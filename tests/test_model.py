import copy

import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as F

from coscaf.backbones import VisualPromptSuffix
from coscaf.model import GENERATOR, CoScafModel, CoScafReadout, regional_moments
from coscaf.training import build_from_state, make_model, schedule, source_loss, train_fit


def readout(variant="original", modes=3, native=False):
    torch.manual_seed(21)
    initializer = {
        "weight": torch.randn(7, 32),
        "bias": torch.randn(32),
        "residual_weight": torch.randn(7, modes, 32),
        "residual_bias": torch.randn(modes, 32),
    }
    return CoScafReadout(
        torch.randn(6, 7),
        initializer,
        torch.zeros(32),
        torch.empty(0, 0),
        variant=variant,
        native=native,
    )


@pytest.mark.parametrize("variant", ["original", "latent"])
@pytest.mark.parametrize("modes", [1, 2, 3, 8])
def test_cached_evidence_replays_direct_scores(variant, modes):
    head = readout(variant, modes)
    patch = torch.randn(2, 16, 32)
    classes = torch.tensor([0, 2, 5])
    direct = head(patch, classes)
    cached, diagnostic = head.score_features(
        head.features(patch, classes), classes, diagnostics=True
    )
    torch.testing.assert_close(direct, cached, atol=2e-7, rtol=1e-6)
    assert torch.isfinite(cached).all()
    for value in diagnostic.values():
        torch.testing.assert_close(value.sum(-1), torch.ones_like(value[..., 0]))
    cached.sum().backward()
    assert all(getattr(head, key).grad is not None for key in GENERATOR)


@pytest.mark.parametrize("variant", ["original", "latent"])
def test_usage_distinguishes_routing_from_posterior_responsibility(variant):
    head = readout(variant)
    patch, classes = torch.randn(2, 16, 32), torch.tensor([0, 2, 5])
    score, usage = head(patch, classes, return_usage=True)
    evidence = head.features(patch, classes)
    cached, diagnostic = head.score_features(evidence, classes, diagnostics=True)
    torch.testing.assert_close(score, cached, rtol=0, atol=0)
    posterior = diagnostic["local_responsibility"]
    if variant == "latent":
        torch.testing.assert_close(usage, posterior, rtol=0, atol=0)
    else:
        similarity = (evidence["local"] * head.prototypes(classes)[None]).sum(-1)
        torch.testing.assert_close(usage, (similarity / 0.5).softmax(-1), rtol=0, atol=0)
        assert not torch.allclose(usage, posterior)


@pytest.mark.parametrize("variant", ["original", "latent"])
def test_native_embedding_behavior_is_specific_to_original_forward(variant):
    head = readout(variant, native=True)
    patch, classes = torch.randn(2, 16, 32), torch.tensor([0, 2, 5])
    native = torch.randn(2, 7)
    cached = head.score_features(head.features(patch, classes), classes)
    direct = head(patch, classes, native)
    if variant == "latent":
        torch.testing.assert_close(direct, cached, rtol=0, atol=0)
        torch.testing.assert_close(head(patch, classes), cached, rtol=0, atol=0)
    else:
        changed = head(patch, classes, -native)
        expected_delta = 2 * (
            F.normalize(native, dim=-1) @ F.normalize(head.semantics[classes], dim=-1).T
        )
        torch.testing.assert_close(direct - changed, expected_delta)
        with pytest.raises(AssertionError):
            head(patch, classes)


def test_all_prototypes_normalized_and_single_mode_residual_cancels():
    head = readout(modes=1)
    classes = torch.arange(6)
    before = head.prototypes(classes)
    with torch.no_grad():
        head.mode_weight.add_(20)
        head.mode_bias.sub_(20)
    torch.testing.assert_close(head.prototypes(classes), before, atol=0, rtol=0)
    torch.testing.assert_close(before.norm(dim=-1), torch.ones(6, 1))


def test_regional_moments_dimension_and_gradients():
    patch = torch.randn(2, 16, 32, requires_grad=True)
    projection = torch.linalg.qr(torch.randn(32, 16), mode="reduced").Q
    output = regional_moments(patch, projection)
    assert output.shape == (2, 760)
    output.square().sum().backward()
    assert torch.isfinite(patch.grad).all()


class ToySuffix(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.prompts = nn.Parameter(torch.randn(2, 2, 32, device=device) * 0.01)

    def forward(self, raw):
        return raw + self.prompts.mean((0, 1))

    def adapter_state(self):
        return {"prompts": self.prompts.detach().clone()}

    def load_adapter(self, state):
        with torch.no_grad():
            self.prompts.copy_(state["prompts"])


def test_source_loss_and_gradient_are_class_balanced_cross_entropy():
    scores = torch.tensor(
        [[0.7, -0.3, 0.2], [0.1, 0.9, -0.2], [-0.4, 0.2, 0.5]],
        dtype=torch.float64,
        requires_grad=True,
    )
    labels = torch.tensor([0, 0, 2])
    weights = torch.tensor([0.5, 1.0, 2.0], dtype=torch.float64)
    temperature = 0.3
    loss = source_loss(scores, labels, weights, {"loss_temperature": temperature})
    probabilities = (scores.detach() / temperature).softmax(-1)
    expected_loss = -(probabilities[torch.arange(3), labels].log() * weights[labels]).mean()
    torch.testing.assert_close(loss, expected_loss)
    loss.backward()
    expected_gradient = (
        (probabilities - F.one_hot(labels, 3)) * weights[labels, None] / (3 * temperature)
    )
    torch.testing.assert_close(scores.grad, expected_gradient)


@pytest.mark.parametrize("variant", ["original", "latent"])
def test_legacy_checkpoint_buffers_do_not_change_restored_scores(variant):
    model = CoScafModel(ToySuffix("cpu"), readout(variant))
    state = copy.deepcopy(model.trainable_state())
    assert not any(name.startswith("initial_") for name in state["head"])
    legacy = copy.deepcopy(state)
    for name in GENERATOR:
        legacy["head"]["initial_" + name] = torch.zeros_like(state["head"][name])
    before = copy.deepcopy(legacy)
    restored = build_from_state(
        legacy, {"routing": variant, "local_weight": 0.25}, ToySuffix
    )
    raw, classes = torch.randn(2, 16, 32), torch.arange(6)
    torch.testing.assert_close(model(raw, classes), restored(raw, classes), atol=0, rtol=0)
    assert set(restored.trainable_state()["head"]) == set(state["head"])
    assert set(legacy["head"]) == set(before["head"])
    for key in before["head"]:
        torch.testing.assert_close(legacy["head"][key], before["head"][key], atol=0, rtol=0)


@pytest.mark.parametrize("damage", ["unexpected", "missing", "wrong_shape"])
def test_legacy_checkpoint_loading_still_validates_active_state(damage):
    model = CoScafModel(ToySuffix("cpu"), readout())
    state = copy.deepcopy(model.trainable_state())
    for name in GENERATOR:
        state["head"]["initial_" + name] = torch.zeros_like(state["head"][name])
    if damage == "unexpected":
        state["head"]["initial_unrelated"] = torch.zeros(1)
    elif damage == "missing":
        del state["head"]["query_low.weight"]
    else:
        state["head"]["query_low.weight"] = torch.zeros(1)
    with pytest.raises(RuntimeError):
        model.load_trainable_state(state)


def test_fresh_training_state_round_trip_and_semantics_frozen():
    rng = np.random.default_rng(7)
    semantics = rng.normal(size=(5, 7))
    labels = np.repeat([0, 2, 4], 8)
    features = rng.normal(size=(24, 32))
    config = {
        "visual_norm": "white_0.5",
        "modes": 2,
        "ridge": 0.001,
        "mode_scale": 0.1,
        "local_weight": 0.25,
        "routing": "latent",
        "prompt_lr": 0.001,
        "generator_lr": 0.001,
        "query_lr": 0.001,
        "moment_lr": 0.001,
        "weight_decay": 0.0001,
        "loss_temperature": 0.1,
        "batch_size": 7,
        "micro_batch_size": 3,
        "epochs": 1,
    }
    model = make_model(semantics, features, labels, config, 3, ToySuffix)
    fixed = model.head.semantics.clone()
    raw = torch.tensor(rng.normal(size=(24, 16, 32)), dtype=torch.float32)
    model, history = train_fit(model, lambda ix: raw[ix], np.arange(24), labels, config, 3)
    assert history[0]["updates"] == 4
    torch.testing.assert_close(model.head.semantics, fixed, atol=0, rtol=0)
    restored = build_from_state(model.trainable_state(), config, ToySuffix)
    classes = torch.arange(5)
    torch.testing.assert_close(model(raw[:2], classes), restored(raw[:2], classes), atol=0, rtol=0)
    np.testing.assert_array_equal(
        np.sort(np.concatenate(schedule(np.arange(24), 3, 1, 7))), np.arange(24)
    )


class MixingBlock(nn.Module):
    def forward(self, tokens):
        return tokens + tokens.mean(1, keepdim=True) * 0.1


class ToyViT(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([MixingBlock() for _ in range(12)])
        self.layernorm = nn.LayerNorm(768)


def test_prompts_leave_patch_count_and_backbone_frozen():
    suffix = VisualPromptSuffix(ToyViT(), count=2)
    raw = torch.randn(1, 197, 768)
    fused, early, late = suffix.layer_views(raw)
    assert fused.shape == (1, 196, 768)
    torch.testing.assert_close(fused, (early + late) * 0.5, atol=0, rtol=0)
    fused.square().sum().backward()
    assert suffix.prompts.grad is not None
    assert all(not p.requires_grad for p in suffix.layers.parameters())
    assert all(not p.requires_grad for p in suffix.norm.parameters())
