"""Generator isolation and complete-target gradient accumulation tests."""

import copy
import inspect

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from coscaf.adaptation import (
    adapt_evidence,
    cached_scores,
    chunk_features,
    decomposed_kl,
    local_targets,
    semantic_relation_loss,
    source_topology_loss,
    targets_with_mass,
)
from coscaf.model import GENERATOR, CoScafReadout, prototypes


def make_problem(variant="latent", engine="geometry_ot"):
    torch.manual_seed(157)
    n, c, a, d, k = 7, 4, 5, 8, 2
    initializer = {
        "weight": torch.randn(a, d) * 0.2,
        "bias": torch.randn(d) * 0.1,
        "residual_weight": torch.randn(a, k, d) * 0.03,
        "residual_bias": torch.randn(k, d) * 0.01,
    }
    head = CoScafReadout(
        torch.randn(c, a),
        initializer,
        torch.zeros(d),
        torch.empty(0),
        query_rank=2,
        moment_rank=2,
        variant=variant,
    )
    ids = torch.arange(c)
    with torch.no_grad():
        evidence = head.features(torch.randn(n, 4, d), ids)
        scores = head.score_features(evidence, ids)
    cache = {name: value.numpy().copy() for name, value in evidence.items()}
    for layer in (10, 12):
        cache[f"view{layer}_global"] = cache["global"].copy()
        cache[f"view{layer}_scores"] = scores.numpy().copy()
    hp = {
        "engine": engine,
        "lr": 0.0001,
        "steps": 2,
        "assignment_temperature": 0.2,
        "consistency_temperature": 0.15,
        "partial_mass": 0.65,
        "view_js_temperature": 0.02,
        "view_margin_temperature": 0.02,
        "ot_epsilon": 0.5,
        "ot_prior_shrink": 0.5,
        "ot_relaxation": 0.8,
        "ot_iterations": 20,
        "mode_temperature": 0.05,
        "min_mode_mass": 0.01,
        "class_alignment_weight": 1.0,
        "mode_alignment_weight": 1.0,
        "consistency_weight": 1.0,
        "ranking_weight": 0.5,
        "ranking_margin": 0.03,
        "hard_negative_neighbors": 2,
        "anchor_weight": 0.1,
        "relation_weight": 0.2,
        "topology_weight": 0.25,
        "gradient_clip": 5.0,
        "agreement_mode": "group",
        "mass_balance_strength": 0.15,
        "mass_balance_iterations": 3,
        "centroid_blend": 0.25,
    }
    setting = {"lr_factor": 1.0, "local_target_mix": 1.0, "group_weight": 1.0}
    return head, ids, cache, np.array([True, True, False, False]), hp, setting


@pytest.mark.parametrize("variant", ["original", "latent"])
@pytest.mark.parametrize("engine", ["geometry_ot", "sun_soft_ot"])
def test_adaptation_only_changes_its_generator_copy(variant, engine):
    head, ids, cache, seen, hp, setting = make_problem(variant, engine)
    before = {name: value.clone() for name, value in head.state_dict().items()}
    cache_before = copy.deepcopy(cache)
    result = adapt_evidence(head, ids, cache, seen, 0.02, hp, setting, batch=3)
    assert result["head_unchanged"]
    assert len(result["history"]) == hp["steps"]
    assert result["scores"].shape == (7, 4)
    assert np.isfinite(result["scores"]).all()
    assert set(result["generator"]) == set(GENERATOR)
    assert any(not torch.equal(result["generator"][key], before[key]) for key in GENERATOR)
    for name, value in head.state_dict().items():
        assert torch.equal(value, before[name])
    assert all(parameter.grad is None for parameter in head.parameters())
    for name, value in cache.items():
        np.testing.assert_array_equal(value, cache_before[name])


def test_no_target_label_api_or_labeled_cache():
    assert "labels" not in inspect.signature(adapt_evidence).parameters
    head, ids, cache, seen, hp, setting = make_problem()
    with pytest.raises(TypeError):
        adapt_evidence(head, ids, cache, seen, 0.02, hp, setting, labels=np.zeros(7))
    cache["labels"] = np.zeros(7)
    with pytest.raises(ValueError, match="Unexpected feature-cache"):
        adapt_evidence(head, ids, cache, seen, 0.02, hp, setting)


def test_chunked_scoring_targets_and_optimizer_match_full_target():
    head, ids, cache, seen, hp, setting = make_problem()
    source = {name: getattr(head, name).detach().clone() for name in GENERATOR}
    np.testing.assert_allclose(
        cached_scores(head, ids, source, cache, "cpu", 2),
        cached_scores(head, ids, source, cache, "cpu", 7),
        atol=2e-7,
    )
    target, _ = targets_with_mass(source, head.semantics, cache, seen, 0.02, hp, "cpu")
    center_small, mass_small = local_targets(
        head, ids, source, cache, target["transport"], "cpu", 2
    )
    center_full, mass_full = local_targets(head, ids, source, cache, target["transport"], "cpu", 7)
    torch.testing.assert_close(center_small, center_full)
    torch.testing.assert_close(mass_small, mass_full)
    small = adapt_evidence(head, ids, cache, seen, 0.02, hp, setting, batch=2)
    full = adapt_evidence(head, ids, cache, seen, 0.02, hp, setting, batch=7)
    for name in GENERATOR:
        torch.testing.assert_close(
            small["generator"][name], full["generator"][name], atol=3e-7, rtol=3e-5
        )
    np.testing.assert_allclose(small["scores"], full["scores"], atol=3e-7, rtol=3e-5)


def test_target_gradient_accumulation_matches_dense_objective():
    head, ids, cache, seen, hp, _ = make_problem()
    source = {name: getattr(head, name).detach().clone() for name in GENERATOR}
    target, _ = targets_with_mass(source, head.semantics, cache, seen, 0.02, hp, "cpu")

    def gradient(batch):
        parameters = {name: value.clone().requires_grad_() for name, value in source.items()}
        denominator = target["reliability"].sum()
        for start in range(0, 7, batch):
            sl = slice(start, min(start + batch, 7))
            scores = head.score_features(chunk_features(cache, sl, "cpu"), ids, parameters)
            logp = F.log_softmax(
                (scores - 0.02 * target["seen"]) / hp["consistency_temperature"], -1
            )
            kl = decomposed_kl(logp, target["teacher"][sl], target["seen"], 1.0)
            index = torch.arange(len(scores))
            ranking = F.relu(
                hp["ranking_margin"]
                + scores[index, target["negative"][sl]]
                - scores[index, target["positive"][sl]]
            )
            objective = (
                (kl + hp["ranking_weight"] * ranking) * target["reliability"][sl]
            ).sum() / denominator
            objective.backward()
        return {name: value.grad for name, value in parameters.items()}

    dense, chunked = gradient(7), gradient(2)
    for name in GENERATOR:
        torch.testing.assert_close(dense[name], chunked[name], atol=2e-6, rtol=2e-5)


def test_zero_steps_replays_source_and_geometry_is_finite():
    head, ids, cache, seen, hp, setting = make_problem()
    source = {name: getattr(head, name).detach().clone() for name in GENERATOR}
    result = adapt_evidence(head, ids, cache, seen, 0.02, hp, setting, steps=0)
    np.testing.assert_array_equal(result["scores"], cached_scores(head, ids, source, cache, "cpu"))
    base, proto = prototypes(source, head.semantics)
    assert source_topology_loss(base, proto, base, proto) == 0
    assert torch.isfinite(semantic_relation_loss(base, head.semantics))
    assert result["history"] == []
