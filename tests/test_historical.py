import copy
import json

import numpy as np
import pytest
import torch
from torch import nn

from coscaf.historical import (
    SCHEMA,
    calibrations,
    check_assets,
    compare_scores,
    load_historical_state,
    load_metadata,
    load_saved_generator,
    main,
    verify_assets,
)
from coscaf.io import sha256
from coscaf.model import GENERATOR, CoScafModel, CoScafReadout


class TinySuffix(nn.Module):
    def __init__(self):
        super().__init__()
        self.prompts = nn.Parameter(torch.zeros(2, 16, 32))

    def forward(self, raw):
        return raw[:, 1:] + self.prompts.mean((0, 1))

    def adapter_state(self):
        return {"prompts": self.prompts.detach().clone()}

    def load_adapter(self, state):
        if set(state) != {"prompts"} or state["prompts"].shape != self.prompts.shape:
            raise ValueError("Incorrect prompt state")
        with torch.no_grad():
            self.prompts.copy_(state["prompts"])


def original_state():
    torch.manual_seed(9)
    initial = {
        "weight": torch.randn(7, 32),
        "bias": torch.randn(32),
        "residual_weight": torch.randn(7, 2, 32),
        "residual_bias": torch.randn(2, 32),
    }
    head = CoScafReadout(
        torch.randn(4, 7),
        initial,
        torch.zeros(32),
        torch.eye(32),
        variant="latent",
        prior_temperature=0.01,
        local_weight=0.25,
    )
    model = CoScafModel(TinySuffix(), head).eval()
    state = copy.deepcopy(model.trainable_state())
    for name in GENERATOR:
        state["head"]["initial_" + name] = torch.randn_like(state["head"][name])
    config = {
        "source_recipe": {"local_weight": 0.25, "prior_temperature": 0.25, "score_batch_size": 2},
        "readout": {"variant": "latent", "prior_temperature": 0.01},
    }
    return model, state, config


def test_legacy_restore_drops_only_loss_buffers_and_preserves_scores():
    original, state, config = original_state()
    restored = load_historical_state(
        state, TinySuffix(), config, state["head"]["semantics"].numpy()
    )
    assert restored.head.prior_temperature == 0.01
    assert restored.head.native is False
    assert not any(p.requires_grad for p in restored.parameters())
    raw, ids = torch.randn(3, 17, 32), torch.arange(4)
    with torch.no_grad():
        torch.testing.assert_close(original(raw, ids), restored(raw, ids), atol=0, rtol=0)
    assert all("initial_" + name not in restored.head.state_dict() for name in GENERATOR)
    assert all("initial_" + name in state["head"] for name in GENERATOR)


@pytest.mark.parametrize("mutation", ["unknown", "missing", "partial_initial", "semantic_order"])
def test_legacy_restore_rejects_active_state_corruption(mutation):
    _, state, config = original_state()
    semantics = state["head"]["semantics"].numpy().copy()
    if mutation == "unknown":
        state["head"]["unrecognized_weight"] = torch.zeros(1)
    elif mutation == "missing":
        del state["head"]["query_high.weight"]
    elif mutation == "partial_initial":
        del state["head"]["initial_mode_bias"]
    else:
        semantics = semantics[::-1].copy()
    with pytest.raises((RuntimeError, ValueError, AssertionError)):
        load_historical_state(state, TinySuffix(), config, semantics)


def test_saved_generator_replay_does_not_change_source_state():
    model, _, _ = original_state()
    before = copy.deepcopy(model.state_dict())
    saved = {name: getattr(model.head, name).detach().clone() for name in GENERATOR}
    saved["center_bias"] += torch.linspace(-1, 1, 32)
    restored = load_saved_generator({"generator": saved}, model.head)
    ids = torch.arange(4)
    with torch.no_grad():
        features = model.head.features(torch.randn(2, 16, 32), ids)
        source = model.head.score_features(features, ids)
        adapted = model.head.score_features(features, ids, restored)
    assert not torch.equal(source, adapted)
    for name, value in before.items():
        torch.testing.assert_close(model.state_dict()[name], value, atol=0, rtol=0)
    saved["center_bias"] = torch.zeros(31)
    with pytest.raises(ValueError, match="shape/dtype"):
        load_saved_generator({"generator": saved}, model.head)


def test_comparison_uses_frozen_gamma_and_separate_czsl_predictions():
    scores = np.array([[0.8, 0.1, 0.7, 0.2], [0.1, 0.9, 0.2, 0.8]], np.float32)
    result = compare_scores(scores, scores.copy(), np.arange(4), np.array([0, 1]), {"gamma": 0.2})
    assert result["gzsl_prediction_disagreements"] == 0
    assert result["czsl_prediction_disagreements"] == 0
    changed = scores.copy()
    changed[0, 2] = -1
    result = compare_scores(changed, scores, np.arange(4), np.array([0, 1]), {"gamma": 0.2})
    assert not result["scores_within_tolerance"]
    assert result["gzsl_prediction_disagreements"] == 1
    assert result["czsl_prediction_disagreements"] == 1


def _json(path, value):
    path.write_text(json.dumps(value))


def fixture_assets(tmp_path):
    root = tmp_path / "assets"
    root.mkdir()
    model, state, config = original_state()
    labels = np.array([0, 1, 3, 0, 2, 1])
    train, test = np.array([0, 1]), np.array([2, 3, 4, 5])
    np.savez(
        root / "metadata.npz",
        labels=labels,
        semantics=state["head"]["semantics"].numpy(),
        train=train,
        test=test,
        seen=np.array([0, 1]),
        unseen=np.array([2, 3]),
    )
    np.save(root / "indices.npy", test)
    raw = np.random.default_rng(23).normal(size=(4, 17, 32)).astype(np.float32)
    np.save(root / "raw.npy", raw)
    torch.save(state, root / "source.pt")
    generator = {name: getattr(model.head, name).detach().clone() for name in GENERATOR}
    generator["center_bias"] += 0.1
    torch.save({"generator": generator, "history": []}, root / "adapted.pt")
    with torch.no_grad():
        features = model.head.features(model.suffix(torch.from_numpy(raw)), torch.arange(4))
        for name, parameters in (("source", None), ("adapted", generator)):
            np.save(
                root / (name + "_scores.npy"),
                model.head.score_features(features, torch.arange(4), parameters).numpy(),
            )
    _json(root / "config.json", config)
    _json(
        root / "calibration.json",
        {
            "source_calibration": {"gamma": 0.1, "probability_temperature": 0.07},
            "tta_calibration": {"gamma": 0.2, "probability_temperature": 0.08},
        },
    )
    backbone = root / "backbone"
    backbone.mkdir()
    (backbone / "model.safetensors").write_bytes(b"synthetic not a real backbone")
    (backbone / "config.json").write_text('{"synthetic": true}')

    def spec(name):
        return {"path": name, "sha256": sha256(root / name)}

    manifest = {
        "schema": SCHEMA,
        "dataset": "CUB",
        "candidate": "synthetic",
        "seed": 9,
        "paper_endpoint": "tta",
        "metadata": spec("metadata.npz"),
        "prefix": {"role": "test", "values": spec("raw.npy"), "indices": spec("indices.npy")},
        "source_checkpoint": spec("source.pt"),
        "adaptation_checkpoint": spec("adapted.pt"),
        "source_config": spec("config.json"),
        "calibration": spec("calibration.json"),
        "reference": {name: spec(name + "_scores.npy") for name in ("source", "adapted")},
        "backbone": {
            "kind": "google_vit",
            "path": "backbone",
            "sha256": sha256(backbone / "model.safetensors"),
        },
    }
    manifest_path = tmp_path / "manifest.json"
    _json(manifest_path, manifest)
    return root, manifest, manifest_path


def test_full_hash_receipt_covers_backbone_dependencies_and_catches_stat_change(tmp_path):
    root, manifest, _ = fixture_assets(tmp_path)
    receipt = tmp_path / "verified.json"
    result = verify_assets(manifest, root, receipt)
    assert "backbone/config.json" in result["files"]
    assert check_assets(manifest, root, receipt)["all_declared_hashes_verified"]
    (root / "backbone/config.json").write_text("changed")
    with pytest.raises(ValueError, match="stat changed"):
        check_assets(manifest, root, receipt)


def test_saved_score_audit_never_loads_backbone_or_runs_adaptation(tmp_path, monkeypatch):
    root, _, manifest_path = fixture_assets(tmp_path)

    def forbidden(*args, **kwargs):
        raise AssertionError("Audit must not load a model or optimize anything")

    monkeypatch.setattr("coscaf.historical._load_suffix", forbidden)
    monkeypatch.setattr(torch.optim.Adam, "__init__", forbidden)
    output = tmp_path / "audit"
    assert (
        main(
            [
                "--manifest",
                str(manifest_path),
                "--assets",
                str(root),
                "--mode",
                "audit",
                "--output",
                str(output),
            ]
        )
        == 0
    )
    result = json.loads((output / "replay.json").read_text())
    assert result["saved_scorebank_metrics_only"]
    assert not result["archived_result_reproduced"]
    assert set(result["metrics"]) == {"source", "adapted"}
    assert result["paper_endpoint_score_key"] == "adapted"


def test_subset_checkpoint_replay_is_not_full_test_reproduction(tmp_path, monkeypatch):
    root, _, manifest_path = fixture_assets(tmp_path)
    monkeypatch.setattr("coscaf.historical._load_suffix", lambda *args: TinySuffix())

    def forbidden(*args, **kwargs):
        raise AssertionError("Checkpoint replay must not create an optimizer")

    monkeypatch.setattr(torch.optim.Adam, "__init__", forbidden)
    monkeypatch.setattr(torch.optim.AdamW, "__init__", forbidden)
    output = tmp_path / "replay"
    assert (
        main(
            [
                "--manifest",
                str(manifest_path),
                "--assets",
                str(root),
                "--output",
                str(output),
                "--limit",
                "2",
            ]
        )
        == 0
    )
    result = json.loads((output / "replay.json").read_text())
    assert result["rows"] == 2
    assert not result["complete_official_population"]
    assert not result["archived_result_reproduced"]
    assert result["metrics"] == {}
    np.testing.assert_array_equal(np.load(output / "indices.npy"), [2, 3])
    assert set(result["comparisons"]) == {"source", "adapted"}


def test_native_metadata_retains_non_grouped_test_order(tmp_path):
    root, _, _ = fixture_assets(tmp_path)
    meta = load_metadata(root / "metadata.npz")
    np.testing.assert_array_equal(meta["test"], [2, 3, 4, 5])
    np.testing.assert_array_equal(meta["labels"][meta["test"]], [3, 0, 2, 1])


def test_frozen_calibration_requires_both_policies_and_valid_temperature():
    with pytest.raises(ValueError, match="Both"):
        calibrations({"source_calibration": {"gamma": 0, "probability_temperature": 1}})
    with pytest.raises(ValueError, match="positive"):
        calibrations(
            {
                "source_calibration": {"gamma": 0, "probability_temperature": 0},
                "tta_calibration": {"gamma": 0, "probability_temperature": 1},
            }
        )


def test_saved_metric_audit_rejects_a_false_historical_metric(tmp_path):
    root, manifest, manifest_path = fixture_assets(tmp_path)
    manifest["expected_metrics"] = {"adapted": {"H": -1.0}}
    _json(manifest_path, manifest)
    output = tmp_path / "mismatch"
    assert (
        main(
            [
                "--manifest",
                str(manifest_path),
                "--assets",
                str(root),
                "--mode",
                "audit",
                "--output",
                str(output),
            ]
        )
        == 2
    )
    result = json.loads((output / "replay.json").read_text())
    assert result["status"] == "MISMATCH"
    assert not result["metric_comparisons"]["adapted"]["within_tolerance"]
