"""CPU protocol tests; no images, real pretrained models, or GPU work."""

import copy
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from coscaf import pipeline, training
from coscaf.backbones import VIT_ID, VIT_REVISION, VIT_WEIGHTS_SHA256
from coscaf.data import DatasetBundle
from coscaf.io import read_json, write_json
from coscaf.preparation import write_prefix_cache_manifest


class TinyClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(2, 2))
        self.head = SimpleNamespace(semantics=torch.ones(2, 1))

    def forward(self, values, classes):
        return (values @ self.weight)[:, classes]

    def trainable_state(self):
        return {"weight": self.weight.detach().clone()}

    def load_trainable_state(self, state):
        with torch.no_grad():
            self.weight.copy_(state["weight"])


def test_epoch_selection_restores_best_weights_and_stops_after_patience(monkeypatch):
    model = TinyClassifier()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.2)
    monkeypatch.setattr(training, "make_optimizer", lambda *args: optimizer)
    values = torch.eye(2)
    states, observed_lrs = {}, []
    metrics = [0.4, 0.6, 0.6 + 5e-13, 0.599, 0.9]

    def validate(current, epoch):
        states[epoch] = current.weight.detach().clone()
        observed_lrs.append(optimizer.param_groups[0]["lr"])
        return {"H": metrics[epoch - 1]}

    config = dict(epochs=100, batch_size=2, loss_temperature=1.0, epoch_decay="exponential_095")
    restored, history = training.train_fit(
        model,
        lambda ix: values[ix],
        np.arange(2),
        np.arange(2),
        config,
        7011,
        validation=validate,
        patience=2,
    )
    assert len(history) == 4
    assert history[-1]["best_epoch"] == 2
    assert history[-1]["epochs_without_improvement"] == 2
    assert observed_lrs == pytest.approx([0.2 * 0.95**i for i in range(4)])
    torch.testing.assert_close(restored.weight, states[2], rtol=0, atol=0)
    assert not torch.equal(restored.weight, states[4])


@pytest.mark.parametrize("policy", [None, "constant", "constant_original"])
def test_constant_schedule_preserves_initial_learning_rate(policy):
    assert [training.epoch_multiplier(policy, e) for e in (1, 2, 100)] == [1, 1, 1]


def test_unknown_schedule_is_rejected():
    with pytest.raises(ValueError, match="Unknown source epoch decay"):
        training.epoch_multiplier("ignored_typo", 1)


@pytest.fixture
def synthetic_bundle(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    (data / "folds").mkdir()
    labels = np.repeat(np.arange(4), 5)
    train = np.array([0, 1, 2, 3, 5, 6, 7, 8, 10, 11, 12, 13])
    np.savez(
        data / "metadata.npz",
        labels=labels,
        semantics=np.random.default_rng(3).normal(size=(4, 6)),
        train=train,
        test_seen=np.array([4, 9, 14]),
        test_unseen=np.arange(15, 20),
    )
    bundle = DatasetBundle.load(data)
    provenance = dict(
        id=VIT_ID,
        revision=VIT_REVISION,
        weights_sha256=VIT_WEIGHTS_SHA256,
        resolution=384,
        prefix_block=10,
        config_sha256="synthetic",
        processor_sha256="synthetic",
    )
    for role, indices in (("train", train), ("test", bundle.test)):
        folder = tmp_path / role
        folder.mkdir()
        np.save(folder / "indices.npy", indices)
        np.save(
            folder / "raw.npy",
            np.random.default_rng(8).normal(size=(len(indices), 577, 16)).astype(np.float32),
        )
        write_prefix_cache_manifest(folder, bundle, role, provenance)
    folds = [
        dict(fit=[0, 1, 5, 6], pseudo_seen=[0, 1], pseudo_unseen=[2]),
        dict(fit=[5, 6, 10, 11], pseudo_seen=[1, 2], pseudo_unseen=[0]),
    ]
    for fold in folds:
        fold.update(cal=[2, 7, 12], tune=[3, 8, 13])
    return data, folds, provenance


class SyntheticSuffix(nn.Module):
    def __init__(self, provenance):
        super().__init__()
        self.prompts = nn.Parameter(torch.randn(2, 16, 16) * 0.01)
        self.enabled = True
        self.backbone_provenance = provenance

    def layer_views(self, raw):
        early = raw[:, 1:]
        late = early + self.prompts.mean((0, 1))[None, None, :] if self.enabled else early
        return (early + late) * 0.5, early, late

    def forward(self, raw):
        return self.layer_views(raw)[0]

    def adapter_state(self):
        return {"prompts": self.prompts.detach().cpu().clone()}

    def load_adapter(self, state):
        with torch.no_grad():
            self.prompts.copy_(state["prompts"])


def small_config():
    config_path = Path(__file__).parents[1] / "configs/CUB-best.json"
    config = read_json(config_path)
    config.update(dataset="synthetic", semantics_dim=6, visual_dim=16, seed=7033, folds=[0, 1])
    config["source"].update(
        modes=1,
        epochs=99,
        batch_size=2,
        micro_batch_size=1,
        score_batch_size=2,
        visual_norm="l2",
        epoch_decay="exponential_095",
    )
    config["adaptation"]["steps"] = 1
    config["adaptation_batch"] = 2
    config["development"] = dict(seed=7011, max_epochs=3, patience=1, improvement_tolerance=1e-12)
    return config


def write_folds(data, folds, extension):
    for i, fold in enumerate(folds):
        path = data / "folds" / f"fold_{i}.{extension}"
        if extension == "npz":
            np.savez(path, **{key: np.asarray(value) for key, value in fold.items()})
        else:
            write_json(path, fold)


@pytest.mark.parametrize("extension", ["npz", "json"])
def test_development_and_final_seeds_horizon_and_calibration_are_separate(
    tmp_path,
    synthetic_bundle,
    monkeypatch,
    extension,
):
    data, folds, provenance = synthetic_bundle
    write_folds(data, folds, extension)
    config = small_config()
    calls = []
    original_fit = pipeline.train_fit

    def capture_fit(model, reader, fit, labels, source, seed, *args, **kwargs):
        calls.append((seed, source["epochs"], kwargs.get("validation") is not None))
        return original_fit(model, reader, fit, labels, source, seed, *args, **kwargs)

    monkeypatch.setattr(pipeline, "train_fit", capture_fit)

    def factory(device):
        return SyntheticSuffix(provenance).to(device)

    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        calibration = pipeline.develop(
            config,
            data,
            tmp_path / "train",
            tmp_path / "dev",
            factory,
        )
        best = calibration["development_selection"]["best_epochs_by_fold"]
        horizon = int(math.ceil(float(np.median(list(best.values())))))
        freeze = pipeline.train(
            config,
            data,
            tmp_path / "train",
            tmp_path / "dev",
            tmp_path / "run",
            factory,
        )
        assert calls == [(7011, 3, True), (7011, 3, True), (7033, horizon, False)]
        assert 1 <= horizon <= 3 and horizon != config["source"]["epochs"]
        assert freeze["effective_seed"] == 7033
        assert freeze["effective_epochs"] == horizon
        for fold_id in config["folds"]:
            history = read_json(tmp_path / "dev" / f"fold_{fold_id}" / "history.json")
            assert all(r["validation"]["calibration_role"] == "CAL" for r in history)
            assert all(r["validation"]["selection_role"] == "TUNE" for r in history)
            assert history[-1]["best_epoch"] == best[str(fold_id)]
        result = pipeline.evaluate_run(
            tmp_path / "run",
            data,
            tmp_path / "test",
            tmp_path / "evaluation",
            factory,
        )
        assert set(result) == {"source", "adapted"}
    finally:
        torch.set_num_threads(old_threads)


def test_final_horizon_uses_ceiling_and_rejects_historical_or_missing_selection():
    config = small_config()
    calibration = {
        "development_selection": {
            "settings": pipeline._development_settings(config),
            "best_epochs_by_fold": {"0": 2, "1": 3},
        },
        "final_training": {"seed": 7033, "epochs": 3},
    }
    assert pipeline._final_fit_config(config, calibration)["source"]["epochs"] == 3
    changed = copy.deepcopy(calibration)
    changed["final_training"]["epochs"] = 99
    with pytest.raises(ValueError, match="current-DEV horizon"):
        pipeline._final_fit_config(config, changed)
    with pytest.raises(ValueError, match="current development"):
        pipeline._final_fit_config(config, {})
    config.pop("development")
    assert pipeline._final_fit_config(config, {}) == config


@pytest.mark.parametrize("failure", ["test_row", "pooled_roles", "test_cache"])
def test_development_rejects_test_access_and_cross_fold_leakage_before_backbone(
    tmp_path,
    synthetic_bundle,
    failure,
):
    data, folds, _ = synthetic_bundle
    if failure == "test_row":
        folds[0]["cal"][0] = 4
    elif failure == "pooled_roles":
        folds[1]["cal"], folds[1]["tune"] = folds[1]["tune"], folds[1]["cal"]
    write_folds(data, folds, "npz")

    def forbidden_factory(*args):
        pytest.fail("Invalid development input reached a backbone")

    with pytest.raises(ValueError, match="Official test|Pooled calibration|incorrect role"):
        pipeline.develop(
            small_config(),
            data,
            tmp_path / ("test" if failure == "test_cache" else "train"),
            tmp_path / "dev",
            forbidden_factory,
        )
