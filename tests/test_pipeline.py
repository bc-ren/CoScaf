import json

import numpy as np
import pytest

from coscaf.io import read_json
from coscaf.pipeline import train
from coscaf.smoke import SyntheticSuffix, run_smoke


@pytest.fixture(scope="module")
def completed(tmp_path_factory):
    folder = tmp_path_factory.mktemp("pipeline") / "run"
    report = run_smoke(folder)
    assert report["status"] == "PASS"
    return folder


def test_complete_pipeline_writes_frozen_scores_and_metrics(completed):
    final = read_json(completed / "model/freeze.json")
    assert final["official_test_used"] is False
    assert (
        read_json(completed / "evaluation/provenance.json")["target_labels_used_for_adaptation"]
        is False
    )
    for mode in ("source", "adapted"):
        scores = np.load(completed / f"evaluation/{mode}_scores.npy")
        assert scores.shape == (56, 6)
        assert np.isfinite(scores).all()
        curve = np.load(completed / f"evaluation/{mode}_su_curve.npy")
        assert (np.diff(curve[:, 2]) >= -1e-12).all()


def test_changed_recipe_cannot_reuse_development_calibration(completed):
    config = read_json(completed / "model/config.json")
    config["source"]["modes"] = 3
    with pytest.raises(ValueError, match="Configuration differs"):
        train(
            config,
            completed / "data",
            completed / "cache_train",
            completed / "development",
            completed / "invalid_model",
            SyntheticSuffix,
        )


def test_changed_fold_cannot_reuse_development_calibration(completed):
    path = completed / "data/folds/fold_0.json"
    original = path.read_text()
    path.write_text(json.dumps(json.loads(original), separators=(",", ":")))
    try:
        config = read_json(completed / "model/config.json")
        with pytest.raises(ValueError, match="fold membership has changed"):
            train(
                config,
                completed / "data",
                completed / "cache_train",
                completed / "development",
                completed / "invalid_fold",
                SyntheticSuffix,
            )
    finally:
        path.write_text(original)
