"""Independent GZSL calculations, calibration isolation and invalid-input tests."""

import copy

import numpy as np
import pytest

from coscaf.metrics import (
    balance_weights,
    curve_from_records,
    evaluate,
    evaluate_folds,
    fit_calibration,
    full_curve,
    group_record,
    joint_calibration,
)


def example():
    scores = np.array(
        [[3.0, 0.0, 1.0, 0.0], [0.0, 2.0, 0.0, 1.0], [2.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 3.0]]
    )
    return scores, np.arange(4), np.arange(4), np.array([0, 1])


def brute_metrics(scores, labels, classes, seen, gamma):
    seen_mask = np.isin(classes, seen)
    prediction = classes[(scores - gamma * seen_mask).argmax(axis=1)]
    unseen_classes = classes[~seen_mask]
    czsl_prediction = unseen_classes[scores[:, ~seen_mask].argmax(axis=1)]
    per_class = {c: np.mean(prediction[labels == c] == c) for c in classes}
    s = np.mean([per_class[c] for c in seen])
    u = np.mean([per_class[c] for c in unseen_classes])
    czsl = np.mean([np.mean(czsl_prediction[labels == c] == c) for c in unseen_classes])
    return dict(S=s, U=u, H=2 * s * u / (s + u) if s + u else 0.0, CZSL=czsl)


def test_metrics_match_direct_predictions_and_keep_frozen_gamma():
    scores, labels, classes, seen = example()
    calibration = {"gamma": 0.0, "probability_temperature": 0.7}
    for gamma in (-3.0, -1.0, 0.0, 1.0, 1.5, 3.0):
        calibration["gamma"] = gamma
        result, _ = evaluate(scores, labels, classes, seen, calibration)
        expected = brute_metrics(scores, labels, classes, seen, gamma)
        for key in expected:
            assert result[key] == pytest.approx(expected[key], abs=1e-14)
        assert result["calibration"]["gamma"] == gamma
    assert result["H"] == 0
    assert result["curve_oracle_H_diagnostic_only"] > result["H"]


def test_metrics_are_class_balanced_under_image_count_imbalance():
    classes = np.arange(4)
    seen = np.array([0, 1])
    labels = np.array([0] * 10 + [1] + [2] * 4 + [3])
    predictions = np.array([0] * 11 + [3] * 5)
    scores = np.eye(4)[predictions]
    result, _ = evaluate(
        scores, labels, classes, seen, {"gamma": 0.0, "probability_temperature": 1.0}
    )
    for key in ("S", "U", "H", "CZSL"):
        assert result[key] == pytest.approx(0.5)
    assert np.mean(predictions == labels) != result["H"]
    weights = balance_weights(labels, classes, seen)
    for c in classes:
        assert weights[labels == c].sum() == pytest.approx(0.25)
    # Reliability metrics use the same group-and-class-balanced population.
    probabilities = np.exp(scores) / np.exp(scores).sum(axis=1, keepdims=True)
    onehot = np.eye(4)[labels]
    expected_brier = weights @ ((probabilities - onehot) ** 2).sum(axis=1)
    assert result["Brier"] == pytest.approx(expected_brier)


def test_sorted_global_indices_define_exact_tie_breaking():
    classes = np.array([2, 5, 8, 11])
    labels = classes.copy()
    seen = np.array([5, 11])
    scores = np.zeros((4, 4))
    result, _ = evaluate(
        scores, labels, classes, seen, {"gamma": 0.0, "probability_temperature": 1.0}
    )
    assert result["S"] == 0
    assert result["U"] == 0.5
    assert result["CZSL"] == 0.5
    record = group_record(scores, labels, classes, seen)
    assert not record["seen_wins_tie"].any()
    # Seen wins ties when its best global index precedes the unseen best index.
    alternate = group_record(scores, labels, classes, np.array([2, 8]))
    assert alternate["seen_wins_tie"].all()
    with pytest.raises(ValueError, match="sorted"):
        evaluate(
            scores[:, [1, 0, 2, 3]],
            labels,
            classes[[1, 0, 2, 3]],
            seen,
            {"gamma": 0.0, "probability_temperature": 1.0},
        )


def test_su_curve_and_ausuc_match_hand_trapezoid():
    scores, labels, classes, seen = example()
    best, curve = full_curve(scores, labels, classes, seen)
    np.testing.assert_allclose(curve[:, 1], [1.0, 1.0, 0.5, 0.0])
    np.testing.assert_allclose(curve[:, 2], [0.0, 0.5, 1.0, 1.0])
    # Area: 0.5 * 1 + 0.5 * (1 + 0.5) / 2 = 0.875, in fraction units.
    assert best["AUSUC"] == pytest.approx(0.875)
    assert best["max_H"] == pytest.approx(2 / 3)
    assert best["gamma"] == pytest.approx(-0.5)
    for gamma, s, u, h in curve:
        actual = brute_metrics(scores, labels, classes, seen, gamma)
        np.testing.assert_allclose([s, u, h], [actual["S"], actual["U"], actual["H"]])


def make_fold(scores, labels, classes, seen):
    return {
        "cal_base": scores,
        "cal_delta": np.zeros_like(scores),
        "cal_y": labels,
        "tune_base": scores.copy(),
        "tune_delta": np.zeros_like(scores),
        "tune_y": labels.copy(),
        "classes": classes,
        "seen": seen,
    }


def test_pooled_calibration_weights_folds_and_classes_equally():
    scores, labels, classes, seen = example()
    second = scores.copy()
    second[:, :2] += 2.0
    folds = [make_fold(scores, labels, classes, seen), make_fold(second, labels, classes, seen)]
    calibration, curve = joint_calibration(folds, 0.0)
    for gamma, s, u, h in curve:
        rows = [brute_metrics(f["cal_base"], f["cal_y"], classes, seen, gamma) for f in folds]
        expected_s = np.mean([row["S"] for row in rows])
        expected_u = np.mean([row["U"] for row in rows])
        expected_h = (
            2 * expected_s * expected_u / (expected_s + expected_u)
            if expected_s + expected_u
            else 0
        )
        np.testing.assert_allclose([s, u, h], [expected_s, expected_u, expected_h], atol=1e-14)
    # Replicating images within just one fold must not increase its influence.
    replicated = copy.deepcopy(folds)
    for key in ("cal_base", "cal_delta", "cal_y"):
        replicated[0][key] = np.repeat(replicated[0][key], 7, axis=0)
    duplicate_cal, duplicate_curve = joint_calibration(replicated, 0.0)
    np.testing.assert_allclose(duplicate_curve, curve, atol=1e-14)
    for key in ("gamma", "max_H", "AUSUC", "probability_temperature"):
        assert duplicate_cal[key] == pytest.approx(calibration[key], rel=1e-8, abs=1e-12)
    metrics = evaluate_folds(folds, 0.0, calibration)
    expected = [
        brute_metrics(f["tune_base"], f["tune_y"], classes, seen, calibration["gamma"])
        for f in folds
    ]
    assert metrics["S"] == pytest.approx(np.mean([row["S"] for row in expected]))
    assert metrics["U"] == pytest.approx(np.mean([row["U"] for row in expected]))


def test_calibration_never_reads_tune_or_test_labels():
    class Forbidden:
        def __array__(self, *args, **kwargs):
            raise AssertionError("Non-calibration data were read")

    scores, labels, classes, seen = example()
    fold = make_fold(scores, labels, classes, seen)
    expected, _ = joint_calibration([fold], 0.0)
    for prefix in ("tune", "test"):
        for suffix in ("base", "delta", "y"):
            fold[f"{prefix}_{suffix}"] = Forbidden()
    actual, _ = joint_calibration([fold], 0.0)
    assert actual == expected
    single, _ = fit_calibration(scores, labels, classes, seen)
    for key in ("gamma", "max_H", "AUSUC", "probability_temperature"):
        assert single[key] == pytest.approx(actual[key])


@pytest.mark.parametrize(
    "invalid",
    [
        "empty",
        "group",
        "missing_class",
        "unknown_label",
        "duplicate",
        "shape",
        "nonfinite",
        "float_ids",
    ],
)
def test_invalid_metric_populations_fail_explicitly(invalid):
    scores, labels, classes, seen = example()
    if invalid == "empty":
        scores, labels = scores[:0], labels[:0]
    elif invalid == "group":
        seen = classes
    elif invalid == "missing_class":
        scores, labels = scores[:-1], labels[:-1]
    elif invalid == "unknown_label":
        labels[0] = 99
    elif invalid == "duplicate":
        classes[1] = classes[0]
    elif invalid == "shape":
        scores = scores[:, :3]
    elif invalid == "nonfinite":
        scores[0, 0] = np.nan
    elif invalid == "float_ids":
        labels = labels.astype(float)
    with pytest.raises(ValueError):
        full_curve(scores, labels, classes, seen)


@pytest.mark.parametrize("temperature", [0.0, -1.0, np.inf, np.nan])
def test_invalid_probability_temperature_is_rejected(temperature):
    with pytest.raises(ValueError, match="temperature"):
        evaluate(*example(), {"gamma": 0.0, "probability_temperature": temperature})


def test_empty_fold_apis_and_zero_normalization_fail():
    with pytest.raises(ValueError):
        joint_calibration([], 0.0)
    with pytest.raises(ValueError):
        evaluate_folds([], 0.0, {})
    with pytest.raises(ValueError):
        curve_from_records([])
    with pytest.raises(ValueError):
        group_record(*example(), nfold=0)
