import numpy as np
import pytest

from coscaf.initialization import Ridge, TargetBank, fit_transform, transform, unit_rows


def test_ridge_matches_regularized_normal_equations():
    rng = np.random.default_rng(19)
    semantics = rng.normal(size=(7, 5))
    targets = rng.normal(size=(7, 3))
    alpha = 0.03
    weight, bias = Ridge(semantics, targets).solve(alpha)
    centered = semantics - semantics.mean(0)
    expected = np.linalg.solve(
        centered.T @ centered / 7 + alpha * np.eye(5), centered.T @ (targets - targets.mean(0)) / 7
    )
    np.testing.assert_allclose(weight, expected, atol=1e-12)
    np.testing.assert_allclose(bias, targets.mean(0) - semantics.mean(0) @ expected)


@pytest.mark.parametrize("kind", ["l2", "white_0.5", "white_0.1"])
def test_coordinate_transform_is_fit_only_and_unit_normalized(kind):
    rng = np.random.default_rng(4)
    fit = rng.normal(size=(15, 8))
    state = fit_transform(fit, kind)
    untouched = {key: value.copy() for key, value in state.items()}
    output = transform(rng.normal(size=(4, 8)), state)
    np.testing.assert_allclose(np.linalg.norm(output, axis=1), 1, atol=1e-12)
    for key in state:
        np.testing.assert_array_equal(state[key], untouched[key])


def test_shared_initialization_generates_unseen_rows():
    rng = np.random.default_rng(72)
    semantics = unit_rows(rng.normal(size=(6, 7)))
    labels = np.repeat([0, 2, 4], 10)
    features = unit_rows(rng.normal(size=(30, 12)))
    recipe = {"modes": 3, "alpha_base": 0.01, "alpha_residual": 0.001}
    model = TargetBank(features, labels, semantics, "random", 11).model(recipe)
    center = semantics @ model["weight"] + model["bias"]
    assert center.shape == (6, 12)
    assert model["residual_weight"].shape == (7, 3, 12)
    repeated = TargetBank(features, labels, semantics, "random", 11).model(recipe)
    for key in model:
        np.testing.assert_array_equal(model[key], repeated[key])


def test_zero_semantics_rejected():
    with pytest.raises(ValueError, match="nonzero"):
        unit_rows(np.zeros((2, 5)))
