"""Class-balanced GZSL metrics and development-only calibration.

Scores use sorted global class IDs. Metric values are fractions, not percentages.
The complete S--U curve is diagnostic at test time; its maximizing threshold is
never used to replace the development-fitted calibration.
"""

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp


def _class_metadata(y, classes, seen):
    """Validate a complete, ordered candidate space and its evaluation labels."""
    y, classes, seen = (np.asarray(value) for value in (y, classes, seen))
    for name, value in (("labels", y), ("classes", classes), ("seen", seen)):
        if value.ndim != 1 or not len(value) or not np.issubdtype(value.dtype, np.integer):
            raise ValueError(f"{name} must be a nonempty one-dimensional integer array")
        if np.any(value < 0):
            raise ValueError(f"{name} contains a negative class index")
    if not np.array_equal(classes, np.sort(np.unique(classes))):
        raise ValueError("Candidates must be sorted and unique")
    if len(np.unique(seen)) != len(seen) or not np.isin(seen, classes).all():
        raise ValueError("Seen classes must be a unique subset of candidates")
    mask = np.isin(classes, seen)
    if not mask.any() or mask.all():
        raise ValueError("Both seen and unseen candidate groups are required")
    if not np.isin(y, classes).all():
        raise ValueError("Evaluation label is outside the candidate class space")
    if not np.isin(classes, y).all():
        raise ValueError("Every candidate class must occur in evaluation")
    return y, classes, seen


def _score_inputs(scores, y, classes, seen):
    y, classes, seen = _class_metadata(y, classes, seen)
    scores = np.asarray(scores)
    if scores.ndim != 2 or scores.shape != (len(y), len(classes)):
        raise ValueError("Score shape must match samples and candidate classes")
    if (
        not np.issubdtype(scores.dtype, np.number)
        or np.iscomplexobj(scores)
        or not np.isfinite(scores).all()
    ):
        raise ValueError("Scores must be finite numeric values")
    return scores, y, classes, seen


def _validate_calibration(calibration):
    if not np.isfinite(calibration["gamma"]):
        raise ValueError("Calibration gamma must be finite")
    temperature = calibration["probability_temperature"]
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("Probability temperature must be finite and positive")


def group_record(scores, y, classes, seen, nfold=1):
    """Summarize every sample's seen/unseen transition and class-balanced mass."""
    scores, y, classes, seen = _score_inputs(scores, y, classes, seen)
    if not np.isfinite(nfold) or nfold <= 0:
        raise ValueError("Fold normalization must be finite and positive")
    y = np.asarray(y)
    classes = np.asarray(classes)
    mask = np.isin(classes, seen)
    if not mask.any() or mask.all():
        raise ValueError("Both groups required")
    si, ui = np.flatnonzero(mask), np.flatnonzero(~mask)
    sp = si[np.argmax(scores[:, si], axis=1)]
    up = ui[np.argmax(scores[:, ui], axis=1)]
    ys = np.isin(y, seen)
    counts = {c: np.sum(y == c) for c in classes}
    if min(counts.values()) == 0:
        raise ValueError("Every candidate class must occur in evaluation")
    weight = np.array(
        [1 / counts[c] / (mask.sum() if c in seen else (~mask).sum()) / nfold for c in y]
    )
    rows = np.arange(len(y))
    margin = scores[rows, sp] - scores[rows, up]
    if not np.isfinite(margin).all():
        raise ValueError("Seen/unseen score margins must be finite")
    return {
        "margin": margin,
        "sw": weight * (classes[sp] == y) * ys,
        "uw": weight * (classes[up] == y) * (~ys),
        "seen_wins_tie": sp < up,
    }


def full_curve(scores, y, classes, seen):
    """Enumerate calibrated-stacking transitions and integrate seen versus unseen accuracy.

    Returned curve columns are gamma, S, U, H. All accuracies and area use
    fractions. The maximum H is diagnostic unless this is development data.
    """
    r = group_record(np.asarray(scores, dtype=np.float64), y, classes, seen)
    margins = r["margin"]
    cross = np.where(r["seen_wins_tie"], np.nextafter(margins, np.inf), margins)
    gs, inv = np.unique(cross, return_inverse=True)
    s = np.clip(
        np.r_[r["sw"].sum(), r["sw"].sum() - np.bincount(inv, weights=r["sw"]).cumsum()], 0, 1
    )
    u = np.clip(np.r_[0.0, np.bincount(inv, weights=r["uw"]).cumsum()], 0, 1)
    h = np.divide(2 * s * u, s + u, out=np.zeros_like(s), where=s + u > 0)
    gs = np.r_[np.nextafter(gs[0], -np.inf), gs]
    nxt = np.r_[gs[1:], gs[-1] + max(1.0, abs(gs[-1])) * 1e-6]
    representatives = gs + (nxt - gs) / 2
    representatives[0] = gs[0] - 1e-6
    curve = np.column_stack([representatives, s, u, h])
    best = np.lexsort((representatives, abs(representatives), -h))[0]
    # np.trapz was removed in NumPy2.4. Support both NumPy APIs.
    area = float(np.sum(np.diff(u) * (s[1:] + s[:-1]) * 0.5))
    return dict(gamma=float(representatives[best]), max_H=float(h[best]), AUSUC=area), curve


def balance_weights(y, classes, seen):
    """Equal seen/unseen weight, then equal class weight within each group."""
    y, classes, seen = _class_metadata(y, classes, seen)
    counts = {int(c): int((y == c).sum()) for c in classes}
    if min(counts.values()) < 1:
        raise ValueError("Every class must occur")
    ns = int(np.isin(classes, seen).sum())
    nu = len(classes) - ns
    w = np.array([0.5 / counts[int(c)] / (ns if c in seen else nu) for c in y])
    np.testing.assert_allclose(w.sum(), 1)
    return w


def fit_calibration(scores, y, classes, seen):
    """Fit gamma and probability temperature on development calibration rows."""
    scores, y, classes, seen = _score_inputs(scores, y, classes, seen)
    best, curve = full_curve(scores, y, classes, seen)
    z = np.asarray(scores, dtype=np.float64) - best["gamma"] * np.isin(classes, seen)
    targets = np.searchsorted(classes, y)
    w = balance_weights(y, classes, seen)

    def loss(t):
        a = z / np.exp(t)
        return float(w @ (logsumexp(a, axis=1) - a[np.arange(len(y)), targets]))

    opt = minimize_scalar(loss, bounds=(np.log(0.001), np.log(10)), method="bounded")
    if not opt.success:
        raise ValueError("Probability temperature fit failed")
    return dict(**best, probability_temperature=float(np.exp(opt.x)), fit_role="cal"), curve


def evaluate(scores, y, classes, seen, calibration):
    """Evaluate at frozen gamma; never replace it with the diagnostic test optimum."""
    scores, y, classes, seen = _score_inputs(scores, y, classes, seen)
    _validate_calibration(calibration)
    z = np.asarray(scores, dtype=np.float64)
    y = np.asarray(y)
    classes = np.asarray(classes)
    if not np.array_equal(classes, np.sort(np.unique(classes))):
        raise ValueError("Candidates must be sorted")
    sm = np.isin(classes, seen)
    gamma = calibration["gamma"]
    zc = z - gamma * sm
    pred = classes[zc.argmax(1)]
    unseen = ~np.isin(y, seen)
    uc = classes[~sm]
    upred = uc[z[:, ~sm].argmax(1)]
    pc = {str(int(c)): float((pred[y == c] == c).mean()) for c in classes}
    S = float(np.mean([pc[str(int(c))] for c in classes[sm]]))
    U = float(np.mean([pc[str(int(c))] for c in classes[~sm]]))
    result = dict(
        S=S,
        U=U,
        H=2 * S * U / (S + U) if S + U else 0.0,
        CZSL=float(np.mean([(upred[y == c] == c).mean() for c in uc])),
    )
    best, curve = full_curve(z, y, classes, seen)
    result["AUSUC"] = best["AUSUC"]
    # Oracle curve maximum is diagnostic only; it never replaces frozen gamma.
    result["curve_oracle_H_diagnostic_only"] = best["max_H"]
    weights = balance_weights(y, classes, seen)
    target = np.searchsorted(classes, y)
    prob = np.exp(
        zc / calibration["probability_temperature"]
        - logsumexp(zc / calibration["probability_temperature"], axis=1, keepdims=True)
    )
    conf = prob.max(1)
    correct = pred == y
    ece = 0.0
    for lo, hi in zip(np.linspace(0, 1, 16)[:-1], np.linspace(0, 1, 16)[1:]):
        mask = (conf > lo) & (conf <= hi)
        if mask.any():
            ece += abs(float(weights[mask] @ (conf[mask] - correct[mask])))
    brier = float(weights @ (np.square(prob).sum(1) - 2 * prob[np.arange(len(y)), target] + 1))
    result.update(
        ECE=ece,
        Brier=brier,
        per_class_accuracy=pc,
        worst_seen=sorted([(int(c), pc[str(int(c))]) for c in classes[sm]], key=lambda x: x[1])[
            :10
        ],
        worst_unseen=sorted([(int(c), pc[str(int(c))]) for c in classes[~sm]], key=lambda x: x[1])[
            :10
        ],
        calibration=calibration,
    )
    margin = z[:, sm].max(1) - z[:, ~sm].max(1)
    diag = {}
    for group, mask in [("seen_images", ~unseen), ("unseen_images", unseen)]:
        zz = z[mask]
        diag[group] = dict(
            seen_logit_mean=float(zz[:, sm].mean()),
            seen_logit_std=float(zz[:, sm].std()),
            unseen_logit_mean=float(zz[:, ~sm].mean()),
            unseen_logit_std=float(zz[:, ~sm].std()),
            margin_mean=float(margin[mask].mean()),
            margin_std=float(margin[mask].std()),
            prediction_seen_proportion=float(np.isin(pred[mask], seen).mean()),
        )
    result["logit_diagnostics"] = diag
    return result, curve


def curve_from_records(records):
    """Build one S--U curve with equal total contribution from each fold."""
    if not records:
        raise ValueError("At least one calibration fold is required")
    for record in records:
        arrays = {
            name: np.asarray(record[name]) for name in ("margin", "sw", "uw", "seen_wins_tie")
        }
        if any(value.ndim != 1 for value in arrays.values()):
            raise ValueError("Transition records must be nonempty aligned vectors")
        size = len(arrays["margin"])
        if not size or any(len(value) != size for value in arrays.values()):
            raise ValueError("Transition records must be nonempty aligned vectors")
        if arrays["seen_wins_tie"].dtype != np.bool_:
            raise ValueError("Transition tie indicators must be Boolean")
        if any(not np.isfinite(arrays[name]).all() for name in ("margin", "sw", "uw")):
            raise ValueError("Transition records must contain finite values")
        if any(np.any(arrays[name] < 0) for name in ("sw", "uw")):
            raise ValueError("Transition accuracy weights must be nonnegative")
    n = len(records)
    margin = np.concatenate([r["margin"] for r in records])
    sw = np.concatenate([r["sw"] / n for r in records])
    uw = np.concatenate([r["uw"] / n for r in records])
    tie = np.concatenate([r["seen_wins_tie"] for r in records])
    cross = np.where(tie, np.nextafter(margin, np.inf), margin)
    gs, inv = np.unique(cross, return_inverse=True)
    s = np.clip(np.r_[sw.sum(), sw.sum() - np.bincount(inv, weights=sw).cumsum()], 0, 1)
    u = np.clip(np.r_[0.0, np.bincount(inv, weights=uw).cumsum()], 0, 1)
    h = np.divide(2 * s * u, s + u, out=np.zeros_like(s), where=s + u > 0)
    gs = np.r_[np.nextafter(gs[0], -np.inf), gs]
    nxt = np.r_[gs[1:], gs[-1] + max(1.0, abs(gs[-1])) * 1e-6]
    gamma = gs + (nxt - gs) / 2
    gamma[0] = gs[0] - 1e-6
    best = np.lexsort((gamma, abs(gamma), -h))[0]
    return dict(
        gamma=float(gamma[best]),
        max_H=float(h[best]),
        AUSUC=float(np.sum(np.diff(u) * (s[1:] + s[:-1]) * 0.5)),
    ), np.column_stack([gamma, s, u, h])


def joint_calibration(folds, coefficient):
    """Fit one gamma/temperature using only each fold's ``cal_*`` arrays.

    Each fold and each class contribute equally, regardless of sample count.
    Tune or test arrays are neither read nor used for threshold selection.
    """
    if not folds or not np.isfinite(coefficient):
        raise ValueError("Require calibration folds and a finite score coefficient")
    records = []
    cached = []
    for f in folds:
        z = f["cal_base"] + coefficient * f["cal_delta"]
        y = f["cal_y"]
        c = f["classes"]
        seen = f["seen"]
        records.append(group_record(z, y, c, seen))
        cached.append((z, y, c, seen))
    best, curve = curve_from_records(records)

    def objective(t):
        losses = []
        for z, y, c, seen in cached:
            a = (z - best["gamma"] * np.isin(c, seen)) / np.exp(t)
            w = balance_weights(y, c, seen)
            losses.append(
                float(w @ (logsumexp(a, axis=1) - a[np.arange(len(y)), np.searchsorted(c, y)]))
            )
        return float(np.mean(losses))

    opt = minimize_scalar(objective, bounds=(np.log(0.001), np.log(10)), method="bounded")
    if not opt.success:
        raise ValueError("OOF temperature fit failed")
    return dict(
        **best,
        probability_temperature=float(np.exp(opt.x)),
        fit_role="joint_OOF_cal_equal_fold_equal_class",
    ), curve


def evaluate_folds(folds, coefficient, calibration, role="tune"):
    """Aggregate development metrics with equal fold and class contributions."""
    if not folds or not np.isfinite(coefficient):
        raise ValueError("Require evaluation folds and a finite score coefficient")
    if role not in ("cal", "tune"):
        raise ValueError("Development fold role must be cal or tune")
    rows = []
    records = []
    for f in folds:
        z = f[role + "_base"] + coefficient * f[role + "_delta"]
        y = f[role + "_y"]
        c = f["classes"]
        seen = f["seen"]
        row, _ = evaluate(z, y, c, seen, calibration)
        rows.append(row)
        records.append(group_record(z, y, c, seen))
    S = float(np.mean([r["S"] for r in rows]))
    U = float(np.mean([r["U"] for r in rows]))
    result = {k: float(np.mean([r[k] for r in rows])) for k in ("CZSL", "ECE", "Brier")}
    result.update(
        S=S,
        U=U,
        H=2 * S * U / (S + U) if S + U else 0,
        AUSUC=curve_from_records(records)[0]["AUSUC"],
        fold_H=[r["H"] for r in rows],
        fold_CZSL=[r["CZSL"] for r in rows],
        calibration=calibration,
    )
    return result
