"""Seen-data coordinate transforms and analytic generator initialization.

All statistics must be fitted on the current source-training partition only.
"""

import numpy as np
from scipy.linalg import eigh


def unit_rows(x):
    """Normalize finite, nonzero rows without changing their orientation."""
    x = np.asarray(x, dtype=np.float64)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    if not np.isfinite(x).all() or (norms < 1e-12).any():
        raise ValueError("Expected finite nonzero rows")
    return x / norms


def fit_transform(x, kind):
    z = unit_rows(np.asarray(x, dtype=np.float64))
    d = z.shape[1]
    mean = np.zeros(d) if kind == "l2" else z.mean(0)
    state = {"mean": mean, "matrix": np.empty((0, 0))}
    if kind.startswith("white"):
        shrink = float(kind.split("_")[1])
        c = z - mean
        ev, u = eigh(c.T @ c / len(c))
        ev = np.maximum(ev, 0)
        denom = (1 - shrink) * ev + shrink * ev.mean()
        if denom.min() <= 0:
            raise ValueError("Degenerate covariance")
        state["matrix"] = (u / np.sqrt(denom)) @ u.T
    elif kind not in ["l2", "center"]:
        raise ValueError(kind)
    return state


def transform(x, state):
    z = unit_rows(np.asarray(x, dtype=np.float64)) - state["mean"]
    if state["matrix"].size:
        z = z @ state["matrix"]
    norms = np.linalg.norm(z, axis=1)
    if (norms < 1e-12).any() or not np.isfinite(z).all():
        raise ValueError("Degenerate transformed vector")
    return z / norms[:, None]


class Ridge:
    def __init__(self, a, targets):
        self.am = a.mean(0)
        self.tm = targets.mean(0)
        self.ac = a - self.am
        self.n = len(a)
        if self.n <= a.shape[1]:
            self.dual = True
            ev, self.u = eigh(self.ac @ self.ac.T / self.n)
            self.rhs = self.u.T @ (targets - self.tm) / self.n
        else:
            self.dual = False
            ev, self.u = eigh(self.ac.T @ self.ac / self.n)
            self.rhs = self.u.T @ (self.ac.T @ (targets - self.tm) / self.n)
        self.ev = np.maximum(ev, 0)

    def solve(self, alpha):
        if alpha <= 0:
            raise ValueError("Positive regularization required")
        w = self.u @ (self.rhs / (self.ev[:, None] + alpha))
        if self.dual:
            w = self.ac.T @ w
        return w, self.tm - self.am @ w


def residual_targets(x, y, k, method, seed):
    classes = np.unique(y)
    means = np.stack([x[y == c].mean(0) for c in classes])
    lookup = {int(c): i for i, c in enumerate(classes)}
    labels = np.array([lookup[int(c)] for c in y])
    r = x - means[labels]
    present = np.ones((len(classes), k), dtype=bool)
    direction = np.empty(0)
    if method == "random":
        direction = np.random.default_rng(seed).normal(size=x.shape[1])
        direction /= np.linalg.norm(direction)
        targets = []
        for i, c in enumerate(classes):
            rr = r[y == c]
            s = rr @ direction
            side = s > 0
            if not side.any() or side.all():
                raise ValueError("Empty root residual mode")
            groups = []
            if k in (2, 4):
                # Split at zero for two modes; bisect each side for four modes.
                for mask in [~side, side]:
                    z = rr[mask][np.argsort(s[mask], kind="stable")]
                    if k == 2:
                        groups.append(z)
                    else:
                        groups.extend([z, z] if len(z) == 1 else np.array_split(z, 2))
            else:
                # Other mode counts partition the ordered residuals into equal-size groups.
                ordered = rr[np.argsort(s, kind="stable")]
                if len(ordered) < k:
                    raise ValueError("Insufficient samples for mode count")
                groups = list(np.array_split(ordered, k))
            targets.append(np.stack([v.mean(0) for v in groups]))
        targets = np.stack(targets)
    else:
        raise ValueError(method)
    return classes, means, targets, present, direction


class TargetBank:
    def __init__(self, x, y, attributes, method, seed):
        self.x = x
        self.y = y
        self.a = attributes
        self.classes = np.unique(y)
        self.method = method
        self.seed = seed
        means = np.stack([x[y == c].mean(0) for c in self.classes])
        self.base = Ridge(attributes[self.classes], means)
        self.res = {}
        self.stats = {}

    def residual(self, k):
        if k not in self.res:
            classes, means, targets, present, direction = residual_targets(
                self.x, self.y, k, self.method, self.seed
            )
            solvers = []
            for j in range(k):
                mask = present[:, j]
                if mask.sum() < 2:
                    raise ValueError("Insufficient classes for a transferable mode")
                solvers.append(Ridge(self.a[classes[mask]], targets[mask, j]))
            self.res[k] = solvers
            self.stats[k] = {
                "missing_class_mode_targets": int((~present).sum()),
                "classes_per_mode": present.sum(0).tolist(),
            }
        return self.res[k]

    def model(self, c):
        w, b = self.base.solve(c["alpha_base"])
        model = {"weight": w, "bias": b}
        if c["modes"] > 1:
            pairs = [r.solve(c["alpha_residual"]) for r in self.residual(c["modes"])]
            model.update(
                residual_weight=np.stack([v[0] for v in pairs], 1),
                residual_bias=np.stack([v[1] for v in pairs]),
            )
        return model
