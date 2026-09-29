"""Portable dataset metadata and role-restricted frozen-prefix caches."""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .io import sha256


def index_array(value, size, name, *, allow_empty=False):
    array = np.asarray(value)
    if array.ndim != 1 or not np.issubdtype(array.dtype, np.integer):
        raise ValueError(f"{name} must be a one-dimensional integer array")
    if not len(array) and not allow_empty:
        raise ValueError(f"{name} cannot be empty")
    if len(array) and (array.min() < 0 or array.max() >= size):
        raise ValueError(f"{name} contains an out-of-range index")
    if len(np.unique(array)) != len(array):
        raise ValueError(f"{name} contains duplicate indices")
    return array.astype(np.int64, copy=False)


@dataclass
class DatasetBundle:
    path: Path
    labels: np.ndarray
    semantics: np.ndarray
    train: np.ndarray
    test_seen: np.ndarray
    test_unseen: np.ndarray
    class_names: np.ndarray | None = None
    groups: np.ndarray | None = None
    images: list[str] | None = None

    @property
    def root(self):
        return self.path.parent

    @property
    def seen(self):
        return np.unique(self.labels[self.train])

    @property
    def unseen(self):
        return np.unique(self.labels[self.test_unseen])

    @property
    def test(self):
        return np.concatenate((self.test_seen, self.test_unseen))

    @classmethod
    def load(cls, path):
        path = Path(path)
        if path.is_dir():
            path = path / "metadata.npz"
        with np.load(path, allow_pickle=False) as arrays:
            required = ("labels", "semantics", "train", "test_seen", "test_unseen")
            missing = set(required) - set(arrays.files)
            if missing:
                raise ValueError(f"Missing metadata fields: {sorted(missing)}")
            values = {key: arrays[key].copy() for key in required}
            values.update(
                {
                    key: arrays[key].copy()
                    for key in ("class_names", "groups")
                    if key in arrays.files
                }
            )
        bundle = cls(path=path, **values)
        images_path = path.parent / "images.json"
        if images_path.exists():
            bundle.images = json.loads(images_path.read_text())
        bundle.validate()
        return bundle

    def validate(self):
        if self.labels.ndim != 1 or not np.issubdtype(self.labels.dtype, np.integer):
            raise ValueError("labels must be a one-dimensional integer array")
        if self.semantics.ndim != 2 or not np.issubdtype(self.semantics.dtype, np.number):
            raise ValueError("semantics must have shape [classes, semantic dimension]")
        if not np.isfinite(self.semantics).all() or np.any(
            np.linalg.norm(self.semantics, axis=1) == 0
        ):
            raise ValueError("semantics contains nonfinite or zero class vectors")
        n, classes = len(self.labels), len(self.semantics)
        if not n or self.labels.min() < 0 or self.labels.max() >= classes:
            raise ValueError("labels must index semantic rows using zero-based class IDs")
        for role in ("train", "test_seen", "test_unseen"):
            setattr(self, role, index_array(getattr(self, role), n, role))
        for left, right in ((self.train, self.test), (self.test_seen, self.test_unseen)):
            if np.intersect1d(left, right).size:
                raise ValueError("Training and official test roles must be disjoint")
        if np.intersect1d(self.seen, self.unseen).size:
            raise ValueError("Seen and unseen classes overlap")
        if not np.array_equal(np.unique(self.labels[self.test_seen]), self.seen):
            raise ValueError("test_seen class IDs differ from training classes")
        if not np.array_equal(np.sort(np.r_[self.seen, self.unseen]), np.arange(classes)):
            raise ValueError("Every semantic row must belong to the seen/unseen partition")
        if self.class_names is not None:
            if self.class_names.shape != (classes,) or self.class_names.dtype.kind not in "US":
                raise ValueError("class_names must be one string per semantic row")
            if len(set(self.class_names.tolist())) != classes:
                raise ValueError("Class names must be unique")
        if self.groups is not None:
            if self.groups.shape != (n,) or self.groups.dtype.kind not in "USiu":
                raise ValueError("groups must contain one string or integer per image")
            if np.intersect1d(self.groups[self.train], self.groups[self.test]).size:
                raise ValueError("A leakage group occurs in both training and official test")
        if self.images is not None:
            if not isinstance(self.images, list) or len(self.images) != n:
                raise ValueError("images.json must contain one relative path per image row")
            for value in self.images:
                if (
                    not isinstance(value, str)
                    or not value
                    or Path(value).is_absolute()
                    or ".." in Path(value).parts
                ):
                    raise ValueError("Image paths must be relative to the image root")

    def validate_fit(self, indices, classes=None):
        indices = index_array(indices, len(self.labels), "fit")
        if not np.isin(indices, self.train).all():
            raise ValueError("Only original seen training rows are permitted for fitting")
        if classes is not None and not np.isin(self.labels[indices], classes).all():
            raise ValueError("A pseudo-unseen class is present in fit data")
        return indices


def load_fold(bundle, path):
    """Load one development fold, whose indices refer to the full image table."""
    record = json.loads(Path(path).read_text())
    return validate_fold(bundle, record)


def validate_fold(bundle, record):
    required = ("fit", "cal", "tune", "pseudo_seen", "pseudo_unseen")
    if not set(required).issubset(record):
        raise ValueError("Fold requires fit/cal/tune and pseudo_seen/pseudo_unseen")
    fold = {}
    for key in required + (("confirm",) if "confirm" in record else ()):
        size = len(bundle.semantics) if key.startswith("pseudo_") else len(bundle.labels)
        fold[key] = index_array(np.asarray(record[key]), size, key)
    ps, pu = fold["pseudo_seen"], fold["pseudo_unseen"]
    if np.intersect1d(ps, pu).size or not np.array_equal(np.sort(np.r_[ps, pu]), bundle.seen):
        raise ValueError("Pseudo classes must partition original seen classes")
    bundle.validate_fit(fold["fit"], ps)
    if not np.array_equal(np.unique(bundle.labels[fold["fit"]]), np.sort(ps)):
        raise ValueError("Every pseudo-seen class must occur in fit")
    roles = [key for key in ("fit", "cal", "tune", "confirm") if key in fold]
    for position, role in enumerate(roles):
        if not np.isin(fold[role], bundle.train).all():
            raise ValueError("Official test data cannot enter development folds")
        for other in roles[:position]:
            if np.intersect1d(fold[role], fold[other]).size:
                raise ValueError("Development roles overlap")
            if (
                bundle.groups is not None
                and np.intersect1d(bundle.groups[fold[role]], bundle.groups[fold[other]]).size
            ):
                raise ValueError("A leakage group crosses development roles")
    return fold


class PrefixCache:
    """Read frozen raw transformer tokens for exactly one authorized role."""

    def __init__(self, folder, bundle, role, verify_hashes=True):
        if role not in ("train", "test"):
            raise ValueError("Cache role must be train or test")
        folder = Path(folder)
        self.manifest = json.loads((folder / "manifest.json").read_text())
        if self.manifest.get("role") != role or self.manifest.get("status") != "complete":
            raise ValueError("Incomplete cache or incorrect role")
        if self.manifest.get("metadata_sha256") != sha256(bundle.path):
            raise ValueError("Cache was generated from different dataset metadata")
        self.indices = np.load(folder / "indices.npy", allow_pickle=False)
        expected = bundle.train if role == "train" else bundle.test
        if not np.array_equal(self.indices, expected):
            raise ValueError("Cached feature row ordering differs from the requested split")
        self.raw = np.load(folder / "raw.npy", mmap_mode="r", allow_pickle=False)
        if self.raw.ndim != 3 or self.raw.shape[0] != len(expected) or self.raw.dtype != np.float32:
            raise ValueError("Cache must contain float32 [images, tokens, hidden dimension]")
        if list(self.raw.shape) != self.manifest.get("shape"):
            raise ValueError("Cache shape differs from its manifest")
        if verify_hashes:
            for name in ("indices.npy", "raw.npy"):
                if sha256(folder / name) != self.manifest.get("sha256", {}).get(name):
                    raise ValueError(f"Cache checksum mismatch: {name}")
        self.lookup = {int(value): row for row, value in enumerate(self.indices)}

    def take(self, indices):
        indices = np.asarray(indices)
        if indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer):
            raise ValueError("Requested rows must be a one-dimensional integer array")
        try:
            positions = [self.lookup[int(value)] for value in indices]
        except KeyError as error:
            raise ValueError("Requested row is outside this cache's authorized role") from error
        values = np.array(self.raw[positions], dtype=np.float32)
        if not np.isfinite(values).all():
            raise ValueError("Nonfinite cached tokens")
        return values
