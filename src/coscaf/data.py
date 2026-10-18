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
    test_seen: np.ndarray | None
    test_unseen: np.ndarray | None
    class_names: np.ndarray | None = None
    groups: np.ndarray | None = None
    images: list[str | None] | None = None
    test_indices: np.ndarray | None = None
    declared_seen: np.ndarray | None = None
    declared_unseen: np.ndarray | None = None
    image_sha256: list[str | None] | None = None
    sparse_images: bool = False

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
        if self.test_indices is not None:
            return self.test_indices
        return np.concatenate((self.test_seen, self.test_unseen))

    @classmethod
    def load(cls, path):
        path = Path(path)
        if path.is_dir():
            path = path / "metadata.npz"
        with np.load(path, allow_pickle=False) as arrays:
            required = ("labels", "semantics", "train")
            missing = set(required) - set(arrays.files)
            if missing:
                raise ValueError(f"Missing metadata fields: {sorted(missing)}")
            values = {key: arrays[key].copy() for key in required}
            for key in ("test_seen", "test_unseen"):
                values[key] = arrays[key].copy() if key in arrays.files else None
            for source, target in (
                ("test", "test_indices"),
                ("seen", "declared_seen"),
                ("unseen", "declared_unseen"),
            ):
                if source in arrays.files:
                    values[target] = arrays[source].copy()
            values.update(
                {
                    key: arrays[key].copy()
                    for key in ("class_names", "groups")
                    if key in arrays.files
                }
            )
            if "class_names" not in values and "classes" in arrays.files:
                names = arrays["classes"]
                if names.dtype.kind in "US":
                    values["class_names"] = names.copy()
        bundle = cls(path=path, **values)
        bundle.validate()
        images_path = path.parent / "images.json"
        if images_path.exists():
            bundle.images = json.loads(images_path.read_text())
        elif (path.parent / "images.jsonl").exists():
            bundle._load_image_manifest(path.parent / "images.jsonl")
        bundle.validate()
        return bundle

    def _load_image_manifest(self, path):
        """Retain original row IDs; excluded rows need not have an image record."""
        self.images = [None] * len(self.labels)
        self.image_sha256 = [None] * len(self.labels)
        self.sparse_images = True
        with Path(path).open() as stream:
            for line in stream:
                if not line.strip():
                    continue
                record = json.loads(line)
                if not isinstance(record, dict) or not {
                    "index",
                    "image_path",
                    "class_id",
                    "sha256",
                }.issubset(record):
                    raise ValueError("Image manifest requires index/image_path/class_id/sha256")
                index, class_id = record["index"], record["class_id"]
                if type(index) is not int or not 0 <= index < len(self.labels):
                    raise ValueError("Image manifest index is out-of-range or not an integer")
                if self.image_sha256[index] is not None:
                    raise ValueError("Image manifest contains duplicate indices")
                if type(class_id) is not int or class_id != int(self.labels[index]):
                    raise ValueError("Image manifest class_id differs from metadata labels")
                digest = record["sha256"]
                if (
                    not isinstance(digest, str)
                    or len(digest) != 64
                    or any(c not in "0123456789abcdefABCDEF" for c in digest)
                ):
                    raise ValueError("Image manifest sha256 must be a 64-character hex digest")
                self.images[index] = record["image_path"]
                self.image_sha256[index] = digest

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
        self.train = index_array(self.train, n, "train")
        for name in ("declared_seen", "declared_unseen"):
            if getattr(self, name) is not None:
                setattr(self, name, index_array(getattr(self, name), classes, name))
        if self.declared_seen is not None and self.declared_unseen is not None:
            partition = np.r_[self.declared_seen, self.declared_unseen]
            if not np.array_equal(np.sort(partition), np.arange(classes)):
                raise ValueError("Explicit seen/unseen classes must partition semantic rows")
        if self.test_indices is not None:
            self.test_indices = index_array(self.test_indices, n, "test")
        if self.test_seen is None or self.test_unseen is None:
            if (
                self.test_indices is None
                or self.declared_seen is None
                or self.declared_unseen is None
            ):
                raise ValueError(
                    "Missing test splits require test and explicit seen/unseen classes"
                )
            for role, ids in (
                ("test_seen", self.declared_seen),
                ("test_unseen", self.declared_unseen),
            ):
                if getattr(self, role) is None:
                    setattr(
                        self, role, self.test_indices[np.isin(self.labels[self.test_indices], ids)]
                    )
        for role in ("test_seen", "test_unseen"):
            setattr(self, role, index_array(getattr(self, role), n, role))
        if self.test_indices is not None and not np.array_equal(
            np.sort(np.r_[self.test_seen, self.test_unseen]), np.sort(self.test_indices)
        ):
            raise ValueError("Explicit test membership differs from test_seen/test_unseen")
        for left, right in ((self.train, self.test), (self.test_seen, self.test_unseen)):
            if np.intersect1d(left, right).size:
                raise ValueError("Training and official test roles must be disjoint")
        if np.intersect1d(self.seen, self.unseen).size:
            raise ValueError("Seen and unseen classes overlap")
        if not np.array_equal(np.unique(self.labels[self.test_seen]), self.seen):
            raise ValueError("test_seen class IDs differ from training classes")
        if not np.array_equal(np.sort(np.r_[self.seen, self.unseen]), np.arange(classes)):
            raise ValueError("Every semantic row must belong to the seen/unseen partition")
        for declared, actual in (
            (self.declared_seen, self.seen),
            (self.declared_unseen, self.unseen),
        ):
            if declared is not None and not np.array_equal(np.sort(declared), actual):
                raise ValueError("Explicit seen/unseen classes differ from split labels")
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
            active = set(np.r_[self.train, self.test].tolist())
            for index, value in enumerate(self.images):
                if value is None and self.sparse_images and index not in active:
                    continue
                if (
                    not isinstance(value, str)
                    or not value
                    or Path(value).is_absolute()
                    or ".." in Path(value).parts
                    or "\\" in value
                ):
                    raise ValueError("Image paths must be relative; active rows require an image")

    def validate_fit(self, indices, classes=None):
        indices = index_array(indices, len(self.labels), "fit")
        if not np.isin(indices, self.train).all():
            raise ValueError("Only original seen training rows are permitted for fitting")
        if classes is not None and not np.isin(self.labels[indices], classes).all():
            raise ValueError("A pseudo-unseen class is present in fit data")
        return indices


def load_fold(bundle, path):
    """Load one development fold, whose indices refer to the full image table."""
    path = Path(path)
    if path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            record = {key: archive[key].copy() for key in archive.files}
    else:
        record = json.loads(path.read_text())
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
        array_name = self.manifest.get("array", "raw.npy")
        if (
            not isinstance(array_name, str)
            or not array_name
            or Path(array_name).name != array_name
            or "\\" in array_name
            or Path(array_name).suffix != ".npy"
        ):
            raise ValueError("Cache array must be a relative .npy filename inside its folder")
        array_path = folder / array_name
        if not array_path.resolve().is_relative_to(folder.resolve()):
            raise ValueError("Cache array must remain inside its folder")
        self.raw = np.load(array_path, mmap_mode="r", allow_pickle=False)
        if self.raw.ndim != 3 or self.raw.shape[0] != len(expected) or self.raw.dtype != np.float32:
            raise ValueError("Cache must contain float32 [images, tokens, hidden dimension]")
        if list(self.raw.shape) != self.manifest.get("shape"):
            raise ValueError("Cache shape differs from its manifest")
        if verify_hashes:
            for name in ("indices.npy", array_name):
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
