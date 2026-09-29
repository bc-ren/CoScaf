"""Import user-supplied data and construct reproducible development metadata."""

import json
from pathlib import Path

import numpy as np
from scipy.io import loadmat

from .data import DatasetBundle, index_array, sha256, validate_fold


def _mat_indices(value, name):
    values = np.asarray(value).ravel()
    if not np.issubdtype(values.dtype, np.number) or not np.isfinite(values).all():
        raise ValueError(f"{name} must contain finite MATLAB indices")
    if not len(values) or (values < 1).any() or not np.equal(values, np.floor(values)).all():
        raise ValueError(f"{name} must contain positive one-based MATLAB indices")
    return values.astype(np.int64) - 1


def _mat_names(value):
    names = []
    for item in np.asarray(value).ravel():
        while isinstance(item, np.ndarray) and item.size == 1:
            item = item.item()
        names.append(str(item))
    return np.asarray(names, dtype=str)


def _write_bundle(output, arrays, provenance, images_file=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    path = output / "metadata.npz"
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite dataset metadata: {path}")
    images = json.loads(Path(images_file).read_text()) if images_file else None
    fields = {
        key: arrays[key] for key in ("labels", "semantics", "train", "test_seen", "test_unseen")
    }
    fields.update({key: arrays[key] for key in ("class_names", "groups") if key in arrays})
    DatasetBundle(path, **fields, images=images).validate()
    np.savez_compressed(path, **arrays)
    if images is not None:
        (output / "images.json").write_text(json.dumps(images, indent=2) + "\n")
    provenance["metadata_sha256"] = sha256(path)
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    return DatasetBundle.load(output)


def import_natural(
    labels_file,
    splits_file,
    output,
    *,
    dataset,
    semantics_file=None,
    exclude_indices=None,
    images_file=None,
):
    """Convert official MATLAB membership and semantics without changing test rows.

    ``exclude_indices`` optionally names a JSON list or .npy file of zero-based
    training rows quarantined by a documented duplicate audit.
    """
    dimensions = {"AWA2": 85, "CUB": 1024, "SUN": 102}
    if dataset not in dimensions:
        raise ValueError("Natural dataset must be AWA2, CUB, or SUN")
    if dataset == "CUB" and semantics_file is None:
        raise ValueError("CUB requires the 1024D sent_splits.mat semantic file")
    split = loadmat(splits_file)
    semantic_path = Path(semantics_file or splits_file)
    source = loadmat(semantic_path)
    labels = _mat_indices(loadmat(labels_file, variable_names=["labels"])["labels"], "labels")
    names = _mat_names(split["allclasses_names"])
    if not np.array_equal(_mat_names(source["allclasses_names"]), names):
        raise ValueError("Semantic class ordering differs from official class ordering")
    for key in ("train_loc", "val_loc", "trainval_loc", "test_seen_loc", "test_unseen_loc"):
        if key not in source or key not in split or not np.array_equal(source[key], split[key]):
            raise ValueError(f"Semantic source official split mismatch: {key}")
    attributes = np.asarray(source["att"], dtype=np.float64)
    if attributes.shape != (dimensions[dataset], len(names)):
        raise ValueError("Wrong semantic matrix orientation or dimension")
    semantics = attributes.T.copy()
    if dataset != "CUB":
        original = np.asarray(split["original_att"])
        if original.shape != attributes.shape:
            raise ValueError("original_att must match the attribute matrix shape")
        semantics[original.T < 0] = 0
    arrays = {"labels": labels, "semantics": semantics, "class_names": names}
    for key, field in (
        ("train", "trainval_loc"),
        ("test_seen", "test_seen_loc"),
        ("test_unseen", "test_unseen_loc"),
    ):
        arrays[key] = _mat_indices(split[field], field)
    excluded = np.empty(0, dtype=np.int64)
    if exclude_indices is not None:
        file = Path(exclude_indices)
        excluded = (
            np.load(file, allow_pickle=False)
            if file.suffix == ".npy"
            else np.asarray(json.loads(file.read_text()))
        )
        excluded = index_array(excluded, len(labels), "excluded_train", allow_empty=True)
        if not np.isin(excluded, arrays["train"]).all():
            raise ValueError("Only official training rows may be quarantined")
        arrays["train"] = arrays["train"][~np.isin(arrays["train"], excluded)]
    arrays["excluded_train"] = excluded
    provenance = {
        "dataset": dataset,
        "index_base": 0,
        "source_index_base": 1,
        "labels_sha256": sha256(labels_file),
        "splits_sha256": sha256(splits_file),
        "semantics_sha256": sha256(semantic_path),
        "excluded_training_rows": excluded.tolist(),
        "official_test_membership_and_order_preserved": True,
    }
    return _write_bundle(output, arrays, provenance, images_file)


def import_retinal(export_file, output, *, images_file=None):
    """Import an already de-identified, zero-based RetiRareV2 metadata export."""
    with np.load(export_file, allow_pickle=False) as archive:
        keys = ("labels", "semantics", "train", "test_seen", "test_unseen", "class_names", "groups")
        arrays = {key: archive[key].copy() for key in keys if key in archive.files}
    if (
        "semantics" not in arrays
        or arrays["semantics"].ndim != 2
        or arrays["semantics"].shape[1] != 512
    ):
        raise ValueError("RetiRareV2 requires 512D official RetiZero projected text vectors")
    if not {"labels", "train", "test_seen", "test_unseen"}.issubset(arrays):
        raise ValueError("Retinal export lacks labels or fixed train/test membership")
    return _write_bundle(
        output,
        arrays,
        {
            "dataset": "RetiRareV2",
            "index_base": 0,
            "export_sha256": sha256(export_file),
            "semantic_variant": "RetiZero disease names",
        },
        images_file,
    )


def make_development_folds(bundle, nfolds):
    """Generate the fixed random-class and semantic-PC1 natural-data folds.

    Per-image roles are consistent across the two class-partition families.
    Use supplied group-disjoint folds instead for grouped retinal datasets.
    """
    if bundle.groups is not None:
        raise ValueError("Grouped data requires explicitly supplied group-disjoint folds")
    if not 2 <= nfolds <= len(bundle.seen):
        raise ValueError("nfolds must be between 2 and the number of seen classes")
    per_class, confirm = {}, []
    for category in bundle.seen:
        rows = bundle.train[bundle.labels[bundle.train] == category].copy()
        np.random.default_rng(9919 + int(category)).shuffle(rows)
        n = max(2, int(round(len(rows) * 0.25)))
        held, remainder = rows[:n], rows[n:]
        validation = held[n // 2 :]
        if len(validation) < 2 or len(remainder) < 2:
            raise ValueError("Class too small for the fixed four-role development protocol")
        shuffled = np.random.default_rng(20260918 + int(category)).permutation(validation)
        confirm.extend(shuffled[: max(1, len(shuffled) // 2)])
        per_class[int(category)] = (remainder, held[: n // 2], validation)
    confirm = np.sort(confirm)
    semantics = bundle.semantics[bundle.seen].astype(np.float64)
    semantics /= np.linalg.norm(semantics, axis=1, keepdims=True)
    centered = semantics - semantics.mean(0)
    _, _, vectors = np.linalg.svd(centered, full_matrices=False)
    direction = vectors[0]
    direction *= 1 if direction[np.argmax(np.abs(direction))] >= 0 else -1
    orders = [
        np.random.default_rng(1947).permutation(bundle.seen),
        bundle.seen[np.lexsort((bundle.seen, centered @ direction))],
    ]
    folds = []
    for order in orders:
        for pu in np.array_split(order, nfolds):
            ps = np.setdiff1d(bundle.seen, pu)
            fit, cal, tune = [], [], []
            for category in bundle.seen:
                remainder, held, validation = per_class[int(category)]
                cal.extend(held)
                tune.extend(np.setdiff1d(validation, confirm))
                if category in ps:
                    fit.extend(remainder)
                else:
                    cal.extend(remainder[: len(remainder) // 2])
                    tune.extend(remainder[len(remainder) // 2 :])
            fold = {
                key: np.asarray(sorted(value), dtype=np.int64)
                for key, value in (
                    ("fit", fit),
                    ("cal", cal),
                    ("tune", tune),
                    ("confirm", confirm),
                    ("pseudo_seen", ps),
                    ("pseudo_unseen", pu),
                )
            }
            validate_fold(bundle, fold)
            folds.append({key: value.tolist() for key, value in fold.items()})
    return folds


def write_prefix_cache_manifest(folder, bundle, role, backbone_metadata):
    """Finalize user-generated raw.npy and indices.npy after validating their rows."""
    folder = Path(folder)
    if role not in ("train", "test"):
        raise ValueError("Cache role must be train or test")
    indices = np.load(folder / "indices.npy", allow_pickle=False)
    expected = bundle.train if role == "train" else bundle.test
    if not np.array_equal(indices, expected):
        raise ValueError("Prefix cache indices must match the exact ordered role")
    raw = np.load(folder / "raw.npy", mmap_mode="r", allow_pickle=False)
    if raw.ndim != 3 or raw.shape[0] != len(indices) or raw.dtype != np.float32:
        raise ValueError("Expected float32 raw tokens with one row per role image")
    for start in range(0, len(raw), 64):
        if not np.isfinite(raw[start : start + 64]).all():
            raise ValueError("Prefix cache contains nonfinite values")
    manifest = {
        "status": "complete",
        "role": role,
        "shape": list(raw.shape),
        "dtype": str(raw.dtype),
        "metadata_sha256": sha256(bundle.path),
        "backbone": backbone_metadata,
        "sha256": {name: sha256(folder / name) for name in ("indices.npy", "raw.npy")},
    }
    (folder / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest
