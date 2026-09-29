"""Synthetic tests for index conversion, semantic alignment, and data isolation."""

import json

import numpy as np
import pytest
from scipy.io import savemat

from coscaf.data import DatasetBundle, PrefixCache, load_fold, validate_fold
from coscaf.preparation import (
    import_natural,
    import_retinal,
    make_development_folds,
    write_prefix_cache_manifest,
)


def metadata():
    labels = np.repeat(np.arange(4), 20)
    return {
        "labels": labels,
        "semantics": np.arange(4 * 85, dtype=float).reshape(4, 85) + 1,
        "train": np.r_[np.arange(16), np.arange(20, 36), np.arange(40, 56)],
        "test_seen": np.r_[np.arange(16, 20), np.arange(36, 40), np.arange(56, 60)],
        "test_unseen": np.arange(60, 80),
        "class_names": np.array(["class_a", "class_b", "class_c", "class_d"]),
    }


@pytest.fixture
def bundle(tmp_path):
    path = tmp_path / "bundle"
    path.mkdir()
    np.savez_compressed(path / "metadata.npz", **metadata())
    return DatasetBundle.load(path)


def test_bundle_infers_partition_and_refuses_test_fit(bundle):
    np.testing.assert_array_equal(bundle.seen, [0, 1, 2])
    np.testing.assert_array_equal(bundle.unseen, [3])
    bundle.validate_fit(np.array([0, 20, 40]))
    with pytest.raises(ValueError, match="original seen"):
        bundle.validate_fit(bundle.test)
    with pytest.raises(ValueError, match="pseudo-unseen"):
        bundle.validate_fit(np.array([0, 20]), [0])


@pytest.mark.parametrize(
    "change,match",
    [
        (lambda a: a.update(train=np.r_[a["train"], 60]), "disjoint"),
        (lambda a: a.update(train=np.r_[a["train"], 0]), "duplicate"),
        (lambda a: a.update(test_seen=np.array([80])), "out-of-range"),
        (lambda a: a.update(semantics=np.zeros((4, 85))), "zero"),
        (lambda a: a.update(class_names=np.array(["x"] * 4)), "unique"),
        (lambda a: a.update(groups=np.array(["same"] * 80)), "leakage group"),
    ],
)
def test_invalid_bundle_is_rejected(tmp_path, change, match):
    arrays = metadata()
    change(arrays)
    np.savez_compressed(tmp_path / "metadata.npz", **arrays)
    with pytest.raises(ValueError, match=match):
        DatasetBundle.load(tmp_path)


def test_fixed_folds_are_deterministic_and_globally_disjoint(bundle, tmp_path):
    folds = make_development_folds(bundle, 3)
    assert len(folds) == 6
    assert folds == make_development_folds(bundle, 3)
    for fold in folds:
        complete = np.concatenate([fold[key] for key in ("fit", "cal", "tune", "confirm")])
        np.testing.assert_array_equal(np.sort(complete), np.sort(bundle.train))
    cal = set(sum([fold["cal"] for fold in folds], []))
    tune = set(sum([fold["tune"] for fold in folds], []))
    assert not cal & tune
    path = tmp_path / "fold.json"
    path.write_text(json.dumps(folds[0]))
    assert set(load_fold(bundle, path)) == set(folds[0])
    folds[0]["cal"].append(int(bundle.test[0]))
    with pytest.raises(ValueError, match="Official test"):
        validate_fold(bundle, folds[0])


def test_fold_rejects_group_leakage(bundle):
    fold = make_development_folds(bundle, 3)[0]
    bundle.groups = np.arange(len(bundle.labels))
    bundle.groups[fold["cal"][0]] = bundle.groups[fold["fit"][0]]
    with pytest.raises(ValueError, match="leakage group"):
        validate_fold(bundle, fold)
    with pytest.raises(ValueError, match="group-disjoint"):
        make_development_folds(bundle, 3)


def matlab_files(tmp_path, dimension=85):
    arrays = metadata()
    splits = {
        "att": np.arange(dimension * 4, dtype=float).reshape(dimension, 4) + 1,
        "original_att": np.ones((dimension, 4)),
        "allclasses_names": arrays["class_names"].astype(object)[:, None],
        "train_loc": arrays["train"][:20, None] + 1,
        "val_loc": arrays["train"][20:, None] + 1,
        "trainval_loc": arrays["train"][:, None] + 1,
        "test_seen_loc": arrays["test_seen"][:, None] + 1,
        "test_unseen_loc": arrays["test_unseen"][:, None] + 1,
    }
    splits["original_att"][0, 0] = -1
    labels_file, split_file = tmp_path / "labels.mat", tmp_path / "splits.mat"
    savemat(labels_file, {"labels": arrays["labels"][:, None] + 1})
    savemat(split_file, splits)
    return arrays, splits, labels_file, split_file


def test_matlab_conversion_and_training_quarantine_preserve_official_test(tmp_path):
    arrays, splits, labels_file, split_file = matlab_files(tmp_path)
    exclusion = tmp_path / "exclude.json"
    exclusion.write_text("[0]")
    out = import_natural(
        labels_file, split_file, tmp_path / "converted", dataset="AWA2", exclude_indices=exclusion
    )
    np.testing.assert_array_equal(out.labels, arrays["labels"])
    np.testing.assert_array_equal(out.test_seen, arrays["test_seen"])
    np.testing.assert_array_equal(out.test_unseen, arrays["test_unseen"])
    assert 0 not in out.train and len(out.train) == len(arrays["train"]) - 1
    assert out.semantics[0, 0] == 0
    np.testing.assert_array_equal(out.semantics[1:], splits["att"].T[1:])
    exclusion.write_text("[60]")
    with pytest.raises(ValueError, match="official training"):
        import_natural(
            labels_file, split_file, tmp_path / "bad", dataset="AWA2", exclude_indices=exclusion
        )


def test_cub_requires_1024d_and_matching_names_and_split_order(tmp_path):
    _, splits, labels_file, split_file = matlab_files(tmp_path, 1024)
    with pytest.raises(ValueError, match="1024D"):
        import_natural(labels_file, split_file, tmp_path / "missing", dataset="CUB")
    semantic = tmp_path / "sent.mat"
    savemat(semantic, splits)
    out = import_natural(
        labels_file, split_file, tmp_path / "good", dataset="CUB", semantics_file=semantic
    )
    np.testing.assert_array_equal(out.semantics, splits["att"].T)
    for key in ("allclasses_names", "test_seen_loc"):
        changed = {**splits, key: splits[key][::-1]}
        savemat(semantic, changed)
        with pytest.raises(ValueError, match="ordering|split mismatch"):
            import_natural(
                labels_file, split_file, tmp_path / key, dataset="CUB", semantics_file=semantic
            )
    savemat(semantic, {**splits, "att": splits["att"][:312]})
    with pytest.raises(ValueError, match="orientation or dimension"):
        import_natural(
            labels_file, split_file, tmp_path / "wrong_dim", dataset="CUB", semantics_file=semantic
        )


def test_retinal_import_preserves_vectors_and_anonymous_groups(tmp_path):
    arrays = metadata()
    arrays["semantics"] = np.ones((4, 512), dtype=np.float32)
    arrays["groups"] = np.array([f"group_{i}" for i in range(80)])
    file = tmp_path / "export.npz"
    np.savez_compressed(file, **arrays)
    out = import_retinal(file, tmp_path / "retinal")
    np.testing.assert_array_equal(out.semantics, arrays["semantics"])
    np.testing.assert_array_equal(out.groups, arrays["groups"])


def test_cache_membership_order_integrity_and_access_guard(bundle, tmp_path):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    np.save(cache_dir / "indices.npy", bundle.train)
    raw = np.arange(len(bundle.train) * 5 * 3, dtype=np.float32).reshape(-1, 5, 3)
    np.save(cache_dir / "raw.npy", raw)
    write_prefix_cache_manifest(cache_dir, bundle, "train", {"model": "synthetic"})
    cache = PrefixCache(cache_dir, bundle, "train")
    np.testing.assert_array_equal(cache.take(bundle.train[[2, 0, 2]]), raw[[2, 0, 2]])
    with pytest.raises(ValueError, match="authorized role"):
        cache.take(bundle.test[:1])
    with pytest.raises(ValueError, match="incorrect role"):
        PrefixCache(cache_dir, bundle, "test")
    raw[0, 0, 0] = -1
    np.save(cache_dir / "raw.npy", raw)
    with pytest.raises(ValueError, match="checksum"):
        PrefixCache(cache_dir, bundle, "train")


def test_image_paths_must_be_relative(bundle):
    (bundle.root / "images.json").write_text(json.dumps(["../image.jpg"] * len(bundle.labels)))
    with pytest.raises(ValueError, match="relative"):
        DatasetBundle.load(bundle.root)
