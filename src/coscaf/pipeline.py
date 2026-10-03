"""Validation calibration, seen-class training, and test-time evaluation."""

import hashlib
from pathlib import Path

import numpy as np
import torch

from .adaptation import adapt_evidence
from .backbones import RETIZERO_REVISION, VIT_ID, VIT_REVISION, VIT_WEIGHTS_SHA256
from .data import DatasetBundle, PrefixCache, load_fold
from .io import cpu_state, environment, fingerprint, read_json, sha256, verify_files, write_json
from .metrics import evaluate, evaluate_folds, joint_calibration
from .training import build_from_state, make_model, train_fit


def _check_config(bundle, config):
    if bundle.semantics.shape[1] != config["semantics_dim"]:
        raise ValueError("Semantic dimension differs from the dataset configuration")
    if config["source"]["modes"] < 1:
        raise ValueError("At least one mode is required")


def _check_cache(cache, config):
    """Reject caches from a different representation before fitting anything."""
    metadata = cache.manifest["backbone"]
    if config["backbone"] not in ("vit_base", "retizero"):
        raise ValueError("Backbone must be vit_base or retizero")
    natural = config["backbone"] == "vit_base"
    expected = {
        "id": VIT_ID if natural else "RetiZero",
        "revision": VIT_REVISION if natural else RETIZERO_REVISION,
        "resolution": config["source"]["resolution"],
        "prefix_block": 10 if natural else 22,
    }
    if any(metadata.get(key) != value for key, value in expected.items()):
        raise ValueError(
            "Cached backbone, revision, resolution, or layer differs from configuration"
        )
    if natural and metadata.get("weights_sha256") != VIT_WEIGHTS_SHA256:
        raise ValueError("Cached ViT weights differ from the pinned backbone")
    if not metadata.get("weights_sha256"):
        raise ValueError("Cache manifest must identify the pretrained checkpoint checksum")
    shape = ((expected["resolution"] // 16) ** 2 + 1, config["visual_dim"])
    if cache.raw.shape[1:] != shape:
        raise ValueError("Cached token shape differs from the configured backbone")


def _checked_factory(factory, cache):
    def build(device):
        suffix = factory(device)
        provenance = getattr(suffix, "backbone_provenance", {})
        weights = provenance.get("weights_sha256")
        if weights != cache.manifest["backbone"].get("weights_sha256"):
            raise ValueError("Frozen suffix weights differ from the prefix cache")
        for key in ("config_sha256", "processor_sha256", "source_sha256"):
            if provenance.get(key) != cache.manifest["backbone"].get(key):
                raise ValueError(f"Frozen suffix and prefix cache differ: {key}")
        return suffix

    return build


def _reader(cache, device):
    def read(indices):
        return torch.as_tensor(cache.take(indices), dtype=torch.float32, device=device)

    return read


@torch.no_grad()
def unprompted_means(suffix_factory, reader, indices, batch, device):
    """Fixed backbone evidence for fit-only whitening and ridge initialization."""
    suffix = suffix_factory(device).eval()
    suffix.enabled = False
    values = []
    for start in range(0, len(indices), batch):
        patch = suffix(reader(indices[start : start + batch]))
        if isinstance(patch, tuple):
            patch = patch[0]
        values.append(patch.mean(1).cpu().numpy())
    del suffix
    return np.concatenate(values)


@torch.no_grad()
def score(model, reader, indices, classes, batch, device):
    ids = torch.as_tensor(classes, device=device)
    rows = []
    for start in range(0, len(indices), batch):
        values = model(reader(indices[start : start + batch]), ids)
        if not torch.isfinite(values).all():
            raise FloatingPointError("Nonfinite source scores")
        rows.append(values.cpu().numpy())
    return np.concatenate(rows).astype(np.float64)


@torch.no_grad()
def extract_evidence(model, reader, indices, classes, folder, batch, device):
    """Cache complete, early, and late evidence without labels.

    Disk-backed arrays avoid keeping the N x C x K x D local evidence tensor
    in RAM. ``view10``/``view12`` denote early/late views; for RetiZero these
    correspond to blocks 22 and 24.
    """
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=False)
    ids = torch.as_tensor(classes, device=device)
    arrays = {}
    for start in range(0, len(indices), batch):
        raw = reader(indices[start : start + batch])
        section = slice(start, start + len(raw))
        for view_index, patch in enumerate(model.suffix.layer_views(raw)):
            features = model.head.features(patch, ids)
            logits = model.head.score_features(features, ids)
            if view_index == 0:
                torch.testing.assert_close(logits, model(raw, ids), atol=2e-6, rtol=0)
                values = {**features, "source_scores": logits}
            else:
                prefix = "view10" if view_index == 1 else "view12"
                values = {prefix + "_global": features["global"], prefix + "_scores": logits}
            for key, value in values.items():
                if not torch.isfinite(value).all():
                    raise FloatingPointError(f"Nonfinite evidence: {key}")
                if key not in arrays:
                    arrays[key] = np.lib.format.open_memmap(
                        folder / (key + ".npy"),
                        mode="w+",
                        dtype=np.float32,
                        shape=(len(indices), *value.shape[1:]),
                    )
                arrays[key][section] = value.cpu().numpy()
    for array in arrays.values():
        array.flush()
    write_json(folder / "manifest.json", {"rows": len(indices), "classes": len(classes)})
    return arrays


def _adapt(model, cache, classes, seen, gamma, config, device):
    return adapt_evidence(
        model.head,
        torch.as_tensor(classes, device=device),
        {key: value for key, value in cache.items() if key != "source_scores"},
        np.isin(classes, seen),
        gamma,
        config["adaptation"],
        config["adaptation_setting"],
        device,
        batch=config["adaptation_batch"],
    )


def _fresh_fit(bundle, reader, fit, means, config, suffix_factory, device):
    bundle.validate_fit(fit)
    # Initialization never reads a previously trained task checkpoint.
    model = make_model(
        bundle.semantics,
        means.astype(np.float64),
        bundle.labels[fit],
        config["source"],
        config["seed"],
        suffix_factory,
        device,
    )
    return train_fit(model, reader, fit, bundle.labels, config["source"], config["seed"], device)


def develop(config, data_dir, cache_dir, output, suffix_factory, device="cpu"):
    """Fit independent seen-only replicas, then freeze source/TTA calibration."""
    bundle = DatasetBundle.load(data_dir)
    _check_config(bundle, config)
    cache = PrefixCache(cache_dir, bundle, "train")
    _check_cache(cache, config)
    suffix_factory = _checked_factory(suffix_factory, cache)
    read = _reader(cache, device)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "config.json", config)
    write_json(output / "environment.json", environment())
    batch = config["source"]["score_batch_size"]
    means = unprompted_means(suffix_factory, read, bundle.train, batch, device)
    lookup = {int(index): position for position, index in enumerate(bundle.train)}
    packs = []
    folds = [
        load_fold(bundle, Path(data_dir) / "folds" / f"fold_{i}.json") for i in config["folds"]
    ]
    all_cal = np.concatenate([fold["cal"] for fold in folds])
    all_tune = np.concatenate([fold["tune"] for fold in folds])
    if np.intersect1d(all_cal, all_tune).size:
        raise ValueError("Pooled calibration and selection rows overlap across development folds")
    if (
        bundle.groups is not None
        and np.intersect1d(bundle.groups[all_cal], bundle.groups[all_tune]).size
    ):
        raise ValueError("A group crosses pooled calibration and selection roles")
    source_hashes = {}
    for fold_id, fold in zip(config["folds"], folds):
        destination = output / f"fold_{fold_id}"
        destination.mkdir()
        model, history = _fresh_fit(
            bundle,
            read,
            fold["fit"],
            means[[lookup[int(i)] for i in fold["fit"]]],
            config,
            suffix_factory,
            device,
        )
        torch.save(cpu_state(model.trainable_state()), destination / "model.pt")
        write_json(destination / "history.json", history)
        pack = {"classes": bundle.seen, "seen": fold["pseudo_seen"]}
        for role in ("cal", "tune"):
            z = score(model, read, fold[role], bundle.seen, batch, device)
            pack[role + "_base"] = z
            pack[role + "_delta"] = np.zeros_like(z)
            pack[role + "_y"] = bundle.labels[fold[role]]
        np.savez_compressed(destination / "source_scores.npz", **pack)
        packs.append(pack)
        for name in ("model.pt", "source_scores.npz"):
            source_hashes[f"fold_{fold_id}/{name}"] = sha256(destination / name)
        del model
    source_calibration, source_curve = joint_calibration(packs, 0.0)
    np.save(output / "source_calibration_curve.npy", source_curve)
    source_metrics = evaluate_folds(packs, 0.0, source_calibration)
    adapted_packs = []
    for fold_id, fold in zip(config["folds"], folds):
        destination = output / f"fold_{fold_id}"
        state = torch.load(destination / "model.pt", map_location=device, weights_only=True)
        model = build_from_state(state, config["source"], suffix_factory, device)
        model.eval().requires_grad_(False)
        pack = {"classes": bundle.seen, "seen": fold["pseudo_seen"]}
        for role in ("cal", "tune"):
            evidence = extract_evidence(
                model,
                read,
                fold[role],
                bundle.seen,
                destination / (role + "_evidence"),
                batch,
                device,
            )
            result = _adapt(
                model,
                evidence,
                bundle.seen,
                fold["pseudo_seen"],
                source_calibration["gamma"],
                config,
                device,
            )
            z = result["scores"].astype(np.float64)
            pack[role + "_base"] = z
            pack[role + "_delta"] = np.zeros_like(z)
            pack[role + "_y"] = bundle.labels[fold[role]]
            write_json(
                destination / (role + "_adaptation.json"),
                {"history": result["history"], "diagnostics": result["diagnostics"]},
            )
        np.savez_compressed(destination / "adapted_scores.npz", **pack)
        adapted_packs.append(pack)
        source_hashes[f"fold_{fold_id}/adapted_scores.npz"] = sha256(
            destination / "adapted_scores.npz"
        )
        del model
    calibration, curve = joint_calibration(adapted_packs, 0.0)
    np.save(output / "adapted_calibration_curve.npy", curve)
    metrics = evaluate_folds(adapted_packs, 0.0, calibration)
    frozen = {
        "config_sha256": fingerprint(config),
        "metadata_sha256": sha256(Path(data_dir) / "metadata.npz"),
        "training_cache_sha256": fingerprint(cache.manifest),
        "fold_sha256": {
            str(f): sha256(Path(data_dir) / "folds" / f"fold_{f}.json") for f in config["folds"]
        },
        "source_calibration": source_calibration,
        "adapted_calibration": calibration,
        "source_tune_metrics": source_metrics,
        "adapted_tune_metrics": metrics,
        "official_test_used": False,
        "files": source_hashes,
    }
    write_json(output / "calibration.json", frozen)
    return frozen


def train(config, data_dir, cache_dir, development, output, suffix_factory, device="cpu"):
    """Train one fresh model on seen-training rows after development calibration."""
    bundle = DatasetBundle.load(data_dir)
    _check_config(bundle, config)
    calibration = read_json(Path(development) / "calibration.json")
    if calibration["config_sha256"] != fingerprint(config):
        raise ValueError("Configuration differs from frozen development configuration")
    if calibration["metadata_sha256"] != sha256(Path(data_dir) / "metadata.npz"):
        raise ValueError("Dataset metadata differs from development")
    verify_files(development, calibration["files"])
    for fold_id, expected in calibration["fold_sha256"].items():
        if sha256(Path(data_dir) / "folds" / f"fold_{fold_id}.json") != expected:
            raise ValueError("Development fold membership has changed")
    cache = PrefixCache(cache_dir, bundle, "train")
    _check_cache(cache, config)
    if calibration["training_cache_sha256"] != fingerprint(cache.manifest):
        raise ValueError("Training cache differs from the frozen development cache")
    suffix_factory = _checked_factory(suffix_factory, cache)
    read = _reader(cache, device)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    means = unprompted_means(
        suffix_factory, read, bundle.train, config["source"]["score_batch_size"], device
    )
    model, history = _fresh_fit(bundle, read, bundle.train, means, config, suffix_factory, device)
    torch.save(cpu_state(model.trainable_state()), output / "model.pt")
    write_json(output / "config.json", config)
    write_json(output / "history.json", history)
    write_json(output / "calibration.json", calibration)
    write_json(output / "environment.json", environment())
    freeze = {
        "metadata_sha256": calibration["metadata_sha256"],
        "backbone": cache.manifest["backbone"],
        "files": {
            name: sha256(output / name) for name in ("model.pt", "config.json", "calibration.json")
        },
        "training": "fresh initialization; one optimizer trajectory",
        "official_test_used": False,
    }
    write_json(output / "freeze.json", freeze)
    return freeze


def evaluate_run(run, data_dir, cache_dir, output, suffix_factory, device="cpu"):
    """Evaluate source and own-source TTA; labels are used only for metrics."""
    run, output = Path(run), Path(output)
    freeze = read_json(run / "freeze.json")
    verify_files(run, freeze["files"])
    if freeze["metadata_sha256"] != sha256(Path(data_dir) / "metadata.npz"):
        raise ValueError("Official dataset membership has changed")
    config, calibration = read_json(run / "config.json"), read_json(run / "calibration.json")
    bundle = DatasetBundle.load(data_dir)
    cache = PrefixCache(cache_dir, bundle, "test")
    _check_cache(cache, config)
    if cache.manifest["backbone"].get("weights_sha256") != freeze["backbone"].get("weights_sha256"):
        raise ValueError("Test and training features use different pretrained checkpoints")
    suffix_factory = _checked_factory(suffix_factory, cache)
    output.mkdir(parents=True, exist_ok=False)
    state = torch.load(run / "model.pt", map_location=device, weights_only=True)
    model = build_from_state(state, config["source"], suffix_factory, device)
    model.eval().requires_grad_(False)
    classes = np.arange(len(bundle.semantics))
    evidence = extract_evidence(
        model,
        _reader(cache, device),
        bundle.test,
        classes,
        output / "evidence",
        config["source"]["score_batch_size"],
        device,
    )
    result = _adapt(
        model,
        evidence,
        classes,
        bundle.seen,
        calibration["source_calibration"]["gamma"],
        config,
        device,
    )
    torch.save(
        cpu_state({key: value for key, value in result.items() if key != "scores"}),
        output / "adaptation.pt",
    )
    # Only after source/TTA predictions are fixed do official labels enter metrics.
    labels = bundle.labels[bundle.test]
    metrics = {}
    for mode, scores, cal in (
        ("source", evidence["source_scores"], calibration["source_calibration"]),
        ("adapted", result["scores"], calibration["adapted_calibration"]),
    ):
        metrics[mode], curve = evaluate(scores, labels, classes, bundle.seen, cal)
        np.save(output / (mode + "_scores.npy"), scores)
        np.save(output / (mode + "_su_curve.npy"), curve)
    write_json(output / "metrics.json", metrics)
    write_json(
        output / "provenance.json",
        {
            "seed": config["seed"],
            "source_model_sha256": sha256(run / "model.pt"),
            "freeze_sha256": sha256(run / "freeze.json"),
            "test_indices_sha256": hashlib.sha256(bundle.test.tobytes()).hexdigest(),
            "transductive": True,
            "target_labels_used_for_adaptation": False,
            "environment": environment(),
        },
    )
    return metrics
