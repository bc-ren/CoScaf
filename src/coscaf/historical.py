"""Replay released historical checkpoints without training or test-time adaptation.

Run ``python -m coscaf.historical --help``. ``verify`` hashes all assets and
backbone dependencies; ``audit`` recomputes metrics from the saved score banks;
``replay`` forwards cached prefix tokens through the saved original models.
Historical TTA parameters are loaded, never optimized. All computations use
FP32; metric arithmetic uses FP64. Subset checks never establish full-test replay.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

SCHEMA = "coscaf-historical-replay-v1"


def _path(root, value):
    path = (root / value).resolve()
    if Path(value).is_absolute() or not path.is_relative_to(root.resolve()):
        raise ValueError(f"Asset path must stay within --assets: {value}")
    return path


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _specs(manifest):
    """Collect declared asset checksums, including optional train prefixes."""
    result = {}

    def visit(value):
        if isinstance(value, dict):
            if "path" in value and "sha256" in value:
                name = value["path"]
                if name in result and result[name] != value["sha256"]:
                    raise ValueError(f"Conflicting asset checksums: {name}")
                result[name] = value["sha256"]
            for key, child in value.items():
                if key != "backbone":
                    visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(manifest)
    backbone = manifest["backbone"]
    weight = backbone["path"]
    if backbone["kind"] in ("vit", "google_vit"):
        weight = str(Path(weight) / "model.safetensors")
    result[weight] = backbone["sha256"]
    for name, expected in result.items():
        if not isinstance(expected, str) or len(expected) != 64:
            raise ValueError(f"A SHA-256 is required for {name}")
    return result


def asset_inventory(manifest, root):
    """Include every local file used by the full backbone/tokenizer/source loaders."""
    declared = _specs(manifest)
    paths = dict(declared)
    backbone = manifest["backbone"]
    for key in ("path", "source_dir", "bert_path"):
        if key not in backbone:
            continue
        location = _path(root, backbone[key])
        if not location.is_dir():
            continue
        for path in location.rglob("*"):
            if not path.is_file() or any(p in (".git", "__pycache__") for p in path.parts):
                continue
            if path.suffix == ".pyc":
                continue
            name = str(path.relative_to(root))
            _path(root, name)
            paths.setdefault(name, None)
    return paths


def verify_assets(manifest, root, receipt_path):
    """Perform full SHA verification once; record stat checks for subsequent replay."""
    from .io import fingerprint, sha256

    files = {}
    for name, expected in asset_inventory(manifest, root).items():
        path = _path(root, name)
        before = path.stat()
        actual = sha256(path)
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError(f"Asset changed while hashing: {name}")
        if expected is not None and actual != expected:
            raise ValueError(f"Asset checksum mismatch: {name}")
        files[name] = {
            "sha256": actual,
            "size": after.st_size,
            "mtime_ns": after.st_mtime_ns,
            "manifest_checksum_matched": expected is not None,
        }
    result = {
        "schema": "coscaf-asset-verification-v1",
        "status": "PASS",
        "manifest_sha256": fingerprint(manifest),
        "files": files,
        "full_backbone_dependency_inventory": True,
        "note": "Stat checks reuse this local hash audit; re-verify after moving or extracting assets.",
    }
    _write(receipt_path, result)
    return result


def check_assets(manifest, root, receipt_path=None, include_backbone=True):
    """Hash small inputs; use a previous full hash audit for large arrays/weights."""
    from .io import fingerprint, sha256

    inventory = asset_inventory(manifest, root) if include_backbone else _specs(manifest)
    if not include_backbone:
        required = [manifest[k] for k in ("metadata", "calibration")]
        required.extend(manifest["reference"].values())
        inventory = {s["path"]: s["sha256"] for s in required}
    previous = _read(receipt_path) if receipt_path and Path(receipt_path).is_file() else None
    if previous:
        if previous.get("status") != "PASS" or previous["manifest_sha256"] != fingerprint(manifest):
            raise ValueError("Verification receipt belongs to a different manifest")
    checked = {}
    for name, expected in inventory.items():
        path = _path(root, name)
        stat = path.stat()
        if previous:
            old = previous["files"].get(name)
            if old is None or (old["size"], old["mtime_ns"]) != (stat.st_size, stat.st_mtime_ns):
                raise ValueError(f"Asset stat changed; run --mode verify again: {name}")
            if expected is not None and old["sha256"] != expected:
                raise ValueError(f"Verification checksum differs from manifest: {name}")
            method = "stat_against_full_sha256_receipt"
        elif stat.st_size <= 64 * 1024 * 1024:
            actual = sha256(path)
            if expected is not None and actual != expected:
                raise ValueError(f"Asset checksum mismatch: {name}")
            method = "sha256_this_run" if expected else "unregistered_file_present"
        else:
            method = "present_only_not_hash_verified_this_run"
        checked[name] = {"size": stat.st_size, "method": method}
    return {
        "full_hash_audit_reused": bool(previous),
        "all_declared_hashes_verified": bool(previous)
        or all(v["method"] == "sha256_this_run" for v in checked.values()),
        "files": checked,
    }


def load_metadata(path):
    """Preserve native row order rather than reconstructing the test concatenation."""
    import numpy as np

    with np.load(path, allow_pickle=False) as data:
        meta = {key: data[key].copy() for key in data.files}
    for key in ("labels", "semantics", "train"):
        if key not in meta:
            raise ValueError(f"Missing metadata field: {key}")
    if "test" not in meta:
        meta["test"] = np.concatenate((meta["test_seen"], meta["test_unseen"]))
    labels, semantics = meta["labels"], meta["semantics"]
    if labels.ndim != 1 or labels.dtype.kind not in "iu" or semantics.ndim != 2:
        raise ValueError("Malformed labels or semantics")
    if not np.isfinite(semantics).all() or (np.linalg.norm(semantics, axis=1) == 0).any():
        raise ValueError("Invalid semantic vectors")
    if labels.min() < 0 or labels.max() >= len(semantics):
        raise ValueError("Labels must index the original semantic rows")
    for role in ("train", "test"):
        rows = meta[role]
        if rows.ndim != 1 or rows.dtype.kind not in "iu" or not len(rows):
            raise ValueError(f"Invalid {role} indices")
        if len(np.unique(rows)) != len(rows) or rows.min() < 0 or rows.max() >= len(labels):
            raise ValueError(f"Duplicate or out-of-range {role} indices")
    if np.intersect1d(meta["train"], meta["test"]).size:
        raise ValueError("Train/test overlap")
    seen = np.unique(labels[meta["train"]])
    if "seen" in meta and not np.array_equal(meta["seen"], seen):
        raise ValueError("Seen-class metadata mismatch")
    meta["seen"] = seen
    meta["classes"] = np.arange(len(semantics), dtype=np.int64)
    return meta


def load_historical_state(state, suffix, config, semantics, device="cpu"):
    """Drop only obsolete loss buffers and strictly load every active tensor."""
    import numpy as np
    import torch

    from .model import GENERATOR, CoScafModel, CoScafReadout

    if set(state) != {"head", "prompt"} or set(state["prompt"]) != {"prompts"}:
        raise ValueError("Expected the original head/prompt checkpoint format")
    h = state["head"]
    obsolete = {"initial_" + name for name in GENERATOR}
    present = set(h) & obsolete
    if present and present != obsolete:
        raise ValueError("Partial historical initial-buffer set")
    for key, value in list(h.items()) + list(state["prompt"].items()):
        if (
            not torch.is_tensor(value)
            or value.dtype != torch.float32
            or not torch.isfinite(value).all()
        ):
            raise ValueError(f"Historical checkpoint requires finite FP32 tensor: {key}")
    torch.testing.assert_close(
        h["semantics"].cpu(),
        torch.as_tensor(np.asarray(semantics), dtype=torch.float32),
        rtol=0,
        atol=0,
    )
    options = dict(config["readout"])
    options["local_weight"] = config["source_recipe"]["local_weight"]
    options["native"] = False
    options["query_rank"] = h["query_low.weight"].shape[0]
    options["moment_rank"] = h["projection"].shape[1]
    initial = {
        name: h[key]
        for name, key in (
            ("weight", "center_weight"),
            ("bias", "center_bias"),
            ("residual_weight", "mode_weight"),
            ("residual_bias", "mode_bias"),
        )
    }
    head = CoScafReadout(
        h["semantics"],
        initial,
        h["visual_mean"],
        h["visual_matrix"],
        **options,
    ).to(device)
    active = {key: value for key, value in h.items() if key not in obsolete}
    head.load_state_dict(active, strict=True)
    suffix.load_adapter(state["prompt"])
    return CoScafModel(suffix, head).to(device).eval().requires_grad_(False)


def load_saved_generator(pack, head, device="cpu"):
    import torch

    from .model import GENERATOR

    generator = pack["generator"]
    if set(generator) != set(GENERATOR):
        raise ValueError("Saved adapted generator has incorrect keys")
    result = {}
    for name, value in generator.items():
        if value.shape != getattr(head, name).shape or value.dtype != torch.float32:
            raise ValueError(f"Adapted generator shape/dtype mismatch: {name}")
        if not torch.isfinite(value).all():
            raise ValueError(f"Nonfinite adapted generator: {name}")
        result[name] = value.to(device).detach()
    return result


def calibrations(record):
    result = {
        "source": record["source_calibration"],
        "adapted": record.get("tta_calibration", record.get("adapted_calibration")),
    }
    for value in result.values():
        if value is None:
            raise ValueError("Both frozen source and TTA calibrations are required")
        from math import isfinite

        if not isfinite(value["gamma"]) or not isfinite(value["probability_temperature"]):
            raise ValueError("Frozen calibration must be finite")
        if value["probability_temperature"] <= 0:
            raise ValueError("Frozen probability temperature must be positive")
    return result


def compare_scores(actual, expected, classes, seen, calibration, atol=2e-6, rtol=1e-5):
    import numpy as np

    if actual.shape != expected.shape or not np.isfinite(expected).all():
        raise ValueError("Saved score bank shape/finiteness mismatch")
    if not np.isfinite(actual).all():
        raise ValueError("Replayed scores are nonfinite")
    sm = np.isin(classes, seen)
    gamma = calibration["gamma"]
    predicted = (np.asarray(actual, dtype=np.float64) - gamma * sm).argmax(1)
    reference = (np.asarray(expected, dtype=np.float64) - gamma * sm).argmax(1)
    cz_actual = actual[:, ~sm].argmax(1)
    cz_reference = expected[:, ~sm].argmax(1)
    return {
        "max_absolute_error": float(np.max(np.abs(actual.astype(float) - expected))),
        "atol": atol,
        "rtol": rtol,
        "scores_within_tolerance": bool(np.allclose(actual, expected, atol=atol, rtol=rtol)),
        "gzsl_prediction_disagreements": int(np.count_nonzero(predicted != reference)),
        "czsl_prediction_disagreements": int(np.count_nonzero(cz_actual != cz_reference)),
    }


def _load_suffix(manifest, root, device):
    from .backbones import RetinaPrompt, VisualPromptSuffix, load_retizero, load_vit

    spec = manifest["backbone"]
    if spec["kind"] in ("vit", "google_vit"):
        encoder, _ = load_vit(_path(root, spec["path"]), device, local_files_only=True)
        return VisualPromptSuffix(encoder, count=16)
    if spec["kind"] != "retizero":
        raise ValueError("Unsupported historical backbone")
    encoder = load_retizero(
        _path(root, spec["source_dir"]),
        _path(root, spec["path"]),
        _path(root, spec["bert_path"]),
        device,
    )
    return RetinaPrompt(encoder, count=16)


def _prefix_spec(manifest, split):
    if "prefixes" in manifest:
        return manifest["prefixes"][split]
    prefix = manifest["prefix"]
    if prefix.get("role", "test") != split:
        raise ValueError("Manifest prefix role differs from --split")
    return prefix


def run(args):
    import numpy as np
    import torch

    from .io import environment, fingerprint
    from .metrics import evaluate

    root = args.assets.resolve()
    manifest = _read(args.manifest)
    if manifest.get("schema") != SCHEMA or manifest.get("paper_endpoint") != "tta":
        raise ValueError("Require the historical replay schema and explicit paper_endpoint='tta'")
    verification = args.verification or args.manifest.with_suffix(".verified.json")
    if args.mode == "verify":
        result = verify_assets(manifest, root, verification)
        print(
            json.dumps(
                {
                    "status": result["status"],
                    "files": len(result["files"]),
                    "verification": str(verification),
                },
                indent=2,
            )
        )
        return 0
    if args.output is None:
        raise ValueError("--output is required for audit or replay")
    if args.output.exists():
        raise FileExistsError("Output already exists; use a new output directory")
    integrity = check_assets(manifest, root, verification, include_backbone=args.mode == "replay")
    meta = load_metadata(_path(root, manifest["metadata"]["path"]))
    calibration = calibrations(_read(_path(root, manifest["calibration"]["path"])))
    classes, seen = meta["classes"], meta["seen"]
    split = args.split
    all_rows = meta[split]
    positions = np.arange(len(all_rows))
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit must be positive")
        positions = positions[: args.limit]
    if args.rows:
        positions = np.load(args.rows, allow_pickle=False)
        if (
            positions.ndim != 1
            or positions.dtype.kind not in "iu"
            or not len(positions)
            or len(np.unique(positions)) != len(positions)
            or positions.min() < 0
            or positions.max() >= len(all_rows)
        ):
            raise ValueError("--rows must contain unique zero-based positions in the chosen split")
    selected = all_rows[positions]
    complete = split == "test" and np.array_equal(positions, np.arange(len(all_rows)))
    references = {}
    if split == "test":
        for name in ("source", "adapted"):
            bank = np.load(_path(root, manifest["reference"][name]["path"]), mmap_mode="r")
            if bank.shape != (len(all_rows), len(classes)):
                raise ValueError("Saved score bank must align to the complete native test ordering")
            references[name] = np.asarray(bank[positions])
    elif args.mode == "audit":
        raise ValueError("Saved score-bank audit is defined only for the official test split")
    args.output.mkdir(parents=True)
    if args.mode == "audit":
        scores = references
    else:
        torch.set_num_threads(args.threads)
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        os.environ["HF_HUB_OFFLINE"] = os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        prefix = _prefix_spec(manifest, split)
        indices = np.load(_path(root, prefix["indices"]["path"]), allow_pickle=False)
        if not np.array_equal(indices, all_rows):
            raise ValueError("Prefix indices do not match original split order")
        raw = np.load(_path(root, prefix["values"]["path"]), mmap_mode="r", allow_pickle=False)
        if raw.ndim != 3 or raw.shape[0] != len(all_rows) or raw.dtype != np.float32:
            raise ValueError("Prefix must be FP32 [rows,tokens,width]")
        config = _read(_path(root, manifest["source_config"]["path"]))
        if config.get("postprocess", {}).get("boundary_strength") is not None:
            raise ValueError("This replay supports the registered null-boundary endpoints only")
        state = torch.load(
            _path(root, manifest["source_checkpoint"]["path"]),
            map_location="cpu",
            weights_only=True,
        )
        model = load_historical_state(
            state, _load_suffix(manifest, root, args.device), config, meta["semantics"], args.device
        )
        pack = torch.load(
            _path(root, manifest["adaptation_checkpoint"]["path"]),
            map_location="cpu",
            weights_only=True,
        )
        generator = load_saved_generator(pack, model.head, args.device)
        ids = torch.tensor(classes, device=args.device)
        batch = args.batch_size or config["source_recipe"]["score_batch_size"]
        if batch < 1:
            raise ValueError("Batch size must be positive")
        scores = {
            name: np.empty((len(selected), len(classes)), np.float32)
            for name in ("source", "adapted")
        }
        with torch.inference_mode():
            for off in range(0, len(selected), batch):
                values = np.array(raw[positions[off : off + batch]], copy=True)
                if not np.isfinite(values).all():
                    raise ValueError("Nonfinite prefix cache values")
                patch = model.suffix(torch.from_numpy(values).to(args.device))
                if isinstance(patch, tuple):
                    patch = patch[0]
                features = model.head.features(patch, ids)
                for name, parameters in (("source", None), ("adapted", generator)):
                    z = model.head.score_features(features, ids, parameters)
                    scores[name][off : off + len(values)] = z.cpu().numpy()
        for name, values in scores.items():
            np.save(args.output / (name + "_scores.npy"), values)
    comparisons, metrics, metric_comparisons = {}, {}, {}
    for name, values in scores.items():
        if not np.isfinite(values).all():
            raise ValueError("Nonfinite score bank")
        if args.mode == "replay" and name in references:
            comparisons[name] = compare_scores(
                values, references[name], classes, seen, calibration[name], args.atol, args.rtol
            )
        if complete:
            metrics[name], _ = evaluate(
                values, meta["labels"][selected], classes, seen, calibration[name]
            )
    expected_metrics = manifest.get("expected_metrics", {})
    for name, expected in expected_metrics.items():
        if name not in ("source", "adapted"):
            raise ValueError("expected_metrics keys must be source/adapted")
        if complete:
            deltas = {}
            for key, target in expected.items():
                if key not in metrics[name] or not isinstance(metrics[name][key], (int, float)):
                    raise ValueError(f"Unknown expected metric: {name}/{key}")
                if not np.isfinite(target):
                    raise ValueError("Expected metrics must be finite fractions")
                deltas[key] = float(metrics[name][key] - target)
            metric_comparisons[name] = {
                "deltas": deltas,
                "atol": args.metric_atol,
                "within_tolerance": all(abs(v) <= args.metric_atol for v in deltas.values()),
            }
    passed = all(
        r["scores_within_tolerance"]
        and not r["gzsl_prediction_disagreements"]
        and not r["czsl_prediction_disagreements"]
        for r in comparisons.values()
    )
    passed = passed and all(r["within_tolerance"] for r in metric_comparisons.values())
    np.save(args.output / "indices.npy", selected)
    result = {
        "status": "PASS" if passed else "MISMATCH",
        "mode": args.mode,
        "dataset": manifest["dataset"],
        "candidate": manifest["candidate"],
        "seed": manifest["seed"],
        "paper_endpoint": "tta",
        "paper_endpoint_score_key": "adapted",
        "rows": len(selected),
        "split": split,
        "complete_official_population": complete,
        "archived_result_reproduced": bool(
            args.mode == "replay" and complete and len(comparisons) == 2 and passed
        ),
        "saved_scorebank_metrics_only": args.mode == "audit",
        "training_performed": False,
        "tta_optimization_performed": False,
        "calibration_refitted": False,
        "historical_anchor_trained_checkpoint": True,
        "new_no_anchor_training_result": False,
        "compute_dtype": "float32",
        "metric_dtype": "float64",
        "device": args.device if args.mode == "replay" else None,
        "score_tolerance": {"atol": args.atol, "rtol": args.rtol},
        "comparisons": comparisons,
        "metrics": metrics,
        "metric_comparisons": metric_comparisons,
        "asset_integrity": integrity,
        "asset_hashes_verified": integrity["all_declared_hashes_verified"],
        "manifest_sha256": fingerprint(manifest),
        "environment": environment(),
    }
    _write(args.output / "replay.json", result)
    print(
        json.dumps(
            {
                k: result[k]
                for k in (
                    "status",
                    "mode",
                    "dataset",
                    "rows",
                    "archived_result_reproduced",
                    "asset_hashes_verified",
                    "comparisons",
                    "metric_comparisons",
                )
            },
            indent=2,
        )
    )
    return 0 if passed else 2


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--mode", choices=("verify", "audit", "replay"), default="replay")
    parser.add_argument("--output", type=Path, help="New output directory; required except verify")
    parser.add_argument(
        "--verification", type=Path, help="Full-SHA receipt; default MANIFEST.verified.json"
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--split", choices=("train", "test"), default="test")
    subset = parser.add_mutually_exclusive_group()
    subset.add_argument(
        "--limit", type=int, help="Check the first N split rows, not full-test replay"
    )
    subset.add_argument("--rows", type=Path, help="NPY of positions in original split order")
    parser.add_argument("--atol", type=float, default=2e-6)
    parser.add_argument("--rtol", type=float, default=1e-5)
    parser.add_argument("--metric-atol", type=float, default=1e-10)
    args = parser.parse_args(argv)
    from math import isfinite

    if args.threads < 1 or any(
        not isfinite(v) or v < 0 for v in (args.atol, args.rtol, args.metric_atol)
    ):
        parser.error("threads must be positive and score tolerances finite/nonnegative")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
