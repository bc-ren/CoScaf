"""CoScaf command-line interface."""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image, features

from .backbones import (
    RETIZERO_REVISION,
    VIT_ID,
    VIT_REVISION,
    RetinaPrompt,
    VisualPromptSuffix,
    load_retizero,
    load_vit,
    retinal_prefix,
    vit_prefix,
)
from .data import DatasetBundle
from .io import environment, read_json, sha256, verify_files, write_json
from .pipeline import develop, evaluate_run, train
from .preparation import (
    import_natural,
    import_retinal,
    make_development_folds,
    write_prefix_cache_manifest,
)


def backbone_arguments(parser):
    parser.add_argument("--vit-snapshot", type=Path, help="Optional pinned Google ViT snapshot")
    parser.add_argument("--retizero-source", type=Path, help="Official RetiZero checkout")
    parser.add_argument("--retizero-checkpoint", type=Path)
    parser.add_argument("--bert-path", type=Path, help="Local official tokenizer/config directory")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--threads", type=int, default=2)


def load_backbone(args, config):
    if config["backbone"] == "vit_base":
        model, processor = load_vit(args.vit_snapshot, args.device)
        model.coscaf_processor = processor
        return model
    if config["backbone"] != "retizero":
        raise ValueError("Backbone must be vit_base or retizero")
    if not all((args.retizero_source, args.retizero_checkpoint, args.bert_path)):
        raise ValueError("RetiZero requires --retizero-source, --retizero-checkpoint, --bert-path")
    return load_retizero(
        args.retizero_source, args.retizero_checkpoint, args.bert_path, args.device
    )


def suffix_factory(args, config):
    def build(device):
        if str(device) != str(args.device):
            raise ValueError("Inconsistent model device")
        encoder = load_backbone(args, config)
        suffix_type = VisualPromptSuffix if config["backbone"] == "vit_base" else RetinaPrompt
        # Each fit creates new prompts; only the pretrained backbone is reused.
        suffix = suffix_type(encoder)
        suffix.backbone_provenance = encoder.coscaf_provenance
        del encoder
        return suffix

    return build


def _write_folds(bundle, count):
    folder = bundle.root / "folds"
    folder.mkdir(exist_ok=False)
    for index, fold in enumerate(make_development_folds(bundle, count)):
        write_json(folder / f"fold_{index}.json", fold)


def cache_images(args, config):
    """Cache unprompted pretrained prefixes, preserving every input row."""
    bundle = DatasetBundle.load(args.data)
    if bundle.images is None:
        raise ValueError("Data bundle must have an ordered images.json list")
    if args.role == "test":
        if args.run is None:
            raise ValueError("Test extraction requires --run with a completed training freeze")
        frozen = read_json(args.run / "freeze.json")
        verify_files(args.run, frozen["files"])
        if frozen["metadata_sha256"] != sha256(bundle.path):
            raise ValueError("Test data differs from the frozen data membership")
        if read_json(args.run / "config.json") != config:
            raise ValueError("Test cache configuration differs from the frozen model")
    encoder = load_backbone(args, config)
    natural = config["backbone"] == "vit_base"
    prefix = vit_prefix if natural else retinal_prefix
    resolution = config["source"]["resolution"]
    indices = bundle.train if args.role == "train" else bundle.test
    destination = args.output
    destination.mkdir(parents=True, exist_ok=False)
    shape = (len(indices), (resolution // 16) ** 2 + 1, config["visual_dim"])
    output = np.lib.format.open_memmap(
        destination / "raw.npy", mode="w+", dtype=np.float32, shape=shape
    )
    image_root = args.image_root.resolve()
    for start in range(0, len(indices), args.batch_size):
        images = []
        for index in indices[start : start + args.batch_size]:
            path = (image_root / bundle.images[index]).resolve()
            if not path.is_relative_to(image_root):
                raise ValueError("Image paths must remain within --image-root")
            with Image.open(path) as image:
                images.append(image.convert("RGB").copy())
        if natural:
            pixels = encoder.coscaf_processor(
                images=images,
                size={"height": resolution, "width": resolution},
                return_tensors="pt",
            )["pixel_values"]
        else:
            pixels = torch.cat([encoder.preprocess_image(image) for image in images])
        if len(pixels) != len(images):
            raise ValueError("Image preprocessing changed the batch row count")
        values = prefix(encoder, pixels.to(args.device))
        if (
            len(values) != len(images)
            or tuple(values.shape[1:]) != shape[1:]
            or not torch.isfinite(values).all()
        ):
            raise ValueError("Prefix tokens do not match the configured frozen backbone")
        output[start : start + len(values)] = values.float().cpu().numpy()
        print(f"Cached {start + len(values)}/{len(indices)} images", flush=True)
    output.flush()
    np.save(destination / "indices.npy", indices)
    metadata = {
        "id": VIT_ID if natural else "RetiZero",
        "revision": VIT_REVISION if natural else RETIZERO_REVISION,
        "resolution": resolution,
        "prefix_block": 10 if natural else 22,
        "environment": environment(),
        "pillow": Image.__version__,
        "jpeg": features.version_codec("jpg"),
        "libjpeg_turbo": features.check_feature("libjpeg_turbo"),
        "preprocessing": "official ViTImageProcessor"
        if natural
        else "official RetiZero preprocess_image",
    }
    metadata.update(encoder.coscaf_provenance)
    if not natural:
        metadata["weights_sha256"] = sha256(args.retizero_checkpoint)
    return write_prefix_cache_manifest(destination, bundle, args.role, metadata)


def make_parser():
    parser = argparse.ArgumentParser(description="CoScaf training and transductive evaluation")
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare-natural", help="Import official MATLAB metadata")
    prepare.add_argument("--dataset", choices=["AWA2", "CUB", "SUN"], required=True)
    prepare.add_argument("--labels", type=Path, required=True, help="MAT file containing labels")
    prepare.add_argument("--splits", type=Path, required=True, help="Official att_splits.mat")
    prepare.add_argument("--semantics", type=Path, help="CUB sent_splits.mat (1024D)")
    prepare.add_argument("--images", type=Path, help="Ordered JSON list of relative image paths")
    prepare.add_argument(
        "--exclude-train", type=Path, help="Optional zero-based training exclusions"
    )
    prepare.add_argument("--output", type=Path, required=True)
    retinal = sub.add_parser("prepare-retinal", help="Import user-supplied RetiRareV2 metadata")
    retinal.add_argument("--metadata", type=Path, required=True)
    retinal.add_argument("--images", type=Path)
    retinal.add_argument(
        "--folds", type=Path, required=True, help="Directory of fixed fold JSON files"
    )
    retinal.add_argument("--output", type=Path, required=True)
    cache = sub.add_parser("cache", help="Extract frozen image prefixes")
    cache.add_argument("--role", choices=["train", "test"], required=True)
    cache.add_argument("--image-root", type=Path, required=True)
    cache.add_argument("--batch-size", type=int, default=8)
    cache.add_argument("--run", type=Path, help="Required for test extraction")
    development = sub.add_parser("develop")
    training = sub.add_parser("train")
    test = sub.add_parser("evaluate", help="Report source and generator-only TTA metrics")
    for command in (cache, development, training, test):
        config_flag = "--run" if command is test else "--config"
        command.add_argument(config_flag, type=Path, required=True)
        command.add_argument("--data", type=Path, required=True)
        command.add_argument("--output", type=Path, required=True)
        backbone_arguments(command)
        if command is not cache:
            command.add_argument("--cache", type=Path, required=True)
    training.add_argument("--development", type=Path, required=True)
    return parser


def main(argv=None):
    args = make_parser().parse_args(argv)
    if args.command == "prepare-natural":
        bundle = import_natural(
            args.labels,
            args.splits,
            args.output,
            dataset=args.dataset,
            semantics_file=args.semantics,
            exclude_indices=args.exclude_train,
            images_file=args.images,
        )
        _write_folds(bundle, {"AWA2": 5, "CUB": 4, "SUN": 10}[args.dataset])
        print(f"Prepared {args.dataset}: {len(bundle.semantics)} classes")
        return
    if args.command == "prepare-retinal":
        from .data import load_fold

        bundle = import_retinal(args.metadata, args.output, images_file=args.images)
        destination = args.output / "folds"
        destination.mkdir()
        for path in sorted(args.folds.glob("fold_*.json")):
            fold = load_fold(bundle, path)
            write_json(
                destination / path.name, {key: value.tolist() for key, value in fold.items()}
            )
        if not list(destination.glob("fold_*.json")):
            raise ValueError("No fixed development folds supplied")
        print("Prepared RetiRareV2 metadata")
        return
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable; use --device cpu for a small validation run")
    config = read_json(args.run / "config.json" if args.command == "evaluate" else args.config)
    if args.command == "cache":
        result = cache_images(args, config)
    else:
        factory = suffix_factory(args, config)
        if args.command == "develop":
            result = develop(config, args.data, args.cache, args.output, factory, args.device)
        elif args.command == "train":
            result = train(
                config, args.data, args.cache, args.development, args.output, factory, args.device
            )
        else:
            result = evaluate_run(
                args.run, args.data, args.cache, args.output, factory, args.device
            )
    print(json.dumps(result, indent=2))
