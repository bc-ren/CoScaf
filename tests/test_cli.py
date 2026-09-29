"""Cache CLI tests with synthetic images and no pretrained model downloads."""

from argparse import Namespace
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from coscaf import cli
from coscaf.data import DatasetBundle, PrefixCache
from coscaf.io import sha256, write_json


@pytest.fixture
def synthetic_images(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    images = tmp_path / "images"
    images.mkdir()
    labels = np.array([0, 0, 0, 0, 1, 1])
    names = [f"image_{index}.png" for index in range(len(labels))]
    for index, name in enumerate(names):
        Image.fromarray(np.full((11, 13, 3), 20 + index, dtype=np.uint8)).save(images / name)
    np.savez(
        data / "metadata.npz",
        labels=labels,
        semantics=np.eye(2, 3),
        train=np.array([2, 0, 3]),
        test_seen=np.array([1]),
        test_unseen=np.array([4, 5]),
    )
    write_json(data / "images.json", names)
    return DatasetBundle.load(data), images


def cache_arguments(tmp_path, bundle, images, role="train", run=None):
    checkpoint = tmp_path / "pretrained.pt"
    checkpoint.write_bytes(b"synthetic-pretrained-file")
    return Namespace(
        data=bundle.root,
        image_root=images,
        output=tmp_path / "cache",
        role=role,
        run=run,
        batch_size=2,
        device="cpu",
        retizero_checkpoint=checkpoint,
    )


def mocked_encoder(monkeypatch, config, calls, prefix_calls):
    resolution = config["source"]["resolution"]

    def processor(*, images, size, return_tensors):
        calls.append((len(images), size, return_tensors))
        # Values deliberately differ from manual image scaling: only use of the
        # official processor interface can reproduce the expected cache below.
        markers = [float(image.getpixel((0, 0))[0]) + 1000 for image in images]
        return {
            "pixel_values": torch.stack(
                [torch.full((3, resolution, resolution), x) for x in markers]
            )
        }

    def preprocess(image):
        calls.append(image.getpixel((0, 0))[0])
        marker = float(image.getpixel((0, 0))[0]) + 2000
        return torch.full((1, 3, resolution, resolution), marker)

    encoder = SimpleNamespace(
        coscaf_processor=processor,
        preprocess_image=preprocess,
        coscaf_provenance={"weights_sha256": "synthetic"},
    )
    monkeypatch.setattr(cli, "load_backbone", lambda args, cfg: encoder)

    def prefix(received_encoder, pixels):
        assert received_encoder is encoder
        assert pixels.ndim == 4 and pixels.shape[1:] == (3, resolution, resolution)
        prefix_calls.append(pixels[:, 0, 0, 0].tolist())
        tokens = (resolution // 16) ** 2 + 1
        return pixels[:, :1, :1, :1].reshape(-1, 1, 1).expand(-1, tokens, config["visual_dim"])

    monkeypatch.setattr(cli, "vit_prefix", prefix)
    monkeypatch.setattr(cli, "retinal_prefix", prefix)
    return encoder


@pytest.mark.parametrize("backbone", ["vit_base", "retizero"])
def test_cache_uses_official_preprocessor_and_preserves_every_row(
    tmp_path, monkeypatch, synthetic_images, backbone
):
    bundle, images = synthetic_images
    resolution = 384 if backbone == "vit_base" else 224
    config = {"backbone": backbone, "visual_dim": 8, "source": {"resolution": resolution}}
    args = cache_arguments(tmp_path, bundle, images)
    calls, prefix_calls = [], []
    mocked_encoder(monkeypatch, config, calls, prefix_calls)
    cli.cache_images(args, config)
    cached = PrefixCache(args.output, bundle, "train")
    np.testing.assert_array_equal(cached.indices, [2, 0, 3])
    assert cached.raw.shape == (3, (resolution // 16) ** 2 + 1, 8)
    offset = 1000 if backbone == "vit_base" else 2000
    expected = np.array([22, 20, 23], dtype=np.float32) + offset
    np.testing.assert_array_equal(cached.raw[:, 0, 0], expected)
    np.testing.assert_array_equal(np.concatenate(prefix_calls), expected)
    if backbone == "vit_base":
        assert calls == [
            (2, {"height": 384, "width": 384}, "pt"),
            (1, {"height": 384, "width": 384}, "pt"),
        ]
    else:
        assert calls == [22, 20, 23]
    with pytest.raises(ValueError, match="role"):
        PrefixCache(args.output, bundle, "test")


def test_test_cache_rejects_missing_freeze_before_backbone_access(
    tmp_path, monkeypatch, synthetic_images
):
    bundle, images = synthetic_images
    args = cache_arguments(tmp_path, bundle, images, role="test")
    monkeypatch.setattr(
        cli, "load_backbone", lambda *args: pytest.fail("Backbone accessed too early")
    )
    with pytest.raises(ValueError, match="completed training freeze"):
        cli.cache_images(args, {"backbone": "vit_base"})
    args.run = tmp_path / "missing_run"
    with pytest.raises(FileNotFoundError):
        cli.cache_images(args, {"backbone": "vit_base"})
    assert not args.output.exists()


def test_test_cache_obeys_frozen_test_membership(tmp_path, monkeypatch, synthetic_images):
    bundle, images = synthetic_images
    config = {"backbone": "vit_base", "visual_dim": 8, "source": {"resolution": 384}}
    run = tmp_path / "run"
    run.mkdir()
    write_json(run / "config.json", config)
    (run / "model.pt").write_bytes(b"synthetic-source-model")
    write_json(
        run / "freeze.json",
        {
            "metadata_sha256": sha256(bundle.path),
            "files": {name: sha256(run / name) for name in ("config.json", "model.pt")},
        },
    )
    args = cache_arguments(tmp_path, bundle, images, role="test", run=run)
    calls, prefix_calls = [], []
    mocked_encoder(monkeypatch, config, calls, prefix_calls)
    cli.cache_images(args, config)
    cached = PrefixCache(args.output, bundle, "test")
    np.testing.assert_array_equal(cached.indices, bundle.test)
    np.testing.assert_array_equal(cached.raw[:, 0, 0], [1021, 1024, 1025])


@pytest.mark.parametrize("failure", ["row_count", "token_count", "nonfinite"])
def test_cache_rejects_invalid_prefix_output(tmp_path, monkeypatch, synthetic_images, failure):
    bundle, images = synthetic_images
    config = {"backbone": "vit_base", "visual_dim": 8, "source": {"resolution": 384}}
    args = cache_arguments(tmp_path, bundle, images)
    mocked_encoder(monkeypatch, config, [], [])

    def wrong_prefix(encoder, pixels):
        rows = len(pixels) - 1 if failure == "row_count" else len(pixels)
        tokens = 576 if failure == "token_count" else 577
        value = float("nan") if failure == "nonfinite" else 0
        return torch.full((rows, tokens, 8), value, dtype=torch.float32)

    monkeypatch.setattr(cli, "vit_prefix", wrong_prefix)
    with pytest.raises(ValueError, match="Prefix"):
        cli.cache_images(args, config)
    assert not (args.output / "manifest.json").exists()
