"""Frozen pretrained backbones and jointly learned late visual prompts.

Download pretrained models separately. Dataset images and pretrained weights are
not distributed with this package.
"""

from __future__ import annotations

import copy
import importlib
import math
import subprocess
import sys
from pathlib import Path

import torch
from torch import nn

from .io import sha256 as file_sha256

VIT_ID = "google/vit-base-patch16-224-in21k"
VIT_REVISION = "b4569560a39a0f1af58e3ddaf17facf20ab919b0"
VIT_WEIGHTS_SHA256 = "fd4e1169c7aa6c2dbfa8a6448be13b35abc0ee256190857c90009d12c094619b"
RETIZERO_REVISION = "d72aadc692fbe33b182c79711bccb397edffb419"


def _run_block(layer, tokens):
    result = layer(tokens)
    return result[0] if isinstance(result, tuple) else result


class VisualPromptSuffix(nn.Module):
    """Prompt blocks 11 and 12, then fuse normalized layer-10/12 patch rows.

    ``raw`` is the unnormalized output of block 10, including CLS. Prompt
    tokens are inserted after CLS and discarded after each prompted block.
    """

    _prefix_error = "Expected prefix tokens with shape [N,197|577,768]"

    def __init__(self, frozen_vit, count=16):
        super().__init__()
        layers = getattr(frozen_vit, "layers", None)
        if layers is None:
            layers = frozen_vit.encoder.layer
        if len(layers) != 12 or count < 1:
            raise ValueError("Expected a 12-block ViT and at least one prompt token")
        self._initialize(layers[10:], frozen_vit.layernorm, frozen_vit, count, 768)

    def _initialize(self, layers, norm, backbone, count, width):
        self.layers = nn.ModuleList([copy.deepcopy(layer) for layer in layers])
        self.norm = copy.deepcopy(norm)
        self.requires_grad_(False)
        reference = next(backbone.parameters())
        self.prompts = nn.Parameter(
            torch.empty(2, count, width, dtype=reference.dtype, device=reference.device)
        )
        radius = math.sqrt(6 / (3 * 16 * 16 + width))
        nn.init.uniform_(self.prompts, -radius, radius)
        self.count = count
        self.enabled = True
        self.eval()

    def layer_views(self, raw):
        """Return fused, early, and late evidence in this order."""
        if (
            raw.ndim != 3
            or raw.shape[1] not in (197, 577)
            or raw.shape[2] != self.prompts.shape[-1]
        ):
            raise ValueError(self._prefix_error)
        h = raw
        for index, layer in enumerate(self.layers):
            if self.enabled:
                prompt = self.prompts[index].unsqueeze(0).expand(len(h), -1, -1)
                extended = _run_block(layer, torch.cat([h[:, :1], prompt, h[:, 1:]], 1))
                h = torch.cat([extended[:, :1], extended[:, 1 + self.count :]], 1)
            else:
                h = _run_block(layer, h)
        early = self.norm(raw)[:, 1:]
        late = self.norm(h)[:, 1:]
        return (early + late) * 0.5, early, late

    def forward(self, raw):
        return self.layer_views(raw)[0]

    def adapter_state(self):
        return {"prompts": self.prompts.detach().cpu().clone()}

    def load_adapter(self, state):
        if set(state) != {"prompts"} or state["prompts"].shape != self.prompts.shape:
            raise ValueError("Prompt checkpoint shape mismatch")
        with torch.no_grad():
            self.prompts.copy_(state["prompts"].to(self.prompts))


class RetinaPrompt(VisualPromptSuffix):
    """Frozen RetiZero blocks 23/24 with newly initialized task prompts.

    The pretrained RetiZero LoRA weights are part of the frozen backbone.
    CoScaf uses the 1024-dimensional patch evidence, not native text scores.
    """

    _prefix_error = "Expected retinal prefix tokens [N,197|577,1024]"

    def __init__(self, model, count=16):
        nn.Module.__init__(self)
        vision = model.vision_model.model.lora_vit
        if len(vision.blocks) != 24 or count < 1:
            raise ValueError("Expected the official RetiZero 24-block encoder")
        self._initialize(
            vision.blocks[-2:],
            vision.fc_norm if vision.global_pool else vision.norm,
            vision,
            count,
            1024,
        )


def load_vit(snapshot=None, device="cpu", local_files_only=False):
    """Load the pinned Google ViT and its official image processor."""
    from transformers import ViTImageProcessor, ViTModel

    if snapshot is None:
        from huggingface_hub import snapshot_download

        snapshot = snapshot_download(
            repo_id=VIT_ID,
            revision=VIT_REVISION,
            allow_patterns=["model.safetensors", "config.json", "preprocessor_config.json"],
            local_files_only=local_files_only,
        )
    snapshot = Path(snapshot)
    weights = snapshot / "model.safetensors"
    if not weights.is_file() or file_sha256(weights) != VIT_WEIGHTS_SHA256:
        raise ValueError("ViT checkpoint does not match the released model SHA-256")
    model, info = ViTModel.from_pretrained(
        snapshot, output_loading_info=True, local_files_only=True
    )
    if any(info.get(k) for k in ("missing_keys", "unexpected_keys", "mismatched_keys")):
        raise ValueError("Pretrained ViT must load without missing or replaced tensors")
    config = model.config
    if (config.hidden_size, config.num_hidden_layers, config.patch_size) != (768, 12, 16):
        raise ValueError("Expected ViT-Base/16")
    model = model.eval().requires_grad_(False).to(device)
    processor = ViTImageProcessor.from_pretrained(snapshot, local_files_only=True)
    if processor.image_mean != [0.5] * 3 or processor.image_std != [0.5] * 3:
        raise ValueError("Unexpected ViT pixel normalization")
    model.coscaf_provenance = {
        "model_id": VIT_ID,
        "revision": VIT_REVISION,
        "weights_sha256": VIT_WEIGHTS_SHA256,
        "config_sha256": file_sha256(snapshot / "config.json"),
        "processor_sha256": file_sha256(snapshot / "preprocessor_config.json"),
    }
    return model, processor


@torch.inference_mode()
def vit_prefix(model, pixels):
    """Extract FP32 block-10 tokens using positional interpolation at 384 px."""
    output = model(pixel_values=pixels, output_hidden_states=True, interpolate_pos_encoding=True)
    raw = output.hidden_states[10]
    if raw.shape[-1] != 768 or not torch.isfinite(raw).all():
        raise ValueError("Invalid frozen ViT prefix output")
    return raw


def load_retizero(source_dir, checkpoint, bert_path, device="cpu"):
    """Load the complete official RetiZero checkpoint strictly.

    ``source_dir`` is a separately obtained checkout at ``RETIZERO_REVISION``.
    The official repository and its license remain separate from this package.
    This compatibility loader does not drop or replace pretrained tensors.
    """
    import types

    from transformers import AutoConfig, AutoModel

    source_dir = Path(source_dir).resolve()
    if not (source_dir / "zeroshot/modeling/model.py").is_file():
        raise FileNotFoundError("Expected official RetiZero source checkout")
    source_commit = None
    if (source_dir / ".git").exists():
        source_commit = subprocess.check_output(
            ["git", "-C", str(source_dir), "rev-parse", "HEAD"], text=True
        ).strip()
        if source_commit != RETIZERO_REVISION:
            raise ValueError("RetiZero checkout does not match the released commit")
    sys.path.insert(0, str(source_dir))
    if "torch._six" not in sys.modules:
        shim = types.ModuleType("torch._six")
        shim.container_abcs = importlib.import_module("collections.abc")
        shim.string_classes = (str,)
        shim.int_classes = (int,)
        sys.modules["torch._six"] = shim
    official = importlib.import_module("zeroshot.modeling.model")
    original = official.AutoModel

    class ConfigOnlyAutoModel:
        @staticmethod
        def from_pretrained(path, **kwargs):
            config = AutoConfig.from_pretrained(path, local_files_only=True)
            config.output_hidden_states = True
            return AutoModel.from_config(config)

    official.AutoModel = ConfigOnlyAutoModel
    official.device = device
    try:
        model = official.CLIPRModel(
            vision_type="lora",
            from_checkpoint=False,
            R=8,
            bert_type=str(bert_path),
            caption="A fundus photograph of [CLS]",
        )
    finally:
        official.AutoModel = original
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    position_key = "text_model.model.embeddings.position_ids"
    if position_key in state:
        embeddings = model.text_model.model.embeddings
        torch.testing.assert_close(
            embeddings.position_ids.cpu(), state[position_key], rtol=0, atol=0
        )
        embeddings._non_persistent_buffers_set.discard("position_ids")
    model.load_state_dict(state, strict=True)
    model.coscaf_provenance = {
        "model_id": "RetiZero",
        "source_commit": source_commit,
        "expected_source_commit": RETIZERO_REVISION,
        "weights_sha256": file_sha256(checkpoint),
        "source_sha256": file_sha256(source_dir / "zeroshot/modeling/model.py"),
    }
    return model.eval().requires_grad_(False).to(device)


@torch.inference_mode()
def retinal_prefix(model, pixels):
    """Extract the official RetiZero block-22 prefix at 224 px."""
    vision = model.vision_model.model.lora_vit
    if pixels.shape[-2:] != (224, 224):
        raise ValueError("The released retinal recipe uses 224-pixel images")
    patch = vision.patch_embed(pixels)
    h = vision.pos_drop(
        torch.cat([vision.cls_token.expand(len(pixels), -1, -1), patch], 1) + vision.pos_embed
    )
    for block in vision.blocks[:22]:
        h = block(h)
    return h
