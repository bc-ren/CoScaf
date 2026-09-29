"""Exercise the installed Transformers ViT API without downloading weights."""

import pytest
import torch
from transformers import ViTConfig, ViTModel

from coscaf.backbones import VIT_WEIGHTS_SHA256, VisualPromptSuffix, load_vit, vit_prefix


@pytest.fixture(scope="module")
def random_vit():
    # Keep the released hidden width, depth and heads. A narrow FFN is sufficient
    # to test tensor interfaces and prompt gradients without a pretrained file.
    torch.manual_seed(83)
    config = ViTConfig(
        hidden_size=768,
        intermediate_size=32,
        num_attention_heads=12,
        num_hidden_layers=12,
        image_size=32,
        patch_size=16,
    )
    return ViTModel(config).eval().requires_grad_(False)


def test_real_huggingface_prefix_and_late_prompt_api(random_vit):
    pixels = torch.randn(1, 3, 224, 224)
    raw = vit_prefix(random_vit, pixels)
    assert raw.shape == (1, 197, 768)
    suffix = VisualPromptSuffix(random_vit, count=2)
    fused, early, late = suffix.layer_views(raw)
    assert fused.shape == early.shape == late.shape == (1, 196, 768)
    torch.testing.assert_close(fused, (early + late) * 0.5, atol=0, rtol=0)
    (fused * torch.randn_like(fused)).sum().backward()
    assert suffix.prompts.grad is not None
    assert torch.isfinite(suffix.prompts.grad).all()
    assert torch.count_nonzero(suffix.prompts.grad) > 0
    assert [name for name, parameter in suffix.named_parameters() if parameter.requires_grad] == [
        "prompts"
    ]
    assert all(parameter.grad is None for parameter in random_vit.parameters())


def test_unprompted_suffix_matches_full_forward(random_vit):
    pixels = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        output = random_vit(
            pixel_values=pixels, output_hidden_states=True, interpolate_pos_encoding=True
        )
        suffix = VisualPromptSuffix(random_vit, count=2)
        suffix.enabled = False
        fused, early, late = suffix.layer_views(output.hidden_states[10])
        torch.testing.assert_close(late, output.last_hidden_state[:, 1:], atol=0, rtol=0)
        torch.testing.assert_close(
            early, random_vit.layernorm(output.hidden_states[10])[:, 1:], atol=0, rtol=0
        )
        torch.testing.assert_close(fused, (early + late) * 0.5, atol=0, rtol=0)


def test_local_snapshot_hash_guard_precedes_model_loading(tmp_path):
    assert len(VIT_WEIGHTS_SHA256) == 64
    (tmp_path / "model.safetensors").write_bytes(b"not-the-pinned-backbone")
    with pytest.raises(ValueError, match="SHA-256"):
        load_vit(snapshot=tmp_path, local_files_only=True)
