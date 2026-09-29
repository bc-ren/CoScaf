"""One source-training trajectory for prompts, evidence maps, and generator."""

from __future__ import annotations

import random
import time

import numpy as np
import torch
from torch.nn import functional as F

from .initialization import TargetBank, fit_transform, transform, unit_rows
from .model import GENERATOR, CoScafModel, CoScafReadout, _BaseReadout


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def schedule(fit, seed, epoch, batch_size):
    """Exactly one exposure per source-training image in every epoch."""
    order = np.random.default_rng(seed + 104729 * epoch).permutation(fit)
    return [order[offset : offset + batch_size] for offset in range(0, len(order), batch_size)]


def _readout_options(config):
    readout = config.get("readout", {})
    return {
        "variant": config.get("routing", readout.get("variant", "original")),
        "prior_temperature": config.get("prior_temperature", readout.get("prior_temperature", 0.5)),
        "local_weight": config["local_weight"],
    }


def make_model(semantics, fit_features, fit_labels, config, seed, suffix_factory, device="cpu"):
    """Initialize a fresh model from the current source-training partition.

    ``fit_features`` contains unprompted fused patch means, one row per entry
    of ``fit_labels``. The callable ``suffix_factory(device)`` must construct
    new task prompts around a frozen pretrained backbone.
    """
    features = np.asarray(fit_features, dtype=np.float64)
    labels = np.asarray(fit_labels, dtype=np.int64)
    semantic = np.asarray(semantics, dtype=np.float64)
    if len(features) != len(labels) or len(features) == 0:
        raise ValueError("Features and labels must have equal nonzero row counts")
    if labels.min() < 0 or labels.max() >= len(semantic):
        raise ValueError("Class labels must index semantic rows")
    visual = fit_transform(features, config["visual_norm"])
    x = transform(features, visual)
    normalized_semantic = unit_rows(semantic)
    recipe = {"modes": config["modes"], "alpha_base": config["ridge"], "alpha_residual": 0.001}
    initial = TargetBank(x, labels, normalized_semantic, "random", seed).model(recipe)
    if config["modes"] == 1:
        initial["residual_weight"] = np.zeros((semantic.shape[1], 1, x.shape[1]))
        initial["residual_bias"] = np.zeros((1, x.shape[1]))
    initial["residual_weight"] *= config["mode_scale"]
    initial["residual_bias"] *= config["mode_scale"]
    seed_all(seed)

    def tensor(value):
        return torch.tensor(value, dtype=torch.float32, device=device)

    inputs = (
        tensor(semantic),
        {key: tensor(value) for key, value in initial.items()},
        tensor(visual["mean"]),
        tensor(visual["matrix"]),
    )
    # Preserve the source implementation's random-draw order: evidence maps,
    # prompts, then independent prior maps. Common maps are retained exactly.
    base = _BaseReadout(*inputs, local_weight=config["local_weight"]).to(device)
    suffix = suffix_factory(device)
    head = CoScafReadout(*inputs, **_readout_options(config)).to(device)
    missing, unexpected = head.load_state_dict(base.state_dict(), strict=False)
    expected = {"prior_low.weight", "prior_high.weight"} if head.variant == "latent" else set()
    if set(missing) != expected or unexpected:
        raise ValueError("Readout initialization mismatch")
    return CoScafModel(suffix, head).eval()


def build_from_state(state, config, suffix_factory, device="cpu"):
    """Restore task state over the specified frozen pretrained backbone."""
    h = {key: value.to(device) for key, value in state["head"].items()}
    initial = {
        "weight": h["center_weight"],
        "bias": h["center_bias"],
        "residual_weight": h["mode_weight"],
        "residual_bias": h["mode_bias"],
    }
    head = CoScafReadout(
        h["semantics"], initial, h["visual_mean"], h["visual_matrix"], **_readout_options(config)
    ).to(device)
    model = CoScafModel(suffix_factory(device), head)
    model.load_trainable_state(state)
    return model.eval()


def make_optimizer(model, config):
    """AdamW groups with separate learning rates and complete parameter coverage."""
    query = list(model.head.query_low.parameters()) + list(model.head.query_high.parameters())
    if model.head.variant == "latent":
        query += list(model.head.prior_low.parameters()) + list(model.head.prior_high.parameters())
    groups = [
        {"params": [model.suffix.prompts], "lr": config["prompt_lr"], "name": "prompt"},
        {
            "params": [getattr(model.head, key) for key in GENERATOR],
            "lr": config["generator_lr"],
            "name": "generator",
        },
        {"params": query, "lr": config["query_lr"], "name": "query"},
        {
            "params": list(model.head.moment.parameters()),
            "lr": config["moment_lr"],
            "name": "moment",
        },
    ]
    covered = [id(p) for group in groups for p in group["params"]]
    required = {id(p) for p in model.parameters() if p.requires_grad}
    if len(covered) != len(set(covered)) or set(covered) != required:
        raise ValueError("Optimizer must cover every trainable parameter exactly once")
    return torch.optim.AdamW(groups, weight_decay=config["weight_decay"])


def source_loss(model, scores, labels, weights, config):
    classification = F.cross_entropy(scores / config["loss_temperature"], labels, reduction="none")
    return (classification * weights[labels]).mean() + config[
        "anchor_weight"
    ] * model.head.anchor_loss()


def train_fit(model, reader, fit_indices, labels, config, seed, device="cpu", progress=None):
    """Train a fresh source model; ``reader(indices)`` returns prefix tokens.

    Call ``make_model`` immediately beforehand. No task checkpoints are read.
    A microbatch contributes its fraction of the full effective-batch mean,
    including the generator anchor. Dropout is disabled throughout training.
    """
    fit = np.asarray(fit_indices, dtype=np.int64)
    labels = np.asarray(labels, dtype=np.int64)
    if len(fit) == 0 or len(np.unique(fit)) != len(fit):
        raise ValueError("Source-training indices must be nonempty and unique")
    classes = np.unique(labels[fit])
    ids = torch.tensor(classes, device=device)
    counts = np.bincount(labels[fit], minlength=len(model.head.semantics))
    weights = torch.tensor(
        len(fit) / (len(classes) * counts[classes]), dtype=torch.float32, device=device
    )
    optimizer = make_optimizer(model, config)
    model.eval()
    history = []
    updates = 0
    start = time.time()
    micro = config.get("micro_batch_size", config["batch_size"])
    if micro < 1 or config["batch_size"] < 1:
        raise ValueError("Batch sizes must be positive")
    for epoch in range(1, config["epochs"] + 1):
        exposure = np.zeros(len(labels), dtype=np.int32)
        losses = []
        for indices in schedule(fit, seed, epoch, config["batch_size"]):
            exposure[indices] += 1
            y = torch.tensor(np.searchsorted(classes, labels[indices]), device=device)
            optimizer.zero_grad(set_to_none=True)
            value = 0.0
            for offset in range(0, len(indices), micro):
                sub = indices[offset : offset + micro]
                scores = model(reader(sub).to(device), ids)
                loss = source_loss(model, scores, y[offset : offset + micro], weights, config)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite source loss")
                scale = len(sub) / len(indices)
                scaled = loss * scale if len(sub) != len(indices) else loss
                scaled.backward()
                value += float(loss.detach()) * scale
            parameters = [p for p in model.parameters() if p.requires_grad]
            if any(p.grad is None or not torch.isfinite(p.grad).all() for p in parameters):
                raise FloatingPointError("Missing or nonfinite source gradient")
            torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
            optimizer.step()
            updates += 1
            losses.append(value)
        if not np.all(exposure[fit] == 1) or exposure.sum() != len(fit):
            raise RuntimeError("Each source image must occur exactly once per epoch")
        history.append(
            {
                "epoch": epoch,
                "loss": float(np.mean(losses)),
                "updates": updates,
                "seconds": time.time() - start,
            }
        )
        if progress is not None:
            progress(history)
    return model, history
