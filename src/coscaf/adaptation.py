"""Generator-only transductive adaptation with fixed visual evidence.

One optimizer step accumulates gradients over the complete unlabeled target
set. The batch size controls memory, not the adaptation protocol. The source
readout, semantic descriptors, prompts, and visual encoder are never updated.
"""

from __future__ import annotations

import copy
import hashlib

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .model import GENERATOR, prototypes
from .transport import (
    cross_view_teacher,
    partial_unbalanced_transport,
    refine_teacher,
    soft_semantic_cross_view_teacher,
)


def _as_tensor(value, device):
    """Make a detached float32 copy of an array-backed feature slice."""
    if torch.is_tensor(value):
        return value.detach().to(device=device, dtype=torch.float32).clone()
    return torch.as_tensor(np.array(value, dtype=np.float32, copy=True), device=device)


def _validate_inputs(head, ids, cache, seen_mask, gamma, hp, setting, device, batch, steps):
    """Reject mismatched class order, labeled inputs and invalid cache shapes."""
    required = {
        "global",
        "local",
        "moment",
        "view10_global",
        "view12_global",
        "view10_scores",
        "view12_scores",
    }
    allowed = required | {"prior"}
    if set(cache) - allowed:
        raise ValueError(f"Unexpected feature-cache fields: {sorted(set(cache) - allowed)}")
    if required - set(cache):
        raise ValueError(f"Missing feature-cache fields: {sorted(required - set(cache))}")
    if head.variant == "latent" and "prior" not in cache:
        raise ValueError("Latent readouts require cached prior evidence")
    n = len(cache["global"])
    c = len(ids)
    k = head.modes
    d = head.center_weight.shape[1]
    if n < 1 or c < 2 or batch < 1:
        raise ValueError("Require target samples, at least two classes, and a positive batch size")
    candidate_ids = torch.as_tensor(ids, dtype=torch.long)
    if candidate_ids.ndim != 1 or candidate_ids.unique().numel() != c:
        raise ValueError("Candidate class indices must be unique and one-dimensional")
    if int(candidate_ids.min()) < 0 or int(candidate_ids.max()) >= len(head.semantics):
        raise ValueError("Candidate class index is outside the semantic matrix")
    seen = torch.as_tensor(seen_mask)
    if seen.dtype != torch.bool or tuple(seen.shape) != (c,):
        raise ValueError("seen_mask must be a Boolean vector matching candidate classes")
    shapes = {
        "global": (n, d),
        "local": (n, c, k, d),
        "moment": (n, c),
        "prior": (n, c, k),
        "view10_global": (n, d),
        "view12_global": (n, d),
        "view10_scores": (n, c),
        "view12_scores": (n, c),
    }
    for key, value in cache.items():
        if tuple(value.shape) != shapes[key]:
            raise ValueError(f"Invalid shape for {key}: expected {shapes[key]}, got {value.shape}")
        # Local evidence may be memory mapped. Bound validation memory
        # independently of the full target-by-class-by-mode cache shape.
        elements_per_row = int(np.prod(value.shape[1:]))
        check_batch = max(1, min(batch, 8_000_000 // max(1, elements_per_row)))
        for start in range(0, n, check_batch):
            if not bool(torch.isfinite(torch.as_tensor(value[start : start + check_batch])).all()):
                raise ValueError(f"Non-finite cached evidence in {key}")
    if head.center_weight.device != torch.empty(0, device=device).device:
        raise ValueError("Source readout and adaptation must use the same device")
    if not np.isfinite(gamma):
        raise ValueError("Source gamma must be finite")
    if hp["engine"] not in ("geometry_ot", "sun_soft_ot"):
        raise ValueError("Unknown transport engine")
    if any(hp[name] <= 0 for name in ("lr", "assignment_temperature", "consistency_temperature")):
        raise ValueError("Adaptation learning rate and temperatures must be positive")
    count = hp["steps"] if steps is None else steps
    if count < 0 or int(count) != count:
        raise ValueError("Adaptation steps must be a nonnegative integer")
    if setting.get("lr_factor", 1.0) <= 0 or not 0 <= setting.get("local_target_mix", 0.0) <= 1:
        raise ValueError("Invalid learning-rate multiplier or local-target mixture")


@torch.no_grad()
def targets_with_mass(source, semantics, cache, seen_mask, gamma, hp, device):
    """Build fixed class/mode targets from the source cross-layer evidence.

    ``gamma`` is the source calibration offset fitted on development data.
    ``seen_mask`` indexes candidate classes; it is not a target-image label.
    The transport's row masses become sample reliability weights, while its
    column masses weight the class and mode alignment objectives.
    """
    seen = torch.as_tensor(seen_mask, device=device, dtype=torch.bool)
    gx = _as_tensor(cache["global"], device)
    views = [_as_tensor(cache[k], device) for k in ("view10_global", "view12_global")]
    probabilities = [
        F.softmax(
            (_as_tensor(cache[k], device) - gamma * seen) / hp["assignment_temperature"],
            -1,
        )
        for k in ("view10_scores", "view12_scores")
    ]
    if hp["engine"] == "sun_soft_ot":
        (teacher, reliability, stats) = soft_semantic_cross_view_teacher(
            probabilities, semantics, seen, hp
        )
        (teacher, refinement) = refine_teacher(teacher, gx, seen, hp)
        stats.update(refinement)
    else:
        (teacher, reliability, stats) = cross_view_teacher(probabilities, seen, hp)
    (transport, ot_stats) = partial_unbalanced_transport(teacher, seen, reliability, hp)
    stats.update(ot_stats)
    (base, proto) = prototypes(source, semantics)
    mode_sim = torch.stack([torch.einsum("nd,ckd->nck", v, proto) for v in views]).mean(0)
    resp = (mode_sim / hp.get("mode_temperature", 0.05)).softmax(-1)
    joint = transport[:, :, None] * resp
    mass = transport.sum(0)
    mode_mass = joint.sum(0)
    class_center = F.normalize(transport.T @ gx / mass[:, None].clamp_min(1e-12), dim=-1)
    mode_center = F.normalize(
        torch.einsum("nck,nd->ckd", joint, gx) / mode_mass[:, :, None].clamp_min(1e-12),
        dim=-1,
    )
    teacher = transport / transport.sum(-1, keepdim=True).clamp_min(1e-12)
    positive = teacher.argmax(-1)
    geometry = F.normalize(semantics, dim=-1) @ F.normalize(semantics, dim=-1).T
    geometry.fill_diagonal_(-torch.inf)
    neighbors = geometry.topk(
        min(hp.get("hard_negative_neighbors", 5), len(semantics) - 1), dim=-1
    ).indices
    candidates = neighbors[positive]
    source_log = torch.stack([p.clamp_min(1e-12).log() for p in probabilities]).mean(0)
    negative = candidates.gather(
        1, source_log.gather(1, candidates).argmax(-1, keepdim=True)
    ).squeeze(1)
    stats.update(
        reliability_mean=float(reliability.mean()),
        active_modes=int((mode_mass >= hp.get("min_mode_mass", 0.05)).sum()),
    )
    return (
        dict(
            transport=transport,
            teacher=teacher,
            reliability=reliability,
            class_mass=mass,
            mode_mass=mode_mass,
            class_center=class_center,
            mode_center=mode_center,
            positive=positive,
            negative=negative,
            source_base=base,
            source_proto=proto,
            seen=seen,
        ),
        stats,
    )


def _state_digest(items):
    h = hashlib.sha256()
    for n, v in sorted(items):
        h.update(n.encode())
        h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def _to_cpu(v):
    if torch.is_tensor(v):
        return v.detach().cpu().clone()
    if isinstance(v, dict):
        return {k: _to_cpu(x) for (k, x) in v.items()}
    if isinstance(v, list):
        return [_to_cpu(x) for x in v]
    if isinstance(v, tuple):
        return tuple((_to_cpu(x) for x in v))
    return copy.deepcopy(v)


def decomposed_kl(logp, teacher, seen, group_weight):
    """Teacher KL with an optional weight on the seen/unseen group term.

    Released configurations use ``group_weight=1``, the full class-wise KL.
    """
    full = F.kl_div(logp, teacher, reduction="none").sum(-1)
    group = 0.0
    for mask in (seen, ~seen):
        mass = teacher[:, mask].sum(-1)
        group = group + mass * (mass.clamp_min(1e-12).log() - torch.logsumexp(logp[:, mask], -1))
    return full + (float(group_weight) - 1) * group


def chunk_features(cache, sl, device):
    """Transfer only detached scoring evidence for a target slice."""
    return {
        k: _as_tensor(cache[k][sl], device)
        for k in ("global", "local", "moment", "prior")
        if k in cache
    }


@torch.no_grad()
def cached_scores(head, ids, params, cache, device, batch=64):
    """Recompute complete class scores with a supplied generator state."""
    if batch < 1:
        raise ValueError("Scoring batch size must be positive")
    rows = []
    for off in range(0, len(cache["global"]), batch):
        z = head.score_features(chunk_features(cache, slice(off, off + batch), device), ids, params)
        if not torch.isfinite(z).all():
            raise FloatingPointError("Non-finite class scores")
        rows.append(z.cpu().numpy())
    if not rows:
        raise ValueError("Cannot score an empty feature cache")
    return np.concatenate(rows)


@torch.no_grad()
def local_targets(head, ids, source, cache, teacher, device, batch):
    """Separate branch responsibilities; vectors come from the matching branch."""
    n = len(cache["global"])
    c = len(ids)
    k = head.modes
    d = head.center_weight.shape[1]
    sums = torch.zeros(c, k, d, device=device)
    mass = torch.zeros(c, k, device=device)
    for off in range(0, n, batch):
        sl = slice(off, min(n, off + batch))
        f = chunk_features(cache, sl, device)
        (_, diag) = head.score_features(f, ids, source, diagnostics=True)
        post = teacher[sl]
        for weight, key, values in [
            (
                1 - head.local_weight,
                "global_responsibility",
                f["global"][:, None, None],
            ),
            (head.local_weight, "local_responsibility", f["local"]),
        ]:
            w = post[:, :, None] * diag[key] * weight
            sums += (w[:, :, :, None] * values).sum(0)
            mass += w.sum(0)
    centers = F.normalize(sums / mass[:, :, None].clamp_min(1e-12), dim=-1)
    return (centers, mass)


def adapt_evidence(
    head,
    ids,
    cache,
    seen_mask,
    gamma,
    hp,
    setting,
    device="cpu",
    batch=64,
    steps=None,
    progress=None,
):
    """Adapt a copy of the shared generator using unlabeled target evidence.

    Args:
        head: Source readout placed on ``device``. It remains unchanged.
        ids: Ordered global candidate-class indices into ``head.semantics``.
        cache: Detached arrays with ``global``, ``local``, ``moment`` and
            cross-layer ``view10_global``, ``view12_global``, ``view10_scores``,
            ``view12_scores``. Latent readouts additionally require ``prior``.
        seen_mask: Boolean vector in the same candidate order as ``ids``.
        gamma: Source calibration offset, fixed before target adaptation.
        hp: Transport and objective hyperparameters from a configuration file.
        setting: Learning-rate multiplier, local-target mixture and KL weight.
        device: PyTorch device shared with the source readout.
        batch: Memory chunk size. Every step uses the entire target set.
        steps: Optional replacement for the configured number of steps.
        progress: Optional callback receiving the accumulated loss history.

    Returns:
        Raw scores, a separate adapted generator, optimizer state, and numeric
        diagnostics. Target labels and target-fitted calibration are not inputs.
    """
    _validate_inputs(head, ids, cache, seen_mask, gamma, hp, setting, device, batch, steps)
    source = {k: getattr(head, k).detach().clone() for k in GENERATOR}
    params = {k: nn.Parameter(v.clone()) for (k, v) in source.items()}
    sem = head.semantics[ids].detach()
    before = _state_digest(head.state_dict().items())
    hp = copy.deepcopy(hp)
    hp["lr"] *= setting.get("lr_factor", 1.0)
    (t, stats) = targets_with_mass(source, sem, cache, seen_mask, gamma, hp, device)
    mix = setting.get("local_target_mix", 0.0)
    if mix:
        (center, mass) = local_targets(head, ids, source, cache, t["transport"], device, batch)
        t["mode_center"] = F.normalize((1 - mix) * t["mode_center"] + mix * center, dim=-1)
        t["mode_mass"] = (1 - mix) * t["mode_mass"] + mix * mass
    opt = torch.optim.Adam(params.values(), lr=hp["lr"])
    history = []
    n = len(cache["global"])
    den = t["reliability"].sum().clamp_min(1e-12)
    count = hp["steps"] if steps is None else steps
    for step in range(1, int(count) + 1):
        opt.zero_grad(set_to_none=True)
        totals = {"consistency": 0.0, "ranking": 0.0}
        for off in range(0, n, batch):
            sl = slice(off, min(n, off + batch))
            r = t["reliability"][sl]
            z = head.score_features(chunk_features(cache, sl, device), ids, params)
            logp = F.log_softmax((z - gamma * t["seen"]) / hp["consistency_temperature"], -1)
            values = decomposed_kl(
                logp, t["teacher"][sl], t["seen"], setting.get("group_weight", 1.0)
            )
            kl = (values * r).sum() / den
            ix = torch.arange(len(z), device=device)
            rank = F.relu(
                hp.get("ranking_margin", 0.03) + z[ix, t["negative"][sl]] - z[ix, t["positive"][sl]]
            )
            if setting.get("group_weight", 1.0) != 1:
                rank = rank * (t["seen"][t["negative"][sl]] == t["seen"][t["positive"][sl]])
            rank = (rank * r).sum() / den
            loss = hp["consistency_weight"] * kl + hp.get("ranking_weight", 0.0) * rank
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite adaptation loss")
            loss.backward()
            totals["consistency"] += float(kl.detach())
            totals["ranking"] += float(rank.detach())
        (base, p) = prototypes(params, sem)
        ca = (
            (1 - (F.normalize(base, dim=-1) * t["class_center"]).sum(-1)) * t["class_mass"]
        ).sum() / t["class_mass"].sum().clamp_min(1e-12)
        active = t["mode_mass"] >= hp.get("min_mode_mass", 0.05)
        ma = ((1 - (p * t["mode_center"]).sum(-1))[active] * t["mode_mass"][active]).sum() / t[
            "mode_mass"
        ][active].sum().clamp_min(1e-12)
        anchor = torch.stack(
            [
                (v - source[k]).square().mean() / source[k].square().mean().clamp_min(1e-08)
                for (k, v) in params.items()
            ]
        ).mean()
        rel = semantic_relation_loss(base, sem)
        top = source_topology_loss(base, p, t["source_base"], t["source_proto"])
        reg = (
            hp["class_alignment_weight"] * ca
            + hp["mode_alignment_weight"] * ma
            + hp["anchor_weight"] * anchor
            + hp["relation_weight"] * rel
            + hp.get("topology_weight", 0.0) * top
        )
        if not torch.isfinite(reg):
            raise FloatingPointError("Non-finite adaptation regularizer")
        reg.backward()
        if any(p.grad is None or not torch.isfinite(p.grad).all() for p in params.values()):
            raise FloatingPointError("Missing or non-finite generator gradient")
        grad = torch.nn.utils.clip_grad_norm_(params.values(), hp.get("gradient_clip", 5.0))
        opt.step()
        totals.update(
            step=step,
            class_alignment=float(ca.detach()),
            mode_alignment=float(ma.detach()),
            gradient_norm=float(grad),
        )
        history.append(totals)
        if progress:
            progress(history)
    z = cached_scores(head, ids, params, cache, device, batch)
    if _state_digest(head.state_dict().items()) != before:
        raise RuntimeError("Source readout was modified during adaptation")
    return dict(
        scores=z,
        generator=_to_cpu(params),
        optimizer=_to_cpu(opt.state_dict()),
        history=history,
        diagnostics=stats,
        head_unchanged=True,
        setting=setting,
        lr=hp["lr"],
    )


def semantic_relation_loss(current, attributes, temperature=0.1):
    """Match off-diagonal class relations to the fixed semantic geometry."""
    n = len(attributes)
    if n < 2:
        return current.sum() * 0
    mask = ~torch.eye(n, dtype=torch.bool, device=attributes.device)
    prior = F.normalize(attributes, dim=-1) @ F.normalize(attributes, dim=-1).T
    geometry = F.normalize(current, dim=-1) @ F.normalize(current, dim=-1).T
    target = F.softmax(prior[mask].reshape(n, n - 1) / temperature, dim=-1).detach()
    prediction = F.log_softmax(geometry[mask].reshape(n, n - 1) / temperature, dim=-1)
    return F.kl_div(prediction, target, reduction="batchmean")


def source_topology_loss(current_base, current_proto, source_base, source_proto):
    """Preserve source center and mean-prototype cosine Gram matrices."""

    def gram(value):
        value = F.normalize(value, dim=-1)
        return value @ value.T

    current_center = F.normalize(current_proto.mean(1), dim=-1)
    source_center = F.normalize(source_proto.mean(1), dim=-1)
    n = len(current_base)
    if n < 2:
        return current_base.sum() * 0
    mask = ~torch.eye(n, dtype=torch.bool, device=current_base.device)
    base = (gram(current_base)[mask] - gram(source_base)[mask]).square().mean()
    proto = (gram(current_center)[mask] - gram(source_center)[mask]).square().mean()
    return (base + proto) / 2
