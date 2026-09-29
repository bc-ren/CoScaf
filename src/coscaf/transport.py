"""Label-free cross-layer teachers and partial class transport.

All class partitions here describe the candidate label space, never target
image labels. Sample reliability controls transported mass; class balancing
acts separately inside the seen and unseen groups.
"""

from __future__ import annotations

import torch
from torch.nn import functional as F


def scale_partial_mass(reliability, rho):
    """Rescale reliability to the requested feasible mean mass, capped at one.

    Zero entries stay rejected when any positive reliability is available. If
    every entry is zero, the implementation falls back to equal reliability;
    the subsequent scaling assigns the requested mean mass uniformly.
    """
    rho = float(rho)
    if not 0 < rho <= 1:
        raise ValueError("partial mass rho must be in (0,1]")
    r = reliability.clamp_min(0)
    if float(r.max()) <= 0:
        r = torch.ones_like(r)
    rho = min(rho, float((r > 0).float().mean()))
    (low, high) = (0.0, 1.0)
    while float(torch.clamp(high * r, max=1).mean()) < rho - 1e-08:
        high *= 2
    for _ in range(40):
        middle = (low + high) / 2
        if float(torch.clamp(middle * r, max=1).mean()) < rho:
            low = middle
        else:
            high = middle
    return torch.clamp(high * r, max=1)


def cross_view_teacher(probabilities, seen_mask, hp):
    """Geometric-mean teacher and reliability from deterministic layer views."""
    if len(probabilities) < 2:
        raise ValueError("Cross-layer consistency requires at least two views")
    views = torch.stack(probabilities)
    teacher = views.clamp_min(1e-12).log().mean(0).softmax(-1)
    top = views.argmax(-1)
    agreement = (top == top[0:1]).all(0)
    group = seen_mask[top]
    group_agreement = (group == group[0:1]).all(0)
    mean = views.mean(0).clamp_min(1e-12)
    js = (views * (views.clamp_min(1e-12).log() - mean.log())).sum(-1).mean(0)
    values = torch.topk(teacher, min(2, teacher.shape[1]), dim=-1).values
    margin = values[:, 0] - values[:, -1]
    reliability = torch.exp(-js / float(hp.get("view_js_temperature", 0.02)))
    reliability *= torch.sigmoid(
        (margin - float(hp.get("view_margin", 0.0)))
        / float(hp.get("view_margin_temperature", 0.02))
    )
    reliability *= (agreement & group_agreement).to(reliability)
    reliability = scale_partial_mass(reliability, hp.get("partial_mass", 0.8))
    return (
        teacher,
        reliability,
        dict(
            view_top1_agreement=float(agreement.float().mean()),
            view_group_agreement=float(group_agreement.float().mean()),
            view_js_mean=float(js.mean()),
            partial_mass=float(reliability.mean()),
        ),
    )


def partial_unbalanced_transport(probability, seen_mask, reliability, hp):
    """Semi-balanced entropic OT with exact sample and relaxed class marginals."""
    epsilon = float(hp.get("ot_epsilon", 0.3))
    prior_shrink = float(hp.get("ot_prior_shrink", 0.25))
    relaxation = float(hp.get("ot_relaxation", 0.8))
    iterations = int(hp.get("ot_iterations", 30))
    if epsilon <= 0 or not 0 <= prior_shrink <= 1 or (not 0 <= relaxation <= 1):
        raise ValueError("Invalid OT hyperparameters")
    result = torch.zeros_like(probability)
    for mask in (seen_mask, ~seen_mask):
        if int(mask.sum()) == 0:
            continue
        group_total = probability[:, mask].sum(-1)
        row_target = reliability * group_total
        conditional = probability[:, mask] / group_total[:, None].clamp_min(1e-12)
        empirical = (conditional * row_target[:, None]).sum(0)
        empirical = empirical / empirical.sum().clamp_min(1e-12)
        uniform = torch.full_like(empirical, 1 / len(empirical))
        prior = (1 - prior_shrink) * empirical + prior_shrink * uniform
        column_target = prior * row_target.sum()
        kernel = conditional.clamp_min(1e-08).pow(1 / epsilon)
        plan = kernel / kernel.sum(-1, keepdim=True).clamp_min(1e-12)
        plan *= row_target[:, None]
        for _ in range(iterations):
            column = plan.sum(0)
            factor = (column_target / column.clamp_min(1e-12)).pow(relaxation)
            plan *= factor.clamp(0.0001, 10000.0)[None]
            plan *= (row_target / plan.sum(-1).clamp_min(1e-12))[:, None]
        result[:, mask] = plan
    row_error = (result.sum(-1) - reliability).abs().max()
    mass = result.sum(0)
    nonzero = mass > 1e-12
    mass_cv = (
        mass[nonzero].std(unbiased=False) / mass[nonzero].mean().clamp_min(1e-12)
        if int(nonzero.sum()) > 1
        else mass.new_zeros(())
    )
    return (
        result,
        dict(
            ot_row_mass_max_error=float(row_error),
            ot_class_mass_cv=float(mass_cv),
            ot_active_classes=int(nonzero.sum()),
        ),
    )


def soft_semantic_cross_view_teacher(probabilities, attributes, seen_mask, hp):
    """Build a cross-view teacher without requiring identical fine-class top-1.

    Reliability remains zero when the views disagree on the seen/unseen group.
    Inside a group, ``semantic`` mode gives partial credit to predictions that
    are close under the immutable class-attribute geometry. ``topk`` accepts an
    intersecting top-k set; ``group`` requires only group agreement; and
    ``top1`` requires exact class agreement.
    """
    if len(probabilities) < 2:
        raise ValueError("Soft cross-view evidence requires at least two views")
    if attributes.ndim != 2 or probabilities[0].shape[1] != len(attributes):
        raise ValueError("Attribute and probability class dimensions disagree")
    views = torch.stack(probabilities)
    teacher = views.clamp_min(1e-12).log().mean(0).softmax(-1)
    top = views.argmax(-1)
    exact = (top == top[0:1]).all(0)
    group = seen_mask[top]
    group_agreement = (group == group[0:1]).all(0)
    mean = views.mean(0).clamp_min(1e-12)
    js = (views * (views.clamp_min(1e-12).log() - mean.log())).sum(-1).mean(0)
    values = torch.topk(teacher, min(2, teacher.shape[1]), dim=-1).values
    margin = values[:, 0] - values[:, -1]
    mode = str(hp.get("agreement_mode", "semantic"))
    if mode == "top1":
        agreement = exact.to(teacher)
        semantic_similarity = exact.to(teacher)
    elif mode == "topk":
        k = min(int(hp.get("agreement_topk", 3)), teacher.shape[1])
        if k < 1:
            raise ValueError("agreement_topk must be positive")
        ranked = views.topk(k, dim=-1).indices
        overlap = (ranked[0, :, :, None] == ranked[1, :, None, :]).any(-1).any(-1)
        for view in range(2, len(probabilities)):
            overlap &= (ranked[0, :, :, None] == ranked[view, :, None, :]).any(-1).any(-1)
        agreement = overlap.to(teacher)
        semantic_similarity = agreement
    elif mode == "group":
        agreement = torch.ones_like(margin)
        semantic_similarity = agreement
    elif mode == "semantic":
        normalized = F.normalize(attributes, dim=-1)
        similarities = []
        for left in range(len(probabilities)):
            for right in range(left + 1, len(probabilities)):
                similarities.append((normalized[top[left]] * normalized[top[right]]).sum(-1))
        semantic_similarity = torch.stack(similarities).mean(0)
        temperature = float(hp.get("semantic_agreement_temperature", 0.05))
        if temperature <= 0:
            raise ValueError("semantic_agreement_temperature must be positive")
        threshold = float(hp.get("semantic_agreement_threshold", 0.6))
        agreement = torch.sigmoid((semantic_similarity - threshold) / temperature)
        floor = float(hp.get("semantic_agreement_floor", 0.0))
        if not 0 <= floor < 1:
            raise ValueError("semantic_agreement_floor must be in [0,1)")
        agreement = floor + (1.0 - floor) * agreement
    else:
        raise ValueError(f"Unknown agreement mode: {mode}")
    js_temperature = float(hp.get("view_js_temperature", 0.05))
    margin_temperature = float(hp.get("view_margin_temperature", 0.02))
    if js_temperature <= 0 or margin_temperature <= 0:
        raise ValueError("View temperatures must be positive")
    reliability = torch.exp(-js / js_temperature)
    reliability *= torch.sigmoid((margin - float(hp.get("view_margin", 0.0))) / margin_temperature)
    reliability *= agreement
    reliability *= group_agreement.to(reliability)
    reliability = scale_partial_mass(reliability, hp.get("partial_mass", 0.8))
    return (
        teacher,
        reliability,
        dict(
            agreement_mode=mode,
            exact_top1_agreement=float(exact.float().mean()),
            view_group_agreement=float(group_agreement.float().mean()),
            semantic_agreement_mean=float(semantic_similarity.mean()),
            reliability_nonzero_fraction=float((reliability > 0).float().mean()),
            view_js_mean=float(js.mean()),
            partial_mass=float(reliability.mean()),
        ),
    )


def _group_mass_cv(probability, seen_mask):
    """Mean within-group coefficient of variation of soft class mass."""
    values = []
    for mask in (seen_mask, ~seen_mask):
        mass = probability[:, mask].sum(0)
        if len(mass) > 1:
            values.append(mass.std(unbiased=False) / mass.mean().clamp_min(1e-12))
    return torch.stack(values).mean() if values else probability.new_zeros(())


def _balance_within_groups(probability, seen_mask, strength, iterations):
    """Debias class mass without changing per-sample seen/unseen evidence.

    The operation rescales conditional class probabilities independently inside
    the seen and unseen candidate groups. Each sample's total probability for
    either group is preserved exactly, so calibrated stacking remains the only
    mechanism that moves evidence between the two groups.
    """
    if strength <= 0 or iterations <= 0:
        return probability
    refined = probability.clone()
    for mask in (seen_mask, ~seen_mask):
        if int(mask.sum()) < 2:
            continue
        group_total = refined[:, mask].sum(-1, keepdim=True)
        conditional = refined[:, mask] / group_total.clamp_min(1e-12)
        for _ in range(int(iterations)):
            mass = conditional.mean(0).clamp_min(1e-12)
            geometric_mean = mass.log().mean().exp()
            factor = (geometric_mean / mass).pow(float(strength))
            conditional = conditional * factor
            conditional = conditional / conditional.sum(-1, keepdim=True).clamp_min(1e-12)
        refined[:, mask] = conditional * group_total
    return refined / refined.sum(-1, keepdim=True).clamp_min(1e-12)


def refine_teacher(probability, x, seen_mask, hp):
    """Refine source responsibilities using class-mass and centroid evidence."""
    before_cv = _group_mass_cv(probability, seen_mask)
    refined = _balance_within_groups(
        probability,
        seen_mask,
        hp.get("mass_balance_strength", 0.0),
        hp.get("mass_balance_iterations", 0),
    )
    blend = float(hp.get("centroid_blend", 0.0))
    if blend > 0:
        normal_x = F.normalize(x, dim=-1)
        mass = refined.sum(0)
        centers = refined.T @ normal_x / mass[:, None].clamp_min(1e-12)
        center_scores = normal_x @ F.normalize(centers, dim=-1).T
        center_probability = torch.zeros_like(refined)
        for mask in (seen_mask, ~seen_mask):
            if int(mask.sum()) == 0:
                continue
            group_total = refined[:, mask].sum(-1, keepdim=True)
            conditional = F.softmax(
                center_scores[:, mask] / float(hp.get("centroid_temperature", 0.05)),
                dim=-1,
            )
            center_probability[:, mask] = conditional * group_total
        mixed = torch.exp(
            (1.0 - blend) * refined.clamp_min(1e-12).log()
            + blend * center_probability.clamp_min(1e-12).log()
        )
        restored = torch.zeros_like(refined)
        for mask in (seen_mask, ~seen_mask):
            if int(mask.sum()) == 0:
                continue
            group_total = refined[:, mask].sum(-1, keepdim=True)
            conditional = mixed[:, mask] / mixed[:, mask].sum(-1, keepdim=True).clamp_min(1e-12)
            restored[:, mask] = conditional * group_total
        refined = restored
        refined = _balance_within_groups(
            refined,
            seen_mask,
            hp.get("mass_balance_strength", 0.0),
            hp.get("mass_balance_iterations", 0),
        )
    after_cv = _group_mass_cv(refined, seen_mask)
    group_error = torch.stack(
        [
            (refined[:, mask].sum(-1) - probability[:, mask].sum(-1)).abs().max()
            for mask in (seen_mask, ~seen_mask)
            if int(mask.sum())
        ]
    ).max()
    return (
        refined,
        dict(
            teacher_mass_cv_before=float(before_cv),
            teacher_mass_cv_after=float(after_cv),
            teacher_group_mass_max_error=float(group_error),
        ),
    )
