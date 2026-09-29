"""Conservation and reliability checks for label-free target transport."""

import pytest
import torch
from torch.nn import functional as F

from coscaf.transport import (
    cross_view_teacher,
    partial_unbalanced_transport,
    refine_teacher,
    scale_partial_mass,
    soft_semantic_cross_view_teacher,
)


def test_partial_mass_respects_hard_rejection_and_budget():
    reliability = torch.tensor([0.0, 0.1, 0.5, 1.0])
    scaled = scale_partial_mass(reliability, 0.6)
    assert scaled[0] == 0
    assert torch.all((scaled >= 0) & (scaled <= 1))
    torch.testing.assert_close(scaled.mean(), torch.tensor(0.6))
    # Only three of four samples are eligible: an infeasible request is capped.
    torch.testing.assert_close(scale_partial_mass(reliability, 0.9).mean(), torch.tensor(0.75))


def test_all_zero_reliability_has_documented_uniform_fallback():
    torch.testing.assert_close(scale_partial_mass(torch.zeros(4), 0.65), torch.full((4,), 0.65))
    with pytest.raises(ValueError, match="partial mass"):
        scale_partial_mass(torch.ones(4), 0)


def test_hard_vs_group_agreement():
    a = torch.tensor([[0.8, 0.1, 0.05, 0.05], [0.7, 0.1, 0.1, 0.1]])
    b = torch.tensor([[0.1, 0.8, 0.05, 0.05], [0.6, 0.2, 0.1, 0.1]])
    seen = torch.tensor([True, True, False, False])
    hp = {"partial_mass": 0.4, "agreement_mode": "group"}
    _, hard, _ = cross_view_teacher([a, b], seen, hp)
    teacher, soft, _ = soft_semantic_cross_view_teacher([a, b], torch.eye(4), seen, hp)
    assert hard[0] == 0
    assert soft[0] > 0
    torch.testing.assert_close(teacher.sum(1), torch.ones(2))


@pytest.mark.parametrize("relaxation", [0.0, 0.8, 1.0])
def test_transport_preserves_sample_and_group_masses(relaxation):
    generator = torch.Generator().manual_seed(34)
    probability = torch.randn(9, 5, generator=generator).softmax(-1)
    seen = torch.tensor([True, False, True, False, False])
    reliability = torch.linspace(0, 1, 9)
    hp = {"ot_epsilon": 0.4, "ot_prior_shrink": 0.5, "ot_relaxation": relaxation}
    plan, stats = partial_unbalanced_transport(probability, seen, reliability, hp)
    torch.testing.assert_close(plan.sum(1), reliability, atol=2e-7, rtol=2e-6)
    for group in (seen, ~seen):
        torch.testing.assert_close(
            plan[:, group].sum(1),
            reliability * probability[:, group].sum(1),
            atol=2e-7,
            rtol=2e-6,
        )
    assert plan.min() >= 0
    assert torch.count_nonzero(plan[0]) == 0
    assert stats["ot_row_mass_max_error"] < 1e-6
    permutation = torch.tensor([4, 2, 0, 3, 1])
    permuted, _ = partial_unbalanced_transport(
        probability[:, permutation], seen[permutation], reliability, hp
    )
    torch.testing.assert_close(permuted, plan[:, permutation])


def test_centroid_refinement_keeps_seen_unseen_evidence():
    generator = torch.Generator().manual_seed(19)
    probability = torch.randn(11, 5, generator=generator).softmax(-1)
    features = F.normalize(torch.randn(11, 7, generator=generator), dim=-1)
    seen = torch.tensor([True, True, False, False, False])
    hp = {
        "mass_balance_strength": 0.15,
        "mass_balance_iterations": 3,
        "centroid_blend": 0.25,
        "centroid_temperature": 0.05,
    }
    refined, stats = refine_teacher(probability, features, seen, hp)
    for group in (seen, ~seen):
        torch.testing.assert_close(refined[:, group].sum(1), probability[:, group].sum(1))
    assert stats["teacher_group_mass_max_error"] < 1e-6
    assert not torch.allclose(refined, probability)
