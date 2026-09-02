"""Oracles for the shared contact-state primitive (P3.2.1-02, P3.2.1-03).

The solvers grade a candidate with one function and descend another's gradient
only if these two stay linked, so this module checks the primitive itself: the
solvers' evaluation path and their autodiff path must be the same map, and the
autodiff Jacobian must match a central finite difference of that map.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch.func import jacrev

from qdgrasp.dataset.pipeline import contact_state as cs
from qdgrasp.robot.spec import RobotSpec

HANDS = ("leap_hand.yaml", "wonik_allegro.yaml", "shadow_hand.yaml")


@pytest.fixture(scope="module")
def specs() -> dict[str, RobotSpec]:
    return {name: RobotSpec.from_config(name, sample_anchors=False) for name in HANDS}


def _interior_states(spec: RobotSpec, count: int, seed: int = 7) -> torch.Tensor:
    """Joint states strictly inside the limits, away from the clamped boundary."""
    rng = np.random.default_rng(seed)
    lows = np.array([spec.joint_limits[j][0] for j in spec.actuated_joint_names])
    highs = np.array([spec.joint_limits[j][1] for j in spec.actuated_joint_names])
    span = highs - lows
    fractions = rng.uniform(0.2, 0.8, size=(count, len(lows)))
    return torch.tensor(lows + fractions * span, dtype=torch.float32)


def _palm(count: int) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.zeros(count, 3, dtype=torch.float32),
        torch.eye(3, dtype=torch.float32).unsqueeze(0).expand(count, 3, 3).contiguous(),
    )


@pytest.mark.parametrize("hand", HANDS)
def test_contact_axes_are_configured_for_every_released_hand(specs, hand) -> None:
    """The 'configured' mode is only meaningful if the profile supplies axes."""
    spec = specs[hand]
    axes = spec.fingertip_contact_axes
    assert set(axes) == set(spec.fingertip_links)
    for tip, axis in axes.items():
        assert np.linalg.norm(axis) > 1e-6, f"{hand}:{tip} has a degenerate contact axis"


@pytest.mark.parametrize("hand", HANDS)
def test_autodiff_jacobian_matches_central_finite_difference(specs, hand) -> None:
    """Section 5.1: autodiff vs central FD at 10 interior states per robot."""
    spec = specs[hand]
    states = _interior_states(spec, 10)
    palm_pos, palm_rot = _palm(1)
    normal_weight = 0.01
    eps = 1e-4

    def features(q_single: torch.Tensor) -> torch.Tensor:
        return cs.contact_residual_features(
            spec,
            q_single,
            palm_pos[0],
            palm_rot[0],
            normal_weight=normal_weight,
            mode=cs.AUTODIFF_DIRECTION_MODE,
        )

    jacobian_fn = jacrev(features)
    for q in states:
        q64 = q.to(torch.float64)
        analytic = jacobian_fn(q).to(torch.float64)
        finite = torch.zeros_like(analytic)
        for j in range(q.numel()):
            step = torch.zeros_like(q64)
            step[j] = eps
            plus = features((q64 + step).to(torch.float32)).to(torch.float64)
            minus = features((q64 - step).to(torch.float32)).to(torch.float64)
            finite[:, j] = (plus - minus) / (2.0 * eps)
        # float32 FK limits the achievable agreement; the tolerance is loose in
        # absolute terms but tight relative to the residual scale the solver
        # descends (position entries are metres, direction entries are 1e-2).
        assert torch.allclose(analytic, finite, atol=2e-3, rtol=2e-2), (
            f"{hand}: autodiff/FD mismatch, max |diff| = "
            f"{float((analytic - finite).abs().max()):.3e}"
        )


@pytest.mark.parametrize("hand", HANDS)
def test_evaluation_and_autodiff_paths_describe_the_same_positions(specs, hand) -> None:
    """Contact points must agree between the batched and single-sample paths."""
    spec = specs[hand]
    q = _interior_states(spec, 3)
    palm_pos, palm_rot = _palm(3)
    positions, _ = cs.contact_state(spec, palm_pos, palm_rot, q)
    num_tips = len(spec.fingertip_links)
    for index in range(3):
        flat = cs.contact_residual_features(
            spec,
            q[index],
            palm_pos[index],
            palm_rot[index],
            normal_weight=1.0,
        )
        assert torch.allclose(
            flat[: 3 * num_tips].reshape(num_tips, 3), positions[index], atol=1e-6
        )


# Measured angle between the graded direction (configured contact axis) and the
# differentiated one (parent-to-tip), averaged over interior joint states.  This
# is the size of RC-01 per hand, and it is not uniform: Allegro's configured
# axis coincides with its parent-to-tip vector, so an RC-01 fix cannot explain
# Allegro's failures at all.
RC01_DIVERGENCE_DEGREES = {
    "leap_hand.yaml": 18.1,
    "wonik_allegro.yaml": 0.0,
    "shadow_hand.yaml": 22.0,
}


@pytest.mark.parametrize("hand", HANDS)
def test_rc01_divergence_is_hand_specific(specs, hand) -> None:
    """Pins how far apart the graded and differentiated directions actually are.

    An RC-01 intervention is expected to move the leap and shadow cells of the
    failure corpus and to leave the Allegro cells alone; this test is what makes
    that prediction falsifiable rather than a story told after the run.
    """
    spec = specs[hand]
    q = _interior_states(spec, 5)
    palm_pos, palm_rot = _palm(5)
    _, configured = cs.contact_state(spec, palm_pos, palm_rot, q, mode="configured")
    _, parent_to_tip = cs.contact_state(
        spec, palm_pos, palm_rot, q, mode="parent_to_tip"
    )
    dots = torch.sum(configured * parent_to_tip, dim=-1).clamp(-1.0, 1.0)
    measured = float(torch.rad2deg(torch.acos(dots)).mean())
    expected = RC01_DIVERGENCE_DEGREES[hand]
    assert measured == pytest.approx(expected, abs=1.0), (
        f"{hand}: direction divergence moved from {expected} to {measured:.2f} deg"
    )


@pytest.mark.parametrize("hand", HANDS)
def test_directions_are_unit_vectors(specs, hand) -> None:
    spec = specs[hand]
    q = _interior_states(spec, 4)
    palm_pos, palm_rot = _palm(4)
    for mode in ("configured", "parent_to_tip"):
        _, directions = cs.contact_state(spec, palm_pos, palm_rot, q, mode=mode)
        norms = torch.linalg.norm(directions, dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)
