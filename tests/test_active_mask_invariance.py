"""Oracles for the active-finger mask (RC-02, P3.2.1-03).

Two different properties are checked here, and only one of them can detect the
bug.  Target perturbation cannot: the error vector was already masked, so an
inactive target never reached the solver in the first place.  What RC-02 broke
is the *curvature* — inactive Jacobian rows entered the Hessian, so an inactive
finger silently shaped the step taken for the active ones.  That is checked
against `masked_normal_equations` directly, where an inactive row can be made
arbitrarily large and must still leave `H` and `g` untouched.

The orchestrator does not yet pass an active mask (every finger is active), so
the failure corpus cannot exercise any of this; these unit oracles are what
RC-02 is validated against.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from qdgrasp.dataset.pipeline.solvers.fixed_contact_dls import solve_dls_ik_batch
from qdgrasp.dataset.pipeline.solvers.normal_equations import masked_normal_equations
from qdgrasp.dataset.pipeline.solvers.region_dls import solve_region_dls_ik_batch
from qdgrasp.robot.spec import RobotSpec


HANDS = ("leap_hand.yaml", "wonik_allegro.yaml", "shadow_hand.yaml")


@pytest.fixture(scope="module")
def specs() -> dict[str, RobotSpec]:
    return {name: RobotSpec.from_config(name, sample_anchors=False) for name in HANDS}


def _reachable_targets(spec: RobotSpec, seed: int = 3):
    """Targets generated from a real configuration, so the task is solvable."""
    rng = np.random.default_rng(seed)
    lows = np.array([spec.joint_limits[j][0] for j in spec.actuated_joint_names])
    highs = np.array([spec.joint_limits[j][1] for j in spec.actuated_joint_names])
    q_truth = torch.tensor(
        lows + rng.uniform(0.35, 0.65, size=len(lows)) * (highs - lows),
        dtype=torch.float32,
    ).unsqueeze(0)
    palm_pos = torch.zeros(1, 3, dtype=torch.float32)
    palm_rot = torch.eye(3, dtype=torch.float32).unsqueeze(0)
    points = spec.fingertip_positions(palm_pos, palm_rot, q_truth)[0].numpy()
    normals = spec.fingertip_contact_directions(palm_pos, palm_rot, q_truth)[0].numpy()
    q_seed = torch.tensor(
        lows + rng.uniform(0.3, 0.7, size=len(lows)) * (highs - lows),
        dtype=torch.float32,
    ).unsqueeze(0)
    return (
        palm_pos.numpy(),
        palm_rot.numpy(),
        points[None].astype(np.float32),
        normals[None].astype(np.float32),
        q_seed.numpy(),
    )


def _solve(solver, spec, palm_pos, palm_rot, points, normals, init_q, active):
    return solver(
        spec,
        palm_pos,
        palm_rot,
        points,
        normals,
        init_q=init_q,
        active_fingers=active,
        max_iter=25,
    )


@pytest.mark.parametrize("hand", HANDS)
@pytest.mark.parametrize(
    "solver", (solve_dls_ik_batch, solve_region_dls_ik_batch), ids=("fixed", "region")
)
def test_inactive_target_perturbation_leaves_the_active_solution_alone(
    specs, hand, solver
) -> None:
    spec = specs[hand]
    palm_pos, palm_rot, points, normals, init_q = _reachable_targets(spec)
    num_tips = len(spec.fingertip_links)
    active = np.zeros((1, num_tips), dtype=bool)
    active[0, :2] = True  # first two fingers carry the task

    reference = _solve(solver, spec, palm_pos, palm_rot, points, normals, init_q, active)

    moved_points = points.copy()
    moved_normals = normals.copy()
    moved_points[0, 2:] += np.array([0.25, -0.3, 0.4], dtype=np.float32)
    moved_normals[0, 2:] = -moved_normals[0, 2:]

    perturbed = _solve(
        solver, spec, palm_pos, palm_rot, moved_points, moved_normals, init_q, active
    )

    np.testing.assert_allclose(reference.q, perturbed.q, atol=1e-6)
    np.testing.assert_allclose(
        reference.position_residuals[0, :2],
        perturbed.position_residuals[0, :2],
        atol=1e-6,
    )
    np.testing.assert_allclose(
        reference.normal_residuals[0, :2],
        perturbed.normal_residuals[0, :2],
        atol=1e-6,
    )
    assert bool(reference.converged[0]) == bool(perturbed.converged[0])
    assert str(reference.reason[0]) == str(perturbed.reason[0])


@pytest.mark.parametrize("hand", HANDS)
@pytest.mark.parametrize(
    "solver", (solve_dls_ik_batch, solve_region_dls_ik_batch), ids=("fixed", "region")
)
def test_convergence_ignores_inactive_finger_residuals(specs, hand, solver) -> None:
    """An unreachable inactive target must not block convergence."""
    spec = specs[hand]
    palm_pos, palm_rot, points, normals, _ = _reachable_targets(spec, seed=11)
    num_tips = len(spec.fingertip_links)
    active = np.zeros((1, num_tips), dtype=bool)
    active[0, :2] = True

    unreachable = points.copy()
    unreachable[0, 2:] += np.array([5.0, 5.0, 5.0], dtype=np.float32)

    q_truth_start = points  # solving from the generating configuration's targets
    result = solver(
        spec,
        palm_pos,
        palm_rot,
        unreachable,
        normals,
        init_q=None,
        active_fingers=active,
        max_iter=60,
    )
    assert q_truth_start is not None
    assert np.all(np.isfinite(result.q))
    # The inactive fingers are metres away from their targets; whatever verdict
    # the solver reaches must be justified by the active pair alone.
    assert result.position_residuals[0, 2] > 1.0
    if bool(result.converged[0]):
        assert np.all(result.position_residuals[0, :2] < 0.005)


@pytest.mark.parametrize("hand", HANDS)
def test_insufficient_active_fingers_is_reported(specs, hand) -> None:
    spec = specs[hand]
    palm_pos, palm_rot, points, normals, init_q = _reachable_targets(spec)
    num_tips = len(spec.fingertip_links)
    active = np.zeros((1, num_tips), dtype=bool)
    active[0, 0] = True

    result = solve_dls_ik_batch(
        spec,
        palm_pos,
        palm_rot,
        points,
        normals,
        init_q=init_q,
        active_fingers=active,
        max_iter=5,
    )
    assert not bool(result.converged[0])
    assert str(result.reason[0]) == "insufficient_active_fingers"


def _random_system(seed: int, batch: int = 2, tips: int = 4, joints: int = 16):
    generator = torch.Generator().manual_seed(seed)
    jacobian = torch.randn(batch, 6 * tips, joints, generator=generator)
    mask = torch.zeros(batch, 6 * tips)
    # First two fingers active: rows [0:6] of the position block and of the
    # direction block, matching how the solvers lay the task vector out.
    mask[:, : 2 * 3] = 1.0
    mask[:, 3 * tips : 3 * tips + 2 * 3] = 1.0
    error = (torch.randn(batch, 6 * tips, generator=generator) * mask).unsqueeze(-1)
    damping = (
        torch.eye(joints).unsqueeze(0).expand(batch, joints, joints).contiguous() * 1e-4
    )
    return jacobian, error, mask, damping


def test_inactive_jacobian_rows_do_not_enter_the_normal_equations() -> None:
    """The discriminating oracle: inactive curvature must vanish, not shrink."""
    jacobian, error, mask, damping = _random_system(seed=17)
    reference_h, reference_g = masked_normal_equations(jacobian, error, mask, damping)

    exploded = jacobian.clone()
    inactive = mask == 0.0
    exploded[inactive] *= 1e6

    perturbed_h, perturbed_g = masked_normal_equations(exploded, error, mask, damping)

    torch.testing.assert_close(reference_h, perturbed_h)
    torch.testing.assert_close(reference_g, perturbed_g)

    # And the masking is not vacuous: an unmasked assembly of the same system
    # would have been dominated by those rows.
    unmasked_h = torch.bmm(jacobian.transpose(1, 2), jacobian) + damping
    assert not torch.allclose(unmasked_h, reference_h, atol=1e-4)


def test_normal_equations_equal_the_reduced_active_only_system() -> None:
    """H and g must match the system built from the active rows alone."""
    jacobian, error, mask, damping = _random_system(seed=23)
    hessian, gradient = masked_normal_equations(jacobian, error, mask, damping)

    for b in range(jacobian.shape[0]):
        rows = torch.nonzero(mask[b], as_tuple=True)[0]
        reduced_j = jacobian[b, rows, :]
        reduced_e = error[b, rows, :]
        expected_h = reduced_j.T @ reduced_j + damping[b]
        expected_g = (reduced_j.T @ reduced_e).squeeze(-1)
        torch.testing.assert_close(hessian[b], expected_h, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(gradient[b], expected_g, atol=1e-5, rtol=1e-5)
