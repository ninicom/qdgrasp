"""Failure-reason and telemetry oracles for P3.2.1-04."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from qdgrasp.dataset.pipeline.solvers.fixed_contact_dls import solve_dls_ik_batch
from qdgrasp.dataset.pipeline.solvers.progress import (
    classify_failure_reasons,
    masked_jacobian_spectrum,
    meaningful_cost_decrease,
)
from qdgrasp.dataset.pipeline.solvers.region_dls import solve_region_dls_ik_batch
from qdgrasp.robot.spec import RobotSpec


SOLVERS = (solve_dls_ik_batch, solve_region_dls_ik_batch)


def _classification(**overrides):
    values = {
        "converged": np.array([False]),
        "insufficient_fingers": np.array([False]),
        "iterations": np.array([10]),
        "accepted_steps": np.array([2]),
        "rejected_steps": np.array([1]),
        "initial_cost": np.array([1.0]),
        "final_cost": np.array([0.5]),
        "raw_step_norm": np.array([0.1]),
        "projected_step_norm": np.array([0.05]),
        "limit_clipped_steps": np.array([0]),
        "jacobian_rank": np.array([6]),
        "finite": np.array([True]),
        "max_iter": 10,
    }
    values.update(overrides)
    return str(classify_failure_reasons(**values)[0])


def test_failure_classification_precedence() -> None:
    assert _classification(finite=np.array([False])) == "singular"
    assert _classification(jacobian_rank=np.array([0])) == "singular"
    assert _classification(
        raw_step_norm=np.array([1.0]),
        projected_step_norm=np.array([0.0]),
        limit_clipped_steps=np.array([2]),
    ) == "joint_limit"
    assert _classification(
        accepted_steps=np.array([0]), rejected_steps=np.array([10])
    ) == "line_search_failed"
    assert _classification(
        initial_cost=np.array([1.0]), final_cost=np.array([1.0])
    ) == "stagnation"
    assert _classification() == "max_iter"
    assert _classification(converged=np.array([True])) == "converged"
    assert _classification(
        insufficient_fingers=np.array([True])
    ) == "insufficient_active_fingers"


def test_masked_spectrum_ignores_inactive_rows() -> None:
    jacobian = torch.zeros(1, 12, 4)
    jacobian[0, 0, 0] = 2.0
    jacobian[0, 1, 1] = 1.0
    jacobian[0, 6:, :] = 1e8
    mask = torch.zeros(1, 12)
    mask[0, :2] = 1.0
    rank, condition = masked_jacobian_spectrum(jacobian, mask)
    assert int(rank[0]) == 2
    assert float(condition[0]) == pytest.approx(2.0)


def test_equal_cost_is_not_accepted_as_progress() -> None:
    current = torch.tensor([1.0, 1.0], dtype=torch.float32)
    trial = torch.tensor([1.0, 0.5], dtype=torch.float32)
    accepted = meaningful_cost_decrease(current, trial)
    assert accepted.tolist() == [False, True]


@pytest.mark.parametrize("solver", SOLVERS, ids=("fixed", "region"))
def test_solver_emits_progress_telemetry_and_specific_line_search_reason(solver) -> None:
    spec = RobotSpec.from_config("leap_hand.yaml", sample_anchors=False)
    palm_pos = torch.zeros(1, 3, dtype=torch.float32)
    palm_rot = torch.eye(3, dtype=torch.float32).unsqueeze(0)
    q_seed = torch.tensor(
        [
            0.5 * (spec.joint_limits[name][0] + spec.joint_limits[name][1])
            for name in spec.actuated_joint_names
        ],
        dtype=torch.float32,
    ).unsqueeze(0)
    target_q = q_seed.clone()
    target_q[:, 0] += 0.15
    points = spec.fingertip_positions(palm_pos, palm_rot, target_q).numpy()
    normals = spec.fingertip_contact_directions(palm_pos, palm_rot, target_q).numpy()

    result = solver(
        spec,
        palm_pos.numpy(),
        palm_rot.numpy(),
        points,
        normals,
        init_q=q_seed.numpy(),
        step_size=0.0,
        max_iter=3,
    )

    assert not bool(result.converged[0])
    assert str(result.reason[0]) == "line_search_failed"
    assert result.solver_metrics is not None
    expected = {
        "initial_cost",
        "final_cost",
        "accepted_steps",
        "rejected_steps",
        "limit_clipped_steps",
        "raw_step_norm",
        "projected_step_norm",
        "gradient_norm",
        "jacobian_rank",
        "jacobian_condition",
        "final_damping",
        "finite",
    }
    assert set(result.solver_metrics) == expected
    assert int(result.solver_metrics["accepted_steps"][0]) == 0
    assert int(result.solver_metrics["rejected_steps"][0]) == 3
    assert int(result.solver_metrics["jacobian_rank"][0]) > 0
    assert bool(result.solver_metrics["finite"][0])
