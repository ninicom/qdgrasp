"""Controllable task-space command oracles for P3.2.1-08."""

from __future__ import annotations

import numpy as np

from qdgrasp.robot.transmission.command import plan_controllable_task_command
from qdgrasp.robot.transmission.contracts import TransmissionState


def _state(*, controls=(0.0, 0.0), moment=None) -> TransmissionState:
    if moment is None:
        moment = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    return TransmissionState(
        joint_names=("j0", "j1", "j2"),
        actuator_names=("u0", "u1"),
        joint_position=np.zeros(3),
        actuator_coordinate=np.asarray(controls, dtype=np.float64),
        moment_matrix=np.asarray(moment, dtype=np.float64),
        rank=np.linalg.matrix_rank(moment),
    )


def _plan(state: TransmissionState, jacobian, desired, **kwargs):
    return plan_controllable_task_command(
        current_state=state,
        task_jacobian=np.asarray(jacobian, dtype=np.float64),
        desired_task_delta=np.asarray(desired, dtype=np.float64),
        joint_limits=np.array([[-1.0, 1.0]] * 3),
        actuator_ctrlrange=np.array([[-1.0, 1.0]] * 2),
        active_fingers=np.array([True, False]),
        max_joint_step=0.25,
        max_task_residual=1e-6,
        **kwargs,
    )


def test_task_with_a_controllable_equivalent_passes() -> None:
    # The task only asks for j0 displacement; j2 is a global transmission
    # null-space direction but must not cause rejection when it is irrelevant.
    plan = _plan(_state(), [[1.0, 0.0, 0.0]], [0.1])
    assert plan.rejection_reason == "converged"
    np.testing.assert_allclose(plan.q_preload, [0.1, 0.0, 0.0], atol=1e-8)
    assert plan.task_residual < 1e-6


def test_task_that_exists_only_in_transmission_nullspace_fails() -> None:
    plan = _plan(_state(), [[0.0, 0.0, 1.0]], [0.1])
    assert plan.rejection_reason == "task_uncontrollable"
    assert plan.task_residual == 0.1


def test_saturation_fails_before_a_clipped_command_can_pass() -> None:
    plan = _plan(
        _state(controls=(0.95, 0.0)),
        [[1.0, 0.0, 0.0]],
        [0.1],
    )
    assert plan.rejection_reason == "actuator_saturation"
    assert plan.saturated.tolist() == [True, False]
    assert plan.control_target[0] == 1.0


def test_broken_moment_mapping_flips_the_task_verdict() -> None:
    healthy = _plan(_state(), [[1.0, 0.0, 0.0]], [0.1])
    broken = _plan(
        _state(moment=np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])),
        [[1.0, 0.0, 0.0]],
        [0.1],
    )
    assert healthy.rejection_reason == "converged"
    assert broken.rejection_reason == "task_uncontrollable"
