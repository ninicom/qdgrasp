"""Controllable-space projection and actuator command solver."""

from __future__ import annotations

from typing import Sequence, Tuple
import numpy as np

from .contracts import ActuatorCommand, GraspCommandPlan, TransmissionState


def plan_controllable_task_command(
    *,
    current_state: TransmissionState,
    task_jacobian: np.ndarray,
    desired_task_delta: np.ndarray,
    joint_limits: np.ndarray,
    actuator_ctrlrange: np.ndarray,
    active_fingers: np.ndarray,
    q_contact: np.ndarray | None = None,
    max_joint_step: float = 0.25,
    max_task_residual: float = 1e-3,
    alpha: float = 1e-8,
    beta: float = 1e-8,
    saturation_tolerance: float = 1e-8,
) -> GraspCommandPlan:
    """Solve an active contact displacement in ``range(M.T)``.

    The optimization variable is actuator-space ``z`` and ``dq = M.T z`` by
    construction.  Joint and step constraints are enforced by one global scale,
    preserving controllability.  A command that would exceed a control range is
    reported as saturation and is never silently accepted after clipping.
    """
    q_start = np.asarray(current_state.joint_position, dtype=np.float64)
    controls = np.asarray(current_state.actuator_coordinate, dtype=np.float64)
    moment = np.asarray(current_state.moment_matrix, dtype=np.float64)
    jacobian = np.asarray(task_jacobian, dtype=np.float64)
    desired = np.asarray(desired_task_delta, dtype=np.float64).reshape(-1)
    limits = np.asarray(joint_limits, dtype=np.float64)
    control_range = np.asarray(actuator_ctrlrange, dtype=np.float64)
    contact = q_start.copy() if q_contact is None else np.asarray(q_contact, dtype=np.float64)

    valid_shapes = (
        q_start.ndim == 1
        and controls.ndim == 1
        and moment.shape == (len(controls), len(q_start))
        and jacobian.ndim == 2
        and jacobian.shape == (len(desired), len(q_start))
        and limits.shape == (len(q_start), 2)
        and control_range.shape == (len(controls), 2)
        and contact.shape == q_start.shape
    )
    finite_state_and_task = all(
        np.all(np.isfinite(value))
        for value in (q_start, controls, moment, jacobian, desired, contact)
    )
    valid_bounds = (
        not np.any(np.isnan(limits))
        and not np.any(np.isnan(control_range))
        and np.all(limits[:, 0] <= limits[:, 1])
        and np.all(control_range[:, 0] <= control_range[:, 1])
    )
    state_in_range = (
        valid_shapes
        and np.all(q_start >= limits[:, 0])
        and np.all(q_start <= limits[:, 1])
        and np.all(controls >= control_range[:, 0])
        and np.all(controls <= control_range[:, 1])
    )

    zero_saturation = np.zeros(len(controls), dtype=bool)
    if not valid_shapes or not finite_state_and_task or not valid_bounds or not state_in_range:
        return GraspCommandPlan(
            q_pregrasp=q_start.copy(),
            q_contact=contact.copy(),
            q_preload=q_start.copy(),
            active_fingers=np.asarray(active_fingers, dtype=bool).copy(),
            control_start=controls.copy(),
            control_target=controls.copy(),
            task_residual=float("inf"),
            nullspace_residual=float("inf"),
            saturated=zero_saturation,
            rejection_reason="invalid_state",
        )

    controllable_basis = moment.T
    task_map = jacobian @ controllable_basis
    gram = moment @ moment.T
    hessian = (
        task_map.T @ task_map
        + alpha * (controllable_basis.T @ controllable_basis)
        + beta * (gram.T @ gram)
    )
    gradient = task_map.T @ desired
    try:
        actuator_coordinate_step = np.linalg.solve(hessian, gradient)
    except np.linalg.LinAlgError:
        actuator_coordinate_step = np.linalg.pinv(hessian) @ gradient
    dq = controllable_basis @ actuator_coordinate_step

    scale = 1.0
    largest_step = float(np.max(np.abs(dq), initial=0.0))
    if largest_step > max_joint_step > 0.0:
        scale = min(scale, max_joint_step / largest_step)
    for index, delta in enumerate(dq):
        if delta > 0.0:
            scale = min(scale, (limits[index, 1] - q_start[index]) / delta)
        elif delta < 0.0:
            scale = min(scale, (limits[index, 0] - q_start[index]) / delta)
    scale = float(np.clip(scale, 0.0, 1.0))
    dq *= scale

    q_preload = q_start + dq
    control_unclipped = controls + moment @ dq
    saturated = (control_unclipped < control_range[:, 0] - saturation_tolerance) | (
        control_unclipped > control_range[:, 1] + saturation_tolerance
    )
    control_target = np.clip(
        control_unclipped, control_range[:, 0], control_range[:, 1]
    )
    task_residual = float(np.linalg.norm(jacobian @ dq - desired))

    unconstrained_joint_step = np.linalg.pinv(jacobian) @ desired
    projector = np.linalg.pinv(moment) @ moment
    nullspace_residual = float(
        np.linalg.norm((np.eye(len(q_start)) - projector) @ unconstrained_joint_step)
    )

    if np.any(saturated):
        reason = "actuator_saturation"
    elif task_residual > max_task_residual:
        reason = "task_uncontrollable"
    else:
        reason = "converged"

    return GraspCommandPlan(
        q_pregrasp=q_start.copy(),
        q_contact=contact.copy(),
        q_preload=q_preload,
        active_fingers=np.asarray(active_fingers, dtype=bool).copy(),
        control_start=controls.copy(),
        control_target=control_target,
        task_residual=task_residual,
        nullspace_residual=nullspace_residual,
        saturated=saturated,
        rejection_reason=reason,
    )


def project_joint_delta_to_actuator_command(
    joint_delta: np.ndarray,
    moment_matrix: np.ndarray,
    actuator_ctrlrange: np.ndarray,
    current_actuator_coordinates: np.ndarray | None = None,
    *,
    damping: float = 1e-6,
    max_nullspace_residual: float = 0.05,
    saturation_tolerance: float = 1e-4,
) -> ActuatorCommand:
    """
    Project desired joint delta dq* [B, J] or [J] into actuator coordinates dl [B, U] or [U].
    Computes controllable delta dq_control = M+ dl, nullspace delta dq_null = dq* - dq_control,
    and applies actuator control range clipping with saturation diagnostics.
    """
    dq = np.asarray(joint_delta, dtype=np.float64)
    is_1d = (dq.ndim == 1)
    if is_1d:
        dq = dq[None, :]  # [1, J]

    B, J = dq.shape
    M = np.asarray(moment_matrix, dtype=np.float64)
    if M.ndim == 2:
        M = np.broadcast_to(M[None, :, :], (B, M.shape[0], M.shape[1]))  # [B, U, J]
    elif M.ndim == 3 and M.shape[0] != B:
        if M.shape[0] == 1:
            M = np.broadcast_to(M, (B, M.shape[1], M.shape[2]))
        else:
            raise ValueError(f"moment_matrix batch size {M.shape[0]} != joint_delta batch size {B}")

    U = M.shape[1]
    ctrl_range = np.asarray(actuator_ctrlrange, dtype=np.float64)  # [U, 2]

    # dl = M @ dq*  [B, U]
    dl = np.einsum("buj,bj->bu", M, dq)

    # Compute Damped Least Squares Pseudoinverse M^+ = M^T (M M^T + lambda^2 I)^-1
    # For each b in B:
    dq_control = np.zeros((B, J), dtype=np.float64)
    dq_null = np.zeros((B, J), dtype=np.float64)
    controllable_residuals = np.zeros(B, dtype=np.float64)
    nullspace_residuals = np.zeros(B, dtype=np.float64)

    I_U = np.eye(U, dtype=np.float64)
    for b in range(B):
        Mb = M[b]  # [U, J]
        MMt = Mb @ Mb.T + (damping**2) * I_U  # [U, U]
        try:
            pinv_Mb = Mb.T @ np.linalg.inv(MMt)  # [J, U]
        except np.linalg.LinAlgError:
            pinv_Mb = np.linalg.pinv(Mb)  # [J, U]

        dq_ctrl_b = pinv_Mb @ dl[b]  # [J]
        dq_null_b = dq[b] - dq_ctrl_b  # [J]

        dq_control[b] = dq_ctrl_b
        dq_null[b] = dq_null_b

        # Residuals
        dl_recon = Mb @ dq_ctrl_b
        controllable_residuals[b] = float(np.linalg.norm(dl[b] - dl_recon))
        nullspace_residuals[b] = float(np.linalg.norm(dq_null_b))

    # Current coordinates
    if current_actuator_coordinates is not None:
        curr_u = np.asarray(current_actuator_coordinates, dtype=np.float64)
        if curr_u.ndim == 1:
            curr_u = curr_u[None, :]
        if curr_u.shape[0] == 1 and B > 1:
            curr_u = np.broadcast_to(curr_u, (B, U))
        u_target = curr_u + dl
    else:
        u_target = dl

    # Saturation & clipping
    u_min = ctrl_range[:, 0]
    u_max = ctrl_range[:, 1]
    u_clipped = np.clip(u_target, u_min[None, :], u_max[None, :])
    saturated = (u_target < (u_min[None, :] - saturation_tolerance)) | (
        u_target > (u_max[None, :] + saturation_tolerance)
    )

    reasons = np.array(["converged"] * B, dtype=object)
    for b in range(B):
        if nullspace_residuals[b] > max_nullspace_residual:
            reasons[b] = "nullspace_rejection"
        elif np.any(saturated[b]):
            reasons[b] = "actuator_saturation"

    if is_1d:
        return ActuatorCommand(
            control_target=u_clipped[0],
            projected_joint_delta=dq_control[0],
            controllable_residual=controllable_residuals[0],
            nullspace_residual=nullspace_residuals[0],
            saturated=saturated[0],
            reason=str(reasons[0]),
        )

    return ActuatorCommand(
        control_target=u_clipped,
        projected_joint_delta=dq_control,
        controllable_residual=controllable_residuals,
        nullspace_residual=nullspace_residuals,
        saturated=saturated,
        reason=reasons,
    )
