"""Controllable-space projection and actuator command solver."""

from __future__ import annotations

from typing import Sequence, Tuple
import numpy as np

from .contracts import ActuatorCommand, TransmissionState


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
