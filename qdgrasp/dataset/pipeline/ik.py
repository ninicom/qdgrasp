"""Damped Least Squares Inverse Kinematics (DLS-IK) for multifingered hands."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import torch

from ...robot.spec import RobotSpec


@dataclass(frozen=True)
class DlsIkResult:
    """Outcome of a Damped Least Squares IK optimization."""

    q: np.ndarray  # [num_actuated_joints]
    converged: bool
    final_error: float
    iterations: int
    fingertip_positions: np.ndarray  # [num_fingertips, 3]


def solve_dls_ik(
    spec: RobotSpec,
    palm_pos: np.ndarray | torch.Tensor,
    palm_rot: np.ndarray | torch.Tensor,
    target_contacts: np.ndarray | torch.Tensor,
    *,
    init_q: np.ndarray | torch.Tensor | None = None,
    damping: float = 0.01,
    step_size: float = 0.5,
    max_iter: int = 50,
    tolerance: float = 0.005,  # 5mm convergence threshold
) -> DlsIkResult:
    """Solve for actuated joint angles q that reach target fingertip positions.

    Uses Damped Least Squares (Levenberg-Marquardt) optimization with projection
    onto the declared joint limits of the robot profile.
    """
    t_palm_pos = torch.as_tensor(palm_pos, dtype=torch.float32).view(1, 3)
    t_palm_rot = torch.as_tensor(palm_rot, dtype=torch.float32).view(1, 3, 3)
    t_targets = torch.as_tensor(target_contacts, dtype=torch.float32).view(1, -1, 3)

    num_joints = len(spec.actuated_joint_names)
    num_tips = len(spec.fingertip_links)

    # Joint limits [J, 2]
    q_mins = torch.tensor([spec.joint_limits[j][0] for j in spec.actuated_joint_names], dtype=torch.float32)
    q_maxs = torch.tensor([spec.joint_limits[j][1] for j in spec.actuated_joint_names], dtype=torch.float32)

    # Initial joint configuration
    if init_q is not None:
        q = torch.as_tensor(init_q, dtype=torch.float32).view(1, num_joints)
    else:
        q = ((q_mins + q_maxs) * 0.5).view(1, num_joints)

    delta = 1e-4
    converged = False
    final_error = float("inf")
    iterations_run = 0

    target_flat = t_targets.view(-1)  # [3 * K]

    for it in range(max_iter):
        iterations_run = it + 1
        with torch.no_grad():
            cur_tips = spec.fingertip_positions(t_palm_pos, t_palm_rot, q)  # [1, K, 3]
            cur_flat = cur_tips.view(-1)  # [3 * K]
            err = target_flat - cur_flat
            err_norm = float(torch.norm(err).item())
            final_error = err_norm

            if err_norm < tolerance:
                converged = True
                break

            # Compute Jacobian J [3K, J]
            J = torch.zeros((3 * num_tips, num_joints), dtype=torch.float32)
            for j_idx in range(num_joints):
                q_pert = q.clone()
                q_pert[0, j_idx] += delta
                tips_pert = spec.fingertip_positions(t_palm_pos, t_palm_rot, q_pert).view(-1)
                J[:, j_idx] = (tips_pert - cur_flat) / delta

            # DLS step: dq = (J^T J + lambda^2 I)^-1 J^T e
            Jt = J.t()
            H = torch.mm(Jt, J) + (damping**2) * torch.eye(num_joints)
            g = torch.mv(Jt, err)

            try:
                dq = torch.linalg.solve(H, g.unsqueeze(-1)).squeeze(-1)
            except Exception:
                dq = torch.zeros(num_joints)

            q = q + step_size * dq.unsqueeze(0)
            # Project onto joint limits
            q = torch.clamp(q, min=q_mins.unsqueeze(0), max=q_maxs.unsqueeze(0))

    final_tips = spec.fingertip_positions(t_palm_pos, t_palm_rot, q)[0].numpy()
    final_q = q[0].numpy().astype(np.float64)

    return DlsIkResult(
        q=final_q,
        converged=converged,
        final_error=final_error,
        iterations=iterations_run,
        fingertip_positions=final_tips.astype(np.float64),
    )
