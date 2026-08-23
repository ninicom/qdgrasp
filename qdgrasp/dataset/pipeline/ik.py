"""Compatibility facade for the canonical batched DLS-IK solver."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from ...robot.spec import RobotSpec
from .solvers.fixed_contact_dls import solve_dls_ik_batch


@dataclass(frozen=True)
class DlsIkResult:
    """Single-candidate view of a canonical batched IK result."""

    q: np.ndarray
    converged: bool
    final_error: float
    iterations: int
    fingertip_positions: np.ndarray


def solve_dls_ik(
    spec: RobotSpec,
    palm_pos: np.ndarray | torch.Tensor,
    palm_rot: np.ndarray | torch.Tensor,
    target_contacts: np.ndarray | torch.Tensor,
    *,
    target_normals: np.ndarray | torch.Tensor | None = None,
    init_q: np.ndarray | torch.Tensor | None = None,
    damping: float = 0.01,
    step_size: float = 0.5,
    max_iter: int = 50,
    tolerance: float = 0.005,
) -> DlsIkResult:
    """Dispatch one candidate through the canonical autodiff DLS solver.

    Legacy callers did not provide surface normals. That path remains
    position-only for API compatibility; artifact generation and correctness
    gates must pass face-derived ``target_normals`` explicitly.
    """
    palm_pos_np = np.asarray(palm_pos, dtype=np.float32).reshape(1, 3)
    palm_rot_np = np.asarray(palm_rot, dtype=np.float32).reshape(1, 3, 3)
    contacts_np = np.asarray(target_contacts, dtype=np.float32).reshape(
        1, len(spec.fingertip_links), 3
    )
    if target_normals is None:
        inferred = contacts_np - palm_pos_np[:, None, :]
        inferred /= np.clip(
            np.linalg.norm(inferred, axis=-1, keepdims=True), 1e-8, None
        )
        normals_np = inferred
        require_normal_alignment = False
    else:
        normals_np = np.asarray(target_normals, dtype=np.float32).reshape(
            1, len(spec.fingertip_links), 3
        )
        require_normal_alignment = True

    init_q_np = (
        None
        if init_q is None
        else np.asarray(init_q, dtype=np.float32).reshape(
            1, len(spec.actuated_joint_names)
        )
    )
    solution = solve_dls_ik_batch(
        spec,
        palm_pos_np,
        palm_rot_np,
        contacts_np,
        normals_np,
        init_q=init_q_np,
        damping=damping,
        step_size=step_size,
        max_iter=max_iter,
        pos_tolerance=tolerance,
        require_normal_alignment=require_normal_alignment,
    )
    iteration_count = (
        max_iter if solution.iterations is None else int(solution.iterations[0])
    )
    return DlsIkResult(
        q=solution.q[0].astype(np.float64),
        converged=bool(solution.converged[0]),
        final_error=float(np.max(solution.position_residuals[0])),
        iterations=iteration_count,
        fingertip_positions=solution.achieved_contacts[0].astype(np.float64),
    )
