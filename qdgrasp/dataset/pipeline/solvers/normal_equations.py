"""Shared damped-least-squares normal equations for the contact IK solvers.

Kept separate from the solvers so the masking rule can be tested directly.
RC-02 was that inactive fingers' Jacobian rows still entered the Hessian: with
only the error vector masked, an inactive finger changed the curvature and moved
the active solution, while target perturbations left no trace to detect it by.
Assembling the system here makes the invariant checkable in isolation.
"""

from __future__ import annotations

import torch


def masked_normal_equations(
    jacobian: torch.Tensor,
    error: torch.Tensor,
    active_flat_mask: torch.Tensor,
    damping_matrix: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Assemble ``(H, g)`` for the active task rows only.

    Args:
        jacobian: [B, 6K, J] task Jacobian.
        error: [B, 6K, 1] task error, already masked by ``active_flat_mask``.
        active_flat_mask: [B, 6K] 1.0 for active task rows, 0.0 otherwise.
        damping_matrix: [B, J, J] damping and regularization term.

    Returns:
        ``H = J^T W J + damping`` and ``g = J^T W e``, where ``W`` is the
        diagonal mask.  Because the mask is idempotent and ``error`` is already
        masked, masking the Jacobian rows once yields exactly the weighted normal
        equations of the plan's section 3.3.
    """
    if active_flat_mask.dtype != jacobian.dtype:
        active_flat_mask = active_flat_mask.to(jacobian.dtype)
    masked_jacobian = jacobian * active_flat_mask.unsqueeze(-1)  # [B, 6K, J]
    jacobian_t = masked_jacobian.transpose(1, 2)  # [B, J, 6K]
    hessian = torch.bmm(jacobian_t, masked_jacobian) + damping_matrix
    gradient = torch.bmm(jacobian_t, error).squeeze(-1)
    return hessian, gradient
