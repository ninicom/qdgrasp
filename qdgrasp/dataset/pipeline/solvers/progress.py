"""Shared progress telemetry and failure classification for DLS solvers.

P3.2.1-04 deliberately keeps classification separate from convergence.  A
candidate is successful only through the geometric tolerances in the solver;
these helpers explain a failure without weakening those tolerances or changing
the candidate/iteration budget.
"""

from __future__ import annotations

import numpy as np
import torch


def masked_jacobian_spectrum(
    jacobian: torch.Tensor,
    active_flat_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return numerical rank and non-zero-spectrum condition number per batch."""
    masked = jacobian * active_flat_mask.to(jacobian.dtype).unsqueeze(-1)
    singular_values = torch.linalg.svdvals(masked)
    if singular_values.shape[-1] == 0:
        batch = jacobian.shape[0]
        return (
            torch.zeros(batch, dtype=torch.int64, device=jacobian.device),
            torch.full((batch,), float("inf"), dtype=jacobian.dtype, device=jacobian.device),
        )

    largest = singular_values[:, 0]
    tolerance = (
        torch.finfo(singular_values.dtype).eps
        * max(masked.shape[-2], masked.shape[-1])
        * largest
    )
    nonzero = singular_values > tolerance.unsqueeze(-1)
    rank = nonzero.sum(dim=-1)
    smallest_nonzero = torch.where(
        nonzero,
        singular_values,
        torch.full_like(singular_values, float("inf")),
    ).amin(dim=-1)
    condition = torch.where(
        rank > 0,
        largest / torch.clamp(smallest_nonzero, min=torch.finfo(singular_values.dtype).tiny),
        torch.full_like(largest, float("inf")),
    )
    return rank, condition


def classify_failure_reasons(
    *,
    converged: np.ndarray,
    insufficient_fingers: np.ndarray,
    iterations: np.ndarray,
    accepted_steps: np.ndarray,
    rejected_steps: np.ndarray,
    initial_cost: np.ndarray,
    final_cost: np.ndarray,
    raw_step_norm: np.ndarray,
    projected_step_norm: np.ndarray,
    limit_clipped_steps: np.ndarray,
    jacobian_rank: np.ndarray,
    finite: np.ndarray,
    max_iter: int,
    meaningful_relative_reduction: float = 1e-6,
    step_tolerance: float = 1e-9,
) -> np.ndarray:
    """Classify non-converged candidates from measured progress signals.

    Precedence is intentional: invalid/singular systems and hard joint-limit
    blocking are more specific than a failed line search; stagnation requires at
    least one accepted step; ``max_iter`` is reserved for candidates that were
    still making measurable progress when their fixed budget expired.
    """
    converged = np.asarray(converged, dtype=bool)
    insufficient = np.asarray(insufficient_fingers, dtype=bool)
    reasons = np.full(converged.shape, "max_iter", dtype=object)
    reasons[converged] = "converged"
    reasons[insufficient] = "insufficient_active_fingers"

    for index in range(len(reasons)):
        if converged[index] or insufficient[index]:
            continue
        if not bool(finite[index]) or int(jacobian_rank[index]) == 0:
            reasons[index] = "singular"
            continue

        raw_norm = float(raw_step_norm[index])
        projected_norm = float(projected_step_norm[index])
        clipped = int(limit_clipped_steps[index]) > 0
        blocked_by_limits = (
            clipped
            and raw_norm > step_tolerance
            and projected_norm <= max(step_tolerance, raw_norm * 1e-4)
        )
        if blocked_by_limits:
            reasons[index] = "joint_limit"
            continue

        if int(accepted_steps[index]) == 0 and int(rejected_steps[index]) > 0:
            reasons[index] = "line_search_failed"
            continue

        initial = float(initial_cost[index])
        final = float(final_cost[index])
        relative_reduction = (
            (initial - final) / max(abs(initial), 1e-12)
            if np.isfinite(initial) and np.isfinite(final)
            else -np.inf
        )
        if (
            int(accepted_steps[index]) > 0
            and (
                relative_reduction <= meaningful_relative_reduction
                or projected_norm <= step_tolerance
            )
        ):
            reasons[index] = "stagnation"
            continue

        # Keep max_iter only for a finite candidate that consumed its budget
        # while retaining measurable descent.  A zero-iteration request is also
        # a budget exhaustion, not a line-search observation.
        if int(iterations[index]) >= max(0, int(max_iter)):
            reasons[index] = "max_iter"
        else:
            reasons[index] = "stagnation"
    return reasons


def solver_metrics_to_numpy(**values: torch.Tensor) -> dict[str, np.ndarray]:
    """Detach a named collection of per-candidate tensors for the contract."""
    return {
        name: tensor.detach().cpu().numpy()
        for name, tensor in values.items()
    }


def meaningful_cost_decrease(
    current_cost: torch.Tensor,
    trial_cost: torch.Tensor,
) -> torch.Tensor:
    """True only when a trial lowers cost beyond floating-point roundoff."""
    scale = torch.clamp(torch.abs(current_cost), min=1.0)
    minimum_decrease = 8.0 * torch.finfo(current_cost.dtype).eps * scale
    return torch.isfinite(trial_cost) & (trial_cost < current_cost - minimum_decrease)
