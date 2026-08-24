"""Bounded joint+palm SE(3) DLS refinement for P3.2.1-06/P3.2.1-10."""

from __future__ import annotations

import numpy as np
import torch
from torch.func import jacrev

from qdgrasp.dataset.pipeline.contact_state import contact_residual_features, contact_state
from qdgrasp.dataset.pipeline.contracts import KinematicSolution
from qdgrasp.dataset.pipeline.solvers.progress import (
    classify_failure_reasons,
    masked_jacobian_spectrum,
    meaningful_cost_decrease,
    solver_metrics_to_numpy,
)
from qdgrasp.robot.spec import RobotSpec


def _skew(vector: torch.Tensor) -> torch.Tensor:
    zero = torch.zeros((), dtype=vector.dtype, device=vector.device)
    x, y, z = vector.unbind()
    return torch.stack((zero, -z, y, z, zero, -x, -y, x, zero)).reshape(3, 3)


def _pose_from_state(
    state: torch.Tensor, base_pos: torch.Tensor, base_rot: torch.Tensor, joints: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    q = state[:joints]
    translation = state[joints : joints + 3]
    rotvec = state[joints + 3 : joints + 6]
    rotation_delta = torch.matrix_exp(_skew(rotvec))
    return q, base_pos + translation, rotation_delta @ base_rot


def solve_joint_palm_dls_batch(
    spec: RobotSpec,
    palm_pos: np.ndarray,
    palm_rot: np.ndarray,
    target_contacts: np.ndarray,
    target_normals: np.ndarray,
    *,
    init_q: np.ndarray,
    active_fingers: np.ndarray,
    max_iter: int = 40,
    pos_tolerance: float = 0.005,
    normal_tolerance_dot: float = 0.866,
    normal_weight: float = 0.01,
    max_translation: float = 0.01,
    max_rotation: float = np.deg2rad(10.0),
    floor_z: float = 0.0,
    min_palm_floor_clearance: float = 0.005,
    damping: float = 0.01,
    step_size: float = 0.5,
) -> KinematicSolution:
    """Jointly refine q and a trust-region palm twist under one task Jacobian."""
    device = torch.device("cpu")
    base_pos = torch.as_tensor(palm_pos, dtype=torch.float32, device=device)
    base_rot = torch.as_tensor(palm_rot, dtype=torch.float32, device=device)
    target_pos = torch.as_tensor(target_contacts, dtype=torch.float32, device=device)
    target_normal = torch.as_tensor(target_normals, dtype=torch.float32, device=device)
    active = torch.as_tensor(active_fingers, dtype=torch.bool, device=device)
    q0 = torch.as_tensor(init_q, dtype=torch.float32, device=device)
    batch, _ = active.shape
    joints = q0.shape[1]
    state = torch.cat([q0, torch.zeros((batch, 6), dtype=torch.float32, device=device)], dim=1)
    state_reference = state.clone()
    q_min = torch.tensor(
        [spec.joint_limits[name][0] for name in spec.actuated_joint_names],
        dtype=torch.float32,
    )
    q_max = torch.tensor(
        [spec.joint_limits[name][1] for name in spec.actuated_joint_names],
        dtype=torch.float32,
    )
    feature_mask = torch.cat([active.unsqueeze(-1).expand(-1, -1, 3).reshape(batch, -1)] * 2, dim=1)
    target_features = torch.cat(
        [target_pos.reshape(batch, -1), (target_normal * normal_weight).reshape(batch, -1)],
        dim=1,
    )

    def features(single_state, single_base_pos, single_base_rot):
        q, pose_pos, pose_rot = _pose_from_state(single_state, single_base_pos, single_base_rot, joints)
        return contact_residual_features(spec, q, pose_pos, pose_rot, normal_weight=normal_weight)

    derivative = jacrev(features, argnums=0)
    converged = torch.zeros(batch, dtype=torch.bool)
    iterations = torch.zeros(batch, dtype=torch.int64)
    accepted = torch.zeros(batch, dtype=torch.int64)
    rejected = torch.zeros(batch, dtype=torch.int64)
    clipped_steps = torch.zeros(batch, dtype=torch.int64)
    damping_values = torch.full((batch,), damping, dtype=torch.float32)
    initial_cost = torch.full((batch,), float("nan"), dtype=torch.float32)
    final_cost = torch.full((batch,), float("nan"), dtype=torch.float32)
    raw_step_norm = torch.zeros(batch)
    projected_step_norm = torch.zeros(batch)
    gradient_norm = torch.zeros(batch)
    jacobian_rank = torch.zeros(batch, dtype=torch.int64)
    jacobian_condition = torch.full((batch,), float("inf"))
    finite = torch.ones(batch, dtype=torch.bool)

    def project(candidate: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        projected = candidate.clone()
        projected[:, :joints] = torch.clamp(projected[:, :joints], min=q_min, max=q_max)
        translation = projected[:, joints : joints + 3]
        translation_norm = torch.linalg.norm(translation, dim=1, keepdim=True)
        translation_scale = torch.clamp(max_translation / torch.clamp(translation_norm, min=1e-12), max=1.0)
        projected[:, joints : joints + 3] = translation * translation_scale
        minimum_delta_z = floor_z + min_palm_floor_clearance - base_pos[:, 2]
        projected[:, joints + 2] = torch.maximum(projected[:, joints + 2], minimum_delta_z)
        rotation = projected[:, joints + 3 : joints + 6]
        rotation_norm = torch.linalg.norm(rotation, dim=1, keepdim=True)
        rotation_scale = torch.clamp(max_rotation / torch.clamp(rotation_norm, min=1e-12), max=1.0)
        projected[:, joints + 3 : joints + 6] = rotation * rotation_scale
        changed = torch.any(torch.abs(projected - candidate) > 1e-8, dim=1)
        return projected, changed

    for _ in range(max_iter):
        current_features = torch.stack([features(state[i], base_pos[i], base_rot[i]) for i in range(batch)])
        error = (target_features - current_features) * feature_mask
        cost = torch.sum(error.square(), dim=1)
        initial_cost = torch.where(torch.isnan(initial_cost), cost, initial_cost)
        q_now, pos_now, rot_now = zip(
            *[_pose_from_state(state[i], base_pos[i], base_rot[i], joints) for i in range(batch)]
        )
        achieved_pos, achieved_normal = contact_state(
            spec, torch.stack(pos_now), torch.stack(rot_now), torch.stack(q_now)
        )
        position_error = torch.linalg.norm(target_pos - achieved_pos, dim=-1)
        dots = torch.sum(target_normal * achieved_normal, dim=-1).clamp(-1.0, 1.0)
        converged_now = torch.all((position_error < pos_tolerance) | ~active, dim=1) & torch.all(
            (dots > normal_tolerance_dot) | ~active, dim=1
        )
        converged |= converged_now
        working = ~converged
        if not bool(torch.any(working)):
            break
        iterations[working] += 1

        jacobian = torch.stack([derivative(state[i], base_pos[i], base_rot[i]) for i in range(batch)])
        weighted_jacobian = jacobian * feature_mask.unsqueeze(-1)
        hessian = torch.bmm(weighted_jacobian.transpose(1, 2), weighted_jacobian)
        identity = torch.eye(joints + 6).unsqueeze(0).expand(batch, -1, -1)
        hessian += damping_values.square()[:, None, None] * identity
        gradient = torch.bmm(weighted_jacobian.transpose(1, 2), error.unsqueeze(-1)).squeeze(-1)
        gradient += 1e-5 * (state_reference - state)
        rank, condition = masked_jacobian_spectrum(jacobian, feature_mask)
        jacobian_rank = torch.where(working, rank, jacobian_rank)
        jacobian_condition = torch.where(working, condition, jacobian_condition)
        try:
            delta = torch.linalg.solve(hessian, gradient.unsqueeze(-1)).squeeze(-1)
        except RuntimeError:
            delta = torch.bmm(torch.linalg.pinv(hessian), gradient.unsqueeze(-1)).squeeze(-1)
        raw_delta = step_size * delta
        trial, clipped = project(state + raw_delta)
        projected_delta = trial - state
        trial_features = torch.stack([features(trial[i], base_pos[i], base_rot[i]) for i in range(batch)])
        trial_error = (target_features - trial_features) * feature_mask
        trial_cost = torch.sum(trial_error.square(), dim=1)
        improved = working & meaningful_cost_decrease(cost, trial_cost)
        state = torch.where(improved.unsqueeze(-1), trial, state)
        accepted += improved.to(torch.int64)
        rejected += (working & ~improved).to(torch.int64)
        clipped_steps += (working & clipped).to(torch.int64)
        raw_step_norm = torch.where(working, torch.linalg.norm(raw_delta, dim=1), raw_step_norm)
        projected_step_norm = torch.where(working, torch.linalg.norm(projected_delta, dim=1), projected_step_norm)
        gradient_norm = torch.where(working, torch.linalg.norm(gradient, dim=1), gradient_norm)
        finite &= (~working) | (
            torch.isfinite(cost) & torch.isfinite(trial_cost) & torch.all(torch.isfinite(jacobian), dim=(1, 2))
        )
        final_cost = torch.where(improved, trial_cost, cost)
        damping_values = torch.where(
            improved,
            torch.clamp(damping_values * 0.7, min=1e-5),
            torch.clamp(damping_values * 2.0, max=1.0),
        )

    q_final, pos_final, rot_final = zip(
        *[_pose_from_state(state[i], base_pos[i], base_rot[i], joints) for i in range(batch)]
    )
    q_out = torch.stack(q_final)
    pos_out = torch.stack(pos_final)
    rot_out = torch.stack(rot_final)
    achieved_pos, achieved_normal = contact_state(spec, pos_out, rot_out, q_out)
    position_error = torch.linalg.norm(target_pos - achieved_pos, dim=-1)
    dots = torch.sum(target_normal * achieved_normal, dim=-1).clamp(-1.0, 1.0)
    normal_error = torch.acos(dots)
    converged = torch.all((position_error < pos_tolerance) | ~active, dim=1) & torch.all(
        (dots > normal_tolerance_dot) | ~active, dim=1
    )
    final_error = (
        target_features
        - torch.cat([achieved_pos.reshape(batch, -1), (achieved_normal * normal_weight).reshape(batch, -1)], dim=1)
    ) * feature_mask
    final_cost = torch.sum(final_error.square(), dim=1)
    reasons = classify_failure_reasons(
        converged=converged.numpy(),
        insufficient_fingers=np.zeros(batch, dtype=bool),
        iterations=iterations.numpy(),
        accepted_steps=accepted.numpy(),
        rejected_steps=rejected.numpy(),
        initial_cost=initial_cost.numpy(),
        final_cost=final_cost.numpy(),
        raw_step_norm=raw_step_norm.numpy(),
        projected_step_norm=projected_step_norm.numpy(),
        limit_clipped_steps=clipped_steps.numpy(),
        jacobian_rank=jacobian_rank.numpy(),
        finite=finite.numpy(),
        max_iter=max_iter,
    )
    return KinematicSolution(
        q=q_out.detach().numpy(),
        palm_pos=pos_out.detach().numpy(),
        palm_rot=rot_out.detach().numpy(),
        achieved_contacts=achieved_pos.detach().numpy(),
        achieved_normals=achieved_normal.detach().numpy(),
        position_residuals=position_error.detach().numpy(),
        normal_residuals=normal_error.detach().numpy(),
        converged=converged.numpy(),
        reason=reasons,
        iterations=iterations.numpy(),
        solver_metrics=solver_metrics_to_numpy(
            initial_cost=initial_cost,
            final_cost=final_cost,
            accepted_steps=accepted,
            rejected_steps=rejected,
            limit_clipped_steps=clipped_steps,
            raw_step_norm=raw_step_norm,
            projected_step_norm=projected_step_norm,
            gradient_norm=gradient_norm,
            jacobian_rank=jacobian_rank,
            jacobian_condition=jacobian_condition,
            final_damping=damping_values,
            finite=finite,
        ),
    )
