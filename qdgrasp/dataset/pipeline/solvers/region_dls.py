from typing import Optional
import numpy as np
import torch
from torch.func import jacrev

from qdgrasp.robot.spec import RobotSpec
from qdgrasp.dataset.pipeline import contact_state as contact_state_module
from qdgrasp.dataset.pipeline.contracts import KinematicSolution
from qdgrasp.dataset.pipeline.solvers.normal_equations import masked_normal_equations

def solve_region_dls_ik_batch(
    spec: RobotSpec,
    palm_pos: np.ndarray,
    palm_rot: np.ndarray,
    target_contacts: np.ndarray,
    target_normals: np.ndarray,
    region_points: Optional[np.ndarray] = None,
    region_normals: Optional[np.ndarray] = None,
    init_q: Optional[np.ndarray] = None,
    active_fingers: Optional[np.ndarray | torch.Tensor] = None,
    damping: float = 0.01,
    step_size: float = 0.5,
    max_iter: int = 50,
    pos_tolerance: float = 0.005,
    normal_tolerance_dot: float = 0.866,
    normal_weight: float = 0.01,
    require_normal_alignment: bool = True,
    regularization_weight: float = 1e-5,
    joint_margin_weight: float = 1e-4,
    min_active_fingers: int = 2,
) -> KinematicSolution:
    """
    Batched DLS IK against a finite set of exact mesh-surface samples per
    finger. Supports active_fingers boolean mask [B, K] or [K].
    """
    B = palm_pos.shape[0]
    num_joints = len(spec.actuated_joint_names)
    num_tips = len(spec.fingertip_links)

    device = torch.device("cpu")

    t_palm_pos = torch.as_tensor(palm_pos, dtype=torch.float32, device=device)
    t_palm_rot = torch.as_tensor(palm_rot, dtype=torch.float32, device=device)
    t_target_anchors = torch.as_tensor(target_contacts, dtype=torch.float32, device=device)
    t_target_normal = torch.as_tensor(target_normals, dtype=torch.float32, device=device)
    if region_points is None:
        t_region_points = t_target_anchors.unsqueeze(2)
        t_region_normals = t_target_normal.unsqueeze(2)
    else:
        t_region_points = torch.as_tensor(region_points, dtype=torch.float32, device=device)
        if region_normals is None:
            raise ValueError("region_normals is required with region_points")
        t_region_normals = torch.as_tensor(region_normals, dtype=torch.float32, device=device)
        if t_region_points.ndim != 4 or t_region_points.shape[:2] != (B, num_tips):
            raise ValueError("region_points must have shape [B, K, R, 3]")

    if active_fingers is not None:
        t_active_fingers = torch.as_tensor(active_fingers, dtype=torch.bool, device=device)
        if t_active_fingers.ndim == 1:
            t_active_fingers = t_active_fingers.unsqueeze(0).expand(B, num_tips)
        elif t_active_fingers.ndim != 2 or t_active_fingers.shape != (B, num_tips):
            raise ValueError(f"active_fingers must have shape ({B}, {num_tips}), got {t_active_fingers.shape}")
    else:
        t_active_fingers = torch.ones((B, num_tips), dtype=torch.bool, device=device)

    active_pos_mask = t_active_fingers.unsqueeze(-1).expand(-1, -1, 3).reshape(B, -1)
    active_norm_mask = t_active_fingers.unsqueeze(-1).expand(-1, -1, 3).reshape(B, -1)
    active_flat_mask = torch.cat([active_pos_mask, active_norm_mask], dim=1) # [B, 6K]

    q_mins = torch.tensor([spec.joint_limits[j][0] for j in spec.actuated_joint_names], dtype=torch.float32, device=device)
    q_maxs = torch.tensor([spec.joint_limits[j][1] for j in spec.actuated_joint_names], dtype=torch.float32, device=device)

    if init_q is not None:
        q = torch.as_tensor(init_q, dtype=torch.float32, device=device).clone()
    else:
        q = ((q_mins + q_maxs) * 0.5).unsqueeze(0).expand(B, num_joints).clone()
    q_reference = q.clone()

    def contact_position(transforms, tip):
        return contact_state_module.contact_position(spec, transforms, tip)

    def contact_direction(transforms, tip):
        return contact_state_module.contact_direction(
            spec, transforms, tip, fallback_origin=t_palm_pos
        )

    def compute_single(q_single, palm_pos_single, palm_rot_single):
        return contact_state_module.contact_residual_features(
            spec,
            q_single,
            palm_pos_single,
            palm_rot_single,
            normal_weight=normal_weight,
        )

    jacobian_single = jacrev(compute_single, argnums=0)

    converged = torch.zeros(B, dtype=torch.bool, device=device)
    reasons = np.array(["max_iter"] * B, dtype=object)
    iterations = torch.zeros(B, dtype=torch.int64, device=device)

    # Check minimum active fingers
    insufficient_fingers = (t_active_fingers.sum(dim=-1) < min_active_fingers)
    for idx in torch.where(insufficient_fingers)[0]:
        reasons[idx.item()] = "insufficient_active_fingers"

    achieved_contacts = torch.zeros_like(t_target_anchors)
    achieved_normals = torch.zeros_like(t_target_normal)
    pos_residuals = torch.full((B, num_tips), float('inf'), device=device)
    norm_residuals = torch.full((B, num_tips), float("inf"), device=device)

    I = torch.eye(num_joints, device=device).unsqueeze(0).expand(B, num_joints, num_joints)
    damping_values = torch.full((B,), float(damping), dtype=torch.float32, device=device)

    for it in range(max_iter):
        active_mask = ~converged & ~insufficient_fingers
        if not active_mask.any():
            break
        iterations[active_mask] += 1

        with torch.no_grad():
            transforms = spec.forward_kinematics(t_palm_pos, t_palm_rot, q)
            pos_list = []
            norm_list = []
            for tip in spec.fingertip_links:
                pos_list.append(contact_position(transforms, tip))
                norm_list.append(contact_direction(transforms, tip))

            p = torch.stack(pos_list, dim=1)
            n = torch.stack(norm_list, dim=1)

            achieved_contacts = p
            achieved_normals = n

            distances_sq = torch.sum(
                (p.unsqueeze(2) - t_region_points) ** 2, dim=-1
            )
            nearest_idx = torch.argmin(distances_sq, dim=2)
            gather_idx = nearest_idx[..., None, None].expand(-1, -1, 1, 3)
            p_target = torch.gather(t_region_points, 2, gather_idx).squeeze(2)
            n_target = torch.gather(t_region_normals, 2, gather_idx).squeeze(2)

            p_errs = torch.norm(p_target - p, dim=-1)
            pos_residuals = p_errs

            n_dots = torch.sum(n_target * n, dim=-1).clamp(-1.0, 1.0)
            norm_residuals = torch.acos(n_dots)

            p_converged = torch.all((p_errs < pos_tolerance) | ~t_active_fingers, dim=1)
            n_converged = (
                torch.all((n_dots > normal_tolerance_dot) | ~t_active_fingers, dim=1)
                if require_normal_alignment
                else torch.ones(B, dtype=torch.bool, device=device)
            )
            new_converged = p_converged & n_converged & active_mask

            if new_converged.any():
                converged[new_converged] = True
                for i in torch.where(new_converged)[0]:
                    reasons[i.item()] = "converged"

            active_mask = ~converged & ~insufficient_fingers
            if not active_mask.any():
                break

            cur_flat = torch.cat([p.view(B, -1), (n * normal_weight).view(B, -1)], dim=1)
            target_flat = torch.cat([p_target.view(B, -1), (n_target * normal_weight).view(B, -1)], dim=1)
            err = ((target_flat - cur_flat) * active_flat_mask).unsqueeze(-1)

        J_batch = torch.stack(
            [jacobian_single(q[idx], t_palm_pos[idx], t_palm_rot[idx]) for idx in range(B)],
            dim=0,
        )

        with torch.no_grad():
            damping_matrix = (
                damping_values.square() + regularization_weight
            )[:, None, None] * I
            # RC-02: inactive fingers must contribute no curvature either, so
            # the mask is applied to the Jacobian rows, not only to `err`.
            H, g = masked_normal_equations(J_batch, err, active_flat_mask, damping_matrix)
            g += regularization_weight * (q_reference - q)

            joint_span = torch.clamp(q_maxs - q_mins, min=1e-6)
            margin = 0.05 * joint_span
            low_repulsion = torch.clamp(margin - (q - q_mins), min=0.0) / margin
            high_repulsion = torch.clamp(margin - (q_maxs - q), min=0.0) / margin
            g += joint_margin_weight * (low_repulsion - high_repulsion)

            try:
                dq = torch.linalg.solve(H, g.unsqueeze(-1)).squeeze(-1)
            except RuntimeError:
                dq = torch.bmm(torch.linalg.pinv(H), g.unsqueeze(-1)).squeeze(-1)

            q_trial = torch.clamp(q + step_size * dq, min=q_mins, max=q_maxs)
            trial_transforms = spec.forward_kinematics(t_palm_pos, t_palm_rot, q_trial)
            trial_pos = torch.stack(
                [contact_position(trial_transforms, tip) for tip in spec.fingertip_links], dim=1
            )
            trial_normals = torch.stack(
                [contact_direction(trial_transforms, tip) for tip in spec.fingertip_links], dim=1
            )
            trial_distances_sq = torch.sum(
                (trial_pos.unsqueeze(2) - t_region_points) ** 2, dim=-1
            )
            trial_nearest = torch.argmin(trial_distances_sq, dim=2)
            trial_gather = trial_nearest[..., None, None].expand(-1, -1, 1, 3)
            trial_targets = torch.gather(t_region_points, 2, trial_gather).squeeze(2)
            trial_target_normals = torch.gather(
                t_region_normals, 2, trial_gather
            ).squeeze(2)
            current_cost = torch.sum(err.squeeze(-1).square(), dim=1)
            trial_error = torch.cat(
                [
                    (trial_targets - trial_pos).reshape(B, -1),
                    ((trial_target_normals - trial_normals) * normal_weight).reshape(B, -1),
                ],
                dim=1,
            ) * active_flat_mask
            trial_cost = torch.sum(trial_error.square(), dim=1)
            improved = active_mask & torch.isfinite(trial_cost) & (trial_cost <= current_cost)
            q = torch.where(improved.unsqueeze(-1), q_trial, q)
            damping_values = torch.where(
                improved,
                torch.clamp(damping_values * 0.7, min=1e-5),
                torch.clamp(damping_values * 2.0, max=1.0),
            )

    with torch.no_grad():
        transforms = spec.forward_kinematics(t_palm_pos, t_palm_rot, q)
        achieved_contacts = torch.stack(
            [contact_position(transforms, tip) for tip in spec.fingertip_links], dim=1
        )
        achieved_normals = torch.stack(
            [
                contact_direction(transforms, tip)
                for tip in spec.fingertip_links
            ],
            dim=1,
        )
        distances_sq = torch.sum(
            (achieved_contacts.unsqueeze(2) - t_region_points) ** 2, dim=-1
        )
        nearest_idx = torch.argmin(distances_sq, dim=2)
        gather_idx = nearest_idx[..., None, None].expand(-1, -1, 1, 3)
        final_targets = torch.gather(t_region_points, 2, gather_idx).squeeze(2)
        final_normals = torch.gather(t_region_normals, 2, gather_idx).squeeze(2)
        pos_residuals = torch.linalg.norm(final_targets - achieved_contacts, dim=-1)
        dots = torch.sum(final_normals * achieved_normals, dim=-1).clamp(-1.0, 1.0)
        norm_residuals = torch.acos(dots)
        final_normal_ok = (
            torch.all((dots > normal_tolerance_dot) | ~t_active_fingers, dim=1)
            if require_normal_alignment
            else torch.ones(B, dtype=torch.bool, device=device)
        )
        final_converged = (
            torch.all((pos_residuals < pos_tolerance) | ~t_active_fingers, dim=1)
            & final_normal_ok
            & ~insufficient_fingers
        )
        for idx in torch.where(final_converged & ~converged)[0]:
            reasons[idx.item()] = "converged"
        converged |= final_converged

    return KinematicSolution(
        q=q.cpu().numpy(),
        palm_pos=t_palm_pos.cpu().numpy(),
        palm_rot=t_palm_rot.cpu().numpy(),
        achieved_contacts=achieved_contacts.cpu().numpy(),
        achieved_normals=achieved_normals.cpu().numpy(),
        position_residuals=pos_residuals.cpu().numpy(),
        normal_residuals=norm_residuals.cpu().numpy(),
        converged=converged.cpu().numpy(),
        reason=reasons,
        iterations=iterations.cpu().numpy(),
    )
