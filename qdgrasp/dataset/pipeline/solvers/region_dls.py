from typing import Optional
import numpy as np
import torch
from torch.func import jacrev

from qdgrasp.robot.spec import RobotSpec
from qdgrasp.dataset.pipeline.contracts import KinematicSolution

def solve_region_dls_ik_batch(
    spec: RobotSpec,
    palm_pos: np.ndarray,
    palm_rot: np.ndarray,
    target_contacts: np.ndarray,
    target_normals: np.ndarray,
    region_points: Optional[np.ndarray] = None,
    region_normals: Optional[np.ndarray] = None,
    init_q: Optional[np.ndarray] = None,
    damping: float = 0.01,
    step_size: float = 0.5,
    max_iter: int = 50,
    pos_tolerance: float = 0.005,
    normal_tolerance_dot: float = 0.866,
    normal_weight: float = 0.01,
    require_normal_alignment: bool = True,
    regularization_weight: float = 1e-5,
    joint_margin_weight: float = 1e-4,
) -> KinematicSolution:
    """
    Batched DLS IK against a finite set of exact mesh-surface samples per
    finger.  Unlike a tangent disk, every selectable target has a face-derived
    normal and therefore remains on the object surface.
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

    q_mins = torch.tensor([spec.joint_limits[j][0] for j in spec.actuated_joint_names], dtype=torch.float32, device=device)
    q_maxs = torch.tensor([spec.joint_limits[j][1] for j in spec.actuated_joint_names], dtype=torch.float32, device=device)

    if init_q is not None:
        q = torch.as_tensor(init_q, dtype=torch.float32, device=device).clone()
    else:
        q = ((q_mins + q_maxs) * 0.5).unsqueeze(0).expand(B, num_joints).clone()
    q_reference = q.clone()

    def contact_position(transforms, tip):
        transform = transforms[tip]
        offset = getattr(spec, "fingertip_contact_offsets", {}).get(tip)
        if offset is None:
            return transform[:, :3, 3]
        offset_t = torch.as_tensor(offset, dtype=transform.dtype, device=transform.device)
        return transform[:, :3, 3] + torch.matmul(
            transform[:, :3, :3], offset_t.view(3, 1)
        ).squeeze(-1)

    def contact_direction(transforms, tip):
        transform = transforms[tip]
        configured_axis = getattr(spec, "fingertip_contact_axes", {}).get(tip)
        if configured_axis is not None:
            axis_t = torch.as_tensor(
                configured_axis, dtype=transform.dtype, device=transform.device
            )
            return torch.nn.functional.normalize(
                torch.matmul(transform[:, :3, :3], axis_t.view(3, 1)).squeeze(-1),
                dim=-1,
                eps=1e-8,
            )
        tip_pos = transform[:, :3, 3]
        link = getattr(spec, "links", {}).get(tip)
        parent = getattr(link, "parent_link", None)
        origin = transforms[parent][:, :3, 3] if parent in transforms else t_palm_pos
        return torch.nn.functional.normalize(tip_pos - origin, dim=-1, eps=1e-8)

    def compute_single(q_single, palm_pos_single, palm_rot_single):
        transforms = spec.forward_kinematics(
            palm_pos_single.unsqueeze(0),
            palm_rot_single.unsqueeze(0),
            q_single.unsqueeze(0),
        )
        values = []
        for tip in spec.fingertip_links:
            tip_pos = contact_position(transforms, tip)[0]
            link = getattr(spec, "links", {}).get(tip)
            parent = getattr(link, "parent_link", None)
            origin = transforms[parent][0, :3, 3] if parent in transforms else palm_pos_single
            direction = torch.nn.functional.normalize(tip_pos - origin, dim=-1, eps=1e-8)
            values.extend((tip_pos, direction * normal_weight))
        positions = torch.stack(values[0::2], dim=0).reshape(-1)
        directions = torch.stack(values[1::2], dim=0).reshape(-1)
        return torch.cat((positions, directions), dim=0)

    jacobian_single = jacrev(compute_single, argnums=0)

    converged = torch.zeros(B, dtype=torch.bool, device=device)
    reasons = np.array(["max_iter"] * B, dtype=object)
    iterations = torch.zeros(B, dtype=torch.int64, device=device)

    achieved_contacts = torch.zeros_like(t_target_anchors)
    achieved_normals = torch.zeros_like(t_target_normal)
    pos_residuals = torch.full((B, num_tips), float('inf'), device=device)
    norm_residuals = torch.full((B, num_tips), float("inf"), device=device)

    I = torch.eye(num_joints, device=device).unsqueeze(0).expand(B, num_joints, num_joints)
    damping_values = torch.full((B,), float(damping), dtype=torch.float32, device=device)

    for it in range(max_iter):
        active_mask = ~converged
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

            p_converged = torch.all(p_errs < pos_tolerance, dim=1)
            n_converged = (
                torch.all(n_dots > normal_tolerance_dot, dim=1)
                if require_normal_alignment
                else torch.ones(B, dtype=torch.bool, device=device)
            )
            new_converged = p_converged & n_converged & active_mask

            if new_converged.any():
                converged[new_converged] = True
                for i in torch.where(new_converged)[0]:
                    reasons[i.item()] = "converged"

            active_mask = ~converged
            if not active_mask.any():
                break

            cur_flat = torch.cat([p.view(B, -1), (n * normal_weight).view(B, -1)], dim=1)
            target_flat = torch.cat([p_target.view(B, -1), (n_target * normal_weight).view(B, -1)], dim=1)
            err = (target_flat - cur_flat).unsqueeze(-1)

        J_batch = torch.stack(
            [jacobian_single(q[idx], t_palm_pos[idx], t_palm_rot[idx]) for idx in range(B)],
            dim=0,
        )

        with torch.no_grad():
            J_t = J_batch.transpose(1, 2)
            damping_matrix = (
                damping_values.square() + regularization_weight
            )[:, None, None] * I
            H = torch.bmm(J_t, J_batch) + damping_matrix
            g = torch.bmm(J_t, err).squeeze(-1)
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
            )
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
            torch.all(dots > normal_tolerance_dot, dim=1)
            if require_normal_alignment
            else torch.ones(B, dtype=torch.bool, device=device)
        )
        final_converged = torch.all(pos_residuals < pos_tolerance, dim=1) & final_normal_ok
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
