from typing import Optional
import numpy as np
import torch
from torch.func import jacrev

from qdgrasp.robot.spec import RobotSpec
from qdgrasp.dataset.pipeline.contracts import KinematicSolution

def solve_dls_ik_batch(
    spec: RobotSpec,
    palm_pos: np.ndarray,
    palm_rot: np.ndarray,
    target_contacts: np.ndarray,
    target_normals: np.ndarray,
    init_q: Optional[np.ndarray] = None,
    active_fingers: Optional[np.ndarray | torch.Tensor] = None,
    damping: float = 0.01,
    step_size: float = 0.5,
    max_iter: int = 50,
    pos_tolerance: float = 0.005,
    normal_tolerance_dot: float = 0.866, # cos(30 degrees)
    normal_weight: float = 0.01,
    require_normal_alignment: bool = True,
    regularization_weight: float = 1e-5,
    joint_margin_weight: float = 1e-4,
    min_active_fingers: int = 2,
) -> KinematicSolution:
    """
    Batched Damped Least Squares IK optimizing both position and normal alignment.
    Uses batched FK and autodiff. Supports active_fingers boolean mask [B, K] or [K].
    """
    B = palm_pos.shape[0]
    num_joints = len(spec.actuated_joint_names)
    num_tips = len(spec.fingertip_links)

    device = torch.device("cpu")

    t_palm_pos = torch.as_tensor(palm_pos, dtype=torch.float32, device=device)
    t_palm_rot = torch.as_tensor(palm_rot, dtype=torch.float32, device=device)
    t_target_pos = torch.as_tensor(target_contacts, dtype=torch.float32, device=device)
    t_target_normal = torch.as_tensor(target_normals, dtype=torch.float32, device=device)

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
        # Midpoint of joint limits
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

    # Check minimum active fingers
    insufficient_fingers = (t_active_fingers.sum(dim=-1) < min_active_fingers)
    for idx in torch.where(insufficient_fingers)[0]:
        reasons[idx.item()] = "insufficient_active_fingers"

    achieved_contacts = torch.zeros_like(t_target_pos)
    achieved_normals = torch.zeros_like(t_target_normal)
    pos_residuals = torch.full((B, num_tips), float('inf'), device=device)
    norm_residuals = torch.full((B, num_tips), float("inf"), device=device)

    target_flat = torch.cat([t_target_pos.view(B, -1), (t_target_normal * normal_weight).view(B, -1)], dim=1) # [B, 6K]

    # Per-candidate adaptive damping avoids letting one singular candidate
    # destabilize every other member of the batch.
    I = torch.eye(num_joints, device=device).unsqueeze(0).expand(B, num_joints, num_joints)
    damping_values = torch.full((B,), float(damping), dtype=torch.float32, device=device)

    for it in range(max_iter):
        active_mask = ~converged & ~insufficient_fingers
        if not active_mask.any():
            break
        iterations[active_mask] += 1

        with torch.no_grad():
            # Vectorized FK for current values
            transforms = spec.forward_kinematics(t_palm_pos, t_palm_rot, q)
            pos_list = []
            norm_list = []
            for tip in spec.fingertip_links:
                pos_list.append(contact_position(transforms, tip))
                norm_list.append(contact_direction(transforms, tip))

            p = torch.stack(pos_list, dim=1) # [B, K, 3]
            n = torch.stack(norm_list, dim=1) # [B, K, 3]

            achieved_contacts = p
            achieved_normals = n

            # Compute errors
            p_errs = torch.norm(t_target_pos - p, dim=-1) # [B, K]
            pos_residuals = p_errs

            n_dots = torch.sum(t_target_normal * n, dim=-1).clamp(-1.0, 1.0)
            norm_residuals = torch.acos(n_dots)

            # Check convergence for active ones
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

            cur_flat = torch.cat([p.view(B, -1), (n * normal_weight).view(B, -1)], dim=1) # [B, 6K]
            err = ((target_flat - cur_flat) * active_flat_mask).unsqueeze(-1) # [B, 6K, 1]

        # Per-sample Jacobians avoid materializing cross-batch derivatives
        # [B, output, B, J], which caused severe memory pressure in ablation.
        J_batch = torch.stack(
            [jacobian_single(q[idx], t_palm_pos[idx], t_palm_rot[idx]) for idx in range(B)],
            dim=0,
        )

        with torch.no_grad():
            J_t = J_batch.transpose(1, 2) # [B, J, 6K]

            damping_matrix = (
                damping_values.square() + regularization_weight
            )[:, None, None] * I
            H = torch.bmm(J_t, J_batch) + damping_matrix # [B, J, J]
            g = torch.bmm(J_t, err).squeeze(-1) # [B, J]
            g += regularization_weight * (q_reference - q)

            joint_span = torch.clamp(q_maxs - q_mins, min=1e-6)
            margin = 0.05 * joint_span
            low_repulsion = torch.clamp(margin - (q - q_mins), min=0.0) / margin
            high_repulsion = torch.clamp(margin - (q_maxs - q), min=0.0) / margin
            g += joint_margin_weight * (low_repulsion - high_repulsion)

            try:
                dq = torch.linalg.solve(H, g.unsqueeze(-1)).squeeze(-1) # [B, J]
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
            current_cost = torch.sum(err.squeeze(-1).square(), dim=1)
            trial_error = torch.cat(
                [
                    (t_target_pos - trial_pos).reshape(B, -1),
                    ((t_target_normal - trial_normals) * normal_weight).reshape(B, -1),
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

    # Recompute evidence from the returned q so q/contact arrays cannot be one
    # iteration out of sync after max_iter.
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
        pos_residuals = torch.linalg.norm(t_target_pos - achieved_contacts, dim=-1)
        dots = torch.sum(t_target_normal * achieved_normals, dim=-1).clamp(-1.0, 1.0)
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
