from typing import Optional
import numpy as np
import torch
from torch.func import jacrev, vmap

from qdgrasp.robot.spec import RobotSpec
from qdgrasp.dataset.pipeline.contracts import KinematicSolution

def solve_region_dls_ik_batch(
    spec: RobotSpec,
    palm_pos: np.ndarray,
    palm_rot: np.ndarray,
    target_contacts: np.ndarray,
    target_normals: np.ndarray,
    init_q: Optional[np.ndarray] = None,
    damping: float = 0.01,
    step_size: float = 0.5,
    max_iter: int = 50,
    pos_tolerance: float = 0.005,
    normal_tolerance_dot: float = 0.866,
    region_radius: float = 0.02,
) -> KinematicSolution:
    """
    Batched Damped Least Squares IK optimizing position within a region.
    Target positions are dynamically projected onto the tangent plane of the anchor
    and clamped to `region_radius`.
    """
    B = palm_pos.shape[0]
    num_joints = len(spec.actuated_joint_names)
    num_tips = len(spec.fingertip_links)
    
    device = torch.device("cpu")
    
    t_palm_pos = torch.as_tensor(palm_pos, dtype=torch.float32, device=device)
    t_palm_rot = torch.as_tensor(palm_rot, dtype=torch.float32, device=device)
    t_target_anchors = torch.as_tensor(target_contacts, dtype=torch.float32, device=device)
    t_target_normal = torch.as_tensor(target_normals, dtype=torch.float32, device=device)
    
    q_mins = torch.tensor([spec.joint_limits[j][0] for j in spec.actuated_joint_names], dtype=torch.float32, device=device)
    q_maxs = torch.tensor([spec.joint_limits[j][1] for j in spec.actuated_joint_names], dtype=torch.float32, device=device)
    
    if init_q is not None:
        q = torch.as_tensor(init_q, dtype=torch.float32, device=device)
    else:
        q = ((q_mins + q_maxs) * 0.5).unsqueeze(0).expand(B, num_joints).clone()
        
    def compute_current(q_batch):
        transforms = spec.forward_kinematics(t_palm_pos, t_palm_rot, q_batch)
        pos_list = []
        norm_list = []
        for tip in spec.fingertip_links:
            T = transforms[tip]
            pos_list.append(T[:, :3, 3])
            norm_list.append(T[:, :3, 2])
        p = torch.stack(pos_list, dim=1)
        n = torch.stack(norm_list, dim=1)
        return torch.cat([p.view(B, -1), n.view(B, -1)], dim=1)

    converged = torch.zeros(B, dtype=torch.bool, device=device)
    reasons = np.array(["max_iter"] * B, dtype=object)
    
    achieved_contacts = torch.zeros_like(t_target_anchors)
    achieved_normals = torch.zeros_like(t_target_normal)
    pos_residuals = torch.full((B, num_tips), float('inf'), device=device)
    norm_residuals = torch.full((B, num_tips), -1.0, device=device)
    
    I = torch.eye(num_joints, device=device).unsqueeze(0).expand(B, num_joints, num_joints)
    damping_matrix = (damping**2) * I

    for it in range(max_iter):
        active_mask = ~converged
        if not active_mask.any():
            break
            
        with torch.no_grad():
            transforms = spec.forward_kinematics(t_palm_pos, t_palm_rot, q)
            pos_list = []
            norm_list = []
            for tip in spec.fingertip_links:
                pos_list.append(transforms[tip][:, :3, 3])
                norm_list.append(transforms[tip][:, :3, 2])
            
            p = torch.stack(pos_list, dim=1)
            n = torch.stack(norm_list, dim=1)
            
            achieved_contacts = p
            achieved_normals = n
            
            # Dynamic target projection
            vec = p - t_target_anchors
            dist_along_normal = torch.sum(vec * t_target_normal, dim=-1, keepdim=True)
            p_proj = p - dist_along_normal * t_target_normal
            
            vec_in_plane = p_proj - t_target_anchors
            dist_in_plane = torch.norm(vec_in_plane, dim=-1, keepdim=True)
            mask = dist_in_plane > region_radius
            vec_in_plane_clamped = torch.where(
                mask, 
                vec_in_plane / torch.clamp(dist_in_plane, min=1e-6) * region_radius, 
                vec_in_plane
            )
            p_target = t_target_anchors + vec_in_plane_clamped
            
            p_errs = torch.norm(p_target - p, dim=-1)
            pos_residuals = p_errs
            
            n_dots = torch.sum(t_target_normal * n, dim=-1)
            norm_residuals = n_dots
            
            p_converged = torch.all(p_errs < pos_tolerance, dim=1)
            n_converged = torch.all(n_dots > normal_tolerance_dot, dim=1)
            new_converged = p_converged & n_converged & active_mask
            
            if new_converged.any():
                converged[new_converged] = True
                for i in torch.where(new_converged)[0]:
                    reasons[i.item()] = "converged"
            
            active_mask = ~converged
            if not active_mask.any():
                break
                
            cur_flat = torch.cat([p.view(B, -1), n.view(B, -1)], dim=1)
            target_flat = torch.cat([p_target.view(B, -1), t_target_normal.view(B, -1)], dim=1)
            err = (target_flat - cur_flat).unsqueeze(-1)
            
        J_full = torch.autograd.functional.jacobian(compute_current, q)
        J_batch = torch.diagonal(J_full, dim1=0, dim2=2).permute(2, 0, 1)
        
        with torch.no_grad():
            J_t = J_batch.transpose(1, 2)
            H = torch.bmm(J_t, J_batch) + damping_matrix
            g = torch.bmm(J_t, err)
            
            try:
                dq = torch.linalg.solve(H, g).squeeze(-1)
            except RuntimeError:
                dq = torch.zeros_like(q)
                
            q_update = q + step_size * dq
            q = torch.where(active_mask.unsqueeze(-1), q_update, q)
            q = torch.clamp(q, min=q_mins, max=q_maxs)
            
    return KinematicSolution(
        q=q.cpu().numpy(),
        achieved_contacts=achieved_contacts.cpu().numpy(),
        achieved_normals=achieved_normals.cpu().numpy(),
        position_residuals=pos_residuals.cpu().numpy(),
        normal_residuals=norm_residuals.cpu().numpy(),
        converged=converged.cpu().numpy(),
        reason=reasons
    )
