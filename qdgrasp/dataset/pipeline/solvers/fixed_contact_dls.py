from typing import Optional
import numpy as np
import torch
from torch.func import jacrev, vmap

from qdgrasp.robot.spec import RobotSpec
from qdgrasp.dataset.pipeline.contracts import KinematicSolution

def solve_dls_ik_batch(
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
    normal_tolerance_dot: float = 0.866, # cos(30 degrees)
) -> KinematicSolution:
    """
    Batched Damped Least Squares IK optimizing both position and normal alignment.
    Uses PyTorch's torch.func.jacrev for Batched Autodiff.
    """
    B = palm_pos.shape[0]
    num_joints = len(spec.actuated_joint_names)
    num_tips = len(spec.fingertip_links)
    
    device = torch.device("cpu")
    
    t_palm_pos = torch.as_tensor(palm_pos, dtype=torch.float32, device=device)
    t_palm_rot = torch.as_tensor(palm_rot, dtype=torch.float32, device=device)
    t_target_pos = torch.as_tensor(target_contacts, dtype=torch.float32, device=device)
    t_target_normal = torch.as_tensor(target_normals, dtype=torch.float32, device=device)
    
    q_mins = torch.tensor([spec.joint_limits[j][0] for j in spec.actuated_joint_names], dtype=torch.float32, device=device)
    q_maxs = torch.tensor([spec.joint_limits[j][1] for j in spec.actuated_joint_names], dtype=torch.float32, device=device)
    
    if init_q is not None:
        q = torch.as_tensor(init_q, dtype=torch.float32, device=device)
    else:
        # Midpoint of joint limits
        q = ((q_mins + q_maxs) * 0.5).unsqueeze(0).expand(B, num_joints).clone()
        
    def compute_current(q_batch):
        # Forward kinematics for a single sample
        transforms = spec.forward_kinematics(t_palm_pos, t_palm_rot, q_batch)
        pos_list = []
        norm_list = []
        for tip in spec.fingertip_links:
            T = transforms[tip]
            pos_list.append(T[:, :3, 3])
            # Default normal is usually local Z axis
            norm_list.append(T[:, :3, 2])
        p = torch.stack(pos_list, dim=1) # [B, K, 3]
        n = torch.stack(norm_list, dim=1) # [B, K, 3]
        return torch.cat([p.view(B, -1), n.view(B, -1)], dim=1) # [B, 6K]

    converged = torch.zeros(B, dtype=torch.bool, device=device)
    reasons = np.array(["max_iter"] * B, dtype=object)
    
    achieved_contacts = torch.zeros_like(t_target_pos)
    achieved_normals = torch.zeros_like(t_target_normal)
    pos_residuals = torch.full((B, num_tips), float('inf'), device=device)
    norm_residuals = torch.full((B, num_tips), -1.0, device=device) # using dot product
    
    target_flat = torch.cat([t_target_pos.view(B, -1), t_target_normal.view(B, -1)], dim=1) # [B, 6K]
    
    # Pre-expand identity and damping factor
    I = torch.eye(num_joints, device=device).unsqueeze(0).expand(B, num_joints, num_joints)
    damping_matrix = (damping**2) * I

    for it in range(max_iter):
        active_mask = ~converged
        if not active_mask.any():
            break
            
        with torch.no_grad():
            # Vectorized FK for current values
            transforms = spec.forward_kinematics(t_palm_pos, t_palm_rot, q)
            pos_list = []
            norm_list = []
            for tip in spec.fingertip_links:
                pos_list.append(transforms[tip][:, :3, 3])
                norm_list.append(transforms[tip][:, :3, 2])
            
            p = torch.stack(pos_list, dim=1) # [B, K, 3]
            n = torch.stack(norm_list, dim=1) # [B, K, 3]
            
            achieved_contacts = p
            achieved_normals = n
            
            # Compute errors
            p_errs = torch.norm(t_target_pos - p, dim=-1) # [B, K]
            pos_residuals = p_errs
            
            n_dots = torch.sum(t_target_normal * n, dim=-1) # [B, K]
            norm_residuals = n_dots
            
            # Check convergence for active ones
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
                
            cur_flat = torch.cat([p.view(B, -1), n.view(B, -1)], dim=1) # [B, 6K]
            err = (target_flat - cur_flat).unsqueeze(-1) # [B, 6K, 1]
            
        # Compute Jacobian only for active (for efficiency, but let's just do all for simplicity of tensor shapes)
        # Computing Jacobian requires autograd, so it can't be in torch.no_grad()
        J_full = torch.autograd.functional.jacobian(compute_current, q) # [B, 6K, B, J]
        J_batch = torch.diagonal(J_full, dim1=0, dim2=2).permute(2, 0, 1) # [B, 6K, J]
        
        with torch.no_grad():
            J_t = J_batch.transpose(1, 2) # [B, J, 6K]
            
            H = torch.bmm(J_t, J_batch) + damping_matrix # [B, J, J]
            g = torch.bmm(J_t, err) # [B, J, 1]
            
            try:
                dq = torch.linalg.solve(H, g).squeeze(-1) # [B, J]
            except RuntimeError:
                # If singular, just use pseudo-inverse or zero
                dq = torch.zeros_like(q)
                
            # Update only active joints
            q_update = q + step_size * dq
            q = torch.where(active_mask.unsqueeze(-1), q_update, q)
            
            # Project onto limits
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
