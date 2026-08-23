"""DexGraspNet 2.0 WidthMapper Engine.

Maps target object grasp widths to natural, collision-free dexterous joint configurations
using canonical open postures and differentiable finger-thumb opposition kinematics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Sequence, Tuple
import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial.transform import Rotation

from qdgrasp.robot.spec import RobotSpec


@dataclass(frozen=True)
class RobotCanonicalMeta:
    """Canonical kinematic configuration and opposition metadata for a dexterous robot hand."""
    name: str
    canonical_qpos: Dict[str, float]
    thumb_link: str
    other_links: Sequence[str]
    opposition_axis_local: Tuple[float, float, float]  # Opposition line in canonical palm frame
    wrist_axis_local: Tuple[float, float, float]       # Wrist vector pointing away from palm


ROBOT_CANONICAL_METAS: Dict[str, RobotCanonicalMeta] = {
    "leap_hand": RobotCanonicalMeta(
        name="leap_hand",
        canonical_qpos={
            "if_mcp": 0.0, "if_rot": 0.0, "if_pip": 0.0, "if_dip": 0.0,
            "mf_mcp": 0.0, "mf_rot": 0.0, "mf_pip": 0.0, "mf_dip": 0.0,
            "rf_mcp": 0.0, "rf_rot": 0.0, "rf_pip": 0.0, "rf_dip": 0.0,
            "th_cmc": 0.8, "th_axl": 0.0, "th_mcp": 0.0, "th_ipl": 0.0,
        },
        thumb_link="th_ds",
        other_links=("if_ds", "mf_ds", "rf_ds"),
        opposition_axis_local=(0.0, 1.0, 0.0),
        wrist_axis_local=(-1.0, 0.0, 0.0),
    ),
    "wonik_allegro": RobotCanonicalMeta(
        name="wonik_allegro",
        canonical_qpos={
            "joint_0.0": 0.0, "joint_1.0": 0.0, "joint_2.0": 0.0, "joint_3.0": 0.0,
            "joint_4.0": 0.0, "joint_5.0": 0.0, "joint_6.0": 0.0, "joint_7.0": 0.0,
            "joint_8.0": 0.0, "joint_9.0": 0.0, "joint_10.0": 0.0, "joint_11.0": 0.0,
            "joint_12.0": 0.8, "joint_13.0": 0.0, "joint_14.0": 0.0, "joint_15.0": 0.0,
        },
        thumb_link="link_15.0_tip",
        other_links=("link_3.0_tip", "link_7.0_tip", "link_11.0_tip"),
        opposition_axis_local=(0.0, 1.0, 0.0),
        wrist_axis_local=(0.0, 0.0, -1.0),
    ),
    "shadow_hand": RobotCanonicalMeta(
        name="shadow_hand",
        canonical_qpos={
            "rh_WRJ2": 0.0, "rh_WRJ1": 0.0,
            "rh_FFJ4": 0.0, "rh_FFJ3": 0.0, "rh_FFJ2": 0.0, "rh_FFJ1": 0.0,
            "rh_MFJ4": 0.0, "rh_MFJ3": 0.0, "rh_MFJ2": 0.0, "rh_MFJ1": 0.0,
            "rh_RFJ4": 0.0, "rh_RFJ3": 0.0, "rh_RFJ2": 0.0, "rh_RFJ1": 0.0,
            "rh_LFJ5": 0.0, "rh_LFJ4": 0.0, "rh_LFJ3": 0.0, "rh_LFJ2": 0.0, "rh_LFJ1": 0.0,
            "rh_THJ5": 0.0, "rh_THJ4": 0.4, "rh_THJ3": 0.0, "rh_THJ2": 0.0, "rh_THJ1": 0.0,
        },
        thumb_link="rh_thdistal",
        other_links=("rh_ffdistal", "rh_mfdistal", "rh_rfdistal", "rh_lfdistal"),
        opposition_axis_local=(0.0, 1.0, 0.0),
        wrist_axis_local=(-1.0, 0.0, 0.0),
    ),
}


class WidthMapper:
    """Computes coordinated multi-finger joint targets for given grasp widths using DexGraspNet 2.0 principles."""

    def __init__(self, spec: RobotSpec) -> None:
        self.spec = spec
        robot_key = spec.config.name
        if robot_key not in ROBOT_CANONICAL_METAS:
            for k in ROBOT_CANONICAL_METAS:
                if k in robot_key:
                    robot_key = k
                    break
        self.meta = ROBOT_CANONICAL_METAS.get(robot_key)
        self.actuated_names = list(spec.actuated_joint_names)

    def get_canonical_open_qpos(self) -> np.ndarray:
        """Return the canonical open pre-grasp joint array [J]."""
        q = np.zeros(len(self.actuated_names), dtype=np.float32)
        if self.meta:
            for i, name in enumerate(self.actuated_names):
                q[i] = self.meta.canonical_qpos.get(name, 0.0)
        return q

    def map_width_to_qpos(
        self,
        target_width: float,
        max_steps: int = 25,
        lr: float = 8.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Differentiable squeeze optimization mapping target width to finger joint angles.
        Returns:
            (q_close [J], fingertip_targets [N_tips, 3])
        """
        q_init = self.get_canonical_open_qpos()
        q_tensor = torch.tensor(q_init[None], dtype=torch.float32, requires_grad=True)

        palm_pos = torch.zeros((1, 3), dtype=torch.float32)
        palm_rot = torch.eye(3, dtype=torch.float32)[None]

        # Determine open fingertip positions
        with torch.no_grad():
            open_tips = self.spec.fingertip_positions(palm_pos, palm_rot, q_tensor)[0]
            thumb_idx = 3 if len(open_tips) >= 4 else (len(open_tips) - 1)
            other_indices = [i for i in range(len(open_tips)) if i != thumb_idx]

            thumb_pos = open_tips[thumb_idx]
            other_pos = open_tips[other_indices]

            # Compute current open width
            mean_other = torch.mean(other_pos, dim=0)
            cur_width = float(torch.norm(thumb_pos - mean_other))
            delta_squeeze = max(0.0, (cur_width - target_width) * 0.5)

            # Squeeze along opposition normal
            opp_dir = F.normalize(mean_other - thumb_pos, dim=0)
            target_thumb = thumb_pos + delta_squeeze * opp_dir
            target_others = other_pos - delta_squeeze * opp_dir[None]

        # Optimize joint angles to match target squeezed fingertip positions
        optimizer = torch.optim.SGD([q_tensor], lr=lr)

        lower_limits = torch.tensor([self.spec.joint_limits.get(n, (-3.14, 3.14))[0] for n in self.actuated_names])
        upper_limits = torch.tensor([self.spec.joint_limits.get(n, (-3.14, 3.14))[1] for n in self.actuated_names])

        for _ in range(max_steps):
            optimizer.zero_grad()
            cur_tips = self.spec.fingertip_positions(palm_pos, palm_rot, q_tensor)[0]
            loss_thumb = torch.sum((cur_tips[thumb_idx] - target_thumb) ** 2)
            loss_others = torch.sum((cur_tips[other_indices] - target_others) ** 2)
            loss = loss_thumb + loss_others
            loss.backward()
            optimizer.step()

            # Clamp to physical joint limits
            with torch.no_grad():
                q_tensor.data.clamp_(lower_limits, upper_limits)

        q_final = q_tensor.detach()[0].numpy()
        with torch.no_grad():
            final_tips = self.spec.fingertip_positions(palm_pos, palm_rot, q_tensor)[0].numpy()

        return q_final, final_tips


def compute_canonical_grasp_frame(
    approach_direction: np.ndarray,
    pinch_axis: np.ndarray,
    elevation_min_deg: float = 30.0,
) -> np.ndarray:
    """
    Construct a 3x3 orthonormal rotation matrix R = [x_app, y_pinch, z_wrist]
    where the wrist vector z is strictly elevated above the table/floor plane.
    """
    x_app = approach_direction / np.linalg.norm(approach_direction)
    y_pinch = pinch_axis - np.dot(pinch_axis, x_app) * x_app
    y_norm = np.linalg.norm(y_pinch)
    if y_norm < 1e-6:
        # fallback orthogonal vector
        y_pinch = np.cross(x_app, np.array([0.0, 0.0, 1.0]))
        if np.linalg.norm(y_pinch) < 1e-6:
            y_pinch = np.cross(x_app, np.array([0.0, 1.0, 0.0]))
        y_norm = np.linalg.norm(y_pinch)
    y_pinch /= y_norm

    z_wrist = np.cross(x_app, y_pinch)
    z_wrist /= np.linalg.norm(z_wrist)

    # Ensure wrist points upward/backwards away from table
    if z_wrist[2] < 0.0:
        y_pinch = -y_pinch
        z_wrist = -z_wrist

    R = np.column_stack([x_app, y_pinch, z_wrist])
    return R
