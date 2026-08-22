from __future__ import annotations

import pytest
import torch

from qdgrasp.robot.graph import HandGraph
from qdgrasp.robot.kinematics import compute_joint_transform, rpy_to_rotation_matrix, transform_points
from qdgrasp.robot.spec import RobotSpec


def test_rpy_to_rotation_matrix_properties() -> None:
    # Identity at (0, 0, 0)
    R_ident = rpy_to_rotation_matrix((0.0, 0.0, 0.0))
    assert torch.allclose(R_ident, torch.eye(3), atol=1e-6)

    # Orthogonality
    rpy = torch.randn(10, 3)
    R = rpy_to_rotation_matrix(rpy)
    I = torch.eye(3).unsqueeze(0).expand(10, 3, 3)
    RtR = torch.bmm(R.transpose(1, 2), R)
    assert torch.allclose(RtR, I, atol=1e-5)


def test_forward_kinematics_batch_parity_all_presets() -> None:
    presets = ("leap_hand.yaml", "wonik_allegro.yaml", "shadow_hand.yaml")
    for name in presets:
        spec = RobotSpec.from_config(name, sample_anchors=False)
        B = 4
        J = len(spec.actuated_joint_names)
        palm_pos = torch.randn(B, 3)
        palm_rot = torch.eye(3).unsqueeze(0).expand(B, 3, 3)
        joints = torch.randn(B, J) * 0.1

        t_batch = spec.forward_kinematics(palm_pos, palm_rot, joints)
        tips_batch = spec.fingertip_positions(palm_pos, palm_rot, joints)

        for b in range(B):
            t_single = spec.forward_kinematics(palm_pos[b : b + 1], palm_rot[b : b + 1], joints[b : b + 1])
            tips_single = spec.fingertip_positions(palm_pos[b : b + 1], palm_rot[b : b + 1], joints[b : b + 1])

            for link_name, mat in t_single.items():
                diff = (t_batch[link_name][b : b + 1] - mat).abs().max().item()
                assert diff < 1e-4, f"diff on {name} {link_name}: {diff}"

            diff_tips = (tips_batch[b : b + 1] - tips_single).abs().max().item()
            assert diff_tips < 1e-4


def test_hand_graph_linear_memory_scaling() -> None:
    spec_leap = RobotSpec.from_config("leap_hand.yaml", sample_anchors=False)
    g_leap = spec_leap.to_hand_graph()

    spec_shadow = RobotSpec.from_config("shadow_hand.yaml", sample_anchors=False)
    g_shadow = spec_shadow.to_hand_graph()

    assert g_leap.num_nodes == 18
    assert g_shadow.num_nodes == 26

    mem_ratio = g_shadow.memory_bytes() / g_leap.memory_bytes()
    node_ratio = 26 / 18
    # Scaling is linear O(L), so mem_ratio ~= node_ratio
    assert mem_ratio < node_ratio * 1.5
