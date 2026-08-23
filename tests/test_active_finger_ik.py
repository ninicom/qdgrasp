"""Tests for active-finger mask in fixed and region DLS IK solvers."""

import numpy as np
import torch
import pytest

from qdgrasp.robot.spec import RobotSpec
from qdgrasp.dataset.pipeline.solvers.fixed_contact_dls import solve_dls_ik_batch
from qdgrasp.dataset.pipeline.solvers.region_dls import solve_region_dls_ik_batch


@pytest.fixture
def leap_spec():
    return RobotSpec.from_config("qdgrasp/presets/robots/leap_hand.yaml")


@pytest.fixture
def shadow_spec():
    return RobotSpec.from_config("qdgrasp/presets/robots/shadow_hand.yaml")


def _compute_nominal_contact_targets(spec, palm_pos, palm_rot, q_nom):
    transforms = spec.forward_kinematics(
        torch.from_numpy(palm_pos), torch.from_numpy(palm_rot), torch.from_numpy(q_nom)
    )
    num_tips = len(spec.fingertip_links)
    target_pos = np.zeros((1, num_tips, 3), dtype=np.float32)
    target_norm = np.zeros((1, num_tips, 3), dtype=np.float32)

    for i, tip in enumerate(spec.fingertip_links):
        transform = transforms[tip]
        offset = getattr(spec, "fingertip_contact_offsets", {}).get(tip)
        if offset is not None:
            p = transform[:, :3, 3] + torch.matmul(
                transform[:, :3, :3], torch.tensor(offset, dtype=torch.float32).view(3, 1)
            ).squeeze(-1)
        else:
            p = transform[:, :3, 3]
        target_pos[0, i] = p[0].detach().cpu().numpy()

        axis = getattr(spec, "fingertip_contact_axes", {}).get(tip)
        if axis is not None:
            n = torch.nn.functional.normalize(
                torch.matmul(
                    transform[:, :3, :3], torch.tensor(axis, dtype=torch.float32).view(3, 1)
                ).squeeze(-1),
                dim=-1,
            )
        else:
            parent = spec.links[tip].parent_link
            origin = transforms[parent][:, :3, 3]
            n = torch.nn.functional.normalize(p - origin, dim=-1)
        target_norm[0, i] = n[0].detach().cpu().numpy()

    return target_pos, target_norm


def test_leap_active_fingers_pinch(leap_spec):
    palm_pos = np.array([[0.0, 0.0, 0.15]], dtype=np.float32)
    palm_rot = np.eye(3, dtype=np.float32)[None, ...]
    q_nom = np.zeros((1, len(leap_spec.actuated_joint_names)), dtype=np.float32)

    target_pos, target_norm = _compute_nominal_contact_targets(leap_spec, palm_pos, palm_rot, q_nom)

    # Inactive fingertips 1 and 2 placed at unreachable targets
    target_pos[0, 1] += 5.0
    target_pos[0, 2] += 5.0

    # 1. With all fingers active: must FAIL because finger 1 and 2 are unreachable
    res_all = solve_dls_ik_batch(
        leap_spec,
        palm_pos,
        palm_rot,
        target_pos,
        target_norm,
        init_q=q_nom,
        max_iter=15,
    )
    assert not res_all.converged[0]

    # 2. With active mask [True, False, False, True] (active fingers 0 and 3): must SUCCEED!
    active_mask = np.array([[True, False, False, True]], dtype=bool)
    res_active = solve_dls_ik_batch(
        leap_spec,
        palm_pos,
        palm_rot,
        target_pos,
        target_norm,
        init_q=q_nom,
        active_fingers=active_mask,
        max_iter=25,
    )
    assert res_active.converged[0]
    assert res_active.reason[0] == "converged"
    assert res_active.position_residuals[0, 0] < 0.005
    assert res_active.position_residuals[0, 3] < 0.005


def test_insufficient_active_fingers_rejection(leap_spec):
    palm_pos = np.array([[0.0, 0.0, 0.15]], dtype=np.float32)
    palm_rot = np.eye(3, dtype=np.float32)[None, ...]
    num_tips = len(leap_spec.fingertip_links)
    target_pos = np.zeros((1, num_tips, 3), dtype=np.float32)
    target_norm = np.zeros((1, num_tips, 3), dtype=np.float32)

    # Only 1 active finger: should be rejected immediately
    active_mask = np.array([[True, False, False, False]], dtype=bool)
    res = solve_dls_ik_batch(
        leap_spec,
        palm_pos,
        palm_rot,
        target_pos,
        target_norm,
        active_fingers=active_mask,
        min_active_fingers=2,
    )
    assert not res.converged[0]
    assert res.reason[0] == "insufficient_active_fingers"


def test_shadow_active_fingers_pinch(shadow_spec):
    palm_pos = np.array([[0.0, 0.0, 0.15]], dtype=np.float32)
    palm_rot = np.eye(3, dtype=np.float32)[None, ...]
    q_nom = np.zeros((1, len(shadow_spec.actuated_joint_names)), dtype=np.float32)

    target_pos, target_norm = _compute_nominal_contact_targets(shadow_spec, palm_pos, palm_rot, q_nom)

    # Inactive middle, ring, little fingers (indices 1, 2, 3) placed at unreachable target
    target_pos[0, 1] += 5.0
    target_pos[0, 2] += 5.0
    target_pos[0, 3] += 5.0

    # Active index and thumb (indices 0 and 4)
    active_mask = np.array([[True, False, False, False, True]], dtype=bool)
    res = solve_dls_ik_batch(
        shadow_spec,
        palm_pos,
        palm_rot,
        target_pos,
        target_norm,
        init_q=q_nom,
        active_fingers=active_mask,
        max_iter=25,
    )
    assert res.converged[0]
    assert res.reason[0] == "converged"
    assert res.position_residuals[0, 0] < 0.005
    assert res.position_residuals[0, 4] < 0.005
