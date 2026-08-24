"""Inverse-generated joint+palm refinement oracle for P3.2.1-06."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation

from qdgrasp.dataset.pipeline.solvers.joint_palm_dls import solve_joint_palm_dls_batch
from qdgrasp.robot.spec import RobotSpec


@pytest.mark.parametrize("profile", ["leap_hand.yaml", "wonik_allegro.yaml", "shadow_hand.yaml"])
def test_joint_palm_refinement_recovers_an_inverse_generated_task(profile: str) -> None:
    spec = RobotSpec.from_config(profile, sample_anchors=False)
    limits = np.asarray(
        [spec.joint_limits[name] for name in spec.actuated_joint_names], dtype=np.float32
    )
    q_truth = limits[:, 0] + 0.55 * (limits[:, 1] - limits[:, 0])
    q_seed = np.clip(q_truth + 0.015, limits[:, 0], limits[:, 1])
    true_pos = np.array([[0.0, 0.0, 0.12]], dtype=np.float32)
    true_rot = np.eye(3, dtype=np.float32)[None]
    with torch.no_grad():
        targets = spec.fingertip_positions(
            torch.from_numpy(true_pos), torch.from_numpy(true_rot), torch.from_numpy(q_truth[None])
        ).numpy()
        normals = spec.fingertip_contact_directions(
            torch.from_numpy(true_pos), torch.from_numpy(true_rot), torch.from_numpy(q_truth[None])
        ).numpy()

    seed_pos = true_pos + np.array([[0.004, -0.003, 0.002]], dtype=np.float32)
    seed_rot = Rotation.from_euler("xyz", [3.0, -2.0, 2.0], degrees=True).as_matrix().astype(
        np.float32
    )[None]
    active = np.zeros((1, len(spec.fingertip_links)), dtype=bool)
    active[0, : min(3, len(spec.fingertip_links))] = True
    result = solve_joint_palm_dls_batch(
        spec,
        seed_pos,
        seed_rot,
        targets,
        normals,
        init_q=q_seed[None],
        active_fingers=active,
        max_iter=40,
        floor_z=0.0,
    )

    assert bool(result.converged[0]), (
        profile,
        result.reason[0],
        result.position_residuals[0, active[0]],
        result.normal_residuals[0, active[0]],
    )
    assert np.max(result.position_residuals[0, active[0]]) < 0.005
    assert np.max(result.normal_residuals[0, active[0]]) < np.deg2rad(30.0)
    assert np.linalg.norm(result.palm_pos[0] - seed_pos[0]) <= 0.01 + 1e-6
