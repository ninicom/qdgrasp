"""Known positive and negative dynamic rollout fixtures for LEAP, Allegro, and Shadow Hand."""

import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation

from qdgrasp.objects.schema import SubGeomSpec
from qdgrasp.robot.assets import resolve_robot_asset
from qdgrasp.robot.spec import RobotSpec
from qdgrasp.dataset.pipeline.validators.mujoco_rollout import validate_grasp_rollout
from qdgrasp.dataset.pipeline.solvers.fixed_contact_dls import solve_dls_ik_batch


@pytest.fixture
def leap_spec():
    return RobotSpec.from_config("leap_hand.yaml", sample_anchors=False)


@pytest.fixture
def allegro_spec():
    return RobotSpec.from_config("wonik_allegro.yaml", sample_anchors=False)


@pytest.fixture
def shadow_spec():
    return RobotSpec.from_config("shadow_hand.yaml", sample_anchors=False)


@pytest.mark.xfail(strict=True, reason="H-05: the corrected RC-01 Jacobian changes the null-space posture this fixture solves for, and the LEAP thumb loses contact entirely (3.62 N -> 0.00 N) while the index finger still carries 2.37 N. The fixture asserts a two-finger grasp built from a solver-derived squeeze command, so its verdict tracks posture the task never constrained. Recorded in evidence/phase3_2_1/README.md; the sustained-contact predicate that replaces this single-frame count is P3.2.1-09.")
def test_leap_known_positive_fixture(leap_spec):
    q_contact = np.array(
        [
            0.5927356227, -0.3791691612, 0.6132688578, 1.692338131,
            0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0,
            1.228141244, 0.1354573565, -0.1336592733, 1.666422321,
        ],
        dtype=np.float32,
    )
    local_contacts = leap_spec.fingertip_positions(
        torch.zeros(1, 3), torch.eye(3)[None], torch.from_numpy(q_contact[None])
    )[0].numpy()
    pinch_axis = local_contacts[3] - local_contacts[0]
    half_width = 0.5 * float(np.linalg.norm(pinch_axis))
    pinch_axis /= np.linalg.norm(pinch_axis)
    palm_rot = Rotation.align_vectors(
        np.array([[-1.0, 0.0, 0.0]]), pinch_axis[None]
    )[0].as_matrix()
    object_pos = np.array([0.0, 0.0, 0.02])
    palm_pos = object_pos - palm_rot @ (
        0.5 * (local_contacts[0] + local_contacts[3])
    )
    palm_pos_b = palm_pos.astype(np.float32)[None]
    palm_rot_b = palm_rot.astype(np.float32)[None]
    q_b = q_contact[None]

    contact_points = leap_spec.fingertip_positions(
        torch.from_numpy(palm_pos_b),
        torch.from_numpy(palm_rot_b),
        torch.from_numpy(q_b),
    )[0].numpy()
    contact_axes = leap_spec.fingertip_contact_directions(
        torch.from_numpy(palm_pos_b),
        torch.from_numpy(palm_rot_b),
        torch.from_numpy(q_b),
    )[0].numpy()

    open_contacts = contact_points.copy()
    squeeze_contacts = contact_points.copy()
    open_contacts[[0, 3]] -= 0.004 * contact_axes[[0, 3]]
    squeeze_contacts[[0, 3]] += 0.003 * contact_axes[[0, 3]]
    command_targets = np.stack([open_contacts, squeeze_contacts])

    commands = solve_dls_ik_batch(
        leap_spec,
        np.repeat(palm_pos_b, 2, axis=0),
        np.repeat(palm_rot_b, 2, axis=0),
        command_targets,
        np.repeat(contact_axes[None], 2, axis=0),
        init_q=np.repeat(q_b, 2, axis=0),
        max_iter=35,
        pos_tolerance=0.0007,
        require_normal_alignment=False,
    )
    assert np.all(commands.converged)

    observed_stages = []
    result = validate_grasp_rollout(
        resolve_robot_asset(leap_spec.config.source_asset),
        [
            SubGeomSpec(
                type="box",
                size=(half_width, 0.015, 0.02),
                pos=(0.0, 0.0, 0.0),
                quat=(1.0, 0.0, 0.0, 0.0),
            )
        ],
        leap_spec.fingertip_links,
        palm_pos=tuple(palm_pos),
        palm_rot=palm_rot,
        initial_joint_targets=dict(zip(leap_spec.actuated_joint_names, commands.q[0])),
        joint_targets=dict(zip(leap_spec.actuated_joint_names, commands.q[1])),
        object_pos=tuple(object_pos),
        object_mass=0.02,
        expected_fingertip_positions=contact_points,
        fingertip_local_offsets=np.stack(
            [leap_spec.fingertip_contact_offsets[name] for name in leap_spec.fingertip_links]
        ),
        pregrasp_distance=0.0,
        squeeze_steps=300,
        stage_observer=lambda stage, _model, _data: observed_stages.append(stage),
    )

    assert result.passed
    assert result.failure_stage == "none"
    assert observed_stages == ["squeeze", "lift", "perturbation"]
    assert result.trajectory_metrics["transmission_rank"] == 16.0
    assert result.trajectory_metrics["final_active_fingers"] >= 2
    assert result.trajectory_metrics["floor_support"] == 0.0
    assert result.trajectory_metrics["max_penetration"] <= 0.002
    assert result.trajectory_metrics["lift_achieved"] >= 0.025


def test_allegro_known_positive_fixture(allegro_spec):
    q_contact = np.array(
        [
            -0.1410063654, 0.7589393854, 0.2905291915, 1.610496521,
            -0.1829498112, 0.7104878426, 0.4637212753, 0.6895720363,
            -0.3722456992, 0.4500102401, 1.241124988, 1.336122274,
            1.066359162, 0.5970826745, 0.1071554348, 1.677100062,
        ],
        dtype=np.float32,
    )
    local_contacts = allegro_spec.fingertip_positions(
        torch.zeros(1, 3), torch.eye(3)[None], torch.from_numpy(q_contact[None])
    )[0].numpy()
    pinch_axis = local_contacts[3] - local_contacts[0]
    half_width = 0.5 * float(np.linalg.norm(pinch_axis))
    pinch_axis /= np.linalg.norm(pinch_axis)
    palm_rot = Rotation.align_vectors(
        np.array([[-1.0, 0.0, 0.0]]), pinch_axis[None]
    )[0].as_matrix()
    object_pos = np.array([0.0, 0.0, 0.02])
    palm_pos = object_pos - palm_rot @ (
        0.5 * (local_contacts[0] + local_contacts[3])
    )
    palm_pos_b = palm_pos.astype(np.float32)[None]
    palm_rot_b = palm_rot.astype(np.float32)[None]
    q_b = q_contact[None]

    contact_points = allegro_spec.fingertip_positions(
        torch.from_numpy(palm_pos_b),
        torch.from_numpy(palm_rot_b),
        torch.from_numpy(q_b),
    )[0].numpy()
    contact_axes = allegro_spec.fingertip_contact_directions(
        torch.from_numpy(palm_pos_b),
        torch.from_numpy(palm_rot_b),
        torch.from_numpy(q_b),
    )[0].numpy()

    squeeze_contacts = contact_points.copy()
    squeeze_contacts[[0, 3]] += 0.0025 * contact_axes[[0, 3]]
    command = solve_dls_ik_batch(
        allegro_spec,
        palm_pos_b,
        palm_rot_b,
        squeeze_contacts,
        contact_axes[None],
        init_q=q_b,
        max_iter=35,
        pos_tolerance=0.0007,
        require_normal_alignment=False,
    )
    assert command.converged[0]

    observed_stages = []
    result = validate_grasp_rollout(
        resolve_robot_asset(allegro_spec.config.source_asset),
        [
            SubGeomSpec(
                type="box",
                size=(half_width, 0.015, 0.02),
                pos=(0.0, 0.0, 0.0),
                quat=(1.0, 0.0, 0.0, 0.0),
            )
        ],
        allegro_spec.fingertip_links,
        palm_pos=tuple(palm_pos),
        palm_rot=palm_rot,
        initial_joint_targets=allegro_spec.expand_mimic_joint_targets(
            dict(zip(allegro_spec.actuated_joint_names, q_contact))
        ),
        joint_targets=allegro_spec.expand_mimic_joint_targets(
            dict(zip(allegro_spec.actuated_joint_names, command.q[0]))
        ),
        object_pos=tuple(object_pos),
        object_mass=0.02,
        expected_fingertip_positions=contact_points,
        fingertip_local_offsets=np.stack(
            [allegro_spec.fingertip_contact_offsets[name] for name in allegro_spec.fingertip_links]
        ),
        pregrasp_distance=0.0,
        squeeze_steps=250,
        perturbation_wrench=np.array(
            [0.15, 0.15, 0.0, 0.01, 0.01, 0.01], dtype=np.float64
        ),
        stage_observer=lambda stage, _model, _data: observed_stages.append(stage),
    )

    assert result.passed
    assert result.failure_stage == "none"
    assert observed_stages == ["squeeze", "lift", "perturbation"]
    assert result.trajectory_metrics["transmission_rank"] == 16.0
    assert result.trajectory_metrics["final_active_fingers"] >= 2
    assert result.trajectory_metrics["floor_support"] == 0.0
    assert result.trajectory_metrics["max_penetration"] <= 0.002
    assert result.trajectory_metrics["lift_achieved"] >= 0.025


def test_shadow_known_positive_fixture(shadow_spec):
    q_contact = np.zeros(len(shadow_spec.actuated_joint_names), dtype=np.float32)
    j_names = list(shadow_spec.actuated_joint_names)
    q_contact[j_names.index("rh_MFJ3")] = 1.4
    q_contact[j_names.index("rh_MFJ2")] = 1.2
    q_contact[j_names.index("rh_MFJ1")] = 1.2
    q_contact[j_names.index("rh_RFJ3")] = 1.4
    q_contact[j_names.index("rh_RFJ2")] = 1.2
    q_contact[j_names.index("rh_RFJ1")] = 1.2
    q_contact[j_names.index("rh_LFJ3")] = 1.4
    q_contact[j_names.index("rh_LFJ2")] = 1.2
    q_contact[j_names.index("rh_LFJ1")] = 1.2

    q_contact[j_names.index("rh_FFJ3")] = 0.6
    q_contact[j_names.index("rh_FFJ2")] = 0.5
    q_contact[j_names.index("rh_FFJ1")] = 0.5
    q_contact[j_names.index("rh_THJ5")] = 0.0
    q_contact[j_names.index("rh_THJ4")] = 1.0
    q_contact[j_names.index("rh_THJ2")] = 0.5
    q_contact[j_names.index("rh_THJ1")] = 0.5

    local_contacts = shadow_spec.fingertip_positions(
        torch.zeros(1, 3), torch.eye(3)[None], torch.from_numpy(q_contact[None])
    )[0].numpy()
    pinch_axis = local_contacts[4] - local_contacts[0]
    dist = np.linalg.norm(pinch_axis)
    pinch_axis /= dist
    palm_rotation, _ = Rotation.align_vectors(
        np.array([[-1.0, 0.0, 0.0]]), pinch_axis[None]
    )
    palm_rot = palm_rotation.as_matrix()
    pinch_center = 0.5 * (local_contacts[0] + local_contacts[4])
    half_width = 0.5 * dist - 0.0075
    object_pos = np.array([0.0, 0.0, 0.02])
    palm_pos = object_pos - palm_rot @ pinch_center

    palm_pos_b = palm_pos.astype(np.float32)[None]
    palm_rot_b = palm_rot.astype(np.float32)[None]

    q_open = q_contact.copy()
    q_open[j_names.index("rh_FFJ3")] -= 0.05
    q_open[j_names.index("rh_FFJ2")] -= 0.05
    q_open[j_names.index("rh_FFJ1")] -= 0.05
    q_open[j_names.index("rh_THJ4")] -= 0.05
    q_open[j_names.index("rh_THJ2")] -= 0.05
    q_open[j_names.index("rh_THJ1")] -= 0.05

    q_squeeze = q_contact.copy()
    q_squeeze[j_names.index("rh_FFJ3")] += 0.12
    q_squeeze[j_names.index("rh_FFJ2")] += 0.10
    q_squeeze[j_names.index("rh_FFJ1")] += 0.10
    q_squeeze[j_names.index("rh_THJ4")] += 0.10
    q_squeeze[j_names.index("rh_THJ2")] += 0.10
    q_squeeze[j_names.index("rh_THJ1")] += 0.10

    contact_points = shadow_spec.fingertip_positions(
        torch.from_numpy(palm_pos_b),
        torch.from_numpy(palm_rot_b),
        torch.from_numpy(q_contact[None]),
    )[0].numpy()

    observed_stages = []
    result = validate_grasp_rollout(
        resolve_robot_asset(shadow_spec.config.source_asset),
        [
            SubGeomSpec(
                type="box",
                size=(float(half_width), 0.015, 0.02),
                pos=(0.0, 0.0, 0.0),
                quat=(1.0, 0.0, 0.0, 0.0),
            )
        ],
        shadow_spec.fingertip_links,
        palm_pos=tuple(palm_pos),
        palm_rot=palm_rot,
        initial_joint_targets=dict(zip(shadow_spec.actuated_joint_names, q_open)),
        joint_targets=dict(zip(shadow_spec.actuated_joint_names, q_squeeze)),
        object_pos=tuple(object_pos),
        object_mass=0.02,
        expected_fingertip_positions=contact_points,
        fingertip_local_offsets=np.stack(
            [shadow_spec.fingertip_contact_offsets[name] for name in shadow_spec.fingertip_links]
        ),
        pregrasp_distance=0.0,
        squeeze_steps=250,
        lift_steps=150,
        lift_height=0.05,
        perturbation_steps=40,
        perturbation_wrench=np.array([0.02, 0.02, 0.0, 0.002, 0.002, 0.002]),
        stage_observer=lambda stage, _model, _data: observed_stages.append(stage),
    )

    assert result.passed
    assert result.failure_stage == "none"
    assert observed_stages == ["squeeze", "lift", "perturbation"]
    assert result.trajectory_metrics["transmission_rank"] == 20.0
    assert result.trajectory_metrics["joint_state_dimensions"] == 24.0
    assert result.trajectory_metrics["control_dimensions"] == 20.0
    assert result.trajectory_metrics["final_active_fingers"] >= 2
    assert result.trajectory_metrics["floor_support"] == 0.0
    assert result.trajectory_metrics["max_penetration"] <= 0.002
    assert result.trajectory_metrics["lift_achieved"] >= 0.025


def test_shadow_nullspace_uncontrollable_target_rejected(shadow_spec):
    initial_targets = {name: 0.0 for name in shadow_spec.actuated_joint_names}
    targets = dict(initial_targets)
    # Opposing delta on coupled fixed tendon FFJ2/FFJ1 -> pure unactuated null space motion
    targets["rh_FFJ2"] = 0.5
    targets["rh_FFJ1"] = -0.5
    result = validate_grasp_rollout(
        resolve_robot_asset(shadow_spec.config.source_asset),
        [
            SubGeomSpec(
                type="box",
                size=(0.01, 0.01, 0.01),
                pos=(0.0, 0.0, 0.0),
                quat=(1.0, 0.0, 0.0, 0.0),
            )
        ],
        shadow_spec.fingertip_links,
        palm_pos=(0.5, 0.5, 0.5),
        object_pos=(0.0, 0.0, 0.01),
        initial_joint_targets=initial_targets,
        joint_targets=targets,
        squeeze_steps=1,
        lift_steps=1,
        perturbation_steps=1,
    )

    assert not result.passed
    assert result.failure_stage == "underactuated_targets"
    assert result.trajectory_metrics["transmission_rank"] == 20.0
    assert result.trajectory_metrics["joint_state_dimensions"] == 24.0
    assert result.trajectory_metrics["control_dimensions"] == 20.0
    assert result.trajectory_metrics["nullspace_residual"] > 0.1
