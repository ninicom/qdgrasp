import mujoco
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
import torch

from qdgrasp.objects.schema import SubGeomSpec
from qdgrasp.dataset.pipeline.validators.mujoco_rollout import (
    build_rollout_scene_model,
    validate_grasp_rollout,
    smoothstep,
)
from qdgrasp.dataset.pipeline.solvers.fixed_contact_dls import solve_dls_ik_batch
from qdgrasp.robot.spec import RobotSpec
from qdgrasp.robot.spec import resolve_robot_asset


def test_smoothstep():
    assert smoothstep(0.0) == 0.0
    assert smoothstep(1.0) == 1.0
    assert 0.0 < smoothstep(0.5) < 1.0
    assert smoothstep(-0.5) == 0.0
    assert smoothstep(1.5) == 1.0


def test_build_rollout_scene_model():
    asset_path = resolve_robot_asset("asset://mujoco-menagerie/shadow_hand/right_hand.xml")

    geoms = [
        SubGeomSpec(type="box", size=(0.02, 0.02, 0.02), pos=(0.0, 0.0, 0.05), quat=(1.0, 0.0, 0.0, 0.0))
    ]

    model = build_rollout_scene_model(asset_path, geoms, object_pos=(0.0, 0.0, 0.05), object_mass=0.1)
    assert model is not None
    assert model.nbody > 0
    # Check that hand_mocap and target_object exist
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand_mocap") >= 0
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target_object") >= 0
    weld_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, "mocap_weld")
    assert weld_id >= 0
    np.testing.assert_allclose(
        model.eq_data[weld_id],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
    )
    np.testing.assert_allclose(model.eq_solref[weld_id], [0.01, 1.0])


def test_validate_grasp_rollout_no_contact():
    asset_path = resolve_robot_asset("asset://mujoco-menagerie/shadow_hand/right_hand.xml")

    geoms = [
        SubGeomSpec(type="box", size=(0.02, 0.02, 0.02), pos=(0.0, 0.0, 0.05), quat=(1.0, 0.0, 0.0, 0.0))
    ]

    # Hand placed far away from object
    result = validate_grasp_rollout(
        hand_xml_path=asset_path,
        collision_geoms=geoms,
        fingertip_body_names=["rh_ffdistal", "rh_mfdistal", "rh_rfdistal", "rh_lfdistal", "rh_thdistal"],
        palm_pos=(0.5, 0.5, 0.5), # Far away
        object_pos=(0.0, 0.0, 0.05),
        squeeze_steps=10,
        lift_steps=10,
        perturbation_steps=5
    )

    assert not result.passed
    assert result.failure_stage == "active_contact"


def test_zero_actuated_damping_fails_before_mujoco_step(tmp_path, monkeypatch):
    xml_path = tmp_path / "zero_damping_hand.xml"
    xml_path.write_text(
        """
<mujoco model="zero_damping_hand">
  <worldbody>
    <body name="hand_root">
      <geom type="sphere" size="0.005" mass="0.01" contype="0" conaffinity="0"/>
      <body name="palm">
        <joint name="finger_joint" type="hinge" damping="0"/>
        <geom type="sphere" size="0.01"/>
        <body name="tip" pos="0 0 -0.03">
          <geom type="sphere" size="0.005"/>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator><position name="finger_actuator" joint="finger_joint" kp="10"/></actuator>
</mujoco>
""".strip(),
        encoding="utf-8",
    )
    step_calls = 0
    original_step = mujoco.mj_step

    def counted_step(model, data):
        nonlocal step_calls
        step_calls += 1
        return original_step(model, data)

    monkeypatch.setattr(mujoco, "mj_step", counted_step)
    result = validate_grasp_rollout(
        xml_path,
        [SubGeomSpec(type="box", size=(0.01, 0.01, 0.01))],
        ["tip"],
        palm_pos=(0.5, 0.5, 0.5),
        object_pos=(0.0, 0.0, 0.01),
        squeeze_steps=1,
        lift_steps=1,
        perturbation_steps=1,
    )

    assert not result.passed
    assert result.failure_stage == "controller_protocol"
    assert result.trajectory_metrics["protocol_error"] == "nonpositive_actuated_damping"
    assert step_calls == 0


def test_shadow_underactuated_joint_targets_fail_before_rollout(monkeypatch):
    step_calls = 0
    original_step = mujoco.mj_step

    def counted_step(model, data):
        nonlocal step_calls
        step_calls += 1
        return original_step(model, data)

    monkeypatch.setattr(mujoco, "mj_step", counted_step)
    spec = RobotSpec.from_config("shadow_hand.yaml", sample_anchors=False)
    initial_targets = {name: 0.0 for name in spec.actuated_joint_names}
    # Create an uncontrollable null-space target: FFJ2 +0.5, FFJ1 -0.5 (tendon length unchanged)
    targets = dict(initial_targets)
    targets["rh_FFJ2"] = 0.5
    targets["rh_FFJ1"] = -0.5
    result = validate_grasp_rollout(
        resolve_robot_asset(spec.config.source_asset),
        [
            SubGeomSpec(
                type="box",
                size=(0.01, 0.01, 0.01),
                pos=(0.0, 0.0, 0.0),
                quat=(1.0, 0.0, 0.0, 0.0),
            )
        ],
        spec.fingertip_links,
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
    assert result.trajectory_metrics["joint_state_dimensions"] == 24.0
    assert result.trajectory_metrics["control_dimensions"] == 20.0
    assert result.trajectory_metrics["transmission_rank"] == 20.0
    assert result.trajectory_metrics["nullspace_residual"] > 0.1
    assert step_calls == 0


@pytest.mark.parametrize("active_count", [0, 1])
def test_task_command_with_too_few_active_fingers_fails_before_step(
    monkeypatch, active_count
):
    step_calls = 0
    original_step = mujoco.mj_step

    def counted_step(model, data):
        nonlocal step_calls
        step_calls += 1
        return original_step(model, data)

    monkeypatch.setattr(mujoco, "mj_step", counted_step)
    spec = RobotSpec.from_config("shadow_hand.yaml", sample_anchors=False)
    active = np.zeros(len(spec.fingertip_links), dtype=bool)
    active[:active_count] = True
    result = validate_grasp_rollout(
        resolve_robot_asset(spec.config.source_asset),
        [SubGeomSpec(type="box", size=(0.01, 0.01, 0.01))],
        spec.fingertip_links,
        palm_pos=(0.5, 0.5, 0.5),
        object_pos=(0.0, 0.0, 0.01),
        active_fingers=active,
        desired_fingertip_displacement=np.zeros(
            (len(spec.fingertip_links), 3), dtype=np.float64
        ),
        squeeze_steps=1,
        lift_steps=1,
        perturbation_steps=1,
    )

    assert not result.passed
    assert result.failure_stage == "insufficient_active_fingers"
    assert result.trajectory_metrics["active_finger_count"] == active_count
    assert step_calls == 0


def test_known_leap_pinch_lifts_box_without_teleportation():
    spec = RobotSpec.from_config("leap_hand.yaml", sample_anchors=False)
    q_contact = np.array(
        [
            0.5927356227, -0.3791691612, 0.6132688578, 1.692338131,
            0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0,
            1.228141244, 0.1354573565, -0.1336592733, 1.666422321,
        ],
        dtype=np.float32,
    )
    local_contacts = spec.fingertip_positions(
        torch.zeros(1, 3),
        torch.eye(3)[None],
        torch.from_numpy(q_contact[None]),
    )[0].numpy()
    pinch_axis = local_contacts[3] - local_contacts[0]
    pinch_axis /= np.linalg.norm(pinch_axis)
    palm_rotation, _ = Rotation.align_vectors(
        np.array([[-1.0, 0.0, 0.0]]), pinch_axis[None]
    )
    palm_rot = palm_rotation.as_matrix()
    pinch_center = 0.5 * (local_contacts[0] + local_contacts[3])
    half_width = 0.5 * np.linalg.norm(local_contacts[3] - local_contacts[0])
    object_pos = np.array([0.0, 0.0, 0.02])
    palm_pos = object_pos - palm_rot @ pinch_center

    palm_pos_b = palm_pos.astype(np.float32)[None]
    palm_rot_b = palm_rot.astype(np.float32)[None]
    q_b = q_contact[None]
    contact_points = spec.fingertip_positions(
        torch.from_numpy(palm_pos_b),
        torch.from_numpy(palm_rot_b),
        torch.from_numpy(q_b),
    )[0].numpy()
    contact_axes = spec.fingertip_contact_directions(
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
        spec,
        np.repeat(palm_pos_b, 2, axis=0),
        np.repeat(palm_rot_b, 2, axis=0),
        command_targets,
        np.repeat(contact_axes[None], 2, axis=0),
        init_q=np.repeat(q_b, 2, axis=0),
        max_iter=35,
        pos_tolerance=0.0007,
        normal_tolerance_dot=0.8,
        require_normal_alignment=False,
    )
    assert np.all(commands.converged)

    observed_stages = []
    result = validate_grasp_rollout(
        resolve_robot_asset(spec.config.source_asset),
        [
            SubGeomSpec(
                type="box",
                size=(float(half_width), 0.015, 0.02),
                pos=(0.0, 0.0, 0.0),
                quat=(1.0, 0.0, 0.0, 0.0),
            )
        ],
        spec.fingertip_links,
        palm_pos=tuple(palm_pos),
        palm_rot=palm_rot,
        initial_joint_targets=dict(zip(spec.actuated_joint_names, commands.q[0])),
        joint_targets=dict(zip(spec.actuated_joint_names, commands.q[1])),
        object_pos=tuple(object_pos),
        object_mass=0.02,
        expected_fingertip_positions=contact_points,
        fingertip_local_offsets=np.stack(
            [spec.fingertip_contact_offsets[name] for name in spec.fingertip_links]
        ),
        pregrasp_distance=0.0,
        squeeze_steps=300,
        stage_observer=lambda stage, _model, _data: observed_stages.append(stage),
    )

    assert result.passed
    assert result.failure_stage == "none"
    assert result.trajectory_metrics["final_active_fingers"] >= 2
    assert result.trajectory_metrics["max_penetration"] <= 0.002
    assert result.trajectory_metrics["floor_support"] == 0.0
    assert observed_stages == ["squeeze", "lift", "perturbation"]


def test_known_allegro_pinch_lifts_box_with_calibrated_disturbance():
    spec = RobotSpec.from_config("wonik_allegro.yaml", sample_anchors=False)
    q_contact = np.array(
        [
            -0.1410063654, 0.7589393854, 0.2905291915, 1.610496521,
            -0.1829498112, 0.7104878426, 0.4637212753, 0.6895720363,
            -0.3722456992, 0.4500102401, 1.241124988, 1.336122274,
            1.066359162, 0.5970826745, 0.1071554348, 1.677100062,
        ],
        dtype=np.float32,
    )
    local_contacts = spec.fingertip_positions(
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
    contact_points = spec.fingertip_positions(
        torch.from_numpy(palm_pos_b),
        torch.from_numpy(palm_rot_b),
        torch.from_numpy(q_b),
    )[0].numpy()
    contact_axes = spec.fingertip_contact_directions(
        torch.from_numpy(palm_pos_b),
        torch.from_numpy(palm_rot_b),
        torch.from_numpy(q_b),
    )[0].numpy()
    squeeze_contacts = contact_points.copy()
    squeeze_contacts[[0, 3]] += 0.0025 * contact_axes[[0, 3]]
    command = solve_dls_ik_batch(
        spec,
        palm_pos_b,
        palm_rot_b,
        squeeze_contacts,
        contact_axes[None],
        init_q=q_b,
        max_iter=35,
        pos_tolerance=0.0007,
        normal_tolerance_dot=0.8,
        require_normal_alignment=False,
    )
    assert command.converged[0]

    result = validate_grasp_rollout(
        resolve_robot_asset(spec.config.source_asset),
        [
            SubGeomSpec(
                type="box",
                size=(half_width, 0.015, 0.02),
                pos=(0.0, 0.0, 0.0),
                quat=(1.0, 0.0, 0.0, 0.0),
            )
        ],
        spec.fingertip_links,
        palm_pos=tuple(palm_pos),
        palm_rot=palm_rot,
        initial_joint_targets=spec.expand_mimic_joint_targets(
            dict(zip(spec.actuated_joint_names, q_contact))
        ),
        joint_targets=spec.expand_mimic_joint_targets(
            dict(zip(spec.actuated_joint_names, command.q[0]))
        ),
        object_pos=tuple(object_pos),
        object_mass=0.02,
        expected_fingertip_positions=contact_points,
        fingertip_local_offsets=np.stack(
            [spec.fingertip_contact_offsets[name] for name in spec.fingertip_links]
        ),
        pregrasp_distance=0.0,
        squeeze_steps=300,
        perturbation_wrench=np.array(
            [0.15, 0.15, 0.0, 0.01, 0.01, 0.01], dtype=np.float64
        ),
    )

    assert result.passed
    assert result.trajectory_metrics["final_active_fingers"] >= 2
    assert result.trajectory_metrics["floor_support"] == 0.0
    assert result.trajectory_metrics["max_penetration"] <= 0.002


def test_known_shadow_pinch_lifts_box_with_transmission_control():
    spec = RobotSpec.from_config("shadow_hand.yaml", sample_anchors=False)
    q_contact = np.zeros(len(spec.actuated_joint_names), dtype=np.float32)
    j_names = list(spec.actuated_joint_names)
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

    local_contacts = spec.fingertip_positions(
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

    contact_points = spec.fingertip_positions(
        torch.from_numpy(palm_pos_b),
        torch.from_numpy(palm_rot_b),
        torch.from_numpy(q_contact[None]),
    )[0].numpy()

    observed_stages = []
    result = validate_grasp_rollout(
        resolve_robot_asset(spec.config.source_asset),
        [
            SubGeomSpec(
                type="box",
                size=(float(half_width), 0.015, 0.02),
                pos=(0.0, 0.0, 0.0),
                quat=(1.0, 0.0, 0.0, 0.0),
            )
        ],
        spec.fingertip_links,
        palm_pos=tuple(palm_pos),
        palm_rot=palm_rot,
        initial_joint_targets=dict(zip(spec.actuated_joint_names, q_open)),
        joint_targets=dict(zip(spec.actuated_joint_names, q_squeeze)),
        object_pos=tuple(object_pos),
        object_mass=0.02,
        expected_fingertip_positions=contact_points,
        fingertip_local_offsets=np.stack(
            [spec.fingertip_contact_offsets[name] for name in spec.fingertip_links]
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
    assert result.trajectory_metrics["final_active_fingers"] >= 2
    assert result.trajectory_metrics["floor_support"] == 0.0
    assert result.trajectory_metrics["max_penetration"] <= 0.002
    assert result.trajectory_metrics["transmission_rank"] == 20.0
    assert result.trajectory_metrics["joint_state_dimensions"] == 24.0
    assert result.trajectory_metrics["control_dimensions"] == 20.0
    assert result.trajectory_metrics["lift_achieved"] >= 0.025
