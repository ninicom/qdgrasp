"""Unit tests for robot transmission layer: direct-drive and fixed-tendon underactuated hands."""

from pathlib import Path
import numpy as np
import pytest
import mujoco

from qdgrasp.robot.assets import resolve_robot_asset
from qdgrasp.robot.spec import RobotSpec
from qdgrasp.robot.transmission import (
    ActuatorCommand,
    DirectJointTransmission,
    FixedTendonTransmission,
    TransmissionState,
    compute_finite_difference_moment_matrix,
    create_transmission_model_from_spec_and_mjcf,
    project_joint_delta_to_actuator_command,
)


@pytest.fixture
def leap_hand():
    spec = RobotSpec.from_config("qdgrasp/presets/robots/leap_hand.yaml")
    xml_path = resolve_robot_asset("asset://mujoco-menagerie/leap_hand/right_hand.xml")
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    return spec, model


@pytest.fixture
def allegro_hand():
    spec = RobotSpec.from_config("qdgrasp/presets/robots/wonik_allegro.yaml")
    xml_path = resolve_robot_asset("asset://mujoco-menagerie/wonik_allegro/right_hand.xml")
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    return spec, model


@pytest.fixture
def shadow_hand():
    spec = RobotSpec.from_config("qdgrasp/presets/robots/shadow_hand.yaml")
    xml_path = resolve_robot_asset("asset://mujoco-menagerie/shadow_hand/right_hand.xml")
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    return spec, model


def test_leap_direct_transmission(leap_hand):
    spec, model = leap_hand
    tm = create_transmission_model_from_spec_and_mjcf(spec, model)
    assert isinstance(tm, DirectJointTransmission)
    assert tm.num_joints == 16
    assert tm.num_actuators == 16
    assert tm.rank == 16

    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    state = tm.extract_state(model, data)
    assert state.joint_position.shape == (16,)
    assert state.actuator_coordinate.shape == (16,)
    assert state.moment_matrix.shape == (16, 16)


def test_allegro_direct_transmission(allegro_hand):
    spec, model = allegro_hand
    tm = create_transmission_model_from_spec_and_mjcf(spec, model)
    assert isinstance(tm, DirectJointTransmission)
    assert tm.num_joints == 16
    assert tm.num_actuators == 16
    assert tm.rank == 16


def test_shadow_fixed_tendon_transmission(shadow_hand):
    spec, model = shadow_hand
    tm = create_transmission_model_from_spec_and_mjcf(spec, model)
    assert isinstance(tm, FixedTendonTransmission)
    assert tm.num_joints == 24
    assert tm.num_actuators == 20
    assert tm.rank == 20

    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    state = tm.extract_state(model, data)
    assert state.joint_position.shape == (24,)
    assert state.actuator_coordinate.shape == (20,)
    assert state.moment_matrix.shape == (20, 24)

    # Verify finite difference parity
    assert tm.verify_finite_difference(eps=1e-6, atol=1e-5)


def test_command_projection_direct_vs_nullspace(shadow_hand):
    spec, model = shadow_hand
    tm = create_transmission_model_from_spec_and_mjcf(spec, model)
    M = tm.moment_matrix  # [20, 24]

    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    state = tm.extract_state(model, data)

    # 1. Fully controllable delta (e.g. wrist and thumb motion)
    dq_controllable = np.zeros(24, dtype=np.float64)
    dq_controllable[0] = 0.05  # rh_WRJ2
    dq_controllable[2] = 0.05  # rh_THJ5

    cmd1 = tm.project_joint_delta(dq_controllable, state)
    assert cmd1.controllable_residual < 1e-5
    assert cmd1.nullspace_residual < 1e-5
    assert cmd1.reason == "converged"

    # 2. Pure nullspace delta: FFJ2 +delta, FFJ1 -delta
    # Indices in spec: rh_FFJ2 is index 9, rh_FFJ1 is index 10
    j_ffj2 = spec.actuated_joint_names.index("rh_FFJ2")
    j_ffj1 = spec.actuated_joint_names.index("rh_FFJ1")

    dq_null = np.zeros(24, dtype=np.float64)
    dq_null[j_ffj2] = 0.1
    dq_null[j_ffj1] = -0.1

    # Actuator delta for rh_FFJ0 should be 0 because dl = dq[FFJ2] + dq[FFJ1] = 0
    cmd2 = tm.project_joint_delta(dq_null, state, max_nullspace_residual=0.01)
    assert cmd2.nullspace_residual > 0.1
    assert cmd2.reason == "nullspace_rejection"


def test_batch_command_projection(shadow_hand):
    spec, model = shadow_hand
    tm = create_transmission_model_from_spec_and_mjcf(spec, model)

    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    state = tm.extract_state(model, data)

    j_ffj2 = spec.actuated_joint_names.index("rh_FFJ2")
    j_ffj1 = spec.actuated_joint_names.index("rh_FFJ1")

    dq_batch = np.zeros((2, 24), dtype=np.float64)
    dq_batch[0, 0] = 0.02  # Controllable
    dq_batch[1, j_ffj2] = 0.1  # Nullspace
    dq_batch[1, j_ffj1] = -0.1

    cmd_batch = tm.project_joint_delta(dq_batch, state, max_nullspace_residual=0.01)
    assert cmd_batch.control_target.shape == (2, 20)
    assert cmd_batch.nullspace_residual[0] < 1e-5
    assert cmd_batch.nullspace_residual[1] > 0.1
    assert cmd_batch.reason[0] == "converged"
    assert cmd_batch.reason[1] == "nullspace_rejection"
