import pytest
import numpy as np
import mujoco
from pathlib import Path

from qdgrasp.objects.schema import SubGeomSpec
from qdgrasp.dataset.pipeline.validators.mujoco_rollout import (
    build_rollout_scene_model,
    validate_grasp_rollout,
    smoothstep,
)
from qdgrasp.dataset.pipeline.observers.contact_load import extract_contact_loads
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
    assert result.failure_stage == "squeeze"
