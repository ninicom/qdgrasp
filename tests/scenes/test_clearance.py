import mujoco
import numpy as np
import pytest

from qdgrasp.scenes.clearance import ClearanceError, check_approach_clearance

SCENE_XML = """
<mujoco>
  <worldbody>
    <body name="hand" pos="-0.3 0 0">
      <freejoint name="hand_root"/>
      <geom name="hand_geom" type="sphere" size="0.04"/>
    </body>
    <body name="target" pos="0.3 0 0">
      <geom name="target_geom" type="sphere" size="0.05"/>
    </body>
    <body name="obstacle" pos="0 0 0">
      <geom name="obstacle_geom" type="sphere" size="0.05"/>
    </body>
  </worldbody>
</mujoco>
"""


def _scene():
    model = mujoco.MjModel.from_xml_string(SCENE_XML)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    hand_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "hand_geom")
    return model, data, [hand_geom]


def _pose(x: float, y: float = 0.0) -> np.ndarray:
    transform = np.eye(4)
    transform[:3, 3] = [x, y, 0.0]
    return transform


def test_empty_approach_path():
    model, data, hand_geoms = _scene()
    with pytest.raises(ClearanceError, match="approach_blocked"):
        check_approach_clearance(model, data, "target", np.array([]), hand_geoms)


def test_clear_path_passes():
    model, data, hand_geoms = _scene()
    path = np.array([_pose(-0.3, 0.2), _pose(0.2, 0.2)])
    assert check_approach_clearance(model, data, "target", path, hand_geoms)


def test_hand_self_contact_is_not_classified_as_scene_collision():
    model = mujoco.MjModel.from_xml_string(
        """
        <mujoco>
          <worldbody>
            <body name="hand" pos="-0.3 0 0">
              <freejoint name="hand_root"/>
              <geom name="hand_a" type="sphere" size="0.04"/>
              <body name="finger"><geom name="hand_b" type="sphere" size="0.04"/></body>
            </body>
            <body name="target" pos="0.3 0 0">
              <geom name="target_geom" type="sphere" size="0.05"/>
            </body>
          </worldbody>
          <contact><pair geom1="hand_a" geom2="hand_b"/></contact>
        </mujoco>
        """
    )
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    hand_geoms = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name) for name in ("hand_a", "hand_b")]
    assert check_approach_clearance(model, data, "target", np.array([_pose(-0.3)]), hand_geoms)


def test_sweep_rejects_intermediate_obstacle_missed_by_endpoints():
    model, data, hand_geoms = _scene()
    path = np.array([_pose(-0.2), _pose(0.2)])
    with pytest.raises(ClearanceError) as caught:
        check_approach_clearance(model, data, "target", path, hand_geoms, max_translation_step=0.01)
    assert caught.value.reason == "hand_scene_collision"
    assert caught.value.telemetry["other_geom"] == "obstacle_geom"
    assert 0.0 < caught.value.telemetry["path_progress"] < 1.0


def test_target_contact_is_allowed_only_at_goal():
    model, data, hand_geoms = _scene()
    terminal_contact = np.array([_pose(0.15), _pose(0.21)])
    assert check_approach_clearance(model, data, "target", terminal_contact, hand_geoms)

    crosses_target = np.array([_pose(0.15), _pose(0.39)])
    with pytest.raises(ClearanceError) as caught:
        check_approach_clearance(model, data, "target", crosses_target, hand_geoms)
    assert caught.value.reason == "approach_blocked"
    assert caught.value.telemetry["other_kind"] == "target"


def test_data_state_is_restored_after_rejection():
    model, data, hand_geoms = _scene()
    initial_qpos = data.qpos.copy()
    initial_time = data.time
    with pytest.raises(ClearanceError):
        check_approach_clearance(model, data, "target", np.array([_pose(-0.2), _pose(0.2)]), hand_geoms)
    np.testing.assert_allclose(data.qpos, initial_qpos)
    assert data.time == initial_time


def test_missing_target_and_invalid_hand_geoms_fail_closed():
    model, data, hand_geoms = _scene()
    with pytest.raises(ClearanceError, match="source_frame_invalid"):
        check_approach_clearance(model, data, "missing", np.array([_pose(-0.3)]), hand_geoms)
    with pytest.raises(ClearanceError, match="source_frame_invalid"):
        check_approach_clearance(model, data, "target", np.array([_pose(-0.3)]), [])
