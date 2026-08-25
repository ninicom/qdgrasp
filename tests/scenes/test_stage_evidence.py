import mujoco
import numpy as np
from PIL import Image

from qdgrasp.scenes.observations.stage_evidence import capture_stage_evidence


def test_stage_evidence_renders_exact_mjdata_and_hashes_overlay(tmp_path):
    model = mujoco.MjModel.from_xml_string(
        """
        <mujoco>
          <worldbody>
            <body name="target_object" pos="0 0 0.03">
              <geom name="target_geom" type="box" size=".02 .02 .03"/>
            </body>
            <body name="tip" pos=".04 0 .04">
              <geom name="tip_geom" type="sphere" size=".015"/>
            </body>
          </worldbody>
        </mujoco>
        """
    )
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    path = np.repeat(np.eye(4)[None], 2, axis=0)
    path[:, 2, 3] = [0.1, 0.04]
    record = capture_stage_evidence(
        model,
        data,
        tmp_path,
        scene_id="scene-1",
        robot_profile="leap",
        stage="pregrasp",
        target_object_id="target",
        fingertip_body_names=["tip"],
        active_fingers=[True],
        approach_path=path,
        failure_reason="none",
        width=160,
        height=120,
    )
    image_path = tmp_path / record["image_ref"]
    assert Image.open(image_path).size == (160, 120)
    assert len(record["image_sha256"]) == 64
    assert record["active_fingers"] == ["tip"]
    assert record["approach_waypoint_count"] == 2
