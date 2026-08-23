import pytest
import numpy as np
import mujoco

from qdgrasp.scenes.contracts import SceneSpec, CameraSpec
from qdgrasp.scenes.environments import get_environment
from qdgrasp.scenes.builders.base import build_base_mujoco_model
from qdgrasp.scenes.observations.renderer import render_camera_view

def test_render_camera_view():
    cam = CameraSpec(
        camera_id="cam_1",
        intrinsics=np.eye(3),
        T_world_camera=np.eye(4)
    )

    spec = SceneSpec(
        scene_id="mock_scene",
        source_dataset="native",
        source_version="1.0",
        source_split="train",
        environment="table",
        objects=[],
        supports=get_environment("table"),
        cameras=[cam]
    )

    model = build_base_mujoco_model(spec)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    try:
        obs = render_camera_view(model, data, "cam_1", width=160, height=120)
        assert "rgb" in obs
        assert "depth" in obs
        assert "segmentation" in obs

        assert obs["rgb"].shape == (120, 160, 3)
        assert obs["depth"].shape == (120, 160)
        assert obs["segmentation"].shape == (120, 160, 2)
    except mujoco.FatalError as e:
        # Skip if headless machine lacks GL context for renderer
        pytest.skip(f"GL context unavailable for rendering: {e}")
