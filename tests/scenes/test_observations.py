from pathlib import Path

import mujoco
import numpy as np
import pytest
from PIL import Image

from qdgrasp.config.schema import ConfigError
from qdgrasp.dataset.scene_shards import write_scene_shard
from qdgrasp.scenes.builders.base import build_scene_mujoco_model
from qdgrasp.scenes.contracts import CameraSpec, SceneObjectSpec, SceneSpec
from qdgrasp.scenes.observations import renderer as renderer_module
from qdgrasp.scenes.observations.renderer import (
    build_scene_observation,
    canonical_instance_mask,
    render_camera_view,
    scene_observation_record,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
OBJECT_MANIFEST = REPO_ROOT / "datasets/dgn-open-tiny/objects/prim_box_01.manifest.json"


def _scene(*, width=160, height=120, with_object=True):
    camera_transform = np.eye(4)
    camera_transform[2, 3] = 0.5
    camera = CameraSpec(
        camera_id="cam_1",
        intrinsics=np.array([[120.0, 0.0, width / 2.0], [0.0, 120.0, height / 2.0], [0.0, 0.0, 1.0]]),
        T_world_camera=camera_transform,
    )
    objects = (
        [
            SceneObjectSpec(
                object_id="prim_box_01",
                asset_ref=str(OBJECT_MANIFEST),
                T_world_object=np.eye(4),
            )
        ]
        if with_object
        else []
    )
    return SceneSpec(
        scene_id="render-fixture",
        source_dataset="native",
        source_version="1.0",
        source_split="train",
        environment="custom",
        objects=objects,
        cameras=[camera],
    )


def test_canonical_instance_mask_maps_geom_to_object_and_excludes_background():
    spec = _scene()
    model = build_scene_mujoco_model(spec, dynamic_objects=False)
    geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "prim_box_01::geom::0")
    segmentation = np.full((2, 3, 2), -1, dtype=np.int32)
    segmentation[0, :2, 0] = geom_id
    segmentation[0, :2, 1] = int(mujoco.mjtObj.mjOBJ_GEOM)
    mask, visibility = canonical_instance_mask(model, segmentation, ["prim_box_01"])
    np.testing.assert_array_equal(mask, [[1, 1, 0], [0, 0, 0]])
    assert visibility == {"prim_box_01": pytest.approx(2 / 6)}


def test_render_camera_view_returns_canonical_channels():
    spec = _scene(with_object=False)
    model = build_scene_mujoco_model(spec)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    try:
        observation = render_camera_view(model, data, "cam_1", width=160, height=120)
    except (mujoco.FatalError, RuntimeError) as exc:
        pytest.skip(f"GL context unavailable for rendering: {exc}")
    assert observation["rgb"].shape == (120, 160, 3)
    assert observation["depth"].shape == (120, 160)
    assert observation["segmentation"].shape == (120, 160, 2)
    assert observation["instance_mask"].shape == (120, 160)
    assert observation["visibility"] == {}


def test_build_scene_observation_packs_references_and_visibility(tmp_path, monkeypatch):
    spec = _scene(width=4, height=2)
    model = build_scene_mujoco_model(spec, dynamic_objects=False)
    data = mujoco.MjData(model)
    rendered = {
        "rgb": np.zeros((2, 4, 3), dtype=np.uint8),
        "depth": np.full((2, 4), 0.25, dtype=np.float32),
        "segmentation": np.full((2, 4, 2), -1, dtype=np.int32),
        "instance_mask": np.array([[1, 1, 0, 0], [0, 0, 0, 0]], dtype=np.uint16),
        "visibility": {"prim_box_01": 0.25},
    }
    monkeypatch.setattr(renderer_module, "render_camera_view", lambda *args, **kwargs: rendered)
    observation = build_scene_observation(spec, model, data, "cam_1", "0000", tmp_path, width=4, height=2)
    assert observation.visibility_by_object == {"prim_box_01": 0.25}
    assert np.load(tmp_path / observation.depth_ref, allow_pickle=False).shape == (2, 4)
    assert (
        np.asarray(Image.open(tmp_path / observation.instance_mask_ref)).tolist() == rendered["instance_mask"].tolist()
    )
    assert (tmp_path / observation.rgb_ref).is_file()
    assert len(observation.calibration_hash) == 64
    record = scene_observation_record(observation)
    assert len(write_scene_shard([record], tmp_path / "observation.jsonl", record_type="observation")) == 64


def test_observation_builder_rejects_unrepresentable_intrinsics_and_unsafe_paths(tmp_path):
    spec = _scene(width=4, height=2)
    spec.cameras[0].intrinsics[0, 2] = 0.0
    model = build_scene_mujoco_model(spec, dynamic_objects=False)
    data = mujoco.MjData(model)
    with pytest.raises(ConfigError, match="not representable"):
        build_scene_observation(spec, model, data, "cam_1", "0000", tmp_path, width=4, height=2)
    with pytest.raises(ConfigError, match="unsafe frame_id"):
        build_scene_observation(spec, model, data, "cam_1", "../escape", tmp_path, width=4, height=2)
    with pytest.raises(ConfigError, match="unsafe frame_id"):
        build_scene_observation(spec, model, data, "cam_1", "..", tmp_path, width=4, height=2)
