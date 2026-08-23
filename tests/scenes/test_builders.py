import numpy as np
from qdgrasp.scenes.contracts import SceneSpec
from qdgrasp.scenes.environments import get_environment
from qdgrasp.scenes.builders.base import build_base_mujoco_model
from qdgrasp.scenes.builders.replay_imported import build_replay_scene
from qdgrasp.scenes.builders.drop_and_settle import drop_and_settle_scene
from qdgrasp.scenes.builders.pose_compose import compose_scene

def create_mock_spec():
    supports = get_environment("table")
    return SceneSpec(
        scene_id="mock_scene",
        source_dataset="native",
        source_version="1.0",
        source_split="train",
        environment="table",
        objects=[],
        supports=supports
    )

def test_build_base_mujoco_model():
    spec = create_mock_spec()
    model = build_base_mujoco_model(spec)

    assert model is not None
    # table_surface body should exist
    assert mujoco_has_body(model, "table_surface")

def mujoco_has_body(model, name):
    try:
        idx = model.body(name).id
        return idx >= 0
    except KeyError:
        return False

def test_build_replay_scene():
    spec = create_mock_spec()
    model, data = build_replay_scene(spec)
    assert data is not None

def test_drop_and_settle():
    spec = create_mock_spec()
    model, data = drop_and_settle_scene(spec, max_steps=10)
    assert data is not None

def test_pose_compose():
    spec = create_mock_spec()
    model, data = compose_scene(spec)
    assert data is not None
