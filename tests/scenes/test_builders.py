from pathlib import Path

import mujoco
import numpy as np
import pytest

from qdgrasp.config.schema import ConfigError
from qdgrasp.scenes.builders.base import build_base_mujoco_model, build_scene_mujoco_model
from qdgrasp.scenes.builders.drop_and_settle import SettlingError, drop_and_settle_scene
from qdgrasp.scenes.builders.pose_compose import compose_scene
from qdgrasp.scenes.builders.replay_imported import build_replay_scene
from qdgrasp.scenes.contracts import SceneObjectSpec, SceneSpec
from qdgrasp.scenes.environments import get_environment

REPO_ROOT = Path(__file__).resolve().parents[2]
OBJECT_MANIFEST = REPO_ROOT / "datasets/dgn-open-tiny/objects/prim_box_01.manifest.json"


def create_spec(*, z=0.15, with_object=True):
    transform = np.eye(4)
    transform[2, 3] = z
    objects = (
        [
            SceneObjectSpec(
                object_id="prim_box_01",
                asset_ref=str(OBJECT_MANIFEST),
                T_world_object=transform,
            )
        ]
        if with_object
        else []
    )
    return SceneSpec(
        scene_id="builder-fixture",
        source_dataset="native",
        source_version="1.0",
        source_split="train",
        environment="table",
        objects=objects,
        supports=get_environment("table"),
    )


def test_base_model_contains_support_but_not_objects():
    model = build_base_mujoco_model(create_spec())
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "table_surface") >= 0
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "prim_box_01") < 0


def test_scene_model_loads_verified_collision_manifest():
    model = build_scene_mujoco_model(create_spec())
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "prim_box_01")
    geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "prim_box_01::geom::0")
    assert body_id >= 0
    assert geom_id >= 0
    assert int(model.body_jntnum[body_id]) == 1


def test_replay_preserves_canonical_pose_without_free_joint():
    model, data = build_replay_scene(create_spec(z=0.123))
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "prim_box_01")
    np.testing.assert_allclose(data.xpos[body_id], [0.0, 0.0, 0.123])
    assert int(model.body_jntnum[body_id]) == 0


def test_drop_and_settle_places_object_on_support():
    model, data = drop_and_settle_scene(create_spec(), max_steps=3000)
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "prim_box_01")
    expected_center_z = 0.02572892784643474
    assert data.xpos[body_id][2] == pytest.approx(expected_center_z, abs=0.002)
    assert np.linalg.norm(data.qvel) < 0.1


def test_settle_timeout_fails_closed_and_compose_uses_same_gate():
    with pytest.raises(SettlingError, match="settle_timeout"):
        drop_and_settle_scene(create_spec(z=1.0), max_steps=1)
    model, data = compose_scene(create_spec(z=0.08), repair_steps=1000)
    assert model.nbody > 1
    assert np.all(np.isfinite(data.qpos))


def test_asset_identity_mismatch_and_invalid_transform_fail_closed():
    spec = create_spec()
    bad_object = SceneObjectSpec(
        object_id="wrong-id",
        asset_ref=str(OBJECT_MANIFEST),
        T_world_object=np.eye(4),
    )
    with pytest.raises(ConfigError, match="does not match"):
        build_scene_mujoco_model(
            SceneSpec(
                scene_id=spec.scene_id,
                source_dataset=spec.source_dataset,
                source_version=spec.source_version,
                source_split=spec.source_split,
                environment=spec.environment,
                objects=[bad_object],
                supports=spec.supports,
            )
        )

    spec.objects[0].T_world_object[0, 0] = 2.0
    with pytest.raises(ConfigError, match="invalid rotation"):
        build_scene_mujoco_model(spec)
