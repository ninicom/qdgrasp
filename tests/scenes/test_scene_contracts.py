import numpy as np
from qdgrasp.scenes.contracts import (
    SceneSpec,
    SceneObjectSpec,
    CameraSpec,
    SupportGeometrySpec,
)


def test_scene_spec_initialization():
    obj = SceneObjectSpec(
        object_id="obj_1",
        asset_ref="mesh.obj",
        T_world_object=np.eye(4),
        mass=1.0,
    )

    cam = CameraSpec(
        camera_id="cam_1",
        intrinsics=np.eye(3),
    )

    sup = SupportGeometrySpec(
        support_id="table",
        geom_type="box",
        params={"size": [1.0, 1.0, 0.1]},
        T_world_support=np.eye(4),
    )

    spec = SceneSpec(
        scene_id="test_scene",
        source_dataset="native",
        source_version="1.0",
        source_split="train",
        environment="table",
        objects=[obj],
        supports=[sup],
        cameras=[cam],
    )

    assert spec.scene_id == "test_scene"
    assert len(spec.objects) == 1
    assert spec.objects[0].mass == 1.0
    assert spec.environment == "table"
    assert spec.gravity == (0.0, 0.0, -9.81)
