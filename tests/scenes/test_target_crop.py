import numpy as np
import pytest

from qdgrasp.config.schema import ConfigError
from qdgrasp.scenes.target_crop import build_target_scene_crop


def test_target_crop_unprojects_to_object_frame_and_keeps_context_deterministically():
    depth = np.zeros((3, 3), dtype=np.float32)
    depth[1, 1:] = 1.0
    mask = np.array([[0, 0, 0], [0, 1, 2], [0, 0, 0]], dtype=np.uint16)
    intrinsics = np.array([[10.0, 0.0, 1.0], [0.0, 10.0, 1.0], [0.0, 0.0, 1.0]])
    world_object = np.eye(4)
    world_object[2, 3] = 1.0
    crop = build_target_scene_crop(
        depth,
        mask,
        target_label=1,
        intrinsics=intrinsics,
        T_world_camera=np.eye(4),
        T_world_object=world_object,
        context_radius=0.11,
    )
    np.testing.assert_allclose(crop.points_object_frame, [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]], atol=1e-7)
    np.testing.assert_array_equal(crop.target_point_mask, [True, False])
    np.testing.assert_array_equal(crop.source_pixel_indices, [[1, 1], [1, 2]])

    dense_depth = np.ones((3, 3), dtype=np.float32)
    limited = build_target_scene_crop(
        dense_depth,
        mask,
        target_label=1,
        intrinsics=intrinsics,
        T_world_camera=np.eye(4),
        T_world_object=world_object,
        context_radius=1.0,
        max_points=2,
    )
    repeated = build_target_scene_crop(
        dense_depth,
        mask,
        target_label=1,
        intrinsics=intrinsics,
        T_world_camera=np.eye(4),
        T_world_object=world_object,
        context_radius=1.0,
        max_points=2,
    )
    np.testing.assert_array_equal(limited.points_object_frame, repeated.points_object_frame)
    assert np.any(limited.target_point_mask)


def test_target_crop_fails_closed_for_missing_target_and_invalid_depth():
    intrinsics = np.eye(3)
    with pytest.raises(ConfigError, match="target label is absent"):
        build_target_scene_crop(
            np.ones((2, 2)),
            np.zeros((2, 2)),
            target_label=1,
            intrinsics=intrinsics,
            T_world_camera=np.eye(4),
            T_world_object=np.eye(4),
            context_radius=1.0,
        )
    with pytest.raises(ConfigError, match="no valid depth"):
        build_target_scene_crop(
            np.zeros((2, 2)),
            np.ones((2, 2)),
            target_label=1,
            intrinsics=intrinsics,
            T_world_camera=np.eye(4),
            T_world_object=np.eye(4),
            context_radius=1.0,
        )
