from __future__ import annotations

import numpy as np
import trimesh

from qdgrasp.dataset.render import CameraModel, sample_analytic_point_cloud
from qdgrasp.dataset.rng import get_generator


def test_analytic_point_cloud_sampling_reproducibility() -> None:
    mesh = trimesh.creation.box(extents=(0.05, 0.05, 0.05))
    cam_pos = np.array([0.0, 0.0, 0.2])
    cam_rot = np.eye(3)

    rng1 = get_generator(42, "render_test")
    pcd1, meta1 = sample_analytic_point_cloud(mesh, cam_pos, cam_rot, num_points=256, rng=rng1)

    rng2 = get_generator(42, "render_test")
    pcd2, meta2 = sample_analytic_point_cloud(mesh, cam_pos, cam_rot, num_points=256, rng=rng2)

    np.testing.assert_array_equal(pcd1, pcd2)
    assert pcd1.shape == (256, 3)
    assert meta1["num_points"] == 256
    assert meta1["frame"] == "camera"


def test_camera_model_intrinsics_matrix() -> None:
    cam = CameraModel(fx=600.0, fy=600.0, cx=320.0, cy=240.0, width=640, height=480)
    K = cam.intrinsics_matrix
    assert K.shape == (3, 3)
    assert K[0, 0] == 600.0
    assert K[1, 1] == 600.0
    assert K[0, 2] == 320.0
    assert K[1, 2] == 240.0
