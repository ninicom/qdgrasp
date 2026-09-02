import numpy as np

from qdgrasp.dataset.pipeline.certifiers.grasp_wrench import compute_grasp_wrench_space_quality


def test_gws_balanced_cube():
    """Test that opposing contacts with friction form a positive GWS epsilon."""
    # 4 contacts on 4 sides of a cube
    target_points = np.array([
        [0.05, 0.0, 0.0],
        [-0.05, 0.0, 0.0],
        [0.0, 0.05, 0.0],
        [0.0, -0.05, 0.0]
    ])
    inward_normals = np.array([
        [-1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 1.0, 0.0]
    ])
    centroid = np.array([0.0, 0.0, 0.0])

    cert = compute_grasp_wrench_space_quality(target_points, inward_normals, centroid, mu=0.5)
    assert cert.passed, "4 opposing contacts with friction should span the 6D wrench space"
    assert cert.quality_margin > 0.0

def test_gws_unbalanced():
    """Test that all contacts pointing in the same direction fail GWS."""
    target_points = np.array([
        [0.05, 0.02, 0.0],
        [0.05, -0.02, 0.0],
        [0.05, 0.0, 0.02],
        [0.05, 0.0, -0.02]
    ])
    inward_normals = np.array([
        [-1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0]
    ])
    centroid = np.array([0.0, 0.0, 0.0])

    cert = compute_grasp_wrench_space_quality(target_points, inward_normals, centroid, mu=0.5)
    assert not cert.passed, "Contacts all on one side cannot resist forces in opposite directions"
    assert cert.quality_margin == 0.0
