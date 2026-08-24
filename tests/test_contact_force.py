import numpy as np
from qdgrasp.dataset.pipeline.certifiers.contact_force import certify_force_closure

def test_contact_force_balanced():
    """
    Test a perfectly balanced cube with 2 opposing fingers and a top and bottom finger.
    """
    # Points on +/- x and +/- y faces of a 10cm cube
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

    cert = certify_force_closure(target_points, inward_normals, centroid, mass=0.1) # 100g
    assert cert.passed, "Opposing contacts should easily achieve force closure to resist gravity"

def test_contact_force_unbalanced():
    """
    Test fingers all on one side, which cannot resist gravity pushing them.
    """
    # Both points on the same face (+x)
    target_points = np.array([
        [0.05, 0.02, 0.0],
        [0.05, -0.02, 0.0]
    ])

    inward_normals = np.array([
        [-1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0]
    ])

    centroid = np.array([0.0, 0.0, 0.0])

    # Gravity is -z, but even with friction, we can't squeeze without opposing forces
    cert = certify_force_closure(target_points, inward_normals, centroid, mass=1.0)
    assert not cert.passed, "Contacts all on one side should not be able to achieve force closure"


def test_single_support_contact_is_not_force_closure():
    cert = certify_force_closure(
        target_points=np.array([[0.0, 0.0, -0.05]]),
        inward_normals=np.array([[0.0, 0.0, 1.0]]),
        centroid=np.zeros(3),
        mass=0.1,
    )

    assert not cert.passed
    assert cert.quality_margin == 0.0


def test_antipodal_soft_finger_pinch_matches_condim4_rollout_model():
    cert = certify_force_closure(
        target_points=np.array([[0.04, 0.0, 0.0], [-0.04, 0.0, 0.0]]),
        inward_normals=np.array([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        centroid=np.zeros(3),
        mass=0.02,
        torsional_friction=0.005,
    )
    assert cert.passed
    assert cert.quality_margin > 0.0
