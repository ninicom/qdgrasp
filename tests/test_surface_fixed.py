import pytest
import numpy as np
import trimesh
from qdgrasp.dataset.pipeline.proposals.surface_fixed import generate_surface_fixed_proposal

@pytest.fixture
def dummy_mesh():
    # A simple box mesh
    return trimesh.creation.box(extents=(0.1, 0.1, 0.1))

def test_surface_fixed_deterministic(dummy_mesh):
    rng1 = np.random.default_rng(42)
    finger_ids = np.array([0, 1, 2, 3])

    proposal1 = generate_surface_fixed_proposal(dummy_mesh, 4, rng1, finger_ids)

    rng2 = np.random.default_rng(42)
    proposal2 = generate_surface_fixed_proposal(dummy_mesh, 4, rng2, finger_ids)

    np.testing.assert_allclose(proposal1.target_points, proposal2.target_points)
    np.testing.assert_allclose(proposal1.inward_normals, proposal2.inward_normals)
    assert np.array_equal(proposal1.face_ids, proposal2.face_ids)

def test_surface_fixed_no_trimesh_sample(dummy_mesh, monkeypatch):
    """Ensure it does not rely on trimesh.sample.sample_surface."""
    def fake_sample(*args, **kwargs):
        raise RuntimeError("Should not be called")

    monkeypatch.setattr(trimesh.sample, "sample_surface", fake_sample)

    rng = np.random.default_rng(0)
    finger_ids = np.array([0, 1, 2])

    # Should not raise RuntimeError
    proposal = generate_surface_fixed_proposal(dummy_mesh, 3, rng, finger_ids)
    assert len(proposal.target_points) == 3

def test_surface_fixed_points_on_surface_and_normals_unit_length(dummy_mesh):
    rng = np.random.default_rng(99)
    finger_ids = np.array([0, 1, 2, 3])

    proposal = generate_surface_fixed_proposal(dummy_mesh, 4, rng, finger_ids)

    # Test unit length normals
    norms = np.linalg.norm(proposal.inward_normals, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-6)

    # Test that normals are inward (opposite of outward face normals)
    outward_normals = dummy_mesh.face_normals[proposal.face_ids]
    # Dot product should be -1
    dots = np.sum(proposal.inward_normals * outward_normals, axis=1)
    np.testing.assert_allclose(dots, -1.0, atol=1e-6)
