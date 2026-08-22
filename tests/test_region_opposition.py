import pytest
import numpy as np
import trimesh
from qdgrasp.dataset.pipeline.proposals.region_opposition import generate_region_opposition_proposal

@pytest.fixture
def dummy_mesh():
    # A simple box mesh
    return trimesh.creation.box(extents=(0.1, 0.1, 0.1))

def test_region_opposition_deterministic(dummy_mesh):
    rng1 = np.random.default_rng(42)
    finger_ids = np.array([0, 1, 2, 3])
    
    proposal1 = generate_region_opposition_proposal(dummy_mesh, 4, rng1, finger_ids)
    
    rng2 = np.random.default_rng(42)
    proposal2 = generate_region_opposition_proposal(dummy_mesh, 4, rng2, finger_ids)
    
    np.testing.assert_allclose(proposal1.target_points, proposal2.target_points)
    np.testing.assert_allclose(proposal1.inward_normals, proposal2.inward_normals)
    assert np.array_equal(proposal1.face_ids, proposal2.face_ids)

def test_region_opposition_thumbs_vs_others(dummy_mesh):
    """
    Ensure that the generated thumb region and opposing regions actually oppose.
    This means the dot product of their inward normals should be < 0 on average.
    """
    rng = np.random.default_rng(42)
    finger_ids = np.array([0, 1, 2, 3])
    thumb_idx = 0
    
    proposal = generate_region_opposition_proposal(dummy_mesh, 4, rng, finger_ids, thumb_index=thumb_idx)
    
    thumb_normal = proposal.inward_normals[thumb_idx]
    
    # Calculate average normal of other fingers
    other_normals = np.delete(proposal.inward_normals, thumb_idx, axis=0)
    avg_other_normal = np.mean(other_normals, axis=0)
    
    # Check if they oppose
    dot_product = np.dot(thumb_normal, avg_other_normal)
    
    # Note: On a simple box, normal dot products might be -1, 0, or 1.
    # The heuristic looks for dot < -0.2, so it should be negative if a valid opposition was found.
    assert dot_product < 0, f"Expected opposition, dot product was {dot_product}"

def test_region_opposition_fallback(dummy_mesh):
    """
    Test that it falls back gracefully if opposition is impossible (e.g. 1 finger).
    """
    rng = np.random.default_rng(42)
    finger_ids = np.array([0])
    
    proposal = generate_region_opposition_proposal(dummy_mesh, 1, rng, finger_ids)
    assert len(proposal.target_points) == 1
    assert proposal.provenance == "surface_fixed"
