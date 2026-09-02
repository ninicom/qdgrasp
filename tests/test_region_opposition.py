import numpy as np
import pytest
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
    assert np.array_equal(proposal1.active_fingers, proposal2.active_fingers)
    assert proposal1.candidate_id == proposal2.candidate_id

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

    assert proposal.active_fingers is not None
    assert proposal.active_fingers[thumb_idx]
    assert int(proposal.active_fingers.sum()) == 2
    active_faces = proposal.face_ids[proposal.active_fingers]
    assert len(np.unique(active_faces)) == len(active_faces)
    active_centers = dummy_mesh.triangles.mean(axis=1)[active_faces]
    active_outward = dummy_mesh.face_normals[active_faces]
    assert not np.any(
        (active_centers[:, 2] <= dummy_mesh.bounds[0, 2] + 1e-8)
        & (active_outward[:, 2] < 0.0)
    )
    assert proposal.opposition_pairs is not None
    assert np.all(proposal.opposition_pairs[:, 0] == thumb_idx)


def test_shadow_sized_proposal_uses_the_same_two_group_task(dummy_mesh):
    proposal = generate_region_opposition_proposal(
        dummy_mesh,
        5,
        np.random.default_rng(9),
        np.arange(5),
        thumb_index=4,
    )
    assert proposal.active_fingers is not None
    assert proposal.active_fingers[4]
    assert int(proposal.active_fingers.sum()) == 2
    assert len(np.unique(proposal.face_ids[proposal.active_fingers])) == 2


def test_region_sampler_can_assign_each_non_thumb_opposition_group(dummy_mesh):
    for opposing_finger in (0, 1, 2, 3):
        proposal = generate_region_opposition_proposal(
            dummy_mesh,
            5,
            np.random.default_rng(20 + opposing_finger),
            np.arange(5),
            thumb_index=4,
            opposing_finger_index=opposing_finger,
        )
        assert np.flatnonzero(proposal.active_fingers).tolist() == [
            opposing_finger,
            4,
        ]
        assert proposal.opposition_pairs.tolist() == [[4, opposing_finger]]

def test_region_opposition_fails_closed(dummy_mesh):
    """
    Test that it falls back gracefully if opposition is impossible (e.g. 1 finger).
    """
    rng = np.random.default_rng(42)
    finger_ids = np.array([0])

    with pytest.raises(ValueError, match="at least two fingers"):
        generate_region_opposition_proposal(dummy_mesh, 1, rng, finger_ids)
