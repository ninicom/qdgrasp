import numpy as np
import pytest
import trimesh

from qdgrasp.dataset.pipeline.orchestrator import _opposing_finger_schedule
from qdgrasp.dataset.pipeline.proposals.wrench_guided import generate_wrench_guided_proposal


@pytest.fixture
def dummy_mesh():
    # A simple box mesh
    return trimesh.creation.box(extents=(0.1, 0.1, 0.1))


def test_wrench_guided_deterministic(dummy_mesh):
    rng1 = np.random.default_rng(42)
    finger_ids = np.array([0, 1, 2, 3])

    proposal1 = generate_wrench_guided_proposal(dummy_mesh, 4, rng1, finger_ids, num_candidates=5)

    rng2 = np.random.default_rng(42)
    proposal2 = generate_wrench_guided_proposal(dummy_mesh, 4, rng2, finger_ids, num_candidates=5)

    np.testing.assert_allclose(proposal1.target_points, proposal2.target_points)
    np.testing.assert_allclose(proposal1.inward_normals, proposal2.inward_normals)
    assert np.array_equal(proposal1.face_ids, proposal2.face_ids)


def test_wrench_guided_provenance(dummy_mesh):
    rng = np.random.default_rng(42)
    finger_ids = np.array([0, 1, 2, 3])

    proposal = generate_wrench_guided_proposal(dummy_mesh, 4, rng, finger_ids, num_candidates=5)
    assert proposal.provenance == "wrench_guided"


def test_wrench_guided_can_target_every_opposing_finger(dummy_mesh):
    assert _opposing_finger_schedule(5, 4) == [0, 1, 2, 3]
    finger_ids = np.arange(5)
    proposal = generate_wrench_guided_proposal(
        dummy_mesh,
        5,
        np.random.default_rng(7),
        finger_ids,
        thumb_index=4,
        opposing_finger_index=3,
        num_candidates=5,
    )
    assert proposal.active_fingers.tolist() == [False, False, False, True, True]
    assert proposal.opposition_pairs.tolist() == [[4, 3]]
