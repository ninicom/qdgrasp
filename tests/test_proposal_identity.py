"""Proposal active-set and content-identity contracts for P3.2.1-05."""

import numpy as np
import pytest

from qdgrasp.dataset.pipeline.proposals.identity import (
    normalize_active_fingers,
    stable_candidate_id,
)


def test_legacy_active_set_defaults_to_all_without_aliasing() -> None:
    active = normalize_active_fingers(None, 4)
    assert active.tolist() == [True, True, True, True]


def test_active_set_fails_closed_on_shape_or_cardinality() -> None:
    with pytest.raises(ValueError, match="shape"):
        normalize_active_fingers(np.ones(3, dtype=bool), 4)
    with pytest.raises(ValueError, match="at least 2"):
        normalize_active_fingers(np.array([True, False, False, False]), 4)


def test_candidate_identity_changes_with_task_membership() -> None:
    points = np.arange(12, dtype=np.float64).reshape(4, 3)
    normals = np.tile([1.0, 0.0, 0.0], (4, 1))
    kwargs = dict(
        provenance="fixture",
        target_points=points,
        inward_normals=normals,
        face_ids=np.arange(4),
        finger_ids=np.arange(4),
    )
    first = stable_candidate_id(
        **kwargs, active_fingers=np.array([True, True, True, False])
    )
    repeated = stable_candidate_id(
        **kwargs, active_fingers=np.array([True, True, True, False])
    )
    changed = stable_candidate_id(
        **kwargs, active_fingers=np.array([True, True, False, True])
    )
    assert first == repeated
    assert first != changed
