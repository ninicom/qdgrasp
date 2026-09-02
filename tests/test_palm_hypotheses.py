"""Grasp-frame palm hypothesis oracles for P3.2.1-06."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from qdgrasp.dataset.pipeline.palm_hypotheses import (
    PalmHypothesisError,
    bounded_local_pose_refinement,
    generate_palm_hypotheses,
    grasp_frame,
)


def _fixture():
    source = np.array(
        [
            [-0.04, 0.00, 0.06],
            [0.04, -0.02, 0.06],
            [0.04, 0.02, 0.06],
            [0.00, 0.06, 0.04],
        ],
        dtype=np.float64,
    )
    directions = np.array(
        [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]],
        dtype=np.float64,
    )
    active = np.array([True, True, True, False])
    pairs = np.array([[0, 1], [0, 2]], dtype=np.int64)
    rotation = Rotation.from_euler("z", 35.0, degrees=True).as_matrix()
    translation = np.array([0.1, -0.03, 0.08])
    target = (rotation @ source.T).T + translation
    normals = (rotation @ directions.T).T
    return source, directions, target, normals, active, pairs, rotation, translation


def test_grasp_frame_is_a_proper_rotation() -> None:
    frame = grasp_frame(np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]))
    np.testing.assert_allclose(frame.T @ frame, np.eye(3), atol=1e-12)
    assert np.linalg.det(frame) == pytest.approx(1.0)


def test_hypotheses_include_exact_rigid_recovery_and_are_deterministic() -> None:
    source, directions, target, normals, active, pairs, rotation, translation = _fixture()
    kwargs = {
        "source_tips": source,
        "source_directions": directions,
        "target_tips": target,
        "target_normals": normals,
        "active_fingers": active,
        "opposition_pairs": pairs,
        "object_centroid": np.zeros(3),
        "floor_z": 0.0,
        "min_palm_floor_clearance": 0.0,
        "hypothesis_prefix": "fixture",
    }
    first = generate_palm_hypotheses(**kwargs)
    second = generate_palm_hypotheses(**kwargs)
    assert [item.hypothesis_id for item in first] == [item.hypothesis_id for item in second]
    exact = min(first, key=lambda item: item.max_position_error)
    np.testing.assert_allclose(exact.palm_rot, rotation, atol=1e-10)
    np.testing.assert_allclose(exact.palm_pos, translation, atol=1e-10)
    assert exact.max_position_error < 1e-10


def test_contact_order_permutation_preserves_the_best_pose() -> None:
    source, directions, target, normals, active, pairs, _, _ = _fixture()
    base = generate_palm_hypotheses(
        source_tips=source,
        source_directions=directions,
        target_tips=target,
        target_normals=normals,
        active_fingers=active,
        opposition_pairs=pairs,
        object_centroid=np.zeros(3),
        floor_z=0.0,
        min_palm_floor_clearance=0.0,
    )[0]
    permutation = np.array([2, 0, 3, 1])
    inverse = np.empty_like(permutation)
    inverse[permutation] = np.arange(len(permutation))
    permuted_pairs = inverse[pairs]
    moved = generate_palm_hypotheses(
        source_tips=source[permutation],
        source_directions=directions[permutation],
        target_tips=target[permutation],
        target_normals=normals[permutation],
        active_fingers=active[permutation],
        opposition_pairs=permuted_pairs,
        object_centroid=np.zeros(3),
        floor_z=0.0,
        min_palm_floor_clearance=0.0,
    )[0]
    np.testing.assert_allclose(base.palm_rot, moved.palm_rot, atol=1e-10)
    np.testing.assert_allclose(base.palm_pos, moved.palm_pos, atol=1e-10)


def test_hypotheses_are_equivariant_to_a_world_rigid_transform() -> None:
    source, directions, target, normals, active, pairs, _, _ = _fixture()
    # Keep position-fit and direction-fit hypotheses distinct; otherwise their
    # mathematically identical rotations can be deduplicated under different
    # mode names at floating-point tolerance.
    target = target.copy()
    target[1] += np.array([0.004, -0.002, 0.001])
    kwargs = {
        "source_tips": source,
        "source_directions": directions,
        "target_tips": target,
        "target_normals": normals,
        "active_fingers": active,
        "opposition_pairs": pairs,
        "object_centroid": np.zeros(3),
        "floor_z": -10.0,
        "min_palm_floor_clearance": 0.0,
        "hypothesis_prefix": "equivariance",
    }
    base = generate_palm_hypotheses(**kwargs)

    world_rotation = Rotation.from_euler("xyz", [17.0, -11.0, 23.0], degrees=True).as_matrix()
    world_translation = np.array([-0.08, 0.12, 0.2])
    transformed = generate_palm_hypotheses(
        **{
            **kwargs,
            "target_tips": (world_rotation @ target.T).T + world_translation,
            "target_normals": (world_rotation @ normals.T).T,
            "object_centroid": world_translation,
            "gravity_axis": world_rotation @ np.array([0.0, 0.0, -1.0]),
        }
    )

    assert [item.mode for item in transformed] == [item.mode for item in base]
    for original, moved in zip(base, transformed):
        np.testing.assert_allclose(
            moved.palm_rot, world_rotation @ original.palm_rot, atol=1e-10
        )
        np.testing.assert_allclose(
            moved.palm_pos,
            world_rotation @ original.palm_pos + world_translation,
            atol=1e-10,
        )
        assert moved.max_position_error == pytest.approx(original.max_position_error)
        assert moved.min_normal_alignment == pytest.approx(original.min_normal_alignment)


def test_floor_constraint_fails_closed() -> None:
    source, directions, target, normals, active, pairs, _, _ = _fixture()
    target[:, 2] -= 1.0
    with pytest.raises(PalmHypothesisError, match="floor clearance"):
        generate_palm_hypotheses(
            source_tips=source,
            source_directions=directions,
            target_tips=target,
            target_normals=normals,
            active_fingers=active,
            opposition_pairs=pairs,
            object_centroid=np.zeros(3),
            floor_z=0.0,
            min_palm_floor_clearance=0.005,
        )


def test_direction_fit_is_an_unweighted_independent_hypothesis() -> None:
    source, directions, target, _, active, pairs, rotation, _ = _fixture()
    directions = directions.copy()
    directions[2] = np.array([-1.0, 0.3, 0.2])
    directions[2] /= np.linalg.norm(directions[2])
    normals = (rotation @ directions.T).T
    hypotheses = generate_palm_hypotheses(
        source_tips=source,
        source_directions=directions,
        target_tips=target,
        target_normals=normals,
        active_fingers=active,
        opposition_pairs=pairs,
        object_centroid=np.zeros(3),
        floor_z=0.0,
        min_palm_floor_clearance=0.0,
    )
    direction_fit = next(item for item in hypotheses if item.mode == "direction_fit")
    assert direction_fit.min_normal_alignment == pytest.approx(1.0, abs=1e-10)


def test_direction_fit_rejects_an_unobservable_roll() -> None:
    source, directions, target, normals, active, pairs, _, _ = _fixture()
    hypotheses = generate_palm_hypotheses(
        source_tips=source,
        source_directions=directions,
        target_tips=target,
        target_normals=normals,
        active_fingers=active,
        opposition_pairs=pairs,
        object_centroid=np.zeros(3),
        floor_z=0.0,
        min_palm_floor_clearance=0.0,
    )
    assert all(item.mode != "direction_fit" for item in hypotheses)


def test_local_refinement_obeys_translation_rotation_and_floor_trust_region() -> None:
    source, _, target, _, active, _, _, _ = _fixture()
    palm_pos = np.array([0.0, 0.0, 0.02])
    palm_rot = np.eye(3)
    moved_pos, moved_rot, metrics = bounded_local_pose_refinement(
        palm_pos=palm_pos,
        palm_rot=palm_rot,
        achieved_tips=source,
        target_tips=target,
        active_fingers=active,
        floor_z=0.0,
        min_palm_floor_clearance=0.005,
        max_translation=0.01,
        max_rotation=np.deg2rad(10.0),
    )
    assert np.linalg.norm(moved_pos - palm_pos) <= 0.01 + 1e-12
    applied_angle = np.linalg.norm(Rotation.from_matrix(moved_rot @ palm_rot.T).as_rotvec())
    assert applied_angle <= np.deg2rad(10.0) + 1e-12
    assert metrics["requested_translation"] >= metrics["applied_translation"]

    rejected_pos, rejected_rot, rejected = bounded_local_pose_refinement(
        palm_pos=np.array([0.0, 0.0, 0.005]),
        palm_rot=np.eye(3),
        achieved_tips=source,
        target_tips=target - np.array([0.0, 0.0, 2.0]),
        active_fingers=active,
        floor_z=0.0,
        min_palm_floor_clearance=0.005,
        max_translation=0.01,
        max_rotation=np.deg2rad(10.0),
    )
    np.testing.assert_allclose(rejected_pos, [0.0, 0.0, 0.005])
    np.testing.assert_allclose(rejected_rot, np.eye(3))
    assert rejected["floor_rejected"] == 1.0
