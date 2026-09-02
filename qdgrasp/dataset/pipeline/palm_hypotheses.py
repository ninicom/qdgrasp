"""Constrained grasp-frame palm hypotheses for P3.2.1-06.

The previous production initializer selected a free Kabsch fit of fingertip
points.  Point registration has no notion of approach side, gravity, floor, or
contact-axis direction.  This module constructs explicit hand/object grasp
frames from the proposal's opposition identity, enumerates both approach sides,
and keeps Kabsch only as a measured fallback hypothesis subject to the same
admission metrics.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation


class PalmHypothesisError(ValueError):
    """No finite palm hypothesis satisfies the hard floor constraint."""


@dataclass(frozen=True)
class PalmHypothesis:
    hypothesis_id: str
    palm_pos: np.ndarray
    palm_rot: np.ndarray
    max_position_error: float
    min_normal_alignment: float
    floor_clearance: float
    mode: str

    @property
    def sort_key(self) -> tuple[float, float, float, str]:
        # A contact axis pointing away from its inward object normal is a hard
        # orientation defect.  Among admissible orientations, position error
        # and then normal alignment rank the hypotheses without unit-mixing.
        normal_violation = max(0.0, -self.min_normal_alignment)
        return (
            normal_violation,
            self.max_position_error,
            -self.min_normal_alignment,
            self.hypothesis_id,
        )


def _unit(vector: np.ndarray, *, name: str) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(value))
    if not np.isfinite(norm) or norm < 1e-9:
        raise PalmHypothesisError(f"degenerate {name}")
    return value / norm


def grasp_frame(closing_axis: np.ndarray, approach_axis: np.ndarray) -> np.ndarray:
    """Build a right-handed frame [closing, lateral, approach]."""
    closing = _unit(closing_axis, name="closing axis")
    approach = np.asarray(approach_axis, dtype=np.float64)
    approach = approach - closing * float(np.dot(closing, approach))
    approach = _unit(approach, name="approach axis")
    lateral = _unit(np.cross(approach, closing), name="lateral axis")
    frame = np.column_stack([closing, lateral, approach])
    if float(np.linalg.det(frame)) < 0.999:
        raise PalmHypothesisError("grasp frame is not a proper rotation")
    return frame


def _proper_kabsch(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    source_centered = source - np.mean(source, axis=0)
    target_centered = target - np.mean(target, axis=0)
    covariance = source_centered.T @ target_centered
    u, _, vt = np.linalg.svd(covariance)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0.0:
        vt[-1] *= -1.0
        rotation = vt.T @ u.T
    return rotation


def _proper_direction_fit(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Orthogonal Procrustes for vectors anchored at the origin."""
    covariance = np.asarray(source, dtype=np.float64).T @ np.asarray(
        target, dtype=np.float64
    )
    # One direction constrains only two rotational degrees of freedom; the
    # remaining roll is an arbitrary SVD basis choice and is not equivariant
    # under a change of world frame.  Such a pose belongs to the grasp-frame
    # hypotheses, where the approach/gravity axis resolves roll explicitly.
    if np.linalg.matrix_rank(covariance) < 2:
        raise PalmHypothesisError("direction fit does not constrain palm roll")
    u, _, vt = np.linalg.svd(covariance)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0.0:
        vt[-1] *= -1.0
        rotation = vt.T @ u.T
    return rotation


def _opposition_geometry(
    points: np.ndarray,
    opposition_pairs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    pairs = np.asarray(opposition_pairs, dtype=np.int64).reshape(-1, 2)
    if len(pairs) == 0:
        raise PalmHypothesisError("opposition_pairs is empty")
    anchors = np.unique(pairs[:, 0])
    if len(anchors) != 1:
        raise PalmHypothesisError("opposition_pairs must share one anchor finger")
    anchor = int(anchors[0])
    opposing = pairs[:, 1]
    anchor_point = np.asarray(points[anchor], dtype=np.float64)
    opposing_center = np.mean(np.asarray(points[opposing], dtype=np.float64), axis=0)
    return anchor_point, opposing_center


def _fallback_tangent(closing: np.ndarray) -> np.ndarray:
    axes = np.eye(3, dtype=np.float64)
    candidate = axes[int(np.argmin(np.abs(axes @ closing)))]
    return candidate - closing * float(np.dot(closing, candidate))


def generate_palm_hypotheses(
    *,
    source_tips: np.ndarray,
    source_directions: np.ndarray,
    target_tips: np.ndarray,
    target_normals: np.ndarray,
    active_fingers: np.ndarray,
    opposition_pairs: np.ndarray | None,
    object_centroid: np.ndarray,
    gravity_axis: np.ndarray | None = None,
    floor_z: float = 0.0,
    min_palm_floor_clearance: float = 0.005,
    hypothesis_prefix: str = "palm",
) -> list[PalmHypothesis]:
    """Enumerate admitted frame and Kabsch hypotheses with explicit metrics."""
    if gravity_axis is None:
        gravity_axis = np.array([0.0, 0.0, -1.0])
    source = np.asarray(source_tips, dtype=np.float64)
    directions = np.asarray(source_directions, dtype=np.float64)
    target = np.asarray(target_tips, dtype=np.float64)
    normals = np.asarray(target_normals, dtype=np.float64)
    active = np.asarray(active_fingers, dtype=bool)
    if source.shape != target.shape or source.shape != directions.shape or target.shape != normals.shape:
        raise PalmHypothesisError("tip/direction/normal arrays must share shape [K, 3]")
    if active.shape != (len(source),) or int(active.sum()) < 2:
        raise PalmHypothesisError("active_fingers must select at least two tips")

    source_center = np.mean(source[active], axis=0)
    target_center = np.mean(target[active], axis=0)
    gravity = _unit(gravity_axis, name="gravity axis")
    rotations: list[tuple[str, np.ndarray]] = []

    if opposition_pairs is not None:
        source_anchor, source_opposing = _opposition_geometry(source, opposition_pairs)
        target_anchor, target_opposing = _opposition_geometry(target, opposition_pairs)
        source_closing = source_opposing - source_anchor
        target_closing = target_opposing - target_anchor
        source_approach = source_center
        if np.linalg.norm(
            source_approach
            - _unit(source_closing, name="source closing")
            * np.dot(_unit(source_closing, name="source closing"), source_approach)
        ) < 1e-8:
            source_approach = np.mean(directions[active], axis=0)
        source_frame = grasp_frame(source_closing, source_approach)

        target_closing_unit = _unit(target_closing, name="target closing")
        target_tangent = np.cross(gravity, target_closing_unit)
        if np.linalg.norm(target_tangent) < 1e-8:
            centroid_direction = target_center - np.asarray(object_centroid, dtype=np.float64)
            target_tangent = centroid_direction - target_closing_unit * np.dot(
                target_closing_unit, centroid_direction
            )
        if np.linalg.norm(target_tangent) < 1e-8:
            target_tangent = _fallback_tangent(target_closing_unit)

        target_tangent = _unit(target_tangent, name="target tangent")
        target_bitangent = _unit(
            np.cross(target_closing_unit, target_tangent),
            name="target bitangent",
        )
        # Finite roll bank around the opposition axis.  The old +/- pair only
        # represented 0 and 180 degrees and systematically put wrists below the
        # floor for side grasps.  45-degree spacing is pinned geometry, not a
        # data-dependent retry or candidate-budget increase.
        for roll_degrees in range(0, 360, 45):
            angle = np.deg2rad(float(roll_degrees))
            rolled_approach = (
                np.cos(angle) * target_tangent
                + np.sin(angle) * target_bitangent
            )
            target_frame = grasp_frame(target_closing, rolled_approach)
            rotations.append(
                (
                    f"grasp_frame:roll{roll_degrees:03d}",
                    target_frame @ source_frame.T,
                )
            )

    try:
        rotations.append(
            (
                "direction_fit",
                _proper_direction_fit(directions[active], normals[active]),
            )
        )
    except PalmHypothesisError:
        pass
    rotations.append(("kabsch", _proper_kabsch(source[active], target[active])))

    hypotheses: list[PalmHypothesis] = []
    for mode, rotation in rotations:
        if not np.all(np.isfinite(rotation)) or np.linalg.det(rotation) < 0.999:
            continue
        # Keep independently derived modes even when two rotations coincide.
        # Their provenance is part of the hypothesis contract and lets the
        # causal suite distinguish direction fit, grasp-frame, and Kabsch.
        translation = target_center - rotation @ source_center
        transformed_tips = (rotation @ source.T).T + translation
        transformed_directions = (rotation @ directions.T).T
        position_error = np.linalg.norm(
            transformed_tips[active] - target[active], axis=1
        )
        normal_alignment = np.sum(
            transformed_directions[active] * normals[active], axis=1
        )
        floor_clearance = float(translation[2] - floor_z)
        if floor_clearance < min_palm_floor_clearance:
            continue
        hypotheses.append(
            PalmHypothesis(
                hypothesis_id=f"{hypothesis_prefix}:{mode}",
                palm_pos=translation.astype(np.float64),
                palm_rot=rotation.astype(np.float64),
                max_position_error=float(np.max(position_error)),
                min_normal_alignment=float(np.min(normal_alignment)),
                floor_clearance=floor_clearance,
                mode=mode,
            )
        )

    if not hypotheses:
        raise PalmHypothesisError("no palm hypothesis satisfies floor clearance")
    return sorted(hypotheses, key=lambda item: item.sort_key)


def best_palm_hypothesis(**kwargs) -> PalmHypothesis:
    """Return the highest-ranked admitted hypothesis."""
    return generate_palm_hypotheses(**kwargs)[0]


def bounded_local_pose_refinement(
    *,
    palm_pos: np.ndarray,
    palm_rot: np.ndarray,
    achieved_tips: np.ndarray,
    target_tips: np.ndarray,
    active_fingers: np.ndarray,
    floor_z: float,
    min_palm_floor_clearance: float = 0.005,
    max_translation: float = 0.01,
    max_rotation: float = np.deg2rad(10.0),
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Apply one trust-region SE(3) correction from achieved to target tips.

    This is local refinement, not a second unconstrained initializer.  Rotation
    and translation are both clipped, and a correction that would cross the
    floor is rejected while still returning telemetry.
    """
    active = np.asarray(active_fingers, dtype=bool)
    achieved = np.asarray(achieved_tips, dtype=np.float64)[active]
    target = np.asarray(target_tips, dtype=np.float64)[active]
    if len(achieved) < 2:
        raise PalmHypothesisError("local refinement requires at least two active tips")

    delta_rotation_full = _proper_kabsch(achieved, target)
    rotvec = Rotation.from_matrix(delta_rotation_full).as_rotvec()
    rotation_angle_full = float(np.linalg.norm(rotvec))
    if rotation_angle_full > max_rotation > 0.0:
        rotvec *= max_rotation / rotation_angle_full
    delta_rotation = Rotation.from_rotvec(rotvec).as_matrix()

    achieved_center = np.mean(achieved, axis=0)
    target_center = np.mean(target, axis=0)
    delta_translation = target_center - delta_rotation @ achieved_center
    proposed_pos = delta_rotation @ np.asarray(palm_pos, dtype=np.float64) + delta_translation
    translation_delta = proposed_pos - np.asarray(palm_pos, dtype=np.float64)
    translation_norm_full = float(np.linalg.norm(translation_delta))
    if translation_norm_full > max_translation > 0.0:
        translation_delta *= max_translation / translation_norm_full
        proposed_pos = np.asarray(palm_pos, dtype=np.float64) + translation_delta

    proposed_rot = delta_rotation @ np.asarray(palm_rot, dtype=np.float64)
    floor_clearance = float(proposed_pos[2] - floor_z)
    floor_rejected = floor_clearance < min_palm_floor_clearance
    if floor_rejected:
        proposed_pos = np.asarray(palm_pos, dtype=np.float64).copy()
        proposed_rot = np.asarray(palm_rot, dtype=np.float64).copy()

    metrics = {
        "requested_translation": translation_norm_full,
        "applied_translation": float(
            np.linalg.norm(proposed_pos - np.asarray(palm_pos, dtype=np.float64))
        ),
        "requested_rotation": rotation_angle_full,
        "applied_rotation": float(np.linalg.norm(rotvec)) if not floor_rejected else 0.0,
        "floor_clearance": floor_clearance,
        "floor_rejected": float(floor_rejected),
    }
    return proposed_pos, proposed_rot, metrics
