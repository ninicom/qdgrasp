"""Fail-closed multi-object scene rollout validation."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np

from qdgrasp.dataset.pipeline.contracts import DynamicValidation

SceneState = Mapping[str, Mapping[str, np.ndarray]]
REQUIRED_STAGES = ("initial", "squeeze", "lift", "perturbation")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _canonical_scene_state(state: SceneState) -> dict[str, dict[str, list[float]]]:
    return {
        object_id: {
            "pos": np.asarray(pose["pos"], dtype=np.float64).tolist(),
            "quat": np.asarray(pose["quat"], dtype=np.float64).tolist(),
        }
        for object_id, pose in sorted(state.items())
    }


def hash_scene_state(state: SceneState) -> str:
    """Return a stable SHA-256 over canonical object poses."""
    payload = json.dumps(_canonical_scene_state(state), sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
        "utf-8"
    )
    return hashlib.sha256(payload).hexdigest()


def _rotation_distance(quat_a: np.ndarray, quat_b: np.ndarray) -> float:
    # Scene contract uses MuJoCo-style wxyz quaternions. q and -q represent
    # the same rotation, hence the absolute dot product.
    dot = float(abs(np.dot(quat_a, quat_b)))
    return float(2.0 * np.arccos(np.clip(dot, -1.0, 1.0)))


class SceneDynamicValidator:
    """Compose genuine object-level rollout evidence with scene safety gates."""

    def __init__(
        self,
        displacement_threshold: float = 0.05,
        rotation_threshold: float = 0.2,
        impulse_threshold: float = 1.0,
        minimum_target_lift: float = 0.025,
        non_target_lift_threshold: float = 0.01,
        lift_consistency_tolerance: float = 0.002,
    ):
        thresholds = {
            "displacement_threshold": displacement_threshold,
            "rotation_threshold": rotation_threshold,
            "impulse_threshold": impulse_threshold,
            "minimum_target_lift": minimum_target_lift,
            "non_target_lift_threshold": non_target_lift_threshold,
            "lift_consistency_tolerance": lift_consistency_tolerance,
        }
        invalid = [name for name, value in thresholds.items() if not math.isfinite(value) or value <= 0]
        if invalid:
            raise ValueError(f"scene validator thresholds must be finite and positive: {invalid}")
        self.displacement_threshold = float(displacement_threshold)
        self.rotation_threshold = float(rotation_threshold)
        self.impulse_threshold = float(impulse_threshold)
        self.minimum_target_lift = float(minimum_target_lift)
        self.non_target_lift_threshold = float(non_target_lift_threshold)
        self.lift_consistency_tolerance = float(lift_consistency_tolerance)

    @staticmethod
    def _result(
        base_validation: DynamicValidation | None,
        failure_stage: str,
        metrics: dict[str, Any],
        *,
        passed: bool = False,
    ) -> DynamicValidation:
        base_metrics = dict(base_validation.trajectory_metrics) if base_validation else {}
        base_metrics.update(metrics)
        loads = (
            np.asarray(base_validation.per_finger_loads, dtype=np.float64).copy()
            if base_validation is not None
            else np.empty((0, 6), dtype=np.float64)
        )
        return DynamicValidation(
            trajectory_metrics=base_metrics,
            per_finger_loads=loads,
            failure_stage=failure_stage,
            passed=passed,
        )

    @staticmethod
    def _validate_states(stage_states: Mapping[str, SceneState]) -> str | None:
        expected_objects: set[str] | None = None
        for stage in REQUIRED_STAGES:
            state = stage_states[stage]
            object_ids = set(state)
            if not object_ids or (expected_objects is not None and object_ids != expected_objects):
                return f"object set changed at stage {stage}"
            expected_objects = object_ids
            for object_id, pose in state.items():
                if "pos" not in pose or "quat" not in pose:
                    return f"missing pose field for {object_id} at stage {stage}"
                pos = np.asarray(pose["pos"], dtype=np.float64)
                quat = np.asarray(pose["quat"], dtype=np.float64)
                if pos.shape != (3,) or quat.shape != (4,):
                    return f"invalid pose shape for {object_id} at stage {stage}"
                if not np.all(np.isfinite(pos)) or not np.all(np.isfinite(quat)):
                    return f"non-finite pose for {object_id} at stage {stage}"
                if not np.isclose(np.linalg.norm(quat), 1.0, atol=1e-6):
                    return f"non-unit quaternion for {object_id} at stage {stage}"
        return None

    def validate(
        self,
        target_object_id: str,
        initial_scene_state: SceneState,
        final_scene_state: SceneState,
        target_lifted: bool | None = None,
        *,
        base_validation: DynamicValidation | None = None,
        stage_scene_states: Mapping[str, SceneState] | None = None,
        state_hashes: Mapping[str, str] | None = None,
        contact_object_ids: Iterable[str] | None = None,
        non_target_impulses: Mapping[str, float] | None = None,
        protocol_hash: str = "",
        recipe_hash: str = "",
        source_hash: str = "",
    ) -> DynamicValidation:
        """Validate complete squeeze/lift/perturbation scene evidence.

        ``initial_scene_state`` and ``final_scene_state`` remain explicit to
        prevent callers from presenting stage hashes for a different rollout.
        All quaternions are wxyz and all impulses are measured scalar magnitudes.
        """
        if base_validation is None:
            return self._result(None, "evidence_incomplete", {"evidence_error": "missing_base_validation"})
        if not base_validation.passed or target_lifted is False:
            return self._result(
                base_validation,
                "target_not_lifted",
                {"base_failure_stage": base_validation.failure_stage},
            )
        base_loads = np.asarray(base_validation.per_finger_loads, dtype=np.float64)
        active_fingers = base_validation.trajectory_metrics.get("final_active_fingers")
        swept_clearance = base_validation.trajectory_metrics.get("swept_clearance_passed")
        if (
            base_validation.failure_stage != "none"
            or base_loads.ndim != 2
            or base_loads.shape[0] == 0
            or base_loads.shape[1:] != (6,)
            or not np.all(np.isfinite(base_loads))
            or not np.any(np.abs(base_loads) > 0.0)
            or not isinstance(active_fingers, (int, float, np.integer, np.floating))
            or not math.isfinite(float(active_fingers))
            or float(active_fingers) <= 0.0
            or swept_clearance != 1.0
        ):
            return self._result(
                base_validation,
                "evidence_incomplete",
                {"evidence_error": "invalid_base_dynamic_evidence"},
            )
        if (
            stage_scene_states is None
            or state_hashes is None
            or contact_object_ids is None
            or non_target_impulses is None
        ):
            return self._result(
                base_validation,
                "evidence_incomplete",
                {"evidence_error": "missing_scene_evidence"},
            )

        stage_states = dict(stage_scene_states)
        stage_states["initial"] = initial_scene_state
        stage_states["perturbation"] = final_scene_state
        missing_stages = [stage for stage in REQUIRED_STAGES if stage not in stage_states]
        if missing_stages:
            return self._result(
                base_validation,
                "evidence_incomplete",
                {"missing_stages": missing_stages},
            )
        state_error = self._validate_states(stage_states)
        if state_error:
            return self._result(base_validation, "scene_unstable", {"state_error": state_error})
        if target_object_id not in initial_scene_state:
            return self._result(
                base_validation,
                "scene_unstable",
                {"state_error": f"target is absent: {target_object_id}"},
            )

        bad_identity_hashes = {
            name: digest
            for name, digest in {
                "protocol_hash": protocol_hash,
                "recipe_hash": recipe_hash,
                "source_hash": source_hash,
            }.items()
            if not _SHA256.fullmatch(digest)
        }
        if bad_identity_hashes:
            return self._result(
                base_validation,
                "evidence_incomplete",
                {"invalid_identity_hashes": sorted(bad_identity_hashes)},
            )
        hash_errors: list[str] = []
        measured_hashes: dict[str, str] = {}
        for stage in REQUIRED_STAGES:
            measured_hashes[stage] = hash_scene_state(stage_states[stage])
            if state_hashes.get(stage) != measured_hashes[stage]:
                hash_errors.append(stage)
        if hash_errors:
            return self._result(
                base_validation,
                "evidence_incomplete",
                {"state_hash_mismatch": hash_errors},
            )

        target_initial = np.asarray(initial_scene_state[target_object_id]["pos"], dtype=np.float64)
        target_final = np.asarray(final_scene_state[target_object_id]["pos"], dtype=np.float64)
        measured_lift = float(target_final[2] - target_initial[2])
        squeeze_stage_pos = np.asarray(stage_states["squeeze"][target_object_id]["pos"], dtype=np.float64)
        lift_stage_pos = np.asarray(stage_states["lift"][target_object_id]["pos"], dtype=np.float64)
        lift_stage_height = float(lift_stage_pos[2] - target_initial[2])
        measured_lift_phase = float(lift_stage_pos[2] - squeeze_stage_pos[2])
        base_lift = base_validation.trajectory_metrics.get("lift_achieved")
        if (
            not isinstance(base_lift, (int, float, np.integer, np.floating))
            or not math.isfinite(float(base_lift))
            or measured_lift < self.minimum_target_lift
            or lift_stage_height < self.minimum_target_lift
            or abs(float(base_lift) - measured_lift_phase) > self.lift_consistency_tolerance
        ):
            return self._result(
                base_validation,
                "target_not_lifted",
                {
                    "measured_target_lift": measured_lift,
                    "measured_lift_phase": measured_lift_phase,
                    "lift_stage_height": lift_stage_height,
                    "base_target_lift": base_lift,
                    "lift_consistency_error": (
                        abs(float(base_lift) - measured_lift_phase)
                        if isinstance(base_lift, (int, float, np.integer, np.floating))
                        else None
                    ),
                },
            )

        contacts = tuple(contact_object_ids)
        wrong_contacts = sorted({object_id for object_id in contacts if object_id != target_object_id})
        if target_object_id not in contacts or wrong_contacts:
            return self._result(
                base_validation,
                "wrong_object_contact",
                {"contact_object_ids": sorted(set(contacts)), "wrong_contacts": wrong_contacts},
            )

        non_target_metrics: dict[str, dict[str, float]] = {}
        for object_id in sorted(set(initial_scene_state) - {target_object_id}):
            if object_id not in non_target_impulses:
                return self._result(
                    base_validation,
                    "evidence_incomplete",
                    {"missing_impulse_object": object_id},
                )
            impulse = float(non_target_impulses[object_id])
            if not math.isfinite(impulse) or impulse < 0.0:
                return self._result(
                    base_validation,
                    "scene_unstable",
                    {"invalid_impulse_object": object_id, "impulse": impulse},
                )
            initial_pose = initial_scene_state[object_id]
            final_pose = final_scene_state[object_id]
            displacement = float(
                np.linalg.norm(
                    np.asarray(final_pose["pos"], dtype=np.float64) - np.asarray(initial_pose["pos"], dtype=np.float64)
                )
            )
            rotation = _rotation_distance(
                np.asarray(initial_pose["quat"], dtype=np.float64),
                np.asarray(final_pose["quat"], dtype=np.float64),
            )
            vertical_displacement = float(
                np.asarray(final_pose["pos"], dtype=np.float64)[2]
                - np.asarray(initial_pose["pos"], dtype=np.float64)[2]
            )
            non_target_metrics[object_id] = {
                "displacement": displacement,
                "rotation": rotation,
                "impulse": impulse,
                "vertical_displacement": vertical_displacement,
            }
            if (
                displacement > self.displacement_threshold
                or rotation > self.rotation_threshold
                or impulse > self.impulse_threshold
                or vertical_displacement > self.non_target_lift_threshold
            ):
                return self._result(
                    base_validation,
                    "non_target_disturbed",
                    {"disturbed_object": object_id, **non_target_metrics[object_id]},
                )

        return self._result(
            base_validation,
            "none",
            {
                "measured_target_lift": measured_lift,
                "measured_lift_phase": measured_lift_phase,
                "lift_stage_height": lift_stage_height,
                "lift_consistency_error": abs(float(base_lift) - measured_lift_phase),
                "non_target_motion": non_target_metrics,
                "scene_state_hashes": measured_hashes,
                "protocol_hash": protocol_hash,
                "recipe_hash": recipe_hash,
                "source_hash": source_hash,
                "validated_stages": list(REQUIRED_STAGES),
            },
            passed=True,
        )
