"""Deterministic sequential-declutter orchestration and state lineage."""

from __future__ import annotations

import copy
import hashlib
import json
import re
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from qdgrasp.dataset.pipeline.validators.scene_dynamic import SceneState, hash_scene_state
from qdgrasp.scenes.contracts import SceneGraspOutcome

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class DeclutterError(ValueError):
    """A malformed state transition or non-genuine validation attempt."""


@dataclass(frozen=True)
class SceneStateRecord:
    step_index: int
    scene_state_hash: str
    lineage_hash: str
    parent_lineage_hash: str | None
    removed_object_id: str | None
    object_poses: dict[str, dict[str, np.ndarray]]


@dataclass(frozen=True)
class DeclutterAttempt:
    step_index: int
    target_object_id: str
    parent_lineage_hash: str
    candidate_id: str
    passed: bool
    failure_reason: str
    child_lineage_hash: str | None = None


@dataclass(frozen=True)
class DeclutterSequence:
    states: tuple[SceneStateRecord, ...]
    attempts: tuple[DeclutterAttempt, ...]


TargetSelector = Callable[[SceneState, int], str]
GraspValidator = Callable[[str, SceneState, int], SceneGraspOutcome]
RemoveAndResettle = Callable[[SceneState, str, int], SceneState]


def _copy_state(state: SceneState) -> dict[str, dict[str, np.ndarray]]:
    copied: dict[str, dict[str, np.ndarray]] = {}
    for object_id, pose in state.items():
        copied[object_id] = {
            "pos": np.asarray(pose["pos"], dtype=np.float64).copy(),
            "quat": np.asarray(pose["quat"], dtype=np.float64).copy(),
        }
    # hash_scene_state rejects NaN and missing pose fields. Normalize here so
    # every callback observes an isolated, canonical state.
    hash_scene_state(copied)
    return copied


def _lineage_hash(
    scene_state_hash: str,
    *,
    step_index: int,
    parent_lineage_hash: str | None,
    removed_object_id: str | None,
) -> str:
    payload = {
        "parent_lineage_hash": parent_lineage_hash,
        "removed_object_id": removed_object_id,
        "scene_state_hash": scene_state_hash,
        "step_index": step_index,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _state_record(
    state: SceneState,
    *,
    step_index: int,
    parent_lineage_hash: str | None,
    removed_object_id: str | None,
) -> SceneStateRecord:
    copied = _copy_state(state)
    state_hash = hash_scene_state(copied)
    return SceneStateRecord(
        step_index=step_index,
        scene_state_hash=state_hash,
        lineage_hash=_lineage_hash(
            state_hash,
            step_index=step_index,
            parent_lineage_hash=parent_lineage_hash,
            removed_object_id=removed_object_id,
        ),
        parent_lineage_hash=parent_lineage_hash,
        removed_object_id=removed_object_id,
        object_poses=copied,
    )


def _genuine_outcome_error(
    outcome: SceneGraspOutcome, target_object_id: str, parent_state_hash: str
) -> str | None:
    if outcome.target_object_id != target_object_id:
        return "validator_target_mismatch"
    if outcome.label_stage != "dynamic_valid" or outcome.failure_reason != "none":
        return outcome.failure_reason or "dynamic_validation_failed"
    if outcome.scene_state_hashes.get("initial") != parent_state_hash:
        return "initial_state_hash_mismatch"
    if not outcome.dynamic_trajectory_evidence:
        return "missing_dynamic_trajectory_evidence"
    for name, digest in {
        "recipe_hash": outcome.recipe_hash,
        "protocol_hash": outcome.protocol_hash,
        "source_hash": outcome.source_hash,
    }.items():
        if not _SHA256.fullmatch(digest):
            return f"invalid_{name}"
    return None


def generate_sequential_declutter(
    initial_state: SceneState,
    *,
    select_target: TargetSelector,
    validate_grasp: GraspValidator,
    remove_and_resettle: RemoveAndResettle,
    max_steps: int | None = None,
) -> DeclutterSequence:
    """Validate, remove, and re-settle one target at a time.

    A failed attempt terminates the sequence without mutating its parent state.
    A successful outcome is never reused after removal: ``validate_grasp`` is
    invoked again with the newly hashed child state on the next iteration.
    """
    if max_steps is not None and max_steps <= 0:
        raise DeclutterError("max_steps must be positive when provided")
    initial_record = _state_record(
        initial_state,
        step_index=0,
        parent_lineage_hash=None,
        removed_object_id=None,
    )
    states = [initial_record]
    attempts: list[DeclutterAttempt] = []
    limit = len(initial_record.object_poses) if max_steps is None else max_steps

    for step_index in range(limit):
        parent = states[-1]
        if not parent.object_poses:
            break
        callback_state = _copy_state(parent.object_poses)
        target_object_id = select_target(copy.deepcopy(callback_state), step_index)
        if target_object_id not in callback_state:
            raise DeclutterError(f"selector returned absent target: {target_object_id}")
        outcome = validate_grasp(
            target_object_id, copy.deepcopy(callback_state), step_index
        )
        outcome_error = _genuine_outcome_error(
            outcome, target_object_id, parent.scene_state_hash
        )
        if outcome_error is not None:
            attempts.append(
                DeclutterAttempt(
                    step_index=step_index,
                    target_object_id=target_object_id,
                    parent_lineage_hash=parent.lineage_hash,
                    candidate_id=outcome.candidate_id,
                    passed=False,
                    failure_reason=outcome_error,
                )
            )
            break

        child_state = _copy_state(
            remove_and_resettle(copy.deepcopy(callback_state), target_object_id, step_index)
        )
        expected_child_objects = set(callback_state) - {target_object_id}
        if set(child_state) != expected_child_objects:
            raise DeclutterError(
                "remove_and_resettle must remove exactly the validated target; "
                f"expected {sorted(expected_child_objects)}, got {sorted(child_state)}"
            )
        child = _state_record(
            child_state,
            step_index=step_index + 1,
            parent_lineage_hash=parent.lineage_hash,
            removed_object_id=target_object_id,
        )
        states.append(child)
        attempts.append(
            DeclutterAttempt(
                step_index=step_index,
                target_object_id=target_object_id,
                parent_lineage_hash=parent.lineage_hash,
                candidate_id=outcome.candidate_id,
                passed=True,
                failure_reason="none",
                child_lineage_hash=child.lineage_hash,
            )
        )

    return DeclutterSequence(states=tuple(states), attempts=tuple(attempts))
