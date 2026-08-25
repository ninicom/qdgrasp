import copy

import numpy as np
import pytest

from qdgrasp.dataset.pipeline.contracts import DynamicValidation
from qdgrasp.dataset.pipeline.validators.scene_dynamic import (
    SceneDynamicValidator,
    hash_scene_state,
)

IDENTITY_HASHES = {
    "protocol_hash": "a" * 64,
    "recipe_hash": "b" * 64,
    "source_hash": "c" * 64,
}


def _pose(x=0.0, y=0.0, z=0.0, quat=None):
    return {
        "pos": np.array([x, y, z], dtype=np.float64),
        "quat": np.array(quat or [1.0, 0.0, 0.0, 0.0], dtype=np.float64),
    }


def _evidence():
    stages = {
        "initial": {"target": _pose(), "obstacle": _pose(y=1.0)},
        "squeeze": {"target": _pose(z=0.005), "obstacle": _pose(y=1.0)},
        "lift": {"target": _pose(z=0.05), "obstacle": _pose(y=1.005)},
        "perturbation": {"target": _pose(z=0.05), "obstacle": _pose(y=1.01)},
    }
    base = DynamicValidation(
        trajectory_metrics={"lift_achieved": 0.05, "final_active_fingers": 2.0},
        per_finger_loads=np.ones((2, 6)),
        failure_stage="none",
        passed=True,
    )
    return {
        "target_object_id": "target",
        "initial_scene_state": stages["initial"],
        "final_scene_state": stages["perturbation"],
        "base_validation": base,
        "stage_scene_states": stages,
        "state_hashes": {stage: hash_scene_state(state) for stage, state in stages.items()},
        "contact_object_ids": ["target"],
        "non_target_impulses": {"obstacle": 0.1},
        **IDENTITY_HASHES,
    }


def test_complete_measured_evidence_passes_without_fabricated_metrics():
    result = SceneDynamicValidator().validate(**_evidence())
    assert result.passed
    assert result.failure_stage == "none"
    assert result.trajectory_metrics["measured_target_lift"] == pytest.approx(0.05)
    assert result.trajectory_metrics["validated_stages"] == [
        "initial",
        "squeeze",
        "lift",
        "perturbation",
    ]
    np.testing.assert_array_equal(result.per_finger_loads, np.ones((2, 6)))


def test_failed_base_rollout_cannot_be_promoted_by_scene_evidence():
    evidence = _evidence()
    evidence["base_validation"] = DynamicValidation(
        trajectory_metrics={"lift_achieved": 0.05},
        per_finger_loads=np.ones((2, 6)),
        failure_stage="perturbation",
        passed=False,
    )
    result = SceneDynamicValidator().validate(**evidence)
    assert not result.passed
    assert result.failure_stage == "target_not_lifted"


def test_passed_flag_without_measured_base_evidence_fails_closed():
    evidence = _evidence()
    evidence["base_validation"] = DynamicValidation(
        trajectory_metrics={"lift_achieved": 0.05, "final_active_fingers": 2.0},
        per_finger_loads=np.empty((0, 6)),
        failure_stage="none",
        passed=True,
    )
    result = SceneDynamicValidator().validate(**evidence)
    assert result.failure_stage == "evidence_incomplete"
    assert result.trajectory_metrics["evidence_error"] == "invalid_base_dynamic_evidence"


def test_missing_or_tampered_stage_evidence_fails_closed():
    evidence = _evidence()
    del evidence["stage_scene_states"]["lift"]
    result = SceneDynamicValidator().validate(**evidence)
    assert result.failure_stage == "evidence_incomplete"
    assert result.trajectory_metrics["missing_stages"] == ["lift"]

    evidence = _evidence()
    evidence["state_hashes"]["squeeze"] = "0" * 64
    result = SceneDynamicValidator().validate(**evidence)
    assert result.failure_stage == "evidence_incomplete"
    assert result.trajectory_metrics["state_hash_mismatch"] == ["squeeze"]


def test_missing_object_or_non_finite_pose_is_scene_unstable():
    evidence = _evidence()
    del evidence["final_scene_state"]["obstacle"]
    evidence["state_hashes"]["perturbation"] = hash_scene_state(
        evidence["final_scene_state"]
    )
    result = SceneDynamicValidator().validate(**evidence)
    assert result.failure_stage == "scene_unstable"

    evidence = _evidence()
    evidence["stage_scene_states"]["lift"]["obstacle"]["pos"][0] = np.nan
    result = SceneDynamicValidator().validate(**evidence)
    assert result.failure_stage == "scene_unstable"
    assert "non-finite" in result.trajectory_metrics["state_error"]


def test_target_lift_must_match_pose_and_base_measurement():
    evidence = _evidence()
    evidence["base_validation"].trajectory_metrics["lift_achieved"] = 0.1
    result = SceneDynamicValidator().validate(**evidence)
    assert result.failure_stage == "target_not_lifted"
    assert result.trajectory_metrics["measured_target_lift"] == pytest.approx(0.05)


def test_wrong_object_contact_is_rejected():
    evidence = _evidence()
    evidence["contact_object_ids"] = ["target", "obstacle"]
    result = SceneDynamicValidator().validate(**evidence)
    assert result.failure_stage == "wrong_object_contact"
    assert result.trajectory_metrics["wrong_contacts"] == ["obstacle"]


@pytest.mark.parametrize(
    ("mutation", "metric"),
    [
        ("translation", "displacement"),
        ("rotation", "rotation"),
        ("impulse", "impulse"),
        ("wrong_lift", "vertical_displacement"),
    ],
)
def test_non_target_disturbance_gates_translation_rotation_and_impulse(mutation, metric):
    evidence = _evidence()
    if mutation == "translation":
        evidence["final_scene_state"]["obstacle"]["pos"][1] = 1.1
    elif mutation == "rotation":
        angle = 0.3
        evidence["final_scene_state"]["obstacle"]["quat"] = np.array(
            [np.cos(angle / 2), 0.0, 0.0, np.sin(angle / 2)]
        )
    else:
        if mutation == "impulse":
            evidence["non_target_impulses"]["obstacle"] = 1.1
        else:
            evidence["final_scene_state"]["obstacle"]["pos"][2] = 0.02
    evidence["state_hashes"]["perturbation"] = hash_scene_state(
        evidence["final_scene_state"]
    )
    result = SceneDynamicValidator().validate(**evidence)
    assert result.failure_stage == "non_target_disturbed"
    threshold_name = (
        "non_target_lift_threshold" if metric == "vertical_displacement" else f"{metric}_threshold"
    )
    assert result.trajectory_metrics[metric] > getattr(SceneDynamicValidator(), threshold_name)


def test_missing_impulse_and_invalid_identity_hash_fail_closed():
    evidence = _evidence()
    evidence["non_target_impulses"] = {}
    result = SceneDynamicValidator().validate(**evidence)
    assert result.failure_stage == "evidence_incomplete"
    assert result.trajectory_metrics["missing_impulse_object"] == "obstacle"

    evidence = _evidence()
    evidence["protocol_hash"] = "not-a-hash"
    result = SceneDynamicValidator().validate(**evidence)
    assert result.failure_stage == "evidence_incomplete"
    assert result.trajectory_metrics["invalid_identity_hashes"] == ["protocol_hash"]


def test_inputs_are_not_mutated():
    evidence = _evidence()
    initial_copy = copy.deepcopy(evidence["initial_scene_state"])
    SceneDynamicValidator().validate(**evidence)
    for object_id in initial_copy:
        np.testing.assert_array_equal(
            evidence["initial_scene_state"][object_id]["pos"], initial_copy[object_id]["pos"]
        )
