import copy

import numpy as np
import pytest

from qdgrasp.scenes.contracts import SceneGraspOutcome
from qdgrasp.scenes.declutter import DeclutterError, generate_sequential_declutter


def _pose(x: float):
    return {"pos": np.array([x, 0.0, 0.0]), "quat": np.array([1.0, 0.0, 0.0, 0.0])}


def _outcome(target, state_hash, step, *, passed=True):
    return SceneGraspOutcome(
        scene_id="scene",
        target_object_id=target,
        robot_profile="leap",
        candidate_id=f"candidate-{step}",
        contact_opportunity=np.zeros((2, 3)),
        contact_opportunity_normals=np.zeros((2, 3)),
        q_command=np.zeros(16),
        palm_T_command=np.eye(4),
        active_fingers=np.array([True, True]),
        dynamic_trajectory_evidence={"validated_stages": ["squeeze", "lift", "perturbation"]},
        scene_state_hashes={"initial": state_hash},
        label_stage="dynamic_valid" if passed else "rejected",
        failure_reason="none" if passed else "target_not_lifted",
        recipe_hash="a" * 64,
        protocol_hash="b" * 64,
        source_hash="c" * 64,
    )


def test_each_child_is_revalidated_and_lineage_is_deterministic():
    initial = {"a": _pose(0.0), "b": _pose(1.0)}
    observed_hashes = []

    def run():
        def validate(target, state, step):
            from qdgrasp.dataset.pipeline.validators.scene_dynamic import hash_scene_state

            observed_hashes.append(hash_scene_state(state))
            return _outcome(target, hash_scene_state(state), step)

        return generate_sequential_declutter(
            initial,
            select_target=lambda state, _step: min(state),
            validate_grasp=validate,
            remove_and_resettle=lambda state, target, _step: {
                key: value for key, value in state.items() if key != target
            },
        )

    first = run()
    second = run()
    assert [attempt.target_object_id for attempt in first.attempts] == ["a", "b"]
    assert len(set(observed_hashes[:2])) == 2
    assert [state.lineage_hash for state in first.states] == [
        state.lineage_hash for state in second.states
    ]
    assert first.states[1].parent_lineage_hash == first.states[0].lineage_hash
    assert first.states[2].parent_lineage_hash == first.states[1].lineage_hash


def test_failed_validation_stops_without_removal():
    initial = {"a": _pose(0.0), "b": _pose(1.0)}

    def validate(target, state, step):
        from qdgrasp.dataset.pipeline.validators.scene_dynamic import hash_scene_state

        return _outcome(target, hash_scene_state(state), step, passed=False)

    result = generate_sequential_declutter(
        initial,
        select_target=lambda _state, _step: "a",
        validate_grasp=validate,
        remove_and_resettle=lambda *_args: pytest.fail("failed target must not be removed"),
    )
    assert len(result.states) == 1
    assert not result.attempts[0].passed
    assert result.attempts[0].failure_reason == "target_not_lifted"


def test_outcome_must_bind_to_parent_state_hash():
    initial = {"a": _pose(0.0)}
    result = generate_sequential_declutter(
        initial,
        select_target=lambda _state, _step: "a",
        validate_grasp=lambda target, _state, step: _outcome(target, "0" * 64, step),
        remove_and_resettle=lambda *_args: {},
    )
    assert not result.attempts[0].passed
    assert result.attempts[0].failure_reason == "initial_state_hash_mismatch"


def test_resettle_must_remove_exactly_target_and_cannot_mutate_input():
    initial = {"a": _pose(0.0), "b": _pose(1.0)}
    original = copy.deepcopy(initial)

    def validate(target, state, step):
        from qdgrasp.dataset.pipeline.validators.scene_dynamic import hash_scene_state

        return _outcome(target, hash_scene_state(state), step)

    with pytest.raises(DeclutterError, match="remove exactly"):
        generate_sequential_declutter(
            initial,
            select_target=lambda _state, _step: "a",
            validate_grasp=validate,
            remove_and_resettle=lambda state, _target, _step: state,
        )
    for object_id in original:
        np.testing.assert_array_equal(initial[object_id]["pos"], original[object_id]["pos"])
