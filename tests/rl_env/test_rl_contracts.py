"""P3.5-09/12: the contracts that keep a measured result measurable.

These are cheap tests of expensive mistakes.  A ``terminated`` that also means
"the clock ran out" teaches a value function that the horizon is a property of
the state; a reward whose total does not equal its logged terms cannot be
audited; an evaluation split that reshuffles when the corpus grows stops being
held out.  Each of those is one assertion here.
"""

from __future__ import annotations

import numpy as np
import pytest

from qdgrasp.rl.contracts import (
    BoxSpace,
    ObservationField,
    ObservationSchema,
    RewardBreakdown,
    RlActionSpec,
    StepResult,
    TerminalReason,
)
from qdgrasp.rl.randomization import (
    STREAM_NAMES,
    DomainRandomization,
    EvaluationSplit,
    Range,
    SeedStreams,
    assert_no_split_leak,
    bucket_by_signature,
    scene_signature,
)
from qdgrasp.scenes.contracts import SceneObjectSpec, SceneSpec

# -- spaces and schema -----------------------------------------------------


def test_a_space_needs_a_positive_shape_and_ordered_bounds() -> None:
    BoxSpace("ok", (3,), -1.0, 1.0).validate()
    with pytest.raises(ValueError):
        BoxSpace("bad", (0,), -1.0, 1.0).validate()
    with pytest.raises(ValueError):
        BoxSpace("bad", (3,), 1.0, -1.0).validate()


def test_observation_fields_must_declare_a_unit_and_a_frame() -> None:
    with pytest.raises(ValueError):
        ObservationSchema(fields=(ObservationField("a", 3, "", "world"),)).validate()
    with pytest.raises(ValueError):
        ObservationSchema(fields=(ObservationField("a", 3, "m", ""),)).validate()
    with pytest.raises(ValueError):
        ObservationSchema(
            fields=(ObservationField("a", 3, "m", "world"), ObservationField("a", 1, "m", "world"))
        ).validate()


def test_the_schema_assembles_in_order_and_checks_sizes() -> None:
    schema = ObservationSchema(fields=(ObservationField("a", 2, "m", "world"), ObservationField("b", 3, "rad", "palm")))
    schema.validate()
    np.testing.assert_array_equal(schema.assemble({"a": [1, 2], "b": [3, 4, 5]}), [1, 2, 3, 4, 5])
    assert schema.offset_of("b") == slice(2, 5)
    with pytest.raises(ValueError):
        schema.assemble({"a": [1, 2, 3], "b": [4, 5, 6]})
    with pytest.raises(KeyError):
        schema.assemble({"a": [1, 2]})


def test_the_schema_hash_changes_with_the_layout() -> None:
    left = ObservationSchema(fields=(ObservationField("a", 2, "m", "world"),))
    right = ObservationSchema(fields=(ObservationField("a", 3, "m", "world"),))
    assert left.content_hash() != right.content_hash()


# -- action spec -----------------------------------------------------------


def test_the_action_dimension_follows_the_palm_and_joint_commands() -> None:
    spec = RlActionSpec(joint_names=("a", "b", "c"), active_joint_mask=(True, True, False), control_dt=0.02)
    spec.validate()
    assert spec.dimension == 6 + 2
    fixed = RlActionSpec(joint_names=("a",), active_joint_mask=(True,), control_dt=0.02, palm_command="fixed")
    assert fixed.dimension == 1


def test_an_action_spec_with_no_active_joint_is_refused() -> None:
    with pytest.raises(ValueError):
        RlActionSpec(joint_names=("a",), active_joint_mask=(False,), control_dt=0.02).validate()
    with pytest.raises(ValueError):
        RlActionSpec(joint_names=("a", "b"), active_joint_mask=(True,), control_dt=0.02).validate()


# -- reward ----------------------------------------------------------------


def test_the_total_is_the_sum_of_the_logged_terms() -> None:
    reward = RewardBreakdown({"reach": 1.5, "lift": 0.5, "action_rate": -0.25})
    assert reward.total == pytest.approx(1.75)
    document = reward.to_document()
    assert document["total"] == pytest.approx(sum(v for k, v in document.items() if k != "total"))


def test_a_penalty_term_may_not_be_positive() -> None:
    """A positive penalty would let a barrier be paid for rather than respected."""

    with pytest.raises(ValueError, match="non-positive"):
        RewardBreakdown({"penetration": 1.0})
    with pytest.raises(ValueError, match="unknown"):
        RewardBreakdown({"free_points": 1.0})
    with pytest.raises(ValueError, match="finite"):
        RewardBreakdown({"reach": float("nan")})


# -- step result -----------------------------------------------------------


def test_terminated_and_truncated_are_mutually_exclusive() -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        StepResult(np.zeros(2), 0.0, True, True, {"terminal_reason": TerminalReason.SUCCESS})


def test_a_terminated_step_must_name_its_reason() -> None:
    with pytest.raises(ValueError, match="terminal reason"):
        StepResult(np.zeros(2), 0.0, True, False, {})
    with pytest.raises(ValueError, match="terminal reason"):
        StepResult(np.zeros(2), 0.0, True, False, {"terminal_reason": TerminalReason.NONE})
    StepResult(np.zeros(2), 0.0, True, False, {"terminal_reason": TerminalReason.OBJECT_DROPPED})


def test_a_truncated_step_reports_the_horizon() -> None:
    with pytest.raises(ValueError, match="horizon"):
        StepResult(np.zeros(2), 0.0, False, True, {"terminal_reason": TerminalReason.SUCCESS})
    StepResult(np.zeros(2), 0.0, False, True, {"terminal_reason": TerminalReason.HORIZON})


# -- randomization ---------------------------------------------------------


def test_seed_streams_are_independent_and_reproducible() -> None:
    streams = SeedStreams(episode_seed=7)
    first = {name: streams.generator(name).random() for name in STREAM_NAMES}
    second = {name: streams.generator(name).random() for name in STREAM_NAMES}
    assert first == second
    assert len(set(first.values())) == len(STREAM_NAMES), "streams must not be correlated"
    other = SeedStreams(episode_seed=8)
    assert other.generator("scene").random() != first["scene"]
    with pytest.raises(KeyError):
        streams.generator("not_a_stream")


def test_randomization_ranges_are_validated_and_hashed() -> None:
    randomization = DomainRandomization(mass_scale=Range(0.5, 2.0), friction_slide=Range(0.4, 1.2))
    randomization.validate()
    sample = randomization.sample(np.random.default_rng(0))
    assert 0.5 <= sample["mass_scale"] <= 2.0
    assert 0.4 <= sample["friction_slide"] <= 1.2
    assert randomization.content_hash() != DomainRandomization().content_hash()
    with pytest.raises(ValueError):
        DomainRandomization(mass_scale=Range(2.0, 0.5)).validate()
    with pytest.raises(ValueError):
        DomainRandomization(mass_scale=Range(0.0, 1.0)).validate()


def _spec(scene_id: str, asset_ref: str, position: float) -> SceneSpec:
    transform = np.eye(4)
    transform[0, 3] = position
    return SceneSpec(
        scene_id=scene_id,
        source_dataset="test",
        source_version="v1",
        source_split="train",
        environment="table",
        objects=[SceneObjectSpec(object_id="a", asset_ref=asset_ref, T_world_object=transform)],
    )


def test_the_scene_signature_ignores_pose_but_not_topology() -> None:
    """Two scenes that differ only in where things sit can share a compiled model."""

    left = _spec("one", "asset_a.obj", 0.0)
    right = _spec("two", "asset_a.obj", 0.25)
    assert scene_signature(left) == scene_signature(right)

    different_asset = _spec("three", "asset_b.obj", 0.0)
    assert scene_signature(left) != scene_signature(different_asset)
    assert scene_signature(left) != scene_signature(left, robot_profile="leap_hand.yaml")

    buckets = bucket_by_signature([left, right, different_asset])
    assert sorted(len(indices) for indices in buckets.values()) == [1, 2]


def test_split_membership_does_not_move_when_the_corpus_grows() -> None:
    split = EvaluationSplit(name="eval", fraction=0.3)
    keys = [f"object_{index}" for index in range(200)]
    before = {key: split.contains(key) for key in keys}
    keys.extend(f"object_{index}" for index in range(200, 400))
    after = {key: split.contains(key) for key in keys}
    assert all(after[key] == value for key, value in before.items())
    assert 0 < sum(before.values()) < len(before)


def test_a_split_leak_is_refused() -> None:
    assert_no_split_leak(["a", "b"], ["c"])
    with pytest.raises(ValueError, match="leaked"):
        assert_no_split_leak(["a", "b"], ["b", "c"])
