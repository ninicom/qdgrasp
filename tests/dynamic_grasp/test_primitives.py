"""Primitive-sequence controller tests (P3.4-07)."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from qdgrasp.dataset.dynamic_contracts import ContactClass, TrajectoryStage
from qdgrasp.dynamic.primitives import (
    Primitive,
    PrimitiveKind,
    PrimitiveSequenceController,
    TransitionCondition,
    condition_met,
    table_pivot_sequence,
)

from .conftest import make_event


def push(**overrides) -> Primitive:
    kwargs = {
        "kind": PrimitiveKind.PUSH,
        "direction": np.array([1.0, 0.0, 0.0]),
        "speed": 0.05,
        "max_duration_s": 0.1,
    }
    kwargs.update(overrides)
    return Primitive(**kwargs)


def test_a_primitive_cannot_carry_a_target_object_pose():
    # The whole phase rests on the target moving because contact moved it, so a
    # primitive must have no field that sets where the object ends up.
    fields = {f.name for f in dataclasses.fields(Primitive)}
    for forbidden in ("object_pose", "target_pose", "goal_pose", "object_position"):
        assert forbidden not in fields


def test_direction_is_normalised_and_degenerate_input_is_rejected():
    assert np.allclose(push(direction=np.array([3.0, 0.0, 0.0])).direction, [1.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="degenerate"):
        push(direction=np.zeros(3))
    with pytest.raises(ValueError, match=r"direction must be \[3\]"):
        push(direction=np.zeros(4))


@pytest.mark.parametrize(
    ("field", "bad"),
    [("speed", -0.1), ("max_duration_s", 0.0), ("grip", 1.5), ("required_contacts", 0)],
)
def test_primitive_rejects_out_of_range_parameters(field, bad):
    with pytest.raises(ValueError, match=field):
        push(**{field: bad})


def test_wrist_velocity_is_direction_times_speed():
    assert np.allclose(
        push(direction=np.array([0.0, 2.0, 0.0]), speed=0.25).wrist_velocity(),
        [0.0, 0.25, 0.0],
    )


def test_every_primitive_kind_maps_to_a_stage():
    for kind in PrimitiveKind:
        assert isinstance(push(kind=kind).stage, TrajectoryStage)


# -- transition conditions ------------------------------------------------


def test_duration_condition_needs_the_full_duration():
    common = {"events": (), "max_duration_s": 0.1, "required_contacts": 2}
    assert not condition_met(TransitionCondition.DURATION_ELAPSED, elapsed_s=0.05, **common)
    assert condition_met(TransitionCondition.DURATION_ELAPSED, elapsed_s=0.1, **common)


def test_target_contact_conditions_are_complementary():
    made = TransitionCondition.TARGET_CONTACT_MADE
    lost = TransitionCondition.TARGET_CONTACT_LOST
    common = {"elapsed_s": 0.0, "max_duration_s": 1.0, "required_contacts": 2}
    none = ()
    touching = (make_event(contact_class=ContactClass.TARGET_INTENTIONAL),)
    assert not condition_met(made, events=none, **common)
    assert condition_met(lost, events=none, **common)
    assert condition_met(made, events=touching, **common)
    assert not condition_met(lost, events=touching, **common)


def test_a_damaging_target_contact_still_counts_as_contact():
    # Reclassification to DAMAGING is a safety verdict; the finger is still
    # touching, so the transition logic must not pretend it is not.
    events = (make_event(contact_class=ContactClass.DAMAGING),)
    assert condition_met(
        TransitionCondition.TARGET_CONTACT_MADE,
        events=events, elapsed_s=0.0, max_duration_s=1.0, required_contacts=2,
    )


def test_support_released_requires_no_support_contact():
    common = {"elapsed_s": 0.0, "max_duration_s": 1.0, "required_contacts": 2}
    on_table = (make_event(contact_class=ContactClass.SUPPORT_ASSISTED),)
    assert not condition_met(TransitionCondition.SUPPORT_RELEASED, events=on_table, **common)
    assert condition_met(TransitionCondition.SUPPORT_RELEASED, events=(), **common)


def test_enclosure_counts_distinct_robot_links_not_contacts():
    common = {"elapsed_s": 0.0, "max_duration_s": 1.0, "required_contacts": 2}
    # Two contacts from the same link must not satisfy a two-link requirement.
    same_link = tuple(
        make_event(body_a="finger_0", body_b="target", geom_a=f"tip0_{i}")
        for i in range(3)
    )
    assert not condition_met(TransitionCondition.ENCLOSURE_REACHED, events=same_link, **common)

    two_links = (
        make_event(body_a="finger_0", body_b="target"),
        make_event(body_a="finger_1", body_b="target"),
    )
    assert condition_met(TransitionCondition.ENCLOSURE_REACHED, events=two_links, **common)


# -- sequencing -----------------------------------------------------------


def test_controller_rejects_an_empty_sequence_or_bad_timestep():
    with pytest.raises(ValueError, match="at least one primitive"):
        PrimitiveSequenceController([], 0.01)
    with pytest.raises(ValueError, match="control_dt"):
        PrimitiveSequenceController([push()], 0.0)


def test_a_primitive_yields_when_its_condition_is_observed():
    controller = PrimitiveSequenceController(
        [push(until=TransitionCondition.TARGET_CONTACT_MADE, max_duration_s=10.0), push()],
        control_dt=0.01,
    )
    first = controller.step(events=())
    assert not first.advanced, "no contact yet, so it must keep pushing"

    second = controller.step(events=(make_event(contact_class=ContactClass.TARGET_INTENTIONAL),))
    assert second.advanced
    assert controller.current is controller.sequence[1]


def test_a_primitive_whose_condition_never_holds_still_times_out():
    # Without the ceiling a search would stall forever on a precondition the
    # physics will not produce.
    controller = PrimitiveSequenceController(
        [push(until=TransitionCondition.TARGET_CONTACT_MADE, max_duration_s=0.03)],
        control_dt=0.01,
    )
    assert not controller.step(events=()).advanced
    assert not controller.step(events=()).advanced
    final = controller.step(events=())
    assert final.advanced and final.finished
    assert controller.finished


def test_stepping_an_exhausted_sequence_is_an_error_not_a_silent_noop():
    controller = PrimitiveSequenceController([push(max_duration_s=0.01)], control_dt=0.01)
    controller.step(events=())
    assert controller.finished
    with pytest.raises(RuntimeError, match="exhausted"):
        controller.step(events=())


def test_reset_restarts_the_sequence():
    controller = PrimitiveSequenceController([push(max_duration_s=0.01), push()], 0.01)
    controller.step(events=())
    assert controller.current is controller.sequence[1]
    controller.reset()
    assert controller.current is controller.sequence[0]


def test_elapsed_time_restarts_per_primitive():
    controller = PrimitiveSequenceController(
        [push(max_duration_s=0.02), push(max_duration_s=0.02)], control_dt=0.01
    )
    controller.step(events=())
    controller.step(events=())  # first advances here
    assert controller.current is controller.sequence[1]
    third = controller.step(events=())
    assert not third.advanced, "the second primitive must get its own full duration"


def test_reference_sequence_walks_reposition_to_lift():
    sequence = table_pivot_sequence(np.array([1.0, 0.0, 0.0]))
    stages = [p.stage for p in sequence]
    assert stages[0] is TrajectoryStage.REPOSITION
    assert TrajectoryStage.ENCLOSE in stages
    assert stages[-1] is TrajectoryStage.LIFT
    assert all(p.grip <= 1.0 for p in sequence)
