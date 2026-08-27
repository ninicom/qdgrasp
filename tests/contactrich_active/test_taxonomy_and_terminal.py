"""S4 — contact identity, self-contact policy and terminal semantics (C03).

The failures pinned here are the ones that let a non-grasp certify.

**B-12**: ``support_assisted`` covered both "the target is resting on the table"
and "a knuckle is resting on the table", so the support-release question could
not be answered, and the self-contact allowlist was the Cartesian product of
every robot geom -- which permits a fingertip driven through the palm.

**B-15**: a primitive whose transition condition never arrived still advanced
when its clock ran out, and the sequence ended as though the condition had been
met.
"""

from __future__ import annotations

import dataclasses
import itertools
from pathlib import Path
from typing import ClassVar

import mujoco
import numpy as np
import pytest

from qdgrasp.dataset.dynamic_contracts import (
    NON_TARGET_PAIRS,
    TARGET_SUPPORTING_PAIRS,
    ContactClass,
    ContactEvent,
    ContactPairKind,
    DynamicGraspTrajectory,
    TrajectoryStage,
    TrajectoryTimebase,
)
from qdgrasp.dynamic.certify import certify_terminal_grasp
from qdgrasp.dynamic.primitives import (
    Primitive,
    PrimitiveKind,
    PrimitiveSequenceController,
    TransitionCondition,
    condition_met,
)
from qdgrasp.dynamic.safety import SceneRoles, classify_contact, classify_pair
from qdgrasp.dynamic.self_contact import (
    SELF_CONTACT_POLICY_SCHEMA,
    SelfContactPolicyError,
    build_self_contact_policy,
    policy_coverage,
    resolve_geom_allowlist,
)
from qdgrasp.robot.spec import RobotSpec, resolve_robot_asset

SAMPLE_PERIOD_S = 0.01


# -- pair taxonomy --------------------------------------------------------


def roles() -> SceneRoles:
    return SceneRoles(
        target_geoms=frozenset({10}),
        support_geoms=frozenset({20}),
        non_target_geoms=frozenset({30}),
        robot_geoms=frozenset({40, 41}),
        self_contact_allowlist=frozenset({(40, 41)}),
        wrist_body=1,
    )


@pytest.mark.parametrize(
    ("geom_a", "geom_b", "expected"),
    [
        (10, 20, ContactPairKind.TARGET_SUPPORT),
        (40, 20, ContactPairKind.ROBOT_SUPPORT),
        (40, 10, ContactPairKind.TARGET_ROBOT),
        (30, 20, ContactPairKind.NON_TARGET_SUPPORT),
        (30, 40, ContactPairKind.NON_TARGET_ROBOT),
        (30, 10, ContactPairKind.NON_TARGET_TARGET),
        (40, 41, ContactPairKind.ROBOT_SELF),
        (99, 10, ContactPairKind.UNKNOWN),
    ],
)
def test_each_pair_of_roles_has_its_own_kind(geom_a, geom_b, expected) -> None:
    assert classify_pair(roles(), geom_a, geom_b) is expected


def test_target_support_and_robot_support_are_different_kinds() -> None:
    scene = roles()
    # Both are treated as support-assisted by the safety budget...
    assert classify_contact(scene, 10, 20) is ContactClass.SUPPORT_ASSISTED
    assert classify_contact(scene, 40, 20) is ContactClass.SUPPORT_ASSISTED
    # ...but only one of them is the target resting on something.
    assert classify_pair(scene, 10, 20) in TARGET_SUPPORTING_PAIRS
    assert classify_pair(scene, 40, 20) not in TARGET_SUPPORTING_PAIRS


def test_non_target_kinds_are_grouped_for_scene_damage() -> None:
    scene = roles()
    for other in (10, 20, 40):
        assert classify_pair(scene, 30, other) in NON_TARGET_PAIRS


def test_an_unknown_role_is_still_forbidden() -> None:
    assert classify_contact(roles(), 99, 10) is ContactClass.FORBIDDEN


def test_self_contact_outside_the_allowlist_is_forbidden() -> None:
    scene = dataclasses.replace(roles(), self_contact_allowlist=frozenset())
    assert classify_contact(scene, 40, 41) is ContactClass.FORBIDDEN


# -- self-contact policy --------------------------------------------------


@pytest.fixture(scope="module", params=["leap_hand", "wonik_allegro"])
def hand(request) -> tuple[str, RobotSpec]:
    return request.param, RobotSpec.from_config(f"{request.param}.yaml", sample_anchors=False)


def test_policy_comes_from_the_profile_and_is_versioned(hand) -> None:
    name, spec = hand
    policy = build_self_contact_policy(spec, robot_profile=name)
    assert policy.schema == SELF_CONTACT_POLICY_SCHEMA
    assert len(policy.policy_hash) == 64
    assert policy.finger_links
    assert policy.robot_profile == name


def test_policy_hash_changes_when_a_pair_is_added(hand) -> None:
    name, spec = hand
    policy = build_self_contact_policy(spec, robot_profile=name)
    widened = dataclasses.replace(
        policy, allowed_link_pairs=policy.allowed_link_pairs | {("zzz_a", "zzz_b")}
    )
    assert widened.policy_hash != policy.policy_hash


def test_a_finger_may_not_fold_onto_a_distant_link_of_itself(hand) -> None:
    name, spec = hand
    policy = build_self_contact_policy(spec, robot_profile=name)
    refused = 0
    for chain in policy.finger_links.values():
        for i, link_a in enumerate(chain):
            for link_b in chain[i + 2 :]:
                assert not policy.permits(link_a, link_b), (link_a, link_b)
                refused += 1
    assert refused > 0, "the profile has no non-adjacent within-finger pairs to refuse"


def test_adjacent_links_and_different_fingers_are_permitted(hand) -> None:
    name, spec = hand
    policy = build_self_contact_policy(spec, robot_profile=name)
    chains = list(policy.finger_links.values())
    for chain in chains:
        for parent, child in itertools.pairwise(chain):
            assert policy.permits(parent, child)
    # A pinch between two fingers is normal dexterous behaviour.
    assert policy.permits(chains[0][-1], chains[1][-1])


def test_every_link_may_touch_the_palm(hand) -> None:
    name, spec = hand
    policy = build_self_contact_policy(spec, robot_profile=name)
    for chain in policy.finger_links.values():
        for link in chain:
            assert policy.permits(link, spec.palm_link)


def test_a_profile_without_a_palm_cannot_produce_a_policy() -> None:
    class Bare:
        links: ClassVar[dict[str, object]] = {"a": None}
        palm_link = None
        base_link = None
        wrist_link = None
        fingertip_links = ()

    with pytest.raises(SelfContactPolicyError, match="palm or base"):
        build_self_contact_policy(Bare(), robot_profile="bare")


def test_the_policy_admits_less_than_the_cartesian_product(hand) -> None:
    name, spec = hand
    model = mujoco.MjModel.from_xml_path(str(resolve_robot_asset(spec.config.source_asset)))
    robot_geoms = frozenset(range(int(model.ngeom)))
    policy = build_self_contact_policy(spec, robot_profile=name)
    coverage = policy_coverage(model, policy, robot_geoms)

    allowed = resolve_geom_allowlist(model, policy, robot_geoms)
    assert coverage["allowed_pairs"] < coverage["candidate_pairs"]
    assert coverage["allowed_fraction"] < 1.0
    # The refused pairs are the ones a hand cannot reach without going through
    # itself; they are refused by identity, not by measured force.
    assert len(allowed) == int(coverage["allowed_pairs"])


# -- transition timeout ---------------------------------------------------


def _primitive(until: TransitionCondition, duration: float = 0.05) -> Primitive:
    return Primitive(
        kind=PrimitiveKind.CAGE,
        direction=np.array([0.0, 0.0, 1.0]),
        speed=0.0,
        max_duration_s=duration,
        grip=0.5,
        until=until,
    )


def test_a_condition_that_never_arrives_is_recorded_as_a_timeout() -> None:
    controller = PrimitiveSequenceController(
        (_primitive(TransitionCondition.ENCLOSURE_REACHED),), control_dt=0.01
    )
    steps = []
    while not controller.finished:
        steps.append(controller.step(events=()))
    assert steps[-1].timed_out is True
    assert steps[-1].timeout_reason == "transition_timeout:enclosure_reached"
    assert controller.timeouts == ("enclosure_reached",)
    assert controller.first_timeout_reason == "transition_timeout:enclosure_reached"


def test_a_duration_primitive_running_its_course_is_not_a_timeout() -> None:
    controller = PrimitiveSequenceController(
        (_primitive(TransitionCondition.DURATION_ELAPSED),), control_dt=0.01
    )
    while not controller.finished:
        step = controller.step(events=())
    assert step.timed_out is False
    assert controller.timeouts == ()


def test_a_condition_that_is_met_is_not_a_timeout() -> None:
    controller = PrimitiveSequenceController(
        (_primitive(TransitionCondition.TARGET_CONTACT_MADE, duration=1.0),), control_dt=0.01
    )
    touching = (
        ContactEvent(
            time_index=0,
            contact_class=ContactClass.TARGET_INTENTIONAL,
            geom_a="tip",
            geom_b="target",
            body_a="distal",
            body_b="target",
            point=np.zeros(3),
            frame=np.eye(3),
            normal_force_N=1.0,
            tangential_force_N=0.0,
            normal_impulse_Ns=0.0,
            tangential_impulse_Ns=0.0,
            penetration_m=0.0,
            relative_velocity_mps=0.0,
            slip_m=0.0,
            work_J=0.0,
            budget_margin=0.5,
            pair_kind=ContactPairKind.TARGET_ROBOT,
        ),
    )
    step = controller.step(events=touching)
    assert step.advanced is True
    assert step.timed_out is False
    assert controller.timeouts == ()


def test_support_release_ignores_a_robot_link_on_the_table() -> None:
    common = {"elapsed_s": 0.0, "max_duration_s": 1.0, "required_contacts": 2}
    knuckle_down = (
        ContactEvent(
            time_index=0,
            contact_class=ContactClass.SUPPORT_ASSISTED,
            geom_a="knuckle",
            geom_b="table",
            body_a="proximal",
            body_b="table",
            point=np.zeros(3),
            frame=np.eye(3),
            normal_force_N=1.0,
            tangential_force_N=0.0,
            normal_impulse_Ns=0.0,
            tangential_impulse_Ns=0.0,
            penetration_m=0.0,
            relative_velocity_mps=0.0,
            slip_m=0.0,
            work_J=0.0,
            budget_margin=0.5,
            pair_kind=ContactPairKind.ROBOT_SUPPORT,
        ),
    )
    assert condition_met(TransitionCondition.SUPPORT_RELEASED, events=knuckle_down, **common)


# -- terminal semantics ---------------------------------------------------


def _event(time_index: int, kind: ContactPairKind, contact_class: ContactClass, **over) -> ContactEvent:
    defaults = {
        "time_index": time_index,
        "contact_class": contact_class,
        "geom_a": "tip",
        "geom_b": "target",
        "body_a": "distal_0",
        "body_b": "target",
        "point": np.zeros(3),
        "frame": np.eye(3),
        "normal_force_N": 1.0,
        "tangential_force_N": 0.0,
        "normal_impulse_Ns": 0.0,
        "tangential_impulse_Ns": 0.0,
        "penetration_m": 0.0,
        "relative_velocity_mps": 0.0,
        "slip_m": 0.0,
        "work_J": 0.0,
        "budget_margin": 0.5,
        "pair_kind": kind,
    }
    defaults.update(over)
    return ContactEvent(**defaults)


def _trajectory(
    *,
    steps: int = 4,
    objects: int = 2,
    lifts: dict[int, float] | None = None,
    events: tuple[ContactEvent, ...] = (),
    stage: tuple[TrajectoryStage, ...] | None = None,
) -> DynamicGraspTrajectory:
    palm = np.zeros((steps, 7))
    palm[:, 3] = 1.0
    pose = np.zeros((steps, objects, 7))
    pose[:, :, 3] = 1.0
    for slot, rise in (lifts or {}).items():
        pose[-1, slot, 2] = pose[0, slot, 2] + rise
    return DynamicGraspTrajectory(
        time=np.arange(steps, dtype=float) * SAMPLE_PERIOD_S,
        palm_pose=palm,
        joint_state=np.zeros((steps, 16)),
        actuator_command=np.zeros((steps, 16)),
        object_pose=pose,
        object_velocity=np.zeros((steps, objects, 6)),
        stage=stage or tuple([TrajectoryStage.APPROACH] * steps),
        timebase=TrajectoryTimebase(simulator_dt=SAMPLE_PERIOD_S, sample_every=1),
        contact_graph=events,
        robot_profile="leap_hand",
        palm_body="palm",
    )


def _enclosing(count: int = 2, time_index: int = 0) -> tuple[ContactEvent, ...]:
    return tuple(
        _event(time_index, ContactPairKind.TARGET_ROBOT, ContactClass.TARGET_INTENTIONAL,
               body_a=f"distal_{i}")
        for i in range(count)
    )


def test_a_hand_resting_on_the_table_does_not_block_the_certificate() -> None:
    events = (
        *_enclosing(),
        _event(3, ContactPairKind.ROBOT_SUPPORT, ContactClass.SUPPORT_ASSISTED,
               geom_a="knuckle", geom_b="table", body_a="proximal_0", body_b="table"),
    )
    result = certify_terminal_grasp(_trajectory(lifts={0: 0.05}, events=events))
    assert result.certified, result.reason


def test_the_target_still_on_its_support_is_refused() -> None:
    events = (
        *_enclosing(),
        _event(3, ContactPairKind.TARGET_SUPPORT, ContactClass.SUPPORT_ASSISTED,
               geom_a="target_geom", geom_b="table", body_a="target", body_b="table"),
    )
    result = certify_terminal_grasp(_trajectory(lifts={0: 0.05}, events=events))
    assert not result.certified
    assert result.reason == "support_not_released"


def test_lifting_the_wrong_object_is_its_own_negative() -> None:
    result = certify_terminal_grasp(
        _trajectory(lifts={0: 0.0, 1: 0.10}, events=_enclosing())
    )
    assert not result.certified
    assert result.reason == "wrong_object_lift"
    assert result.metrics["max_non_target_lift_m"] == pytest.approx(0.10)


def test_a_target_lift_alongside_a_neighbour_lift_is_still_refused() -> None:
    result = certify_terminal_grasp(
        _trajectory(lifts={0: 0.06, 1: 0.06}, events=_enclosing())
    )
    assert not result.certified
    assert result.reason == "wrong_object_lift"


def test_an_unresolved_pair_cannot_be_read_as_released() -> None:
    events = (
        *_enclosing(),
        _event(3, ContactPairKind.UNKNOWN, ContactClass.FORBIDDEN,
               geom_a="mystery", geom_b="target"),
    )
    result = certify_terminal_grasp(_trajectory(lifts={0: 0.05}, events=events))
    assert not result.certified


def test_stage_progression_is_checked_when_asked() -> None:
    scrambled = (
        TrajectoryStage.LIFT,
        TrajectoryStage.ENCLOSE,
        TrajectoryStage.SUPPORT_RELEASE,
        TrajectoryStage.PERTURB,
    )
    ordered = (
        TrajectoryStage.ENCLOSE,
        TrajectoryStage.SUPPORT_RELEASE,
        TrajectoryStage.LIFT,
        TrajectoryStage.PERTURB,
    )
    bad = certify_terminal_grasp(
        _trajectory(lifts={0: 0.05}, events=_enclosing(), stage=scrambled),
        require_stage_progression=True,
    )
    assert not bad.certified
    assert bad.reason == "no_closure"

    good = certify_terminal_grasp(
        _trajectory(lifts={0: 0.05}, events=_enclosing(), stage=ordered),
        require_stage_progression=True,
    )
    assert good.certified, good.reason


def test_a_support_only_hold_never_certifies() -> None:
    # The object never left the table and nothing enclosed it.
    events = (
        _event(3, ContactPairKind.TARGET_SUPPORT, ContactClass.SUPPORT_ASSISTED,
               geom_a="target_geom", geom_b="table", body_a="target", body_b="table"),
    )
    result = certify_terminal_grasp(_trajectory(events=events))
    assert not result.certified
    assert result.reason == "insufficient_enclosure"


def test_the_primitive_vocabulary_never_writes_an_object_pose() -> None:
    # A primitive that set object pose would manufacture the result the whole
    # phase exists to measure (C03.5).
    source = (
        Path(__file__).resolve().parents[2] / "qdgrasp" / "dynamic" / "primitives.py"
    ).read_text(encoding="utf-8")
    assert "qpos" not in source
    assert not [f for f in dataclasses.fields(Primitive) if "object" in f.name]
