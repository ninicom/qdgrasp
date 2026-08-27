"""Contact observation and safety budget tests (P3.4-06).

Driven by real MuJoCo contacts on the shared micro scene, because the point of
the observer is that its numbers come from the solver.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import mujoco
import numpy as np
import pytest

from qdgrasp.dataset.dynamic_contracts import ContactClass
from qdgrasp.dynamic.safety import (
    ContactObserver,
    SceneRoles,
    budget_margin,
    classify_contact,
    summarise_safety,
)

MICRO_SCENE = (Path(__file__).parent / "micro_scene.xml").read_text(encoding="utf-8")


@pytest.fixture
def model() -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_string(MICRO_SCENE)


def gid(model: mujoco.MjModel, name: str) -> int:
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)


def bid(model: mujoco.MjModel, name: str) -> int:
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)


@pytest.fixture
def roles(model: mujoco.MjModel) -> SceneRoles:
    return SceneRoles(
        target_geoms=frozenset({gid(model, "target_geom")}),
        support_geoms=frozenset({gid(model, "table")}),
        non_target_geoms=frozenset(),
        robot_geoms=frozenset({gid(model, "pusher_geom")}),
        # The micro scene has no separate wrist link, so the wrist budget
        # resolves at the body the stage is driven through. Naming it is what
        # makes the wrist limits measurable at all.
        wrist_body=bid(model, "pusher"),
        palm_body=bid(model, "pusher"),
    )


# -- classification ------------------------------------------------------


def test_robot_touching_target_is_intentional(model, roles):
    assert classify_contact(roles, gid(model, "pusher_geom"), gid(model, "target_geom")) is (
        ContactClass.TARGET_INTENTIONAL
    )


def test_anything_touching_a_support_is_support_assisted(model, roles):
    assert classify_contact(roles, gid(model, "target_geom"), gid(model, "table")) is (
        ContactClass.SUPPORT_ASSISTED
    )
    assert classify_contact(roles, gid(model, "pusher_geom"), gid(model, "table")) is (
        ContactClass.SUPPORT_ASSISTED
    )


def test_robot_self_contact_is_forbidden_unless_allowlisted(model, roles):
    pusher = gid(model, "pusher_geom")
    assert classify_contact(roles, pusher, pusher) is ContactClass.FORBIDDEN
    permissive = dataclasses.replace(
        roles, self_contact_allowlist=frozenset({(pusher, pusher)})
    )
    assert classify_contact(permissive, pusher, pusher) is ContactClass.SELF_CONTACT_ALLOWED


def test_an_unclassified_geom_is_forbidden_not_waved_through(model, roles):
    # A geom with no declared role means the budget has nothing to say about it,
    # so it must not be admitted by default.
    unknown = 999
    assert classify_contact(roles, gid(model, "pusher_geom"), unknown) is ContactClass.FORBIDDEN


def test_forbidden_pairs_beat_every_other_role(model, roles):
    pair = tuple(sorted((gid(model, "pusher_geom"), gid(model, "target_geom"))))
    strict = dataclasses.replace(roles, forbidden_pairs=frozenset({pair}))
    assert classify_contact(strict, *pair) is ContactClass.FORBIDDEN


# -- budget margin -------------------------------------------------------


def test_margin_is_positive_below_every_limit(budget):
    margin = budget_margin(
        budget,
        normal_force_N=1.0,
        tangential_force_N=0.1,
        normal_impulse_Ns=0.01,
        tangential_impulse_Ns=0.01,
        penetration_m=0.0001,
        work_J=0.001,
        duration_s=0.1,
    )
    assert 0.0 < margin < 1.0


def test_margin_is_driven_by_the_worst_quantity(budget):
    # Penetration alone at the limit must zero the margin even when every other
    # quantity is tiny: a single blown limit is not averaged away.
    margin = budget_margin(
        budget,
        normal_force_N=0.01,
        tangential_force_N=0.01,
        normal_impulse_Ns=0.0,
        tangential_impulse_Ns=0.0,
        penetration_m=budget.max_penetration_m,
        work_J=0.0,
        duration_s=0.0,
    )
    assert margin == pytest.approx(0.0)


def test_margin_goes_negative_when_a_limit_is_exceeded(budget):
    margin = budget_margin(
        budget,
        normal_force_N=budget.peak_normal_force_N * 2.0,
        tangential_force_N=0.0,
        normal_impulse_Ns=0.0,
        tangential_impulse_Ns=0.0,
        penetration_m=0.0,
        work_J=0.0,
        duration_s=0.0,
    )
    assert margin == pytest.approx(-1.0)


# -- observation on real physics -----------------------------------------


def settle(model: mujoco.MjModel, steps: int = 60) -> mujoco.MjData:
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    for _ in range(steps):
        mujoco.mj_step(model, data)
    return data


def test_observer_reads_real_contacts_and_names_them(model, roles, budget):
    data = settle(model)
    observer = ContactObserver(model, roles, budget)
    events = observer.observe(data, time_index=0, dt=model.opt.timestep)

    assert events, "the resting box must produce a support contact"
    names = {tuple(sorted((e.geom_a, e.geom_b))) for e in events}
    assert ("table", "target_geom") in names
    for event in events:
        assert np.isfinite(event.normal_force_N)
        assert event.penetration_m >= 0.0
        assert event.frame.shape == (3, 3)


def test_a_resting_box_is_support_assisted_and_within_budget(model, roles, budget):
    data = settle(model)
    observer = ContactObserver(model, roles, budget)
    events = observer.observe(data, time_index=0, dt=model.opt.timestep)
    support = [e for e in events if {e.geom_a, e.geom_b} == {"table", "target_geom"}]
    assert support
    assert all(e.contact_class is ContactClass.SUPPORT_ASSISTED for e in support)
    assert all(e.budget_margin > 0.0 for e in support), "a 50 g box must not blow the budget"


def test_impulse_and_duration_accumulate_across_steps(model, roles, budget):
    data = settle(model)
    observer = ContactObserver(model, roles, budget)
    first = observer.observe(data, time_index=0, dt=0.002)
    for _ in range(10):
        mujoco.mj_step(model, data)
    later = observer.observe(data, time_index=1, dt=0.002)

    def support_of(events):
        return next(e for e in events if {e.geom_a, e.geom_b} == {"table", "target_geom"})

    assert support_of(later).normal_impulse_Ns > support_of(first).normal_impulse_Ns
    assert support_of(later).duration_s > support_of(first).duration_s


def test_reset_clears_accumulated_history(model, roles, budget):
    def support_impulse(events) -> float:
        # Select the load-bearing contact by identity: contact order is not
        # guaranteed, and the pusher rests against the table at ~zero force.
        return next(
            e for e in events if {e.geom_a, e.geom_b} == {"table", "target_geom"}
        ).normal_impulse_Ns

    data = settle(model)
    observer = ContactObserver(model, roles, budget)
    for step in range(5):
        observer.observe(data, time_index=step, dt=0.002)
    before = support_impulse(observer.observe(data, time_index=5, dt=0.002))
    assert before > 0.0, "a resting 50 g box must carry a measurable normal impulse"

    observer.reset()
    after = support_impulse(observer.observe(data, time_index=0, dt=0.002))
    assert after < before


def test_a_permitted_contact_that_blows_the_budget_becomes_damaging(model, roles):
    # Same physics, an absurdly tight budget: the class must follow the measured
    # force, not the geometry alone.
    from qdgrasp.dataset.dynamic_contracts import ContactSafetyBudget

    tight = ContactSafetyBudget(
        budget_id="tight",
        robot_profile="micro_pusher",
        peak_normal_force_N=1e-6,
        peak_tangential_force_N=1e-6,
        normal_impulse_Ns=1e-9,
        tangential_impulse_Ns=1e-9,
        contact_duration_s=1e-6,
        contact_work_J=1e-9,
        max_penetration_m=1e-9,
        max_wrist_force_N=1.0,
        max_wrist_torque_Nm=1.0,
        max_joint_or_tendon_load=1.0,
        max_non_target_translation_m=1e-6,
        max_non_target_rotation_rad=1e-6,
        max_non_target_velocity_mps=1e-6,
    )
    data = settle(model)
    events = ContactObserver(model, roles, tight).observe(data, time_index=0, dt=0.002)
    assert events
    assert all(e.contact_class is ContactClass.DAMAGING for e in events)
    assert all(e.budget_margin < 0.0 for e in events)
    assert all(e.is_hard_reject for e in events)


def test_observer_rejects_a_nonpositive_timestep(model, roles, budget):
    data = settle(model)
    with pytest.raises(ValueError, match="dt must be positive"):
        ContactObserver(model, roles, budget).observe(data, time_index=0, dt=0.0)


def test_summary_reports_peak_and_cumulative_without_double_counting(model, roles, budget):
    data = settle(model)
    observer = ContactObserver(model, roles, budget)
    collected = []
    for step in range(8):
        collected.extend(observer.observe(data, time_index=step, dt=0.002))
        mujoco.mj_step(model, data)

    peak, cumulative = summarise_safety(collected)
    assert peak["peak_normal_force_N"] >= 0.0
    assert peak["min_budget_margin"] <= 1.0
    # Impulse accumulates inside the observer, so the cumulative figure must be
    # the final value per pair rather than a sum over every step.
    per_step_sum = sum(e.normal_impulse_Ns for e in collected)
    assert cumulative["normal_impulse_Ns"] < per_step_sum


def test_summary_of_an_empty_stream_is_empty():
    assert summarise_safety([]) == ({}, {})
