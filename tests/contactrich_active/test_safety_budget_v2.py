"""S3 — every declared limit is measured, and the impulse window is a window.

Two failures are pinned here.

**B-01**: six of the thirteen declared limits had no sensor. The coverage tests
assert the mapping is total, that a budget whose limits a model cannot measure
fails preflight, and that mutating any single threshold produces that field's
own failure -- not a generic one, and not silence.

**B-02**: impulse was accumulated into a block that reset when it filled, so an
impact landing across the reset was split in two and both halves passed. The
window tests drive an identical impulse at different offsets and require the
same verdict.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import mujoco
import numpy as np
import pytest

from qdgrasp.dataset.dynamic_contracts import ContactSafetyBudget
from qdgrasp.dynamic.safety import ContactObserver, SceneRoles
from qdgrasp.dynamic.safety_budget import (
    SAFETY_FIELD_SPECS,
    Aggregation,
    SafetyCoverageError,
    SensorScope,
    coverage_matrix,
    evaluate_budget,
    missing_specs,
    require_full_coverage,
)

MICRO_SCENE = (
    Path(__file__).resolve().parents[1] / "dynamic_grasp" / "micro_scene.xml"
).read_text(encoding="utf-8")


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
        wrist_body=bid(model, "pusher"),
        palm_body=bid(model, "pusher"),
    )


@pytest.fixture
def budget() -> ContactSafetyBudget:
    return ContactSafetyBudget(
        budget_id="micro-conservative-v1",
        robot_profile="micro_pusher",
        peak_normal_force_N=20.0,
        peak_tangential_force_N=12.0,
        normal_impulse_Ns=2.0,
        tangential_impulse_Ns=1.2,
        contact_duration_s=5.0,
        contact_work_J=0.5,
        max_penetration_m=0.002,
        max_wrist_force_N=400.0,
        max_wrist_torque_Nm=60.0,
        max_joint_or_tendon_load=150.0,
        max_non_target_translation_m=0.01,
        max_non_target_rotation_rad=0.15,
        max_non_target_velocity_mps=0.05,
        impulse_window_s=0.1,
    )


# -- coverage -------------------------------------------------------------


def test_every_declared_limit_has_a_sensor(budget: ContactSafetyBudget) -> None:
    assert budget.limit_fields
    assert missing_specs(budget) == ()
    assert set(budget.limit_fields) <= set(SAFETY_FIELD_SPECS)


def test_the_six_limits_v1_never_measured_are_mapped() -> None:
    for field in (
        "max_wrist_force_N",
        "max_wrist_torque_Nm",
        "max_joint_or_tendon_load",
        "max_non_target_translation_m",
        "max_non_target_rotation_rad",
        "max_non_target_velocity_mps",
    ):
        spec = SAFETY_FIELD_SPECS[field]
        assert spec.sensor
        assert spec.unit
        assert spec.failure_reason


def test_each_aggregation_semantics_is_used_by_something() -> None:
    used = {spec.aggregation for spec in SAFETY_FIELD_SPECS.values()}
    assert used == set(Aggregation)


def test_coverage_matrix_is_complete_for_the_evidence_packet(budget) -> None:
    matrix = coverage_matrix(budget)
    assert set(matrix) == set(budget.limit_fields)
    for entry in matrix.values():
        assert entry["sensor"] and entry["unit"] and entry["failure_reason"]


def test_a_model_without_a_wrist_fails_preflight(budget) -> None:
    with pytest.raises(SafetyCoverageError, match="no sensor for"):
        require_full_coverage(
            budget, frozenset({SensorScope.CONTACT, SensorScope.NON_TARGET, SensorScope.ACTUATION})
        )


def test_observer_refuses_a_budget_it_cannot_enforce(model, roles, budget) -> None:
    wristless = dataclasses.replace(roles, wrist_body=None)
    with pytest.raises(SafetyCoverageError, match="max_wrist_force_N"):
        ContactObserver(model, wristless, budget)


def test_observer_reports_the_scopes_it_can_read(model, roles, budget) -> None:
    observer = ContactObserver(model, roles, budget)
    assert SensorScope.CONTACT in observer.available_scopes
    assert SensorScope.WRIST in observer.available_scopes
    assert SensorScope.ACTUATION in observer.available_scopes


def test_an_unmeasured_limit_is_not_treated_as_zero(budget) -> None:
    # Everything measured except the wrist: the verdict must be unsafe, because
    # an unmeasured quantity is unknown, and unknown is not within budget.
    measurements = {field: 0.0 for field in budget.limit_fields if field != "max_wrist_force_N"}
    evaluation = evaluate_budget(budget, measurements)
    assert "max_wrist_force_N" in evaluation.unavailable_fields
    assert not evaluation.safe


def test_a_non_finite_measurement_is_unavailable_not_passing(budget) -> None:
    measurements = {field: 0.0 for field in budget.limit_fields}
    measurements["peak_normal_force_N"] = float("nan")
    evaluation = evaluate_budget(budget, measurements)
    assert "peak_normal_force_N" in evaluation.unavailable_fields
    assert not evaluation.safe


# -- threshold mutation ---------------------------------------------------


@pytest.mark.parametrize("field", sorted(SAFETY_FIELD_SPECS))
def test_mutating_one_threshold_violates_exactly_that_field(field: str, budget) -> None:
    # Every limit measured at exactly its declared value: nothing is violated.
    measurements = {name: float(getattr(budget, name)) for name in budget.limit_fields}
    evaluation = evaluate_budget(budget, measurements)
    assert evaluation.violated_fields == ()
    assert evaluation.min_margin == pytest.approx(0.0)

    # Halve one threshold; that field, and only that field, now fails.
    tightened = dataclasses.replace(budget, **{field: float(getattr(budget, field)) / 2.0})
    mutated = evaluate_budget(tightened, measurements)
    assert mutated.violated_fields == (field,)
    assert mutated.min_margin_field == field
    assert mutated.failure_reasons == (SAFETY_FIELD_SPECS[field].failure_reason,)
    assert not mutated.safe


def test_the_tightest_limit_is_named(budget) -> None:
    measurements = {name: 0.0 for name in budget.limit_fields}
    measurements["max_penetration_m"] = budget.max_penetration_m * 0.9
    evaluation = evaluate_budget(budget, measurements)
    assert evaluation.min_margin_field == "max_penetration_m"
    assert evaluation.min_margin == pytest.approx(0.1)
    assert evaluation.safe


# -- rolling impulse window ----------------------------------------------


def _point(*, normal_force: float, tangential_force: float = 0.0, slip_rate: float = 0.0) -> dict:
    """One synthetic contact-point reading in the shape the observer consumes."""
    return {
        "normal_force": normal_force,
        "tangential_force": tangential_force,
        "slip_rate": slip_rate,
        "penetration": 0.0,
    }


class _Window:
    """Drive the observer's accumulator directly with a synthetic force trace.

    A real contact cannot be scheduled to the millisecond, and the property
    under test is about *when* an impulse arrives relative to a window boundary,
    so the trace is analytic and the accumulator is the thing being measured.
    """

    def __init__(self, observer: ContactObserver, dt: float) -> None:
        self.observer = observer
        self.dt = dt

    def run(self, forces: list[float]) -> float:
        key = (0, 1)
        peak = 0.0
        for force in forces:
            self.observer._step += 1
            self.observer._elapsed_s += self.dt
            now = self.observer._elapsed_s
            readings = {key: [_point(normal_force=force)]}
            self.observer._accumulate(readings, dt=self.dt, now=now)
            episode = self.observer._episodes[key]
            peak = max(peak, episode.windowed()[0])
        return peak


def test_an_impulse_across_a_window_boundary_is_still_detected(model, roles, budget) -> None:
    dt = 0.01
    window = round(budget.impulse_window_s / dt)  # 10 steps

    # The same 6 steps of force, placed so that it straddles what a block
    # accumulator would have treated as a reset. Both placements must report the
    # same windowed impulse, because a sliding window has no privileged origin.
    burst = [100.0] * 6
    aligned = [0.0] * 2 + burst + [0.0] * 12
    straddling = [0.0] * (window - 3) + burst + [0.0] * 12

    first = _Window(ContactObserver(model, roles, budget), dt).run(aligned)
    second = _Window(ContactObserver(model, roles, budget), dt).run(straddling)
    assert first == pytest.approx(second)
    assert first == pytest.approx(sum(burst) * dt)


def test_shifting_the_same_waveform_by_one_step_does_not_change_the_verdict(
    model, roles, budget
) -> None:
    dt = 0.01
    burst = [80.0] * 5
    peaks = [
        _Window(ContactObserver(model, roles, budget), dt).run([0.0] * offset + burst + [0.0] * 10)
        for offset in range(6)
    ]
    assert all(peak == pytest.approx(peaks[0]) for peak in peaks)


def test_the_window_forgets_what_has_aged_out(model, roles, budget) -> None:
    dt = 0.01
    observer = ContactObserver(model, roles, budget)
    driver = _Window(observer, dt)
    driver.run([50.0] * 5)
    # Long enough for everything to age out of a 0.1 s window.
    driver.run([0.0] * 20)
    assert observer._episodes[(0, 1)].windowed()[0] == pytest.approx(0.0)


def test_a_long_gentle_hold_does_not_accumulate_into_a_violation(model, roles, budget) -> None:
    # Impulse is force times time, so a cumulative limit would reject every
    # sustained hold no matter how gentle. The windowed figure must stay bounded.
    dt = 0.01
    observer = ContactObserver(model, roles, budget)
    peak = _Window(observer, dt).run([1.0] * 500)
    assert peak <= budget.normal_impulse_Ns
    assert peak == pytest.approx(1.0 * budget.impulse_window_s, rel=0.15)


# -- episodes and duration ------------------------------------------------


def test_a_recontact_does_not_inherit_the_previous_episode(model, roles, budget) -> None:
    data = mujoco.MjData(model)
    observer = ContactObserver(model, roles, budget)
    key = (0, 1)

    driver = _Window(observer, 0.01)
    driver.run([10.0] * 5)
    first = observer._episodes[key]
    assert first.duration_s == pytest.approx(0.05)

    # The pair stops touching: the episode closes.
    observer._close_stale_episodes(present=set())
    assert key not in observer._episodes

    driver.run([10.0] * 2)
    second = observer._episodes[key]
    assert second.index == first.index + 1
    assert second.duration_s == pytest.approx(0.02)
    assert second.work_J == pytest.approx(0.0)
    del data


def test_several_contact_points_on_one_pair_do_not_double_the_duration(
    model, roles, budget
) -> None:
    observer = ContactObserver(model, roles, budget)
    key = (0, 1)
    points = [_point(normal_force=5.0) for _ in range(3)]
    observer._step = 1
    observer._elapsed_s = 0.01
    observer._accumulate({key: points}, dt=0.01, now=0.01)
    episode = observer._episodes[key]
    assert episode.duration_s == pytest.approx(0.01)
    # The impulse is the sum over points, because they really are three forces.
    assert episode.normal_impulse_Ns == pytest.approx(15.0 * 0.01)


# -- measured against real physics ----------------------------------------


def test_the_observer_measures_all_thirteen_limits_on_a_real_rollout(
    model, roles, budget
) -> None:
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    observer = ContactObserver(model, roles, budget)
    observer.reset(data)

    dt = float(model.opt.timestep)
    data.ctrl[0] = 0.2
    for step in range(400):
        mujoco.mj_step(model, data)
        observer.observe(data, time_index=step, dt=dt, simulator_step=step)

    evaluation = observer.evaluation
    assert evaluation.unavailable_fields == ()
    assert set(evaluation.measured_fields) == set(budget.limit_fields)
    assert evaluation.min_margin_field
    assert np.isfinite(evaluation.min_margin)


def test_a_peek_does_not_charge_the_interval(model, roles, budget) -> None:
    # The static-seeded controller reads contacts before it decides. Charging
    # that read would double every impulse and duration (C03.10).
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    observer = ContactObserver(model, roles, budget)
    observer.reset(data)

    dt = float(model.opt.timestep)
    data.ctrl[0] = 0.2
    for step in range(200):
        mujoco.mj_step(model, data)
        observer.observe(data, time_index=step, dt=dt, accumulate=False)
    assert observer.elapsed_s == pytest.approx(0.0)

    peeking = ContactObserver(model, roles, budget)
    charging = ContactObserver(model, roles, budget)
    replay = mujoco.MjData(model)
    mujoco.mj_forward(model, replay)
    peeking.reset(replay)
    charging.reset(replay)
    replay.ctrl[0] = 0.2
    for step in range(200):
        mujoco.mj_step(model, replay)
        peeking.observe(replay, time_index=step, dt=dt, accumulate=False)
        peeking.observe(replay, time_index=step, dt=dt)
        charging.observe(replay, time_index=step, dt=dt)
    assert peeking.elapsed_s == pytest.approx(charging.elapsed_s)
    assert peeking.measurements.get("contact_duration_s", 0.0) == pytest.approx(
        charging.measurements.get("contact_duration_s", 0.0)
    )


def test_the_evaluation_is_serialisable_for_evidence(model, roles, budget) -> None:
    observer = ContactObserver(model, roles, budget)
    payload = observer.evaluation.as_dict()
    assert payload["budget_id"] == budget.budget_id
    assert len(payload["budget_hash"]) == 64
    assert "unavailable_fields" in payload
