"""S7 — objective, CEM and refinement fail closed (C04, B-16).

The failure pinned here is a search that keeps going on evidence it should have
refused: a NaN term scored as though it were a number, a missing term defaulted
to zero and therefore to a perfect score, an iteration that refits its
distribution from a cohort where every candidate was rejected.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from qdgrasp.dataset.dynamic_contracts import (
    CpuReplayCertificate,
    DynamicGraspTrajectory,
    DynamicSearchOutcome,
    TrajectoryStage,
    TrajectoryTimebase,
)
from qdgrasp.dynamic.cem import CemConfig, ParameterSpace, search_cem
from qdgrasp.dynamic.objective import (
    BARRIER_REASONS,
    REQUIRED_OBJECTIVE_TERMS,
    ObjectiveWeights,
    evaluate_objective,
    missing_objective_terms,
    require_objective_terms,
)
from qdgrasp.dynamic.primitives import (
    CAPABILITY_MATRIX,
    DEFERRED_STRATEGIES,
    Primitive,
    PrimitiveKind,
    TransitionCondition,
    capability_report,
    primitive_sequence_hash,
)
from qdgrasp.dynamic.refine import RefineConfig, refine_local

SAMPLE_PERIOD_S = 0.01


def certificate() -> CpuReplayCertificate:
    return CpuReplayCertificate(
        backend_id="mujoco_cpu",
        capsule_sha256="a" * 64,
        command_sha256="b" * 64,
        model_sha256="c" * 64,
        timestep_s=0.002,
        terminal_certified=True,
        safety_certified=True,
        outcome_class="pass",
    )


def trajectory(steps: int = 4) -> DynamicGraspTrajectory:
    palm = np.zeros((steps, 7))
    palm[:, 3] = 1.0
    pose = np.zeros((steps, 1, 7))
    pose[:, :, 3] = 1.0
    return DynamicGraspTrajectory(
        time=np.arange(steps, dtype=float) * SAMPLE_PERIOD_S,
        palm_pose=palm,
        joint_state=np.zeros((steps, 16)),
        actuator_command=np.zeros((steps, 16)),
        object_pose=pose,
        object_velocity=np.zeros((steps, 1, 6)),
        stage=tuple([TrajectoryStage.APPROACH] * steps),
        timebase=TrajectoryTimebase(simulator_dt=SAMPLE_PERIOD_S, sample_every=1),
    )


def complete_outcome(passed: bool = True, **over) -> DynamicSearchOutcome:
    defaults = {
        "trajectory_ref": "t:0",
        "passed": passed,
        "failure_stage": "none" if passed else "lift",
        "failure_reason": "none" if passed else "insufficient_lift",
        "objective_terms": {
            "lift_m": 0.05,
            "enclosure_links": 2.0,
            "control_energy_J": 0.08,
            "elapsed_time_s": 1.2,
        },
        "peak_safety_metrics": {
            "min_budget_margin": 0.5,
            "peak_normal_force_N": 1.0,
            "max_penetration_m": 1e-4,
        },
        "cumulative_safety_metrics": {"total_slip_m": 0.0},
        "cpu_replay_evidence": certificate() if passed else None,
    }
    defaults.update(over)
    return DynamicSearchOutcome(**defaults)


def template() -> tuple[Primitive, ...]:
    return (
        Primitive(
            kind=PrimitiveKind.PUSH,
            direction=np.array([1.0, 0.0, 0.0]),
            speed=0.1,
            max_duration_s=0.2,
        ),
    )


# -- objective ------------------------------------------------------------


def test_weights_are_finite_non_negative_and_hashed() -> None:
    weights = ObjectiveWeights()
    assert len(weights.weights_hash) == 64
    assert weights.weights_hash != dataclasses.replace(
        weights, safety_margin=0.31
    ).weights_hash
    with pytest.raises(ValueError, match="non-negative"):
        ObjectiveWeights(hand_load_cost=-1.0)
    with pytest.raises(ValueError, match="finite"):
        ObjectiveWeights(safety_margin=float("nan"))


def test_control_energy_and_elapsed_time_are_weighted_separately() -> None:
    weights = ObjectiveWeights()
    assert weights.control_energy_cost != 0.0
    assert weights.elapsed_time_cost != 0.0
    assert not hasattr(weights, "control_energy_and_time_cost")

    slow_and_gentle = complete_outcome(
        objective_terms={
            "lift_m": 0.05, "enclosure_links": 2.0,
            "control_energy_J": 0.01, "elapsed_time_s": 5.0,
        }
    )
    fast_and_violent = complete_outcome(
        objective_terms={
            "lift_m": 0.05, "enclosure_links": 2.0,
            "control_energy_J": 5.0, "elapsed_time_s": 0.5,
        }
    )
    # v1 charged both through one step count, so these scored the same.
    assert evaluate_objective(slow_and_gentle, weights).score != pytest.approx(
        evaluate_objective(fast_and_violent, weights).score
    )


@pytest.mark.parametrize("term", REQUIRED_OBJECTIVE_TERMS)
def test_a_missing_required_term_is_rejected_in_the_release_path(term: str) -> None:
    terms = dict(complete_outcome().objective_terms)
    terms.pop(term)
    outcome = complete_outcome(objective_terms=terms)
    assert term in missing_objective_terms(outcome)

    verdict = evaluate_objective(outcome, ObjectiveWeights(), strict=True)
    assert verdict.rejected
    assert verdict.reason == "missing_objective_term"
    with pytest.raises(ValueError, match="missing required terms"):
        require_objective_terms(outcome)


def test_a_complete_outcome_passes_the_strict_path() -> None:
    require_objective_terms(complete_outcome())
    verdict = evaluate_objective(complete_outcome(), ObjectiveWeights(), strict=True)
    assert not verdict.rejected
    assert np.isfinite(verdict.score)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_a_non_finite_term_hard_rejects_the_candidate(bad: float) -> None:
    # The contract refuses to build such an outcome at all, which is the
    # earliest place to stop it; the objective refuses it too, because a search
    # may be handed metrics from a backend the contract never saw.
    with pytest.raises(ValueError, match="objective_terms"):
        complete_outcome(objective_terms={"lift_m": bad})

    verdict = evaluate_objective(
        _unchecked_outcome(objective_terms={"lift_m": bad}), ObjectiveWeights()
    )
    assert verdict.rejected
    assert verdict.reason == "non_finite_objective"


def _replace_raw(outcome: DynamicSearchOutcome, **over) -> DynamicSearchOutcome:
    """Build an outcome past its own validation.

    The contract already refuses these payloads, which is the right place to
    stop them. The objective has to refuse them too, because a search can be
    handed metrics from a backend whose summary the contract never saw.
    """
    clone = object.__new__(DynamicSearchOutcome)
    for field in dataclasses.fields(outcome):
        object.__setattr__(clone, field.name, over.get(field.name, getattr(outcome, field.name)))
    return clone


def _unchecked_outcome(**over) -> DynamicSearchOutcome:
    return _replace_raw(complete_outcome(), **over)


@pytest.mark.parametrize("reason", sorted(BARRIER_REASONS))
def test_a_barrier_reason_always_scores_negative_infinity(reason: str) -> None:
    outcome = complete_outcome(passed=False, failure_stage="contact", failure_reason=reason)
    verdict = evaluate_objective(outcome, ObjectiveWeights())
    assert verdict.score == float("-inf")
    assert verdict.rejected


def test_no_terminal_quality_buys_back_a_damaging_contact() -> None:
    spectacular = complete_outcome(
        passed=False,
        failure_stage="contact",
        failure_reason="damaging_contact",
        objective_terms={
            "lift_m": 10.0, "enclosure_links": 20.0,
            "control_energy_J": 0.0, "elapsed_time_s": 0.0,
        },
    )
    assert evaluate_objective(spectacular, ObjectiveWeights()).score == float("-inf")


def test_a_self_contradicting_outcome_is_rejected() -> None:
    contradictory = _replace_raw(
        complete_outcome(), passed=True, failure_reason="insufficient_lift"
    )
    verdict = evaluate_objective(contradictory, ObjectiveWeights())
    assert verdict.rejected
    assert verdict.reason == "unexpected_control_outcome"


# -- CEM ------------------------------------------------------------------


def rollout_passing(primitives):
    return trajectory(), complete_outcome()


def rollout_failing(reason: str = "insufficient_lift"):
    def inner(primitives):
        return trajectory(), complete_outcome(
            passed=False, failure_stage="lift", failure_reason=reason
        )

    return inner


def test_the_budget_is_bounded_and_hashed() -> None:
    config = CemConfig(population=4, iterations=2)
    assert len(config.config_hash) == 64
    assert config.config_hash != CemConfig(population=4, iterations=3).config_hash
    with pytest.raises(ValueError, match="max_worlds"):
        CemConfig(population=64, max_worlds=16)


def test_a_search_burns_exactly_its_declared_budget() -> None:
    result = search_cem(
        template=template(), rollout=rollout_passing, config=CemConfig(population=5, iterations=3)
    )
    assert result.evaluated == 15
    assert result.iterations_run == 3
    assert len(result.candidates) == 15


def test_all_rejected_stops_with_no_feasible_elite() -> None:
    result = search_cem(
        template=template(),
        rollout=rollout_failing("damaging_contact"),
        config=CemConfig(population=4, iterations=5),
    )
    assert result.stop_reason == "no_feasible_elite"
    assert result.iterations_run == 1
    assert result.best_outcome is None
    # Nothing was refit from a cohort of rejects.
    assert result.mean_history == ()


def test_a_search_that_never_succeeds_reports_budget_exhausted() -> None:
    result = search_cem(
        template=template(),
        rollout=rollout_failing("insufficient_lift"),
        config=CemConfig(population=4, iterations=2),
    )
    assert result.stop_reason == "budget_exhausted"
    assert result.iterations_run == 2
    assert not result.succeeded


def test_a_successful_search_carries_the_outcome_it_succeeded_with() -> None:
    result = search_cem(
        template=template(), rollout=rollout_passing, config=CemConfig(population=4, iterations=2)
    )
    assert result.succeeded
    assert result.best_outcome is not None
    assert result.stop_reason == "none"


def test_a_success_verdict_cannot_be_built_without_an_outcome() -> None:
    from qdgrasp.dynamic.cem import CemResult

    fields = {
        "best_sample": np.zeros(3),
        "best_score": 1.0,
        "best_outcome": None,
        "best_trajectory": None,
        "iterations_run": 1,
        "evaluated": 1,
        "reason_ledger": {},
        "mean_history": (),
    }
    with pytest.raises(ValueError, match="must carry the outcome"):
        CemResult(**fields, stop_reason="none")

    # The same result is fine once it says why it stopped.
    assert CemResult(**fields, stop_reason="budget_exhausted").best_outcome is None


def test_an_unknown_stop_reason_is_refused() -> None:
    from qdgrasp.dynamic.cem import CemResult

    with pytest.raises(ValueError, match="unknown stop reason"):
        CemResult(
            best_sample=np.zeros(3),
            best_score=1.0,
            best_outcome=None,
            best_trajectory=None,
            iterations_run=1,
            evaluated=1,
            reason_ledger={},
            mean_history=(),
            stop_reason="gave_up",
        )


def test_the_same_seed_gives_the_same_samples() -> None:
    seen: list[list[float]] = []

    def record(primitives):
        seen.append([primitives[0].speed, primitives[0].grip, primitives[0].max_duration_s])
        return trajectory(), complete_outcome()

    first_len = 0
    for _ in range(2):
        seen.clear()
        search_cem(
            template=template(),
            rollout=record,
            config=CemConfig(population=6, iterations=2, seed=11),
        )
        if first_len == 0:
            first_len = len(seen)
            reference = [row[:] for row in seen]
        else:
            assert seen == reference


def test_a_different_seed_gives_different_samples() -> None:
    def collect(seed: int) -> list[float]:
        seen: list[float] = []

        def record(primitives):
            seen.append(primitives[0].speed)
            return trajectory(), complete_outcome()

        search_cem(
            template=template(),
            rollout=record,
            config=CemConfig(population=6, iterations=1, seed=seed),
        )
        return seen

    assert collect(1) != collect(2)


def test_candidate_world_and_iteration_mapping_is_stable() -> None:
    result = search_cem(
        template=template(), rollout=rollout_passing, config=CemConfig(population=4, iterations=2)
    )
    assert [record.candidate_index for record in result.candidates] == list(range(8))
    assert [record.world_index for record in result.candidates] == [0, 1, 2, 3, 0, 1, 2, 3]
    assert [record.iteration for record in result.candidates] == [0] * 4 + [1] * 4


def test_samples_stay_inside_the_declared_bounds() -> None:
    seen: list[float] = []

    def record(primitives):
        seen.append(primitives[0].speed)
        return trajectory(), complete_outcome()

    space = ParameterSpace(speed_bounds=(0.05, 0.10))
    search_cem(
        template=template(),
        rollout=record,
        space=space,
        config=CemConfig(population=20, iterations=3, seed=5),
    )
    assert min(seen) >= 0.05 - 1e-9
    assert max(seen) <= 0.10 + 1e-9


# -- refinement -----------------------------------------------------------


def test_refinement_needs_a_cpu_confirmed_positive() -> None:
    uncertified = _replace_raw(complete_outcome(), cpu_replay_evidence=None)
    with pytest.raises(ValueError, match="CPU-confirmed"):
        refine_local(
            template=template(),
            seed_sample=np.array([0.1, 0.5, 0.2]),
            seed_outcome=uncertified,
            seed_trajectory=trajectory(),
            rollout=rollout_passing,
        )


def test_refinement_records_the_objective_and_bounds_it_ran_under() -> None:
    result = refine_local(
        template=template(),
        seed_sample=np.array([0.1, 0.5, 0.2]),
        seed_outcome=complete_outcome(),
        seed_trajectory=trajectory(),
        rollout=rollout_passing,
        config=RefineConfig(iterations=1),
    )
    assert len(result.weights_hash) == 64
    assert len(result.space_hash) == 64


def test_refinement_never_accepts_a_failing_neighbour() -> None:
    result = refine_local(
        template=template(),
        seed_sample=np.array([0.1, 0.5, 0.2]),
        seed_outcome=complete_outcome(),
        seed_trajectory=trajectory(),
        rollout=rollout_failing(),
        config=RefineConfig(iterations=2),
    )
    assert result.outcome.passed
    assert result.rejected_regressions > 0
    assert not result.improved


# -- capability matrix ----------------------------------------------------


def test_every_primitive_kind_declares_what_it_actually_does() -> None:
    assert set(CAPABILITY_MATRIX) == set(PrimitiveKind)
    for kind, capability in CAPABILITY_MATRIX.items():
        assert capability.kind is kind
        assert capability.command_semantics
        assert capability.transition_semantics


@pytest.mark.parametrize(
    "kind",
    [
        PrimitiveKind.PUSH,
        PrimitiveKind.SLIDE,
        PrimitiveKind.PIVOT_ON_SUPPORT,
        PrimitiveKind.HOOK,
        PrimitiveKind.CAGE,
        PrimitiveKind.SQUEEZE,
        PrimitiveKind.SUPPORT_RELEASE,
        PrimitiveKind.LIFT,
        PrimitiveKind.PERTURB,
    ],
)
def test_the_required_primitives_emit_a_real_command(kind: PrimitiveKind) -> None:
    # The plan names these nine explicitly; each has to produce a command a
    # controller can apply, not just an enum member (C04.1).
    primitive = Primitive(
        kind=kind,
        direction=np.array([1.0, 0.0, 0.0]),
        speed=0.1,
        max_duration_s=0.2,
        grip=0.5,
    )
    command = primitive.wrist_velocity()
    assert command.shape == (3,)
    assert np.isfinite(command).all()
    assert primitive.stage is CAPABILITY_MATRIX[kind].stage


def test_a_clock_only_transition_is_declared_as_a_duration() -> None:
    for capability in CAPABILITY_MATRIX.values():
        if "clock is only a ceiling" not in capability.transition_semantics:
            assert "duration *is* the condition" in capability.transition_semantics


def test_mppi_is_recorded_as_deferred_rather_than_silently_absent() -> None:
    assert "mppi" in DEFERRED_STRATEGIES
    report = capability_report()
    assert "mppi" in report["deferred_not_claimed"]
    assert "mppi" not in report["implemented"]


def test_a_sequence_hash_separates_sequences_that_command_differently() -> None:
    base = template()
    assert primitive_sequence_hash(base) == primitive_sequence_hash(template())

    faster = (dataclasses.replace(base[0], speed=0.2),)
    other_way = (dataclasses.replace(base[0], direction=np.array([0.0, 1.0, 0.0])),)
    other_condition = (
        dataclasses.replace(base[0], until=TransitionCondition.TARGET_CONTACT_MADE),
    )
    hashes = {
        primitive_sequence_hash(base),
        primitive_sequence_hash(faster),
        primitive_sequence_hash(other_way),
        primitive_sequence_hash(other_condition),
    }
    assert len(hashes) == 4
