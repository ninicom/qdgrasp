"""CEM search tests (P3.4-09)."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from qdgrasp.dataset.dynamic_contracts import DynamicSearchOutcome
from qdgrasp.dynamic.cem import CemConfig, ParameterSpace, search_cem
from qdgrasp.dynamic.objective import ObjectiveWeights, ReasonLedger, score_outcome
from qdgrasp.dynamic.primitives import Primitive, PrimitiveKind

from .conftest import make_certificate, make_trajectory


def template() -> tuple[Primitive, ...]:
    return (
        Primitive(
            kind=PrimitiveKind.PUSH,
            direction=np.array([1.0, 0.0, 0.0]),
            speed=0.1,
            max_duration_s=0.2,
        ),
        Primitive(
            kind=PrimitiveKind.SQUEEZE,
            direction=np.array([0.0, 0.0, 1.0]),
            speed=0.0,
            max_duration_s=0.2,
            grip=0.5,
        ),
    )


def outcome(passed=False, reason="insufficient_lift", **terms) -> DynamicSearchOutcome:
    return DynamicSearchOutcome(
        trajectory_ref="t",
        passed=passed,
        failure_stage="lift" if not passed else "none",
        failure_reason=reason if not passed else "none",
        objective_terms=terms,
        peak_safety_metrics={"min_budget_margin": 0.5, "peak_normal_force_N": 1.0},
        cumulative_safety_metrics={"total_slip_m": 0.0},
        cpu_replay_evidence=make_certificate() if passed else {},
    )


# -- objective ------------------------------------------------------------


def test_a_hard_reject_cannot_be_bought_back_by_a_high_score():
    weights = ObjectiveWeights()
    rich = outcome(reason="damaging_contact", lift_m=10.0, enclosure_links=10.0)
    assert score_outcome(rich, weights) == float("-inf")
    assert score_outcome(outcome(reason="forbidden_contact"), weights) == float("-inf")


def test_score_rewards_lift_and_penalises_disturbance():
    weights = ObjectiveWeights()
    base = score_outcome(outcome(lift_m=0.0), weights)
    lifted = score_outcome(outcome(lift_m=0.05), weights)
    disturbed = score_outcome(outcome(lift_m=0.0, non_target_disturbance_m=0.05), weights)
    assert lifted > base > disturbed


def test_ledger_keeps_a_denominator_at_every_stage():
    ledger = ReasonLedger()
    ledger.record(outcome(passed=True))
    ledger.record(outcome(reason="damaging_contact"))
    ledger.record(outcome(reason="insufficient_lift"))
    report = ledger.to_dict()

    assert report["stages"]["sampled"] == 3
    # The damaging candidate dies before safe_contact_feasible; the other two
    # reach it, so the stage count must be 2 rather than a bare failure tally.
    assert report["stages"]["safe_contact_feasible"] == 2
    assert report["stages"]["cpu_replay_confirmed"] == 1
    assert report["failures"] == {"damaging_contact": 1, "insufficient_lift": 1}
    assert report["yield"] == pytest.approx(1 / 3)


# -- parameter space ------------------------------------------------------


def test_space_rejects_inverted_bounds():
    with pytest.raises(ValueError, match="speed_bounds"):
        ParameterSpace(speed_bounds=(1.0, 0.0))


def test_applying_a_sample_keeps_the_primitive_shape():
    space = ParameterSpace()
    rewritten = space.apply(template(), np.array([0.2, 0.8, 0.5]))
    assert len(rewritten) == len(template())
    assert all(p.max_duration_s == pytest.approx(0.5) for p in rewritten)
    # A primitive that is stationary by design stays stationary.
    assert rewritten[1].speed == 0.0


# -- search ---------------------------------------------------------------


def test_config_rejects_a_degenerate_budget():
    for kwargs, match in (
        ({"population": 1}, "population"),
        ({"elite_fraction": 0.0}, "elite_fraction"),
        ({"iterations": 0}, "iterations"),
        ({"min_std": 0.0}, "min_std"),
    ):
        with pytest.raises(ValueError, match=match):
            CemConfig(**kwargs)


def test_search_burns_exactly_its_declared_budget():
    # Fixed budget, never "until something works": a search that stops on first
    # success reports a yield that means nothing.
    calls = []

    def rollout(primitives):
        calls.append(primitives)
        return make_trajectory(), outcome(passed=True, lift_m=0.05)

    config = CemConfig(population=6, iterations=3, seed=1)
    result = search_cem(template=template(), rollout=rollout, config=config)

    assert len(calls) == 18
    assert result.evaluated == 18
    assert result.iterations_run == 3
    assert result.reason_ledger["stages"]["sampled"] == 18


def test_search_is_deterministic_for_a_seed():
    def rollout(primitives):
        return make_trajectory(), outcome(lift_m=float(primitives[0].speed))

    kwargs = {"template": template(), "rollout": rollout, "config": CemConfig(seed=7)}
    first, second = search_cem(**kwargs), search_cem(**kwargs)
    assert np.allclose(first.best_sample, second.best_sample)
    assert first.best_score == second.best_score


def test_a_different_seed_explores_differently():
    def rollout(primitives):
        return make_trajectory(), outcome(lift_m=float(primitives[0].speed))

    a = search_cem(template=template(), rollout=rollout, config=CemConfig(seed=1))
    b = search_cem(template=template(), rollout=rollout, config=CemConfig(seed=2))
    assert not np.allclose(a.best_sample, b.best_sample)


def test_the_distribution_moves_toward_better_samples():
    # Reward high speed; the fitted mean speed should rise across iterations.
    def rollout(primitives):
        return make_trajectory(), outcome(lift_m=float(primitives[0].speed))

    result = search_cem(
        template=template(), rollout=rollout,
        config=CemConfig(population=24, iterations=4, seed=3),
    )
    speeds = [m[0] for m in result.mean_history]
    assert speeds[-1] > speeds[0]


def test_all_candidates_hard_rejected_still_returns_a_usable_report():
    # Every score is -inf. The search must not crash, and must not claim a best.
    def rollout(primitives):
        return make_trajectory(), outcome(reason="damaging_contact")

    result = search_cem(
        template=template(), rollout=rollout, config=CemConfig(population=4, iterations=2)
    )
    assert result.best_score == float("-inf")
    assert result.best_outcome is None
    assert result.reason_ledger["yield"] == 0.0
    assert result.reason_ledger["failures"]["damaging_contact"] == 8


def test_samples_stay_inside_the_declared_bounds():
    seen = []

    def rollout(primitives):
        seen.append(primitives[0].speed)
        return make_trajectory(), outcome()

    space = ParameterSpace(speed_bounds=(0.05, 0.10))
    search_cem(
        template=template(), rollout=rollout, space=space,
        config=CemConfig(population=20, iterations=3, seed=5),
    )
    assert min(seen) >= 0.05 - 1e-9
    assert max(seen) <= 0.10 + 1e-9


def test_weights_are_frozen_so_a_manifest_can_hash_them():
    weights = ObjectiveWeights()
    with pytest.raises(dataclasses.FrozenInstanceError):
        weights.safety_margin = 0.0  # type: ignore[misc]
