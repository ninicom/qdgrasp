"""Declarative search objective and reason accounting (P3.4-08 support, §8).

The objective is data, not code buried in a strategy: every term is named,
weighted and reported separately so a result can be audited term by term.

Hard rejection is not a large negative weight. A forbidden or damaging contact
removes a candidate outright, because a barrier expressed as a penalty can
always be outbid by a high terminal score.
"""

from __future__ import annotations

import dataclasses
import math

from qdgrasp.dataset.dynamic_contracts import DynamicSearchOutcome, canonical_hash


@dataclasses.dataclass(frozen=True)
class ObjectiveWeights:
    """Pinned before a search so the manifest can hash them.

    Every weight is finite and non-negative: the sign of a term is fixed by the
    scoring function (progress adds, cost subtracts), so a negative weight would
    silently invert what the objective rewards.

    Control energy and elapsed time are weighted separately. v1 charged both
    through one step count, which is neither: a slow gentle rollout and a fast
    violent one had the same cost (C04.2).
    """

    terminal_grasp_quality: float = 1.0
    target_accessibility_progress: float = 0.5
    enclosure_progress: float = 0.5
    safety_margin: float = 0.3
    hand_load_cost: float = 0.2
    non_target_disturbance_cost: float = 1.0
    slip_and_penetration_cost: float = 0.4
    control_energy_cost: float = 0.05
    elapsed_time_cost: float = 0.05

    def __post_init__(self) -> None:
        for field in dataclasses.fields(self):
            value = float(getattr(self, field.name))
            if not math.isfinite(value):
                raise ValueError(f"{field.name} must be finite, got {value!r}")
            if value < 0.0:
                raise ValueError(
                    f"{field.name} must be non-negative: the sign of each term is fixed by "
                    f"the scoring function, so a negative weight inverts it. Got {value!r}"
                )

    @property
    def weights_hash(self) -> str:
        return canonical_hash(dataclasses.asdict(self))


#: Terms a release-path candidate has to carry. Defaulting a missing one to zero
#: makes an unmeasured quantity look like a perfect score (C04.2).
REQUIRED_OBJECTIVE_TERMS: tuple[str, ...] = (
    "lift_m",
    "enclosure_links",
    "control_energy_J",
    "elapsed_time_s",
)
REQUIRED_PEAK_METRICS: tuple[str, ...] = (
    "min_budget_margin",
    "peak_normal_force_N",
    "max_penetration_m",
)
REQUIRED_CUMULATIVE_METRICS: tuple[str, ...] = ("total_slip_m",)


class ObjectiveError(ValueError):
    """A candidate's objective cannot be evaluated as declared."""


@dataclasses.dataclass(frozen=True)
class ObjectiveScore:
    """A score, or the reason there isn't one."""

    score: float
    rejected: bool
    reason: str

    @property
    def is_barrier(self) -> bool:
        return self.score == float("-inf")


def missing_objective_terms(outcome: DynamicSearchOutcome) -> tuple[str, ...]:
    """Required terms this outcome does not carry."""
    missing = [t for t in REQUIRED_OBJECTIVE_TERMS if t not in outcome.objective_terms]
    missing += [t for t in REQUIRED_PEAK_METRICS if t not in outcome.peak_safety_metrics]
    missing += [
        t for t in REQUIRED_CUMULATIVE_METRICS if t not in outcome.cumulative_safety_metrics
    ]
    return tuple(sorted(missing))


def require_objective_terms(outcome: DynamicSearchOutcome) -> None:
    """Fail closed when a release candidate is missing a declared term."""
    missing = missing_objective_terms(outcome)
    if missing:
        raise ObjectiveError(
            f"objective is missing required terms {list(missing)}; defaulting them to zero "
            "would score an unmeasured quantity as a perfect one"
        )


#: Stages of the reason ledger, in the order a candidate passes through them.
REASON_STAGES = (
    "sampled",
    "numerically_stable",
    "safe_contact_feasible",
    "terminal_enclosure",
    "support_released",
    "lift_passed",
    "perturbation_passed",
    "cpu_replay_confirmed",
)

#: Which stage each failure reason belongs to, so the ledger has a denominator
#: at every step rather than only a final count.
_REASON_STAGE: dict[str, str] = {
    "empty_trajectory": "sampled",
    "world_rejected": "numerically_stable",
    "target_teleported": "numerically_stable",
    "forbidden_contact": "safe_contact_feasible",
    "damaging_contact": "safe_contact_feasible",
    "non_target_disturbance": "safe_contact_feasible",
    "no_environmental_assistance": "terminal_enclosure",
    "insufficient_enclosure": "terminal_enclosure",
    "support_not_released": "support_released",
    "insufficient_lift": "lift_passed",
    "perturbation_failed": "perturbation_passed",
    "backend_divergence": "cpu_replay_confirmed",
}


#: Reasons that remove a candidate outright, whatever else it scored.
BARRIER_REASONS: frozenset[str] = frozenset(
    {"forbidden_contact", "damaging_contact", "safety_budget_violation", "scene_damage"}
)


def evaluate_objective(
    outcome: DynamicSearchOutcome,
    weights: ObjectiveWeights,
    *,
    strict: bool = False,
) -> ObjectiveScore:
    """Score a rollout, or reject it outright and say why.

    A hard rejection returns negative infinity rather than a large penalty: no
    terminal quality may buy back a forbidden or damaging contact (C04.4).

    ``strict`` is the release path: a candidate missing a declared term is
    rejected rather than scored as though the term were zero (C04.2).
    """
    if outcome.failure_reason in BARRIER_REASONS:
        return ObjectiveScore(float("-inf"), rejected=True, reason=outcome.failure_reason)

    # A passed outcome with a failure reason, or a failed one without, means the
    # producer disagreed with itself; scoring it would pick a side at random.
    if outcome.passed != (outcome.failure_reason == "none"):
        return ObjectiveScore(
            float("-inf"), rejected=True, reason="unexpected_control_outcome"
        )

    if strict:
        missing = missing_objective_terms(outcome)
        if missing:
            return ObjectiveScore(
                float("-inf"), rejected=True, reason="missing_objective_term"
            )

    terms = outcome.objective_terms
    peak = outcome.peak_safety_metrics
    cumulative = outcome.cumulative_safety_metrics
    for source in (terms, peak, cumulative):
        for name, value in source.items():
            if not math.isfinite(float(value)):
                return ObjectiveScore(
                    float("-inf"), rejected=True, reason="non_finite_objective"
                )
            del name

    score = 0.0
    score += weights.terminal_grasp_quality * (1.0 if outcome.passed else 0.0)
    score += weights.target_accessibility_progress * float(terms.get("lift_m", 0.0))
    score += weights.enclosure_progress * float(terms.get("enclosure_links", 0.0))
    score += weights.safety_margin * float(peak.get("min_budget_margin", 0.0))
    score -= weights.hand_load_cost * float(peak.get("peak_normal_force_N", 0.0))
    score -= weights.non_target_disturbance_cost * float(
        terms.get("non_target_disturbance_m", 0.0)
    )
    score -= weights.slip_and_penetration_cost * (
        float(cumulative.get("total_slip_m", 0.0)) + float(peak.get("max_penetration_m", 0.0))
    )
    score -= weights.control_energy_cost * float(terms.get("control_energy_J", 0.0))
    score -= weights.elapsed_time_cost * float(terms.get("elapsed_time_s", 0.0))
    if not math.isfinite(score):
        return ObjectiveScore(float("-inf"), rejected=True, reason="non_finite_objective")
    return ObjectiveScore(float(score), rejected=False, reason="none")


def score_outcome(
    outcome: DynamicSearchOutcome, weights: ObjectiveWeights, *, strict: bool = False
) -> float:
    """The score alone, for ranking. See :func:`evaluate_objective` for the reason."""
    return evaluate_objective(outcome, weights, strict=strict).score


class ReasonLedger:
    """Counts how many candidates survive each stage, with a denominator."""

    def __init__(self) -> None:
        self._counts = dict.fromkeys(REASON_STAGES, 0)
        self._failures: dict[str, int] = {}
        self._sampled = 0

    def record(self, outcome: DynamicSearchOutcome) -> None:
        self._sampled += 1
        self._counts["sampled"] += 1
        if outcome.passed:
            for stage in REASON_STAGES[1:]:
                self._counts[stage] += 1
            return
        reason = outcome.failure_reason
        self._failures[reason] = self._failures.get(reason, 0) + 1
        stop = _REASON_STAGE.get(reason)
        if stop is None:
            return
        for stage in REASON_STAGES[1:]:
            if stage == stop:
                break
            self._counts[stage] += 1

    @property
    def sampled(self) -> int:
        return self._sampled

    def to_dict(self) -> dict[str, object]:
        return {
            "stages": dict(self._counts),
            "failures": dict(sorted(self._failures.items())),
            "yield": (
                self._counts["cpu_replay_confirmed"] / self._sampled
                if self._sampled
                else 0.0
            ),
        }
