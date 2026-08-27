"""Declarative search objective and reason accounting (P3.4-08 support, §8).

The objective is data, not code buried in a strategy: every term is named,
weighted and reported separately so a result can be audited term by term.

Hard rejection is not a large negative weight. A forbidden or damaging contact
removes a candidate outright, because a barrier expressed as a penalty can
always be outbid by a high terminal score.
"""

from __future__ import annotations

import dataclasses

from qdgrasp.dataset.dynamic_contracts import DynamicSearchOutcome


@dataclasses.dataclass(frozen=True)
class ObjectiveWeights:
    """Pinned before a search so the manifest can hash them."""

    terminal_grasp_quality: float = 1.0
    target_accessibility_progress: float = 0.5
    enclosure_progress: float = 0.5
    safety_margin: float = 0.3
    hand_load_cost: float = 0.2
    non_target_disturbance_cost: float = 1.0
    slip_and_penetration_cost: float = 0.4
    control_energy_and_time_cost: float = 0.05


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


def score_outcome(outcome: DynamicSearchOutcome, weights: ObjectiveWeights) -> float:
    """Score a rollout, or reject it outright.

    A hard rejection returns negative infinity rather than a large penalty: no
    terminal quality may buy back a forbidden or damaging contact.
    """
    if outcome.failure_reason in ("forbidden_contact", "damaging_contact"):
        return float("-inf")

    terms = outcome.objective_terms
    peak = outcome.peak_safety_metrics
    cumulative = outcome.cumulative_safety_metrics

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
    score -= weights.control_energy_and_time_cost * float(terms.get("steps", 0.0))
    return float(score)


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
