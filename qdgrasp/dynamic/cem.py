"""Batched cross-entropy search over primitive parameters (P3.4-09).

CEM samples primitive parameters, rolls the batch out, keeps the elite fraction
and refits the sampling distribution -- for a fixed number of iterations, never
until something works. An unbounded search that stops on first success reports a
yield that means nothing.

Rejected candidates are counted, not discarded: the reason ledger keeps a
denominator at every stage so a yield figure can be read honestly.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Sequence

import numpy as np

from qdgrasp.dataset.dynamic_contracts import (
    DynamicGraspTrajectory,
    DynamicSearchOutcome,
    canonical_hash,
)
from qdgrasp.dynamic.objective import ObjectiveWeights, ReasonLedger, evaluate_objective
from qdgrasp.dynamic.primitives import Primitive


@dataclasses.dataclass(frozen=True)
class CemConfig:
    """Search budget, pinned before the run."""

    population: int = 16
    elite_fraction: float = 0.25
    iterations: int = 3
    #: Standard deviation floor, so the distribution cannot collapse onto one
    #: sample and report a confident result from a single lucky rollout.
    min_std: float = 1e-3
    seed: int = 0
    #: Hard ceiling on simultaneous worlds. A search that sizes its batch from
    #: whatever fits is a search that behaves differently on every machine.
    max_worlds: int = 1024

    def __post_init__(self) -> None:
        if self.population < 2:
            raise ValueError(f"population must be >= 2, got {self.population}")
        if self.max_worlds < 1:
            raise ValueError(f"max_worlds must be >= 1, got {self.max_worlds}")
        if self.population > self.max_worlds:
            raise ValueError(
                f"population {self.population} exceeds max_worlds {self.max_worlds}"
            )
        if not 0.0 < self.elite_fraction <= 1.0:
            raise ValueError(
                f"elite_fraction must lie in (0, 1], got {self.elite_fraction}"
            )
        if self.iterations < 1:
            raise ValueError(f"iterations must be >= 1, got {self.iterations}")
        if self.min_std <= 0.0:
            raise ValueError(f"min_std must be positive, got {self.min_std}")

    @property
    def elite_count(self) -> int:
        return max(1, round(self.population * self.elite_fraction))

    @property
    def config_hash(self) -> str:
        return canonical_hash(dataclasses.asdict(self))


@dataclasses.dataclass(frozen=True)
class ParameterSpace:
    """What CEM is allowed to vary, and within which bounds.

    Bounds are hard: a sampled parameter is clipped, never allowed to wander
    outside the envelope the plan pinned.
    """

    speed_bounds: tuple[float, float] = (0.0, 0.3)
    grip_bounds: tuple[float, float] = (0.0, 1.0)
    duration_bounds: tuple[float, float] = (0.05, 1.5)

    def __post_init__(self) -> None:
        for name in ("speed_bounds", "grip_bounds", "duration_bounds"):
            low, high = getattr(self, name)
            if not (np.isfinite(low) and np.isfinite(high) and low < high):
                raise ValueError(f"{name} must be a finite increasing pair, got {(low, high)}")

    @property
    def dimensions(self) -> int:
        return 3

    def lower(self) -> np.ndarray:
        return np.array(
            [self.speed_bounds[0], self.grip_bounds[0], self.duration_bounds[0]]
        )

    def upper(self) -> np.ndarray:
        return np.array(
            [self.speed_bounds[1], self.grip_bounds[1], self.duration_bounds[1]]
        )

    def apply(self, template: Sequence[Primitive], sample: np.ndarray) -> tuple[Primitive, ...]:
        """Rewrite a primitive sequence with one sampled parameter vector."""
        speed, grip, duration = (float(v) for v in sample)
        return tuple(
            dataclasses.replace(
                primitive,
                speed=speed if primitive.speed > 0.0 else 0.0,
                grip=grip if primitive.grip > 0.0 else primitive.grip,
                max_duration_s=duration,
            )
            for primitive in template
        )


#: Why a search stopped. ``budget_exhausted`` means the declared budget ran out
#: with nothing feasible; ``no_feasible_elite`` means an iteration produced no
#: candidate worth refitting from, which is a stop rather than a licence to
#: refit from rejects (C04.5).
STOP_REASONS: frozenset[str] = frozenset(
    {"budget_exhausted", "no_feasible_elite", "none"}
)


@dataclasses.dataclass(frozen=True)
class CandidateRecord:
    """One evaluated candidate, and where it lived.

    The mapping from candidate index to world index to capsule is stable and
    recorded, because a ranking that cannot be traced back to the world it came
    from cannot be replayed (C04.5).
    """

    candidate_index: int
    iteration: int
    world_index: int
    sample: tuple[float, ...]
    score: float
    rejected: bool
    reason: str


@dataclasses.dataclass(frozen=True)
class CemResult:
    """Outcome of a bounded search."""

    best_sample: np.ndarray
    best_score: float
    best_outcome: DynamicSearchOutcome | None
    best_trajectory: DynamicGraspTrajectory | None
    iterations_run: int
    evaluated: int
    reason_ledger: dict[str, object]
    mean_history: tuple[tuple[float, ...], ...]
    stop_reason: str = "none"
    candidates: tuple[CandidateRecord, ...] = ()
    config_hash: str = ""
    weights_hash: str = ""

    def __post_init__(self) -> None:
        if self.stop_reason not in STOP_REASONS:
            raise ValueError(f"unknown stop reason {self.stop_reason!r}")
        # ``stop_reason == "none"`` is the success verdict: the search neither
        # ran out of budget nor ran out of feasible candidates. A search cannot
        # report that with nothing to hand back (C04.5).
        if self.stop_reason == "none" and self.best_outcome is None:
            raise ValueError(
                "a search that reports no failure stop must carry the outcome it succeeded with"
            )

    @property
    def succeeded(self) -> bool:
        return bool(self.best_outcome is not None and self.best_outcome.passed)

    @property
    def elite_count(self) -> int:
        return sum(1 for record in self.candidates if not record.rejected)


RolloutFn = Callable[
    [Sequence[Primitive]], "tuple[DynamicGraspTrajectory, DynamicSearchOutcome]"
]


def search_cem(
    *,
    template: Sequence[Primitive],
    rollout: RolloutFn,
    space: ParameterSpace | None = None,
    config: CemConfig | None = None,
    weights: ObjectiveWeights | None = None,
) -> CemResult:
    """Run a fixed-budget CEM search and report every candidate it burned."""
    space = space or ParameterSpace()
    config = config or CemConfig()
    weights = weights or ObjectiveWeights()

    rng = np.random.default_rng(config.seed)
    lower, upper = space.lower(), space.upper()
    mean = 0.5 * (lower + upper)
    std = 0.5 * (upper - lower)

    ledger = ReasonLedger()
    best_score = float("-inf")
    best_sample = mean.copy()
    best_outcome: DynamicSearchOutcome | None = None
    best_trajectory: DynamicGraspTrajectory | None = None
    evaluated = 0
    means: list[tuple[float, ...]] = []
    records: list[CandidateRecord] = []
    stop_reason = "none"
    iterations_run = 0

    for iteration in range(config.iterations):
        iterations_run = iteration + 1
        samples = rng.normal(mean, std, size=(config.population, space.dimensions))
        samples = np.clip(samples, lower, upper)

        scores = np.empty(config.population)
        for index in range(config.population):
            trajectory, outcome = rollout(space.apply(template, samples[index]))
            ledger.record(outcome)
            evaluated += 1
            evaluation = evaluate_objective(outcome, weights)
            scores[index] = evaluation.score
            records.append(
                CandidateRecord(
                    candidate_index=evaluated - 1,
                    iteration=iteration,
                    # One candidate per world, in sampling order, every
                    # iteration: the mapping is positional and stable.
                    world_index=index,
                    sample=tuple(float(v) for v in samples[index]),
                    score=evaluation.score,
                    rejected=evaluation.rejected,
                    reason=evaluation.reason,
                )
            )
            if evaluation.score > best_score:
                best_score, best_sample = evaluation.score, samples[index].copy()
                best_outcome, best_trajectory = outcome, trajectory

        feasible = np.flatnonzero(np.isfinite(scores))
        if feasible.size == 0:
            # Every candidate was rejected. Refitting from rejects would move
            # the distribution towards whatever failed least badly, which is not
            # information (C04.5).
            stop_reason = "no_feasible_elite"
            break

        take = min(config.elite_count, int(feasible.size))
        order = feasible[np.argsort(scores[feasible])]
        elite = samples[order[-take:]]
        mean = elite.mean(axis=0)
        std = np.maximum(elite.std(axis=0), config.min_std)
        means.append(tuple(float(v) for v in mean))

    if stop_reason == "none" and (best_outcome is None or not best_outcome.passed):
        stop_reason = "budget_exhausted"

    return CemResult(
        best_sample=best_sample,
        best_score=best_score,
        best_outcome=best_outcome,
        best_trajectory=best_trajectory,
        iterations_run=iterations_run,
        evaluated=evaluated,
        reason_ledger=ledger.to_dict(),
        mean_history=tuple(means),
        stop_reason=stop_reason,
        candidates=tuple(records),
        config_hash=config.config_hash,
        weights_hash=weights.weights_hash,
    )
