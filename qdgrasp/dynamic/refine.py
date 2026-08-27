"""Local refinement around a successful trajectory (P3.4-11).

Takes a candidate that already passed and looks for a safer version of the same
behaviour: more budget margin, less impulse, less energy. It is a coordinate
search in a shrinking neighbourhood, deliberately not a new search.

Two constraints keep it honest. It never accepts a candidate that fails, so
refinement cannot turn a pass into a near-miss it then relabels. And it never
touches a threshold -- only the control parameters.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence

import numpy as np

from qdgrasp.dataset.dynamic_contracts import (
    CpuReplayCertificate,
    DynamicGraspTrajectory,
    DynamicSearchOutcome,
    canonical_hash,
)
from qdgrasp.dynamic.cem import ParameterSpace, RolloutFn
from qdgrasp.dynamic.objective import ObjectiveWeights, score_outcome
from qdgrasp.dynamic.primitives import Primitive


@dataclasses.dataclass(frozen=True)
class RefineConfig:
    """Bounded neighbourhood search."""

    iterations: int = 3
    #: Initial step as a fraction of each parameter's range.
    initial_step_fraction: float = 0.1
    shrink: float = 0.5

    def __post_init__(self) -> None:
        if self.iterations < 1:
            raise ValueError(f"iterations must be >= 1, got {self.iterations}")
        if not 0.0 < self.initial_step_fraction <= 0.5:
            raise ValueError(
                f"initial_step_fraction must lie in (0, 0.5], got {self.initial_step_fraction}"
            )
        if not 0.0 < self.shrink < 1.0:
            raise ValueError(f"shrink must lie in (0, 1), got {self.shrink}")


@dataclasses.dataclass(frozen=True)
class RefineResult:
    sample: np.ndarray
    score: float
    outcome: DynamicSearchOutcome
    trajectory: DynamicGraspTrajectory
    improved: bool
    evaluated: int
    rejected_regressions: int
    #: Hashes of the objective and the bounds the refinement ran under. They are
    #: recorded rather than assumed so a reviewer can check that refinement did
    #: not quietly change what "better" means (C04.7).
    weights_hash: str = ""
    space_hash: str = ""


def refine_local(
    *,
    template: Sequence[Primitive],
    seed_sample: np.ndarray,
    seed_outcome: DynamicSearchOutcome,
    seed_trajectory: DynamicGraspTrajectory,
    rollout: RolloutFn,
    space: ParameterSpace | None = None,
    config: RefineConfig | None = None,
    weights: ObjectiveWeights | None = None,
) -> RefineResult:
    """Improve a passing trajectory without ever accepting a failing one."""
    if not seed_outcome.passed:
        raise ValueError(
            "local refinement starts from a passing trajectory; refining a "
            "failure would let a near-miss be relabelled as a success"
        )
    if not isinstance(seed_outcome.cpu_replay_evidence, CpuReplayCertificate):
        # A ValueError, not a TypeError: the argument is the right type, it just
        # has not been through the CPU oracle yet.
        raise ValueError(  # noqa: TRY004
            "local refinement starts from a CPU-confirmed positive: refining "
            "something only the GPU ranked would improve a result nobody has "
            "checked (C04.7)"
        )
    space = space or ParameterSpace()
    config = config or RefineConfig()
    weights = weights or ObjectiveWeights()

    lower, upper = space.lower(), space.upper()
    best_sample = np.clip(np.asarray(seed_sample, dtype=float), lower, upper)
    best_score = score_outcome(seed_outcome, weights)
    best_outcome, best_trajectory = seed_outcome, seed_trajectory

    step = (upper - lower) * config.initial_step_fraction
    evaluated = 0
    rejected = 0

    for _ in range(config.iterations):
        improved_this_round = False
        for axis in range(space.dimensions):
            for direction in (+1.0, -1.0):
                candidate = best_sample.copy()
                candidate[axis] = float(
                    np.clip(candidate[axis] + direction * step[axis], lower[axis], upper[axis])
                )
                if np.allclose(candidate, best_sample):
                    continue
                trajectory, outcome = rollout(space.apply(template, candidate))
                evaluated += 1
                if not outcome.passed:
                    rejected += 1
                    continue
                score = score_outcome(outcome, weights)
                if score > best_score:
                    best_sample, best_score = candidate, score
                    best_outcome, best_trajectory = outcome, trajectory
                    improved_this_round = True
        step = step * config.shrink
        if not improved_this_round:
            # Nothing in this neighbourhood helped; shrinking further is the
            # only remaining move, and the loop bound already caps that.
            continue

    return RefineResult(
        sample=best_sample,
        score=best_score,
        outcome=best_outcome,
        trajectory=best_trajectory,
        improved=best_score > score_outcome(seed_outcome, weights),
        evaluated=evaluated,
        rejected_regressions=rejected,
        weights_hash=weights.weights_hash,
        space_hash=canonical_hash(dataclasses.asdict(space)),
    )
