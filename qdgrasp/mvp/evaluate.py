"""Locked evaluation of an MVP candidate (``ROADMAP-MVP-001`` §7).

The report this produces is deliberately unflattering by construction: every
timeout, safety termination, invalid state and simulator error stays in the
denominator, and the failure buckets are printed beside the headline rate so a
single pretty average cannot stand alone.

A tier is run from the seeds in the immutable evaluation manifest, so "we ran
Tier B" is a checkable statement rather than a description of a mood.
"""

from __future__ import annotations

import dataclasses
import json
import math
import multiprocessing
import os
from collections import Counter
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from qdgrasp.mvp.config import EvalTier, MvpScopeConfig, load_mvp_scope
from qdgrasp.mvp.env import DexAcquireMvpEnv, EpisodeResult, environment_fingerprint, write_episode_ledger
from qdgrasp.mvp.prior import DEFAULT_PRIOR_PATH, PinchPriorTable

EVAL_REPORT_SCHEMA_V0 = "qdgrasp/mvp-eval-report/v0"
EVAL_REPORT_SCHEMA_V1 = "qdgrasp/mvp-eval-report/v1"
EVAL_REPORT_SCHEMA = EVAL_REPORT_SCHEMA_V1

#: A policy is anything that maps an observation to a unit action.  ``None``
#: means the controller prior alone, which is the baseline every tier is read
#: against.
Policy = Callable[[np.ndarray], np.ndarray]


def wilson_lower_bound(successes: int, trials: int, z: float = 1.959963984540054) -> float:
    """Wilson score lower bound at 95% confidence."""

    if trials <= 0:
        return 0.0
    phat = successes / trials
    denominator = 1.0 + z * z / trials
    centre = phat + z * z / (2.0 * trials)
    spread = z * math.sqrt((phat * (1.0 - phat) + z * z / (4.0 * trials)) / trials)
    return float(max(0.0, (centre - spread) / denominator))


def wilson_upper_bound(successes: int, trials: int, z: float = 1.959963984540054) -> float:
    if trials <= 0:
        return 1.0
    phat = successes / trials
    denominator = 1.0 + z * z / trials
    centre = phat + z * z / (2.0 * trials)
    spread = z * math.sqrt((phat * (1.0 - phat) + z * z / (4.0 * trials)) / trials)
    return float(min(1.0, (centre + spread) / denominator))


PAIRED_COMPARISON_SCHEMA = "qdgrasp/mvp-paired-comparison/v1"


def paired_uplift(
    prior: Sequence[bool],
    candidate: Sequence[bool],
    *,
    resamples: int,
    seed: int,
    confidence: float = 0.95,
) -> dict[str, Any]:
    """Paired bootstrap of the candidate's uplift over the prior, in points.

    Paired, because the two arms ran the same seeds: comparing two independent
    Wilson intervals on matched data throws away the pairing and widens the
    interval for nothing.  Percentile bootstrap, because the statistic is a
    difference of proportions on a few hundred correlated Bernoulli pairs,
    where a normal approximation is exactly the assumption in doubt.

    Deterministic by construction -- the resample count and the seed are inputs
    and come back out in the result -- so the interval can be recomputed from
    the ledgers and cannot be quietly re-rolled until it clears zero.
    """

    if len(prior) != len(candidate):
        raise ValueError(f"paired comparison needs matched arms: prior={len(prior)}, candidate={len(candidate)}")
    if not prior:
        raise ValueError("paired comparison needs at least one episode")
    if resamples <= 0:
        raise ValueError("paired comparison needs a positive resample count")
    if not 0.5 < confidence < 1.0:
        raise ValueError(f"confidence must lie in (0.5, 1.0): {confidence}")

    prior_outcomes = np.asarray(prior, dtype=bool)
    candidate_outcomes = np.asarray(candidate, dtype=bool)
    difference = candidate_outcomes.astype(np.float64) - prior_outcomes.astype(np.float64)
    episodes = difference.size

    generator = np.random.default_rng(seed)
    indices = generator.integers(0, episodes, size=(resamples, episodes))
    resampled = difference[indices].mean(axis=1)
    alpha = 1.0 - confidence
    lower, upper = np.quantile(resampled, [alpha / 2.0, 1.0 - alpha / 2.0])

    discordant_candidate = int(np.sum(candidate_outcomes & ~prior_outcomes))
    discordant_prior = int(np.sum(prior_outcomes & ~candidate_outcomes))
    return {
        "schema": PAIRED_COMPARISON_SCHEMA,
        "method": "paired_bootstrap_percentile",
        "episodes": episodes,
        "prior_successes": int(prior_outcomes.sum()),
        "candidate_successes": int(candidate_outcomes.sum()),
        # Percentage points, which is the unit the release gate is written in.
        "uplift_pp": float(difference.mean() * 100.0),
        "ci_lower_pp": float(lower * 100.0),
        "ci_upper_pp": float(upper * 100.0),
        "confidence": float(confidence),
        "resamples": int(resamples),
        "seed": int(seed),
        # The pairs that carry the whole signal: everything else cancels.
        "candidate_only_successes": discordant_candidate,
        "prior_only_successes": discordant_prior,
    }


@dataclasses.dataclass(frozen=True)
class TierReport:
    """Aggregate outcome of one acceptance tier."""

    tier: str
    episodes: int
    successes: int
    success_rate: float
    wilson_lower: float
    wilson_upper: float
    invalid_state: int
    safety_violation: int
    checkpoint_reload_mismatch: int
    failure_buckets: dict[str, int]
    #: ``None`` on a tier gated on uplift over the prior rather than on a level.
    min_success_rate: float | None
    min_wilson_lower_bound: float | None
    passed: bool
    ledger_path: str | None = None

    def to_document(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def _aggregate(
    tier: EvalTier,
    scope: MvpScopeConfig,
    results: Sequence[EpisodeResult],
    *,
    reload_mismatch: int = 0,
    ledger_path: str | None = None,
) -> TierReport:
    spec = scope.tier(tier)
    successes = sum(1 for result in results if result.success)
    episodes = len(results)
    rate = successes / episodes if episodes else 0.0
    lower = wilson_lower_bound(successes, episodes)
    invalid = sum(1 for result in results if result.invalid_state)
    violations = sum(1 for result in results if result.safety_violation)
    buckets = Counter(result.failure_bucket for result in results if not result.success)
    # A challenge tier's verdict is a paired comparison against the prior, which
    # this function cannot see: it holds one arm.  What is settled here is that
    # the tier ran cleanly, at its declared size, and cleared whatever absolute
    # floor it has.  The uplift gate is applied by the release checker.
    passed = (
        episodes == spec.episodes
        and (spec.min_success_rate is None or rate >= spec.min_success_rate)
        and (spec.min_wilson_lower_bound is None or lower >= spec.min_wilson_lower_bound)
        and invalid == 0
        and violations == 0
        and reload_mismatch == 0
    )
    return TierReport(
        tier=tier,
        episodes=episodes,
        successes=successes,
        success_rate=rate,
        wilson_lower=lower,
        wilson_upper=wilson_upper_bound(successes, episodes),
        invalid_state=invalid,
        safety_violation=violations,
        checkpoint_reload_mismatch=reload_mismatch,
        failure_buckets=dict(sorted(buckets.items())),
        min_success_rate=spec.min_success_rate,
        min_wilson_lower_bound=spec.min_wilson_lower_bound,
        passed=passed,
        ledger_path=ledger_path,
    )


# -- worker plumbing ------------------------------------------------------
#
# Episodes are independent, so they parallelise trivially; what does not
# parallelise is a MuJoCo model, so each worker builds its own environment once
# and keeps it for the life of the process.
#
# The pool is spawned, never forked.  A caller that has just trained a torch
# model holds OpenMP and allocator locks, and a forked child inherits them in a
# locked state and then blocks forever on its first allocation -- observed here
# as eight workers idling at 0.1% CPU while the parent waited for results that
# were never coming.  Spawn costs a few seconds of re-import per worker and buys
# a pool that actually runs.
_SPAWN = multiprocessing.get_context("spawn")

_WORKER: dict[str, Any] = {}


def verify_checkpoint_fingerprint(
    checkpoint_path: str | Path | None,
    expected: dict[str, str],
) -> dict[str, Any]:
    """Verify a candidate identity before an environment or episode exists."""

    if checkpoint_path is None:
        return {"stored": None, "effective": dict(expected), "verdict": "not_applicable"}
    from qdgrasp.mvp.policy import load_checkpoint

    payload = load_checkpoint(checkpoint_path)
    stored = payload.get("fingerprint")
    if not isinstance(stored, dict):
        raise TypeError(f"checkpoint {checkpoint_path} has no fingerprint mapping")
    mismatched = {
        key: {"stored": stored.get(key), "effective": value}
        for key, value in expected.items()
        if stored.get(key) != value
    }
    extra = sorted(set(stored) - set(expected))
    if mismatched or extra:
        raise ValueError(
            f"checkpoint fingerprint does not match this environment: mismatched={mismatched}, extra={extra}"
        )
    return {"stored": dict(stored), "effective": dict(expected), "verdict": "match"}


def _limit_worker_threads() -> None:
    """One torch thread per worker: the parallelism is across episodes."""

    import torch

    torch.set_num_threads(1)


def _worker_init(scope_path: str | None, prior_path: str, checkpoint_path: str | None) -> None:
    _limit_worker_threads()
    _WORKER.clear()
    scope = load_mvp_scope(scope_path)
    prior = PinchPriorTable.load(prior_path)
    expected = environment_fingerprint(scope, prior)
    verification = verify_checkpoint_fingerprint(checkpoint_path, expected)
    policy = None
    if checkpoint_path is not None:
        from qdgrasp.mvp.policy import load_policy  # local import: torch is heavy

        policy = load_policy(checkpoint_path, fingerprint=expected)
    # Construct the simulator only after the artifact has been admitted.  A
    # foreign checkpoint must not be able to produce even an empty ledger row.
    env = DexAcquireMvpEnv(scope, prior)
    _WORKER.update({"env": env, "policy": policy, "fingerprint_verification": verification})


def _worker_episode(job: tuple[int, str, bool, str | None]) -> dict[str, Any]:
    seed, split, randomized, variant_id = job
    env: DexAcquireMvpEnv = _WORKER["env"]
    policy = _WORKER["policy"]
    result = env.run_episode(
        seed,
        split,  # type: ignore[arg-type]
        policy=policy,
        randomized=randomized,
        variant_id=variant_id,
    )
    return result.to_document()


def _rehydrate(document: dict[str, Any]) -> EpisodeResult:
    from qdgrasp.mvp.env import EpisodeSetup

    setup = EpisodeSetup(**{**document["setup"], "position": tuple(document["setup"]["position"])})
    return EpisodeResult(**{**document, "setup": setup})


def run_episodes(
    jobs: Sequence[tuple[int, str, bool, str | None]],
    *,
    scope_path: str | None = None,
    prior_path: str = str(DEFAULT_PRIOR_PATH),
    checkpoint_path: str | None = None,
    workers: int | None = None,
) -> list[EpisodeResult]:
    """Run a batch of episodes, in parallel when more than one worker is asked."""

    # Parent-side preflight makes failure synchronous and guarantees no worker
    # has constructed an environment before identity is known.
    scope = load_mvp_scope(scope_path)
    prior = PinchPriorTable.load(prior_path)
    verify_checkpoint_fingerprint(checkpoint_path, environment_fingerprint(scope, prior))

    count = workers if workers is not None else max(1, (os.cpu_count() or 2) - 1)
    if count <= 1:
        _worker_init(scope_path, prior_path, checkpoint_path)
        return [_rehydrate(_worker_episode(job)) for job in jobs]
    with ProcessPoolExecutor(
        max_workers=count,
        mp_context=_SPAWN,
        initializer=_worker_init,
        initargs=(scope_path, prior_path, checkpoint_path),
    ) as pool:
        return [_rehydrate(document) for document in pool.map(_worker_episode, jobs, chunksize=4)]


def evaluate_tier(
    tier: EvalTier,
    scope: MvpScopeConfig,
    *,
    scope_path: str | None = None,
    prior_path: str = str(DEFAULT_PRIOR_PATH),
    checkpoint_path: str | None = None,
    workers: int | None = None,
    ledger_dir: str | Path | None = None,
) -> TierReport:
    """Run one acceptance tier on its locked seeds and aggregate the verdict."""

    spec = scope.tier(tier)
    if spec.challenge_domain:
        # The challenge domain is calibrated after the scope is frozen and
        # lives in its own locked document.  Running this tier under the base
        # randomization would produce a report that looks like Tier D and is
        # not, so it fails closed rather than silently measuring the wrong
        # thing (``ROADMAP-MVP-RELEASE-001`` §5 MR-03).
        raise NotImplementedError(
            f"tier {tier} draws a locked challenge domain that this evaluator has not been given; "
            "it cannot be run from the scope's base randomization"
        )
    split = f"eval_{tier.lower()}"
    seeds = scope.locked_seeds(tier)
    jobs = [(seed, split, spec.randomized, None) for seed in seeds]
    results = run_episodes(
        jobs,
        scope_path=scope_path,
        prior_path=prior_path,
        checkpoint_path=checkpoint_path,
        workers=workers,
    )
    ledger_path: str | None = None
    if ledger_dir is not None:
        ledger_path = str(write_episode_ledger(Path(ledger_dir) / f"tier-{tier.lower()}.jsonl", results))
    return _aggregate(tier, scope, results, ledger_path=ledger_path)


def evaluate_candidate(
    scope: MvpScopeConfig,
    prior: PinchPriorTable,
    *,
    scope_path: str | None = None,
    prior_path: str = str(DEFAULT_PRIOR_PATH),
    checkpoint_path: str | None = None,
    workers: int | None = None,
    ledger_dir: str | Path | None = None,
    reload_mismatch: int = 0,
    label: str = "controller_prior",
) -> dict[str, Any]:
    """Run every tier and produce the locked-evaluation report document."""

    effective_fingerprint = environment_fingerprint(scope, prior)
    verification = verify_checkpoint_fingerprint(checkpoint_path, effective_fingerprint)
    reports = [
        evaluate_tier(
            tier,
            scope,
            scope_path=scope_path,
            prior_path=prior_path,
            checkpoint_path=checkpoint_path,
            workers=workers,
            ledger_dir=ledger_dir,
        )
        for tier in sorted(spec.tier for spec in scope.eval_tiers)
    ]
    if reload_mismatch:
        reports = [
            dataclasses.replace(item, checkpoint_reload_mismatch=reload_mismatch, passed=False) for item in reports
        ]
    return {
        "schema": EVAL_REPORT_SCHEMA,
        "candidate": label,
        "checkpoint": checkpoint_path,
        # Retained as the convenient effective identity for existing report
        # consumers; ``checkpoint_fingerprint`` is the audit trail that proves
        # whether those values came from the artifact or only from the live env.
        "fingerprint": effective_fingerprint,
        "checkpoint_fingerprint": verification,
        "tiers": [report.to_document() for report in reports],
        "all_tiers_passed": all(report.passed for report in reports),
    }


def write_report(path: str | Path, report: dict[str, Any]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return target


def format_report(report: dict[str, Any]) -> str:
    """A short human-readable rendering that keeps the failure buckets visible."""

    lines = [f"candidate: {report['candidate']}"]
    for tier in report["tiers"]:
        gate = f">={tier['min_success_rate']:.0%}"
        if tier["min_wilson_lower_bound"] is not None:
            gate += f", wilson>={tier['min_wilson_lower_bound']:.0%}"
        lines.append(
            f"  tier {tier['tier']}: {tier['successes']}/{tier['episodes']} = {tier['success_rate']:.1%} "
            f"[{tier['wilson_lower']:.1%}, {tier['wilson_upper']:.1%}]  gate {gate}  "
            f"{'PASS' if tier['passed'] else 'FAIL'}"
        )
        lines.append(
            f"    invalid={tier['invalid_state']} safety={tier['safety_violation']} "
            f"reload_mismatch={tier['checkpoint_reload_mismatch']} buckets={tier['failure_buckets']}"
        )
    lines.append(f"  overall: {'PASS' if report['all_tiers_passed'] else 'FAIL'}")
    return "\n".join(lines)
