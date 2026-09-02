#!/usr/bin/env python3
"""Development probe: what can a constant residual do on the challenge domain?

``ROADMAP-MVP-RELEASE-001`` §8.8 requires a `NO-GO` to keep its evidence so the
cause can be decided, and forbids repeating a run without changing the
hypothesis.  This script is how the hypothesis gets changed on evidence rather
than on a hunch: it asks whether the *locked action space* contains any residual
that recovers the failures the controller prior leaves on the challenge domain,
before anything is trained.

It answers a question about the environment, not about a candidate.  It reads
development seeds only -- the calibration root, proven disjoint from every
locked tier -- and it writes no checkpoint and selects nothing.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from qdgrasp.mvp.challenge import challenge_development_seeds, load_challenge_domain
from qdgrasp.mvp.config import MvpScopeConfig, load_mvp_scope
from qdgrasp.mvp.env import DexAcquireMvpEnv
from qdgrasp.mvp.evaluate import wilson_lower_bound
from qdgrasp.mvp.prior import DEFAULT_PRIOR_PATH, PinchPriorTable

PROBE_SCHEMA = "qdgrasp/mvp-residual-probe/v1"

_WORKER: dict[str, Any] = {}


def _init(scope_path: str, prior_path: str, domain_path: str | None) -> None:
    scope = load_mvp_scope(scope_path)
    prior = PinchPriorTable.load(prior_path)
    domain = load_challenge_domain(domain_path, scope) if domain_path is not None else None
    _WORKER["env"] = DexAcquireMvpEnv(scope, prior, challenge=domain)
    _WORKER["scope"] = scope


def _segment(step_index: int, scope: MvpScopeConfig) -> int:
    return 0 if step_index < scope.episode.approach_steps + scope.episode.enclose_steps else 1


def _rollout(job: tuple[int, str, list[list[float]] | None]) -> dict[str, Any]:
    seed, split, candidate = job
    env, scope = _WORKER["env"], _WORKER["scope"]
    env.reset(seed, split)  # type: ignore[arg-type]
    zeros = np.zeros(scope.action.dimension)
    while not env.done:
        action = zeros if candidate is None else np.asarray(candidate[_segment(env.step_index, scope)])
        env.step(action)
    result = env.result
    assert result is not None
    return {
        "success": bool(result.success),
        "failure_bucket": result.failure_bucket,
        "safety_violation": bool(result.safety_violation),
        "invalid_state": bool(result.invalid_state),
        "max_contact_force_n": float(result.max_contact_force_n),
    }


def carry_synergy(first: float, second: float) -> list[list[float]]:
    """Leave the approach alone and close harder once the grasp is closed.

    The approach deliberately spreads the fingers wider than the grasp needs so
    a descending fingertip clears the target's side; closing during that phase
    is the jam the controller prior was designed to avoid.
    """

    candidate = np.zeros((2, 8))
    candidate[1, 6] = first
    candidate[1, 7] = second
    return candidate.tolist()


def _run(
    seeds: list[int],
    split: str,
    candidate: list[list[float]] | None,
    *,
    scope_path: str,
    prior_path: str,
    domain_path: str | None,
    workers: int,
) -> list[dict[str, Any]]:
    jobs = [(seed, split, candidate) for seed in seeds]
    if workers <= 1:
        _init(scope_path, prior_path, domain_path)
        return [_rollout(job) for job in jobs]
    with ProcessPoolExecutor(
        max_workers=workers, initializer=_init, initargs=(scope_path, prior_path, domain_path)
    ) as pool:
        return list(pool.map(_rollout, jobs, chunksize=4))


def _compare(baseline: list[dict[str, Any]], arm: list[dict[str, Any]]) -> dict[str, Any]:
    before = [row["success"] for row in baseline]
    after = [row["success"] for row in arm]
    successes = sum(after)
    return {
        "episodes": len(after),
        "successes": successes,
        "success_rate": successes / len(after),
        "baseline_successes": sum(before),
        "net_pp": 100.0 * (successes - sum(before)) / len(after),
        "gained": sum(1 for x, y in zip(before, after, strict=True) if not x and y),
        "lost": sum(1 for x, y in zip(before, after, strict=True) if x and not y),
        "safety_violation": sum(row["safety_violation"] for row in arm),
        "invalid_state": sum(row["invalid_state"] for row in arm),
        "max_contact_force_n": max(row["max_contact_force_n"] for row in arm),
        "failure_buckets": dict(sorted(Counter(r["failure_bucket"] for r in arm if not r["success"]).items())),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", type=Path, default=Path("configs/mvp/dexacquire-mvp-v1.yaml"))
    parser.add_argument("--prior", type=Path, default=DEFAULT_PRIOR_PATH)
    parser.add_argument("--challenge", type=Path, default=Path("configs/mvp/dexacquire-mvp-v1.challenge.json"))
    parser.add_argument("--challenge-episodes", type=int, default=300)
    parser.add_argument("--dev-episodes", type=int, default=200)
    parser.add_argument("--workers", type=int, default=7)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    scope = load_mvp_scope(args.scope)
    common = {
        "scope_path": str(args.scope),
        "prior_path": str(args.prior),
        "workers": args.workers,
    }
    grid = [(0.0, 0.75), (0.25, 0.75), (0.5, 0.5), (1.0, 0.0)]

    challenge_seeds = challenge_development_seeds(scope, args.challenge_episodes)
    challenge_prior = _run(challenge_seeds, "eval_d", None, domain_path=str(args.challenge), **common)
    dev_seeds = [scope.episode_seed("dev", index) for index in range(args.dev_episodes)]
    dev_prior = _run(dev_seeds, "dev", None, domain_path=None, **common)

    report: dict[str, Any] = {
        "schema": PROBE_SCHEMA,
        "scope_hash": scope.content_hash(),
        "challenge_domain_hash": load_challenge_domain(args.challenge, scope).content_hash(),
        "controller_prior": {
            "challenge": _compare(challenge_prior, challenge_prior),
            "base_dev": _compare(dev_prior, dev_prior),
        },
        "constant_residuals": {},
    }
    for first, second in grid:
        label = f"carry_synergy_{first}_{second}"
        candidate = carry_synergy(first, second)
        challenge_arm = _run(challenge_seeds, "eval_d", candidate, domain_path=str(args.challenge), **common)
        dev_arm = _run(dev_seeds, "dev", candidate, domain_path=None, **common)
        report["constant_residuals"][label] = {
            "residual": candidate,
            "challenge": _compare(challenge_prior, challenge_arm),
            "base_dev": _compare(dev_prior, dev_arm),
        }
        challenge = report["constant_residuals"][label]["challenge"]
        base = report["constant_residuals"][label]["base_dev"]
        print(
            f"{label:26s} challenge {challenge['successes']:3d}/{challenge['episodes']} "
            f"({challenge['net_pp']:+5.2f} pp, +{challenge['gained']}/-{challenge['lost']})  "
            f"base_dev ({base['net_pp']:+5.2f} pp, +{base['gained']}/-{base['lost']})  "
            f"unsafe {challenge['safety_violation'] + base['safety_violation']}  "
            f"maxF {max(challenge['max_contact_force_n'], base['max_contact_force_n']):.2f} N"
        )

    best = max(report["constant_residuals"].items(), key=lambda item: item[1]["challenge"]["net_pp"])
    report["best_constant"] = {
        "label": best[0],
        "challenge_net_pp": best[1]["challenge"]["net_pp"],
        "challenge_wilson_lower": wilson_lower_bound(
            best[1]["challenge"]["successes"], best[1]["challenge"]["episodes"]
        ),
        "base_dev_net_pp": best[1]["base_dev"]["net_pp"],
    }
    # The finding this probe exists to settle.
    report["locked_action_space_can_recover_failures"] = bool(best[1]["challenge"]["net_pp"] > 0.0)
    print(f"\nbest constant: {best[0]} at {best[1]['challenge']['net_pp']:+.2f} pp on the challenge domain")

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
