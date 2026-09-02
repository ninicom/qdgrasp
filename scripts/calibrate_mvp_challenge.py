#!/usr/bin/env python3
"""Survey one candidate Tier D domain on development-only seeds.

``ROADMAP-MVP-RELEASE-001`` §5 MR-03: the challenge tier only means something on
a domain the controller prior has not saturated, and that domain has to be found
before the candidate is trained and locked before the candidate is judged.  This
script measures one configuration and says whether it satisfies the rule the
scope froze -- it never picks the domain, and it never writes the locked
document.

The seeds come from the scope's ``challenge.development_seed_root``, which is
deliberately not the seed root Tier D draws from.  Nothing explored here may
appear in the tier that later judges the candidate, and
``test_mvp_release_contract.py`` asserts that the two sets are disjoint.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from qdgrasp.mvp.challenge import (
    ChallengeDomain,
    load_challenge_domain,
)
from qdgrasp.mvp.challenge import (
    challenge_development_seeds as development_seeds,
)
from qdgrasp.mvp.config import load_mvp_scope
from qdgrasp.mvp.evaluate import run_episodes, wilson_lower_bound, wilson_upper_bound
from qdgrasp.mvp.prior import DEFAULT_PRIOR_PATH

CALIBRATION_SCHEMA = "qdgrasp/mvp-challenge-calibration/v1"


def survey(
    scope: Any,
    domain: ChallengeDomain,
    *,
    scope_path: str,
    prior_path: str,
    domain_path: str,
    episodes: int,
    workers: int | None,
) -> dict[str, Any]:
    """Run the controller prior over one candidate domain and report it."""

    seeds = development_seeds(scope, episodes)
    # ``eval_d`` is the split the environment treats as the challenge tier, so
    # the domain is what gets sampled.  The seeds are development seeds; the
    # split name only selects which ranges apply.
    jobs = [(seed, "eval_d", True, None) for seed in seeds]
    started = time.time()
    results = run_episodes(
        jobs,
        scope_path=scope_path,
        prior_path=prior_path,
        challenge_path=domain_path,
        workers=workers,
    )
    return _report(scope, domain, results, elapsed=time.time() - started)


def _report(scope: Any, domain: ChallengeDomain, results: list[Any], *, elapsed: float) -> dict[str, Any]:
    episodes = len(results)
    successes = sum(1 for result in results if result.success)
    failures = episodes - successes
    invalid = sum(1 for result in results if result.invalid_state)
    violations = sum(1 for result in results if result.safety_violation)
    buckets = Counter(result.failure_bucket for result in results if not result.success)
    rate = successes / episodes if episodes else 0.0
    band_low, band_high = scope.challenge.prior_success_band

    admissible = {
        "prior_inside_band": band_low <= rate <= band_high,
        "enough_measurable_failures": failures >= scope.challenge.min_prior_failures,
        "zero_safety_violation": violations == 0,
        "zero_invalid_state": invalid == 0,
    }
    return {
        "schema": CALIBRATION_SCHEMA,
        "configuration_id": domain.configuration_id,
        "scope_hash": scope.content_hash(),
        "domain": domain.to_document(),
        "domain_hash": domain.content_hash(),
        "variant_ids": [variant.variant_id for variant in domain.variants(scope)],
        "episodes": episodes,
        "successes": successes,
        "failures": failures,
        "prior_success_rate": rate,
        "wilson_lower": wilson_lower_bound(successes, episodes),
        "wilson_upper": wilson_upper_bound(successes, episodes),
        "invalid_state": invalid,
        "safety_violation": violations,
        "failure_buckets": dict(sorted(buckets.items())),
        "required_band": [band_low, band_high],
        "min_prior_failures": scope.challenge.min_prior_failures,
        "admissible": admissible,
        "verdict": "admissible" if all(admissible.values()) else "rejected",
        "elapsed_s": round(elapsed, 1),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", type=Path, default=Path("configs/mvp/dexacquire-mvp-v1.yaml"))
    parser.add_argument("--prior", type=str, default=str(DEFAULT_PRIOR_PATH))
    parser.add_argument("--domain", type=Path, required=True, help="candidate challenge domain document")
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--out", type=Path, default=None, help="write the calibration report here")
    args = parser.parse_args(argv)

    scope = load_mvp_scope(args.scope)
    if scope.challenge is None:
        print("FAIL the scope declares no challenge contract")
        return 2
    domain = load_challenge_domain(args.domain, scope)

    report = survey(
        scope,
        domain,
        scope_path=str(args.scope),
        prior_path=args.prior,
        domain_path=str(args.domain),
        episodes=args.episodes,
        workers=args.workers,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0 if report["verdict"] == "admissible" else 1


if __name__ == "__main__":
    sys.exit(main())
