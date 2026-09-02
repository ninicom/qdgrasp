#!/usr/bin/env python3
"""MVP-03/04: clone the expert, then fine-tune the residual with PPO.

Both stages write self-describing checkpoints and their learning curves.  The
promotion rule is the plan's, not the optimiser's: PPO is only promoted if its
measured ``dev`` success is within two percentage points of the behaviour-cloned
baseline, and the BC checkpoint is always kept as the rollback.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from qdgrasp.mvp.bc import BehaviorCloningSpec, train_behavior_cloning
from qdgrasp.mvp.config import load_mvp_scope
from qdgrasp.mvp.contracts import TRAINING_REPORT_SCHEMA
from qdgrasp.mvp.env import environment_fingerprint
from qdgrasp.mvp.evaluate import run_episodes
from qdgrasp.mvp.expert import DemonstrationSet
from qdgrasp.mvp.policy import (
    MvpPolicy,
    build_from_checkpoint,
    checkpoint_reload_matches,
    load_checkpoint,
    save_checkpoint,
    verify_reload_probe,
)
from qdgrasp.mvp.ppo import PpoSpec, train_residual_ppo
from qdgrasp.mvp.prior import DEFAULT_PRIOR_PATH, PinchPriorTable

DEFAULT_DEMOS = Path("runs/mvp/demonstrations")
DEFAULT_OUTPUT = Path("runs/mvp/policy")

#: ``ROADMAP-MVP-001`` MVP-04: PPO may not be promoted if it costs more than two
#: percentage points of locked-eval-style success against the BC baseline.
PPO_PROMOTION_TOLERANCE = 0.02


def _commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.SubprocessError):  # pragma: no cover - git-less checkout
        return "unknown"


def measure_dev_success(
    checkpoint: Path, scope_path: str | None, prior_path: str, episodes: int, workers: int
) -> dict[str, Any]:
    """Measure a candidate on ``dev`` seeds -- never on the locked eval tiers."""

    scope = load_mvp_scope(scope_path)
    jobs = [(scope.episode_seed("dev", index), "dev", True, None) for index in range(episodes)]
    results = run_episodes(
        jobs, scope_path=scope_path, prior_path=prior_path, checkpoint_path=str(checkpoint), workers=workers
    )
    successes = sum(1 for result in results if result.success)
    buckets: dict[str, int] = {}
    for result in results:
        if not result.success:
            buckets[result.failure_bucket] = buckets.get(result.failure_bucket, 0) + 1
    return {
        "episodes": len(results),
        "successes": successes,
        "success_rate": successes / len(results) if results else 0.0,
        "safety_violation": sum(1 for result in results if result.safety_violation),
        "invalid_state": sum(1 for result in results if result.invalid_state),
        "failure_buckets": dict(sorted(buckets.items())),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", type=Path, default=None)
    parser.add_argument("--prior", type=Path, default=DEFAULT_PRIOR_PATH)
    parser.add_argument("--demos", type=Path, default=DEFAULT_DEMOS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bc-epochs", type=int, default=60)
    parser.add_argument("--ppo-iterations", type=int, default=40)
    parser.add_argument("--ppo-episodes", type=int, default=64)
    parser.add_argument("--dev-episodes", type=int, default=150)
    parser.add_argument("--skip-ppo", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = parser.parse_args(argv)

    scope = load_mvp_scope(args.scope)
    prior = PinchPriorTable.load(args.prior)
    fingerprint = environment_fingerprint(scope, prior)
    scope_path = str(args.scope) if args.scope is not None else None
    args.out.mkdir(parents=True, exist_ok=True)

    demonstrations = DemonstrationSet.load(args.demos / "train")
    demo_summary = demonstrations.summary()
    started = time.time()

    bc_spec = BehaviorCloningSpec(epochs=args.bc_epochs, seed=args.seed)
    network, normalizer, bc_metrics = train_behavior_cloning(demonstrations, bc_spec)
    bc_path = args.out / "bc.pt"
    save_checkpoint(
        bc_path,
        network,
        normalizer,
        fingerprint=fingerprint,
        stage="bc",
        metadata={
            "commit": _commit(),
            "dataset_summary": demo_summary,
            "bc_metrics": {key: value for key, value in bc_metrics.items() if key != "curve"},
        },
    )
    sample = demonstrations.observations[:64]
    in_memory = MvpPolicy(network, normalizer)
    reload_ok = checkpoint_reload_matches(
        bc_path, sample, np.stack([in_memory(row) for row in sample])
    ) and verify_reload_probe(bc_path)
    bc_dev = measure_dev_success(bc_path, scope_path, str(args.prior), args.dev_episodes, args.workers)
    print(f"[bc] reload_parity={reload_ok} dev={json.dumps(bc_dev, sort_keys=True)}")

    report: dict[str, Any] = {
        "schema": TRAINING_REPORT_SCHEMA,
        "commit": _commit(),
        "fingerprint": fingerprint,
        "demonstrations": demo_summary,
        "bc": {"metrics": bc_metrics, "reload_parity": reload_ok, "dev": bc_dev, "checkpoint": str(bc_path)},
    }

    if not args.skip_ppo:
        ppo_spec = PpoSpec(iterations=args.ppo_iterations, episodes_per_iteration=args.ppo_episodes, seed=args.seed)
        payload = load_checkpoint(bc_path)
        ppo_network, ppo_normalizer = build_from_checkpoint(payload)
        ppo_network, ppo_metrics = train_residual_ppo(
            ppo_network,
            ppo_normalizer,
            scope,
            spec=ppo_spec,
            scope_path=scope_path,
            prior_path=str(args.prior),
            fingerprint=fingerprint,
            workers=args.workers,
            output_dir=args.out / "ppo-staging",
            # Start past the seeds the demonstrations consumed so PPO trains on
            # episodes the expert search has not already been shown.
            seed_offset=10_000,
        )
        ppo_path = args.out / "ppo.pt"
        save_checkpoint(
            ppo_path,
            ppo_network,
            ppo_normalizer,
            fingerprint=fingerprint,
            stage="ppo",
            metadata={
                "commit": _commit(),
                "parent": str(bc_path),
                "ppo_metrics": {key: value for key, value in ppo_metrics.items() if key != "curve"},
            },
        )
        ppo_dev = measure_dev_success(ppo_path, scope_path, str(args.prior), args.dev_episodes, args.workers)
        promoted = ppo_dev["success_rate"] >= bc_dev["success_rate"] - PPO_PROMOTION_TOLERANCE
        print(f"[ppo] dev={json.dumps(ppo_dev, sort_keys=True)} promoted={promoted}")
        report["ppo"] = {
            "metrics": ppo_metrics,
            "dev": ppo_dev,
            "checkpoint": str(ppo_path),
            "promoted": promoted,
            "promotion_tolerance": PPO_PROMOTION_TOLERANCE,
        }
        report["candidate"] = str(ppo_path if promoted else bc_path)
    else:
        report["candidate"] = str(bc_path)

    # A report may only advertise the action contract and lineage that the
    # candidate itself carries.  Reloading through the current safe loader also
    # prevents a report from being written around a stale checkpoint schema.
    candidate_payload = load_checkpoint(report["candidate"])
    report["action_distribution"] = candidate_payload["architecture"]["action_distribution"]
    report["lineage"] = candidate_payload["lineage"]
    report["elapsed_s"] = round(time.time() - started, 1)
    (args.out / "training-report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"candidate: {report['candidate']}  ({report['elapsed_s']}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
