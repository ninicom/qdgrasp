#!/usr/bin/env python3
"""MVP-02: record expert demonstrations and the generator ledger.

Runs the residual search on ``train`` and ``dev`` seeds, keeps only episodes
that satisfy the measured success predicate, and writes both the accepted
transitions and the full attempt ledger.  The ledger is the point: an expert set
whose rejection rate is invisible cannot be calibrated by anyone downstream.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from qdgrasp.mvp.challenge import load_challenge_domain
from qdgrasp.mvp.config import load_mvp_scope
from qdgrasp.mvp.env import DexAcquireMvpEnv, environment_fingerprint
from qdgrasp.mvp.expert import (
    DEMONSTRATION_INDEX_SCHEMA,
    DemonstrationSet,
    ExpertSearchSpec,
    search_expert_episode,
)
from qdgrasp.mvp.prior import DEFAULT_PRIOR_PATH, PinchPriorTable

DEFAULT_OUTPUT = Path("runs/mvp/demonstrations")

#: Challenge-domain demonstrations draw from the same train/dev seed roots but
#: at indices no base episode uses, so the two halves of a split can never
#: collide while both stay disjoint from every locked tier.
CHALLENGE_SEED_OFFSET = 1_000_000

_WORKER: dict[str, Any] = {}


def _commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):  # pragma: no cover - git-less checkout
        return "unknown"


def _init(
    scope_path: str | None,
    prior_path: str,
    candidates: int,
    sigma: float,
    challenge_path: str | None = None,
) -> None:
    scope = load_mvp_scope(scope_path)
    challenge = load_challenge_domain(challenge_path, scope) if challenge_path is not None else None
    _WORKER["env"] = DexAcquireMvpEnv(scope, PinchPriorTable.load(prior_path), challenge=challenge)
    _WORKER["spec"] = ExpertSearchSpec(candidates=candidates, sigma=sigma)


def _search(job: tuple[int, str, bool]) -> tuple[dict[str, Any], np.ndarray | None, np.ndarray | None]:
    seed, split, challenged = job
    episode, row = search_expert_episode(
        _WORKER["env"],
        seed,
        split,  # type: ignore[arg-type]
        _WORKER["spec"],
        challenged=challenged,
    )
    if episode is None:
        return row, None, None
    return row, episode.observations, episode.actions


def collect(
    split: str,
    jobs: list[tuple[int, str, bool]],
    workers: int,
    out: Path,
    args: argparse.Namespace,
) -> DemonstrationSet:
    initargs = (
        args.scope and str(args.scope),
        str(args.prior),
        args.candidates,
        args.sigma,
        args.challenge and str(args.challenge),
    )
    if workers <= 1:
        _init(*initargs)
        outputs = [_search(job) for job in jobs]
    else:
        with ProcessPoolExecutor(max_workers=workers, initializer=_init, initargs=initargs) as pool:
            outputs = list(pool.map(_search, jobs, chunksize=2))

    observations: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    episode_index: list[np.ndarray] = []
    variant_ids: list[str] = []
    accepted_seeds: list[int] = []
    ledger: list[dict[str, Any]] = []
    for row, episode_observations, episode_actions in outputs:
        ledger.append(row)
        if episode_observations is None or episode_actions is None:
            continue
        index = len(accepted_seeds)
        observations.append(episode_observations)
        actions.append(episode_actions)
        episode_index.append(np.full(episode_observations.shape[0], index, dtype=np.int64))
        variant_ids.append(str(row["variant_id"]))
        accepted_seeds.append(int(row["seed"]))
    if not observations:
        raise RuntimeError(f"no demonstration was accepted for split {split!r}")
    dataset = DemonstrationSet(
        observations=np.concatenate(observations),
        actions=np.concatenate(actions),
        episode_index=np.concatenate(episode_index),
        variant_ids=tuple(variant_ids),
        seeds=tuple(accepted_seeds),
        ledger=ledger,
    )
    dataset.save(out / split)
    return dataset


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", type=Path, default=None)
    parser.add_argument("--prior", type=Path, default=DEFAULT_PRIOR_PATH)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--train-episodes", type=int, default=400)
    parser.add_argument("--dev-episodes", type=int, default=120)
    parser.add_argument("--challenge", type=Path, default=None, help="locked challenge domain document")
    parser.add_argument(
        "--challenge-train-episodes",
        type=int,
        default=0,
        help="extra train episodes drawn from the challenge domain",
    )
    parser.add_argument(
        "--challenge-dev-episodes",
        type=int,
        default=0,
        help="extra dev episodes drawn from the challenge domain",
    )
    parser.add_argument("--candidates", type=int, default=12)
    parser.add_argument("--sigma", type=float, default=0.35)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = parser.parse_args(argv)

    scope = load_mvp_scope(args.scope)
    prior = PinchPriorTable.load(args.prior)
    challenge = load_challenge_domain(args.challenge, scope) if args.challenge is not None else None
    if challenge is None and (args.challenge_train_episodes or args.challenge_dev_episodes):
        parser.error("challenge episodes were requested without a challenge domain")
    search_spec = ExpertSearchSpec(candidates=args.candidates, sigma=args.sigma)
    started = time.time()
    summaries: dict[str, Any] = {}
    counts = {
        "train": (args.train_episodes, args.challenge_train_episodes),
        "dev": (args.dev_episodes, args.challenge_dev_episodes),
    }
    for split, (base_count, challenge_count) in counts.items():
        # The base half teaches the policy to leave a working grasp alone; the
        # challenge half is the only place the expert search has anything to
        # rescue, because on the base domain the prior already succeeds.
        jobs: list[tuple[int, str, bool]] = [
            (scope.episode_seed(split, index), split, False)  # type: ignore[arg-type]
            for index in range(base_count)
        ]
        jobs += [
            (scope.episode_seed(split, CHALLENGE_SEED_OFFSET + index), split, True)  # type: ignore[arg-type]
            for index in range(challenge_count)
        ]
        dataset = collect(split, jobs, args.workers, args.out, args)
        summaries[split] = dataset.summary()
        print(f"[{split}] {json.dumps(summaries[split], sort_keys=True)}")

    index = {
        "schema": DEMONSTRATION_INDEX_SCHEMA,
        "commit": _commit(),
        "fingerprint": environment_fingerprint(scope, prior),
        "scope_hash": scope.content_hash(),
        "prior_hash": prior.content_hash(),
        "challenge_domain_hash": challenge.content_hash() if challenge is not None else None,
        "generator_config": {
            "train_episodes": args.train_episodes,
            "dev_episodes": args.dev_episodes,
            "challenge_train_episodes": args.challenge_train_episodes,
            "challenge_dev_episodes": args.challenge_dev_episodes,
            "challenge_seed_offset": CHALLENGE_SEED_OFFSET,
            "expert_search": search_spec.to_document(),
        },
        "elapsed_s": round(time.time() - started, 1),
        "splits": summaries,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "index.json").write_text(json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {args.out / 'index.json'} in {index['elapsed_s']}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
