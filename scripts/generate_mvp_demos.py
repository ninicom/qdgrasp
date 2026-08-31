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
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from qdgrasp.mvp.config import load_mvp_scope
from qdgrasp.mvp.env import DexAcquireMvpEnv
from qdgrasp.mvp.expert import DemonstrationSet, ExpertSearchSpec, search_expert_episode
from qdgrasp.mvp.prior import DEFAULT_PRIOR_PATH, PinchPriorTable

DEFAULT_OUTPUT = Path("runs/mvp/demonstrations")

_WORKER: dict[str, Any] = {}


def _init(scope_path: str | None, prior_path: str, candidates: int, sigma: float) -> None:
    scope = load_mvp_scope(scope_path)
    _WORKER["env"] = DexAcquireMvpEnv(scope, PinchPriorTable.load(prior_path))
    _WORKER["spec"] = ExpertSearchSpec(candidates=candidates, sigma=sigma)


def _search(job: tuple[int, str]) -> tuple[dict[str, Any], np.ndarray | None, np.ndarray | None]:
    seed, split = job
    episode, row = search_expert_episode(_WORKER["env"], seed, split, _WORKER["spec"])  # type: ignore[arg-type]
    if episode is None:
        return row, None, None
    return row, episode.observations, episode.actions


def collect(split: str, seeds: list[int], workers: int, out: Path, args: argparse.Namespace) -> DemonstrationSet:
    jobs = [(seed, split) for seed in seeds]
    if workers <= 1:
        _init(args.scope and str(args.scope), str(args.prior), args.candidates, args.sigma)
        outputs = [_search(job) for job in jobs]
    else:
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_init,
            initargs=(args.scope and str(args.scope), str(args.prior), args.candidates, args.sigma),
        ) as pool:
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
    parser.add_argument("--candidates", type=int, default=12)
    parser.add_argument("--sigma", type=float, default=0.35)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = parser.parse_args(argv)

    scope = load_mvp_scope(args.scope)
    started = time.time()
    summaries: dict[str, Any] = {}
    for split, count in (("train", args.train_episodes), ("dev", args.dev_episodes)):
        seeds = [scope.episode_seed(split, index) for index in range(count)]  # type: ignore[arg-type]
        dataset = collect(split, seeds, args.workers, args.out, args)
        summaries[split] = dataset.summary()
        print(f"[{split}] {json.dumps(summaries[split], sort_keys=True)}")

    index = {
        "scope_hash": scope.content_hash(),
        "prior_hash": PinchPriorTable.load(args.prior).content_hash(),
        "candidates": args.candidates,
        "sigma": args.sigma,
        "elapsed_s": round(time.time() - started, 1),
        "splits": summaries,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "index.json").write_text(json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {args.out / 'index.json'} in {index['elapsed_s']}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
