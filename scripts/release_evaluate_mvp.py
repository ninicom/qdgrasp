#!/usr/bin/env python3
"""MR-05: the locked evaluation, run once, with everything the gate needs.

``ROADMAP-MVP-RELEASE-001`` §5 MR-05 is a single ordered pass and this script is
that pass: the controller prior, the BC rollback and the final candidate over
tiers A to D on the locked seeds; an ablation that runs the exact candidate with
its learned residual switched off; and a contribution report whose paired
intervals are recomputed from the raw ledgers with the estimator, resample count
and seed the scope froze before any of this was trained.

It writes the artifacts the release gate reads and then stops.  It does not
decide anything -- ``scripts/check_mvp.py --release`` does that, from the files
this leaves behind, and it recomputes every number rather than believing the
summaries here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from qdgrasp.mvp.challenge import load_challenge_domain
from qdgrasp.mvp.config import load_mvp_scope
from qdgrasp.mvp.contracts import ABLATION_REPORT_SCHEMA, CONTRIBUTION_REPORT_SCHEMA
from qdgrasp.mvp.env import DexAcquireMvpEnv
from qdgrasp.mvp.evaluate import evaluate_candidate, format_report, paired_uplift, write_report
from qdgrasp.mvp.policy import load_policy, verify_reload_probe
from qdgrasp.mvp.prior import DEFAULT_PRIOR_PATH, PinchPriorTable

_WORKER: dict[str, Any] = {}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_ledger(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _outcomes(report: dict[str, Any], tier: str, root: Path) -> tuple[list[int], list[bool]]:
    """Seeds and successes of one tier, read back out of its raw ledger."""

    for entry in report["tiers"]:
        if entry["tier"] != tier:
            continue
        rows = _read_ledger(root / entry["ledger_path"] if not Path(entry["ledger_path"]).is_absolute()
                            else Path(entry["ledger_path"]))
        return [int(row["setup"]["seed"]) for row in rows], [row["success"] is True for row in rows]
    raise KeyError(f"report has no tier {tier}")


# -- the ablation and the residual statistics -----------------------------


def _init(scope_path: str, prior_path: str, domain_path: str, checkpoint: str) -> None:
    import torch

    torch.set_num_threads(1)
    scope = load_mvp_scope(scope_path)
    prior = PinchPriorTable.load(prior_path)
    _WORKER["env"] = DexAcquireMvpEnv(scope, prior, challenge=load_challenge_domain(domain_path, scope))
    _WORKER["policy"] = load_policy(checkpoint)
    _WORKER["scope"] = scope


def _measured_episode(job: tuple[int, bool]) -> dict[str, Any]:
    """One episode of the contribution tier, with the residual on or off.

    ``ablate`` still calls the policy and still runs the whole trajectory
    contract; only the residual it produces is replaced by zero.  Skipping the
    call would measure a different program, and the question is what the
    learned residual contributes to *this* one.
    """

    seed, ablate = job
    env, policy, scope = _WORKER["env"], _WORKER["policy"], _WORKER["scope"]
    zeros = np.zeros(scope.action.dimension)
    issued: list[np.ndarray] = []
    observation = env.reset(seed, "eval_d")
    while not env.done:
        action = np.asarray(policy(observation), dtype=np.float64)
        issued.append(action)
        observation, _, _, _ = env.step(zeros if ablate else action)
    result = env.result
    assert result is not None
    actions = np.stack(issued)
    return {
        "success": bool(result.success),
        "safety_violation": bool(result.safety_violation),
        "invalid_state": bool(result.invalid_state),
        "mean_magnitude": float(np.abs(actions).mean()),
        "saturated": float(np.mean(np.abs(actions) >= 0.999)),
    }


def _run_tier_d(seeds: list[int], ablate: bool, initargs: tuple[str, ...], workers: int) -> list[dict[str, Any]]:
    jobs = [(seed, ablate) for seed in seeds]
    if workers <= 1:
        _init(*initargs)
        return [_measured_episode(job) for job in jobs]
    with ProcessPoolExecutor(max_workers=workers, initializer=_init, initargs=initargs) as pool:
        return list(pool.map(_measured_episode, jobs, chunksize=4))


def _evaluate(
    label: str,
    checkpoint: Path | None,
    *,
    scope: Any,
    prior: Any,
    args: argparse.Namespace,
) -> dict[str, Any]:
    ledger_dir = args.out / label
    reload_mismatch = 0
    extra: dict[str, Any] = {}
    if checkpoint is not None:
        extra["checkpoint_sha256"] = _sha256(checkpoint)
        if not verify_reload_probe(checkpoint):
            reload_mismatch = 1

    # A locked tier is run once.  If this arm already has a report for this
    # exact checkpoint, that run happened, and re-rolling it because a later
    # step crashed would be a second draw on the same dice.
    existing = args.out / f"{label}.json"
    if existing.is_file():
        stored = json.loads(existing.read_text(encoding="utf-8"))
        same = stored.get("checkpoint_sha256") == extra.get("checkpoint_sha256")
        if same and stored.get("fingerprint") is not None:
            print(f"[{label}] already measured on these locked seeds; reusing {existing}")
            print(format_report(stored))
            return stored
    report = evaluate_candidate(
        scope,
        prior,
        scope_path=str(args.scope),
        prior_path=str(args.prior),
        checkpoint_path=None if checkpoint is None else str(checkpoint),
        challenge_path=str(args.challenge),
        workers=args.workers,
        ledger_dir=ledger_dir,
        reload_mismatch=reload_mismatch,
        label=label,
    )
    report.update(extra)
    write_report(args.out / f"{label}.json", report)
    print(format_report(report))
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", type=Path, default=Path("configs/mvp/dexacquire-mvp-v1.yaml"))
    parser.add_argument("--prior", type=Path, default=DEFAULT_PRIOR_PATH)
    parser.add_argument("--challenge", type=Path, default=Path("configs/mvp/dexacquire-mvp-v1.challenge.json"))
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--bc", type=Path, required=True, help="the retained BC rollback checkpoint")
    parser.add_argument("--runs", type=Path, required=True, help="evidence root; evaluation/ is written under it")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = parser.parse_args(argv)
    args.out = args.runs / "evaluation"

    scope = load_mvp_scope(args.scope)
    prior = PinchPriorTable.load(args.prior)
    if scope.release is None or scope.challenge is None:
        print("FAIL the scope carries no release contract")
        return 2
    criteria = scope.release
    domain = load_challenge_domain(args.challenge, scope)

    # 1-3: the arms, each locked tier run exactly once.  When PPO was refused
    # promotion the candidate *is* the BC rollback, and evaluating one file
    # under two labels would run the locked seeds twice on one checkpoint --
    # which the plan allows once, and which would be a second roll of the same
    # dice even though the result is deterministic.
    prior_report = _evaluate("controller_prior", None, scope=scope, prior=prior, args=args)
    candidate_label = args.candidate.stem
    candidate_report = _evaluate(candidate_label, args.candidate, scope=scope, prior=prior, args=args)
    if args.bc.resolve() == args.candidate.resolve():
        print(f"[bc] the candidate is the retained rollback; reusing evaluation/{candidate_label}.json")
        if candidate_label != "bc":
            write_report(args.out / "bc.json", candidate_report)
    else:
        _evaluate("bc", args.bc, scope=scope, prior=prior, args=args)

    # 4: the ablation, on the contribution tier, through the candidate's own
    # code path with the residual switched off.
    tier_d = criteria.contribution_tier
    seeds = list(scope.locked_seeds(tier_d))
    initargs = (str(args.scope), str(args.prior), str(args.challenge), str(args.candidate))
    print(f"[ablation] running tier {tier_d} with the learned residual disabled")
    disabled = _run_tier_d(seeds, True, initargs, args.workers)
    print("[ablation] measuring the residual the candidate actually issues")
    live = _run_tier_d(seeds, False, initargs, args.workers)

    _, prior_outcomes = _outcomes(prior_report, tier_d, PROJECT_ROOT)
    disabled_outcomes = [row["success"] for row in disabled]
    ablation = {
        "schema": ABLATION_REPORT_SCHEMA,
        "tier": tier_d,
        "candidate_sha256": _sha256(args.candidate),
        "challenge_domain_sha256": _sha256(args.challenge),
        "residual_disabled": True,
        "episodes": len(seeds),
        "disabled_successes": sum(disabled_outcomes),
        "live_successes": sum(row["success"] for row in live),
        "paired_vs_prior": paired_uplift(
            prior_outcomes,
            disabled_outcomes,
            resamples=criteria.paired_resamples,
            seed=criteria.paired_seed,
            confidence=criteria.paired_confidence,
        ),
        "residual_statistics": {
            "mean_magnitude": float(np.mean([row["mean_magnitude"] for row in live])),
            "saturation_rate": float(np.mean([row["saturated"] for row in live])),
        },
        "safety_violation": sum(row["safety_violation"] for row in live + disabled),
        "invalid_state": sum(row["invalid_state"] for row in live + disabled),
    }
    write_report(args.out / "ablation.json", ablation)
    print(
        f"[ablation] residual off: {ablation['disabled_successes']}/{len(seeds)} "
        f"({ablation['paired_vs_prior']['uplift_pp']:+.2f} pp vs prior); "
        f"residual on: {ablation['live_successes']}/{len(seeds)}; "
        f"magnitude {ablation['residual_statistics']['mean_magnitude']:.4f}, "
        f"saturation {ablation['residual_statistics']['saturation_rate']:.4f}"
    )

    # 5: paired comparisons, recomputed from the raw ledgers.
    paired: dict[str, Any] = {}
    for spec in sorted(scope.eval_tiers, key=lambda item: item.tier):
        base_seeds, base = _outcomes(prior_report, spec.tier, PROJECT_ROOT)
        arm_seeds, arm = _outcomes(candidate_report, spec.tier, PROJECT_ROOT)
        if base_seeds != arm_seeds:
            print(f"FAIL tier {spec.tier}: the two arms did not run the same seeds")
            return 1
        paired[spec.tier] = paired_uplift(
            base,
            arm,
            resamples=criteria.paired_resamples,
            seed=criteria.paired_seed,
            confidence=criteria.paired_confidence,
        )
        entry = paired[spec.tier]
        print(
            f"[paired] tier {spec.tier}: {entry['candidate_successes']}/{entry['episodes']} vs "
            f"{entry['prior_successes']}/{entry['episodes']}  {entry['uplift_pp']:+.2f} pp "
            f"[{entry['ci_lower_pp']:+.2f}, {entry['ci_upper_pp']:+.2f}]"
        )

    write_report(
        args.runs / "contribution.json",
        {
            "schema": CONTRIBUTION_REPORT_SCHEMA,
            "scope_hash": scope.content_hash(),
            "eval_manifest_hash": scope.eval_manifest_hash(),
            "challenge_domain_sha256": _sha256(args.challenge),
            "challenge_domain_content_hash": domain.content_hash(),
            "candidate_sha256": _sha256(args.candidate),
            "prior_report": "evaluation/controller_prior.json",
            "candidate_report": f"evaluation/{candidate_label}.json",
            "bc_report": "evaluation/bc.json",
            "paired": paired,
        },
    )
    print(f"wrote {args.runs / 'contribution.json'}")
    print("locked evaluation complete; run scripts/check_mvp.py --release for the verdict")
    return 0


if __name__ == "__main__":
    sys.exit(main())
