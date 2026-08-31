#!/usr/bin/env python3
"""MVP-07: the closure gate for the temporary Grasp Policy MVP.

Returns ``0`` only when every work package in ``ROADMAP-MVP-001`` §6 has a real
artifact and §7 passes after a checkpoint reload.  It is deliberately willing to
fail: an MVP that does not reach the gate is ``blocked_with_evidence``, which is
a state this checker can report and a state the plan explicitly allows.

Nothing here recomputes physics.  It reads the artifacts the pipeline wrote,
checks they agree with each other and with the locked scope, and applies the
plan's thresholds to numbers it did not produce.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from qdgrasp.mvp.config import load_mvp_scope
from qdgrasp.mvp.prior import PinchPriorTable

#: MVP-03's own gate.
BC_DEV_SUCCESS_FLOOR = 0.75
#: MVP-04: PPO may not cost more than two points against the BC baseline.
PPO_REGRESSION_TOLERANCE = 0.02
#: MVP-02: the controller prior must clear this on the canonical fixture set.
CONTROLLER_CANONICAL_FLOOR = 0.90


@dataclass
class Check:
    package: str
    name: str
    passed: bool
    detail: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def run_checks(root: Path, runs: Path) -> list[Check]:
    checks: list[Check] = []

    def record(package: str, name: str, passed: bool, detail: str) -> None:
        checks.append(Check(package, name, passed, detail))

    # -- MVP-00: locked scope and immutable eval manifest ------------------
    scope_path = root / "configs/mvp/dexacquire-mvp-v0.yaml"
    manifest_path = root / "configs/mvp/dexacquire-mvp-v0.eval-manifest.json"
    try:
        scope = load_mvp_scope(scope_path)
    except Exception as error:  # noqa: BLE001 - any failure here is the finding
        record("MVP-00", "scope_loads", False, f"{error}")
        return checks
    record("MVP-00", "scope_loads", True, f"scope_hash={scope.content_hash()}")
    record(
        "MVP-00",
        "release_class_is_experimental",
        scope.release_class == "experimental_non_release",
        scope.release_class,
    )
    stored_manifest = _load_json(manifest_path)
    record(
        "MVP-00",
        "eval_manifest_immutable",
        stored_manifest == scope.eval_manifest(),
        f"eval_manifest_hash={scope.eval_manifest_hash()}",
    )

    # -- MVP-01: prior artifact fitted on the train widths only ------------
    prior_path = root / "configs/mvp/leap-pinch-prior-v0.json"
    prior: PinchPriorTable | None = None
    if prior_path.is_file():
        prior = PinchPriorTable.load(prior_path)
        fitted = sorted(knot.half_width for knot in prior.knots)
        train = sorted(variant.half_width for variant in scope.train_variants)
        held = {variant.half_width for variant in scope.heldout_variants}
        record("MVP-01", "prior_fits_train_widths", fitted == train, f"{fitted}")
        record("MVP-01", "prior_excludes_heldout_widths", not held & set(fitted), f"held_out={sorted(held)}")
    else:
        record("MVP-01", "prior_artifact_present", False, str(prior_path))

    # -- MVP-02: demonstrations and their generator ledger -----------------
    demos = runs / "demonstrations"
    index = _load_json(demos / "index.json")
    record("MVP-02", "demonstration_index_present", index is not None, str(demos / "index.json"))
    if index is not None:
        record(
            "MVP-02",
            "demonstrations_match_locked_scope",
            index.get("scope_hash") == scope.content_hash(),
            f"index scope_hash={index.get('scope_hash')}",
        )
        for split in ("train", "dev"):
            summary = index.get("splits", {}).get(split, {})
            accepted = int(summary.get("episodes_accepted", 0))
            record("MVP-02", f"{split}_demonstrations_accepted", accepted > 0, f"{accepted} episodes")
            ledger = demos / split / "ledger.jsonl"
            record("MVP-02", f"{split}_generator_ledger_present", ledger.is_file(), str(ledger))
        # A minimum-intervention expert labels most episodes with the zero
        # residual, so a large non-zero fraction is not the property to demand.
        # What must be true is that the search actually rescued episodes the
        # prior alone failed -- otherwise the expert is the prior, and the
        # demonstrations teach nothing the controller did not already do.
        rescued = int(index.get("splits", {}).get("train", {}).get("search_rescued", 0))
        residual = float(index.get("splits", {}).get("train", {}).get("non_zero_residual_fraction", 0.0))
        record(
            "MVP-02",
            "expert_improves_on_the_prior",
            rescued > 0,
            f"search_rescued={rescued}, non_zero_residual_fraction={residual:.3f}",
        )

    # -- MVP-03/04: training report ----------------------------------------
    training = _load_json(runs / "policy/training-report.json")
    record("MVP-03", "training_report_present", training is not None, str(runs / "policy/training-report.json"))
    candidate_path: Path | None = None
    if training is not None:
        bc = training.get("bc", {})
        record("MVP-03", "bc_reload_parity", bool(bc.get("reload_parity")), f"{bc.get('reload_parity')}")
        bc_rate = float(bc.get("dev", {}).get("success_rate", 0.0))
        record("MVP-03", "bc_dev_success", bc_rate >= BC_DEV_SUCCESS_FLOOR, f"{bc_rate:.3f} >= {BC_DEV_SUCCESS_FLOOR}")
        record(
            "MVP-03",
            "fingerprint_matches_scope",
            training.get("fingerprint", {}).get("scope_hash") == scope.content_hash(),
            f"{training.get('fingerprint', {}).get('scope_hash')}",
        )
        ppo = training.get("ppo")
        if ppo is None:
            record("MVP-04", "ppo_confirmation_run", False, "no PPO stage in the training report")
        else:
            ppo_rate = float(ppo.get("dev", {}).get("success_rate", 0.0))
            promoted = bool(ppo.get("promoted"))
            record(
                "MVP-04",
                "ppo_promotion_rule_respected",
                (ppo_rate >= bc_rate - PPO_REGRESSION_TOLERANCE) == promoted,
                f"ppo_dev={ppo_rate:.3f} bc_dev={bc_rate:.3f} promoted={promoted}",
            )
            record(
                "MVP-04",
                "ppo_introduced_no_safety_violation",
                int(ppo.get("dev", {}).get("safety_violation", 1)) == 0,
                f"safety_violation={ppo.get('dev', {}).get('safety_violation')}",
            )
            record("MVP-04", "bc_rollback_retained", Path(bc.get("checkpoint", "")).name == "bc.pt", "bc.pt")
        candidate = training.get("candidate")
        if candidate:
            candidate_path = Path(candidate)
            record("MVP-04", "candidate_checkpoint_exists", candidate_path.is_file(), str(candidate_path))

    # -- MVP-02 gate: the controller prior on the canonical fixtures --------
    prior_report = _load_json(runs / "evaluation/controller_prior.json")
    record("MVP-02", "controller_prior_measured", prior_report is not None, "evaluation/controller_prior.json")
    if prior_report is not None:
        tier_a = next((tier for tier in prior_report["tiers"] if tier["tier"] == "A"), None)
        rate = float(tier_a["success_rate"]) if tier_a else 0.0
        record(
            "MVP-02",
            "controller_prior_canonical_floor",
            rate >= CONTROLLER_CANONICAL_FLOOR,
            f"tier A {rate:.3f} >= {CONTROLLER_CANONICAL_FLOOR}",
        )

    # -- MVP-05: locked evaluation of the candidate ------------------------
    candidate_label = candidate_path.stem if candidate_path is not None else "ppo"
    evaluation = _load_json(runs / f"evaluation/{candidate_label}.json")
    record("MVP-05", "locked_evaluation_present", evaluation is not None, f"evaluation/{candidate_label}.json")
    if evaluation is not None:
        record(
            "MVP-05",
            "evaluation_matches_locked_manifest",
            evaluation.get("fingerprint", {}).get("eval_manifest_hash") == scope.eval_manifest_hash(),
            f"{evaluation.get('fingerprint', {}).get('eval_manifest_hash')}",
        )
        if candidate_path is not None and candidate_path.is_file():
            record(
                "MVP-05",
                "evaluated_checkpoint_is_the_candidate",
                evaluation.get("checkpoint_sha256") == _sha256(candidate_path),
                "checkpoint hash recorded in the evaluation report",
            )
        for tier in evaluation.get("tiers", []):
            name = tier["tier"]
            record(
                "MVP-05",
                f"tier_{name.lower()}_gate",
                bool(tier["passed"]),
                f"{tier['successes']}/{tier['episodes']} = {tier['success_rate']:.3f} "
                f"(wilson {tier['wilson_lower']:.3f}) buckets={tier['failure_buckets']}",
            )
            record(
                "MVP-05",
                f"tier_{name.lower()}_zero_invalid_and_safe",
                tier["invalid_state"] == 0
                and tier["safety_violation"] == 0
                and tier["checkpoint_reload_mismatch"] == 0,
                f"invalid={tier['invalid_state']} safety={tier['safety_violation']} "
                f"reload={tier['checkpoint_reload_mismatch']}",
            )
            record("MVP-05", f"tier_{name.lower()}_ledger", bool(tier.get("ledger_path")), str(tier.get("ledger_path")))

    # -- MVP-06/07: cloud handoff and the model card -----------------------
    runner = root / "notebooks/mvp_grasp_policy.ipynb"
    record("MVP-06", "cloud_runner_present", runner.is_file(), str(runner))
    card = root / "docs/reports/MVP-GRASP-POLICY-MODEL-CARD.md"
    record("MVP-07", "model_card_present", card.is_file(), str(card))
    if card.is_file():
        text = card.read_text(encoding="utf-8")
        record(
            "MVP-07",
            "model_card_declares_experimental",
            "experimental_non_release" in text,
            "release_class stated in the card",
        )
        record(
            "MVP-07",
            "model_card_states_limitations",
            any(marker in text.casefold() for marker in ("giới hạn", "limitations")),
            "limitations section",
        )
    return checks


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--runs", type=Path, default=Path("runs/mvp"))
    parser.add_argument("--json", type=Path, default=None, help="also write the result as JSON")
    args = parser.parse_args(argv)

    checks = run_checks(args.root, args.runs)
    failed = [check for check in checks if not check.passed]
    for check in checks:
        print(f"{'PASS' if check.passed else 'FAIL'}  [{check.package}] {check.name}: {check.detail}")
    print(f"\n{len(checks) - len(failed)}/{len(checks)} checks passed")
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(
                {
                    "schema": "qdgrasp/mvp-closure/v0",
                    "passed": not failed,
                    "checks": [
                        {"package": c.package, "name": c.name, "passed": c.passed, "detail": c.detail} for c in checks
                    ],
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    if failed:
        print("MVP status: blocked_with_evidence")
        return 1
    print("MVP status: complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
