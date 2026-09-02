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
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from qdgrasp.mvp.config import load_mvp_scope
from qdgrasp.mvp.contracts import TRAINING_REPORT_SCHEMA
from qdgrasp.mvp.env import environment_fingerprint
from qdgrasp.mvp.evaluate import EVAL_REPORT_SCHEMA, wilson_lower_bound
from qdgrasp.mvp.policy import ACTION_DISTRIBUTION, load_checkpoint
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
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return document if isinstance(document, dict) else None


def _resolve_checkpoint(reference: object, *, repository_root: Path, evidence_root: Path) -> Path:
    """Resolve one declared checkpoint without escaping or basename fallback.

    Producers historically wrote repository-relative paths (``runs/mvp/...``),
    while a published evidence directory naturally uses paths relative to that
    directory (``policy/...``).  Both forms are accepted only when the exact
    resolved file stays below ``evidence_root``.  An outside path is never
    replaced with a same-named local file.
    """

    if not isinstance(reference, str) or not reference.strip():
        raise ValueError("checkpoint reference must be a non-empty string")
    declared = Path(reference)
    if ".." in declared.parts:
        raise ValueError(f"checkpoint {reference!r} contains a parent traversal")
    evidence = evidence_root.resolve()
    if declared.is_absolute():
        candidates = [declared]
    else:
        repository_candidate = (repository_root / declared).resolve()
        candidates = [repository_candidate] if repository_candidate.is_relative_to(evidence) else []
        candidates.append(evidence / declared)
        if declared.parts[:2] == ("runs", "mvp"):
            candidates.append(evidence.joinpath(*declared.parts[2:]))
    failures: list[str] = []
    for candidate in candidates:
        try:
            resolved = candidate.resolve(strict=True)
        except OSError as error:
            failures.append(str(error))
            continue
        if not resolved.is_relative_to(evidence):
            failures.append(f"{resolved} escapes {evidence}")
            continue
        if resolved.is_file():
            return resolved
        failures.append(f"{resolved} is not a regular file")
    raise ValueError(f"checkpoint {reference!r} is not an exact evidence artifact: {failures}")


def _validate_evidence_manifest(runs: Path) -> tuple[bool, str]:
    manifest_path = runs / "MANIFEST.json"
    manifest = _load_json(manifest_path)
    if manifest is None:
        return False, f"missing or invalid {manifest_path}"
    if manifest.get("schema") != "qdgrasp/mvp-evidence-manifest/v0":
        return False, f"unsupported manifest schema {manifest.get('schema')!r}"
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list):
        return False, "manifest artifacts must be a list"
    declared: set[str] = set()
    errors: list[str] = []
    for entry in artifacts:
        if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
            errors.append("malformed artifact entry")
            continue
        name = entry["path"]
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts or name in declared:
            errors.append(f"unsafe or duplicate artifact path {name!r}")
            continue
        declared.add(name)
        candidate = (runs / relative).resolve()
        if not candidate.is_relative_to(runs) or not candidate.is_file():
            errors.append(f"missing artifact {name!r}")
            continue
        if entry.get("bytes") != candidate.stat().st_size or entry.get("sha256") != _sha256(candidate):
            errors.append(f"content mismatch for {name!r}")
    actual = {
        path.relative_to(runs).as_posix()
        for path in runs.rglob("*")
        if path.is_file() and path.name != "MANIFEST.json"
    }
    if declared != actual:
        errors.append(f"file set mismatch: undeclared={sorted(actual - declared)}, missing={sorted(declared - actual)}")
    return not errors, "; ".join(errors) if errors else f"{len(declared)} artifacts verified"


def run_checks(root: Path, runs: Path) -> list[Check]:
    root = root.resolve()
    runs = (runs if runs.is_absolute() else root / runs).resolve()
    checks: list[Check] = []

    def record(package: str, name: str, passed: bool, detail: str) -> None:
        checks.append(Check(package, name, passed, detail))

    # Raw ``runs/mvp`` is checked before publication.  Every other artifact
    # directory is a published evidence set and must carry an exact content
    # manifest; otherwise a report can be edited after the checker ran.
    manifest_path = runs / "MANIFEST.json"
    if runs != (root / "runs/mvp").resolve() or manifest_path.exists():
        manifest_ok, manifest_detail = _validate_evidence_manifest(runs)
        record("MVP-07", "evidence_manifest_integrity", manifest_ok, manifest_detail)

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

    expected_fingerprint = environment_fingerprint(scope, prior) if prior is not None else None

    loaded: dict[Path, tuple[dict[str, Any] | None, str | None]] = {}

    def validate_checkpoint(
        package: str,
        label: str,
        reference: object,
    ) -> tuple[Path | None, dict[str, Any] | None]:
        try:
            path = _resolve_checkpoint(reference, repository_root=root, evidence_root=runs)
        except ValueError as error:
            record(package, f"{label}_checkpoint_is_contained", False, str(error))
            return None, None
        record(package, f"{label}_checkpoint_is_contained", True, str(path))
        if path not in loaded:
            try:
                loaded[path] = (load_checkpoint(path), None)
            except Exception as error:  # noqa: BLE001 - the gate records every safe-loader refusal
                loaded[path] = (None, f"{type(error).__name__}: {error}")
        payload, error = loaded[path]
        record(
            package,
            f"{label}_checkpoint_loads",
            payload is not None,
            f"schema={payload.get('schema')!r}" if payload is not None else str(error),
        )
        if payload is not None:
            record(
                package,
                f"{label}_checkpoint_matches_scope",
                expected_fingerprint is not None and payload.get("fingerprint") == expected_fingerprint,
                f"stored={payload.get('fingerprint')}",
            )
        return path, payload

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
    candidate_payload: dict[str, Any] | None = None
    if training is not None:
        record(
            "MVP-03",
            "training_report_schema",
            training.get("schema") == TRAINING_REPORT_SCHEMA,
            f"stored={training.get('schema')!r}, current={TRAINING_REPORT_SCHEMA!r}",
        )
        record(
            "MVP-03",
            "training_action_distribution",
            training.get("action_distribution") == ACTION_DISTRIBUTION,
            f"stored={training.get('action_distribution')!r}, current={ACTION_DISTRIBUTION!r}",
        )
        bc = training.get("bc") if isinstance(training.get("bc"), dict) else {}
        record("MVP-03", "bc_reload_parity", bool(bc.get("reload_parity")), f"{bc.get('reload_parity')}")
        bc_rate = float(bc.get("dev", {}).get("success_rate", 0.0))
        record("MVP-03", "bc_dev_success", bc_rate >= BC_DEV_SUCCESS_FLOOR, f"{bc_rate:.3f} >= {BC_DEV_SUCCESS_FLOOR}")
        record(
            "MVP-03",
            "fingerprint_matches_scope",
            expected_fingerprint is not None and training.get("fingerprint") == expected_fingerprint,
            f"stored={training.get('fingerprint')}",
        )
        bc_path, bc_payload = validate_checkpoint("MVP-03", "bc", bc.get("checkpoint"))
        record(
            "MVP-04",
            "bc_rollback_retained",
            bc_path is not None and bc_payload is not None and bc_path.name == "bc.pt",
            str(bc_path or bc.get("checkpoint")),
        )

        ppo = training.get("ppo") if isinstance(training.get("ppo"), dict) else None
        promoted = False
        ppo_path: Path | None = None
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
            ppo_path, _ = validate_checkpoint("MVP-04", "ppo", ppo.get("checkpoint"))

        candidate_path, candidate_payload = validate_checkpoint("MVP-04", "candidate", training.get("candidate"))
        record(
            "MVP-04",
            "candidate_checkpoint_exists",
            candidate_path is not None,
            str(candidate_path or training.get("candidate")),
        )
        expected_candidate = ppo_path if promoted else bc_path
        record(
            "MVP-04",
            "candidate_matches_promotion",
            candidate_path is not None and expected_candidate is not None and candidate_path == expected_candidate,
            f"candidate={candidate_path}, selected={expected_candidate}",
        )
        record(
            "MVP-04",
            "training_lineage_matches_candidate",
            candidate_payload is not None
            and isinstance(training.get("lineage"), dict)
            and training.get("lineage") == candidate_payload.get("lineage"),
            f"report={training.get('lineage')}",
        )

    # -- MVP-02 gate: the controller prior on the canonical fixtures --------
    prior_report = _load_json(runs / "evaluation/controller_prior.json")
    record("MVP-02", "controller_prior_measured", prior_report is not None, "evaluation/controller_prior.json")
    if prior_report is not None:
        record(
            "MVP-02",
            "controller_prior_report_schema",
            prior_report.get("schema") == EVAL_REPORT_SCHEMA,
            f"stored={prior_report.get('schema')!r}, current={EVAL_REPORT_SCHEMA!r}",
        )
        expected_prior_verification = {
            "stored": None,
            "effective": expected_fingerprint,
            "verdict": "not_applicable",
        }
        record(
            "MVP-02",
            "controller_prior_fingerprint_verdict",
            expected_fingerprint is not None
            and prior_report.get("fingerprint") == expected_fingerprint
            and prior_report.get("checkpoint_fingerprint") == expected_prior_verification,
            f"checkpoint_fingerprint={prior_report.get('checkpoint_fingerprint')}",
        )
        prior_tiers = prior_report.get("tiers") if isinstance(prior_report.get("tiers"), list) else []
        tier_a = next(
            (tier for tier in prior_tiers if isinstance(tier, dict) and tier.get("tier") == "A"),
            None,
        )
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
            "evaluation_report_schema",
            evaluation.get("schema") == EVAL_REPORT_SCHEMA,
            f"stored={evaluation.get('schema')!r}, current={EVAL_REPORT_SCHEMA!r}",
        )
        record(
            "MVP-05",
            "evaluation_matches_locked_manifest",
            expected_fingerprint is not None and evaluation.get("fingerprint") == expected_fingerprint,
            f"stored={evaluation.get('fingerprint')}",
        )
        expected_verification = {
            "stored": candidate_payload.get("fingerprint") if candidate_payload is not None else None,
            "effective": expected_fingerprint,
            "verdict": "match",
        }
        record(
            "MVP-05",
            "checkpoint_fingerprint_verdict",
            candidate_payload is not None
            and expected_fingerprint is not None
            and evaluation.get("checkpoint_fingerprint") == expected_verification,
            f"stored={evaluation.get('checkpoint_fingerprint')}",
        )
        evaluated_path, _ = validate_checkpoint("MVP-05", "evaluated", evaluation.get("checkpoint"))
        record(
            "MVP-05",
            "evaluated_checkpoint_is_the_candidate",
            candidate_path is not None
            and evaluated_path == candidate_path
            and evaluation.get("checkpoint_sha256") == _sha256(candidate_path),
            f"declared={evaluated_path}, candidate={candidate_path}",
        )
        tiers = evaluation.get("tiers") if isinstance(evaluation.get("tiers"), list) else []
        tier_names = [tier.get("tier") for tier in tiers if isinstance(tier, dict)]
        record(
            "MVP-05",
            "evaluation_tiers_complete",
            len(tiers) == 3 and set(tier_names) == {"A", "B", "C"},
            f"tiers={tier_names}",
        )
        derived_passes: list[bool] = []
        for tier in tiers:
            if not isinstance(tier, dict):
                continue
            name = str(tier.get("tier", "unknown"))
            try:
                spec = scope.tier(name)  # type: ignore[arg-type]
                episodes = int(tier["episodes"])
                successes = int(tier["successes"])
                invalid = int(tier["invalid_state"])
                safety = int(tier["safety_violation"])
                reload_mismatch = int(tier["checkpoint_reload_mismatch"])
                rate = successes / episodes if episodes else 0.0
                lower = wilson_lower_bound(successes, episodes)
                derived_pass = (
                    episodes == spec.episodes
                    and 0 <= successes <= episodes
                    and rate >= spec.min_success_rate
                    and (spec.min_wilson_lower_bound is None or lower >= spec.min_wilson_lower_bound)
                    and invalid == 0
                    and safety == 0
                    and reload_mismatch == 0
                )
                buckets = tier.get("failure_buckets")
                contract_ok = (
                    math.isclose(float(tier["success_rate"]), rate, abs_tol=1e-12)
                    and math.isclose(float(tier["wilson_lower"]), lower, abs_tol=1e-12)
                    and tier.get("min_success_rate") == spec.min_success_rate
                    and tier.get("min_wilson_lower_bound") == spec.min_wilson_lower_bound
                    and isinstance(buckets, dict)
                    and sum(int(value) for value in buckets.values()) == episodes - successes
                    and bool(tier.get("passed")) == derived_pass
                )
                contract_detail = (
                    f"episodes={episodes}/{spec.episodes}, successes={successes}, "
                    f"rate={rate:.3f}, wilson={lower:.3f}, derived_pass={derived_pass}"
                )
            except (KeyError, TypeError, ValueError) as error:
                contract_ok = False
                derived_pass = False
                contract_detail = f"malformed tier: {type(error).__name__}: {error}"
            derived_passes.append(derived_pass)
            record("MVP-05", f"tier_{name.lower()}_contract", contract_ok, contract_detail)
            record(
                "MVP-05",
                f"tier_{name.lower()}_gate",
                derived_pass and bool(tier.get("passed")),
                contract_detail,
            )
            record(
                "MVP-05",
                f"tier_{name.lower()}_zero_invalid_and_safe",
                tier.get("invalid_state") == 0
                and tier.get("safety_violation") == 0
                and tier.get("checkpoint_reload_mismatch") == 0,
                f"invalid={tier.get('invalid_state')} safety={tier.get('safety_violation')} "
                f"reload={tier.get('checkpoint_reload_mismatch')}",
            )
            try:
                ledger = _resolve_checkpoint(tier.get("ledger_path"), repository_root=root, evidence_root=runs)
                ledger_rows = sum(1 for line in ledger.read_text(encoding="utf-8").splitlines() if line.strip())
                ledger_ok = ledger_rows == int(tier["episodes"])
                ledger_detail = f"{ledger}: {ledger_rows} rows"
            except (OSError, UnicodeError, ValueError, TypeError, KeyError) as error:
                ledger_ok = False
                ledger_detail = f"{type(error).__name__}: {error}"
            record("MVP-05", f"tier_{name.lower()}_ledger", ledger_ok, ledger_detail)
        record(
            "MVP-05",
            "evaluation_all_tiers_verdict",
            len(derived_passes) == 3
            and all(derived_passes)
            and evaluation.get("all_tiers_passed") is True,
            f"stored={evaluation.get('all_tiers_passed')!r}, derived={derived_passes}",
        )

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
