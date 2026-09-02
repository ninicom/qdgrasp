#!/usr/bin/env python3
"""The closure gate for the Grasp Policy MVP, in two distinct modes.

``--experimental`` (the default) is ``MVP-07``: it returns ``0`` only when every
work package in ``ROADMAP-MVP-001`` §6 has a real artifact and §7 passes after a
checkpoint reload.  It is deliberately willing to fail: an MVP that does not
reach the gate is ``blocked_with_evidence``, which is a state this checker can
report and a state the plan explicitly allows.  What it is *not* is a release
gate, and it says so in its verdict -- ``ROADMAP-MVP-RELEASE-001`` §5 MR-02
requires the two to be separable by a machine, not by a reader's good faith.

``--release`` is that release gate.  It runs against scope v1, whose
``release_class`` is ``release_candidate``, and it adds everything the
experimental gate has no opinion about: the challenge tier, the paired
comparison against the controller prior recomputed from the raw ledgers, the
ablation that makes the learned residual prove it is the thing doing the work,
and the safety budget as a bound rather than a habit.  A v0 artifact set cannot
reach it, and passing the experimental gate does not imply passing this one.

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
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from qdgrasp.mvp.challenge import load_challenge_domain
from qdgrasp.mvp.config import (
    EXPERIMENTAL_RELEASE_CLASS,
    RELEASE_CANDIDATE_CLASS,
    MvpScopeConfig,
    load_mvp_scope,
)
from qdgrasp.mvp.contracts import (
    ABLATION_REPORT_SCHEMA,
    CONTRIBUTION_REPORT_SCHEMA,
    TRAINING_REPORT_SCHEMA,
)
from qdgrasp.mvp.env import environment_fingerprint
from qdgrasp.mvp.evaluate import EVAL_REPORT_SCHEMA, paired_uplift, wilson_lower_bound
from qdgrasp.mvp.expert import (
    DEMONSTRATION_INDEX_SCHEMA,
    DEMONSTRATION_MANIFEST_SCHEMA,
    DEMONSTRATION_SCHEMA,
    DemonstrationSet,
)
from qdgrasp.mvp.policy import ACTION_DISTRIBUTION, load_checkpoint
from qdgrasp.mvp.prior import PinchPriorTable

#: The two gates this script can be.  They read different scope documents on
#: purpose: an experimental artifact set must not be able to reach the release
#: verdict by being handed a flag.
GateMode = Literal["experimental", "release"]

SCOPE_DOCUMENT: dict[str, Path] = {
    "experimental": Path("configs/mvp/dexacquire-mvp-v0.yaml"),
    "release": Path("configs/mvp/dexacquire-mvp-v1.yaml"),
}
REQUIRED_RELEASE_CLASS: dict[str, str] = {
    "experimental": EXPERIMENTAL_RELEASE_CLASS,
    "release": RELEASE_CANDIDATE_CLASS,
}

#: Release-mode artifacts, relative to the evidence root.
CONTRIBUTION_REPORT = Path("contribution.json")
ABLATION_REPORT = Path("evaluation/ablation.json")
BC_EVALUATION_REPORT = Path("evaluation/bc.json")
PRIOR_EVALUATION_REPORT = Path("evaluation/controller_prior.json")

#: MVP-03's own gate.
BC_DEV_SUCCESS_FLOOR = 0.75
#: MVP-04: PPO may not cost more than two points against the BC baseline.
#: The release contract does not inherit that tolerance -- ``MR-02`` states it
#: plainly -- so under the release gate the allowance is zero.
PPO_REGRESSION_TOLERANCE = 0.02
PPO_REGRESSION_TOLERANCE_BY_MODE: dict[str, float] = {"experimental": PPO_REGRESSION_TOLERANCE, "release": 0.0}
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


def _canonical_sha256(document: Any) -> str:
    encoded = json.dumps(
        document,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _read_ledger(path: Path) -> list[dict[str, Any]]:
    """Every raw episode row, in the order the evaluator wrote them."""

    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _recompute_tier(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Rebuild a tier's aggregate from raw episodes, ignoring any summary.

    The summary is the thing under suspicion.  A report that says a tier passed
    is a claim; this is the measurement, taken from the rows the evaluator could
    not edit after the checker read them.
    """

    successes = sum(1 for row in rows if row.get("success") is True)
    buckets: Counter[str] = Counter(
        str(row.get("failure_bucket")) for row in rows if row.get("success") is not True
    )
    return {
        "episodes": len(rows),
        "successes": successes,
        "invalid_state": sum(1 for row in rows if row.get("invalid_state") is True),
        "safety_violation": sum(1 for row in rows if row.get("safety_violation") is True),
        "failure_buckets": dict(sorted(buckets.items())),
        "seeds": [row.get("setup", {}).get("seed") for row in rows],
        "outcomes": [row.get("success") is True for row in rows],
    }


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


def _release_checks(
    record: Callable[[str, str, bool, str], None],
    *,
    root: Path,
    runs: Path,
    scope: MvpScopeConfig,
    candidate_path: Path | None,
    candidate_payload: dict[str, Any] | None,
    evaluation: dict[str, Any] | None,
    prior_report: dict[str, Any] | None,
    promoted: bool,
) -> None:
    """MVP-08: everything the experimental gate has no opinion about.

    ``ROADMAP-MVP-RELEASE-001`` §2.2 and §2.3 are the whole content of this
    function: the challenge tier, the paired comparison against the controller
    prior, the ablation, and the safety budget read as a bound out of the
    frozen scope rather than assumed.  Every number is recomputed from the raw
    ledgers -- a tier summary that says a tier passed is the claim under test,
    not evidence for it.
    """

    criteria = scope.release
    challenge = scope.challenge
    if criteria is None or challenge is None:
        record("MVP-08", "release_contract_present", False, "the scope carries no release contract")
        return
    record(
        "MVP-08",
        "release_contract_present",
        True,
        f"contribution_tier={criteria.contribution_tier}, regression_tiers={list(criteria.regression_tiers)}",
    )

    # -- the challenge domain Tier D is drawn from -------------------------
    domain_path = root / challenge.domain_document
    domain_hash = _sha256(domain_path) if domain_path.is_file() else None
    record("MVP-08", "challenge_domain_present", domain_path.is_file(), str(domain_path))
    # Parsed through the model rather than pattern-matched as a dict: the model
    # is what knows that a challenge domain must *narrow* the scope.  A domain
    # declaring only authorised axis names, but reaching outside their locked
    # ranges, is a different world, and a Tier D measured in it could not be
    # compared with the tiers it is supposed to be read beside.
    domain_ok = False
    domain_detail = f"missing: {domain_path}"
    if domain_path.is_file():
        try:
            loaded = load_challenge_domain(domain_path, scope)
            domain_ok = True
            domain_detail = (
                f"{loaded.configuration_id}: axes={sorted(loaded.axes)}, "
                f"variants={[v.variant_id for v in loaded.variants(scope)]}"
            )
        except Exception as error:  # noqa: BLE001 - every refusal is the finding
            domain_detail = f"{type(error).__name__}: {error}"
    record("MVP-08", "challenge_domain_contract", domain_ok, domain_detail)

    # -- the contribution report and what it must be bound to --------------
    contribution = _load_json(runs / CONTRIBUTION_REPORT)
    record("MVP-08", "contribution_report_present", contribution is not None, str(runs / CONTRIBUTION_REPORT))
    candidate_hash = _sha256(candidate_path) if candidate_path is not None else None
    record(
        "MVP-08",
        "contribution_report_contract",
        isinstance(contribution, dict)
        and contribution.get("schema") == CONTRIBUTION_REPORT_SCHEMA
        and contribution.get("scope_hash") == scope.content_hash()
        and contribution.get("eval_manifest_hash") == scope.eval_manifest_hash()
        and contribution.get("challenge_domain_sha256") == domain_hash
        and candidate_hash is not None
        and contribution.get("candidate_sha256") == candidate_hash,
        f"candidate_sha256={contribution.get('candidate_sha256') if isinstance(contribution, dict) else None}",
    )
    paired_documents = contribution.get("paired") if isinstance(contribution, dict) else None
    paired_documents = paired_documents if isinstance(paired_documents, dict) else {}

    def measured(report: dict[str, Any] | None, tier: str) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        """The stored tier summary and the aggregate recomputed from its ledger."""

        if not isinstance(report, dict):
            return None, None
        for entry in report.get("tiers") if isinstance(report.get("tiers"), list) else []:
            if not isinstance(entry, dict) or entry.get("tier") != tier:
                continue
            try:
                ledger = _resolve_checkpoint(entry.get("ledger_path"), repository_root=root, evidence_root=runs)
                return entry, _recompute_tier(_read_ledger(ledger))
            except (OSError, UnicodeError, ValueError, TypeError, json.JSONDecodeError):
                return entry, None
        return None, None

    # -- every tier, recomputed and then compared arm to arm ---------------
    for spec in sorted(scope.eval_tiers, key=lambda item: item.tier):
        name = spec.tier
        lowered = name.lower()
        stored_tier, candidate_rows = measured(evaluation, name)
        _, prior_rows = measured(prior_report, name)

        recomputed_ok = (
            isinstance(stored_tier, dict)
            and candidate_rows is not None
            and candidate_rows["episodes"] == spec.episodes
            and candidate_rows["successes"] == stored_tier.get("successes")
            and candidate_rows["invalid_state"] == stored_tier.get("invalid_state")
            and candidate_rows["safety_violation"] == stored_tier.get("safety_violation")
            and candidate_rows["failure_buckets"] == stored_tier.get("failure_buckets")
        )
        record(
            "MVP-08",
            f"tier_{lowered}_recomputed_from_raw_ledger",
            recomputed_ok,
            f"ledger={candidate_rows if candidate_rows is None else {k: candidate_rows[k] for k in ('episodes', 'successes', 'invalid_state', 'safety_violation')}}, "
            f"summary_successes={stored_tier.get('successes') if isinstance(stored_tier, dict) else None}",
        )
        record(
            "MVP-08",
            f"tier_{lowered}_within_safety_budget",
            candidate_rows is not None
            and isinstance(stored_tier, dict)
            and candidate_rows["safety_violation"] <= criteria.safety.max_safety_violation
            and candidate_rows["invalid_state"] <= criteria.safety.max_invalid_state
            and int(stored_tier.get("checkpoint_reload_mismatch", 1))
            <= criteria.safety.max_checkpoint_reload_mismatch,
            f"safety={None if candidate_rows is None else candidate_rows['safety_violation']}, "
            f"invalid={None if candidate_rows is None else candidate_rows['invalid_state']}",
        )

        seeds_paired = (
            prior_rows is not None
            and candidate_rows is not None
            and prior_rows["seeds"] == candidate_rows["seeds"]
            and len(candidate_rows["seeds"]) == spec.episodes
            and all(isinstance(seed, int) for seed in candidate_rows["seeds"])
        )
        record(
            "MVP-08",
            f"tier_{lowered}_arms_share_the_locked_seeds",
            seeds_paired,
            "prior and candidate ledgers agree seed for seed"
            if seeds_paired
            else "the two arms did not run the same seeds in the same order",
        )
        if not seeds_paired or prior_rows is None or candidate_rows is None:
            continue

        comparison = paired_uplift(
            prior_rows["outcomes"],
            candidate_rows["outcomes"],
            resamples=criteria.paired_resamples,
            seed=criteria.paired_seed,
            confidence=criteria.paired_confidence,
        )
        record(
            "MVP-08",
            f"tier_{lowered}_paired_comparison_recomputes",
            paired_documents.get(name) == comparison,
            f"uplift={comparison['uplift_pp']:.3f}pp ci=[{comparison['ci_lower_pp']:.3f}, "
            f"{comparison['ci_upper_pp']:.3f}] seed={comparison['seed']}",
        )
        if name == criteria.contribution_tier:
            record(
                "MVP-08",
                f"tier_{lowered}_uplift_gate",
                spec.min_uplift_pp is not None
                and spec.min_paired_ci_lower is not None
                and comparison["uplift_pp"] >= spec.min_uplift_pp
                and comparison["ci_lower_pp"] > spec.min_paired_ci_lower,
                f"uplift={comparison['uplift_pp']:.3f}pp >= {spec.min_uplift_pp}pp, "
                f"ci_lower={comparison['ci_lower_pp']:.3f}pp > {spec.min_paired_ci_lower}",
            )
        else:
            record(
                "MVP-08",
                f"tier_{lowered}_no_paired_regression",
                comparison["candidate_successes"] >= comparison["prior_successes"],
                f"candidate={comparison['candidate_successes']} prior={comparison['prior_successes']} "
                f"(prior_only_successes={comparison['prior_only_successes']})",
            )

    # -- the ablation: the residual has to be the thing doing the work -----
    ablation = _load_json(runs / ABLATION_REPORT)
    record("MVP-08", "ablation_report_present", ablation is not None, str(runs / ABLATION_REPORT))
    statistics = ablation.get("residual_statistics") if isinstance(ablation, dict) else None
    disabled = ablation.get("paired_vs_prior") if isinstance(ablation, dict) else None
    record(
        "MVP-08",
        "ablation_report_contract",
        isinstance(ablation, dict)
        and ablation.get("schema") == ABLATION_REPORT_SCHEMA
        and ablation.get("tier") == criteria.contribution_tier
        and candidate_hash is not None
        and ablation.get("candidate_sha256") == candidate_hash
        and ablation.get("residual_disabled") is True
        and criteria.ablation.require_disabled_residual_run,
        f"tier={ablation.get('tier') if isinstance(ablation, dict) else None}, "
        f"candidate_sha256={ablation.get('candidate_sha256') if isinstance(ablation, dict) else None}",
    )
    record(
        "MVP-08",
        "disabling_the_residual_removes_the_uplift",
        isinstance(disabled, dict)
        and isinstance(disabled.get("uplift_pp"), (int, float))
        and float(disabled["uplift_pp"]) <= criteria.ablation.max_disabled_uplift_pp,
        f"uplift_with_residual_off={disabled.get('uplift_pp') if isinstance(disabled, dict) else None}pp "
        f"<= {criteria.ablation.max_disabled_uplift_pp}pp",
    )
    record(
        "MVP-08",
        "residual_has_not_degenerated",
        isinstance(statistics, dict)
        and isinstance(statistics.get("mean_magnitude"), (int, float))
        and isinstance(statistics.get("saturation_rate"), (int, float))
        and float(statistics["mean_magnitude"]) >= criteria.ablation.min_residual_magnitude
        and float(statistics["saturation_rate"]) <= criteria.ablation.max_saturation_rate,
        f"magnitude={statistics.get('mean_magnitude') if isinstance(statistics, dict) else None} "
        f">= {criteria.ablation.min_residual_magnitude}, "
        f"saturation={statistics.get('saturation_rate') if isinstance(statistics, dict) else None} "
        f"<= {criteria.ablation.max_saturation_rate}",
    )

    # -- promotion may not trade a regression tier for the challenge tier --
    bc_evaluation = _load_json(runs / BC_EVALUATION_REPORT)
    record("MVP-08", "bc_rollback_evaluated_on_the_locked_tiers", bc_evaluation is not None, str(BC_EVALUATION_REPORT))
    if promoted:
        losses: list[str] = []
        for tier_name in criteria.regression_tiers:
            _, bc_rows = measured(bc_evaluation, tier_name)
            _, candidate_rows = measured(evaluation, tier_name)
            if bc_rows is None or candidate_rows is None:
                losses.append(f"{tier_name}: missing arm")
            elif candidate_rows["successes"] < bc_rows["successes"]:
                losses.append(f"{tier_name}: {candidate_rows['successes']} < {bc_rows['successes']}")
        record(
            "MVP-08",
            "ppo_is_at_least_bc_on_every_regression_tier",
            not losses,
            "; ".join(losses) if losses else "PPO matches or beats BC on every regression tier",
        )

    # -- the candidate was chosen without ever reading the locked seeds ----
    training = _load_json(runs / "policy/training-report.json")
    selection = training.get("candidate_selection") if isinstance(training, dict) else None
    record(
        "MVP-08",
        "candidate_selected_on_development_evidence_only",
        isinstance(selection, dict)
        and selection.get("evidence") == criteria.candidate_evidence
        and selection.get("ppo_promotion") == criteria.ppo_promotion
        and selection.get("locked_seeds_read") is False,
        f"selection={selection}",
    )
    record(
        "MVP-08",
        "candidate_carries_release_scope_identity",
        candidate_payload is not None
        and candidate_payload.get("fingerprint", {}).get("scope_hash") == scope.content_hash()
        and candidate_payload.get("fingerprint", {}).get("environment_id") == scope.environment_id,
        f"fingerprint={candidate_payload.get('fingerprint') if candidate_payload else None}",
    )


def run_checks(root: Path, runs: Path, *, mode: GateMode = "experimental") -> list[Check]:
    if mode not in SCOPE_DOCUMENT:
        raise ValueError(f"unknown gate mode: {mode!r}")
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
    scope_path = root / SCOPE_DOCUMENT[mode]
    manifest_path = scope_path.with_suffix("").with_suffix(".eval-manifest.json")
    try:
        scope = load_mvp_scope(scope_path)
    except Exception as error:  # noqa: BLE001 - any failure here is the finding
        record("MVP-00", "scope_loads", False, f"{error}")
        return checks
    record("MVP-00", "scope_loads", True, f"scope_hash={scope.content_hash()}")
    # The gate's own identity check.  Handing ``--release`` an experimental
    # scope, or running the experimental gate against the release contract,
    # fails here rather than producing a verdict about the wrong document.
    record(
        "MVP-00",
        f"scope_release_class_is_{REQUIRED_RELEASE_CLASS[mode]}",
        scope.release_class == REQUIRED_RELEASE_CLASS[mode],
        f"mode={mode}, release_class={scope.release_class}, document={SCOPE_DOCUMENT[mode]}",
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
    train_content_hash: str | None = None
    if index is not None:
        record(
            "MVP-02",
            "demonstration_index_schema",
            index.get("schema") == DEMONSTRATION_INDEX_SCHEMA,
            f"stored={index.get('schema')!r}, current={DEMONSTRATION_INDEX_SCHEMA!r}",
        )
        record(
            "MVP-02",
            "demonstrations_match_locked_scope",
            index.get("scope_hash") == scope.content_hash(),
            f"index scope_hash={index.get('scope_hash')}",
        )
        record(
            "MVP-02",
            "demonstrations_match_prior",
            prior is not None and index.get("prior_hash") == prior.content_hash(),
            f"index prior_hash={index.get('prior_hash')}",
        )
        record(
            "MVP-02",
            "demonstrations_match_environment_fingerprint",
            expected_fingerprint is not None and index.get("fingerprint") == expected_fingerprint,
            f"stored={index.get('fingerprint')}",
        )
        for split in ("train", "dev"):
            summary = index.get("splits", {}).get(split, {})
            accepted = int(summary.get("episodes_accepted", 0))
            record("MVP-02", f"{split}_demonstrations_accepted", accepted > 0, f"{accepted} episodes")
            ledger = demos / split / "ledger.jsonl"
            record("MVP-02", f"{split}_generator_ledger_present", ledger.is_file(), str(ledger))
            summary_file = _load_json(demos / split / "summary.json")
            manifest = _load_json(demos / split / "manifest.json")
            content_hash = summary.get("content_hash")
            if split == "train" and _is_sha256(content_hash):
                train_content_hash = str(content_hash)
            record(
                "MVP-02",
                f"{split}_summary_matches_index",
                isinstance(summary, dict) and summary_file == summary,
                f"content_hash={content_hash!r}",
            )
            manifest_contract = (
                manifest is not None
                and manifest.get("schema") == DEMONSTRATION_MANIFEST_SCHEMA
                and manifest.get("dataset_schema") == DEMONSTRATION_SCHEMA
                and _is_sha256(manifest.get("content_hash"))
                and manifest.get("content_hash") == content_hash
                and isinstance(manifest.get("arrays"), dict)
                and set(manifest["arrays"]) == {"actions", "episode_index", "observations"}
            )
            record(
                "MVP-02",
                f"{split}_content_manifest",
                manifest_contract,
                f"stored={manifest.get('content_hash') if manifest else None!r}",
            )
            ledger_contract = False
            if ledger.is_file() and manifest is not None and isinstance(manifest.get("ledger"), dict):
                try:
                    rows = [json.loads(line) for line in ledger.read_text(encoding="utf-8").splitlines() if line.strip()]
                    ledger_contract = (
                        manifest["ledger"].get("rows") == len(rows)
                        and manifest["ledger"].get("sha256") == _canonical_sha256(rows)
                    )
                except (OSError, UnicodeError, json.JSONDecodeError):
                    ledger_contract = False
            record(
                "MVP-02",
                f"{split}_ledger_content_hash",
                ledger_contract,
                f"rows={manifest.get('ledger') if manifest else None}",
            )
            if runs == (root / "runs/mvp").resolve():
                try:
                    raw_dataset = DemonstrationSet.load(demos / split)
                    raw_content_ok = raw_dataset.content_hash() == content_hash
                    raw_detail = f"content_hash={raw_dataset.content_hash()}"
                except Exception as error:  # noqa: BLE001 - this is a fail-closed artifact gate
                    raw_content_ok = False
                    raw_detail = f"{type(error).__name__}: {error}"
                record("MVP-02", f"{split}_raw_arrays_match_manifest", raw_content_ok, raw_detail)
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
    promoted = False
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
            "MVP-03",
            "bc_dataset_lineage_matches_demonstrations",
            bc_payload is not None
            and train_content_hash is not None
            and bc_payload.get("lineage", {}).get("dataset_content_hash") == train_content_hash
            and training.get("demonstrations", {}).get("content_hash") == train_content_hash,
            f"demo={train_content_hash}, checkpoint={bc_payload.get('lineage') if bc_payload else None}",
        )
        record(
            "MVP-03",
            "bc_training_config_matches_checkpoint",
            bc_payload is not None
            and isinstance(bc.get("training_config"), dict)
            and bc.get("training_config") == bc_payload.get("metadata", {}).get("training_config"),
            f"report={bc.get('training_config')}",
        )
        record(
            "MVP-04",
            "bc_rollback_retained",
            bc_path is not None and bc_payload is not None and bc_path.name == "bc.pt",
            str(bc_path or bc.get("checkpoint")),
        )

        ppo = training.get("ppo") if isinstance(training.get("ppo"), dict) else None
        ppo_path: Path | None = None
        if ppo is None:
            record("MVP-04", "ppo_confirmation_run", False, "no PPO stage in the training report")
        else:
            ppo_rate = float(ppo.get("dev", {}).get("success_rate", 0.0))
            promoted = bool(ppo.get("promoted"))
            tolerance = PPO_REGRESSION_TOLERANCE_BY_MODE[mode]
            record(
                "MVP-04",
                "ppo_promotion_rule_respected",
                (ppo_rate >= bc_rate - tolerance) == promoted,
                f"ppo_dev={ppo_rate:.3f} bc_dev={bc_rate:.3f} promoted={promoted} tolerance={tolerance}",
            )
            record(
                "MVP-04",
                "ppo_introduced_no_safety_violation",
                int(ppo.get("dev", {}).get("safety_violation", 1)) == 0,
                f"safety_violation={ppo.get('dev', {}).get('safety_violation')}",
            )
            ppo_path, ppo_payload = validate_checkpoint("MVP-04", "ppo", ppo.get("checkpoint"))
            record(
                "MVP-04",
                "ppo_dataset_lineage_matches_demonstrations",
                ppo_payload is not None
                and train_content_hash is not None
                and ppo_payload.get("lineage", {}).get("dataset_content_hash") == train_content_hash,
                f"demo={train_content_hash}, checkpoint={ppo_payload.get('lineage') if ppo_payload else None}",
            )
            record(
                "MVP-04",
                "ppo_training_config_matches_checkpoint",
                ppo_payload is not None
                and isinstance(ppo.get("training_config"), dict)
                and ppo.get("training_config") == ppo_payload.get("metadata", {}).get("training_config"),
                f"report={ppo.get('training_config')}",
            )
            parent_ok = False
            parent_detail = "ppo checkpoint did not load"
            if ppo_payload is not None and bc_path is not None:
                try:
                    parent_path = _resolve_checkpoint(
                        ppo_payload.get("lineage", {}).get("parent"),
                        repository_root=root,
                        evidence_root=runs,
                    )
                    stored_parent_hash = ppo_payload.get("lineage", {}).get("parent_checkpoint_hash")
                    parent_ok = parent_path == bc_path and stored_parent_hash == _sha256(bc_path)
                    parent_detail = (
                        f"parent={parent_path}, bc={bc_path}, stored_hash={stored_parent_hash}, "
                        f"actual_hash={_sha256(bc_path)}"
                    )
                except (OSError, ValueError) as error:
                    parent_detail = f"{type(error).__name__}: {error}"
            record("MVP-04", "ppo_parent_checkpoint_lineage", parent_ok, parent_detail)

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
        expected_tiers = {spec.tier for spec in scope.eval_tiers}
        record(
            "MVP-05",
            "evaluation_tiers_complete",
            len(tiers) == len(expected_tiers) and set(tier_names) == expected_tiers,
            f"tiers={tier_names}, expected={sorted(expected_tiers)}",
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
                # A challenge tier has no absolute floor: its gate is the
                # paired uplift, applied in release mode once both arms exist.
                derived_pass = (
                    episodes == spec.episodes
                    and 0 <= successes <= episodes
                    and (spec.min_success_rate is None or rate >= spec.min_success_rate)
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
            len(derived_passes) == len(expected_tiers)
            and all(derived_passes)
            and evaluation.get("all_tiers_passed") is True,
            f"stored={evaluation.get('all_tiers_passed')!r}, derived={derived_passes}",
        )

    # -- MVP-06/07: cloud handoff and the model card -----------------------
    runner = root / "notebooks/mvp_grasp_policy.ipynb"
    record("MVP-06", "cloud_runner_present", runner.is_file(), str(runner))
    card = (
        root / "docs/reports/MVP-GRASP-POLICY-MODEL-CARD.md"
        if mode == "experimental"
        else root / "docs/reports/MVP-GRASP-POLICY-MODEL-CARD-V1.md"
    )
    record("MVP-07", "model_card_present", card.is_file(), str(card))
    if card.is_file():
        card_text = card.read_text(encoding="utf-8")
        required_class = REQUIRED_RELEASE_CLASS[mode]
        record(
            "MVP-07",
            f"model_card_declares_{required_class}",
            required_class in card_text
            # A release card that also carries the experimental class is
            # ambiguous about what it is describing, so it is refused.
            and (mode == "experimental" or EXPERIMENTAL_RELEASE_CLASS not in card_text),
            f"release_class stated in {card.name}",
        )
        record(
            "MVP-07",
            "model_card_states_limitations",
            any(marker in card_text.casefold() for marker in ("giới hạn", "limitations")),
            "limitations section",
        )

    if mode == "release":
        _release_checks(
            record,
            root=root,
            runs=runs,
            scope=scope,
            candidate_path=candidate_path,
            candidate_payload=candidate_payload,
            evaluation=evaluation,
            prior_report=prior_report,
            promoted=promoted,
        )
    return checks


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--runs", type=Path, default=Path("runs/mvp"))
    parser.add_argument("--json", type=Path, default=None, help="also write the result as JSON")
    parser.add_argument(
        "--release",
        dest="mode",
        action="store_const",
        const="release",
        default="experimental",
        help="run the release gate against scope v1 instead of the experimental gate against scope v0",
    )
    args = parser.parse_args(argv)

    mode: GateMode = args.mode
    checks = run_checks(args.root, args.runs, mode=mode)
    failed = [check for check in checks if not check.passed]
    for check in checks:
        print(f"{'PASS' if check.passed else 'FAIL'}  [{check.package}] {check.name}: {check.detail}")
    print(f"\n{len(checks) - len(failed)}/{len(checks)} checks passed")
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(
                {
                    "schema": "qdgrasp/mvp-closure/v1",
                    "mode": mode,
                    # Machine-readable, so that "the gate passed" cannot be
                    # quoted out of an experimental run as a release verdict.
                    "is_release_gate": mode == "release",
                    "scope_document": str(SCOPE_DOCUMENT[mode]),
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
        print(f"MVP status: blocked_with_evidence ({mode} gate)")
        return 1
    if mode == "release":
        print("MVP status: release_gate_passed")
        return 0
    print("MVP status: complete (experimental gate; this is not a release verdict)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
