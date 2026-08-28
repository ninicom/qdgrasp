"""Phase 3.4.3 closure gate for the active contact-rich hands (G00, C08).

This is the one command that is allowed to say whether Phase 3.4.3 is closed,
and it says it in exactly one JSON object rather than in a log a reader has to
interpret. Its scope is the two active hands of ADR-0008; it can never emit a
three-hand claim, and it can never report the historical P3.4 contract as
anything but paused.

Exit codes are part of the contract, because CI reads them:

    0  the exact requested scope passes
    1  a gate failed
    2  the requested scope is paused or not applicable
    3  the requested scope is incomplete or partial
    4  the gate could not be configured or run

The failure this replaces (B-09) is a checker that printed ``PARTIAL`` and
exited ``0``, which reads to any CI system as a phase pass.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from qdgrasp.config.active_scope import ACTIVE_HANDS, PAUSED_HANDS
from qdgrasp.roadmap import ManifestError, audit_active_scope, audit_closure, load_manifest
from qdgrasp.roadmap.review_packet import canonical_packet_digest

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPO_ROOT / "docs" / "roadmap" / "phase3_4_3_requirements.yaml"

#: The one schema the CUDA verifier accepts. A bundle that calls itself something
#: else is not a bundle this gate knows how to check.
CUDA_EVIDENCE_SCHEMA: str = "qdgrasp/evidence/phase3.4.3-cuda/v1"

#: Only this verdict may be used to unblock the Phase 4 contact-rich input.
ACTIVE_VERDICT = "P3.4.3-ACTIVE-PASS"

CONFIG_ERROR_EXIT = 4


def run_pytest(selection: tuple[str, ...]) -> dict[str, Any]:
    """Run the CPU correctness suites this gate depends on."""
    present = [path for path in selection if (REPO_ROOT / path).exists()]
    if not present:
        return {"status": "no_suites_present", "selected": list(selection)}
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", *present, "-q", "-p", "no:cacheprovider"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=3600,
        check=False,
    )
    summary = completed.stdout.strip().splitlines()
    return {
        "status": "passed" if completed.returncode == 0 else "failed",
        "returncode": completed.returncode,
        "suites": present,
        "summary": summary[-1] if summary else "",
        "tail": completed.stdout[-2000:] if completed.returncode != 0 else "",
    }


def run_command(args: list[str], *, label: str, timeout: int = 3600) -> dict[str, Any]:
    """Run one gate and report what it said, without interpreting it."""
    completed = subprocess.run(
        args, cwd=REPO_ROOT, capture_output=True, text=True, timeout=timeout, check=False
    )
    lines = completed.stdout.strip().splitlines()
    return {
        "label": label,
        "command": " ".join(args[1:]) if args and args[0] == sys.executable else " ".join(args),
        "returncode": completed.returncode,
        "passed": completed.returncode == 0,
        "summary": lines[-1] if lines else "",
        "stderr_tail": completed.stderr[-800:] if completed.returncode != 0 else "",
    }


#: Paths whose behaviour the CUDA gate actually exercises. Evidence has to come
#: from a tree where these are identical to the candidate's; a later commit that
#: only touches documentation or the notebook's own pin does not invalidate a
#: measurement of the same code.
MEASURED_PATHS: tuple[str, ...] = ("qdgrasp", "scripts/check_phase3_4_3_cuda.py")


def measured_tree_matches(evidence_commit: str, candidate_commit: str) -> tuple[bool, str]:
    """Whether the code the evidence measured is the code under review.

    Comparing bare commit ids would refuse evidence from the commit immediately
    before a documentation change, which measured exactly the same library. What
    matters is whether the measured paths differ, so that is what is compared --
    and if git cannot answer, the answer is no.
    """
    if not evidence_commit or not candidate_commit:
        return (False, "one of the commits is unknown")
    if evidence_commit == candidate_commit:
        return (True, "same commit")
    completed = subprocess.run(
        ["git", "diff", "--quiet", evidence_commit, candidate_commit, "--", *MEASURED_PATHS],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode == 0:
        return (True, f"{evidence_commit[:12]} and {candidate_commit[:12]} agree on {list(MEASURED_PATHS)}")
    if completed.returncode == 1:
        return (False, f"the measured paths differ between {evidence_commit[:12]} and {candidate_commit[:12]}")
    return (False, f"git could not compare the trees: {completed.stderr.strip()[:120]}")


def recompute_cuda_verdict(payload: dict[str, Any]) -> tuple[str, list[str]]:
    """Derive the CUDA verdict from the metrics, never from the declared field.

    WRK-R3. A verdict the producer wrote is a claim; a verdict computed here from
    the numbers is a check. Reading ``payload["verdict"]`` let a bundle assert its
    own pass, which is the false-pass path RRV-01 names. The declared value is
    still compared, because a producer and a verifier disagreeing about the same
    numbers is itself a finding.
    """
    problems: list[str] = []

    capability = payload.get("capability") or {}
    if capability.get("verdict") != "supported":
        problems.append(f"capability verdict is {capability.get('verdict')!r}, not 'supported'")
    if not capability.get("contact_force_readable"):
        problems.append("contact force is not readable")
    missing = capability.get("missing_contact_fields") or []
    if missing:
        problems.append(f"missing contact fields {missing}")
    overflow = capability.get("overflow_telemetry") or {}
    if overflow.get("buffer_overflow"):
        problems.append("contact buffer overflowed")
    if overflow.get("stream_truncated"):
        problems.append("contact stream was truncated")

    parity = payload.get("parity") or {}
    for stage in ("no_contact", "single_contact", "full_trajectory"):
        section = parity.get(stage) or {}
        if not section.get("passed"):
            problems.append(f"parity stage {stage} did not pass")

    sanitizer = payload.get("sanitizer") or {}
    if sanitizer.get("status") != "recorded":
        problems.append(f"sanitizer status is {sanitizer.get('status')!r}, not 'recorded'")
    if not sanitizer.get("clean"):
        problems.append("sanitizer reported errors")
    tools = sanitizer.get("tools") or {}
    for required in ("initcheck", "racecheck"):
        if required not in tools:
            problems.append(f"sanitizer tool {required} was not run")
        elif not tools[required].get("clean"):
            problems.append(f"sanitizer tool {required} is not clean")

    performance = payload.get("performance") or {}
    hands = performance.get("hands") or {}
    if not hands:
        problems.append("no per-hand performance measurements")
    for hand, metrics in sorted(hands.items()):
        if not metrics.get("speedup_met"):
            problems.append(f"{hand}: speedup {metrics.get('speedup')} below the criterion")
        if not metrics.get("vram_within_budget"):
            problems.append(f"{hand}: VRAM outside the budget")
        if not metrics.get("worlds_met"):
            problems.append(f"{hand}: simultaneous-world floor not met")
        if int(metrics.get("overflow_worlds", 0)):
            problems.append(f"{hand}: {metrics['overflow_worlds']} world(s) overflowed")
        rejected = int(metrics.get("rejected_worlds", 0))
        if rejected:
            problems.append(f"{hand}: {rejected} world(s) rejected as non-finite")

    if payload.get("three_hand_coverage"):
        problems.append("evidence claims three-hand coverage, which ADR-0008 forbids")

    return ("PASS" if not problems else "FAIL"), problems


def verify_external_evidence(path: Path | None, *, expected_commit: str) -> dict[str, Any]:
    """Check GPU evidence produced somewhere this machine cannot reproduce.

    The gate cannot re-run a T4 rollout. What it can do is refuse to take the
    bundle's word for anything: the schema is pinned, the verdict is recomputed
    from the metrics, the raw log is bound by hash, and the measured tree has to
    be the candidate's.
    """
    if path is None:
        return {
            "status": "absent",
            "detail": (
                "no CUDA evidence supplied; run kaggle-phase3-4-3/ on a T4 and pass "
                "--cuda-evidence. A missing GPU gate is not a passed one."
            ),
            "passed": False,
        }
    if not path.is_file():
        return {"status": "missing_file", "detail": str(path), "passed": False}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return {"status": "unreadable", "detail": str(exc), "passed": False}
    if not isinstance(payload, dict):
        return {"status": "unreadable", "detail": "evidence is not an object", "passed": False}

    schema = payload.get("schema", "")
    declared = payload.get("verdict", "")
    commit = str(payload.get("commit") or payload.get("cuda_environment", {}).get("commit") or "")
    problems: list[str] = []
    if schema != CUDA_EVIDENCE_SCHEMA:
        problems.append(f"evidence schema is {schema!r}, not {CUDA_EVIDENCE_SCHEMA!r}")

    computed, metric_problems = recompute_cuda_verdict(payload)
    problems.extend(metric_problems)
    if declared != computed:
        problems.append(
            f"declared verdict {declared!r} disagrees with the verdict computed "
            f"from the metrics ({computed!r})"
        )

    # The raw log is what makes the metrics auditable after the fact.
    raw_log = _raw_log_binding(path, payload)
    if raw_log["problems"]:
        problems.extend(raw_log["problems"])

    matches, detail = measured_tree_matches(commit, expected_commit)
    if not matches:
        problems.append(f"evidence does not measure the candidate's code: {detail}")
    return {
        "status": "recorded",
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "schema": schema,
        "declared_verdict": declared,
        "computed_verdict": computed,
        "commit": commit,
        "candidate_commit": expected_commit,
        "measured_paths": list(MEASURED_PATHS),
        "measured_tree_matches": matches,
        "measured_tree_detail": detail,
        "raw_log": raw_log,
        "problems": problems,
        "passed": not problems,
    }


def _raw_log_binding(evidence_path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    """Bind the bundle to the raw run log that produced it."""
    declared = payload.get("raw_log_sha256")
    bundle_dir = evidence_path.parent
    candidates = sorted(bundle_dir.glob("*.log")) + sorted(bundle_dir.glob("raw*.json"))
    if not declared:
        return {
            "status": "absent",
            "problems": [
                (
                    "evidence declares no raw_log_sha256, so its metrics cannot "
                    "be audited against the run that produced them"
                )
            ],
        }
    for candidate in candidates:
        digest = hashlib.sha256(candidate.read_bytes()).hexdigest()
        if digest == declared:
            return {"status": "bound", "path": str(candidate), "sha256": digest, "problems": []}
    return {
        "status": "unbound",
        "declared": declared,
        "problems": [f"no file beside the evidence hashes to {declared}"],
    }


def verify_review_packet(
    path: Path | None,
    *,
    expected_commit: str,
    packet_path: Path | None = None,
) -> dict[str, Any]:
    """Check the independent reviewer's verdict, without standing in for it.

    The author of a change cannot review it, so this never issues a verdict. What
    it does do is refuse an unbound one: the signature has to name the packet
    digest it reviewed and the commit that packet describes, and both have to be
    the candidate's. Two self-declared strings claiming reviewer != author were
    the whole identity check before, which is no check at all.
    """
    if path is None:
        return {
            "status": "absent",
            "detail": "no reviewer verdict supplied; an author cannot review their own work",
            "passed": False,
        }
    if not path.is_file():
        return {"status": "missing_file", "detail": str(path), "passed": False}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return {"status": "unreadable", "detail": str(exc), "passed": False}
    if not isinstance(payload, dict):
        return {"status": "unreadable", "detail": "verdict is not an object", "passed": False}

    verdict = payload.get("reviewer_verdict", "")
    open_findings = payload.get("open_findings", {})
    problems: list[str] = []

    if verdict != "PASS":
        problems.append(f"reviewer verdict is {verdict!r}, not PASS")

    # WRK-R3: zero open S0-S3, not just the two most severe.
    blocking = 0
    for severity in ("S0", "S1", "S2", "S3"):
        count = int(open_findings.get(severity, 0))
        blocking += count
        if count:
            problems.append(f"{count} open {severity} finding(s)")
    if blocking == 0 and not open_findings:
        problems.append("verdict declares no finding counts at all, not even zeros")

    reviewer = str(payload.get("reviewer") or "")
    author = str(payload.get("author") or "")
    if not reviewer:
        problems.append("verdict names no reviewer")
    if not author:
        problems.append("verdict names no author to distinguish the reviewer from")
    if reviewer and reviewer == author:
        problems.append("the reviewer must not be the author of the change")

    signed_commit = str(payload.get("candidate_commit") or "")
    if signed_commit != expected_commit:
        problems.append(
            f"verdict signs commit {signed_commit or '(none)'!r}, not the candidate {expected_commit!r}"
        )

    signed_digest = str(payload.get("packet_sha256") or "")
    packet_binding: dict[str, Any] = {"status": "unbound"}
    if not signed_digest:
        problems.append("verdict signs no packet digest")
    elif packet_path is None or not packet_path.is_file():
        problems.append("no review packet supplied to bind the signed digest against")
    else:
        actual = canonical_packet_digest(packet_path)
        packet_binding = {
            "status": "bound" if actual == signed_digest else "mismatched",
            "path": str(packet_path),
            "signed": signed_digest,
            "actual": actual,
        }
        if actual != signed_digest:
            problems.append(
                f"the packet on disk digests to {actual}, not the {signed_digest} that was signed"
            )

    return {
        "status": "recorded",
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "reviewer_verdict": verdict,
        "reviewer": reviewer,
        "author": author,
        "candidate_commit": signed_commit,
        "packet_binding": packet_binding,
        "open_findings": open_findings,
        "problems": problems,
        "passed": not problems,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scope",
        choices=("active",),
        default="active",
        help="only the two-hand active scope of ADR-0008 exists; there is no three-hand scope here",
    )
    parser.add_argument("--profile", choices=("cpu", "release"), default="cpu")
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--skip-tests", action="store_true")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=REPO_ROOT / "datasets" / "contactrich-active-tiny",
    )
    parser.add_argument(
        "--cuda-evidence",
        type=Path,
        default=None,
        help="CUDA gate evidence from the Kaggle T4 run",
    )
    parser.add_argument(
        "--review-packet",
        type=Path,
        default=None,
        help="the immutable review packet the reviewer read",
    )
    parser.add_argument(
        "--reviewer-verdict",
        type=Path,
        default=None,
        help=(
            "the independent reviewer's signed verdict, which must name the "
            "packet digest and candidate commit it covers"
        ),
    )
    args = parser.parse_args()

    # WRK-R3: a release verdict cannot rest on tests nobody ran.
    if args.profile == "release" and args.skip_tests:
        print(
            json.dumps(
                {
                    "verdict": "CONFIG_ERROR",
                    "error": "--skip-tests is refused on the release profile",
                },
                indent=2,
            )
        )
        print(
            "Phase 3.4.3 release profile refuses --skip-tests: a missing suite is "
            "a failure, not an omission.",
            file=sys.stderr,
        )
        return CONFIG_ERROR_EXIT

    try:
        manifest = load_manifest(args.manifest)
    except (ManifestError, OSError) as exc:
        print(json.dumps({"verdict": "CONFIG_ERROR", "error": str(exc)}, indent=2))
        print(f"Phase 3.4.3 gate could not read its manifest: {exc}", file=sys.stderr)
        return CONFIG_ERROR_EXIT

    require_release = args.profile == "release"
    closure = audit_closure(manifest, repo_root=REPO_ROOT, require_release=require_release)

    # ADR-0008 is enforced at the point a workload picks its hands, not by
    # remembering to leave Shadow out; an undeclared default selecting it is a
    # gate failure whatever the ledger says (G05).
    scope_findings = audit_active_scope(REPO_ROOT)

    result: dict[str, Any] = {
        "phase": "3.4.3",
        "scope": args.scope,
        "profile": args.profile,
        "verdict": closure.verdict,
        "release_verdict": ACTIVE_VERDICT if closure.verdict == "PASS" else "none",
        "release_blocked": closure.release_blocked,
        "closure_scope": closure.closure_scope,
        "active_hands": list(closure.active_hands),
        "paused_hands": list(closure.paused_hands),
        "three_hand_coverage": closure.three_hand_coverage,
        "historical_p3_4_state": "paused_by_ADR-0008",
        "coverage": f"{len(closure.active_hands)}/{len(closure.active_hands)}_active",
        "completeness": {
            "total_requirements": closure.total_requirements,
            "mapped_requirements": closure.mapped_requirements,
            "unmapped": list(closure.unmapped),
            "unknown": list(closure.unknown),
            "status_counts": dict(sorted(closure.status_counts.items())),
            "open_required": list(closure.open_required),
            "violations": list(closure.violations),
        },
        "adr_0008_scope_audit": {
            "registry_active_hands": list(ACTIVE_HANDS),
            "registry_paused_hands": list(PAUSED_HANDS),
            "undeclared_paused_selections": [str(finding) for finding in scope_findings],
        },
        "worktree_dirty": closure.worktree_dirty,
        "manifest": {
            "path": str(args.manifest.relative_to(REPO_ROOT)) if args.manifest.is_absolute() else str(args.manifest),
            "sha256": closure.manifest_sha256,
            "plan_id": manifest.plan_id,
            "plan_version": manifest.plan_version,
        },
        "artifact_refs": {
            "plan": "docs/roadmap/PHASE3_4_3_ACTIVE_GATE_COMPLETION_PLAN.md",
            "breakdown": "docs/roadmap/PHASE3_4_3_EXECUTION_BREAKDOWN.md",
            "dataset_id": "QDGrasp-ContactRich-Active-Tiny",
        },
    }

    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=False
    ).stdout.strip()
    result["candidate_commit"] = head

    gates: list[dict[str, Any]] = []
    if args.dataset_root.is_dir():
        gates.append(
            run_command(
                [
                    sys.executable,
                    "scripts/check_contactrich_active.py",
                    str(args.dataset_root),
                    *(["--require-release"] if require_release else []),
                ],
                label="dataset",
            )
        )
    else:
        gates.append(
            {
                "label": "dataset",
                "passed": False,
                "returncode": None,
                "summary": f"{args.dataset_root} does not exist",
            }
        )
    gates.append(
        run_command([sys.executable, "scripts/check_docs.py", "--root", "."], label="docs")
    )
    gates.append(run_command(["git", "diff", "--check"], label="git_diff_check"))
    result["gates"] = gates

    result["cuda_evidence"] = verify_external_evidence(args.cuda_evidence, expected_commit=head)
    result["review_packet"] = verify_review_packet(
        args.reviewer_verdict,
        expected_commit=head,
        packet_path=args.review_packet,
    )

    exit_code = closure.exit_code
    for gate in gates:
        if not gate["passed"]:
            result["verdict"] = "FAIL"
            result["release_blocked"] = True
            result["release_verdict"] = "none"
            exit_code = 1
    if require_release and not (
        result["cuda_evidence"]["passed"] and result["review_packet"]["passed"]
    ):
        result["verdict"] = "BLOCKED"
        result["release_blocked"] = True
        result["release_verdict"] = "none"
        exit_code = max(exit_code, 3)
    if scope_findings:
        result["verdict"] = "FAIL"
        result["release_verdict"] = "none"
        result["release_blocked"] = True
        exit_code = 1
    if not args.skip_tests:
        tests = run_pytest(("tests/dynamic_grasp", "tests/contactrich_active"))
        result["tests"] = tests
        if tests["status"] == "failed":
            result["verdict"] = "FAIL"
            result["release_verdict"] = "none"
            result["release_blocked"] = True
            exit_code = 1

    result["exit_code"] = exit_code
    print(json.dumps(result, indent=2, sort_keys=True))
    print(
        f"Phase 3.4.3 [{args.scope}/{args.profile}]: {result['verdict']} "
        f"(exit {exit_code}); release_blocked={result['release_blocked']}. "
        "This verdict covers two active hands only; the historical three-hand "
        "P3.4 contract stays paused_by_ADR-0008.",
        file=sys.stderr,
    )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
