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

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPO_ROOT / "docs" / "roadmap" / "phase3_4_3_requirements.yaml"

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


def verify_external_evidence(path: Path | None, *, expected_commit: str) -> dict[str, Any]:
    """Check GPU evidence produced somewhere this machine cannot reproduce.

    The gate cannot re-run a T4 rollout, so it checks what it can: that the
    evidence exists, that it is the schema it claims, that its verdict is a
    pass, and that it was produced from the commit under review. Evidence from
    a different commit is evidence about a different tree.
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

    schema = payload.get("schema", "")
    verdict = payload.get("verdict", "")
    commit = str(payload.get("commit") or payload.get("cuda_environment", {}).get("commit") or "")
    problems: list[str] = []
    if not schema.startswith("qdgrasp/evidence/phase3.4.3-"):
        problems.append(f"unexpected evidence schema {schema!r}")
    if verdict != "PASS":
        problems.append(f"CUDA gate verdict is {verdict!r}, not PASS")
    if expected_commit and commit and commit != expected_commit:
        problems.append(
            f"evidence was produced from commit {commit[:12]}, not the candidate "
            f"{expected_commit[:12]}"
        )
    return {
        "status": "recorded",
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "schema": schema,
        "verdict": verdict,
        "commit": commit,
        "problems": problems,
        "passed": not problems,
    }


def verify_review_packet(path: Path | None) -> dict[str, Any]:
    """Check the independent reviewer's verdict, without standing in for it.

    The author of a change cannot review it, so this only reads what a reviewer
    signed: a PASS on an exact packet hash with no open S0-S1 findings.
    """
    if path is None:
        return {
            "status": "absent",
            "detail": "no review packet supplied; an author cannot review their own work",
            "passed": False,
        }
    if not path.is_file():
        return {"status": "missing_file", "detail": str(path), "passed": False}
    payload = json.loads(path.read_text(encoding="utf-8"))
    verdict = payload.get("reviewer_verdict", "")
    open_findings = payload.get("open_findings", {})
    blocking = sum(int(open_findings.get(severity, 0)) for severity in ("S0", "S1"))
    problems: list[str] = []
    if verdict != "PASS":
        problems.append(f"reviewer verdict is {verdict!r}, not PASS")
    if blocking:
        problems.append(f"{blocking} unresolved S0/S1 finding(s)")
    if not payload.get("reviewer") or payload.get("reviewer") == payload.get("author"):
        problems.append("the reviewer must not be the author of the change")
    return {
        "status": "recorded",
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "reviewer_verdict": verdict,
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
        help="the independent reviewer's signed verdict",
    )
    args = parser.parse_args()

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
    result["review_packet"] = verify_review_packet(args.review_packet)

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
