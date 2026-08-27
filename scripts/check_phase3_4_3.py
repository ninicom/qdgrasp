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
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from qdgrasp.roadmap import ManifestError, audit_closure, load_manifest

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
    args = parser.parse_args()

    try:
        manifest = load_manifest(args.manifest)
    except (ManifestError, OSError) as exc:
        print(json.dumps({"verdict": "CONFIG_ERROR", "error": str(exc)}, indent=2))
        print(f"Phase 3.4.3 gate could not read its manifest: {exc}", file=sys.stderr)
        return CONFIG_ERROR_EXIT

    require_release = args.profile == "release"
    closure = audit_closure(manifest, repo_root=REPO_ROOT, require_release=require_release)

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

    exit_code = closure.exit_code
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
