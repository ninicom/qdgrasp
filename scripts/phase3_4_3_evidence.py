"""Record sub-phase evidence for Phase 3.4.3 (S0..S12).

A requirement may only move to ``passed`` when it can name evidence that exists.
This script produces that evidence the same way every time: it runs the named
test selection, captures the result together with the commit, the worktree
state and a hash of every implementation file the sub-phase claims, and writes
one JSON record under ``evidence/phase3_4_3/``.

The record is deliberately unflattering when it should be: a failing selection
is written as ``passed=false`` rather than not written at all, and a dirty
worktree is recorded rather than hidden, because a gate result that cannot be
reproduced from a commit is not evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_ROOT = REPO_ROOT / "evidence" / "phase3_4_3"


def _git(*args: str) -> str:
    completed = subprocess.run(
        ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, timeout=120, check=False
    )
    return completed.stdout.strip() if completed.returncode == 0 else ""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _hash_refs(refs: list[str]) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for ref in refs:
        target = REPO_ROOT / ref
        if target.is_file():
            hashes[ref] = _sha256_file(target)
        elif target.is_dir():
            parts = [
                f"{child.relative_to(REPO_ROOT)}:{_sha256_file(child)}"
                for child in sorted(target.rglob("*.py"))
                if child.is_file()
            ]
            hashes[ref] = hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()
        else:
            hashes[ref] = "missing"
    return hashes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sub-phase", required=True, help="e.g. S0")
    parser.add_argument("--gates", required=True, help="comma-separated gate ids, e.g. G00,C08")
    parser.add_argument("--blockers", default="", help="comma-separated blocker ids, e.g. B-09")
    parser.add_argument("--tests", required=True, help="comma-separated pytest selections")
    parser.add_argument("--implementation", required=True, help="comma-separated repo-relative paths")
    parser.add_argument("--note", default="")
    args = parser.parse_args()

    tests = [item.strip() for item in args.tests.split(",") if item.strip()]
    implementation = [item.strip() for item in args.implementation.split(",") if item.strip()]

    completed = subprocess.run(
        [sys.executable, "-m", "pytest", *tests, "-q", "-p", "no:cacheprovider"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=5400,
        check=False,
    )
    stdout_lines = completed.stdout.strip().splitlines()

    record: dict[str, Any] = {
        "schema": "qdgrasp/phase3_4_3-subphase-evidence/v1",
        "sub_phase": args.sub_phase,
        "gates": [item.strip() for item in args.gates.split(",") if item.strip()],
        "blockers": [item.strip() for item in args.blockers.split(",") if item.strip()],
        "recorded_at": datetime.now(UTC).isoformat(),
        "commit": _git("rev-parse", "HEAD"),
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "worktree_dirty": bool(_git("status", "--porcelain")),
        "scope": {
            "active_hands": ["leap_hand", "wonik_allegro"],
            "paused_hands": ["shadow_hand"],
            "three_hand_coverage": False,
            "historical_p3_4_state": "paused_by_ADR-0008",
        },
        "tests": {
            "selection": tests,
            "returncode": completed.returncode,
            "passed": completed.returncode == 0,
            "summary": stdout_lines[-1] if stdout_lines else "",
            "tail": completed.stdout[-4000:] if completed.returncode != 0 else "",
        },
        "implementation_refs": implementation,
        "implementation_hashes": _hash_refs(implementation),
        "python": sys.version.split()[0],
        "device": "cpu",
        "note": args.note,
    }

    target_dir = EVIDENCE_ROOT / args.sub_phase.lower()
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / "result.json"
    target.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(json.dumps({"written": str(target.relative_to(REPO_ROOT)), "passed": record["tests"]["passed"]}, indent=2))
    return 0 if completed.returncode == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
