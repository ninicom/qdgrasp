#!/usr/bin/env python3
"""Fail-closed static/runtime checks for the Phase 0 library foundation."""

from __future__ import annotations

import hashlib
import importlib.metadata
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REQUIRED_FILES = {
    "LICENSE": "GNU AFFERO GENERAL PUBLIC LICENSE",
    "NOTICE": "Ultralytics",
    "THIRD_PARTY.yml": "AGPL-3.0-only",
    "pyproject.toml": 'license = "AGPL-3.0-only"',
    "robot_assets.lock.yaml": "removed_from_active_scope",
}
SECRET_PATTERNS = {
    "private key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "Kaggle credential": re.compile(r'"(?:key|token)"\s*:\s*"[^"\n]{12,}"', re.IGNORECASE),
    "GitHub token": re.compile(r"\b(?:ghp|github_pat)_[A-Za-z0-9_]{20,}\b"),
}
RH_ALLOWLIST = {
    Path("PLAN.md"),
    Path("README.md"),
    Path("THIRD_PARTY.yml"),
    Path("robot_assets.lock.yaml"),
    Path("scripts/check_phase0.py"),
    Path("docs/archive/README.md"),
    Path("docs/README.md"),
    Path("docs/decisions/0007-agpl-community-library.md"),
    Path("docs/roadmap/PROJECT_PHASES.md"),
    Path("docs/revisions/REV-20260822-009-agpl-library-first-phase0.md"),
    Path("docs/sessions/SESSION-20260822-018-phase0-agpl-library.md"),
}


def tracked_files() -> list[Path]:
    output = subprocess.check_output(["git", "ls-files", "--cached", "--others", "--exclude-standard"], cwd=ROOT)
    return [Path(line) for line in output.decode().splitlines() if line]


def main() -> int:
    problems: list[str] = []
    for name, marker in REQUIRED_FILES.items():
        path = ROOT / name
        if not path.is_file():
            problems.append(f"missing {name}")
        elif marker not in path.read_text(encoding="utf-8"):
            problems.append(f"{name} missing marker {marker!r}")

    if importlib.metadata.version("qdgrasp") != "0.1.0a1":
        problems.append("installed qdgrasp version is not 0.1.0a1")

    for relative in tracked_files():
        lowered = str(relative).lower()
        if lowered.endswith("kaggle.json") or lowered.endswith(".env"):
            problems.append(f"credential file is publishable: {relative}")
        if "rh56e2" in lowered or "rh56-e2" in lowered:
            problems.append(f"RH56E2 artifact path is forbidden: {relative}")
        path = ROOT / relative
        if not path.is_file() or path.stat().st_size > 2_000_000:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for label, pattern in SECRET_PATTERNS.items():
            if pattern.search(text):
                problems.append(f"possible {label} in {relative}")
        if relative not in RH_ALLOWLIST and not lowered.startswith("docs/archive/"):
            if re.search(r"rh56[\s_-]*e2", text, re.IGNORECASE):
                problems.append(f"active RH56E2 reference in {relative}")

    plan_hash = hashlib.sha256((ROOT / "PLAN.md").read_bytes()).hexdigest()
    print(f"Phase 0 foundation: {'PASS' if not problems else 'FAIL'}")
    print(f"PLAN sha256: {plan_hash}")
    for problem in problems:
        print(f"- {problem}")
    return 0 if not problems else 1


if __name__ == "__main__":
    raise SystemExit(main())
