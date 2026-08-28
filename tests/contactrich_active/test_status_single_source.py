"""WRK-R6: the manifest is the only source of truth for status.

RRV-06 found plan, packet, guide and ledger disagreeing. They could, because
docs/roadmap/ -- where every one of those documents lives -- was outside the docs
gate entirely. It is swept now, and a hand-typed count has to name the manifest
it came from.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from qdgrasp.roadmap import audit_closure, load_manifest
from qdgrasp.roadmap.review_packet import manifest_digest

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST = REPO_ROOT / "docs" / "roadmap" / "phase3_4_3_requirements.yaml"


def test_the_digest_moves_when_the_manifest_does(tmp_path):
    original = manifest_digest(MANIFEST)
    copy = tmp_path / "manifest.yaml"
    copy.write_text(MANIFEST.read_text(encoding="utf-8") + "\n# a change\n", encoding="utf-8")
    assert manifest_digest(copy) != original


def test_roadmap_documents_are_inside_the_docs_gate():
    """The prose WRK-R6 governs was not being checked at all."""
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    try:
        import check_docs
    finally:
        sys.path.pop(0)

    _issues, count = check_docs.validate_root(REPO_ROOT)
    roadmap_docs = list((REPO_ROOT / "docs" / "roadmap").glob("*.md"))
    assert roadmap_docs, "there should be roadmap documents to check"
    assert count >= len(roadmap_docs)


def test_an_unstamped_status_snapshot_is_refused(tmp_path):
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    try:
        import check_docs
    finally:
        sys.path.pop(0)

    issues: list = []
    check_docs.validate_status_snapshots(
        REPO_ROOT,
        Path("docs/roadmap/made-up.md"),
        "The ledger now reads 58 passed, 21 failed, 5 blocked.\n",
        "roadmap",
        issues,
    )
    assert issues, "a count with no manifest hash must be refused"


def test_a_stamped_status_snapshot_is_accepted(tmp_path):
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    try:
        import check_docs
    finally:
        sys.path.pop(0)

    digest = manifest_digest(MANIFEST)
    issues: list = []
    check_docs.validate_status_snapshots(
        REPO_ROOT,
        Path("docs/roadmap/made-up.md"),
        f"The ledger reads 58 passed, 21 failed, 5 blocked -- `manifest {digest}`.\n",
        "roadmap",
        issues,
    )
    assert issues == []


def test_a_revision_record_may_state_the_counts_of_its_own_day():
    """History is not stale prose, and revision records are never rewritten."""
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    try:
        import check_docs
    finally:
        sys.path.pop(0)

    issues: list = []
    check_docs.validate_status_snapshots(
        REPO_ROOT,
        Path("docs/revisions/REV-something.md"),
        "On the day: 67 passed, 12 failed, 5 blocked.\n",
        "revisions",
        issues,
    )
    assert issues == []


def test_the_plan_states_the_same_counts_the_manifest_does():
    """The one check that would have caught RRV-06 before a reviewer did."""
    manifest = load_manifest(MANIFEST)
    result = audit_closure(manifest, repo_root=REPO_ROOT)
    counts = result.status_counts
    plan = (REPO_ROOT / "docs" / "roadmap" / "PHASE3_4_3_ACTIVE_GATE_COMPLETION_PLAN.md").read_text(
        encoding="utf-8"
    )
    stamped = f"{counts.get('passed', 0)} passed, {counts.get('failed', 0)} failed"
    assert stamped in plan, f"the plan should state {stamped!r}"
    assert manifest_digest(MANIFEST) in plan


def test_the_docs_gate_passes_on_this_tree():
    completed = subprocess.run(
        [sys.executable, "scripts/check_docs.py"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout[-3000:]
