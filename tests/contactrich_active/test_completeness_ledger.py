"""S0 — the requirement ledger and the closure exit codes (G00, C08, B-09)."""

from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from qdgrasp.roadmap import ALLOWED_STATUS, ManifestError, audit_closure, load_manifest

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST = REPO_ROOT / "docs" / "roadmap" / "phase3_4_3_requirements.yaml"


@pytest.fixture(scope="module")
def manifest():
    return load_manifest(MANIFEST)


def _write(tmp_path: Path, document: dict) -> Path:
    path = tmp_path / "manifest.yaml"
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    return path


@pytest.fixture()
def document() -> dict:
    return yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))


def test_shipped_manifest_parses_and_is_fully_mapped(manifest) -> None:
    verdict = audit_closure(manifest, repo_root=REPO_ROOT, worktree_dirty=False)
    assert verdict.total_requirements == len(manifest.requirements)
    assert verdict.unmapped == ()
    assert verdict.unknown == ()
    assert verdict.mapped_requirements == verdict.total_requirements


def test_scope_never_claims_three_hand_coverage(manifest) -> None:
    verdict = audit_closure(manifest, repo_root=REPO_ROOT, worktree_dirty=False)
    assert verdict.three_hand_coverage is False
    assert set(verdict.active_hands) == {"leap_hand", "wonik_allegro"}
    assert verdict.paused_hands == ("shadow_hand",)


def test_pending_requirements_cannot_produce_a_pass(manifest) -> None:
    verdict = audit_closure(manifest, repo_root=REPO_ROOT, worktree_dirty=False)
    assert verdict.verdict != "PASS"
    assert verdict.release_blocked is True
    assert verdict.exit_code != 0


def test_duplicate_requirement_id_is_rejected(tmp_path: Path, document: dict) -> None:
    document = copy.deepcopy(document)
    document["requirements"].append(copy.deepcopy(document["requirements"][0]))
    with pytest.raises(ManifestError, match="duplicate requirement id"):
        load_manifest(_write(tmp_path, document))


def test_unknown_status_is_rejected(tmp_path: Path, document: dict) -> None:
    document = copy.deepcopy(document)
    document["requirements"][0]["status"] = "mostly_done"
    with pytest.raises(ManifestError, match="not one of"):
        load_manifest(_write(tmp_path, document))


def test_status_vocabulary_matches_the_plan() -> None:
    assert ALLOWED_STATUS == {
        "pending",
        "passed",
        "failed",
        "blocked",
        "paused",
        "deferred_not_claimed",
    }


def test_passed_without_evidence_is_a_violation(tmp_path: Path, document: dict) -> None:
    document = copy.deepcopy(document)
    entry = document["requirements"][0]
    entry.update(
        status="passed",
        implementation_refs=["qdgrasp/roadmap/requirements.py"],
        test_ids=["tests/contactrich_active/test_completeness_ledger.py"],
        evidence_refs=[],
    )
    verdict = audit_closure(
        load_manifest(_write(tmp_path, document)), repo_root=REPO_ROOT, worktree_dirty=False
    )
    assert verdict.verdict == "FAIL"
    assert any("passed without evidence_refs" in v for v in verdict.violations)


def test_passed_with_a_nonexistent_ref_is_a_violation(tmp_path: Path, document: dict) -> None:
    document = copy.deepcopy(document)
    entry = document["requirements"][0]
    entry.update(
        status="passed",
        implementation_refs=["qdgrasp/does_not_exist.py"],
        test_ids=["tests/contactrich_active/test_completeness_ledger.py"],
        evidence_refs=["evidence/phase3_4_3/nope.json"],
    )
    verdict = audit_closure(
        load_manifest(_write(tmp_path, document)), repo_root=REPO_ROOT, worktree_dirty=False
    )
    assert verdict.verdict == "FAIL"
    assert any("refs do not exist" in v for v in verdict.violations)


def test_passed_on_a_dirty_worktree_is_a_violation(tmp_path: Path, document: dict) -> None:
    document = copy.deepcopy(document)
    entry = document["requirements"][0]
    entry.update(
        status="passed",
        implementation_refs=["qdgrasp/roadmap/requirements.py"],
        test_ids=["tests/contactrich_active/test_completeness_ledger.py"],
        evidence_refs=["docs/roadmap/phase3_4_3_requirements.yaml"],
    )
    verdict = audit_closure(
        load_manifest(_write(tmp_path, document)), repo_root=REPO_ROOT, worktree_dirty=True
    )
    assert verdict.verdict == "FAIL"
    assert any("dirty worktree" in v for v in verdict.violations)


def test_untracked_mapping_target_is_reported(tmp_path: Path, document: dict) -> None:
    document = copy.deepcopy(document)
    document["requirements"][0]["mapped_to"] = ["G99"]
    verdict = audit_closure(
        load_manifest(_write(tmp_path, document)), repo_root=REPO_ROOT, worktree_dirty=False
    )
    assert verdict.unknown
    assert verdict.verdict == "FAIL"


def test_deferred_item_may_not_carry_evidence(tmp_path: Path, document: dict) -> None:
    document = copy.deepcopy(document)
    document["requirements"][0].update(
        status="deferred_not_claimed",
        required=False,
        blocker_reason="optional strategy not implemented",
        evidence_refs=["docs/roadmap/phase3_4_3_requirements.yaml"],
    )
    verdict = audit_closure(
        load_manifest(_write(tmp_path, document)), repo_root=REPO_ROOT, worktree_dirty=False
    )
    assert any("deferral is not coverage" in v for v in verdict.violations)


def test_a_required_item_cannot_be_deferred(tmp_path: Path, document: dict) -> None:
    # Deferral is an allowed disposition for an optional package -- the plan
    # says so for MPPI -- but dropping something the contract requires needs a
    # revision, not a status change.
    document = copy.deepcopy(document)
    document["requirements"][0].update(
        status="deferred_not_claimed",
        required=True,
        blocker_reason="we ran out of time",
    )
    verdict = audit_closure(
        load_manifest(_write(tmp_path, document)), repo_root=REPO_ROOT, worktree_dirty=False
    )
    assert verdict.verdict == "FAIL"
    assert any("while still required" in v for v in verdict.violations)


def test_an_optional_deferred_item_does_not_block_closure(tmp_path: Path, document: dict) -> None:
    document = copy.deepcopy(document)
    for entry in document["requirements"]:
        entry.update(
            status="passed",
            implementation_refs=["qdgrasp/roadmap/requirements.py"],
            test_ids=["tests/contactrich_active/test_completeness_ledger.py"],
            evidence_refs=["docs/roadmap/phase3_4_3_requirements.yaml"],
            blocker_reason="",
        )
    document["requirements"][0].update(
        status="deferred_not_claimed",
        required=False,
        evidence_refs=[],
        blocker_reason="optional in the plan and not implemented",
    )
    verdict = audit_closure(
        load_manifest(_write(tmp_path, document)), repo_root=REPO_ROOT, worktree_dirty=False
    )
    assert verdict.violations == ()
    assert verdict.verdict == "PASS"


def test_historical_three_hand_state_cannot_be_relabelled(tmp_path: Path, document: dict) -> None:
    document = copy.deepcopy(document)
    document["scope"]["historical_p3_4_state"] = "pass"
    verdict = audit_closure(
        load_manifest(_write(tmp_path, document)), repo_root=REPO_ROOT, worktree_dirty=False
    )
    assert verdict.verdict == "FAIL"
    assert any("paused_by_ADR-0008" in v for v in verdict.violations)


def test_three_hand_coverage_claim_is_refused(tmp_path: Path, document: dict) -> None:
    document = copy.deepcopy(document)
    document["scope"]["three_hand_coverage"] = True
    verdict = audit_closure(
        load_manifest(_write(tmp_path, document)), repo_root=REPO_ROOT, worktree_dirty=False
    )
    assert verdict.verdict == "FAIL"


def test_blocked_requirement_needs_a_reason(tmp_path: Path, document: dict) -> None:
    document = copy.deepcopy(document)
    document["requirements"][0].update(status="blocked", blocker_reason="")
    verdict = audit_closure(
        load_manifest(_write(tmp_path, document)), repo_root=REPO_ROOT, worktree_dirty=False
    )
    assert any("without blocker_reason" in v for v in verdict.violations)


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )


def test_closure_gate_does_not_exit_zero_while_incomplete() -> None:
    completed = _run("scripts/check_phase3_4_3.py", "--skip-tests")
    payload = json.loads(completed.stdout)
    # A dirty worktree is itself a FAIL, because a passed claim on uncommitted
    # code cannot be reproduced from the commit under review; either way the
    # invariant is that an open ledger never exits 0.
    assert completed.returncode in {1, 3}, completed.stderr
    assert payload["verdict"] in {"FAIL", "INCOMPLETE", "BLOCKED"}
    assert payload["release_blocked"] is True
    assert payload["release_verdict"] == "none"
    assert payload["three_hand_coverage"] is False


def test_clean_tree_with_open_requirements_reads_as_incomplete(manifest) -> None:
    verdict = audit_closure(manifest, repo_root=REPO_ROOT, worktree_dirty=False)
    assert verdict.verdict == "INCOMPLETE"
    assert verdict.exit_code == 3
    assert verdict.violations == ()


def test_partial_phase_3_4_gate_does_not_exit_zero() -> None:
    completed = _run("scripts/check_phase3_4.py", "--skip-tests")
    assert completed.returncode == 3, completed.stderr


def test_historical_status_command_reports_paused() -> None:
    completed = _run("scripts/check_phase3_4.py", "--command", "historical-status")
    assert completed.returncode == 2, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["status"] == "PAUSED"
    assert payload["three_hand_coverage"] is False
