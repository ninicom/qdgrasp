"""WRK-R3 mutation corpus: every way a forged pass must fail.

RRV-01 found the closure runner taking evidence at its word. These are the
forgeries it has to refuse. Each one is a single mutation of an otherwise valid
artifact, so a test that goes green tells you exactly which check stopped
caring.
"""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = REPO_ROOT / "evidence" / "phase3_4_3" / "s10" / "kaggle-run-v8" / "cuda-gate.json"


def _closure():
    spec = importlib.util.spec_from_file_location(
        "check_phase3_4_3", REPO_ROOT / "scripts" / "check_phase3_4_3.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _passing_cuda_payload() -> dict:
    """A bundle whose metrics genuinely say pass, built from the real one."""
    payload = copy.deepcopy(json.loads(EVIDENCE.read_text(encoding="utf-8")))
    payload["verdict"] = "PASS"
    payload["parity"]["single_contact"]["passed"] = True
    payload["parity"]["passed"] = True
    payload["sanitizer"]["clean"] = True
    payload["sanitizer"]["tools"]["initcheck"]["clean"] = True
    payload["performance"]["passed"] = True
    for metrics in payload["performance"]["hands"].values():
        metrics["rejected_worlds"] = 0
        metrics["overflow_worlds"] = 0
    return payload


def test_the_real_bundle_recomputes_to_fail():
    closure = _closure()
    payload = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    verdict, problems = closure.recompute_cuda_verdict(payload)
    assert verdict == "FAIL"
    assert problems


def test_a_forged_verdict_does_not_survive_recomputation():
    """The whole of RRV-01 in one case: declaring PASS must not produce one."""
    closure = _closure()
    payload = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    payload["verdict"] = "PASS"
    verdict, problems = closure.recompute_cuda_verdict(payload)
    assert verdict == "FAIL"
    assert problems


@pytest.mark.parametrize(
    "mutate,expected",
    [
        (lambda p: p["parity"]["no_contact"].__setitem__("passed", False), "no_contact"),
        (lambda p: p["sanitizer"].__setitem__("clean", False), "sanitizer"),
        (lambda p: p["sanitizer"]["tools"].pop("racecheck"), "racecheck"),
        (lambda p: p["capability"].__setitem__("contact_force_readable", False), "contact force"),
        (lambda p: p["capability"]["overflow_telemetry"].__setitem__("buffer_overflow", True), "overflow"),
        (lambda p: p["performance"]["hands"]["leap_hand"].__setitem__("speedup_met", False), "speedup"),
        (lambda p: p["performance"]["hands"]["leap_hand"].__setitem__("rejected_worlds", 3), "non-finite"),
        (lambda p: p.__setitem__("three_hand_coverage", True), "three-hand"),
    ],
)
def test_each_metric_mutation_is_caught(mutate, expected):
    closure = _closure()
    payload = _passing_cuda_payload()
    assert closure.recompute_cuda_verdict(payload)[0] == "PASS", "fixture must start clean"
    mutate(payload)
    verdict, problems = closure.recompute_cuda_verdict(payload)
    assert verdict == "FAIL"
    assert any(expected in problem for problem in problems), problems


def test_missing_raw_log_is_a_problem(tmp_path):
    closure = _closure()
    payload = _passing_cuda_payload()
    path = tmp_path / "cuda-gate.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    result = closure.verify_external_evidence(path, expected_commit=payload["commit"])
    assert not result["passed"]
    assert any("raw_log_sha256" in problem for problem in result["problems"])


def test_wrong_schema_is_refused(tmp_path):
    closure = _closure()
    payload = _passing_cuda_payload()
    payload["schema"] = "qdgrasp/evidence/something-else/v1"
    path = tmp_path / "cuda-gate.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    result = closure.verify_external_evidence(path, expected_commit=payload["commit"])
    assert not result["passed"]
    assert any("schema" in problem for problem in result["problems"])


def _verdict_file(tmp_path: Path, **overrides) -> Path:
    payload = {
        "reviewer_verdict": "PASS",
        "reviewer": "an-independent-reviewer",
        "author": "claude-implementation-agent",
        "candidate_commit": "a" * 40,
        "packet_sha256": "b" * 64,
        "open_findings": {"S0": 0, "S1": 0, "S2": 0, "S3": 0},
    }
    payload.update(overrides)
    path = tmp_path / "verdict.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _packet_file(tmp_path: Path) -> tuple[Path, str]:
    from qdgrasp.roadmap.review_packet import canonical_digest

    packet = {"phase": "3.4.3", "commit": "a" * 40, "assembled_at": "now", "packet_sha256": "stale"}
    path = tmp_path / "review-packet.json"
    path.write_text(json.dumps(packet), encoding="utf-8")
    return path, canonical_digest(packet)


def test_a_correctly_signed_verdict_passes(tmp_path):
    closure = _closure()
    packet_path, digest = _packet_file(tmp_path)
    verdict = _verdict_file(tmp_path, packet_sha256=digest)
    result = closure.verify_review_packet(
        verdict, expected_commit="a" * 40, packet_path=packet_path
    )
    assert result["passed"], result["problems"]


@pytest.mark.parametrize(
    "overrides,expected",
    [
        ({"reviewer_verdict": "FAIL"}, "not PASS"),
        ({"reviewer": "claude-implementation-agent"}, "must not be the author"),
        ({"reviewer": ""}, "names no reviewer"),
        ({"candidate_commit": "c" * 40}, "not the candidate"),
        ({"packet_sha256": ""}, "signs no packet digest"),
        ({"open_findings": {"S0": 0, "S1": 0, "S2": 1, "S3": 0}}, "open S2"),
        ({"open_findings": {"S0": 0, "S1": 0, "S2": 0, "S3": 2}}, "open S3"),
    ],
)
def test_each_verdict_forgery_is_refused(tmp_path, overrides, expected):
    closure = _closure()
    packet_path, digest = _packet_file(tmp_path)
    payload = {"packet_sha256": digest}
    payload.update(overrides)
    verdict = _verdict_file(tmp_path, **payload)
    result = closure.verify_review_packet(
        verdict, expected_commit="a" * 40, packet_path=packet_path
    )
    assert not result["passed"]
    assert any(expected in problem for problem in result["problems"]), result["problems"]


def test_a_signature_over_a_different_packet_is_refused(tmp_path):
    """The digest has to bind to the packet actually on disk."""
    closure = _closure()
    packet_path, _ = _packet_file(tmp_path)
    verdict = _verdict_file(tmp_path, packet_sha256="d" * 64)
    result = closure.verify_review_packet(
        verdict, expected_commit="a" * 40, packet_path=packet_path
    )
    assert not result["passed"]
    assert any("digests to" in problem for problem in result["problems"])


def test_an_absent_verdict_is_not_a_pass():
    closure = _closure()
    result = closure.verify_review_packet(None, expected_commit="a" * 40)
    assert not result["passed"]


def test_release_profile_refuses_skip_tests():
    """A release verdict cannot rest on suites nobody ran."""
    import subprocess
    import sys

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/check_phase3_4_3.py",
            "--profile",
            "release",
            "--skip-tests",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode != 0
    assert "skip-tests" in (completed.stdout + completed.stderr)


def test_the_ledger_rejects_a_test_path_that_does_not_exist(tmp_path):
    """A passed claim pointing at a nonexistent test is not evidence of anything."""
    import yaml

    from qdgrasp.roadmap import audit_closure, load_manifest

    source = REPO_ROOT / "docs" / "roadmap" / "phase3_4_3_requirements.yaml"
    document = yaml.safe_load(source.read_text(encoding="utf-8"))

    forged = tmp_path / "manifest.yaml"
    raw = source.read_text(encoding="utf-8")
    raw = raw.replace(
        "test_ids: [tests/contactrich_active/test_cuda_gate_harness.py]",
        "test_ids: [tests/contactrich_active/test_a_file_that_is_not_there.py]",
        1,
    )
    forged.write_text(raw, encoding="utf-8")
    assert document is not None

    manifest = load_manifest(forged)
    result = audit_closure(manifest, repo_root=REPO_ROOT)
    assert result.violations, "a missing test path must invalidate the claim"
