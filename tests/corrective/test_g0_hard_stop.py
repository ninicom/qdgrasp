"""G0: the public path stops on a corpus that fails its own checks.

``PLAN.md`` §9.1 records the state this gate exists for: the canonical audit
fails on three source hashes, and the locked protocol's train split holds one
positive for LEAP and two for Allegro against a floor of twenty-five.  Both
numbers were available before this change, and the public facade trained on that
corpus anyway.  The distance between "a script reports it" and "the run cannot
start" is the whole finding.

Unlike the characterization tests beside them, these are ordinary regression
tests: R1 delivers this behaviour, so it passes now and must keep passing.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from qdgrasp.corrective import CorrectiveGateError, evaluate, gate, registry

REPO_ROOT = Path(__file__).resolve().parents[2]
CORPUS_CONFIG = REPO_ROOT / "configs" / "data" / "dgn_open_tiny.yaml"


def _corpus_config():
    from qdgrasp.config import load_data_config

    return load_data_config(CORPUS_CONFIG)


def test_public_training_will_not_start_on_the_current_corpus() -> None:
    from qdgrasp.api import QDGrasp

    with pytest.raises(CorrectiveGateError) as error:
        QDGrasp().train(CORPUS_CONFIG, max_steps=1, run_name="corrective-blocked", project_dir="runs")

    message = str(error.value)
    assert "canonical_dataset_audit" in message
    assert "phase5_positive_gate" in message


def test_public_validation_will_not_start_on_the_current_corpus() -> None:
    from qdgrasp.api import QDGrasp

    with pytest.raises(CorrectiveGateError):
        QDGrasp().val(CORPUS_CONFIG, batch_size=2)


def test_the_report_names_the_failing_check_and_its_reason() -> None:
    report = evaluate(_corpus_config(), purpose="training")

    assert report.gated
    assert report.dataset_id == "dgn-open-tiny-v1"
    assert not report.allowed
    checks = {check.name: check for check in report.checks}
    assert set(checks) == {"canonical_dataset_audit", "phase5_positive_gate"}

    audit = checks["canonical_dataset_audit"]
    assert audit.failed
    assert any(word in audit.detail for word in ("drift", "mismatch", "provenance")), audit.detail

    # One root cause is reported once: an unverifiable corpus is not counted.
    positive = checks["phase5_positive_gate"]
    assert positive.status == "skip"
    assert "canonical audit" in positive.detail


def test_the_positive_gate_reports_the_floor_on_a_corpus_it_can_verify(verified_corpus) -> None:
    """The other half of the stop, measured where the audit does not block it."""

    report = evaluate({"dataset_root": str(verified_corpus)}, purpose="training")

    checks = {check.name: check for check in report.checks}
    assert checks["canonical_dataset_audit"].status == "pass"
    assert checks["phase5_positive_gate"].failed
    assert "floor" in checks["phase5_positive_gate"].detail
    assert not report.allowed


def test_a_dataset_that_claims_no_provenance_is_not_gated() -> None:
    """A synthetic fixture has nothing to verify and nothing it could mislabel."""

    from qdgrasp.api import QDGrasp

    result = QDGrasp().train("dummy-tiny.yaml", max_steps=2, run_name="corrective-ungated", project_dir="runs")
    assert result.global_step == 2

    grasper = QDGrasp()
    grasper.train("dummy-tiny.yaml", max_steps=1, run_name="corrective-ungated", project_dir="runs")
    assert grasper.gate_report is not None
    assert not grasper.gate_report.gated


def test_the_gate_has_no_environment_override() -> None:
    """A gate with a documented bypass is a warning, and §9.2 forbids warnings here."""

    source = inspect.getsource(gate)
    for bypass in ("os.environ", "getenv", "QDGRASP_SKIP", "force=True"):
        assert bypass not in source, f"the corrective gate exposes a bypass through {bypass}"


def test_no_run_may_claim_release_evidence_while_findings_are_open() -> None:
    report = evaluate({"dataset_root": "does-not-exist"}, purpose="training")

    assert registry.release_is_blocked()
    assert not report.release_evidence_allowed
