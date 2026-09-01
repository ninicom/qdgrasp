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
    # Both checks are named whichever one failed, so a reader can see what was
    # asked as well as what refused.
    assert "canonical_dataset_audit" in message
    assert "phase5_positive_gate" in message


def test_public_validation_will_not_start_on_the_current_corpus() -> None:
    from qdgrasp.api import QDGrasp

    with pytest.raises(CorrectiveGateError):
        QDGrasp().val(CORPUS_CONFIG, batch_size=2)


def test_the_report_names_the_failing_check_and_its_reason() -> None:
    """R8 regenerated the corpus, so the audit passes and the floor does not.

    The stop has two halves and they fail at different times.  Provenance is now
    intact -- the dataset was produced from a recorded clean commit and its
    sources hash to what is on disk -- and the corpus still holds five positives
    where the protocol's train split needs twenty-five per hand.  Training is
    refused for that reason alone, and the report says which reason.
    """

    report = evaluate(_corpus_config(), purpose="training")

    assert report.gated
    assert report.dataset_id == "dgn-open-tiny-v1"
    assert not report.allowed
    checks = {check.name: check for check in report.checks}
    assert set(checks) == {"canonical_dataset_audit", "phase5_positive_gate"}

    assert checks["canonical_dataset_audit"].status == "pass"

    positive = checks["phase5_positive_gate"]
    assert positive.failed
    assert "floor" in positive.detail


def test_a_corpus_whose_sources_drifted_is_not_counted(tmp_path: Path) -> None:
    """One root cause, reported once: an unverifiable corpus is not measured.

    Source drift is how a dataset stops describing the code that made it, and
    once that is true its sample counts are about whatever bytes are on disk.
    The positive gate therefore declines to measure rather than reporting a
    number beside a failed audit.
    """

    import json
    import shutil

    root = tmp_path / "datasets" / "drifted"
    shutil.copytree(REPO_ROOT / "datasets" / "dgn-open-tiny", root)
    manifest_path = root / "dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    # The sources have to be present for "drifted" to mean drifted rather than
    # "missing": the audit resolves them relative to the project the dataset
    # sits in.
    for name in manifest["generator_source_hashes"]:
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO_ROOT / name, target)
    drifted = min(manifest["generator_source_hashes"])
    manifest["generator_source_hashes"][drifted] = "0" * 64
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    report = evaluate({"dataset_root": str(root)}, purpose="training")
    checks = {check.name: check for check in report.checks}

    assert checks["canonical_dataset_audit"].failed
    assert "mismatch" in checks["canonical_dataset_audit"].detail
    assert checks["phase5_positive_gate"].status == "skip"
    assert "canonical audit" in checks["phase5_positive_gate"].detail
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
