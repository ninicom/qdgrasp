"""P4-11: the gate and the CUDA harness, including what they refuse.

A gate is only worth running if it can fail, so most of these tests are about
the failure paths: a phase that is not finished must not report zero, and a CPU
run must never be able to present itself as CUDA evidence.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load(name: str):
    path = REPO_ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"_qdgrasp_script_{name}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def gate():
    return _load("check_phase4")


@pytest.fixture(scope="module")
def cuda_gate():
    return _load("phase4_cuda_gate")


def test_the_gate_fails_while_packages_are_open(gate) -> None:
    """P4-11b and P4-12 cannot be closed from here, so the exit code is 1."""

    assert gate.main(["--profile", "contract", "--root", str(REPO_ROOT)]) == 1
    results = gate.run_checks("contract", REPO_ROOT)
    outstanding = {item.package for item in results if not item.passed}
    assert outstanding == {"P4-11b", "P4-12"}


def test_every_declared_module_and_preset_is_present(gate) -> None:
    results = {item.package: item for item in gate.run_checks("contract", REPO_ROOT)}
    for package in ("P4-01", "P4-02", "P4-03", "P4-04/05/07", "P4-06/09", "P4-08"):
        assert results[package].status == gate.STATUS_DELIVERED, results[package].detail


def test_a_cuda_labelled_file_is_refused_from_the_cpu_evidence_slot(gate, tmp_path: Path) -> None:
    """The one way a CPU gate could launder a CUDA claim, closed off."""

    evidence = tmp_path / "evidence/phase4"
    evidence.mkdir(parents=True)
    (evidence / "overfit-leap-cpu.json").write_text(
        json.dumps({"converged": True, "cuda": True, "last": {}}), encoding="utf-8"
    )
    results = gate._evidence(tmp_path)
    assert results[0].status == gate.STATUS_OPEN
    assert "claims cuda=true" in results[0].detail


def test_a_missing_overfit_record_is_open_not_silently_passed(gate, tmp_path: Path) -> None:
    results = gate._evidence(tmp_path)
    assert results[0].status == gate.STATUS_OPEN


def test_cuda_evidence_is_blocked_until_a_record_exists(gate, tmp_path: Path) -> None:
    outstanding = {item.package: item for item in gate._outstanding(tmp_path)}
    assert outstanding["P4-11b"].status == gate.STATUS_BLOCKED
    assert "ADR-0006" in outstanding["P4-11b"].detail
    (tmp_path / "evidence/phase4").mkdir(parents=True)
    (tmp_path / "evidence/phase4/cuda-a100-20260901.json").write_text("{}", encoding="utf-8")
    assert gate._outstanding(tmp_path)[1].status == gate.STATUS_DELIVERED


def test_the_review_verdict_stays_blocked_because_the_author_may_not_sign(gate) -> None:
    outstanding = {item.package: item for item in gate._outstanding(REPO_ROOT)}
    assert outstanding["P4-12"].status == gate.STATUS_BLOCKED
    assert "may not sign it" in outstanding["P4-12"].detail
    assert "review packet and reviewer guide prepared" in outstanding["P4-12"].detail


def test_the_cuda_harness_refuses_a_cpu_device(cuda_gate) -> None:
    with pytest.raises(cuda_gate.CudaGateError, match="ADR-0006"):
        cuda_gate._require_cuda("cpu")


@pytest.mark.skipif(torch.cuda.is_available(), reason="this asserts the no-CUDA refusal")
def test_the_cuda_harness_refuses_a_cuda_label_without_cuda(cuda_gate) -> None:
    with pytest.raises(cuda_gate.CudaGateError, match="must not be labelled a GPU run"):
        cuda_gate._require_cuda("cuda:0")


@pytest.mark.skipif(torch.cuda.is_available(), reason="this asserts the no-CUDA refusal")
def test_a_refused_run_writes_a_refusal_and_exits_nonzero(cuda_gate, tmp_path: Path) -> None:
    report = tmp_path / "phase4_cuda_evidence.json"
    assert cuda_gate.main(["--device", "cuda:0", "--evidence", str(report)]) == 1
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["verdict"] == "refused"
    assert "hands" not in payload, "a refused run must not carry measurements"
    assert payload["scope"]["paused_hands"] == ["shadow_hand"]
    assert payload["scope"]["selected_hands"] == ["leap_hand", "wonik_allegro"]


def test_the_notebook_pins_a_commit_and_says_why() -> None:
    notebook = json.loads((REPO_ROOT / "notebooks/phase4_cuda_gate.ipynb").read_text(encoding="utf-8"))
    sources = ["".join(cell["source"]) for cell in notebook["cells"]]
    setup = next(text for text in sources if "CODE_REVISION" in text)
    assert "REPLACE_WITH_PUSHED_COMMIT" not in setup.split("assert")[0].split("=")[1].split("\n")[0], (
        "the notebook still ships the placeholder revision; pin a pushed commit"
    )
    assert "rev-parse" in setup, "the notebook must verify the checkout landed on the pinned commit"
    assert any("ADR-0006" in text for text in sources)
