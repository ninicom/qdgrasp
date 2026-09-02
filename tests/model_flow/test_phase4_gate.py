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
TARGET_VALIDITY_FIELDS = {
    "kinematics_valid",
    "pose_target_valid",
    "joint_target_valid",
    "fk_target_valid",
}


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


def test_the_micro_gate_executes_both_hands_under_the_v3_supervision_contract(gate) -> None:
    """Synthetic measured targets carry validity masks through backward."""

    results = gate._micro_checks()
    assert len(results) == 2
    assert all(item.status == gate.STATUS_DELIVERED for item in results), [item.detail for item in results]


def test_the_cuda_fixture_executes_a_training_step_under_the_v3_supervision_contract(cuda_gate) -> None:
    """The GPU fixture must be trainable before scarce CUDA time is booked."""

    from qdgrasp.models.config import FlowModelSettings, QDGraspFlow
    from qdgrasp.robot.spec import RobotSpec

    robot = RobotSpec.from_config("leap_hand.yaml", sample_anchors=False)
    batch = cuda_gate._fixture(robot, samples=2, points=64, seed=0, device=torch.device("cpu"))
    assert TARGET_VALIDITY_FIELDS <= set(batch)
    for field in TARGET_VALIDITY_FIELDS:
        assert batch[field].dtype == torch.bool
        assert batch[field].shape == (2,)
        assert bool(batch[field].all())

    loss = QDGraspFlow(FlowModelSettings(), robot).training_step(batch)
    loss.backward()
    assert torch.isfinite(loss)


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


def test_cuda_evidence_is_blocked_until_a_record_measures_something(gate, tmp_path: Path) -> None:
    outstanding = {item.package: item for item in gate._outstanding(tmp_path)}
    assert outstanding["P4-11b"].status == gate.STATUS_BLOCKED
    assert "ADR-0006" in outstanding["P4-11b"].detail
    # An empty file named like a record is not a record.
    (tmp_path / "evidence/phase4").mkdir(parents=True)
    (tmp_path / "evidence/phase4/cuda-a100-20260901.json").write_text("{}", encoding="utf-8")
    still = {item.package: item for item in gate._outstanding(tmp_path)}["P4-11b"]
    assert still.status == gate.STATUS_BLOCKED
    assert "cuda-a100-20260901.json" in still.detail


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


def _cuda_record(**overrides) -> dict:
    record = {
        "verdict": "measured",
        "device": {"cuda": True, "name": "NVIDIA T4"},
        "hands": [
            {"robot": "leap_hand.yaml", "passed": True},
            {"robot": "wonik_allegro.yaml", "passed": True},
        ],
    }
    record.update(overrides)
    return record


def _write_cuda(tmp_path: Path, name: str, record: dict) -> Path:
    directory = tmp_path / "evidence/phase4"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_text(json.dumps(record), encoding="utf-8")
    return path


def test_a_measured_two_hand_record_satisfies_the_cuda_gate(gate, tmp_path: Path) -> None:
    _write_cuda(tmp_path, "cuda-t4-20260901.json", _cuda_record())
    measured, rejected = gate._cuda_records(tmp_path)
    assert [path.name for path in measured] == ["cuda-t4-20260901.json"]
    assert rejected == []
    row = {item.package: item for item in gate._outstanding(tmp_path)}["P4-11b"]
    assert row.status == gate.STATUS_DELIVERED


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({"verdict": "refused"}, "verdict"),
        ({"verdict": "failed"}, "verdict"),
        ({"device": {"cuda": False}}, "device.cuda"),
        ({"hands": []}, "no hand was measured"),
        ({"hands": [{"robot": "leap_hand.yaml", "passed": False}]}, "did not pass"),
        ({"hands": [{"robot": "leap_hand.yaml", "passed": True}]}, "of 2 active hands"),
    ],
)
def test_a_record_that_did_not_measure_both_hands_is_not_evidence(gate, tmp_path: Path, overrides, expected) -> None:
    """The file existing is not the evidence; what it says is.

    The harness writes a record on every run, refusals included, so a gate that
    counted files would be satisfied by the machine that cannot run it.
    """

    _write_cuda(tmp_path, "cuda-something.json", _cuda_record(**overrides))
    measured, rejected = gate._cuda_records(tmp_path)
    assert measured == []
    assert len(rejected) == 1 and expected in rejected[0]
    row = {item.package: item for item in gate._outstanding(tmp_path)}["P4-11b"]
    assert row.status == gate.STATUS_BLOCKED
    assert "cuda-something.json" in row.detail


def test_an_unreadable_cuda_record_is_named_not_skipped(gate, tmp_path: Path) -> None:
    directory = tmp_path / "evidence/phase4"
    directory.mkdir(parents=True)
    (directory / "cuda-truncated.json").write_text("{not json", encoding="utf-8")
    measured, rejected = gate._cuda_records(tmp_path)
    assert measured == []
    assert "unreadable" in rejected[0]


def test_the_refusal_record_this_machine_produced_does_not_count(gate) -> None:
    """The record committed from the development machine must stay inert."""

    record = json.loads(
        (REPO_ROOT / "evidence/phase4/cuda-refused-devmachine-20260831.json").read_text(encoding="utf-8")
    )
    assert record["verdict"] == "refused"
    assert "hands" not in record
    assert record["hardware_probe"]["torch_device_count"] == 0
    measured, rejected = gate._cuda_records(REPO_ROOT)
    assert measured == []
    assert any("cuda-refused-devmachine" in item for item in rejected)


# -- P4-12: the reviewer's mechanical check --------------------------------


@pytest.fixture(scope="module")
def packet_script():
    return _load("phase4_review_packet")


def test_verify_refuses_the_committed_packet_now_that_its_sources_moved(packet_script) -> None:
    """The one command a reviewer runs before signing, answering honestly.

    The corrective track of ``PLAN.md`` §9 changed the joint parameterization,
    the quality objective and the tokeniser this packet was signed over, and §9.1
    records the consequence: the Phase 4 evidence is superseded for release and
    has to be produced again, not re-signed. So the packet must now be refused,
    and it must be refused *by name* -- a verifier that shrugged at a moved
    source would let the old numbers keep vouching for new code.

    ``R8`` regenerates this packet after the remaining gates close; until then a
    green result here would be the finding, not the fix.
    """

    path = REPO_ROOT / "evidence/phase4/review/review-packet.json"
    ok, findings = packet_script.verify_packet(path)
    drifted = [item for item in findings if "does not match the packet's" in item]

    assert not ok
    assert drifted, f"the packet was signed over code that has since changed; findings were {findings}"
    assert any("qdgrasp/models/flow.py" in item or "qdgrasp/models/tokenizer.py" in item for item in drifted)


def test_verify_accepts_a_packet_built_from_the_tree_it_describes(packet_script, tmp_path: Path) -> None:
    """The mechanical check still passes on a packet that matches its sources."""

    packet = packet_script.build_packet()
    target = tmp_path / "review-packet.json"
    target.write_text(json.dumps(packet, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    ok, findings = packet_script.verify_packet(target)
    # Whether the tree is clean and which commit is recorded are separate
    # statements from "the bytes this packet describes are the bytes on disk",
    # and only the last one is what this test is about.
    mismatches = [
        item
        for item in findings
        if not item.startswith("packet records commit") and "working tree is dirty" not in item
    ]
    assert not mismatches, mismatches
    assert ok == (findings == [])


def test_verify_refuses_a_packet_whose_artifact_moved(packet_script, tmp_path: Path) -> None:
    source = REPO_ROOT / "evidence/phase4/review/review-packet.json"
    packet = json.loads(source.read_text(encoding="utf-8"))
    packet["artifacts"][0]["sha256"] = "0" * 64
    forged = tmp_path / "review-packet.json"
    forged.write_text(json.dumps(packet), encoding="utf-8")
    ok, findings = packet_script.verify_packet(forged)
    assert not ok
    assert any("does not match the packet" in item for item in findings)


def test_verify_refuses_a_packet_that_already_carries_a_verdict(packet_script, tmp_path: Path) -> None:
    """A packet is the material a verdict is written against, not its container."""

    source = REPO_ROOT / "evidence/phase4/review/review-packet.json"
    packet = json.loads(source.read_text(encoding="utf-8"))
    packet["verdict"] = "pass"
    forged = tmp_path / "review-packet.json"
    forged.write_text(json.dumps(packet), encoding="utf-8")
    ok, findings = packet_script.verify_packet(forged)
    assert not ok
    assert any("already carries a verdict" in item for item in findings)


def test_verify_refuses_a_packet_whose_digest_was_edited(packet_script, tmp_path: Path) -> None:
    source = REPO_ROOT / "evidence/phase4/review/review-packet.json"
    packet = json.loads(source.read_text(encoding="utf-8"))
    packet["claim"] = "the model achieves state of the art"
    forged = tmp_path / "review-packet.json"
    forged.write_text(json.dumps(packet), encoding="utf-8")
    ok, findings = packet_script.verify_packet(forged)
    assert not ok
    assert any("does not match its own contents" in item for item in findings)
