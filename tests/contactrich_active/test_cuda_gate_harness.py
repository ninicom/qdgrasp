"""S10 — the parts of the CUDA gate that can be checked without a device.

The gate itself has to run on a real NVIDIA device, and this machine has none.
What can be checked here is everything that decides *whether a result counts*:
the exit codes, the refusal to run on a CPU host, the resource estimate, the
atomic checkpoint, and the thresholds the plan pinned.

**B-08** is pinned here. The old harness read ``peak_vram_gib`` on its own
success path -- a key it never set -- so a passing run crashed; and when the
benchmark had not run at all it fell through to the success message. Both are
paths that only execute when things go *right*, which is why nobody hit them.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
GATE = REPO_ROOT / "scripts" / "check_phase3_4_3_cuda.py"
OLD_HARNESS = REPO_ROOT / "scripts" / "phase3_4_cuda_contact_search.py"
NOTEBOOK_DIR = REPO_ROOT / "kaggle-phase3-4-3"


def run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(GATE), *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )


# -- refusing a CPU host --------------------------------------------------


def test_a_cpu_host_can_never_produce_cuda_evidence() -> None:
    completed = run("--device", "cuda:0")
    assert completed.returncode != 0
    payload = json.loads(completed.stdout)
    assert payload["verdict"] in {"FAIL", "CONFIG_ERROR"}
    assert "not CUDA evidence" in json.dumps(payload)


def test_a_non_cuda_device_is_refused() -> None:
    completed = run("--device", "cpu")
    assert completed.returncode != 0


# -- thresholds are pinned, not negotiated --------------------------------


def test_the_pinned_thresholds_match_the_plan() -> None:
    import importlib.util

    spec = importlib.util.spec_from_file_location("cuda_gate", GATE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert module.MIN_GPU_SPEEDUP == 2.0
    assert module.VRAM_BUDGET_GIB == 14.0
    assert module.MIN_SIMULTANEOUS_WORLDS == 64
    # The operating point is chosen before the run, not after seeing it.
    assert module.BENCHMARK_WORLDS >= module.MIN_SIMULTANEOUS_WORLDS
    assert module.BENCHMARK_RUNS >= 3


def test_a_world_count_below_the_declared_floor_is_a_config_error() -> None:
    completed = run("--worlds", "8")
    assert completed.returncode == 4
    payload = json.loads(completed.stdout)
    assert payload["verdict"] == "CONFIG_ERROR"
    assert "below the declared floor" in payload["error"]


# -- resource estimate before the run -------------------------------------


def test_dry_run_reports_the_cost_before_anything_is_allocated() -> None:
    completed = run("--dry-run", "--worlds", "1024")
    assert completed.returncode == 0
    payload = json.loads(completed.stdout)
    assert payload["worlds"] == 1024
    assert payload["world_steps"] == payload["worlds"] * payload["steps"]
    assert payload["estimated_total_bytes"] > 0
    assert payload["vram_budget_gib"] == 14.0


# -- checkpointing --------------------------------------------------------


def test_a_checkpoint_is_written_atomically(tmp_path: Path) -> None:
    import importlib.util

    spec = importlib.util.spec_from_file_location("cuda_gate", GATE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    target = tmp_path / "nested" / "checkpoint.json"
    digest = module.write_atomically(target, '{"stage": "capability"}')
    assert target.is_file()
    assert len(digest) == 64
    # Nothing partial is left behind for a resume to pick up.
    assert not list(tmp_path.rglob("*.partial"))


def test_a_truncated_checkpoint_is_discarded_not_half_trusted(tmp_path: Path) -> None:
    import importlib.util

    spec = importlib.util.spec_from_file_location("cuda_gate", GATE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    broken = tmp_path / "checkpoint.json"
    broken.write_text('{"stage": "capa', encoding="utf-8")
    assert module.load_checkpoint(broken) == {}
    assert module.load_checkpoint(None) == {}


def test_a_complete_checkpoint_is_resumed(tmp_path: Path) -> None:
    import importlib.util

    spec = importlib.util.spec_from_file_location("cuda_gate", GATE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    path = tmp_path / "checkpoint.json"
    path.write_text(json.dumps({"capability": {"verdict": "supported"}}), encoding="utf-8")
    assert module.load_checkpoint(path) == {"capability": {"verdict": "supported"}}


# -- the defect the old harness carried -----------------------------------


def test_the_old_harness_no_longer_reads_a_key_it_never_sets() -> None:
    source = OLD_HARNESS.read_text(encoding="utf-8")
    # The success path used bench['peak_vram_gib']; the benchmark only ever set
    # device_peak_vram_gib, so a *passing* run raised KeyError (blocker B-08).
    assert "bench['peak_vram_gib']" not in source
    assert "device_peak_vram_gib" in source


def test_a_benchmark_that_did_not_run_is_not_a_benchmark_that_passed() -> None:
    source = OLD_HARNESS.read_text(encoding="utf-8")
    assert 'if bench.get("status") != "measured"' in source


def test_the_gate_success_path_only_reads_keys_it_sets() -> None:
    source = GATE.read_text(encoding="utf-8")
    # The summary line reads speedup and device_peak_vram_gib, both of which
    # run_performance always writes for every hand.
    assert "entry['speedup']" in source
    assert "entry['device_peak_vram_gib']" in source
    assert '"speedup": round(float(speedup), 3)' in source
    assert '"device_peak_vram_gib"' in source


# -- the notebook that runs it --------------------------------------------


@pytest.mark.skipif(
    not (NOTEBOOK_DIR / "qdgrasp-phase-3-4-3-cuda-gate.ipynb").is_file(),
    reason="the notebook has not been built in this checkout",
)
def test_the_notebook_carries_no_credentials_or_private_paths() -> None:
    notebook = json.loads(
        (NOTEBOOK_DIR / "qdgrasp-phase-3-4-3-cuda-gate.ipynb").read_text(encoding="utf-8")
    )
    source = "".join("".join(cell["source"]) for cell in notebook["cells"])
    for forbidden in ("kaggle.json", "KAGGLE_KEY", "/home/", "/run/media", "api_token"):
        assert forbidden not in source, forbidden


@pytest.mark.skipif(
    not (NOTEBOOK_DIR / "qdgrasp-phase-3-4-3-cuda-gate.ipynb").is_file(),
    reason="the notebook has not been built in this checkout",
)
def test_the_notebook_runs_the_prior_gates_and_the_sanitizer() -> None:
    notebook = json.loads(
        (NOTEBOOK_DIR / "qdgrasp-phase-3-4-3-cuda-gate.ipynb").read_text(encoding="utf-8")
    )
    source = "".join("".join(cell["source"]) for cell in notebook["cells"])
    assert "phase1_cuda_smoke.py" in source
    assert "phase2_cuda_fk_parity.py" in source
    assert "compute-sanitizer" in source
    assert "check_phase3_4_3_cuda.py" in source
    # The Warp defect is confronted rather than worked around.
    assert "mujoco-warp==" in source
    assert "initcheck" in source


@pytest.mark.skipif(
    not (NOTEBOOK_DIR / "qdgrasp-phase-3-4-3-cuda-gate.ipynb").is_file(),
    reason="the notebook has not been built in this checkout",
)
def test_the_notebook_pins_an_exact_commit() -> None:
    notebook = json.loads(
        (NOTEBOOK_DIR / "qdgrasp-phase-3-4-3-cuda-gate.ipynb").read_text(encoding="utf-8")
    )
    source = "".join("".join(cell["source"]) for cell in notebook["cells"])
    assert "CODE_REVISION = " in source
    marker = source.split("CODE_REVISION = ")[1].split("\n")[0].strip().strip('"')
    assert len(marker) == 40, marker
    assert all(c in "0123456789abcdef" for c in marker)


@pytest.mark.skipif(
    not (NOTEBOOK_DIR / "kernel-metadata.json").is_file(),
    reason="the notebook has not been built in this checkout",
)
def test_the_kernel_asks_for_a_gpu() -> None:
    metadata = json.loads((NOTEBOOK_DIR / "kernel-metadata.json").read_text(encoding="utf-8"))
    assert metadata["enable_gpu"] is True
    assert metadata["machine_shape"] == "NvidiaTeslaT4"


# -- the closure runner's view of external evidence ------------------------


def _closure_module():
    import importlib.util

    path = REPO_ROOT / "scripts" / "check_phase3_4_3.py"
    spec = importlib.util.spec_from_file_location("closure_gate", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_absent_cuda_evidence_is_not_passed_evidence() -> None:
    module = _closure_module()
    verdict = module.verify_external_evidence(None, expected_commit="deadbeef")
    assert verdict["passed"] is False
    assert "not a passed one" in verdict["detail"]


def test_a_failed_cuda_verdict_is_refused(tmp_path: Path) -> None:
    module = _closure_module()
    path = tmp_path / "cuda.json"
    path.write_text(
        json.dumps(
            {
                "schema": "qdgrasp/evidence/phase3.4.3-cuda/v1",
                "verdict": "FAIL",
                "commit": "deadbeef",
            }
        ),
        encoding="utf-8",
    )
    verdict = module.verify_external_evidence(path, expected_commit="deadbeef")
    assert verdict["passed"] is False
    # WRK-R3: the declared verdict is no longer what decides this. A bundle with
    # no metrics recomputes to FAIL because there is nothing in it that passes.
    assert verdict["computed_verdict"] == "FAIL"
    assert verdict["problems"]


def test_evidence_measuring_different_code_is_refused(tmp_path: Path) -> None:
    module = _closure_module()
    import subprocess

    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=False
    ).stdout.strip()
    old = subprocess.run(
        ["git", "rev-list", "--max-parents=0", "HEAD"],
        cwd=REPO_ROOT, capture_output=True, text=True, check=False,
    ).stdout.strip().splitlines()[0]

    path = tmp_path / "cuda.json"
    path.write_text(
        json.dumps(
            {
                "schema": "qdgrasp/evidence/phase3.4.3-cuda/v1",
                "verdict": "PASS",
                "commit": old,
            }
        ),
        encoding="utf-8",
    )
    verdict = module.verify_external_evidence(path, expected_commit=head)
    assert verdict["passed"] is False
    assert any("does not measure the candidate" in p for p in verdict["problems"])


def test_a_documentation_only_commit_does_not_invalidate_a_measurement() -> None:
    # Comparing bare commit ids would refuse evidence from the commit right
    # before a docs change, which measured exactly the same library.
    module = _closure_module()
    import subprocess

    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=False
    ).stdout.strip()
    matches, detail = module.measured_tree_matches(head, head)
    assert matches, detail
    assert module.MEASURED_PATHS[0] == "qdgrasp"


def test_a_review_packet_from_the_author_is_refused(tmp_path: Path) -> None:
    module = _closure_module()
    path = tmp_path / "review.json"
    path.write_text(
        json.dumps(
            {
                "reviewer": "claude-implementation-agent",
                "author": "claude-implementation-agent",
                "reviewer_verdict": "PASS",
                "open_findings": {},
            }
        ),
        encoding="utf-8",
    )
    verdict = module.verify_review_packet(path, expected_commit="deadbeef")
    assert verdict["passed"] is False
    assert any("must not be the author" in p for p in verdict["problems"])


def test_an_unresolved_blocking_finding_is_refused(tmp_path: Path) -> None:
    module = _closure_module()
    path = tmp_path / "review.json"
    path.write_text(
        json.dumps(
            {
                "reviewer": "someone-else",
                "author": "claude-implementation-agent",
                "reviewer_verdict": "PASS",
                "open_findings": {"S0": 0, "S1": 2},
            }
        ),
        encoding="utf-8",
    )
    verdict = module.verify_review_packet(path, expected_commit="deadbeef")
    assert verdict["passed"] is False
    # WRK-R3 widened the blocking band from S0/S1 to S0-S3.
    assert any("open S1" in p for p in verdict["problems"])


# -- the notebook has to be runnable before it is run ---------------------


@pytest.mark.skipif(
    not (NOTEBOOK_DIR / "qdgrasp-phase-3-4-3-cuda-gate.ipynb").is_file(),
    reason="the notebook has not been built in this checkout",
)
def test_every_code_cell_parses() -> None:
    """A cell that does not parse is a T4 run spent on a syntax error.

    Two runs were lost that way: an escape written as ``\\n`` in the builder
    became a real newline in the generated cell, which the builder itself has no
    reason to notice.
    """
    import ast

    notebook = json.loads(
        (NOTEBOOK_DIR / "qdgrasp-phase-3-4-3-cuda-gate.ipynb").read_text(encoding="utf-8")
    )
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        try:
            ast.parse(source)
        except SyntaxError as exc:  # pragma: no cover - the assertion is the point
            raise AssertionError(f"cell {index} does not parse: {exc}") from exc


@pytest.mark.skipif(
    not (NOTEBOOK_DIR / "qdgrasp-phase-3-4-3-cuda-gate.ipynb").is_file(),
    reason="the notebook has not been built in this checkout",
)
def test_the_warp_matrix_probes_the_scene_that_showed_the_defect() -> None:
    # REV-20260827-010 isolated the defect on the LEAP hand model with its
    # meshes. A three-geom scene may never reach the kernel in question, so a
    # clean result there would say nothing about the defect being re-tested.
    notebook = json.loads(
        (NOTEBOOK_DIR / "qdgrasp-phase-3-4-3-cuda-gate.ipynb").read_text(encoding="utf-8")
    )
    source = "".join("".join(cell["source"]) for cell in notebook["cells"])
    probe = source.split("PROBE = (", 1)[1].split(")", 1)[0]
    assert "leap_hand.yaml" in probe
    assert "build_rollout_scene_model" in probe
    assert "micro_scene.xml" not in probe


@pytest.mark.skipif(
    not (NOTEBOOK_DIR / "qdgrasp-phase-3-4-3-cuda-gate.ipynb").is_file(),
    reason="the notebook has not been built in this checkout",
)
def test_clean_needs_positive_proof_not_an_absent_error_line() -> None:
    # A probe that dies before its first instrumented call prints no errors
    # either, and that produced a false "clean" for all three pins.
    notebook = json.loads(
        (NOTEBOOK_DIR / "qdgrasp-phase-3-4-3-cuda-gate.ipynb").read_text(encoding="utf-8")
    )
    source = "".join("".join(cell["source"]) for cell in notebook["cells"])
    assert "probe_did_not_run" in source
    assert "inconclusive_no_error_summary" in source
    assert "ERROR SUMMARY" in source
    assert "verbatim_tail" in source


@pytest.mark.skipif(
    not (NOTEBOOK_DIR / "qdgrasp-phase-3-4-3-cuda-gate.ipynb").is_file(),
    reason="the notebook has not been built in this checkout",
)
def test_evidence_goes_where_kaggle_can_hand_it_back() -> None:
    # /tmp is not persisted, so a run that wrote its packet there produced
    # evidence nobody could download.
    notebook = json.loads(
        (NOTEBOOK_DIR / "qdgrasp-phase-3-4-3-cuda-gate.ipynb").read_text(encoding="utf-8")
    )
    source = "".join("".join(cell["source"]) for cell in notebook["cells"])
    assert "/kaggle/working/phase3_4_3_evidence" in source
    assert "/tmp/phase3_4_3_evidence" not in source


def test_the_gate_records_the_commit_it_measured() -> None:
    """Evidence that does not say which code it ran on cannot be tied to it.

    The first T4 run produced evidence with no commit field at all, so the
    closure runner could only refuse it -- not because the measurement was bad
    but because nothing connected it to a tree.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location("cuda_gate", GATE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    commit = module._repo_commit()
    assert len(commit) == 40, commit
    assert all(c in "0123456789abcdef" for c in commit)

    source = GATE.read_text(encoding="utf-8")
    assert '"commit": _repo_commit(),' in source


# -- the sanitizer is a gate criterion, not a side note -------------------


def _gate_module():
    import importlib.util

    spec = importlib.util.spec_from_file_location("cuda_gate", GATE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_a_run_with_no_sanitizer_report_cannot_be_clean() -> None:
    # Absence of a check is not a clean check (G08.7).
    module = _gate_module()
    result = module.read_sanitizer_report(None)
    assert result["clean"] is False
    assert "has not shown there are none" in result["detail"]


def test_uninitialised_reads_are_not_clean(tmp_path: Path) -> None:
    module = _gate_module()
    path = tmp_path / "sanitizer.json"
    path.write_text(
        json.dumps(
            {
                "initcheck": {
                    "head": [
                        "========= COMPUTE-SANITIZER",
                        "========= Uninitialized __global__ memory read of size 4 bytes",
                        "========= ERROR SUMMARY: 68224 errors",
                    ]
                },
                "racecheck": {
                    "head": [
                        "========= COMPUTE-SANITIZER",
                        "========= RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)",
                    ]
                },
            }
        ),
        encoding="utf-8",
    )
    result = module.read_sanitizer_report(path)
    assert result["clean"] is False
    assert result["tools"]["initcheck"]["clean"] is False
    # racecheck being clean is exactly what separates an uninitialised read from
    # a race, so it is reported separately rather than folded in.
    assert result["tools"]["racecheck"]["clean"] is True


def test_a_clean_sanitizer_report_is_clean(tmp_path: Path) -> None:
    module = _gate_module()
    path = tmp_path / "sanitizer.json"
    path.write_text(
        json.dumps(
            {
                "initcheck": {
                    "head": [
                        "========= COMPUTE-SANITIZER",
                        "========= ERROR SUMMARY: 0 errors",
                    ]
                },
                "racecheck": {
                    "head": [
                        "========= COMPUTE-SANITIZER",
                        "========= RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)",
                    ]
                },
            }
        ),
        encoding="utf-8",
    )
    assert module.read_sanitizer_report(path)["clean"] is True


def test_a_missing_tool_is_not_silently_clean(tmp_path: Path) -> None:
    module = _gate_module()
    path = tmp_path / "sanitizer.json"
    path.write_text(
        json.dumps({"initcheck": {"head": ["========= ERROR SUMMARY: 0 errors"]}}),
        encoding="utf-8",
    )
    result = module.read_sanitizer_report(path)
    assert result["clean"] is False
    assert result["tools"]["racecheck"]["status"] == "absent"


def test_the_verdict_folds_the_sanitizer_in() -> None:
    source = GATE.read_text(encoding="utf-8")
    # Parity and performance are necessary, not sufficient.
    assert 'evidence["verdict"] = "PASS" if evidence["sanitizer"]["clean"] else "BLOCKED"' in source


@pytest.mark.skipif(
    not (NOTEBOOK_DIR / "qdgrasp-phase-3-4-3-cuda-gate.ipynb").is_file(),
    reason="the notebook has not been built in this checkout",
)
def test_the_sanitizer_runs_before_the_gate_reads_it() -> None:
    """Ordering is the difference between a criterion and a decoration.

    The gate reads the sanitizer report as a pass condition, so the cell that
    writes it has to run first; with the cells the other way round the gate can
    only ever see a missing file.
    """
    notebook = json.loads(
        (NOTEBOOK_DIR / "qdgrasp-phase-3-4-3-cuda-gate.ipynb").read_text(encoding="utf-8")
    )
    sanitizer_at = gate_at = None
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        if "compute-sanitizer" in source and "WARP_MATRIX" not in source:
            sanitizer_at = index if sanitizer_at is None else sanitizer_at
        if "check_phase3_4_3_cuda.py" in source and "--dry-run" not in source:
            gate_at = index if gate_at is None else gate_at

    assert sanitizer_at is not None and gate_at is not None
    assert sanitizer_at < gate_at, (sanitizer_at, gate_at)

    source = "".join("".join(c["source"]) for c in notebook["cells"])
    assert "--sanitizer-report" in source
    assert "phase3_4_3_evidence/sanitizer.json" in source
