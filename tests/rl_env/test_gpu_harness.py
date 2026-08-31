"""P3.5-13/14/16: the GPU harness must refuse to fake a GPU result.

There is no NVIDIA device here, so what can be tested is exactly the part that
matters most: the harness's refusals.  A gate that quietly ran on the CPU while
reporting a CUDA device is the failure mode ``ADR-0006`` exists to prevent, and
Phase 3.4.3 already spent a cycle on evidence that had to be thrown away for a
related reason.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.phase3_5_gpu_rl_readiness import (
    BACKENDS,
    GpuGateError,
    _require_cuda,
    main,
)


def test_a_cpu_device_is_reported_as_a_cpu_device() -> None:
    assert _require_cuda("cpu") == {"device": "cpu", "cuda": False}


def test_requesting_cuda_without_cuda_is_refused() -> None:
    import torch

    if torch.cuda.is_available():  # pragma: no cover - only on a GPU machine
        pytest.skip("CUDA is available; the refusal path cannot be exercised")
    with pytest.raises(GpuGateError, match="must not be labelled a GPU run"):
        _require_cuda("cuda:0")


def test_the_cpu_oracle_runs_and_records_its_own_verdict(tmp_path) -> None:
    evidence = tmp_path / "cpu.json"
    status = main(["--backend", "mujoco-cpu", "--device", "cpu", "--evidence", str(evidence)])
    assert status == 0
    document = json.loads(evidence.read_text(encoding="utf-8"))
    assert document["verdict"] == "cpu_oracle_only"
    assert document["cpu_oracle_passed"] is True
    assert {item["profile"] for item in document["cpu_oracle"]} == set(document["active_hands"])
    assert document["shadow_hand"] == "paused_by_ADR-0008"
    # Compile and settle are timed apart from stepping, so a slow backend cannot
    # hide its warm-up inside a throughput number.
    for item in document["cpu_oracle"]:
        assert {"compile_s", "settle_s", "episode_s", "steps_per_s"} <= set(item)


def test_a_gpu_request_without_a_gpu_writes_a_refusal(tmp_path) -> None:
    import torch

    if torch.cuda.is_available():  # pragma: no cover
        pytest.skip("CUDA is available; the refusal path cannot be exercised")
    evidence = tmp_path / "gpu.json"
    status = main(["--backend", "mjx-warp", "--device", "cuda:0", "--evidence", str(evidence)])
    assert status == 1
    document = json.loads(evidence.read_text(encoding="utf-8"))
    assert document["verdict"] == "refused"
    assert "CUDA is not available" in document["error"]
    assert "gpu_candidate" not in document


def test_the_backend_list_names_the_two_candidates_from_the_plan() -> None:
    assert BACKENDS == ("mujoco-cpu", "mjx-warp", "maniskill-gpu")


def test_the_cloud_notebook_pins_an_immutable_commit() -> None:
    import re

    notebook = PROJECT_ROOT / "notebooks/phase3_5_rl_readiness.ipynb"
    assert notebook.is_file()
    document = json.loads(notebook.read_text(encoding="utf-8"))
    source = "".join("".join(cell["source"]) for cell in document["cells"])
    match = re.search(r'CODE_REVISION = "([0-9a-f]{40})"', source)
    assert match is not None, "the notebook must pin a 40-character commit"
    assert match.group(1) != "0" * 40, "the placeholder revision must be replaced before publication"
    assert "ADR-0006" in source, "the notebook must say that a CPU run is not GPU evidence"
