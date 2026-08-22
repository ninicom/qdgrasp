#!/usr/bin/env python3
"""CPU gate for the Phase 1 framework: API/config round-trip and lifecycle smoke.

The CUDA half of the Phase 1 gate is deliberately not here; it lives in
``scripts/phase1_cuda_smoke.py`` and must run on physical NVIDIA hardware.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import torch
import yaml

from qdgrasp import QDGrasp, __version__
from qdgrasp.config import ConfigError, RunConfig, dump_document, load_model_config, parse_document, resolve_runtime
from qdgrasp.engine.callbacks import LossHistory


def check_config_round_trip(problems: list[str]) -> None:
    original = load_model_config("qdgrasp-dummy-n.yaml")
    reparsed = type(original).model_validate(yaml.safe_load(dump_document(original)))
    if reparsed != original or reparsed.content_hash() != original.content_hash():
        problems.append("model config does not round-trip through YAML")

    try:
        parse_document({"schema": "qdgrasp/run/v1", "epochs": 3}, RunConfig, origin="gate")
    except ConfigError:
        pass
    else:
        problems.append("unknown run key was accepted")

    runtime = resolve_runtime(RunConfig(device="cpu", amp=True))
    if runtime.amp or not runtime.adjustments:
        problems.append("CPU run did not force amp=False with a recorded adjustment")

    try:
        resolve_runtime(RunConfig(device="cuda:0"))
    except RuntimeError:
        pass
    except ConfigError:
        pass
    else:
        if not torch.cuda.is_available():
            problems.append("CUDA request succeeded without CUDA hardware")


def check_lifecycle(problems: list[str], workdir: Path) -> dict[str, object]:
    os.chdir(workdir)
    grasper = QDGrasp()
    result = grasper.train("dummy-tiny.yaml", max_steps=6, batch_size=4, val_interval=3, run_name="gate")
    for artifact in ("runs/gate/results.json", "runs/gate/resume.pt", "runs/gate/bundle/bundle.json"):
        if not Path(artifact).is_file():
            problems.append(f"missing training artifact {artifact}")
    if result.global_step != 6:
        problems.append(f"expected 6 steps, got {result.global_step}")
    if not result.metrics:
        problems.append("training produced no validation metrics")

    metrics = grasper.val("dummy-tiny.yaml", batch_size=4)
    if not all(float(value) == float(value) for value in metrics.values()):
        problems.append("validation produced a non-finite metric")

    results = grasper.predict(torch.randn(64, 3))
    lower = torch.tensor(grasper.robot_config.lower_limits)
    upper = torch.tensor(grasper.robot_config.upper_limits)
    if not bool((results.joint_values >= lower).all() and (results.joint_values <= upper).all()):
        problems.append("predicted joints escaped the declared limits")

    export = grasper.export(fmt="torchscript", out_dir="runs/gate/export")
    deviations = export.metadata["round_trip_max_abs_deviation"]
    if max(deviations.values()) > 1e-5:
        problems.append(f"TorchScript round-trip deviation too large: {deviations}")

    continuous = LossHistory()
    QDGrasp().train("dummy-tiny.yaml", max_steps=8, run_name="full", callbacks=[continuous])
    first = LossHistory()
    QDGrasp().train("dummy-tiny.yaml", max_steps=8, stop_after_steps=4, run_name="part", callbacks=[first])
    second = LossHistory()
    QDGrasp().train("dummy-tiny.yaml", max_steps=8, resume="runs/part/resume.pt", run_name="part-2", callbacks=[second])
    if continuous.history != first.history + second.history:
        problems.append("resume is not bit-exact against an uninterrupted CPU run")

    return {
        "final_loss": result.final_loss,
        "metrics": result.metrics,
        "bundle_hash": result.hashes["bundle"],
        "export_deviation": deviations,
    }


def check_import_purity(problems: list[str], project_root: Path) -> None:
    script = (
        "import os, sys, json; before = os.getcwd(); import qdgrasp; "
        "forbidden = [n for n in ('ultralytics','cv2','MinkowskiEngine','spconv','open3d') if n in sys.modules]; "
        "print(json.dumps({'cwd_stable': before == os.getcwd(), 'forbidden': forbidden}))"
    )
    output = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, cwd=project_root)
    if output.returncode != 0:
        problems.append(f"importing qdgrasp failed: {output.stderr.strip().splitlines()[-1:]}")
        return
    payload = json.loads(output.stdout)
    if not payload["cwd_stable"]:
        problems.append("importing qdgrasp changed the working directory")
    if payload["forbidden"]:
        problems.append(f"base import pulled unapproved modules: {payload['forbidden']}")


def main() -> int:
    logging.basicConfig(level=logging.WARNING)
    project_root = Path(__file__).resolve().parents[1]
    problems: list[str] = []
    summary: dict[str, object] = {}

    check_config_round_trip(problems)
    check_import_purity(problems, project_root)
    original_cwd = Path.cwd()
    with tempfile.TemporaryDirectory(prefix="qdgrasp-phase1-") as workdir:
        try:
            summary = check_lifecycle(problems, Path(workdir))
        finally:
            os.chdir(original_cwd)

    print(f"Phase 1 CPU framework: {'PASS' if not problems else 'FAIL'}")
    print(f"qdgrasp version: {__version__}")
    if summary:
        print(json.dumps(summary, indent=2, sort_keys=True))
    print("CUDA dummy train-step is NOT covered here; run scripts/phase1_cuda_smoke.py on NVIDIA hardware.")
    for problem in problems:
        print(f"- {problem}")
    return 0 if not problems else 1


if __name__ == "__main__":
    raise SystemExit(main())
