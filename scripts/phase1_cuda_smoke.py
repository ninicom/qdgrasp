#!/usr/bin/env python3
"""CUDA half of the Phase 1 gate; must run on a physical NVIDIA GPU.

Run this from the separate CUDA notebook repository against an installed
``qdgrasp`` wheel.  It refuses to produce evidence on a CPU-only host: a CPU
fallback is not admissible as CUDA evidence
(``docs/decisions/0006-cuda-hardware-required.md``).

    python scripts/phase1_cuda_smoke.py --out evidence/phase1_cuda_evidence.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import torch

from qdgrasp import QDGrasp, __version__, environment_info, require_cuda
from qdgrasp.engine.callbacks import LossHistory


def run_smoke(device: str) -> dict[str, object]:
    """Train, validate, resume, predict and export on a real CUDA device."""

    evidence: dict[str, object] = {}

    fp32 = LossHistory()
    grasper = QDGrasp()
    result = grasper.train(
        "dummy-tiny.yaml", device=device, max_steps=8, batch_size=4, val_interval=4, run_name="cuda-fp32", callbacks=[fp32]
    )
    evidence["fp32"] = {
        "steps": result.global_step,
        "final_loss": result.final_loss,
        "metrics": result.metrics,
        "bundle_hash": result.hashes["bundle"],
        "effective_runtime": result.runtime["effective"],
    }

    amp = LossHistory()
    amp_result = QDGrasp().train(
        "dummy-tiny.yaml", device=device, amp=True, max_steps=8, batch_size=4, run_name="cuda-amp", callbacks=[amp]
    )
    evidence["amp"] = {
        "steps": amp_result.global_step,
        "final_loss": amp_result.final_loss,
        "precision": amp_result.runtime["effective"]["precision"],
    }
    if amp_result.runtime["effective"]["precision"] != "16-mixed":
        raise RuntimeError("AMP was requested on CUDA but the effective precision is not 16-mixed")

    continuous = LossHistory()
    QDGrasp().train("dummy-tiny.yaml", device=device, max_steps=8, run_name="cuda-full", callbacks=[continuous])
    first = LossHistory()
    QDGrasp().train(
        "dummy-tiny.yaml", device=device, max_steps=8, stop_after_steps=4, run_name="cuda-part", callbacks=[first]
    )
    second = LossHistory()
    QDGrasp().train(
        "dummy-tiny.yaml", device=device, max_steps=8, resume="runs/cuda-part/resume.pt", run_name="cuda-part-2", callbacks=[second]
    )
    evidence["resume"] = {
        "continuous": continuous.history,
        "resumed": first.history + second.history,
        "bit_exact": continuous.history == first.history + second.history,
    }

    predictions = grasper.predict(torch.randn(64, 3), device=device)
    lower = torch.tensor(grasper.robot_config.lower_limits, device=predictions.joint_values.device)
    upper = torch.tensor(grasper.robot_config.upper_limits, device=predictions.joint_values.device)
    evidence["predict"] = {
        "device": str(predictions.device),
        "count": len(predictions),
        "joints_within_limits": bool((predictions.joint_values >= lower).all() and (predictions.joint_values <= upper).all()),
        "rotation_finite": bool(torch.isfinite(predictions.rotation).all()),
    }

    export = grasper.export(fmt="torchscript", out_dir="runs/cuda-export")
    evidence["export"] = {
        "format": export.metadata["format"],
        "artifact_sha256": export.metadata["artifact_sha256"],
        "round_trip_max_abs_deviation": export.metadata["round_trip_max_abs_deviation"],
    }

    cpu_reference = QDGrasp().predict(torch.zeros(64, 3), device="cpu")
    cuda_reference = QDGrasp().predict(torch.zeros(64, 3), device=device)
    deviation = float((cpu_reference.translation - cuda_reference.translation.cpu()).abs().max())
    evidence["cpu_cuda_fp32_parity"] = {"translation_max_abs_deviation": deviation, "tolerance": 1e-4}
    if deviation > 1e-4:
        raise RuntimeError(f"CPU/CUDA FP32 parity deviation {deviation:.3e} exceeds 1e-4")

    return evidence


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0", help="CUDA device to run on; CPU is rejected.")
    parser.add_argument("--out", type=Path, default=None, help="Write the evidence JSON to this relative path.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    if not args.device.startswith("cuda"):
        print("ERROR: this gate only accepts a CUDA device; CPU results are not CUDA evidence.")
        return 2
    require_cuda()

    original_cwd = Path.cwd()
    with tempfile.TemporaryDirectory(prefix="qdgrasp-phase1-cuda-") as workdir:
        os.chdir(workdir)
        try:
            evidence = run_smoke(args.device)
        finally:
            os.chdir(original_cwd)

    payload = {
        "schema": "qdgrasp/evidence/phase1-cuda/v1",
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "qdgrasp_version": __version__,
        "environment": environment_info().to_dict(),
        "checks": evidence,
    }
    payload["payload_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    print(rendered)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered + "\n", encoding="utf-8")
        print(f"evidence written to {args.out}")
    return 0 if evidence["resume"]["bit_exact"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
