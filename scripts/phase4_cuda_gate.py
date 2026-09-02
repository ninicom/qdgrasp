#!/usr/bin/env python3
"""CUDA gate for Phase 4 (``ROADMAP-P4-001`` §6, §7.4, §7.5).

This is the harness, not the verdict.  It cannot be run on the development
machine: there is no NVIDIA GPU here, and ``ADR-0006`` forbids a CPU fallback
standing in for one.  Run it in the Kaggle/Colab notebook at
``notebooks/phase4_cuda_gate.ipynb``, which pins a commit.

Four refusals are built in, each of them a way a GPU gate has been faked before:

* ``--device cuda:*`` with no CUDA present is an error, never a CPU run wearing
  a GPU label;
* the CPU reference runs first and unconditionally, so a CUDA number always has
  something to be compared against;
* parity is measured against that reference at a pinned tolerance, so a CUDA
  path that silently computes something else fails rather than passes faster;
* the memory scaling check reports the measured ratio, not a verdict about it,
  and a run that could not measure memory says so instead of omitting the row.

What it measures, for each active hand:

1. forward and backward in FP32, with full gradient coverage;
2. output validity -- finite, ``RᵀR = I``, ``det R = 1``, joints inside their
   named limits;
3. CPU/CUDA FP32 parity within ``atol/rtol`` from ``PLAN.md`` §6;
4. peak memory at ``T`` and ``2T`` tokens, which is the ``N x N`` gate;
5. a short overfit, to show the architecture trains on CUDA as it does on CPU.

None of these is a grasping result.  §7 forbids citing any P4 number as one.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

#: Tolerance for CPU/CUDA FP32 parity (``PLAN.md`` §6).
PARITY_ATOL = 1e-4
PARITY_RTOL = 1e-4


class CudaGateError(RuntimeError):
    """The harness refuses to produce evidence it cannot stand behind."""


def _commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.SubprocessError):  # pragma: no cover - git-less checkout
        return "unknown"


def _require_cuda(device: str) -> dict[str, Any]:
    """Confirm a real NVIDIA device, or refuse to continue."""

    import torch

    if not device.startswith("cuda"):
        raise CudaGateError(
            f"this is the CUDA gate and it was asked for device '{device}'. Running it on CPU would produce a "
            "file that looks like CUDA evidence and is not (ADR-0006)."
        )
    if not torch.cuda.is_available():
        raise CudaGateError(
            f"{device} was requested and CUDA is not available. A CPU run must not be labelled a GPU run "
            "(ADR-0006); rerun this on real NVIDIA hardware."
        )
    index = int(device.split(":")[1]) if ":" in device else 0
    return {
        "device": device,
        "cuda": True,
        "name": torch.cuda.get_device_name(index),
        "capability": list(torch.cuda.get_device_capability(index)),
        "torch": torch.__version__,
        "driver_allocator": torch.cuda.get_allocator_backend(),
    }


def _fixture(robot, samples: int, points: int, seed: int, device):
    """A fixed batch with reachable labels, built exactly like the CPU overfit."""

    import torch
    from scipy.spatial.transform import Rotation

    generator = torch.Generator(device="cpu").manual_seed(seed)
    names = robot.actuated_joint_names
    limits = [robot.config.joint_limits[name] for name in names]
    lower = torch.tensor([value[0] for value in limits], dtype=torch.float32)
    upper = torch.tensor([value[1] for value in limits], dtype=torch.float32)
    palm_pos = torch.randn(samples, 3, generator=generator) * 0.05
    palm_rot = torch.tensor(Rotation.random(samples, random_state=seed).as_matrix(), dtype=torch.float32)
    joints = lower + torch.rand(samples, len(names), generator=generator) * (upper - lower)
    target_valid = torch.ones(samples, dtype=torch.bool, device=device)
    return {
        "points": (torch.randn(samples, points, 3, generator=generator) * 0.04).to(device),
        "palm_pos": palm_pos.to(device),
        "palm_rot": palm_rot.to(device),
        "joint_angles": joints.to(device),
        "fingertip_positions": robot.fingertip_positions(palm_pos, palm_rot, joints).to(device),
        "success": (torch.rand(samples, generator=generator) > 0.5).float().to(device),
        "kinematics_valid": target_valid,
        "pose_target_valid": target_valid,
        "joint_target_valid": target_valid,
        "fk_target_valid": target_valid,
    }


def _output_validity(prediction, robot) -> dict[str, Any]:
    import torch

    rotation = prediction.palm_rotation
    identity = torch.eye(3, device=rotation.device).expand_as(rotation)
    lower = torch.tensor([robot.joint_limits[name][0] for name in robot.actuated_joint_names], device=rotation.device)
    upper = torch.tensor([robot.joint_limits[name][1] for name in robot.actuated_joint_names], device=rotation.device)
    return {
        "finite": bool(prediction.is_finite()),
        "orthonormal": bool(torch.allclose(rotation.transpose(-1, -2) @ rotation, identity, atol=1e-4)),
        "determinant_one": bool(
            torch.allclose(torch.linalg.det(rotation), torch.ones(rotation.shape[0], device=rotation.device), atol=1e-4)
        ),
        "joints_in_limits": bool(
            (prediction.joint_angles >= lower - 1e-5).all() and (prediction.joint_angles <= upper + 1e-5).all()
        ),
    }


def run_hand(profile: str, device_name: str, *, samples: int, points: int, steps: int, seed: int) -> dict[str, Any]:
    """Forward, backward, parity, memory and a short overfit for one hand."""

    import torch

    from qdgrasp.models.config import FlowModelSettings, QDGraspFlow
    from qdgrasp.models.losses import LossWeights, forward_and_loss, gradient_coverage
    from qdgrasp.robot.spec import RobotSpec

    device = torch.device(device_name)
    robot = RobotSpec.from_config(profile, sample_anchors=False)

    torch.manual_seed(seed)
    reference = QDGraspFlow(FlowModelSettings(), robot)
    state = {name: tensor.clone() for name, tensor in reference.state_dict().items()}

    cpu_batch = _fixture(robot, samples, points, seed, torch.device("cpu"))
    noise = torch.randn(
        samples, reference.model.flow_config.state_dimension, generator=torch.Generator().manual_seed(seed)
    )

    def evaluate(module, batch, sample_noise):
        prediction, losses = forward_and_loss(
            module.model,
            module.robot,
            module.graph,
            weights=LossWeights(),
            sample_noise=sample_noise,
            **batch,
        )
        return prediction, losses

    # -- CPU reference, first and unconditionally --------------------------
    cpu_prediction, cpu_losses = evaluate(reference, cpu_batch, noise)
    cpu_losses.total.backward()
    cpu_coverage = gradient_coverage(reference)

    # -- the same weights on CUDA -----------------------------------------
    module = QDGraspFlow(FlowModelSettings(), robot)
    module.load_state_dict(state)
    module = module.to(device)
    batch = {name: tensor.to(device) for name, tensor in cpu_batch.items()}
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    prediction, losses = evaluate(module, batch, noise.to(device))
    losses.total.backward()
    torch.cuda.synchronize(device)
    step_s = time.perf_counter() - started
    coverage = gradient_coverage(module)
    peak_small = int(torch.cuda.max_memory_allocated(device))

    parity = {
        name: float(
            torch.linalg.norm(
                (getattr(prediction, name).detach().cpu() - getattr(cpu_prediction, name).detach()).reshape(-1),
                ord=float("inf"),
            )
        )
        for name in ("palm_translation", "palm_rotation", "joint_angles", "fingertips")
    }
    parity["total_loss"] = abs(float(losses.total.detach().cpu()) - float(cpu_losses.total.detach()))
    scale = max(float(cpu_prediction.palm_translation.detach().abs().max()), 1.0)
    parity_passed = all(value <= PARITY_ATOL + PARITY_RTOL * scale for value in parity.values())

    # -- memory scaling: twice the points, not four times the memory -------
    module.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    wide = _fixture(robot, samples, points * 2, seed, device)
    wide_prediction, wide_losses = evaluate(module, wide, noise.to(device))
    wide_losses.total.backward()
    torch.cuda.synchronize(device)
    peak_large = int(torch.cuda.max_memory_allocated(device))

    # -- a short overfit, to show it trains here too -----------------------
    module.zero_grad(set_to_none=True)
    optimizer = torch.optim.AdamW(module.parameters(), lr=3e-4)
    first_loss = None
    overfit_started = time.perf_counter()
    for index in range(steps):
        optimizer.zero_grad()
        _, step_losses = evaluate(module, batch, noise.to(device))
        step_losses.total.backward()
        torch.nn.utils.clip_grad_norm_(module.parameters(), 1.0)
        optimizer.step()
        if index == 0:
            first_loss = float(step_losses.total.detach())
    torch.cuda.synchronize(device)
    final_prediction, final_losses = evaluate(module, batch, noise.to(device))
    overfit_s = time.perf_counter() - overfit_started

    return {
        "robot": profile,
        "parameters": sum(p.numel() for p in module.parameters()),
        "step_s": step_s,
        "gradient_coverage": {"covered": sum(coverage.values()), "total": len(coverage)},
        "cpu_gradient_coverage": {"covered": sum(cpu_coverage.values()), "total": len(cpu_coverage)},
        "output_validity": _output_validity(prediction, robot),
        "output_validity_wide": _output_validity(wide_prediction, robot),
        "parity": parity,
        "parity_tolerance": {"atol": PARITY_ATOL, "rtol": PARITY_RTOL},
        "parity_passed": parity_passed,
        "memory": {
            "points": points,
            "peak_bytes": peak_small,
            "points_doubled": points * 2,
            "peak_bytes_doubled": peak_large,
            "ratio": peak_large / peak_small if peak_small else None,
        },
        "overfit": {
            "steps": steps,
            "elapsed_s": overfit_s,
            "first_total": first_loss,
            "last_total": float(final_losses.total.detach()),
            "last_terms": final_losses.to_document(),
            # Checked again after training: a model that drifts out of SO(3) or
            # out of its joint limits while the loss falls has not learned, it
            # has escaped.
            "output_validity": _output_validity(final_prediction, robot),
        },
        "passed": bool(
            sum(coverage.values()) == len(coverage)
            and all(_output_validity(prediction, robot).values())
            and parity_passed
            and first_loss is not None
            and float(final_losses.total.detach()) < first_loss
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--points", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--evidence", type=Path, default=Path("phase4_cuda_evidence.json"))
    args = parser.parse_args(argv)

    from qdgrasp.config.active_scope import resolve_workload_hands

    scope = resolve_workload_hands()
    evidence: dict[str, Any] = {
        "schema": "qdgrasp/phase4-cuda-evidence/v0",
        "commit": _commit(),
        "platform": f"{platform.system()}-{platform.machine()}-py{platform.python_version()}",
        "scope": scope.as_disclosure(),
        "settings": {"samples": args.samples, "points": args.points, "steps": args.steps, "seed": args.seed},
        "note": "Architecture evidence only. ROADMAP-P4-001 §7 forbids citing any of it as a grasping result.",
    }

    try:
        evidence["device"] = _require_cuda(args.device)
        evidence["hands"] = [
            run_hand(
                profile,
                args.device,
                samples=args.samples,
                points=args.points,
                steps=args.steps,
                seed=args.seed,
            )
            for profile in scope.robot_profiles
        ]
        passed = all(item["passed"] for item in evidence["hands"])
        evidence["verdict"] = "measured" if passed else "failed"
        status = 0 if passed else 1
    except CudaGateError as error:
        evidence["verdict"] = "refused"
        evidence["error"] = str(error)
        status = 1

    args.evidence.parent.mkdir(parents=True, exist_ok=True)
    args.evidence.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(evidence, indent=2, sort_keys=True))
    print(f"\nwrote {args.evidence}; verdict={evidence['verdict']}")
    if evidence["verdict"] != "measured":
        print("P4-11 stays open. ROADMAP-P4-001 §6 requires a measured CUDA run for both active hands.")
    return status


if __name__ == "__main__":
    sys.exit(main())
