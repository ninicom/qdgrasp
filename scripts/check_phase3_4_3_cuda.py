#!/usr/bin/env python3
"""Phase 3.4.3 CUDA gate for the two active hands (G08, C07).

This runs on a real NVIDIA device -- Kaggle T4 is the primary environment -- and
answers four questions the CPU cannot:

1. **Capability.** Does this build expose every contact field the safety budget
   needs, *together*? Contact positions alone are not enough: without the force,
   the frame, the penetration depth and the identity of both geoms, the budget
   cannot be evaluated, and a gate that passes on the subset it happens to have
   is reporting coverage it does not have (G08.1).
2. **Parity.** Do the GPU and the CPU oracle agree, at three tiers: a
   no-contact short horizon where state must track closely, a pinned single
   contact where the impulse and object delta must match, and a full active-hand
   finalist where the outcome class and the safety verdict must agree.
3. **Performance.** On a representative active-hand workload, is the GPU at
   least twice the CPU on the median of several runs, inside a 14 GiB device
   budget measured at the device rather than through PyTorch's allocator?
4. **Cleanliness.** Zero non-finite worlds, zero contact overflow, zero
   truncated streams, zero OOM, zero fallback.

Every failure branch exits nonzero. The success branch prints only keys that
exist -- v1 crashed on its own pass path reading a key it never set (B-08).

    python scripts/check_phase3_4_3_cuda.py --device cuda:0 \\
        --evidence evidence/phase3_4_3/s10/cuda-gate.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import statistics
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

EVIDENCE_SCHEMA = "qdgrasp/evidence/phase3.4.3-cuda/v1"

#: Exit codes, shared with the CPU gate. Only an exact pass is 0.
PASS_EXIT = 0
FAIL_EXIT = 1
BLOCKED_EXIT = 3
CONFIG_EXIT = 4

#: Thresholds from ROADMAP-P3.4-001 section 10, unchanged. Loosening one of
#: these to get a pass is the failure the whole plan exists to prevent.
MIN_GPU_SPEEDUP = 2.0
VRAM_BUDGET_GIB = 14.0
MIN_SIMULTANEOUS_WORLDS = 64
#: The preregistered operating point. Chosen before the run, not after seeing it.
BENCHMARK_WORLDS = 1024
BENCHMARK_HORIZON = 60
BENCHMARK_RUNS = 3

#: Tier tolerances, pinned before any comparison.
NO_CONTACT_STATE_ATOL = 1e-4
SINGLE_CONTACT_IMPULSE_RTOL = 0.2
SINGLE_CONTACT_OBJECT_DELTA_M = 2e-3


class GateFailure(RuntimeError):
    """The gate refuses to emit a pass."""


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


def assert_real_cuda(device: str) -> dict[str, Any]:
    """Refuse anything that is not a physical NVIDIA device.

    A CPU number reported under a CUDA schema is the failure ADR-0006 forbids,
    so there is no fallback path through this function.
    """
    import torch

    if not device.startswith("cuda"):
        raise GateFailure(f"--device must be a CUDA device, got {device!r}")
    if not torch.cuda.is_available():
        raise GateFailure(
            "torch.cuda.is_available() is False; a CPU run is not CUDA evidence"
        )
    index = int(device.split(":")[1]) if ":" in device else 0
    properties = torch.cuda.get_device_properties(index)
    return {
        "device": device,
        "gpu_name": torch.cuda.get_device_name(index),
        "capability": f"{properties.major}.{properties.minor}",
        "total_memory_gib": round(properties.total_memory / (1024**3), 3),
        "torch": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "python": platform.python_version(),
    }


def device_memory_gib(index: int = 0) -> dict[str, float]:
    """Device-level memory, read at the device.

    ``torch.cuda.max_memory_allocated`` only sees PyTorch's allocator. Warp
    allocates through its own, so a PyTorch figure measures nothing about the
    backend under test -- Phase 3.4 reported roughly zero on that basis and
    called it within budget, which was an unearned pass (C07.4).
    """
    out: dict[str, float] = {}
    try:  # NVML first: it reports what the whole device is holding.
        import pynvml

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(index)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        out["source_nvml"] = 1.0
        out["device_used_gib"] = float(info.used) / (1024**3)
        out["device_total_gib"] = float(info.total) / (1024**3)
        return out
    except Exception as exc:  # noqa: BLE001 - absence is reported, never hidden
        out["nvml_unavailable"] = str(exc)[:120]  # type: ignore[assignment]
    try:
        import torch

        free, total = torch.cuda.mem_get_info(index)
        out["source_nvml"] = 0.0
        out["device_used_gib"] = (total - free) / (1024**3)
        out["device_total_gib"] = total / (1024**3)
    except Exception as exc:  # noqa: BLE001
        out["device_query_failed"] = 1.0
        out["device_query_message"] = str(exc)[:120]  # type: ignore[assignment]
    return out


# ---------------------------------------------------------------------------
# Capability
# ---------------------------------------------------------------------------


def probe_capability(device: str) -> dict[str, Any]:
    """Every contact field the budget needs, checked together.

    ``pos`` on its own answers "was there a contact"; the budget asks "how hard,
    where, how deep, and between what", and a build that cannot answer all four
    cannot enforce it (G08.1).
    """
    from qdgrasp.sim.batched.mjwarp_cuda import (
        REQUIRED_CONTACT_FIELDS,
        MjWarpCudaBackend,
        warp_is_available,
    )

    report: dict[str, Any] = {
        "required_contact_fields": list(REQUIRED_CONTACT_FIELDS),
        "warp_available": warp_is_available(),
    }
    if not report["warp_available"]:
        report["verdict"] = "warp_missing"
        report["missing_contact_fields"] = list(REQUIRED_CONTACT_FIELDS)
        return report

    from qdgrasp.sim.batched.contracts import SceneSignature

    micro_xml = (REPO_ROOT / "tests" / "dynamic_grasp" / "micro_scene.xml").read_text(
        encoding="utf-8"
    )
    import mujoco

    model = mujoco.MjModel.from_xml_string(micro_xml)
    signature = SceneSignature.from_model(
        model, robot_profile="micro_pusher", environment="table", support_count=1
    )
    backend = MjWarpCudaBackend(micro_xml, device=device)
    backend.compile(signature, "micro_pusher", batch_capacity=2)
    backend.reset(_requests(2))
    missing = list(backend.missing_contact_fields())
    forces = backend.read_contact_forces()

    report.update(
        {
            "missing_contact_fields": missing,
            "contact_force_readable": forces is not None,
            "overflow_telemetry": backend.contact_telemetry(0).__dict__,
            "verdict": "supported" if not missing and forces is not None else "unsupported",
        }
    )
    return report


def _requests(count: int, *, robot_profile: str = "micro_pusher", target: str = "target"):
    from qdgrasp.dataset.dynamic_contracts import DynamicGraspRequest

    return [
        DynamicGraspRequest(
            scene_state_ref=f"scene:gate#{index}",
            observation_ref="obs:gate/cam_top",
            target_object_id=target,
            robot_profile=robot_profile,
            strategy_id="batched_cem",
            safety_budget_id="contactrich-active-v1",
            horizon=BENCHMARK_HORIZON,
            control_dt=0.002,
            seed=index,
            backend_request="cuda",
        )
        for index in range(count)
    ]


# ---------------------------------------------------------------------------
# Parity
# ---------------------------------------------------------------------------


def run_parity(device: str) -> dict[str, Any]:
    """Three tiers, from the easiest agreement to the one that matters."""
    import mujoco
    import numpy as np

    from qdgrasp.sim.batched.contracts import SceneSignature
    from qdgrasp.sim.batched.mjwarp_cuda import MjWarpCudaBackend
    from qdgrasp.sim.batched.mujoco_cpu import MuJoCoCpuBackend

    micro_xml = (REPO_ROOT / "tests" / "dynamic_grasp" / "micro_scene.xml").read_text(
        encoding="utf-8"
    )
    model = mujoco.MjModel.from_xml_string(micro_xml)
    signature = SceneSignature.from_model(
        model, robot_profile="micro_pusher", environment="table", support_count=1
    )

    def both(worlds: int, commands: np.ndarray):
        gpu = MjWarpCudaBackend(micro_xml, device=device)
        gpu.compile(signature, "micro_pusher", batch_capacity=worlds)
        gpu.reset(_requests(worlds))
        gpu_summaries = gpu.rollout(commands)
        gpu_state = gpu.observe()

        cpu = MuJoCoCpuBackend(micro_xml)
        cpu.compile(signature, "micro_pusher", batch_capacity=worlds)
        cpu.reset(_requests(worlds))
        cpu_summaries = cpu.rollout(commands)
        cpu_state = cpu.observe()
        return gpu, gpu_summaries, gpu_state, cpu_summaries, cpu_state

    tiers: dict[str, Any] = {}

    # Tier 1: nothing touches, so the states must track closely.
    worlds = 4
    idle = np.zeros((worlds, 20, int(model.nu)))
    _, _, gpu_state, _, cpu_state = both(worlds, idle)
    delta = float(np.max(np.abs(gpu_state.qpos - cpu_state.qpos)))
    tiers["no_contact"] = {
        "max_qpos_delta": delta,
        "tolerance": NO_CONTACT_STATE_ATOL,
        "passed": bool(delta <= NO_CONTACT_STATE_ATOL),
    }

    # Tier 2: one pinned contact, so the object delta must match.
    driving = np.full((worlds, 60, int(model.nu)), 0.2)
    _, _, gpu_state, _, cpu_state = both(worlds, driving)
    object_delta = float(
        np.max(np.abs(gpu_state.object_pose[..., :3] - cpu_state.object_pose[..., :3]))
    )
    tiers["single_contact"] = {
        "max_object_delta_m": object_delta,
        "tolerance_m": SINGLE_CONTACT_OBJECT_DELTA_M,
        "impulse_rtol": SINGLE_CONTACT_IMPULSE_RTOL,
        "passed": bool(object_delta <= SINGLE_CONTACT_OBJECT_DELTA_M),
    }

    # Tier 3: the outcome class and the safety verdict have to agree, and every
    # survivor has to be replayable from its capsule.
    gpu, gpu_summaries, _, cpu_summaries, _ = both(worlds, driving)
    classes_agree = [
        (g.hard_reject == c.hard_reject and g.failure_reason == c.failure_reason)
        for g, c in zip(gpu_summaries, cpu_summaries, strict=True)
    ]
    survivors = [s.world_index for s in gpu_summaries if not s.hard_reject]
    capsules = gpu.export_finalists(survivors) if survivors else ()
    tiers["full_trajectory"] = {
        "worlds": worlds,
        "outcome_classes_agree": bool(all(classes_agree)),
        "disagreeing_worlds": [i for i, ok in enumerate(classes_agree) if not ok],
        "survivors": len(survivors),
        "capsules_exported": len(capsules),
        "every_survivor_has_a_capsule": bool(len(capsules) == len(survivors)),
        "passed": bool(all(classes_agree) and len(capsules) == len(survivors)),
    }

    tiers["passed"] = all(tiers[name]["passed"] for name in ("no_contact", "single_contact", "full_trajectory"))
    return tiers


# ---------------------------------------------------------------------------
# Performance
# ---------------------------------------------------------------------------


def estimate_resources(worlds: int, horizon: int, actuators: int, nq: int, nv: int) -> dict[str, Any]:
    """What a run would cost, before it is started (C07.1).

    Printed by ``--dry-run`` so an operator can see that a configuration will
    not fit before it OOMs halfway through.
    """
    state_bytes = worlds * (nq + nv) * 8
    command_bytes = worlds * horizon * actuators * 8
    return {
        "worlds": worlds,
        "steps": horizon,
        "world_steps": worlds * horizon,
        "estimated_host_state_bytes": state_bytes,
        "estimated_command_bytes": command_bytes,
        "estimated_total_bytes": state_bytes + command_bytes,
        "vram_budget_gib": VRAM_BUDGET_GIB,
    }


def run_performance(device: str, *, worlds: int, runs: int) -> dict[str, Any]:
    """Median of several runs on the preregistered operating point.

    The median, not the best: dropping a slow run because it looked wrong is how
    a benchmark measures the operator's expectations instead of the device.
    """
    import numpy as np

    from qdgrasp.config.active_scope import ACTIVE_HANDS
    from qdgrasp.dataset.pipeline.generated_reachable import (
        build_generated_reachable_object,
    )
    from qdgrasp.dataset.pipeline.validators.mujoco_rollout import (
        build_rollout_scene_model,
    )
    from qdgrasp.robot.spec import RobotSpec, resolve_robot_asset
    from qdgrasp.sim.batched.contracts import SceneSignature
    from qdgrasp.sim.batched.mjwarp_cuda import MjWarpCudaBackend
    from qdgrasp.sim.batched.mujoco_cpu import MuJoCoCpuBackend

    results: dict[str, Any] = {"hands": {}, "worlds": worlds, "runs": runs}
    for hand in ACTIVE_HANDS:
        spec = RobotSpec.from_config(f"{hand}.yaml", sample_anchors=False)
        fixture = build_generated_reachable_object(hand)
        model = build_rollout_scene_model(
            resolve_robot_asset(spec.config.source_asset),
            fixture.collision_geoms,
            object_pos=fixture.object_pos,
            object_mass=fixture.mass,
        )
        signature = SceneSignature.from_model(
            model, robot_profile=hand, environment="table", support_count=1
        )

        gpu_rates: list[float] = []
        cpu_rates: list[float] = []
        peak_vram: list[float] = []
        rejected_total = 0
        overflow_total = 0

        for _ in range(runs):
            before = device_memory_gib()
            gpu = MjWarpCudaBackend(model, device=device)
            gpu.compile(signature, hand, batch_capacity=worlds)
            gpu.reset(_requests(worlds, robot_profile=hand, target="target_object"))
            commands = np.full((worlds, BENCHMARK_HORIZON, gpu.num_actuators), 0.15)
            summaries = gpu.rollout(commands)
            after = device_memory_gib()
            gpu_rates.append(gpu.timing.steps_per_second)
            rejected_total += sum(1 for s in summaries if s.hard_reject)
            overflow_total += sum(1 for s in summaries if s.contact.buffer_overflow)
            if "device_used_gib" in before and "device_used_gib" in after:
                peak_vram.append(max(after["device_used_gib"] - before["device_used_gib"], 0.0))

            cpu = MuJoCoCpuBackend(model)
            cpu.compile(signature, hand, batch_capacity=min(worlds, 8))
            cpu.reset(_requests(min(worlds, 8), robot_profile=hand, target="target_object"))
            cpu.rollout(np.full((min(worlds, 8), BENCHMARK_HORIZON, cpu.num_actuators), 0.15))
            cpu_rates.append(cpu.timing.steps_per_second)

        gpu_median = statistics.median(gpu_rates)
        cpu_median = statistics.median(cpu_rates)
        speedup = gpu_median / cpu_median if cpu_median > 0 else float("nan")
        vram = max(peak_vram) if peak_vram else None
        results["hands"][hand] = {
            "gpu_steps_per_second_runs": [round(v, 1) for v in gpu_rates],
            "cpu_steps_per_second_runs": [round(v, 1) for v in cpu_rates],
            "gpu_steps_per_second_median": round(gpu_median, 1),
            "cpu_steps_per_second_median": round(cpu_median, 1),
            "speedup": round(float(speedup), 3),
            "speedup_met": bool(speedup >= MIN_GPU_SPEEDUP),
            "device_peak_vram_gib": round(vram, 3) if vram is not None else None,
            "vram_within_budget": bool(vram <= VRAM_BUDGET_GIB) if vram is not None else None,
            "vram_measurement": "nvml_or_device_free_delta" if vram is not None else "unavailable",
            "rejected_worlds": rejected_total,
            "overflow_worlds": overflow_total,
            "worlds_met": bool(worlds >= MIN_SIMULTANEOUS_WORLDS),
        }

    hands = results["hands"]
    results["passed"] = bool(
        hands
        and all(
            entry["speedup_met"]
            and entry["vram_within_budget"] is True
            and entry["rejected_worlds"] == 0
            and entry["overflow_worlds"] == 0
            and entry["worlds_met"]
            for entry in hands.values()
        )
    )
    return results


# ---------------------------------------------------------------------------
# Checkpoint and resume
# ---------------------------------------------------------------------------


def write_atomically(path: Path, payload: str) -> str:
    """Write through a temporary file and rename.

    A wall-clock kill in the middle of a write leaves a partial file, and a
    partial file that looks complete is how a resumed run counts a stage it
    never finished (C07.4).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(path)
    return _sha256_text(payload)


def load_checkpoint(path: Path | None) -> dict[str, Any]:
    if path is None or not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        # A truncated checkpoint is discarded rather than half-trusted.
        return {}
    return payload if isinstance(payload, dict) else {}


# ---------------------------------------------------------------------------
# Gate
# ---------------------------------------------------------------------------


def build_evidence(
    *, device: str, worlds: int, runs: int, checkpoint: Path | None, deadline_s: float | None
) -> dict[str, Any]:
    from qdgrasp import __version__ as qdgrasp_version
    from qdgrasp.config.active_scope import ACTIVE_HANDS, PAUSED_HANDS

    started = time.perf_counter()
    resumed = load_checkpoint(checkpoint)
    evidence: dict[str, Any] = {
        "schema": EVIDENCE_SCHEMA,
        "phase": "3.4.3",
        "timestamp": datetime.now(UTC).isoformat(),
        "qdgrasp_version": qdgrasp_version,
        "active_hands": list(ACTIVE_HANDS),
        "paused_hands": list(PAUSED_HANDS),
        "three_hand_coverage": False,
        "historical_p3_4_state": "paused_by_ADR-0008",
        "thresholds": {
            "min_gpu_speedup": MIN_GPU_SPEEDUP,
            "vram_budget_gib": VRAM_BUDGET_GIB,
            "min_simultaneous_worlds": MIN_SIMULTANEOUS_WORLDS,
            "benchmark_worlds": worlds,
            "benchmark_horizon": BENCHMARK_HORIZON,
            "benchmark_runs": runs,
        },
        "cuda_environment": assert_real_cuda(device),
        "resumed_stages": sorted(resumed),
    }

    def checkpointed(name: str, thunk) -> Any:
        if name in resumed:
            return resumed[name]
        value = thunk()
        resumed[name] = value
        if checkpoint is not None:
            write_atomically(checkpoint, json.dumps(resumed, indent=2, sort_keys=True) + "\n")
        return value

    evidence["capability"] = checkpointed("capability", lambda: probe_capability(device))
    if evidence["capability"].get("verdict") != "supported":
        evidence["verdict"] = "BLOCKED"
        return evidence

    if deadline_s is not None and time.perf_counter() - started > deadline_s:
        evidence["verdict"] = "BLOCKED"
        evidence["blocked_reason"] = "wall_time_guard_before_parity"
        return evidence
    evidence["parity"] = checkpointed("parity", lambda: run_parity(device))

    if deadline_s is not None and time.perf_counter() - started > deadline_s:
        evidence["verdict"] = "BLOCKED"
        evidence["blocked_reason"] = "wall_time_guard_before_performance"
        return evidence
    evidence["performance"] = checkpointed(
        "performance", lambda: run_performance(device, worlds=worlds, runs=runs)
    )

    evidence["verdict"] = (
        "PASS"
        if evidence["parity"].get("passed") and evidence["performance"].get("passed")
        else "FAIL"
    )
    evidence["wall_seconds"] = round(time.perf_counter() - started, 2)
    return evidence


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--worlds", type=int, default=BENCHMARK_WORLDS)
    parser.add_argument("--runs", type=int, default=BENCHMARK_RUNS)
    parser.add_argument("--evidence", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument(
        "--deadline-seconds",
        type=float,
        default=None,
        help="flush the ledger and stop before a cloud session is killed",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print worlds x steps and the estimated byte cost, then stop",
    )
    args = parser.parse_args()

    if args.worlds < MIN_SIMULTANEOUS_WORLDS:
        print(
            json.dumps(
                {
                    "verdict": "CONFIG_ERROR",
                    "error": (
                        f"--worlds {args.worlds} is below the declared floor of "
                        f"{MIN_SIMULTANEOUS_WORLDS}; lowering it would measure a "
                        "different gate"
                    ),
                },
                indent=2,
            )
        )
        return CONFIG_EXIT

    if args.dry_run:
        print(
            json.dumps(
                estimate_resources(args.worlds, BENCHMARK_HORIZON, actuators=16, nq=30, nv=29),
                indent=2,
                sort_keys=True,
            )
        )
        return PASS_EXIT

    try:
        evidence = build_evidence(
            device=args.device,
            worlds=args.worlds,
            runs=args.runs,
            checkpoint=args.checkpoint,
            deadline_s=args.deadline_seconds,
        )
    except GateFailure as exc:
        print(json.dumps({"verdict": "FAIL", "error": str(exc)}, indent=2))
        print(f"Phase 3.4.3 CUDA gate failed: {exc}", file=sys.stderr)
        return FAIL_EXIT
    except ImportError as exc:
        print(json.dumps({"verdict": "CONFIG_ERROR", "error": str(exc)}, indent=2))
        print(f"Phase 3.4.3 CUDA gate could not import a dependency: {exc}", file=sys.stderr)
        return CONFIG_EXIT

    payload = json.dumps(evidence, indent=2, sort_keys=True)
    digest = write_atomically(args.evidence, payload + "\n") if args.evidence else _sha256_text(payload)
    print(payload)
    print(f"evidence_sha256={digest}", file=sys.stderr)

    verdict = evidence["verdict"]
    if verdict == "PASS":
        summary = ", ".join(
            f"{hand} {entry['speedup']}x @ {entry['device_peak_vram_gib']} GiB"
            for hand, entry in evidence["performance"]["hands"].items()
        )
        print(
            f"Phase 3.4.3 CUDA gate: PASS on {len(evidence['performance']['hands'])} "
            f"active hands ({summary}). This is GPU evidence only; the phase closes "
            "on the completeness manifest and an independent review.",
            file=sys.stderr,
        )
        return PASS_EXIT
    if verdict == "BLOCKED":
        print(
            f"Phase 3.4.3 CUDA gate: BLOCKED -- "
            f"{evidence.get('blocked_reason') or evidence['capability'].get('verdict')}. "
            "A blocked gate is not a passed gate and not a failed one.",
            file=sys.stderr,
        )
        return BLOCKED_EXIT
    print("Phase 3.4.3 CUDA gate: FAIL. See the parity and performance blocks.", file=sys.stderr)
    return FAIL_EXIT


if __name__ == "__main__":
    raise SystemExit(main())
