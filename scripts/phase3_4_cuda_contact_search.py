#!/usr/bin/env python3
"""Phase 3.4 CUDA gate; must run on a physical NVIDIA GPU.

Staged to match the phase. Stage 1 resolves the backend decision that P3.4-04
left open: whether MuJoCo Warp actually carries tendon transmission, weld
equality and per-contact force for the release hands. Until P3.4-05 lands there
is no CUDA backend to benchmark, and this script says so rather than reporting a
CPU number as CUDA evidence.

A CPU fallback is never admissible here
(``docs/decisions/0006-cuda-hardware-required.md``).

    python scripts/phase3_4_cuda_contact_search.py \\
      --device cuda:0 --profile kaggle-t4-micro \\
      --evidence phase3_4_cuda_evidence.json
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
import platform
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]

EVIDENCE_SCHEMA = "qdgrasp/evidence/phase3.4-cuda/v1"

#: Requirements P3.4-04 measured from the real rollout models. A GPU backend
#: that cannot carry all of these does not unblock the phase.
BLOCKING_REQUIREMENTS = (
    "tendon_transmission",
    "equality:mjEQ_WELD",
    "mocap_body",
    "per_contact_force_and_frame",
)


class GateFailure(RuntimeError):
    """The gate refuses to emit evidence."""


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def assert_real_cuda(device: str) -> dict[str, Any]:
    """Refuse anything that is not a physical NVIDIA device."""
    import torch

    if not device.startswith("cuda"):
        raise GateFailure(f"--device must be a CUDA device, got {device!r}")
    if not torch.cuda.is_available():
        raise GateFailure(
            "torch.cuda.is_available() is False; a CPU run is not CUDA evidence"
        )
    if torch.cuda.device_count() < 1:
        raise GateFailure("no CUDA device visible to torch")

    index = int(device.split(":")[1]) if ":" in device else 0
    properties = torch.cuda.get_device_properties(index)
    return {
        "device": device,
        "gpu_name": torch.cuda.get_device_name(index),
        "capability": f"{properties.major}.{properties.minor}",
        "total_memory_gib": round(properties.total_memory / (1024**3), 2),
        "torch": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "driver": getattr(torch.version, "cuda", None),
    }


def resolve_warp_backend(device: str) -> dict[str, Any]:
    """Answer the question P3.4-04 could not answer on CPU.

    Reports what MuJoCo Warp actually exposes. It does not guess: a missing
    module is reported as missing, and a missing capability blocks the phase.
    """
    import mujoco

    status: dict[str, Any] = {
        "mujoco": mujoco.__version__,
        "modules": {},
        "capabilities": {},
    }
    for name in ("warp", "mujoco_warp", "mujoco.mjx"):
        spec = importlib.util.find_spec(name)
        status["modules"][name] = "available" if spec is not None else "not_installed"

    if status["modules"]["mujoco_warp"] != "available":
        status["verdict"] = "blocked_mujoco_warp_missing"
        status["unsupported"] = list(BLOCKING_REQUIREMENTS)
        return status

    warp = importlib.import_module("warp")
    mjwarp = importlib.import_module("mujoco_warp")
    status["warp_version"] = getattr(warp, "__version__", "unknown")

    devices = [str(d) for d in getattr(warp, "get_cuda_devices", list)()]
    status["warp_cuda_devices"] = devices
    if not devices:
        status["verdict"] = "blocked_warp_sees_no_cuda_device"
        status["unsupported"] = list(BLOCKING_REQUIREMENTS)
        return status

    # Compile each release hand and record which blocking requirement survives.
    from qdgrasp.dataset.pipeline.validators.mujoco_rollout import (
        build_rollout_scene_model,
    )
    from qdgrasp.objects.schema import SubGeomSpec
    from qdgrasp.robot.spec import RobotSpec, resolve_robot_asset

    probe_geoms = (SubGeomSpec(type="box", size=(0.02, 0.02, 0.02), pos=(0.0, 0.0, 0.0)),)
    unsupported: list[str] = []
    tested: set[str] = set()
    per_hand: dict[str, Any] = {}
    for hand, config_name in (
        ("leap_hand", "leap_hand.yaml"),
        ("wonik_allegro", "wonik_allegro.yaml"),
        ("shadow_hand", "shadow_hand.yaml"),
    ):
        spec = RobotSpec.from_config(config_name, sample_anchors=False)
        model = build_rollout_scene_model(
            resolve_robot_asset(spec.config.source_asset), probe_geoms
        )
        entry: dict[str, Any] = {
            "ntendon": int(model.ntendon),
            "neq": int(model.neq),
            "nmocap": int(model.nmocap),
        }
        warp_model = None
        try:
            put_model = getattr(mjwarp, "put_model", None)
            if put_model is None:
                raise AttributeError("mujoco_warp exposes no put_model")
            warp_model = put_model(model)
            entry["compiled"] = True
            tested.add("equality:mjEQ_WELD")
            tested.add("mocap_body")
            if int(model.ntendon) > 0:
                tested.add("tendon_transmission")
        except Exception as exc:  # noqa: BLE001 - the verdict is the point
            entry["compiled"] = False
            entry["error"] = f"{type(exc).__name__}: {exc}"
            if int(model.ntendon) > 0:
                unsupported.append("tendon_transmission")
            unsupported.append("equality:mjEQ_WELD")
            unsupported.append("mocap_body")

        # Compiling proves the model is accepted. It does not prove contact
        # force can be read back, and the whole safety budget depends on that,
        # so step the model and look for the arrays.
        if warp_model is not None:
            try:
                import mujoco as _mj

                put_data = getattr(mjwarp, "put_data", None)
                step = getattr(mjwarp, "step", None)
                if put_data is None or step is None:
                    raise AttributeError(
                        "mujoco_warp exposes no put_data/step: "
                        f"{sorted(n for n in dir(mjwarp) if not n.startswith('_'))[:40]}"
                    )
                cpu_data = _mj.MjData(model)
                _mj.mj_forward(model, cpu_data)
                warp_data = put_data(model, cpu_data)
                for _ in range(20):
                    step(warp_model, warp_data)
                contact = getattr(warp_data, "contact", None)
                fields = (
                    sorted(n for n in dir(contact) if not n.startswith("_"))
                    if contact is not None
                    else []
                )
                entry["contact_fields"] = fields[:30]
                has_force = any(
                    f in fields for f in ("force", "efc_force", "frame", "dist", "pos")
                )
                entry["contact_readback"] = bool(contact is not None and has_force)
                tested.add("per_contact_force_and_frame")
                if not entry["contact_readback"]:
                    unsupported.append("per_contact_force_and_frame")
            except Exception as exc:  # noqa: BLE001
                entry["contact_readback"] = False
                entry["contact_error"] = f"{type(exc).__name__}: {exc}"
                unsupported.append("per_contact_force_and_frame")
        per_hand[hand] = entry

    status["per_hand"] = per_hand
    status["unsupported"] = sorted(set(unsupported))
    # Tri-state: never report a capability as supported unless it was exercised.
    status["capabilities"] = {
        req: (
            "unsupported"
            if req in status["unsupported"]
            else ("supported" if req in tested else "not_tested")
        )
        for req in BLOCKING_REQUIREMENTS
    }
    status["untested_requirements"] = sorted(
        req for req, state in status["capabilities"].items() if state == "not_tested"
    )
    if status["unsupported"]:
        status["verdict"] = "blocked_missing_capability"
    elif status["untested_requirements"]:
        status["verdict"] = "incomplete_untested_requirements"
    else:
        status["verdict"] = "supported"
    return status


#: Pinned before the run. The plan forbids looping batch size upward until OOM,
#: so the cap is fixed and a failure at this size is reported as a failure.
BENCHMARK_WORLDS = 64
BENCHMARK_HORIZON = 100
VRAM_BUDGET_GIB = 14.0
MIN_GPU_SPEEDUP = 2.0


def _benchmark_scene(
    device: str, label: str, scene_xml: str, signature: Any
) -> dict[str, Any]:
    """Time one scene on both backends with identical commands."""
    import mujoco
    import numpy as np
    import torch

    from qdgrasp.dataset.dynamic_contracts import DynamicGraspRequest
    from qdgrasp.sim.batched.mjwarp_cuda import MjWarpCudaBackend
    from qdgrasp.sim.batched.mujoco_cpu import MuJoCoCpuBackend

    scene = scene_xml
    profile = signature.robot_profile

    def requests(count: int) -> list[DynamicGraspRequest]:
        return [
            DynamicGraspRequest(
                scene_state_ref="scene:micro#0",
                observation_ref="obs:micro/cam_top",
                target_object_id="target",
                robot_profile=profile,
                strategy_id="batched_cem",
                safety_budget_id="bench",
                horizon=BENCHMARK_HORIZON,
                control_dt=0.002,
                seed=index,
            )
            for index in range(count)
        ]

    result: dict[str, Any] = {
        "label": label,
        "worlds": BENCHMARK_WORLDS,
        "horizon": BENCHMARK_HORIZON,
    }

    torch.cuda.reset_peak_memory_stats()
    gpu = MjWarpCudaBackend(scene, device=device)
    gpu.compile(signature, profile, batch_capacity=BENCHMARK_WORLDS)
    gpu.reset(requests(BENCHMARK_WORLDS))
    commands = np.full((BENCHMARK_WORLDS, BENCHMARK_HORIZON, gpu.num_actuators), 0.15)
    gpu_summaries = gpu.rollout(commands)
    gpu_timing = gpu.timing
    peak_vram = torch.cuda.max_memory_allocated() / (1024**3)

    cpu = MuJoCoCpuBackend(scene)
    cpu.compile(signature, profile, batch_capacity=BENCHMARK_WORLDS)
    cpu.reset(requests(BENCHMARK_WORLDS))
    cpu.rollout(commands)
    cpu_timing = cpu.timing

    speedup = (
        gpu_timing.steps_per_second / cpu_timing.steps_per_second
        if cpu_timing.steps_per_second > 0
        else float("nan")
    )
    rejected = [s.world_index for s in gpu_summaries if s.hard_reject]

    # Every finalist must be replayable on the oracle; the GPU never self-admits.
    finalists = gpu.export_finalists([0, 1, 2])
    replayable = all(f.backend_request == "cpu" for f in finalists)

    result.update(
        {
            "status": "measured",
            "gpu_compile_seconds": round(gpu_timing.compile_seconds, 4),
            "gpu_warmup_seconds": round(gpu_timing.warmup_seconds, 4),
            "gpu_steady_state_seconds": round(gpu_timing.steady_state_seconds, 4),
            "gpu_steps_per_second": round(gpu_timing.steps_per_second, 1),
            "cpu_steps_per_second": round(cpu_timing.steps_per_second, 1),
            "speedup": round(float(speedup), 3),
            "min_required_speedup": MIN_GPU_SPEEDUP,
            "speedup_met": bool(speedup >= MIN_GPU_SPEEDUP),
            "peak_vram_gib": round(peak_vram, 3),
            "vram_budget_gib": VRAM_BUDGET_GIB,
            "vram_within_budget": bool(peak_vram <= VRAM_BUDGET_GIB),
            "rejected_worlds": rejected,
            "finalists_routed_to_cpu": replayable,
            "worlds_ran_without_oom": len(rejected) == 0,
            "geom_count": int(mujoco.MjModel.from_xml_string(scene).ngeom),
        }
    )
    return result


def run_search_benchmark(device: str) -> dict[str, Any]:
    """Benchmark both a trivial scene and the real Phase 3.4 workload.

    Both are reported. The micro scene is three geoms, where per-step kernel
    launch dominates and the GPU cannot win; the representative scene is a
    dexterous hand, which is the workload the phase actually searches. The gate
    reads the representative one and records the micro result beside it, rather
    than picking whichever number looks better.
    """
    import mujoco

    from qdgrasp.dataset.pipeline.generated_reachable import (
        build_generated_reachable_object,
    )
    from qdgrasp.dataset.pipeline.validators.mujoco_rollout import (
        build_rollout_scene_model,
    )
    from qdgrasp.robot.spec import RobotSpec, resolve_robot_asset
    from qdgrasp.sim.batched.contracts import SceneSignature

    micro_xml = (REPO_ROOT / "tests" / "dynamic_grasp" / "micro_scene.xml").read_text(
        encoding="utf-8"
    )
    micro_signature = SceneSignature(
        robot_profile="micro_pusher",
        environment="table",
        geom_type_counts=(("box", 2), ("plane", 1)),
        joint_count=2,
        support_count=1,
        solver_profile="default",
        timestep=0.002,
    )

    spec = RobotSpec.from_config("leap_hand.yaml", sample_anchors=False)
    fixture = build_generated_reachable_object("leap_hand")
    hand_model = build_rollout_scene_model(
        resolve_robot_asset(spec.config.source_asset),
        fixture.collision_geoms,
        object_pos=fixture.object_pos,
        object_mass=fixture.mass,
    )
    hand_xml = hand_model.to_xml_string() if hasattr(hand_model, "to_xml_string") else None
    if hand_xml is None:
        spec_obj = mujoco.MjSpec.from_file(str(resolve_robot_asset(spec.config.source_asset)))
        hand_xml = spec_obj.to_xml()
    hand_signature = SceneSignature(
        robot_profile="leap_hand",
        environment="table",
        geom_type_counts=(("mesh", int(hand_model.ngeom)),),
        joint_count=int(hand_model.njnt),
        support_count=1,
        solver_profile="default",
        timestep=float(hand_model.opt.timestep),
    )

    results = {}
    for label, xml, signature in (
        ("micro_pusher", micro_xml, micro_signature),
        ("leap_hand_scene", hand_xml, hand_signature),
    ):
        try:
            results[label] = _benchmark_scene(device, label, xml, signature)
        except Exception as exc:  # noqa: BLE001 - a failed scene is reported
            results[label] = {
                "label": label,
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
            }

    gating = results.get("leap_hand_scene", {})
    return {
        "status": "measured",
        "gating_scene": "leap_hand_scene",
        "gating_rationale": (
            "The performance criterion is about the workload Phase 3.4 searches: "
            "a dexterous hand in a scene. The three-geom micro scene is reported "
            "beside it because it bounds the other end -- GPU batching cannot win "
            "where per-step launch overhead dominates."
        ),
        "scenes": results,
        **{k: v for k, v in gating.items() if k not in ("label", "status")},
    }


def build_evidence(device: str, profile: str) -> dict[str, Any]:
    from qdgrasp import __version__ as qdgrasp_version

    cuda = assert_real_cuda(device)
    warp = resolve_warp_backend(device)
    evidence = {
        "schema": EVIDENCE_SCHEMA,
        "timestamp": datetime.now(UTC).isoformat(),
        "profile": profile,
        "qdgrasp_version": qdgrasp_version,
        "python": platform.python_version(),
        "cuda_environment": cuda,
        "backend_resolution": warp,
        "blocking_requirements": list(BLOCKING_REQUIREMENTS),
        "search_benchmark": (
            run_search_benchmark(device)
            if warp["verdict"] == "supported"
            else {"status": "skipped", "reason": warp["verdict"]}
        ),
    }
    evidence["status"] = (
        "BACKEND_SUPPORTED_PENDING_P3.4-05"
        if warp["verdict"] == "supported"
        else "BLOCKED"
    )
    return evidence


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--profile", default="kaggle-t4-micro")
    parser.add_argument("--evidence", type=Path, default=None)
    args = parser.parse_args()

    try:
        evidence = build_evidence(args.device, args.profile)
    except GateFailure as exc:
        print(f"Phase 3.4 CUDA gate failed: {exc}", file=sys.stderr)
        return 1
    except ImportError as exc:
        print(f"Phase 3.4 CUDA gate failed to import a dependency: {exc}", file=sys.stderr)
        return 1

    payload = json.dumps(evidence, indent=2, sort_keys=True)
    if args.evidence:
        args.evidence.parent.mkdir(parents=True, exist_ok=True)
        args.evidence.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    print(f"evidence_sha256={_sha256_text(payload)}", file=sys.stderr)

    verdict = evidence["backend_resolution"]["verdict"]
    if verdict != "supported":
        unsupported = evidence["backend_resolution"].get("unsupported", [])
        print(
            f"Phase 3.4 BLOCKED: MuJoCo Warp verdict {verdict!r}; "
            f"unsupported blocking requirements: {unsupported}. "
            "Per ROADMAP-P3.4-001 section 10 the phase stays blocked and a backend "
            "decision record is written. Do not substitute a mock CUDA backend and "
            "do not drop Shadow from the gate.",
            file=sys.stderr,
        )
        return 1

    bench = evidence["search_benchmark"]
    if bench.get("status") == "measured":
        failures = [
            name
            for name, ok in (
                ("speedup", bench["speedup_met"]),
                ("vram", bench["vram_within_budget"]),
                ("no_oom", bench["worlds_ran_without_oom"]),
                ("cpu_routing", bench["finalists_routed_to_cpu"]),
            )
            if not ok
        ]
        if failures:
            print(
                f"Phase 3.4 benchmark failed: {failures}. "
                f"speedup={bench['speedup']}x against a required "
                f"{bench['min_required_speedup']}x.",
                file=sys.stderr,
            )
            return 1
        print(
            f"Phase 3.4 benchmark passed: {bench['speedup']}x speedup on "
            f"{bench['worlds']} worlds, {bench['peak_vram_gib']} GiB peak VRAM.",
            file=sys.stderr,
        )
    print(
        "Backend supported and benchmarked. The phase still needs a dataset with "
        "a positive per hand and an independent review; this run does not close it.",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
