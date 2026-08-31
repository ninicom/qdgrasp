#!/usr/bin/env python3
"""GPU backend compatibility spike for Phase 3.5 (P3.5-13/14, gate in §12).

This is the harness, not the verdict.  It runs the CPU oracle first, then the
requested GPU backend, and writes the evidence a backend decision record has to
be made from.  It cannot be run here: there is no NVIDIA GPU on the development
machine, and ``ADR-0006`` forbids a CPU fallback standing in for one.

Three refusals are built in, because each of them is a way a GPU gate has been
quietly faked before:

* ``--device cuda:*`` with no CUDA present is an error, never a CPU run wearing
  a GPU label;
* the CPU oracle must pass before the GPU candidate is measured at all, so a
  GPU result can never be the only thing that was checked;
* compile and warm-up time are reported separately from steady-state throughput,
  because folding them together is how a slow backend is made to look fast.

A backend is *not* chosen by this script.  §7 requires two-hand parity on
compile, step, contact, drop and lift before a decision record may be written,
and the decision lives in an ADR that a human signs.
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

BACKENDS = ("mujoco-cpu", "mjx-warp", "maniskill-gpu")


class GpuGateError(RuntimeError):
    """The harness refuses to produce evidence it cannot stand behind."""


def _commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.SubprocessError):  # pragma: no cover - git-less checkout
        return "unknown"


def _require_cuda(device: str) -> dict[str, Any]:
    """Confirm a real NVIDIA device, or refuse to continue."""

    if not device.startswith("cuda"):
        return {"device": device, "cuda": False}
    try:
        import torch
    except ImportError as error:  # pragma: no cover
        raise GpuGateError(f"{device} was requested but torch is not installed") from error
    if not torch.cuda.is_available():
        raise GpuGateError(
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
    }


def _scene_fixture(directory: Path):
    """One dropped object on a table, built the same way the CPU gate builds it."""

    import numpy as np

    from qdgrasp.objects.generate import generate_box
    from qdgrasp.objects.manifest import create_object_asset, save_object_asset
    from qdgrasp.scenes.virtual_drop import DropObjectRequest, SpawnRegion, VirtualDropSceneSpec

    rng = np.random.default_rng(0)
    mesh, geoms, params, mass, inertia = generate_box(rng, size_range=(0.028, 0.034), density=650.0)
    mesh_bytes, manifest = create_object_asset("target", "primitive", "box", mesh, geoms, params, mass, inertia)
    asset_ref = str(save_object_asset(mesh_bytes, manifest, directory))
    objects = (DropObjectRequest(object_id="target", asset_ref=asset_ref),)
    scene = VirtualDropSceneSpec(
        spawn_region=SpawnRegion(half_extents=(0.05, 0.05, 0.0)), drop_height_range_m=(0.02, 0.04)
    )
    return objects, scene


def run_cpu_oracle(profile: str, steps: int) -> dict[str, Any]:
    """Compile, settle and step one scene per hand on the CPU oracle."""

    import tempfile

    import mujoco

    from qdgrasp.rl.envs import DexAcquireConfig, DexAcquireEnv
    from qdgrasp.rl.tasks import ScriptedAcquireSpec, run_scripted_episode
    from qdgrasp.scenes.resolver import resolve_scene
    from qdgrasp.scenes.settle import certify_settle

    directory = Path(tempfile.mkdtemp(prefix="qdgrasp-p35-gpu-"))
    objects, scene = _scene_fixture(directory)

    compile_started = time.perf_counter()
    resolved = resolve_scene(objects=objects, virtual_scene_config=scene, seed=1)
    compile_s = time.perf_counter() - compile_started

    settle_started = time.perf_counter()
    snapshot = certify_settle(
        resolved.spec,
        resolved.model,
        mujoco.MjData(resolved.model),
        scene.settle_thresholds,
        spawn_region=scene.spawn_region,
    )
    settle_s = time.perf_counter() - settle_started

    spec = ScriptedAcquireSpec()
    config = DexAcquireConfig(
        robot_profile=profile,
        objects=objects,
        target_object_id="target",
        virtual_scene=scene,
        max_steps=min(steps, spec.total_steps),
        settle_steps=400,
    )
    env = DexAcquireEnv(config)
    step_started = time.perf_counter()
    scripted = run_scripted_episode(env, seed=21, spec=spec)
    step_s = time.perf_counter() - step_started

    return {
        "profile": profile,
        "compile_s": compile_s,
        "settle_s": settle_s,
        "settle_outcome": snapshot.outcome.value,
        "scene_hash": snapshot.scene_hash,
        "episode_s": step_s,
        "steps": scripted["steps"],
        "steps_per_s": scripted["steps"] / step_s if step_s > 0 else 0.0,
        "terminal_reason": scripted["terminal_reason"],
        "observations_finite": scripted["observations_finite"],
        "passed": bool(snapshot.outcome.value == "settled" and scripted["observations_finite"]),
    }


def run_gpu_candidate(backend: str, device: str, profile: str, steps: int) -> dict[str, Any]:
    """Measure the GPU candidate.  Absent backends are reported, never faked."""

    del steps
    if backend == "mjx-warp":
        try:
            import mujoco_warp  # type: ignore[import-not-found]  # noqa: F401
        except ImportError as error:
            raise GpuGateError(
                "backend 'mjx-warp' was requested and mujoco_warp is not installed. "
                "Install the pinned build on the GPU runtime; an unavailable backend is not a passing gate."
            ) from error
    elif backend == "maniskill-gpu":
        try:
            import mani_skill  # type: ignore[import-not-found]  # noqa: F401
        except ImportError as error:
            raise GpuGateError("backend 'maniskill-gpu' was requested and mani_skill is not installed.") from error
    else:
        raise GpuGateError(f"{backend!r} is not a GPU backend")

    raise GpuGateError(
        f"the {backend!r} adapter is not implemented. P3.5-13/14 requires measured two-hand parity on "
        "compile, step, contact, drop and lift before §7 permits a backend decision, and this harness "
        "will not emit a number it did not measure. "
        f"(device={device}, profile={profile})"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", default="mujoco-cpu", choices=BACKENDS)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--profile", default="notebook-micro")
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--evidence", type=Path, default=Path("phase3_5_gpu_evidence.json"))
    args = parser.parse_args(argv)

    from qdgrasp.rl.envs import ACTIVE_ROBOT_PROFILES

    evidence: dict[str, Any] = {
        "schema": "qdgrasp/phase3-5-gpu-evidence/v0",
        "commit": _commit(),
        "backend": args.backend,
        "profile": args.profile,
        "platform": f"{platform.system()}-{platform.machine()}-py{platform.python_version()}",
        "active_hands": list(ACTIVE_ROBOT_PROFILES),
        "shadow_hand": "paused_by_ADR-0008",
    }

    try:
        evidence["device"] = _require_cuda(args.device)
        # The CPU oracle runs first and unconditionally: a GPU number with no
        # oracle behind it is a number with nothing to be compared against.
        evidence["cpu_oracle"] = [run_cpu_oracle(profile, args.steps) for profile in ACTIVE_ROBOT_PROFILES]
        oracle_passed = all(item["passed"] for item in evidence["cpu_oracle"])
        evidence["cpu_oracle_passed"] = oracle_passed
        if not oracle_passed:
            raise GpuGateError("the CPU oracle did not pass; a GPU candidate may not be measured against it")
        if args.backend != "mujoco-cpu":
            evidence["gpu_candidate"] = run_gpu_candidate(
                args.backend, args.device, ACTIVE_ROBOT_PROFILES[0], args.steps
            )
        evidence["verdict"] = "cpu_oracle_only" if args.backend == "mujoco-cpu" else "measured"
        status = 0
    except GpuGateError as error:
        evidence["verdict"] = "refused"
        evidence["error"] = str(error)
        status = 1

    args.evidence.parent.mkdir(parents=True, exist_ok=True)
    args.evidence.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(evidence, indent=2, sort_keys=True))
    print(f"\nwrote {args.evidence}; verdict={evidence['verdict']}")
    if evidence["verdict"] != "measured":
        print(
            "No backend decision follows from this run. ROADMAP-P3.5-001 §7 requires measured two-hand "
            "parity before P3.5-15 may record one."
        )
    return status


if __name__ == "__main__":
    sys.exit(main())
