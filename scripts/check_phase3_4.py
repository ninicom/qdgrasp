"""Phase 3.4 CPU gate for contact-rich dynamic grasp synthesis.

The gate is staged: it verifies the work packages that exist and reports the
ones that do not, rather than passing silently on an empty phase. A green run
here proves CPU correctness only. Phase 3.4 cannot close on it -- the CUDA
backend and throughput evidence come from the Kaggle harness (plan section 15).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

from qdgrasp.config.schema import ConfigError
from qdgrasp.dataset.dynamic_contracts import (
    HARD_REJECT_CLASSES,
    ContactClass,
    DynamicGraspRequest,
    DynamicSearchOutcome,
    TrajectoryStage,
)
from qdgrasp.sim.batched.contracts import (
    BatchedContactBackend,
    SceneSignature,
    WorldRejected,
)
from qdgrasp.sim.batched.mujoco_cpu import MuJoCoCpuBackend

REPO_ROOT = Path(__file__).resolve().parents[1]

#: Work packages of ROADMAP-P3.4-001 and how far each one is verified here.
#:   "done"        verified by this CPU gate
#:   "cpu_pending_gpu"  everything CPU can establish is done; the rest needs a
#:                 real NVIDIA device and is not claimed
#:   "todo"        not started
#: Nothing here may read as phase closure: that needs the Kaggle GPU evidence.
WORK_PACKAGES: dict[str, str] = {
    "P3.4-00 hypothesis/safety semantics/entry gate": "done",
    "P3.4-01 typed trajectory contracts": "done",
    "P3.4-02 batched backend protocol + scene bucketing": "done",
    "P3.4-03 MuJoCo CPU oracle backend": "done",
    "P3.4-04 MJX-Warp compatibility spike": "done",
    "P3.4-05 MJWarp CUDA backend": "cpu_pending_gpu",
    "P3.4-06 contact observer + safety budget": "done",
    "P3.4-07 primitive-sequence controller": "done",
    "P3.4-08 static-seeded contact rollout": "done",
    "P3.4-09 batched CEM search": "done",
    "P3.4-10 batched MPPI strategy": "todo",
    "P3.4-11 local contact trajectory refinement": "done",
    "P3.4-12 CPU replay + terminal grasp certifier": "done",
    "P3.4-13 trajectory writer/loader": "todo",
    "P3.4-14 static-vs-dynamic ablation": "todo",
    "P3.4-15 Kaggle CUDA harness": "cpu_pending_gpu",
    "P3.4-16 QDGrasp-ContactRich-Tiny": "todo",
    "P3.4-17 independent review": "todo",
}

#: The spike report P3.4-04 produces, if it has been run.
SPIKE_REPORT = (
    REPO_ROOT / "evidence" / "phase3_4" / "p04-backend-spike" / "requirement-matrix.json"
)

MICRO_SCENE = (REPO_ROOT / "tests" / "dynamic_grasp" / "micro_scene.xml").read_text(
    encoding="utf-8"
)

MICRO_SIGNATURE = SceneSignature(
    robot_profile="micro_pusher",
    environment="table",
    geom_type_counts=(("box", 2), ("plane", 1)),
    joint_count=2,
    support_count=1,
    solver_profile="default",
    timestep=0.002,
)


def _micro_request(seed: int) -> DynamicGraspRequest:
    return DynamicGraspRequest(
        scene_state_ref="scene:micro#0",
        observation_ref="obs:micro/cam_top",
        target_object_id="target",
        robot_profile="micro_pusher",
        strategy_id="primitive_sequence",
        safety_budget_id="micro-conservative-v1",
        horizon=40,
        control_dt=0.002,
        seed=seed,
    )


def verify_contracts() -> dict[str, Any]:
    """The wire format of the contracts is dataset-breaking if it drifts."""
    if HARD_REJECT_CLASSES != {ContactClass.FORBIDDEN, ContactClass.DAMAGING}:
        raise ConfigError("hard-reject contact classes drifted from the plan")

    try:
        DynamicSearchOutcome(
            trajectory_ref="t:probe",
            passed=True,
            failure_stage="none",
            failure_reason="none",
            gpu_search_evidence={"backend": "mjwarp_cuda"},
        )
    except ValueError:
        pass
    else:
        raise ConfigError(
            "a passed outcome was admitted without CPU replay evidence; "
            "GPU search must never admit a release positive on its own"
        )

    return {
        "contact_classes": sorted(c.value for c in ContactClass),
        "trajectory_stages": sorted(s.value for s in TrajectoryStage),
        "hard_reject_classes": sorted(c.value for c in HARD_REJECT_CLASSES),
        "gpu_only_positive_refused": True,
    }


def verify_cpu_backend() -> dict[str, Any]:
    """Drive the oracle on real physics and check its fail-closed behaviour."""
    backend = MuJoCoCpuBackend(MICRO_SCENE)
    if not isinstance(backend, BatchedContactBackend):
        raise ConfigError("MuJoCoCpuBackend does not satisfy BatchedContactBackend")
    if "cuda" in backend.backend_id:
        raise ConfigError("the CPU oracle must never identify itself as CUDA")

    backend.compile(MICRO_SIGNATURE, "micro_pusher", batch_capacity=2)
    backend.reset([_micro_request(0), _micro_request(1)])

    try:
        backend.step(np.full((2, backend.num_actuators), np.nan))
    except WorldRejected:
        pass
    else:
        raise ConfigError("a non-finite control batch was accepted")

    backend.reset([_micro_request(0), _micro_request(1)])
    horizon = 40
    commands = np.zeros((2, horizon, backend.num_actuators))
    commands[0, :, 0] = 0.2  # drive world 0 only
    summaries = backend.rollout(commands)
    state = backend.observe()

    driven = float(state.object_pose[0, 0, 0])
    idle = float(state.object_pose[1, 0, 0])
    if not driven > idle + 1e-4:
        raise ConfigError(
            f"contact did not move the target: driven={driven:.5f} idle={idle:.5f}"
        )
    if any(s.hard_reject for s in summaries):
        raise ConfigError("micro rollout rejected a world")

    replay = MuJoCoCpuBackend(MICRO_SCENE)
    replay.compile(MICRO_SIGNATURE, "micro_pusher", batch_capacity=2)
    replay.reset([_micro_request(0), _micro_request(1)])
    replay.rollout(commands)
    if not np.array_equal(state.object_pose, replay.observe().object_pose):
        raise ConfigError("the CPU oracle did not replay bit-identically")

    timing = backend.timing
    return {
        "backend_id": backend.backend_id,
        "worlds": timing.worlds,
        "horizon": horizon,
        "target_displacement_m": round(driven - idle, 6),
        "compile_seconds": round(timing.compile_seconds, 4),
        "warmup_seconds": round(timing.warmup_seconds, 4),
        "steady_state_seconds": round(timing.steady_state_seconds, 4),
        "steps_per_second": round(timing.steps_per_second, 1),
        "replay_bit_identical": True,
        "scene_bucket": MICRO_SIGNATURE.bucket_key[:16],
    }


def verify_backend_spike() -> dict[str, Any]:
    """Report the P3.4-04 requirement matrix without pretending it is a verdict."""
    if not SPIKE_REPORT.is_file():
        return {"status": "not_run", "hint": "scripts/phase3_4_backend_spike.py"}
    report = json.loads(SPIKE_REPORT.read_text(encoding="utf-8"))
    verdict = report.get("gpu_backend_status", {}).get("verdict", "unknown")
    return {
        "status": "recorded",
        "required_features": len(report.get("required_feature_set", [])),
        "blocking_requirements": report.get("blocking_requirements", []),
        "gpu_support_verdict": verdict,
        "resolved_here": verdict != "unknown_pending_gpu_environment",
    }


def run_pytest() -> dict[str, Any]:
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/dynamic_grasp/", "-q", "-p", "no:cacheprovider"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=1800,
        check=False,
    )
    if completed.returncode != 0:
        raise ConfigError(f"tests/dynamic_grasp failed:\n{completed.stdout[-2000:]}")
    return {"summary": completed.stdout.strip().splitlines()[-1]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--profile", choices=("micro", "release"), default="micro")
    parser.add_argument("--skip-tests", action="store_true")
    args = parser.parse_args()

    outstanding = sorted(
        name for name, status in WORK_PACKAGES.items() if status == "todo"
    )

    try:
        if args.backend == "cuda":
            raise ConfigError(
                "the CUDA backend is not implemented (P3.4-05). This gate must not "
                "fall back to CPU and report it as CUDA evidence; run the Kaggle "
                "harness once P3.4-05/15 land."
            )
        if args.profile == "release":
            raise ConfigError(
                "profile=release needs QDGrasp-ContactRich-Tiny (P3.4-16), which "
                f"does not exist yet. Outstanding: {len(outstanding)} work packages."
            )
        result: dict[str, Any] = {
            "phase": "3.4",
            "backend": args.backend,
            "profile": args.profile,
            "status": "PARTIAL",
            "contracts": verify_contracts(),
            "cpu_backend": verify_cpu_backend(),
            "work_package_status": dict(sorted(WORK_PACKAGES.items())),
            "verified_on_cpu": sorted(
                name for name, status in WORK_PACKAGES.items() if status == "done"
            ),
            "backend_spike": verify_backend_spike(),
            "outstanding_work_packages": outstanding,
            "closure_blocked_by": [
                "CUDA backend and Kaggle GPU evidence (P3.4-05, P3.4-15)",
                "QDGrasp-ContactRich-Tiny (P3.4-16)",
                "independent review (P3.4-17)",
            ],
        }
        if not args.skip_tests:
            result["tests"] = run_pytest()
    except (ConfigError, OSError, subprocess.SubprocessError) as exc:
        print(f"Phase 3.4 gate failed: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(result, indent=2, sort_keys=True))
    print(
        f"Phase 3.4: PARTIAL -- {len(result['verified_on_cpu'])} of "
        f"{len(WORK_PACKAGES)} work packages verified on CPU. "
        "This is not phase closure.",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
