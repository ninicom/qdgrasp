#!/usr/bin/env python3
"""Minimized reproducer for the GPU world divergence, under Compute Sanitizer.

Section 3.3 step 5. The classification already excluded a benchmark reading
defect and an index-range error, and showed the rejected set changes between
runs, which points at a race or uninitialized memory. Sanitizer output is what
distinguishes those from each other; contact numbers cannot.

Deliberately small: one scene, a few worlds, a short horizon. Sanitizer slows
execution by one to two orders of magnitude, and a 1024-world run under it would
not finish. This is a diagnostic and is never performance evidence.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]


def build_reproducer(worlds: int, horizon: int) -> dict[str, Any]:
    import numpy as np

    from qdgrasp.dataset.dynamic_contracts import DynamicGraspRequest
    from qdgrasp.dataset.pipeline.generated_reachable import (
        build_generated_reachable_object,
    )
    from qdgrasp.dataset.pipeline.validators.mujoco_rollout import (
        build_rollout_scene_model,
    )
    from qdgrasp.robot.spec import RobotSpec, resolve_robot_asset
    from qdgrasp.sim.batched.contracts import SceneSignature
    from qdgrasp.sim.batched.mjwarp_cuda import MjWarpCudaBackend

    spec = RobotSpec.from_config("leap_hand.yaml", sample_anchors=False)
    fixture = build_generated_reachable_object("leap_hand")
    model = build_rollout_scene_model(
        resolve_robot_asset(spec.config.source_asset),
        fixture.collision_geoms,
        object_pos=fixture.object_pos,
        object_mass=fixture.mass,
    )
    # Derived from the compiled model rather than written out by hand: a field
    # nobody remembered to fill in is how two different models share a bucket.
    signature = SceneSignature.from_model(
        model, robot_profile="leap_hand", environment="table", support_count=1
    )

    backend = MjWarpCudaBackend(model, device="cuda:0")
    backend.compile(signature, "leap_hand", batch_capacity=worlds)
    requests = [
        DynamicGraspRequest(
            scene_state_ref="scene:repro#0",
            observation_ref="obs:repro",
            target_object_id="target_object",
            robot_profile="leap_hand",
            strategy_id="sanitizer_repro",
            safety_budget_id="repro",
            horizon=horizon,
            control_dt=float(model.opt.timestep),
            seed=index,
        )
        for index in range(worlds)
    ]
    backend.reset(requests)

    commands = np.full((worlds, horizon, backend.num_actuators), 0.15)
    backend.rollout(commands)
    state = backend.observe()

    finite = np.isfinite(state.qpos).all(axis=1)
    rows = np.round(state.qpos[finite], 9)
    distinct = int(np.unique(rows, axis=0).shape[0]) if finite.any() else 0

    # Identical inputs must give identical outputs. Report the spread, not just
    # the count of NaN worlds: the earlier run showed every survivor differing.
    spread = 0.0
    if finite.sum() > 1:
        spread = float(np.max(np.ptp(state.qpos[finite], axis=0)))

    return {
        "worlds": worlds,
        "horizon": horizon,
        "geoms": int(model.ngeom),
        "nonfinite_worlds": int((~finite).sum()),
        "finite_worlds": int(finite.sum()),
        "distinct_finite_rows": distinct,
        "max_qpos_spread_across_identical_worlds": spread,
        "all_identical": bool(distinct <= 1),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worlds", type=int, default=8)
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    try:
        report = build_reproducer(args.worlds, args.horizon)
    except Exception as exc:  # noqa: BLE001 - the reproducer reports its own failure
        report = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
    payload = json.dumps(report, indent=2, sort_keys=True)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
