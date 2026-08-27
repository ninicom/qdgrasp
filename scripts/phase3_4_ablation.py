#!/usr/bin/env python3
"""Static-vs-dynamic controlled ablation (P3.4-14).

The Phase 3.4 hypothesis is that letting the scene react produces valid grasps a
frozen-object pipeline misses. This script measures that on the same scenes,
same targets, same safety budget, with only the frozen-object assumption
changing -- and it is written to be able to report failure.

The plan is explicit: if yield does not rise, or rises only through unsafe
contact, the record says so rather than moving a threshold.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from qdgrasp.dataset.dynamic_contracts import ContactSafetyBudget
from qdgrasp.dynamic.objective import ReasonLedger
from qdgrasp.dynamic.primitives import Primitive, PrimitiveKind, TransitionCondition
from qdgrasp.dynamic.safety import SceneRoles
from qdgrasp.dynamic.static_seeded import (
    RolloutLimits,
    SeedPose,
    run_static_seeded_rollout,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
MICRO_SCENE = REPO_ROOT / "tests" / "dynamic_grasp" / "micro_scene.xml"

#: Pinned before the run. Changing any of these after reading a result is the
#: move the plan forbids, so they are hashed into the report.
BUDGET = ContactSafetyBudget(
    budget_id="ablation-conservative-v1",
    robot_profile="micro_pusher",
    peak_normal_force_N=20.0,
    peak_tangential_force_N=12.0,
    normal_impulse_Ns=2.0,
    tangential_impulse_Ns=1.2,
    contact_duration_s=5.0,
    contact_work_J=0.5,
    max_penetration_m=0.002,
    max_wrist_force_N=40.0,
    max_wrist_torque_Nm=6.0,
    max_joint_or_tendon_load=15.0,
    max_non_target_translation_m=0.01,
    max_non_target_rotation_rad=0.15,
    max_non_target_velocity_mps=0.05,
)
LIMITS = RolloutLimits()
SEEDS = (0, 1, 2, 3, 4, 5)


def _sequence(speed: float) -> tuple[Primitive, ...]:
    return (
        Primitive(
            kind=PrimitiveKind.PUSH,
            direction=np.array([1.0, 0.0, 0.0]),
            speed=speed,
            max_duration_s=0.4,
            until=TransitionCondition.TARGET_CONTACT_MADE,
        ),
        Primitive(
            kind=PrimitiveKind.SQUEEZE,
            direction=np.array([1.0, 0.0, 0.0]),
            speed=speed * 0.5,
            max_duration_s=0.4,
            grip=1.0,
        ),
    )


def _roles(model: mujoco.MjModel) -> SceneRoles:
    def gid(name: str) -> int:
        return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)

    def bid(name: str) -> int:
        return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)

    return SceneRoles(
        target_geoms=frozenset({gid("target_geom")}),
        support_geoms=frozenset({gid("table")}),
        non_target_geoms=frozenset(),
        robot_geoms=frozenset({gid("pusher_geom")}),
        # The micro scene has no separate wrist link, so the wrist budget
        # resolves at the body the stage is driven through. Naming it is what
        # makes the wrist limits measurable at all (G01).
        wrist_body=bid("pusher"),
        palm_body=bid("pusher"),
    )


def _run_arm(*, frozen_target: bool) -> dict[str, Any]:
    """One arm of the ablation.

    ``frozen_target`` emulates the static assumption by welding the target in
    place, so the only difference between arms is whether the scene may react.
    """
    xml = MICRO_SCENE.read_text(encoding="utf-8")
    if frozen_target:
        xml = xml.replace('<freejoint name="target_free"/>', "")
    model = mujoco.MjModel.from_xml_string(xml)
    roles = _roles(model)

    ledger = ReasonLedger()
    unsafe = 0
    lifts: list[float] = []
    for seed in SEEDS:
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        pose = SeedPose(
            qpos=np.array(data.qpos),
            ctrl=np.zeros(model.nu),
            source_candidate_id=f"ablation:{'static' if frozen_target else 'dynamic'}:{seed}",
        )
        _, outcome = run_static_seeded_rollout(
            model,
            roles=roles,
            budget=BUDGET,
            seed=pose,
            primitives=_sequence(0.05 + 0.02 * seed),
            horizon=60,
            control_dt=0.01,
            limits=LIMITS,
        )
        ledger.record(outcome)
        if outcome.failure_reason in ("damaging_contact", "forbidden_contact"):
            unsafe += 1
        lifts.append(float(outcome.objective_terms.get("lift_m", 0.0)))

    report = ledger.to_dict()
    return {
        "frozen_target": frozen_target,
        "candidates": len(SEEDS),
        "reason_ledger": report,
        "yield": report["yield"],
        "unsafe_rejections": unsafe,
        "median_lift_m": float(np.median(lifts)),
    }


def run_ablation() -> dict[str, Any]:
    static_arm = _run_arm(frozen_target=True)
    dynamic_arm = _run_arm(frozen_target=False)

    delta = dynamic_arm["yield"] - static_arm["yield"]
    if delta > 0.0 and dynamic_arm["unsafe_rejections"] <= static_arm["unsafe_rejections"]:
        verdict = "dynamic_improves_yield_safely"
    elif delta > 0.0:
        verdict = "dynamic_improves_yield_but_adds_unsafe_contact"
    elif delta == 0.0:
        verdict = "no_measured_difference"
    else:
        verdict = "dynamic_reduces_yield"

    config_hash = hashlib.sha256(
        json.dumps(
            {
                "budget": dataclasses.asdict(BUDGET),
                "limits": dataclasses.asdict(LIMITS),
                "seeds": list(SEEDS),
            },
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()

    return {
        "schema": "qdgrasp/evidence/phase3.4-ablation/v1",
        "work_package": "P3.4-14",
        "config_hash": config_hash,
        "static_arm": static_arm,
        "dynamic_arm": dynamic_arm,
        "yield_delta": delta,
        "verdict": verdict,
        "interpretation": (
            "The arms differ only in whether the target may move. A verdict other "
            "than dynamic_improves_yield_safely is a legitimate result and must be "
            "recorded as such: no threshold may be moved to produce a better one."
        ),
        "scope_limit": (
            "Measured on the micro pusher scene, which has one actuator and cannot "
            "enclose or lift. It exercises the ablation machinery and the reason "
            "accounting end to end; it is not a claim about the release hands."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    report = run_ablation()
    payload = json.dumps(report, indent=2, sort_keys=True)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    print(f"verdict={report['verdict']}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
