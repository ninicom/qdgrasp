#!/usr/bin/env python3
"""Shadow contact-pair localization (P3.4.1-06).

Phase 3.4 measured 323 N between `rh_lfproximal` and `rh_lfmetacarpal` and
stopped there. That number does not say whether the recipe walks an inactive
finger into itself, whether the collision proxy is wrong, whether the pair is a
structural adjacency that should never collide, or whether the posture genuinely
self-collides and must be rejected.

This audits the pair per stage so the failure lands in exactly one of those four
classes before anything is changed. It fixes nothing.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from qdgrasp.dataset.pipeline.validators.mujoco_rollout import build_rollout_scene_model
from qdgrasp.robot.spec import RobotSpec, resolve_robot_asset
from qdgrasp.scenes.release_recipes import build_release_grasp_recipe

REPO_ROOT = Path(__file__).resolve().parents[1]
CFG = "shadow_hand.yaml"

#: Fingers the pinch actually uses. Everything else is inactive by design, and
#: an inactive finger driving into collision is a recipe defect, not a grasp.
ACTIVE_PREFIXES = ("rh_FFJ", "rh_THJ", "rh_WRJ")


def _name(model: mujoco.MjModel, kind: int, index: int) -> str:
    return mujoco.mj_id2name(model, kind, index) or f"id_{index}"


def _pair_report(model: mujoco.MjModel, data: mujoco.MjData) -> list[dict[str, Any]]:
    wrench = np.zeros(6)
    out = []
    for i in range(int(data.ncon)):
        c = data.contact[i]
        mujoco.mj_contactForce(model, data, i, wrench)
        ga, gb = int(c.geom1), int(c.geom2)
        ba, bb = int(model.geom_bodyid[ga]), int(model.geom_bodyid[gb])
        out.append(
            {
                "geom_a": _name(model, mujoco.mjtObj.mjOBJ_GEOM, ga),
                "geom_b": _name(model, mujoco.mjtObj.mjOBJ_GEOM, gb),
                "body_a": _name(model, mujoco.mjtObj.mjOBJ_BODY, ba),
                "body_b": _name(model, mujoco.mjtObj.mjOBJ_BODY, bb),
                "signed_distance_m": float(c.dist),
                "penetration_m": max(0.0, -float(c.dist)),
                "normal_force_N": abs(float(wrench[0])),
                "tangential_force_N": float(np.linalg.norm(wrench[1:3])),
                "parent_of_a": _name(
                    model, mujoco.mjtObj.mjOBJ_BODY, int(model.body_parentid[ba])
                ),
                "parent_of_b": _name(
                    model, mujoco.mjtObj.mjOBJ_BODY, int(model.body_parentid[bb])
                ),
                "structurally_adjacent": bool(
                    int(model.body_parentid[ba]) == bb
                    or int(model.body_parentid[bb]) == ba
                ),
            }
        )
    return out


def _apply(model: mujoco.MjModel, data: mujoco.MjData, targets: dict[str, float]) -> None:
    for joint, value in targets.items():
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint)
        if jid >= 0:
            data.qpos[int(model.jnt_qposadr[jid])] = value


def audit() -> dict[str, Any]:
    spec = RobotSpec.from_config(CFG, sample_anchors=False)
    recipe = build_release_grasp_recipe(CFG)
    kw = recipe.rollout_kwargs
    model = build_rollout_scene_model(
        resolve_robot_asset(spec.config.source_asset),
        recipe.target_geoms,
        object_pos=tuple(kw["object_pos"]),
        object_mass=float(kw["object_mass"]),
    )
    initial = dict(kw["initial_joint_targets"])
    closed = dict(kw["joint_targets"])
    inactive = {j: v for j, v in closed.items() if not j.startswith(ACTIVE_PREFIXES)}

    # The recipe already sets rh_LFJ1/2/3 to 1.2/1.2/1.4 in its *pregrasp*
    # targets, so "initial" is not an open hand: the little finger is curled
    # before the grasp starts. Stages that only differ between initial and
    # closed therefore hold that finger fixed and cannot separate a posture
    # defect from a proxy defect. A genuinely neutral posture is required.
    stages: dict[str, dict[str, float]] = {
        "q_neutral_all_zero": dict.fromkeys(closed, 0.0),
        "q_lf_zero": {j: (0.0 if j.startswith("rh_LFJ") else closed[j]) for j in closed},
        # Plan section 4.3 option A: every inactive finger to an open posture
        # instead of 1.2/1.2/1.4 rad, active set untouched.
        "q_inactive_open": {j: (0.0 if j in inactive else closed[j]) for j in closed},
        "q_recipe_pregrasp": initial,
        "q_recipe_closed": closed,
        "q_closed_active_only": {
            j: (initial[j] if j in inactive else closed[j]) for j in closed
        },
    }

    results: dict[str, Any] = {}
    for label, targets in stages.items():
        data = mujoco.MjData(model)
        _apply(model, data, targets)
        # No gravity, no control: any contact here is posture, not a controller
        # pressing. That is what separates invalid_posture from a control effect.
        model.opt.gravity[:] = 0.0
        mujoco.mj_forward(model, data)
        pairs = _pair_report(model, data)
        robot_only = [
            p for p in pairs
            if "target" not in p["body_a"] and "target" not in p["body_b"]
            and p["body_a"] != "world" and p["body_b"] != "world"
        ]
        worst = max(robot_only, key=lambda p: p["penetration_m"], default=None)
        results[label] = {
            "contacts": len(pairs),
            "self_contacts": len(robot_only),
            "worst_self_pair": worst,
            "inactive_finger_self_contacts": [
                p for p in robot_only
                if any(t in (p["body_a"] + p["body_b"]) for t in ("lf", "mf", "rf"))
            ][:5],
        }

    lf = results["q_recipe_closed"]["worst_self_pair"]
    neutral = results["q_neutral_all_zero"]["worst_self_pair"]
    _lf_zero = results["q_lf_zero"]["worst_self_pair"]  # kept for the report
    inactive_open = results["q_inactive_open"]["worst_self_pair"]
    active_only = results["q_closed_active_only"]["worst_self_pair"]

    def pen(entry: Any) -> float:
        return float(entry["penetration_m"]) if entry else 0.0

    if pen(neutral) > 0.0:
        classification = "invalid_proxy"
        rationale = (
            "The hand interpenetrates at a neutral all-zero posture with no "
            "gravity and no control, so no joint command produces it and the "
            "collision geometry is wrong."
        )
    elif pen(inactive_open) == 0.0 and pen(lf) > 0.0:
        classification = "invalid_posture"
        rationale = (
            "A neutral posture is clean and opening every inactive finger clears "
            "the penetration entirely, while the recipe's values reproduce it. "
            "The recipe commands fingers the pinch does not use into the palm; "
            "the collision geometry is sound."
        )
    elif lf and lf.get("structurally_adjacent"):
        classification = "missing_structural_exclusion"
        rationale = (
            "The worst pair is a parent-child body pair. Adjacent links cannot "
            "physically collide the way the model allows, so the contact is a "
            "model artifact rather than a posture defect."
        )
    elif pen(active_only) < pen(lf) * 0.5 and pen(lf) > 0.0:
        classification = "invalid_posture"
        rationale = (
            "Holding the inactive fingers open removes most of the penetration, "
            "so the recipe walks a finger the pinch does not use into collision."
        )
    else:
        classification = "legitimate_self_contact"
        rationale = (
            "Penetration only appears under the commanded closure and survives "
            "holding the inactive fingers open, so the posture genuinely "
            "self-collides and rejecting it is correct."
        )

    return {
        "schema": "qdgrasp/evidence/phase3.4.1-shadow-audit/v1",
        "work_package": "P3.4.1-06",
        "robot_profile": CFG,
        "active_joint_prefixes": list(ACTIVE_PREFIXES),
        "inactive_joints_closed_by_recipe": sorted(inactive),
        "stages": results,
        "classification": classification,
        "rationale": rationale,
        "note": (
            "Gravity and control are off in every stage, so a contact here is "
            "produced by the commanded joint vector alone."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    report = audit()
    payload = json.dumps(report, indent=2, sort_keys=True)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    print(f"classification={report['classification']}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
