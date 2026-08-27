#!/usr/bin/env python3
"""Generate QDGrasp-ContactRich-Tiny (P3.4-16).

Every sample is produced by the validated `mocap-weld-v3` rollout with the
Phase 3.4 contact observer attached, so the physics is the protocol Phase 3.2.1
already certified and only the contact-rich accounting is new.

Positives and negatives are both released. A negative carries its measured
reason; the plan needs the failures for a critic or safety model later, and
dropping them would also hide what the safety budget actually rejects.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import mujoco

from qdgrasp.dataset.dynamic_contracts import ContactSafetyBudget
from qdgrasp.dataset.dynamic_shards import write_trajectory_shard
from qdgrasp.dynamic.safety import SceneRoles
from qdgrasp.dynamic.wrapped_rollout import run_wrapped_contact_rollout
from qdgrasp.robot.spec import RobotSpec, resolve_robot_asset
from qdgrasp.scenes.release_recipes import build_release_grasp_recipe

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_ID = "qdgrasp-contactrich-tiny"

#: ADR-0008 pauses shadow_hand from default scope on 2026-08-27. It is not
#: dropped from the record: a missing Shadow result must read
#: `paused_by_ADR-0008`, never `pass`, `zero`, `unsupported` or a bare
#: `not_run`, and the three-hand P3.4 contract does not close while it holds.
ACTIVE_HANDS = ("leap_hand", "wonik_allegro")
PAUSED_HANDS = ("shadow_hand",)
HANDS = ACTIVE_HANDS
CFG = {h: f"{h}.yaml" for h in (*ACTIVE_HANDS, *PAUSED_HANDS)}
PAUSE_DECISION = "ADR-0008"

#: Pinned per robot profile before generation, and hashed into the manifest.
#: Impulse is judged over a rolling window: a cumulative limit would reject
#: every sustained hold regardless of how gentle it was.
BUDGETS = {
    hand: ContactSafetyBudget(
        budget_id=f"contactrich-tiny-{hand}-v1",
        robot_profile=hand,
        peak_normal_force_N=50.0,
        peak_tangential_force_N=30.0,
        normal_impulse_Ns=5.0,
        tangential_impulse_Ns=3.0,
        contact_duration_s=20.0,
        contact_work_J=2.0,
        max_penetration_m=0.005,
        max_wrist_force_N=100.0,
        max_wrist_torque_Nm=20.0,
        max_joint_or_tendon_load=50.0,
        max_non_target_translation_m=0.02,
        max_non_target_rotation_rad=0.3,
        max_non_target_velocity_mps=0.2,
    )
    for hand in (*ACTIVE_HANDS, *PAUSED_HANDS)
}

#: Declared negative controls. Each one changes a physical condition, never a
#: threshold, so a rejection is measured rather than arranged.
NEGATIVE_CONTROLS: tuple[tuple[str, str, dict[str, Any]], ...] = (
    ("heavy_object", "target ten times heavier than the validated envelope",
     {"object_mass": 0.05}),
    ("no_closure", "fingers never leave the pregrasp pose",
     {"use_initial_as_closed": True}),
)


def _roles_from_model(model: mujoco.MjModel) -> SceneRoles:
    target_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target_object")
    floor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    target, support, robot = set(), set(), set()
    for geom in range(int(model.ngeom)):
        if int(model.geom_bodyid[geom]) == target_body:
            target.add(geom)
        elif geom == floor:
            support.add(geom)
        else:
            robot.add(geom)
    # A dexterous hand touches itself constantly; self-contact is allowed by
    # identity and still judged by the budget, which is what caught Shadow.
    allow = frozenset((min(a, b), max(a, b)) for a in robot for b in robot)
    return SceneRoles(
        frozenset(target), frozenset(support), frozenset(), frozenset(robot), allow
    )


def _generate_one(hand: str, variant: str, overrides: dict[str, Any]):
    spec = RobotSpec.from_config(CFG[hand], sample_anchors=False)
    recipe = build_release_grasp_recipe(CFG[hand])
    kwargs = dict(recipe.rollout_kwargs)
    if overrides.pop("use_initial_as_closed", False):
        kwargs["joint_targets"] = dict(kwargs["initial_joint_targets"])
    kwargs.update(overrides)
    return run_wrapped_contact_rollout(
        hand_xml_path=resolve_robot_asset(spec.config.source_asset),
        collision_geoms=recipe.target_geoms,
        fingertip_body_names=spec.fingertip_links,
        roles_from_model=_roles_from_model,
        budget=BUDGETS[hand],
        rollout_kwargs=kwargs,
        trajectory_ref=f"{hand}/{variant}",
    )


def generate(output_dir: Path) -> dict[str, Any]:
    samples: dict[str, list] = {"train": [], "val": []}
    records: list[dict[str, Any]] = []

    for hand in HANDS:
        variants = [("baseline", "validated release recipe, target free to react", {})]
        variants += [(name, why, dict(over)) for name, why, over in NEGATIVE_CONTROLS]
        for variant, rationale, overrides in variants:
            trajectory, outcome, validation = _generate_one(hand, variant, overrides)
            # Split by variant so a negative control never shares a split with
            # the baseline it was derived from.
            split = "val" if variant == "baseline" and hand == "shadow_hand" else "train"
            samples[split].append((trajectory, outcome))
            records.append(
                {
                    "hand": hand,
                    "variant": variant,
                    "rationale": rationale,
                    "split": split,
                    "passed": bool(outcome.passed),
                    "failure_reason": outcome.failure_reason,
                    "validated_rollout_passed": bool(validation.passed),
                    "lift_m": round(float(outcome.objective_terms.get("lift_m", 0.0)), 5),
                    "peak_normal_force_N": round(
                        float(outcome.peak_safety_metrics.get("peak_normal_force_N", 0.0)), 3
                    ),
                    "min_budget_margin": round(
                        float(outcome.peak_safety_metrics.get("min_budget_margin", 0.0)), 4
                    ),
                    "contact_events": len(trajectory.contact_graph),
                    "steps": trajectory.num_steps,
                }
            )
            print(
                f"  {hand:14s} {variant:14s} passed={outcome.passed!s:5s} "
                f"reason={outcome.failure_reason}",
                flush=True,
            )

    shard_hashes = {}
    for split, entries in samples.items():
        if not entries:
            continue
        path = output_dir / "shards" / f"{split}.json"
        shard_hashes[f"shards/{split}.json"] = write_trajectory_shard(path, entries)

    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=REPO_ROOT, check=True, capture_output=True, text=True,
        ).stdout.strip()
    )

    for hand in PAUSED_HANDS:
        records.append(
            {
                "hand": hand,
                "variant": "not_generated",
                "rationale": f"paused from default scope by {PAUSE_DECISION}",
                "split": "none",
                "status": f"paused_by_{PAUSE_DECISION}",
                "passed": False,
            }
        )

    positives = [r for r in records if r["passed"]]
    hands_with_positive = sorted({r["hand"] for r in positives})
    manifest = {
        "dataset_id": DATASET_ID,
        "schema": "qdgrasp/contactrich-manifest/v1",
        "generator_commit": commit,
        "generator_worktree_dirty": dirty,
        "protocol": "mocap-weld-v3 via validate_grasp_rollout",
        "generation_mode": "static_seeded_contact_rollout",
        "safety_budgets": {
            h: dataclasses.asdict(b) for h, b in sorted(BUDGETS.items())
        },
        "records": records,
        "shards": shard_hashes,
        "counts": {
            "samples": len(records),
            "positives": len(positives),
            "negatives": len(records) - len(positives),
            "hands_with_positive": hands_with_positive,
        },
        "active_hands": list(ACTIVE_HANDS),
        "paused_hands": list(PAUSED_HANDS),
        "paused_by": PAUSE_DECISION,
        "coverage_claim": (
            "two active hands; this is NOT three-hand coverage and must not be "
            "described as such"
        ),
        # The three-hand P3.4 contract cannot close while ADR-0008 holds, and a
        # two-hand release needs its own successor scope rather than reusing the
        # P3.4 verdict. So release stays blocked even with both active hands
        # positive.
        "release_blocked": True,
        "release_block_reason": (
            f"{PAUSE_DECISION} pauses {', '.join(PAUSED_HANDS)}; the three-hand "
            "P3.4 contract does not close during the pause, and a two-hand "
            "dynamic-data release requires a successor scope rather than the "
            "existing P3.4 verdict"
        ),
        "active_hands_with_positive": hands_with_positive,
        "active_coverage_complete": sorted(hands_with_positive) == sorted(ACTIVE_HANDS),
    }
    path = output_dir / "dataset_manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/contactrich-tiny"))
    args = parser.parse_args()
    print(f"Generating {DATASET_ID} into {args.output_dir}")
    manifest = generate(args.output_dir)
    print(json.dumps(manifest["counts"], indent=2))
    print(f"release_blocked={manifest['release_blocked']}", file=sys.stderr)
    if manifest["release_blocked"]:
        print(manifest["release_block_reason"], file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
