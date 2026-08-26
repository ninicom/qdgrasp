"""Probe: reproduce the validated positive-control cell and test val-side variants.

Read-only experiment. Writes one JSON report; changes nothing in the repo.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import trimesh

from qdgrasp.dataset.pipeline.generated_reachable import (
    build_generated_reachable_object,
    generated_reachable_rng,
)
from qdgrasp.dataset.pipeline.orchestrator import run_pipeline_chunk
from qdgrasp.objects.schema import SubGeomSpec
from qdgrasp.robot.spec import RobotSpec, resolve_robot_asset

BUDGETS = {"leap_hand": 4, "wonik_allegro": 14, "shadow_hand": 10}
CFGS = {
    "leap_hand": "leap_hand.yaml",
    "wonik_allegro": "wonik_allegro.yaml",
    "shadow_hand": "shadow_hand.yaml",
}
WIDTHS = {"leap_hand": 0.040, "wonik_allegro": 0.040, "shadow_hand": 0.050}
UPPER_Z = {"leap_hand": 0.140, "wonik_allegro": 0.085, "shadow_hand": 0.140}


def build_bar(width: float, upper_center_z: float, upper_height: float):
    """Same construction as build_generated_reachable_object, parametrized."""
    stem_width = 0.008
    stem_height = upper_center_z - 0.5 * upper_height
    upper = trimesh.creation.box(extents=(width, width, upper_height))
    upper.apply_translation((0.0, 0.0, upper_center_z))
    stem = trimesh.creation.box(extents=(stem_width, stem_width, stem_height))
    stem.apply_translation((0.0, 0.0, 0.5 * stem_height))
    mesh = trimesh.util.concatenate((upper, stem))
    geoms = (
        SubGeomSpec(
            type="box",
            size=(0.5 * width, 0.5 * width, 0.5 * upper_height),
            pos=(0.0, 0.0, upper_center_z),
        ),
        SubGeomSpec(
            type="box",
            size=(0.5 * stem_width, 0.5 * stem_width, 0.5 * stem_height),
            pos=(0.0, 0.0, 0.5 * stem_height),
        ),
    )
    return mesh, geoms


def variants(hand: str, sweep: str) -> list[dict]:
    w = WIDTHS[hand]
    z = UPPER_Z[hand]
    if sweep == "upper_height":
        out = [{"name": "A_exact", "kind": "fixture"}]
        for h in (0.045, 0.055):
            out.append(
                {
                    "name": f"B_upper_height_{h:.3f}",
                    "kind": "sweep",
                    "width": w,
                    "upper_center_z": z,
                    "upper_height": h,
                }
            )
        return out
    if sweep == "upper_center_z":
        # Pre-registered second sweep: exactly two values, +/- 5 mm on the
        # per-hand calibrated graspable height. No further values are run.
        return [
            {
                "name": f"C_upper_center_z_{z + d:.3f}",
                "kind": "sweep",
                "width": w,
                "upper_center_z": z + d,
                "upper_height": 0.050,
            }
            for d in (-0.005, 0.005)
        ]
    raise ValueError(f"unsupported sweep: {sweep}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hand", required=True, choices=tuple(BUDGETS))
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--recipe", default="region_opposition_v1")
    ap.add_argument("--budget", type=int, default=None,
                    help="Override candidate budget; must not exceed the validated ceiling.")
    ap.add_argument("--sweep", default="upper_height",
                    choices=("upper_height", "upper_center_z"))
    args = ap.parse_args()

    spec = RobotSpec.from_config(CFGS[args.hand], sample_anchors=False)
    xml_path = resolve_robot_asset(spec.config.source_asset)
    fixture = build_generated_reachable_object(args.hand)
    budget = args.budget if args.budget is not None else BUDGETS[args.hand]
    if budget > fixture.candidate_budget:
        raise SystemExit(
            f"budget {budget} exceeds validated ceiling {fixture.candidate_budget}"
        )

    results = []
    for var in variants(args.hand, args.sweep):
        if var["kind"] == "fixture":
            mesh, geoms = fixture.mesh, fixture.collision_geoms
        else:
            mesh, geoms = build_bar(
                var["width"], var["upper_center_z"], var["upper_height"]
            )
        rng = generated_reachable_rng(args.hand, args.seed)
        started = time.perf_counter()
        outcomes, reasons = run_pipeline_chunk(
            recipe_id=args.recipe,
            spec=spec,
            mesh=mesh,
            collision_geoms=geoms,
            hand_xml_path=xml_path,
            rng=rng,
            num_candidates=budget,
            object_mass=fixture.mass,
            object_pos=fixture.object_pos,
            run_dynamic=True,
        )
        row = {
            "hand": args.hand,
            "variant": var["name"],
            "params": {k: v for k, v in var.items() if k not in ("name", "kind")},
            "budget": budget,
            "observed": len(outcomes),
            "reasons": dict(reasons),
            "dynamic_positive": sum(1 for o in outcomes if o.dynamic_valid),
            "runtime_seconds": round(time.perf_counter() - started, 3),
        }
        results.append(row)
        print(json.dumps(row), flush=True)

    Path(args.out).write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
