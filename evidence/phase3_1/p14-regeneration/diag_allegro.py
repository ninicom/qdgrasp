"""Characterize WHERE Allegro candidates die across the grasp-bar envelope.

Kinematics only (run_dynamic=False). This maps the failure surface; it does
not search for or admit a positive. Bounded, declared grid.
"""
from __future__ import annotations

import argparse, json, time
from pathlib import Path

from qdgrasp.dataset.pipeline.generated_reachable import build_grasp_bar, generated_reachable_rng
from qdgrasp.dataset.pipeline.orchestrator import run_pipeline_chunk
from qdgrasp.robot.spec import RobotSpec, resolve_robot_asset

CFGS = {"leap_hand": "leap_hand.yaml", "wonik_allegro": "wonik_allegro.yaml",
        "shadow_hand": "shadow_hand.yaml"}

WIDTHS = [0.030, 0.035, 0.040, 0.045, 0.050]
HEIGHTS = [0.085, 0.100, 0.115, 0.130, 0.140]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hand", default="wonik_allegro", choices=tuple(CFGS))
    ap.add_argument("--budget", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--recipe", default="region_opposition_v1")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    spec = RobotSpec.from_config(CFGS[args.hand], sample_anchors=False)
    xml = resolve_robot_asset(spec.config.source_asset)

    rows = []
    for w in WIDTHS:
        for z in HEIGHTS:
            bar = build_grasp_bar(args.hand, upper_center_z=z)
            # rebuild at the swept width by scaling the pinned constructor input
            from qdgrasp.dataset.pipeline import generated_reachable as gr
            saved = gr._PROFILE_WIDTHS[args.hand]
            gr._PROFILE_WIDTHS[args.hand] = w
            try:
                bar = build_grasp_bar(args.hand, upper_center_z=z)
            finally:
                gr._PROFILE_WIDTHS[args.hand] = saved

            t0 = time.perf_counter()
            outcomes, reasons = run_pipeline_chunk(
                recipe_id=args.recipe, spec=spec, mesh=bar.mesh,
                collision_geoms=bar.collision_geoms, hand_xml_path=xml,
                rng=generated_reachable_rng(args.hand, args.seed),
                num_candidates=args.budget, object_mass=bar.mass,
                object_pos=bar.object_pos, run_dynamic=False,
            )
            stages = {}
            for o in outcomes:
                key = o.failure_stage or "reached_dynamic"
                stages[key] = stages.get(key, 0) + 1
            row = {
                "hand": args.hand, "width": w, "upper_center_z": z,
                "reasons": {k: v for k, v in reasons.items() if v},
                "stages": stages,
                "reached_dynamic": sum(
                    1 for o in outcomes
                    if o.proposal_valid and o.ik_valid and o.collision_valid
                    and o.static_force_valid
                ),
                "seconds": round(time.perf_counter() - t0, 1),
            }
            rows.append(row)
            print(json.dumps(row), flush=True)
    Path(args.out).write_text(json.dumps(rows, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
