"""Bounded dynamic confirmation of the three best Allegro cells from the grid.

Declared before running: exactly the three (width, upper_center_z) cells with
the most survivors in the kinematics grid, at the fixture's ceiling budget 16.
No further cells are run.
"""
from __future__ import annotations
import json, time
from pathlib import Path

from qdgrasp.dataset.pipeline import generated_reachable as gr
from qdgrasp.dataset.pipeline.generated_reachable import build_grasp_bar, generated_reachable_rng
from qdgrasp.dataset.pipeline.orchestrator import run_pipeline_chunk
from qdgrasp.robot.spec import RobotSpec, resolve_robot_asset

HAND = "wonik_allegro"
CELLS = [(0.045, 0.140), (0.045, 0.130), (0.050, 0.115)]
BUDGET = 16
SEED = 42

spec = RobotSpec.from_config("wonik_allegro.yaml", sample_anchors=False)
xml = resolve_robot_asset(spec.config.source_asset)
rows = []
for w, z in CELLS:
    saved = gr._PROFILE_WIDTHS[HAND]
    gr._PROFILE_WIDTHS[HAND] = w
    try:
        bar = build_grasp_bar(HAND, upper_center_z=z)
    finally:
        gr._PROFILE_WIDTHS[HAND] = saved
    t0 = time.perf_counter()
    outcomes, reasons = run_pipeline_chunk(
        recipe_id="region_opposition_v1", spec=spec, mesh=bar.mesh,
        collision_geoms=bar.collision_geoms, hand_xml_path=xml,
        rng=generated_reachable_rng(HAND, SEED), num_candidates=BUDGET,
        object_mass=bar.mass, object_pos=bar.object_pos, run_dynamic=True,
    )
    row = {"width": w, "upper_center_z": z, "budget": BUDGET,
           "dynamic_positive": sum(1 for o in outcomes if o.dynamic_valid),
           "reasons": {k: v for k, v in reasons.items() if v},
           "seconds": round(time.perf_counter() - t0, 1)}
    rows.append(row)
    print(json.dumps(row), flush=True)
Path("/tmp/claude-1000/-run-media-quyen-H-H-Project-2026-qdgrasp/7c23d7f7-2214-43f6-ad41-9bebb740212c/scratchpad/confirm_allegro.json").write_text(json.dumps(rows, indent=2))
