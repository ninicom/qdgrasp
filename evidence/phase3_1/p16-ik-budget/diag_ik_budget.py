"""Is IK failing on budget, or is it stuck?

Records terminal residuals of IK-rejected candidates on the canonical
procedural objects, split into position vs normal. Then reruns the same cells
with a raised iteration budget (monkeypatched in this script only -- production
code is untouched) to measure whether the budget is the binding constraint.

Kinematics only. Diagnostic, not an admission path.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np

from qdgrasp.dataset.pipeline import orchestrator as orch
from qdgrasp.dataset.pipeline.orchestrator import run_pipeline_chunk
from qdgrasp.dataset.rng import get_generator
from qdgrasp.objects.generate import generate_box, generate_cylinder, generate_sphere, generate_capsule
from qdgrasp.robot.spec import RobotSpec, resolve_robot_asset

CFGS = {"leap_hand": "leap_hand.yaml", "wonik_allegro": "wonik_allegro.yaml",
        "shadow_hand": "shadow_hand.yaml"}
OBJS = [("box", generate_box), ("sphere", generate_sphere),
        ("cylinder", generate_cylinder), ("capsule", generate_capsule)]
POS_TOL, NORMAL_TOL_DOT = 0.001, 0.866
BUDGET = 8
SEED = 42


def patch_max_iter(value: int):
    """Rewrite both hardcoded 40s in the orchestrator source at import level."""
    src = Path(orch.__file__).read_text()
    return src.count('"max_iter": 40') + src.count("max_iter=40,")


def run(hand: str, per_stage_iter: int) -> list[dict]:
    spec = RobotSpec.from_config(CFGS[hand], sample_anchors=False)
    xml = resolve_robot_asset(spec.config.source_asset)
    rows = []
    for oname, gen in OBJS:
        rng0 = get_generator(SEED, "ikdiag", oname)
        mesh, geoms, _, mass, _ = gen(rng0)
        outcomes, reasons = run_pipeline_chunk(
            recipe_id="region_opposition_v1", spec=spec, mesh=mesh,
            collision_geoms=geoms, hand_xml_path=xml,
            rng=get_generator(SEED, "ikdiag", hand, oname),
            num_candidates=BUDGET, object_mass=mass, run_dynamic=False,
        )
        pos_res, norm_res = [], []
        for o in outcomes:
            if o.kinematics is None:
                continue
            k = o.kinematics
            act = np.asarray(k.active_fingers, dtype=bool) if hasattr(k, "active_fingers") else None
            pr = np.asarray(k.position_residuals, dtype=float).ravel()
            nr = np.asarray(k.normal_residuals, dtype=float).ravel()
            if act is not None and act.size == pr.size:
                pr, nr = pr[act.ravel()], nr[act.ravel()]
            if pr.size:
                pos_res.append(float(np.max(pr)))
            if nr.size:
                norm_res.append(float(np.max(nr)))
        rows.append({
            "hand": hand, "object": oname, "per_stage_iter": per_stage_iter,
            "reasons": {k: v for k, v in reasons.items() if v},
            "max_pos_residual_p50": float(np.median(pos_res)) if pos_res else None,
            "max_pos_residual_min": float(np.min(pos_res)) if pos_res else None,
            "max_norm_residual_p50": float(np.median(norm_res)) if norm_res else None,
            "max_norm_residual_min": float(np.min(norm_res)) if norm_res else None,
            "n_with_kinematics": len(pos_res),
        })
        print(json.dumps(rows[-1]), flush=True)
    return rows


if __name__ == "__main__":
    hand = sys.argv[1]
    out = Path(sys.argv[2])
    out.write_text(json.dumps(run(hand, 40), indent=2))
