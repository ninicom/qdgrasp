"""Ablation study across three grasp proposal and IK recipes in Phase 3.1."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import trimesh

from qdgrasp.dataset.pipeline.contracts import ALLOWED_RECIPES
from qdgrasp.dataset.pipeline.orchestrator import run_pipeline_chunk
from qdgrasp.dataset.rng import get_generator
from qdgrasp.objects.generate import generate_box, generate_sphere, generate_cylinder, generate_capsule
from qdgrasp.objects.schema import SubGeomSpec
from qdgrasp.robot.spec import RobotSpec, resolve_robot_asset

logger = logging.getLogger("ablate_recipes")

def run_ablation(
    recipes: list[str] = list(ALLOWED_RECIPES.keys()),
    num_candidates_per_obj: int = 4,
    seed: int = 42,
) -> dict[str, Any]:
    robot_configs = [
        ("leap_hand", "leap_hand.yaml"),
        ("wonik_allegro", "wonik_allegro.yaml"),
        ("shadow_hand", "shadow_hand.yaml"),
    ]

    # Generate test objects
    obj_defs = [
        ("box", generate_box, {}),
        ("sphere", generate_sphere, {}),
        ("cylinder", generate_cylinder, {}),
        ("capsule", generate_capsule, {}),
    ]

    test_objects = []
    for obj_name, gen_fn, kwargs in obj_defs:
        rng = get_generator(seed, "ablation", obj_name)
        mesh, geoms, _, mass, _ = gen_fn(rng, **kwargs)
        test_objects.append((obj_name, mesh, geoms, mass))

    results: dict[str, Any] = {}

    for recipe_id in recipes:
        logger.info(f"Ablating recipe: {recipe_id}")
        recipe_reasons = {
            "proposal_rejected": 0,
            "ik_rejected": 0,
            "collision_rejected": 0,
            "static_force_rejected": 0,
            "dynamic_rejected": 0,
            "accepted": 0,
        }
        total_candidates = 0

        for r_name, r_cfg in robot_configs:
            spec = RobotSpec.from_config(r_cfg, sample_anchors=False)
            xml_path = resolve_robot_asset(spec.config.source_asset)

            for obj_name, mesh, geoms, mass in test_objects:
                rng = get_generator(seed, recipe_id, r_name, obj_name)
                outcomes, reasons = run_pipeline_chunk(
                    recipe_id=recipe_id,
                    spec=spec,
                    mesh=mesh,
                    collision_geoms=geoms,
                    hand_xml_path=xml_path,
                    rng=rng,
                    num_candidates=num_candidates_per_obj,
                    object_mass=mass,
                    run_dynamic=True,
                )
                total_candidates += len(outcomes)
                for k, v in reasons.items():
                    recipe_reasons[k] += v

        results[recipe_id] = {
            "total_candidates": total_candidates,
            "reasons": recipe_reasons,
            "ik_convergence_rate": float((total_candidates - recipe_reasons["ik_rejected"]) / max(1, total_candidates)),
            "static_pass_rate": float((total_candidates - recipe_reasons["ik_rejected"] - recipe_reasons["collision_rejected"] - recipe_reasons["static_force_rejected"]) / max(1, total_candidates)),
            "dynamic_pass_rate": float(recipe_reasons["accepted"] / max(1, total_candidates)),
        }

    return results

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    results = run_ablation(num_candidates_per_obj=2, seed=42)
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()
