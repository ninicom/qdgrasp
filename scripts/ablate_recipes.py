"""Ablation study across three grasp proposal and IK recipes in Phase 3.1."""

from __future__ import annotations

import argparse
import json
import logging
from typing import Any

from qdgrasp.dataset.pipeline.contracts import ALLOWED_RECIPES
from qdgrasp.dataset.pipeline.orchestrator import run_pipeline_chunk
from qdgrasp.dataset.rng import get_generator
from qdgrasp.objects.generate import generate_box, generate_sphere, generate_cylinder, generate_capsule
from qdgrasp.robot.spec import RobotSpec, resolve_robot_asset

logger = logging.getLogger("ablate_recipes")

MAX_TOTAL_CANDIDATES = 96


def _stage_rates(total: int, reasons: dict[str, int]) -> dict[str, float]:
    """Compute conditional conversion rates without crediting skipped stages."""
    proposal_attempted = total
    proposal_passed = proposal_attempted - reasons["proposal_rejected"]
    ik_attempted = proposal_passed
    ik_passed = ik_attempted - reasons["ik_rejected"]
    collision_attempted = ik_passed
    collision_passed = collision_attempted - reasons["collision_rejected"]
    static_attempted = collision_passed
    static_passed = static_attempted - reasons["static_force_rejected"]
    dynamic_attempted = reasons["dynamic_rejected"] + reasons["accepted"]

    def rate(passed: int, attempted: int) -> float:
        return float(passed / attempted) if attempted else 0.0

    return {
        "proposal_yield": rate(proposal_passed, proposal_attempted),
        "ik_convergence_rate": rate(ik_passed, ik_attempted),
        "collision_pass_rate": rate(collision_passed, collision_attempted),
        "static_pass_rate": rate(static_passed, static_attempted),
        "dynamic_pass_rate": rate(reasons["accepted"], dynamic_attempted),
    }

def run_ablation(
    recipes: list[str] | None = None,
    num_candidates_per_obj: int = 4,
    seed: int = 42,
    run_dynamic: bool = True,
) -> dict[str, Any]:
    if recipes is None:
        recipes = list(ALLOWED_RECIPES.keys())
    unknown = sorted(set(recipes) - set(ALLOWED_RECIPES))
    if unknown:
        raise ValueError(f"unknown recipes: {unknown}")
    if not 1 <= num_candidates_per_obj <= 4:
        raise ValueError("num_candidates_per_obj must be in [1, 4]")

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
    planned_candidates = (
        len(recipes)
        * len(robot_configs)
        * len(obj_defs)
        * num_candidates_per_obj
    )
    if planned_candidates > MAX_TOTAL_CANDIDATES:
        raise ValueError(
            f"planned ablation has {planned_candidates} candidates; "
            f"hard limit is {MAX_TOTAL_CANDIDATES}"
        )

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
            "dynamic_skipped": 0,
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
                    run_dynamic=run_dynamic,
                )
                total_candidates += len(outcomes)
                for k, v in reasons.items():
                    recipe_reasons[k] += v
                logger.info(
                    "completed recipe=%s robot=%s object=%s candidates=%d",
                    recipe_id,
                    r_name,
                    obj_name,
                    len(outcomes),
                )

        accounted = sum(recipe_reasons.values())
        if accounted != total_candidates:
            raise RuntimeError(
                f"reason accounting mismatch for {recipe_id}: "
                f"{accounted} != {total_candidates}"
            )

        results[recipe_id] = {
            "total_candidates": total_candidates,
            "reasons": recipe_reasons,
            **_stage_rates(total_candidates, recipe_reasons),
        }

    return results

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--num-candidates-per-object", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--recipe",
        action="append",
        choices=sorted(ALLOWED_RECIPES),
        dest="recipes",
    )
    parser.add_argument("--skip-dynamic", action="store_true")
    args = parser.parse_args()
    recipes = args.recipes or list(ALLOWED_RECIPES)
    planned = len(recipes) * 3 * 4 * args.num_candidates_per_object
    if not args.execute:
        print(
            json.dumps(
                {
                    "status": "DRY_RUN",
                    "planned_candidates": planned,
                    "recipes": recipes,
                    "run_dynamic": not args.skip_dynamic,
                    "message": "pass --execute to start the bounded ablation",
                },
                indent=2,
            )
        )
        return
    results = run_ablation(
        recipes=recipes,
        num_candidates_per_obj=args.num_candidates_per_object,
        seed=args.seed,
        run_dynamic=not args.skip_dynamic,
    )
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()
