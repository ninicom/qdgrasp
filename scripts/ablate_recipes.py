"""Ablation study across three grasp proposal and IK recipes in Phase 3.1."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import logging
import subprocess
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from qdgrasp.config.active_scope import historical_reproduction_scope
from qdgrasp.dataset.pipeline.contracts import ALLOWED_RECIPES
from qdgrasp.dataset.pipeline.generated_reachable import (
    build_generated_reachable_object,
    generated_reachable_rng,
)
from qdgrasp.dataset.pipeline.orchestrator import run_pipeline_chunk
from qdgrasp.dataset.pipeline.validators.dynamic_predicate import RolloutProtocol
from qdgrasp.dataset.rng import get_generator
from qdgrasp.objects.generate import generate_box, generate_capsule, generate_cylinder, generate_sphere
from qdgrasp.robot.spec import RobotSpec, resolve_robot_asset

logger = logging.getLogger("ablate_recipes")

MAX_TOTAL_CANDIDATES = 96
PINNED_CANDIDATES_PER_CELL = 2
ABLATION_MATRICES = ("canonical", "positive-control")
POSITIVE_CONTROL_BUDGETS = {
    "leap_hand": 4,
    "wonik_allegro": 14,
    "shadow_hand": 10,
}
REQUIRED_RECIPES = tuple(ALLOWED_RECIPES)
#: The P3.2 recipe ablation was run and pinned before ADR-0008, so reproducing
#: it needs the same three hands. The selection is declared through the registry
#: as a historical reproduction; nothing it produces is release evidence for the
#: active scope (G05).
ABLATION_ARTIFACT_ID = "phase3-2-recipe-ablation"
ABLATION_SCOPE = historical_reproduction_scope(
    ABLATION_ARTIFACT_ID, ("leap_hand", "wonik_allegro", "shadow_hand")
)
REQUIRED_HANDS = ABLATION_SCOPE.hands
REPO_ROOT = Path(__file__).resolve().parents[1]
PROVENANCE_SOURCES = (
    "scripts/ablate_recipes.py",
    "qdgrasp/dataset/pipeline/orchestrator.py",
    "qdgrasp/dataset/pipeline/contracts.py",
    "qdgrasp/dataset/pipeline/generated_reachable.py",
    "qdgrasp/dataset/pipeline/proposals/surface_fixed.py",
    "qdgrasp/dataset/pipeline/proposals/region_opposition.py",
    "qdgrasp/dataset/pipeline/proposals/wrench_guided.py",
    "qdgrasp/dataset/pipeline/solvers/fixed_contact_dls.py",
    "qdgrasp/dataset/pipeline/solvers/region_dls.py",
    "qdgrasp/dataset/pipeline/validators/collision_admission.py",
    "qdgrasp/dataset/pipeline/validators/dynamic_predicate.py",
    "qdgrasp/dataset/pipeline/validators/mujoco_rollout.py",
)


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


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _mesh_sha256(mesh: Any) -> str:
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(mesh.vertices, dtype=np.float64).tobytes())
    digest.update(np.ascontiguousarray(mesh.faces, dtype=np.int64).tobytes())
    return digest.hexdigest()


def _active_contact_signature(outcome: Any) -> str | None:
    if not outcome.ik_valid or outcome.kinematics is None:
        return None
    contacts = outcome.kinematics.surface_contacts
    if contacts is None:
        contacts = outcome.kinematics.achieved_contacts
    contacts_array = np.asarray(contacts, dtype=np.float64)
    if contacts_array.ndim == 3:
        contacts_array = contacts_array[0]
    proposal = outcome.proposal
    if proposal is None or proposal.active_fingers is None:
        active = np.ones(contacts_array.shape[0], dtype=bool)
    else:
        active = np.asarray(proposal.active_fingers, dtype=bool)
    active_contacts = np.round(contacts_array[active], decimals=4)
    if active_contacts.size == 0:
        return None
    return hashlib.sha256(np.ascontiguousarray(active_contacts).tobytes()).hexdigest()[:20]


def _summarize_outcomes(outcomes: list[Any]) -> dict[str, Any]:
    ik_signatures: set[str] = set()
    accepted_signatures: set[str] = set()
    penetrations: list[float] = []
    wrench_signatures: set[str] = set()
    failure_signature: Counter[str] = Counter()
    candidate_diagnostics: list[dict[str, Any]] = []
    for outcome in outcomes:
        failure_signature[f"{outcome.failure_stage}:{outcome.failure_reason}"] += 1
        signature = _active_contact_signature(outcome)
        if signature is not None:
            ik_signatures.add(signature)
            if outcome.dynamic_valid:
                accepted_signatures.add(signature)
        if outcome.collision_admission is not None:
            penetrations.append(float(outcome.collision_admission.max_penetration))
        if outcome.static_certificate is not None:
            wrench = np.round(
                np.asarray(outcome.static_certificate.object_wrench, dtype=np.float64),
                decimals=6,
            )
            wrench_signatures.add(hashlib.sha256(np.ascontiguousarray(wrench).tobytes()).hexdigest()[:20])
        diagnostic: dict[str, Any] = {
            "failure_stage": outcome.failure_stage,
            "failure_reason": outcome.failure_reason,
        }
        if outcome.proposal is not None:
            diagnostic["proposal_id"] = outcome.proposal.candidate_id
            diagnostic["active_fingers"] = [
                bool(value) for value in np.asarray(outcome.proposal.active_fingers).reshape(-1)
            ]
        if outcome.kinematics is not None and outcome.proposal is not None:
            active = np.asarray(outcome.proposal.active_fingers, dtype=bool)
            diagnostic["solver_reason"] = str(outcome.kinematics.reason)
            diagnostic["max_active_position_residual"] = float(
                np.max(np.asarray(outcome.kinematics.position_residuals)[active])
            )
            diagnostic["max_active_normal_residual"] = float(
                np.max(np.asarray(outcome.kinematics.normal_residuals)[active])
            )
        if outcome.collision_admission is not None:
            diagnostic["collision_reason"] = outcome.collision_admission.reason
        candidate_diagnostics.append(diagnostic)
    return {
        "failure_signature": dict(sorted(failure_signature.items())),
        "candidate_diagnostics": candidate_diagnostics,
        "ik_contact_signatures": sorted(ik_signatures),
        "accepted_contact_signatures": sorted(accepted_signatures),
        "wrench_signatures": sorted(wrench_signatures),
        "penetration_observations": len(penetrations),
        "max_penetration": max(penetrations) if penetrations else None,
        "median_penetration": float(np.median(penetrations)) if penetrations else None,
    }


def _selection_decision(recipe_results: dict[str, dict[str, Any]], *, run_dynamic: bool) -> dict[str, Any]:
    """Apply the pre-registered fail-closed release-recipe selection rule."""
    criterion = (
        "require all three hands to have measured dynamic positives; require at "
        "least two distinct accepted contact signatures; then maximize "
        "(accepted_cells, accepted, distinct_accepted_contacts, static_passed, "
        "collision_passed, ik_passed, proposal_passed); exact ties are inconclusive"
    )
    if not run_dynamic:
        return {"status": "inconclusive", "winner": None, "reason": "dynamic_disabled", "criterion": criterion}
    if set(recipe_results) != set(REQUIRED_RECIPES):
        return {"status": "inconclusive", "winner": None, "reason": "incomplete_recipe_matrix", "criterion": criterion}

    eligible: dict[str, tuple[int, ...]] = {}
    for recipe_id, result in recipe_results.items():
        reasons = result["reasons"]
        accepted_hands = set(result["accepted_hands"])
        distinct_contacts = int(result["distinct_accepted_contact_count"])
        if accepted_hands != set(REQUIRED_HANDS) or distinct_contacts < 2:
            continue
        eligible[recipe_id] = (
            int(result["accepted_cell_count"]),
            int(reasons["accepted"]),
            distinct_contacts,
            int(result["stage_counts"]["static_passed"]),
            int(result["stage_counts"]["collision_passed"]),
            int(result["stage_counts"]["ik_passed"]),
            int(result["stage_counts"]["proposal_passed"]),
        )
    if not eligible:
        return {
            "status": "inconclusive",
            "winner": None,
            "reason": "no_recipe_has_three_hand_dynamic_evidence",
            "criterion": criterion,
        }
    best_score = max(eligible.values())
    winners = sorted(name for name, score in eligible.items() if score == best_score)
    if len(winners) != 1:
        return {
            "status": "inconclusive",
            "winner": None,
            "reason": "selection_score_tie",
            "criterion": criterion,
            "scores": eligible,
        }
    return {
        "status": "selected",
        "winner": winners[0],
        "reason": "unique_pre_registered_score",
        "criterion": criterion,
        "scores": eligible,
    }


def _stage_counts(total: int, reasons: dict[str, int]) -> dict[str, int]:
    proposal_passed = total - reasons["proposal_rejected"]
    ik_passed = proposal_passed - reasons["ik_rejected"]
    collision_passed = ik_passed - reasons["collision_rejected"]
    static_passed = collision_passed - reasons["static_force_rejected"]
    return {
        "proposal_passed": proposal_passed,
        "ik_passed": ik_passed,
        "collision_passed": collision_passed,
        "static_passed": static_passed,
        "dynamic_attempted": reasons["dynamic_rejected"] + reasons["accepted"],
    }


def run_ablation(
    recipes: list[str] | None = None,
    num_candidates_per_obj: int = 4,
    seed: int = 42,
    run_dynamic: bool = True,
    matrix: str = "canonical",
) -> dict[str, Any]:
    if recipes is None:
        recipes = list(ALLOWED_RECIPES.keys())
    unknown = sorted(set(recipes) - set(ALLOWED_RECIPES))
    if unknown:
        raise ValueError(f"unknown recipes: {unknown}")
    if matrix not in ABLATION_MATRICES:
        raise ValueError(f"matrix must be one of {ABLATION_MATRICES}")
    if not 1 <= num_candidates_per_obj <= 4:
        raise ValueError("num_candidates_per_obj must be in [1, 4]")

    if len(recipes) != len(set(recipes)):
        raise ValueError("recipes must not contain duplicates")

    robot_configs = [
        ("leap_hand", "leap_hand.yaml"),
        ("wonik_allegro", "wonik_allegro.yaml"),
        ("shadow_hand", "shadow_hand.yaml"),
    ]

    cells_by_hand: dict[str, list[dict[str, Any]]] = {}
    if matrix == "canonical":
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
            test_objects.append(
                {
                    "name": obj_name,
                    "mesh": mesh,
                    "geoms": geoms,
                    "mass": mass,
                    "object_pos": None,
                    "budget": num_candidates_per_obj,
                }
            )
        cells_by_hand = {hand: test_objects for hand, _ in robot_configs}
    else:
        for hand, _ in robot_configs:
            fixture = build_generated_reachable_object(hand)
            budget = POSITIVE_CONTROL_BUDGETS[hand]
            if budget > fixture.candidate_budget:
                raise RuntimeError(f"positive-control budget for {hand} exceeds its validated fixture budget")
            cells_by_hand[hand] = [
                {
                    "name": "generated_reachable",
                    "mesh": fixture.mesh,
                    "geoms": fixture.collision_geoms,
                    "mass": fixture.mass,
                    "object_pos": fixture.object_pos,
                    "budget": budget,
                }
            ]

    planned_candidates = len(recipes) * sum(
        int(cell["budget"]) for hand_cells in cells_by_hand.values() for cell in hand_cells
    )
    if planned_candidates > MAX_TOTAL_CANDIDATES:
        raise ValueError(f"planned ablation has {planned_candidates} candidates; hard limit is {MAX_TOTAL_CANDIDATES}")

    object_hashes = {
        f"{hand}/{cell['name']}": _mesh_sha256(cell["mesh"])
        for hand, hand_cells in cells_by_hand.items()
        for cell in hand_cells
    }
    recipe_results: dict[str, Any] = {}
    cells: list[dict[str, Any]] = []

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
        recipe_started = time.perf_counter()
        accepted_hands: set[str] = set()
        accepted_cells = 0
        all_ik_signatures: set[str] = set()
        all_accepted_signatures: set[str] = set()
        all_wrench_signatures: set[str] = set()

        for r_name, r_cfg in robot_configs:
            spec = RobotSpec.from_config(r_cfg, sample_anchors=False)
            xml_path = resolve_robot_asset(spec.config.source_asset)

            for cell in cells_by_hand[r_name]:
                obj_name = str(cell["name"])
                mesh = cell["mesh"]
                geoms = cell["geoms"]
                mass = float(cell["mass"])
                object_pos = cell["object_pos"]
                candidate_budget = int(cell["budget"])
                cell_started = time.perf_counter()
                rng = (
                    generated_reachable_rng(r_name, seed)
                    if matrix == "positive-control"
                    else get_generator(seed, recipe_id, r_name, obj_name)
                )
                outcomes, reasons = run_pipeline_chunk(
                    recipe_id=recipe_id,
                    spec=spec,
                    mesh=mesh,
                    collision_geoms=geoms,
                    hand_xml_path=xml_path,
                    rng=rng,
                    num_candidates=candidate_budget,
                    object_mass=mass,
                    object_pos=object_pos,
                    run_dynamic=run_dynamic,
                )
                total_candidates += len(outcomes)
                for k, v in reasons.items():
                    recipe_reasons[k] += v
                summary = _summarize_outcomes(outcomes)
                all_ik_signatures.update(summary["ik_contact_signatures"])
                all_accepted_signatures.update(summary["accepted_contact_signatures"])
                all_wrench_signatures.update(summary["wrench_signatures"])
                if reasons["accepted"]:
                    accepted_hands.add(r_name)
                    accepted_cells += 1
                cells.append(
                    {
                        "recipe": recipe_id,
                        "hand": r_name,
                        "object": obj_name,
                        "candidate_budget": candidate_budget,
                        "observed_candidates": len(outcomes),
                        "reasons": reasons,
                        "stage_counts": _stage_counts(len(outcomes), reasons),
                        "stage_rates": _stage_rates(len(outcomes), reasons),
                        "metrics": summary,
                        "runtime_seconds": round(time.perf_counter() - cell_started, 6),
                    }
                )
                logger.info(
                    "completed recipe=%s robot=%s object=%s candidates=%d",
                    recipe_id,
                    r_name,
                    obj_name,
                    len(outcomes),
                )

        accounted = sum(recipe_reasons.values())
        if accounted != total_candidates:
            raise RuntimeError(f"reason accounting mismatch for {recipe_id}: {accounted} != {total_candidates}")

        recipe_results[recipe_id] = {
            "total_candidates": total_candidates,
            "reasons": recipe_reasons,
            "stage_counts": _stage_counts(total_candidates, recipe_reasons),
            **_stage_rates(total_candidates, recipe_reasons),
            "accepted_hands": sorted(accepted_hands),
            "accepted_cell_count": accepted_cells,
            "distinct_ik_contact_count": len(all_ik_signatures),
            "distinct_accepted_contact_count": len(all_accepted_signatures),
            "distinct_wrench_count": len(all_wrench_signatures),
            "runtime_seconds": round(time.perf_counter() - recipe_started, 6),
        }

    protocol = dataclasses.asdict(RolloutProtocol())
    protocol_encoded = json.dumps(protocol, sort_keys=True, separators=(",", ":")).encode()
    provenance = {
        "git_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
        "source_hashes": {name: _sha256_file(REPO_ROOT / name) for name in PROVENANCE_SOURCES},
        "robot_profile_hashes": {
            hand: _sha256_file(REPO_ROOT / "qdgrasp" / "presets" / "robots" / config) for hand, config in robot_configs
        },
        "object_mesh_hashes": object_hashes,
        "rollout_protocol_sha256": hashlib.sha256(protocol_encoded).hexdigest(),
    }
    return {
        "schema": "qdgrasp/controlled-ablation/v2",
        "status": "COMPLETE",
        "protocol": {
            "seed": seed,
            "matrix": matrix,
            "scope": (
                "morphology-specific pipeline positive controls"
                if matrix == "positive-control"
                else "procedural cross-object generalization"
            ),
            "recipes": recipes,
            "hands": [name for name, _ in robot_configs],
            "objects": sorted({str(cell["name"]) for cells in cells_by_hand.values() for cell in cells}),
            "candidates_per_cell": num_candidates_per_obj if matrix == "canonical" else None,
            "candidate_budgets_by_hand": POSITIVE_CONTROL_BUDGETS if matrix == "positive-control" else None,
            "planned_candidates": planned_candidates,
            "run_dynamic": run_dynamic,
            "interpretation_limit": (
                "May select a recipe for the validated positive-control envelope; does not establish procedural-object generalization."
                if matrix == "positive-control"
                else None
            ),
        },
        "provenance": provenance,
        "cells": cells,
        "recipes": recipe_results,
        "decision": _selection_decision(recipe_results, run_dynamic=run_dynamic),
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--num-candidates-per-object", type=int, default=PINNED_CANDIDATES_PER_CELL)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--matrix", choices=ABLATION_MATRICES, default="canonical")
    parser.add_argument(
        "--recipe",
        action="append",
        choices=sorted(ALLOWED_RECIPES),
        dest="recipes",
    )
    parser.add_argument("--skip-dynamic", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    recipes = args.recipes or list(ALLOWED_RECIPES)
    planned = (
        len(recipes) * sum(POSITIVE_CONTROL_BUDGETS.values())
        if args.matrix == "positive-control"
        else len(recipes) * 3 * 4 * args.num_candidates_per_object
    )
    if not args.execute:
        print(
            json.dumps(
                {
                    "status": "DRY_RUN",
                    "planned_candidates": planned,
                    "matrix": args.matrix,
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
        matrix=args.matrix,
    )
    encoded = json.dumps(results, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")


if __name__ == "__main__":
    main()
