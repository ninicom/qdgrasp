"""Phase 3 verification gate: data layer, procedural objects, and dataset loader."""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import trimesh

from qdgrasp.api import QDGrasp
from qdgrasp.dataset.pipeline.filter import filter_grasp_candidate
from qdgrasp.dataset.pipeline.ik import solve_dls_ik
from qdgrasp.dataset.pipeline.sample import sample_grasp_candidates
from qdgrasp.dataset.rng import get_generator
from qdgrasp.objects.collision import validate_collision_representation
from qdgrasp.objects.generate import (
    generate_box,
    generate_capsule,
    generate_compound_convex,
    generate_cylinder,
    generate_sphere,
    generate_superquadric,
)
from qdgrasp.objects.schema import SubGeomSpec
from qdgrasp.robot.spec import RobotSpec
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.check_dataset_manifest import audit_dataset_manifest

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


def verify_procedural_generators() -> None:
    for gen in (generate_box, generate_sphere, generate_cylinder, generate_capsule, generate_superquadric):
        rng1 = get_generator(42, "gate_test")
        m1, g1, _, mass1, in1 = gen(rng1)
        rng2 = get_generator(42, "gate_test")
        m2, g2, _, mass2, in2 = gen(rng2)

        if mass1 <= 0 or not all(i > 0 for i in in1):
            raise AssertionError(f"invalid mass/inertia from {gen.__name__}")
        if not np.allclose(m1.vertices, m2.vertices):
            raise AssertionError(f"non-deterministic vertices from {gen.__name__}")
        validate_collision_representation(m1, g1)

    # Compound
    for f in ("t_shape", "l_shape", "dumbbell"):
        rng = get_generator(99, f)
        m, g, _, mass, in_t = generate_compound_convex(rng, shape_family=f)
        if len(g) < 2:
            raise AssertionError(f"compound shape {f} has fewer than 2 collision geoms")
        validate_collision_representation(m, g)


def verify_pipeline_and_ik() -> None:
    test_mesh = trimesh.creation.box(extents=(0.04, 0.04, 0.04))
    for preset in ("leap_hand.yaml", "wonik_allegro.yaml", "shadow_hand.yaml"):
        spec = RobotSpec.from_config(preset, sample_anchors=False)
        rng = get_generator(555, preset)
        candidates = sample_grasp_candidates(spec, test_mesh, rng, num_candidates=2)
        if len(candidates) != 2:
            raise AssertionError(f"failed to sample candidates for {preset}")

        cand = candidates[0]
        ik_res = solve_dls_ik(
            spec,
            cand.palm_pos,
            cand.palm_rot,
            cand.target_contacts,
            target_normals=cand.target_normals,
            max_iter=20,
        )
        # Check joint limits
        for j_idx, j_name in enumerate(spec.actuated_joint_names):
            lo, hi = spec.joint_limits[j_name]
            if ik_res.q[j_idx] < lo - 1e-4 or ik_res.q[j_idx] > hi + 1e-4:
                raise AssertionError(f"IK joint limit violation on {j_name}")

        filter_res = filter_grasp_candidate(spec, cand.palm_pos, cand.palm_rot, ik_res.q, test_mesh)
        if not filter_res.valid and filter_res.reason.startswith("joint_limit"):
            raise AssertionError("unexpected joint limit violation in filter")


def verify_train_step_with_dgn_tiny() -> dict[str, float]:
    grasper = QDGrasp("qdgrasp-dummy-n.yaml", robot="leap_hand.yaml", seed=42)
    result = grasper.train(
        "configs/data/dgn_open_tiny.yaml",
        device="cpu",
        max_steps=2,
        batch_size=2,
        learning_rate=1e-3,
        run_name="check_phase3_train",
        project_dir="runs/check_phase3",
    )
    return result.metrics


def verify_recipes_and_contracts() -> None:
    test_mesh = trimesh.creation.box(extents=(0.04, 0.04, 0.04))
    geoms = [
        SubGeomSpec(
            type="box",
            size=(0.02, 0.02, 0.02),
            pos=(0.0, 0.0, 0.0),
            quat=(1.0, 0.0, 0.0, 0.0)
        )
    ]
    from qdgrasp.dataset.pipeline.contracts import ALLOWED_RECIPES
    from qdgrasp.dataset.pipeline.orchestrator import run_pipeline_chunk
    from qdgrasp.robot.spec import resolve_robot_asset

    for recipe_id in ALLOWED_RECIPES.keys():
        for preset in ("leap_hand.yaml", "wonik_allegro.yaml", "shadow_hand.yaml"):
            spec = RobotSpec.from_config(preset, sample_anchors=False)
            xml_path = resolve_robot_asset(spec.config.source_asset)
            rng = get_generator(42, f"{recipe_id}_{preset}")
            outcomes, reasons = run_pipeline_chunk(
                recipe_id=recipe_id,
                spec=spec,
                mesh=test_mesh,
                collision_geoms=geoms,
                hand_xml_path=xml_path,
                rng=rng,
                num_candidates=2,
                run_dynamic=False,
            )
            if len(outcomes) != 2:
                raise AssertionError(f"Failed pipeline run for recipe {recipe_id} with robot {preset}")
            if sum(reasons.values()) != len(outcomes):
                raise AssertionError(
                    f"reason accounting drift for {recipe_id}/{preset}: {reasons}"
                )
            for outcome in outcomes:
                flags = (
                    outcome.proposal_valid,
                    outcome.ik_valid,
                    outcome.collision_valid,
                    outcome.static_force_valid,
                    outcome.dynamic_valid,
                )
                if any(flags[index] and not flags[index - 1] for index in range(1, 5)):
                    raise AssertionError(
                        f"non-monotonic outcome for {recipe_id}/{preset}: {flags}"
                    )
                if outcome.dynamic_valid or outcome.failure_reason != "dynamic_skipped" and outcome.static_force_valid:
                    raise AssertionError(
                        "run_dynamic=False must never create a positive sample"
                    )


def verify_p31_regression_bundle() -> None:
    tests = [
        "tests/test_fixed_contact_dls.py",
        "tests/test_region_dls.py",
        "tests/test_contact_force.py",
        "tests/test_contact_load.py",
        "tests/test_physics_rollout.py",
        "tests/test_dataset_generator_logic.py",
        "tests/test_dataset_manifest_audit.py",
        "tests/test_ablation_accounting.py",
    ]
    environment = os.environ.copy()
    environment.update(
        {
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "PYTHONHASHSEED": "0",
        }
    )
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", *tests],
        cwd=REPO_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    if completed.returncode != 0:
        raise AssertionError(
            "P3.1 correctness regression failed:\n"
            f"{completed.stdout}\n{completed.stderr}"
        )


def main() -> None:
    print("Running Phase 3 / 3.1 verification suite...")
    verify_procedural_generators()
    print("  1. Procedural generators & collision guards: PASS")

    verify_pipeline_and_ik()
    print("  2. Candidate sampling & DLS-IK across 3 hands: PASS")

    verify_recipes_and_contracts()
    print("  3. 3-Recipe and Staged Pipeline Verification: PASS")

    verify_p31_regression_bundle()
    print("  4. P3.1 IK/static/dynamic correctness regressions: PASS")

    manifest_summary = audit_dataset_manifest("datasets/dgn-open-tiny")
    print(f"  5. Dataset manifest audit: PASS ({manifest_summary['total_samples']} samples)")

    train_metrics = verify_train_step_with_dgn_tiny()
    print(f"  6. CPU Train step with DGN-Open-Tiny: PASS (loss={train_metrics.get('loss', 0.0):.4f})")

    summary = {
        "status": "PASS",
        "dataset_id": manifest_summary["dataset_id"],
        "objects": manifest_summary["total_objects"],
        "shards": manifest_summary["shards"],
        "samples": manifest_summary["total_samples"],
        "train_metrics": train_metrics,
    }
    print("Phase 3 / 3.1 Data Layer: PASS")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
