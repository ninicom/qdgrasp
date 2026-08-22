"""Phase 3 verification gate: data layer, procedural objects, and dataset loader."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import trimesh

from qdgrasp.api import QDGrasp
from qdgrasp.dataset.manifest import load_dataset_manifest
from qdgrasp.dataset.pipeline.filter import filter_grasp_candidate
from qdgrasp.dataset.pipeline.ik import solve_dls_ik
from qdgrasp.dataset.pipeline.sample import sample_grasp_candidates
from qdgrasp.dataset.render import sample_analytic_point_cloud
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
        ik_res = solve_dls_ik(spec, cand.palm_pos, cand.palm_rot, cand.target_contacts, max_iter=20)
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


def main() -> None:
    print("Running Phase 3 verification suite...")
    verify_procedural_generators()
    print("  1. Procedural generators & collision guards: PASS")

    verify_pipeline_and_ik()
    print("  2. Candidate sampling & DLS-IK across 3 hands: PASS")

    manifest_summary = audit_dataset_manifest("datasets/dgn-open-tiny")
    print(f"  3. Dataset manifest audit: PASS ({manifest_summary['total_samples']} samples)")

    train_metrics = verify_train_step_with_dgn_tiny()
    print(f"  4. CPU Train step with DGN-Open-Tiny: PASS (loss={train_metrics.get('loss', 0.0):.4f})")

    summary = {
        "status": "PASS",
        "dataset_id": manifest_summary["dataset_id"],
        "objects": manifest_summary["total_objects"],
        "shards": manifest_summary["shards"],
        "samples": manifest_summary["total_samples"],
        "train_metrics": train_metrics,
    }
    print("Phase 3 Data Layer: PASS")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
