"""Generate the standard DGN-Open-Tiny cross-embodiment dataset release."""

from __future__ import annotations

import argparse
import hashlib
import logging
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import torch

from qdgrasp.config import (
    DEFAULT_ROBOT_PROFILES,
    historical_reproduction_scope,
    require_release_scope,
)
from qdgrasp.dataset.manifest import DatasetManifestSpec, ShardMetadata, save_dataset_manifest
from qdgrasp.dataset.pipeline.contracts import ALLOWED_RECIPES, PipelineOutcome, get_recipe
from qdgrasp.dataset.pipeline.generated_reachable import build_grasp_bar, generated_reachable_rng
from qdgrasp.dataset.pipeline.orchestrator import run_pipeline_chunk
from qdgrasp.dataset.render import sample_analytic_point_cloud
from qdgrasp.dataset.rng import get_generator
from qdgrasp.dataset.shards import write_shard_file
from qdgrasp.dataset.split import create_object_family_splits
from qdgrasp.objects.generate import (
    generate_box,
    generate_capsule,
    generate_compound_convex,
    generate_cylinder,
    generate_sphere,
    generate_superquadric,
)
from qdgrasp.objects.manifest import create_object_asset, save_object_asset
from qdgrasp.objects.schema import ObjectManifestSpec
from qdgrasp.robot.provenance import validate_profile_for_release
from qdgrasp.robot.spec import RobotSpec, resolve_robot_asset
from qdgrasp.runtime import environment_info

logger = logging.getLogger("generate_dgn_open_tiny")

# Candidate budgets of the validated positive-control envelope (P3.1-13).  They
# are the budgets the recipe selection was measured at and must not be raised
# above ``GeneratedReachableObject.candidate_budget``.
POSITIVE_CONTROL_BUDGETS = {"leap_hand": 4, "wonik_allegro": 14, "shadow_hand": 10}

# Release variants of the positive-control grasp bar.  Each entry was measured
# end-to-end by this pipeline before being admitted (P3.1-14); none of them
# carries a stored grasp.  Two variants per hand let both splits hold a
# positive.
#
# `pc_allegro_02` moves `width`, not the block height: a kinematics grid over
# the envelope showed Allegro dies at IK, not at palm floor clearance, and that
# the calibrated 40 mm opposition is a poor operating point.  A 45 mm task
# measures two dynamic positives where four fixed-width variants measured none.
POSITIVE_CONTROL_VARIANTS: dict[str, tuple[tuple[str, dict[str, float]], ...]] = {
    "leap_hand": (
        ("pc_leap_01", {}),
        ("pc_leap_02", {"upper_height": 0.055}),
    ),
    "wonik_allegro": (
        ("pc_allegro_01", {}),
        ("pc_allegro_02", {"width": 0.045, "upper_center_z": 0.130}),
    ),
    "shadow_hand": (
        ("pc_shadow_01", {}),
        ("pc_shadow_02", {"upper_center_z": 0.135}),
    ),
}


def outcome_to_sample(
    outcome: PipelineOutcome,
    *,
    spec: RobotSpec,
    mesh: Any,
    rng: np.random.Generator,
    object_id: str,
    robot_name: str,
    recipe_id: str,
) -> dict[str, Any]:
    """Serialize one staged outcome without manufacturing missing evidence."""
    recipe = get_recipe(recipe_id)
    stage_flags = (
        outcome.proposal_valid,
        outcome.ik_valid,
        outcome.collision_valid,
        outcome.static_force_valid,
        outcome.dynamic_valid,
    )
    if any(stage_flags[index] and not stage_flags[index - 1] for index in range(1, 5)):
        raise RuntimeError("pipeline outcome has non-monotonic stage flags")
    if outcome.dynamic_valid and (outcome.dynamic_validation is None or not outcome.dynamic_validation.passed):
        raise RuntimeError("dynamic-valid outcome lacks passing rollout evidence")

    is_success = bool(outcome.dynamic_valid)
    kinematics_valid = outcome.kinematics is not None
    quality = (
        float(outcome.dynamic_validation.trajectory_metrics.get("lift_achieved", 0.0))
        if is_success and outcome.dynamic_validation is not None
        else 0.0
    )
    q = (
        np.asarray(outcome.kinematics.q, dtype=np.float64)
        if outcome.kinematics is not None
        else np.zeros(len(spec.actuated_joint_names), dtype=np.float64)
    )
    achieved_contacts = (
        np.asarray(outcome.kinematics.achieved_contacts, dtype=np.float64)
        if outcome.kinematics is not None
        else np.zeros((len(spec.fingertip_links), 3), dtype=np.float64)
    )
    palm_pos = (
        np.asarray(outcome.kinematics.palm_pos, dtype=np.float64)
        if outcome.kinematics is not None
        else np.zeros(3, dtype=np.float64)
    )
    palm_rot = (
        np.asarray(outcome.kinematics.palm_rot, dtype=np.float64)
        if outcome.kinematics is not None
        else np.eye(3, dtype=np.float64)
    )

    cam_pos = palm_pos + np.array([0.0, 0.0, 0.15])
    pcd_cam, camera_meta = sample_analytic_point_cloud(
        mesh,
        camera_pos=cam_pos,
        camera_rot=np.eye(3),
        num_points=1024,
        rng=rng,
    )
    camera_rot = np.asarray(camera_meta["camera_rot"], dtype=np.float64)
    camera_pos = np.asarray(camera_meta["camera_pos"], dtype=np.float64)
    pcd_object = (camera_rot @ pcd_cam.astype(np.float64).T).T + camera_pos

    return {
        "points": torch.from_numpy(pcd_object).float(),
        "palm_pos": torch.from_numpy(palm_pos).float(),
        "palm_rot": torch.from_numpy(palm_rot).float(),
        "joint_angles": torch.from_numpy(q).float(),
        "fingertip_positions": torch.from_numpy(achieved_contacts).float(),
        "success": torch.tensor(float(is_success), dtype=torch.float32),
        "quality": torch.tensor(quality, dtype=torch.float32),
        "object_id": object_id,
        "robot_name": robot_name,
        "frame": "object",
        "recipe_id": recipe_id,
        "proposal_module": recipe["proposal"],
        "solver_module": recipe["solver"],
        "certifier_version": "gws-gravity-v1",
        "dynamic_protocol_version": "mocap-weld-v3",
        "success_schema_version": "dynamic-only-v1",
        "failure_stage": outcome.failure_stage,
        "failure_reason": outcome.failure_reason,
        "proposal_valid": outcome.proposal_valid,
        "ik_valid": outcome.ik_valid,
        "collision_valid": outcome.collision_valid,
        "static_force_valid": outcome.static_force_valid,
        "dynamic_valid": outcome.dynamic_valid,
        # Target validity is explicit.  Zero joint angles and an identity palm
        # are legitimate measurements, so consumers must never infer whether a
        # target exists from its numeric value.
        "kinematics_valid": kinematics_valid,
        "pose_target_valid": kinematics_valid,
        "joint_target_valid": kinematics_valid,
        "fk_target_valid": kinematics_valid,
    }


def generate_tiny_dataset(
    output_dir: str | Path = "datasets/dgn-open-tiny",
    base_seed: int = 42,
    samples_per_pair: int = 4,
    recipe_id: str = "wrench_guided_v1",
    historical_reproduction: str | None = None,
) -> Path:
    """Generate all objects, grasp samples, and manifest for DGN-Open-Tiny.

    The default corpus is the active one. Reproducing a pre-ADR-0008 three-hand
    artifact needs its declared id passed explicitly, and everything that run
    produces is ``non_release``: reproducing history does not give the pause an
    exception, and does not create new three-hand coverage.
    """
    repo_root = Path(__file__).resolve().parent.parent
    generator_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty_output = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if dirty_output:
        changed = ", ".join(line[3:] for line in dirty_output.splitlines()[:8])
        raise RuntimeError(
            "refusing to generate a release corpus from a dirty worktree; "
            f"commit or remove the pending paths first ({changed})"
        )

    recipe = get_recipe(recipe_id)
    if historical_reproduction is None:
        profiles = DEFAULT_ROBOT_PROFILES
        scope = None
        require_release_scope(profiles)
    else:
        profiles = ("leap_hand.yaml", "wonik_allegro.yaml", "shadow_hand.yaml")
        scope = historical_reproduction_scope(historical_reproduction, profiles)
    robot_configs = [(name.removesuffix(".yaml"), name) for name in profiles]
    robot_specs = {name: RobotSpec.from_config(cfg_name, sample_anchors=False) for name, cfg_name in robot_configs}
    for spec in robot_specs.values():
        validate_profile_for_release(spec.config)

    out_p = Path(output_dir).resolve()
    obj_dir = out_p / "objects"
    shards_dir = out_p / "shards"
    obj_dir.mkdir(parents=True, exist_ok=True)
    shards_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Generating 12 procedural objects...")
    objects: list[ObjectManifestSpec] = []
    meshes: dict[str, Any] = {}

    # 1. Generate 4 primitives
    prim_defs = [
        ("prim_box_01", "primitive", "box", generate_box, {}),
        ("prim_sphere_01", "primitive", "sphere", generate_sphere, {}),
        ("prim_cylinder_01", "primitive", "cylinder", generate_cylinder, {}),
        ("prim_capsule_01", "primitive", "capsule", generate_capsule, {}),
    ]
    # 2. Generate 4 superquadrics
    sq_defs = [(f"sq_{i:02d}", "superquadric", "superquadric", generate_superquadric, {}) for i in range(1, 5)]
    # 3. Generate 4 compound convex shapes
    comp_defs = [
        ("comp_t_shape_01", "compound", "t_shape", generate_compound_convex, {"shape_family": "t_shape"}),
        ("comp_t_shape_02", "compound", "t_shape", generate_compound_convex, {"shape_family": "t_shape"}),
        ("comp_l_shape_01", "compound", "l_shape", generate_compound_convex, {"shape_family": "l_shape"}),
        ("comp_dumbbell_01", "compound", "dumbbell", generate_compound_convex, {"shape_family": "dumbbell"}),
    ]

    all_specs = prim_defs + sq_defs + comp_defs

    for obj_id, family, shape_type, gen_fn, kwargs in all_specs:
        rng = get_generator(base_seed, "object", obj_id)
        mesh, geoms, params, mass, inertia = gen_fn(rng, **kwargs)
        mesh_bytes, manifest = create_object_asset(
            object_id=obj_id,
            family=family,
            shape_type=shape_type,
            mesh=mesh,
            collision_geoms=geoms,
            params=params,
            mass=mass,
            inertia=inertia,
        )
        save_object_asset(mesh_bytes, manifest, obj_dir)
        objects.append(manifest)
        meshes[obj_id] = mesh

    benchmark_objects = tuple(objects)

    # Positive-control objects: same generator contract as every other object,
    # but each one is only ever paired with the hand it was calibrated for.
    positive_control: dict[str, dict[str, Any]] = {}
    for robot_name in robot_specs:
        variants = POSITIVE_CONTROL_VARIANTS[robot_name]
        for obj_id, params in variants:
            bar = build_grasp_bar(robot_name, **params)
            inertia_scale = bar.mass / float(bar.mesh.volume)
            unit_inertia = bar.mesh.moment_inertia
            inertia = tuple(float(unit_inertia[i][i] * inertia_scale) for i in range(3))
            mesh_bytes, manifest = create_object_asset(
                object_id=obj_id,
                family="positive_control",
                shape_type=f"grasp_bar_{robot_name}",
                mesh=bar.mesh,
                collision_geoms=bar.collision_geoms,
                params={"profile": robot_name, **params},
                mass=bar.mass,
                inertia=inertia,
            )
            save_object_asset(mesh_bytes, manifest, obj_dir)
            objects.append(manifest)
            meshes[obj_id] = bar.mesh
            positive_control[obj_id] = {
                "robot_name": robot_name,
                "object_pos": bar.object_pos,
                "budget": POSITIVE_CONTROL_BUDGETS[robot_name],
            }
            if positive_control[obj_id]["budget"] > bar.candidate_budget:
                raise RuntimeError(f"positive-control budget for {robot_name} exceeds its validated ceiling")

    # A normal release contains only the active hands.  Remove exact files left
    # by an earlier declared three-hand reproduction so the release-file audit
    # cannot mistake historical Shadow artifacts for active scope.
    for robot_name in sorted(set(POSITIVE_CONTROL_VARIANTS) - set(robot_specs)):
        for split_name in ("train", "val"):
            (shards_dir / f"{split_name}_{robot_name}.pt").unlink(missing_ok=True)
        for object_id, _params in POSITIVE_CONTROL_VARIANTS[robot_name]:
            (obj_dir / f"{object_id}.obj").unlink(missing_ok=True)
            (obj_dir / f"{object_id}.manifest.json").unlink(missing_ok=True)

    # The locked Phase-5 generalisation claim holds the complete compound
    # family out.  The splitter assigns whole families, never members within a
    # shape, so this physical split now agrees with the claim it carries.
    splits = create_object_family_splits(
        benchmark_objects,
        val_fraction=0.25,
        seed=base_seed,
        val_families=("compound",),
    )
    # Calibration controls are outside the protocol's object matrix.  Each hand
    # has two independently generated controls so both physical shards retain a
    # measured positive/negative health check without leaking a benchmark
    # family across the train/validation boundary.
    for robot_name in robot_specs:
        variants = POSITIVE_CONTROL_VARIANTS[robot_name]
        if len(variants) != 2:
            raise RuntimeError(f"{robot_name} must declare exactly two positive-control variants")
        splits["train"].append(variants[0][0])
        splits["val"].append(variants[1][0])
    splits = {name: sorted(object_ids) for name, object_ids in splits.items()}
    logger.info(f"Split objects: train={splits['train']}, val={splits['val']}")

    robot_hashes = {name: spec.config.content_hash() for name, spec in robot_specs.items()}

    shard_metas: list[ShardMetadata] = []

    # Generate samples per (split, robot)
    for split_name, obj_ids in splits.items():
        for r_name, r_cfg in robot_configs:
            spec = robot_specs[r_name]
            xml_path = resolve_robot_asset(spec.config.source_asset)
            if not xml_path.is_file():
                raise RuntimeError(f"dynamic robot asset unavailable: {xml_path}")

            samples: list[dict[str, Any]] = []
            positives = 0

            for obj_id in obj_ids:
                control = positive_control.get(obj_id)
                if control is not None and control["robot_name"] != r_name:
                    continue

                obj_manifest = next(o for o in objects if o.object_id == obj_id)
                mesh = meshes[obj_id]
                if control is None:
                    rng = get_generator(base_seed, split_name, r_name, obj_id)
                    num_candidates = samples_per_pair
                    object_pos = None
                else:
                    # Frozen proposal stream and budget of the validated
                    # positive-control envelope; a different stream would be an
                    # unmeasured experiment, not this release.
                    rng = generated_reachable_rng(r_name, base_seed)
                    num_candidates = control["budget"]
                    object_pos = control["object_pos"]

                outcomes, _reasons = run_pipeline_chunk(
                    recipe_id=recipe_id,
                    spec=spec,
                    mesh=mesh,
                    collision_geoms=obj_manifest.collision_geoms,
                    hand_xml_path=xml_path,
                    rng=rng,
                    num_candidates=num_candidates,
                    object_mass=obj_manifest.mass,
                    object_pos=object_pos,
                    run_dynamic=True,
                )

                for outcome in outcomes:
                    sample = outcome_to_sample(
                        outcome,
                        spec=spec,
                        mesh=mesh,
                        rng=rng,
                        object_id=obj_id,
                        robot_name=r_name,
                        recipe_id=recipe_id,
                    )
                    positives += int(bool(sample["dynamic_valid"]))
                    samples.append(sample)

            shard_filename = f"shards/{split_name}_{r_name}.pt"
            shard_path = out_p / shard_filename
            sha = write_shard_file(samples, shard_path)

            shard_meta = ShardMetadata(
                filename=shard_filename,
                sha256=sha,
                num_samples=len(samples),
                positive_samples=positives,
                robot_name=r_name,
                split=split_name,
                recipe_id=recipe_id,
            )
            shard_metas.append(shard_meta)
            logger.info(f"Generated {shard_filename}: {len(samples)} samples ({positives} pos), sha256={sha[:12]}...")

    # Top-level dataset manifest
    env_info = environment_info().to_dict()
    object_hashes = {
        obj.object_id: hashlib.sha256((obj_dir / f"{obj.object_id}.manifest.json").read_bytes()).hexdigest()
        for obj in objects
    }
    source_names = [
        "scripts/generate_dgn_open_tiny.py",
        "qdgrasp/dataset/manifest.py",
        "qdgrasp/dataset/pipeline/contracts.py",
        "qdgrasp/dataset/pipeline/generated_reachable.py",
        "qdgrasp/dataset/pipeline/filter.py",
        "qdgrasp/dataset/pipeline/orchestrator.py",
        "qdgrasp/dataset/pipeline/proposals/surface_fixed.py",
        "qdgrasp/dataset/pipeline/proposals/region_opposition.py",
        "qdgrasp/dataset/pipeline/proposals/wrench_guided.py",
        "qdgrasp/dataset/pipeline/solvers/fixed_contact_dls.py",
        "qdgrasp/dataset/pipeline/solvers/region_dls.py",
        "qdgrasp/dataset/pipeline/certifiers/contact_force.py",
        "qdgrasp/dataset/pipeline/certifiers/grasp_wrench.py",
        "qdgrasp/dataset/pipeline/observers/contact_load.py",
        "qdgrasp/dataset/pipeline/validators/mujoco_rollout.py",
    ]
    source_hashes = {name: hashlib.sha256((repo_root / name).read_bytes()).hexdigest() for name in source_names}
    release_blocked = any(
        shard.positive_samples == 0 or shard.positive_samples == shard.num_samples for shard in shard_metas
    )
    generator_worktree_dirty = False
    # A historical reproduction can never be release evidence, whatever else
    # the shard statistics say.
    if scope is not None and scope.non_release:
        release_blocked = True
    dataset_manifest = DatasetManifestSpec(
        dataset_id="dgn-open-tiny-v1",
        generator_version="0.1.0a1",
        generator_commit=generator_commit,
        generator_worktree_dirty=generator_worktree_dirty,
        seed=base_seed,
        environment_fingerprint=env_info,
        robot_profile_hashes=robot_hashes,
        object_manifest_hashes=object_hashes,
        generator_source_hashes=source_hashes,
        recipe_id=recipe_id,
        proposal_module=recipe["proposal"],
        solver_module=recipe["solver"],
        certifier_version="gws-gravity-v1",
        dynamic_protocol_version="mocap-weld-v3",
        splits=splits,
        shards=shard_metas,
        success_criteria={
            "min_contacts": 2.0,
            "max_penetration": 0.002,
            "min_lift_ratio": 0.5,
        },
        license="CC0-1.0",
        release_blocked=release_blocked,
    )
    save_dataset_manifest(dataset_manifest, out_p / "dataset_manifest.json")
    logger.info(f"Saved dataset manifest at {out_p / 'dataset_manifest.json'}")
    return out_p


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Generate DGN-Open-Tiny dataset release.")
    parser.add_argument("--output-dir", default="datasets/dgn-open-tiny", help="Target output directory.")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed.")
    parser.add_argument("--samples-per-pair", type=int, default=4, help="Samples per object-robot pair.")
    parser.add_argument(
        "--recipe",
        default="wrench_guided_v1",
        choices=tuple(sorted(ALLOWED_RECIPES)),
        help="Allowlisted proposal/solver recipe.",
    )
    parser.add_argument(
        "--historical-reproduction",
        default=None,
        metavar="ARTIFACT_ID",
        help=(
            "Reproduce a declared pre-ADR-0008 three-hand artifact by id. Without "
            "this the run covers the active corpus only. Anything produced with "
            "it is non-release."
        ),
    )
    args = parser.parse_args()

    generate_tiny_dataset(
        output_dir=args.output_dir,
        base_seed=args.seed,
        samples_per_pair=args.samples_per_pair,
        recipe_id=args.recipe,
        historical_reproduction=args.historical_reproduction,
    )


if __name__ == "__main__":
    main()
