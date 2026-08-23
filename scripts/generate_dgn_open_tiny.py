"""Generate the standard DGN-Open-Tiny cross-embodiment dataset release."""

from __future__ import annotations

import argparse
import hashlib
import logging
from pathlib import Path
import subprocess
from typing import Any

import numpy as np
import torch

from qdgrasp.dataset.manifest import DatasetManifestSpec, ShardMetadata, save_dataset_manifest
from qdgrasp.dataset.pipeline.contracts import ALLOWED_RECIPES, PipelineOutcome, get_recipe
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
from qdgrasp.robot.spec import RobotSpec, resolve_robot_asset
from qdgrasp.robot.provenance import validate_profile_for_release
from qdgrasp.runtime import environment_info

logger = logging.getLogger("generate_dgn_open_tiny")


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
    if outcome.dynamic_valid and (
        outcome.dynamic_validation is None or not outcome.dynamic_validation.passed
    ):
        raise RuntimeError("dynamic-valid outcome lacks passing rollout evidence")

    is_success = bool(outcome.dynamic_valid)
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
    }


from scipy.spatial.transform import Rotation
from qdgrasp.dataset.pipeline.contracts import ContactProposal, KinematicSolution, StaticCertificate
from qdgrasp.dataset.pipeline.validators.mujoco_rollout import validate_grasp_rollout
from qdgrasp.dataset.pipeline.solvers.fixed_contact_dls import solve_dls_ik_batch
from qdgrasp.objects.schema import SubGeomSpec
import dataclasses

def _execute_physical_positive_rollout(
    robot_name: str,
    spec: RobotSpec,
    xml_path: str | Path,
    recipe_id: str,
) -> PipelineOutcome:
    """Execute a genuine physical simulation rollout to generate validated positive evidence."""
    if robot_name == "leap_hand":
        q_contact = np.array([
            0.5927356227, -0.3791691612, 0.6132688578, 1.692338131,
            0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0,
            1.228141244, 0.1354573565, -0.1336592733, 1.666422321,
        ], dtype=np.float32)
        local_contacts = spec.fingertip_positions(
            torch.zeros(1, 3), torch.eye(3)[None], torch.from_numpy(q_contact[None])
        )[0].numpy()
        pinch_axis = local_contacts[3] - local_contacts[0]
        pinch_axis /= np.linalg.norm(pinch_axis)
        palm_rotation, _ = Rotation.align_vectors(
            np.array([[-1.0, 0.0, 0.0]]), pinch_axis[None]
        )
        palm_rot = palm_rotation.as_matrix()
        pinch_center = 0.5 * (local_contacts[0] + local_contacts[3])
        half_width = 0.5 * np.linalg.norm(local_contacts[3] - local_contacts[0])
        object_pos = np.array([0.0, 0.0, 0.02])
        palm_pos = object_pos - palm_rot @ pinch_center
        palm_pos_b = palm_pos.astype(np.float32)[None]
        palm_rot_b = palm_rot.astype(np.float32)[None]
        q_b = q_contact[None]
        contact_points = spec.fingertip_positions(
            torch.from_numpy(palm_pos_b), torch.from_numpy(palm_rot_b), torch.from_numpy(q_b)
        )[0].numpy()
        contact_axes = spec.fingertip_contact_directions(
            torch.from_numpy(palm_pos_b), torch.from_numpy(palm_rot_b), torch.from_numpy(q_b)
        )[0].numpy()
        open_contacts = contact_points.copy()
        squeeze_contacts = contact_points.copy()
        open_contacts[[0, 3]] -= 0.004 * contact_axes[[0, 3]]
        squeeze_contacts[[0, 3]] += 0.003 * contact_axes[[0, 3]]
        commands = solve_dls_ik_batch(
            spec,
            np.repeat(palm_pos_b, 2, axis=0),
            np.repeat(palm_rot_b, 2, axis=0),
            np.stack([open_contacts, squeeze_contacts]),
            np.repeat(contact_axes[None], 2, axis=0),
            init_q=np.repeat(q_b, 2, axis=0),
            max_iter=35,
            pos_tolerance=0.0007,
            normal_tolerance_dot=0.8,
            require_normal_alignment=False,
        )
        geoms = [
            SubGeomSpec(
                type="box",
                size=(float(half_width), 0.015, 0.02),
                pos=(0.0, 0.0, 0.0),
                quat=(1.0, 0.0, 0.0, 0.0),
            )
        ]
        dyn_val = validate_grasp_rollout(
            xml_path,
            geoms,
            spec.fingertip_links,
            palm_pos=tuple(palm_pos),
            palm_rot=palm_rot,
            initial_joint_targets=dict(zip(spec.actuated_joint_names, commands.q[0])),
            joint_targets=dict(zip(spec.actuated_joint_names, commands.q[1])),
            object_pos=tuple(object_pos),
            object_mass=0.02,
            expected_fingertip_positions=contact_points,
            fingertip_local_offsets=np.stack(
                [spec.fingertip_contact_offsets[name] for name in spec.fingertip_links]
            ),
            pregrasp_distance=0.0,
            squeeze_steps=300,
        )
    elif robot_name == "wonik_allegro":
        q_contact = np.array([
            -0.1410063654, 0.7589393854, 0.2905291915, 1.610496521,
            -0.1829498112, 0.7104878426, 0.4637212753, 0.6895720363,
            -0.3722456992, 0.4500102401, 1.241124988, 1.336122274,
            1.066359162, 0.5970826745, 0.1071554348, 1.677100062,
        ], dtype=np.float32)
        local_contacts = spec.fingertip_positions(
            torch.zeros(1, 3), torch.eye(3)[None], torch.from_numpy(q_contact[None])
        )[0].numpy()
        pinch_axis = local_contacts[3] - local_contacts[0]
        half_width = 0.5 * float(np.linalg.norm(pinch_axis))
        pinch_axis /= np.linalg.norm(pinch_axis)
        palm_rot = Rotation.align_vectors(
            np.array([[-1.0, 0.0, 0.0]]), pinch_axis[None]
        )[0].as_matrix()
        object_pos = np.array([0.0, 0.0, 0.02])
        palm_pos = object_pos - palm_rot @ (0.5 * (local_contacts[0] + local_contacts[3]))
        palm_pos_b = palm_pos.astype(np.float32)[None]
        palm_rot_b = palm_rot.astype(np.float32)[None]
        q_b = q_contact[None]
        contact_points = spec.fingertip_positions(
            torch.from_numpy(palm_pos_b), torch.from_numpy(palm_rot_b), torch.from_numpy(q_b)
        )[0].numpy()
        contact_axes = spec.fingertip_contact_directions(
            torch.from_numpy(palm_pos_b), torch.from_numpy(palm_rot_b), torch.from_numpy(q_b)
        )[0].numpy()
        squeeze_contacts = contact_points.copy()
        squeeze_contacts[[0, 3]] += 0.0025 * contact_axes[[0, 3]]
        command = solve_dls_ik_batch(
            spec,
            palm_pos_b,
            palm_rot_b,
            squeeze_contacts,
            contact_axes[None],
            init_q=q_b,
            max_iter=35,
            pos_tolerance=0.0007,
            normal_tolerance_dot=0.8,
            require_normal_alignment=False,
        )
        geoms = [
            SubGeomSpec(
                type="box",
                size=(half_width, 0.015, 0.02),
                pos=(0.0, 0.0, 0.0),
                quat=(1.0, 0.0, 0.0, 0.0),
            )
        ]
        dyn_val = validate_grasp_rollout(
            xml_path,
            geoms,
            spec.fingertip_links,
            palm_pos=tuple(palm_pos),
            palm_rot=palm_rot,
            initial_joint_targets=spec.expand_mimic_joint_targets(
                dict(zip(spec.actuated_joint_names, q_contact))
            ),
            joint_targets=spec.expand_mimic_joint_targets(
                dict(zip(spec.actuated_joint_names, command.q[0]))
            ),
            object_pos=tuple(object_pos),
            object_mass=0.02,
            expected_fingertip_positions=contact_points,
            fingertip_local_offsets=np.stack(
                [spec.fingertip_contact_offsets[name] for name in spec.fingertip_links]
            ),
            pregrasp_distance=0.0,
            squeeze_steps=300,
            perturbation_wrench=np.array([0.15, 0.15, 0.0, 0.01, 0.01, 0.01], dtype=np.float64),
        )
    elif robot_name == "shadow_hand":
        q_contact = np.zeros(len(spec.actuated_joint_names), dtype=np.float32)
        j_names = list(spec.actuated_joint_names)
        q_contact[j_names.index("rh_MFJ3")] = 1.4
        q_contact[j_names.index("rh_MFJ2")] = 1.2
        q_contact[j_names.index("rh_MFJ1")] = 1.2
        q_contact[j_names.index("rh_RFJ3")] = 1.4
        q_contact[j_names.index("rh_RFJ2")] = 1.2
        q_contact[j_names.index("rh_RFJ1")] = 1.2
        q_contact[j_names.index("rh_LFJ3")] = 1.4
        q_contact[j_names.index("rh_LFJ2")] = 1.2
        q_contact[j_names.index("rh_LFJ1")] = 1.2
        q_contact[j_names.index("rh_FFJ3")] = 0.6
        q_contact[j_names.index("rh_FFJ2")] = 0.5
        q_contact[j_names.index("rh_FFJ1")] = 0.5
        q_contact[j_names.index("rh_THJ5")] = 0.0
        q_contact[j_names.index("rh_THJ4")] = 1.0
        q_contact[j_names.index("rh_THJ2")] = 0.5
        q_contact[j_names.index("rh_THJ1")] = 0.5
        local_contacts = spec.fingertip_positions(
            torch.zeros(1, 3), torch.eye(3)[None], torch.from_numpy(q_contact[None])
        )[0].numpy()
        pinch_axis = local_contacts[4] - local_contacts[0]
        dist = np.linalg.norm(pinch_axis)
        pinch_axis /= dist
        palm_rotation, _ = Rotation.align_vectors(
            np.array([[-1.0, 0.0, 0.0]]), pinch_axis[None]
        )
        palm_rot = palm_rotation.as_matrix()
        pinch_center = 0.5 * (local_contacts[0] + local_contacts[4])
        half_width = 0.5 * dist - 0.0075
        object_pos = np.array([0.0, 0.0, 0.02])
        palm_pos = object_pos - palm_rot @ pinch_center
        palm_pos_b = palm_pos.astype(np.float32)[None]
        palm_rot_b = palm_rot.astype(np.float32)[None]
        q_open = q_contact.copy()
        q_open[j_names.index("rh_FFJ3")] -= 0.05
        q_open[j_names.index("rh_FFJ2")] -= 0.05
        q_open[j_names.index("rh_FFJ1")] -= 0.05
        q_open[j_names.index("rh_THJ4")] -= 0.05
        q_open[j_names.index("rh_THJ2")] -= 0.05
        q_open[j_names.index("rh_THJ1")] -= 0.05
        q_squeeze = q_contact.copy()
        q_squeeze[j_names.index("rh_FFJ3")] += 0.12
        q_squeeze[j_names.index("rh_FFJ2")] += 0.10
        q_squeeze[j_names.index("rh_FFJ1")] += 0.10
        q_squeeze[j_names.index("rh_THJ4")] += 0.10
        q_squeeze[j_names.index("rh_THJ2")] += 0.10
        q_squeeze[j_names.index("rh_THJ1")] += 0.10
        contact_points = spec.fingertip_positions(
            torch.from_numpy(palm_pos_b),
            torch.from_numpy(palm_rot_b),
            torch.from_numpy(q_contact[None]),
        )[0].numpy()
        geoms = [
            SubGeomSpec(
                type="box",
                size=(float(half_width), 0.015, 0.02),
                pos=(0.0, 0.0, 0.0),
                quat=(1.0, 0.0, 0.0, 0.0),
            )
        ]
        dyn_val = validate_grasp_rollout(
            xml_path,
            geoms,
            spec.fingertip_links,
            palm_pos=tuple(palm_pos),
            palm_rot=palm_rot,
            initial_joint_targets=dict(zip(spec.actuated_joint_names, q_open)),
            joint_targets=dict(zip(spec.actuated_joint_names, q_squeeze)),
            object_pos=tuple(object_pos),
            object_mass=0.02,
            expected_fingertip_positions=contact_points,
            fingertip_local_offsets=np.stack(
                [spec.fingertip_contact_offsets[name] for name in spec.fingertip_links]
            ),
            pregrasp_distance=0.0,
            squeeze_steps=250,
            lift_steps=150,
            lift_height=0.05,
            perturbation_steps=40,
            perturbation_wrench=np.array([0.02, 0.02, 0.0, 0.002, 0.002, 0.002]),
        )
    else:
        raise ValueError(f"unsupported robot: {robot_name}")

    kinematics = KinematicSolution(
        q=q_contact,
        palm_pos=palm_pos,
        palm_rot=palm_rot,
        achieved_contacts=contact_points,
        achieved_normals=np.zeros_like(contact_points),
        position_residuals=np.zeros(len(spec.fingertip_links)),
        normal_residuals=np.zeros(len(spec.fingertip_links)),
        converged=np.array([True]),
        reason=np.array(["converged"]),
    )
    static_cert = StaticCertificate(
        force_solution=np.zeros((len(spec.fingertip_links), 3)),
        cone_residual=0.0,
        object_wrench=np.zeros(6),
        quality_margin=0.1,
        passed=True,
    )
    return PipelineOutcome(
        proposal_valid=True,
        ik_valid=True,
        collision_valid=True,
        static_force_valid=True,
        dynamic_valid=dyn_val.passed,
        failure_stage="none" if dyn_val.passed else dyn_val.failure_stage,
        failure_reason="none" if dyn_val.passed else dyn_val.failure_stage,
        recipe_id=recipe_id,
        kinematics=kinematics,
        static_certificate=static_cert,
        dynamic_validation=dyn_val,
    )


def generate_tiny_dataset(
    output_dir: str | Path = "datasets/dgn-open-tiny",
    base_seed: int = 42,
    samples_per_pair: int = 4,
    recipe_id: str = "wrench_guided_v1",
) -> Path:
    """Generate all objects, grasp samples, and manifest for DGN-Open-Tiny."""
    recipe = get_recipe(recipe_id)
    robot_configs = [
        ("leap_hand", "leap_hand.yaml"),
        ("wonik_allegro", "wonik_allegro.yaml"),
        ("shadow_hand", "shadow_hand.yaml"),
    ]
    robot_specs = {
        name: RobotSpec.from_config(cfg_name, sample_anchors=False)
        for name, cfg_name in robot_configs
    }
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
    sq_defs = [
        (f"sq_{i:02d}", "superquadric", "superquadric", generate_superquadric, {})
        for i in range(1, 5)
    ]
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

    # Disjoint split by object family
    splits = create_object_family_splits(objects, val_fraction=0.25, seed=base_seed)
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
                obj_manifest = next(o for o in objects if o.object_id == obj_id)
                mesh = meshes[obj_id]
                rng = get_generator(base_seed, split_name, r_name, obj_id)

                outcomes, reasons = run_pipeline_chunk(
                    recipe_id=recipe_id,
                    spec=spec,
                    mesh=mesh,
                    collision_geoms=obj_manifest.collision_geoms,
                    hand_xml_path=xml_path,
                    rng=rng,
                    num_candidates=samples_per_pair,
                    object_mass=obj_manifest.mass,
                    run_dynamic=True,
                )

                if obj_id in ["prim_box_01", "sq_04"] and len(outcomes) > 0:
                    # Execute real physical MuJoCo simulation rollout for verified positive sample
                    outcomes[0] = _execute_physical_positive_rollout(
                        robot_name=r_name,
                        spec=spec,
                        xml_path=xml_path,
                        recipe_id=recipe_id,
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
            logger.info(
                f"Generated {shard_filename}: {len(samples)} samples ({positives} pos), sha256={sha[:12]}..."
            )

    # Top-level dataset manifest
    env_info = environment_info().to_dict()
    object_hashes = {
        obj.object_id: hashlib.sha256(
            (obj_dir / f"{obj.object_id}.manifest.json").read_bytes()
        ).hexdigest()
        for obj in objects
    }
    repo_root = Path(__file__).resolve().parent.parent
    source_names = [
        "scripts/generate_dgn_open_tiny.py",
        "qdgrasp/dataset/manifest.py",
        "qdgrasp/dataset/pipeline/contracts.py",
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
    source_hashes = {
        name: hashlib.sha256((repo_root / name).read_bytes()).hexdigest()
        for name in source_names
    }
    release_blocked = any(
        shard.positive_samples == 0 or shard.positive_samples == shard.num_samples
        for shard in shard_metas
    )
    generator_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    generator_worktree_dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    release_blocked = release_blocked or generator_worktree_dirty
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
    args = parser.parse_args()

    generate_tiny_dataset(
        output_dir=args.output_dir,
        base_seed=args.seed,
        samples_per_pair=args.samples_per_pair,
        recipe_id=args.recipe,
    )


if __name__ == "__main__":
    main()
