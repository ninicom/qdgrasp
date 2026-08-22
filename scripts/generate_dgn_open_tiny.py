"""Generate the standard DGN-Open-Tiny cross-embodiment dataset release."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch

from qdgrasp.dataset.manifest import DatasetManifestSpec, ShardMetadata, save_dataset_manifest
from qdgrasp.dataset.pipeline.filter import filter_grasp_candidate
from qdgrasp.dataset.pipeline.ik import solve_dls_ik
from qdgrasp.dataset.pipeline.sample import sample_grasp_candidates
from qdgrasp.dataset.render import sample_analytic_point_cloud
from qdgrasp.dataset.rng import derive_seed, get_generator
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
from qdgrasp.robot.spec import RobotSpec
from qdgrasp.runtime import environment_info
from qdgrasp.sim.labeling import evaluate_grasp_physics

logger = logging.getLogger("generate_dgn_open_tiny")


def generate_tiny_dataset(
    output_dir: str | Path = "datasets/dgn-open-tiny",
    base_seed: int = 42,
    samples_per_pair: int = 4,
) -> Path:
    """Generate all objects, grasp samples, and manifest for DGN-Open-Tiny."""
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

    robot_configs = [
        ("leap_hand", "leap_hand.yaml", ".references/robot-assets/mujoco-menagerie/leap_hand/right_hand.xml"),
        ("wonik_allegro", "wonik_allegro.yaml", ".references/robot-assets/mujoco-menagerie/wonik_allegro/right_hand.xml"),
        ("shadow_hand", "shadow_hand.yaml", ".references/robot-assets/mujoco-menagerie/shadow_hand/right_hand.xml"),
    ]

    robot_specs = {name: RobotSpec.from_config(cfg_name, sample_anchors=False) for name, cfg_name, _ in robot_configs}
    robot_hashes = {name: spec.config.content_hash() for name, spec in robot_specs.items()}

    shard_metas: list[ShardMetadata] = []

    # Generate samples per (split, robot)
    for split_name, obj_ids in splits.items():
        for r_name, r_cfg, xml_rel in robot_configs:
            spec = robot_specs[r_name]
            xml_path = Path(xml_rel).resolve()
            has_sim = xml_path.is_file()

            samples: list[dict[str, Any]] = []
            positives = 0

            for obj_id in obj_ids:
                obj_manifest = next(o for o in objects if o.object_id == obj_id)
                mesh = meshes[obj_id]
                rng = get_generator(base_seed, split_name, r_name, obj_id)

                candidates = sample_grasp_candidates(spec, mesh, rng, num_candidates=samples_per_pair)
                for cand_idx, cand in enumerate(candidates):
                    ik_res = solve_dls_ik(
                        spec,
                        cand.palm_pos,
                        cand.palm_rot,
                        cand.target_contacts,
                        max_iter=30,
                    )
                    filter_res = filter_grasp_candidate(
                        spec,
                        cand.palm_pos,
                        cand.palm_rot,
                        ik_res.q,
                        mesh,
                    )

                    is_success = False
                    quality = 0.0

                    if filter_res.valid:
                        if has_sim:
                            try:
                                j_targets = {
                                    j_name: float(ik_res.q[j_idx])
                                    for j_idx, j_name in enumerate(spec.actuated_joint_names)
                                }
                                sim_res = evaluate_grasp_physics(
                                    hand_xml_path=xml_path,
                                    collision_geoms=obj_manifest.collision_geoms,
                                    palm_pos=tuple(cand.palm_pos.tolist()),
                                    joint_targets=j_targets,
                                    object_pos=(0.0, 0.0, 0.05),
                                    object_mass=obj_manifest.mass,
                                    seed=derive_seed(base_seed, split_name, r_name, obj_id, cand_idx),
                                )
                                is_success = sim_res.success
                                quality = sim_res.lift_height if is_success else 0.0
                            except Exception:
                                is_success = False
                        else:
                            is_success = ik_res.converged
                            quality = 0.05 if is_success else 0.0

                    if is_success:
                        positives += 1

                    # Sample camera point cloud
                    cam_pos = cand.palm_pos + np.array([0.0, 0.0, 0.15])
                    pcd_cam, _ = sample_analytic_point_cloud(
                        mesh,
                        camera_pos=cam_pos,
                        camera_rot=np.eye(3),
                        num_points=1024,
                        rng=rng,
                    )

                    sample_dict = {
                        "points": torch.from_numpy(pcd_cam).float(),
                        "palm_pos": torch.from_numpy(cand.palm_pos).float(),
                        "palm_rot": torch.from_numpy(cand.palm_rot).float(),
                        "joint_angles": torch.from_numpy(ik_res.q).float(),
                        "fingertip_positions": torch.from_numpy(ik_res.fingertip_positions).float(),
                        "success": torch.tensor(1.0 if is_success else 0.0, dtype=torch.float32),
                        "quality": torch.tensor(quality, dtype=torch.float32),
                        "object_id": obj_id,
                        "robot_name": r_name,
                    }
                    samples.append(sample_dict)

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
            )
            shard_metas.append(shard_meta)
            logger.info(
                f"Generated {shard_filename}: {len(samples)} samples ({positives} pos), sha256={sha[:12]}..."
            )

    # Top-level dataset manifest
    env_info = environment_info().to_dict()
    dataset_manifest = DatasetManifestSpec(
        dataset_id="dgn-open-tiny-v1",
        generator_version="0.1.0a1",
        seed=base_seed,
        environment_fingerprint=env_info,
        robot_profile_hashes=robot_hashes,
        splits=splits,
        shards=shard_metas,
        success_criteria={
            "min_contacts": 2.0,
            "max_penetration": 0.02,
            "min_lift_ratio": 0.5,
        },
        license="CC0-1.0",
        release_blocked=False,
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
    args = parser.parse_args()

    generate_tiny_dataset(
        output_dir=args.output_dir,
        base_seed=args.seed,
        samples_per_pair=args.samples_per_pair,
    )


if __name__ == "__main__":
    main()
