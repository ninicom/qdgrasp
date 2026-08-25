"""Bounded deterministic generator for the QDGrasp-Scene-Tiny release."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import trimesh

from qdgrasp.config.schema import ConfigError
from qdgrasp.dataset.pipeline.validators.mujoco_rollout import RolloutSceneObject
from qdgrasp.dataset.pipeline.validators.scene_dynamic import (
    SceneDynamicValidator,
    hash_scene_state,
)
from qdgrasp.dataset.pipeline.validators.scene_rollout import (
    SceneRolloutResult,
    run_scene_grasp_rollout,
)
from qdgrasp.dataset.rng import get_generator
from qdgrasp.dataset.scene_loader import audit_scene_dataset
from qdgrasp.dataset.scene_manifest import (
    SceneDatasetManifest,
    SceneShardMetadata,
    save_scene_manifest,
)
from qdgrasp.dataset.scene_shards import write_scene_shard
from qdgrasp.objects.generate import generate_box, generate_superquadric
from qdgrasp.objects.manifest import create_object_asset, save_object_asset
from qdgrasp.objects.schema import ObjectManifestSpec, SubGeomSpec
from qdgrasp.robot.spec import resolve_robot_asset
from qdgrasp.scenes.builders.base import build_scene_mujoco_model
from qdgrasp.scenes.contracts import CameraSpec, SceneObjectSpec, SceneSpec
from qdgrasp.scenes.environments import get_environment
from qdgrasp.scenes.observations.renderer import (
    build_scene_observation,
    scene_observation_record,
)
from qdgrasp.scenes.observations.stage_evidence import capture_stage_evidence
from qdgrasp.scenes.release_recipes import SceneGraspRecipe, build_release_grasp_recipe

DATASET_ID = "QDGrasp-Scene-Tiny"
GENERATOR_VERSION = "1.0.0"
ROBOT_PROFILES = ("leap_hand.yaml", "wonik_allegro.yaml", "shadow_hand.yaml")


@dataclass(frozen=True)
class SceneBlueprint:
    scene_id: str
    environment: str
    clutter_tier: str
    object_count: int
    split: str
    template_id: str
    robot_profile: str | None = None


def release_blueprints() -> tuple[SceneBlueprint, ...]:
    """Return the complete ordered 12-scene release index without touching disk."""
    return (
        SceneBlueprint("table-leap-sparse", "table", "sparse", 2, "train", "table-pinch-a", ROBOT_PROFILES[0]),
        SceneBlueprint("table-allegro-sparse", "table", "sparse", 3, "train", "table-pinch-b", ROBOT_PROFILES[1]),
        SceneBlueprint("table-shadow-dense", "table", "dense", 6, "train", "table-pinch-c", ROBOT_PROFILES[2]),
        SceneBlueprint("table-single-val", "table", "single", 1, "validation", "table-single-d"),
        SceneBlueprint("bin-single", "bin", "single", 1, "train", "bin-single-a"),
        SceneBlueprint("bin-sparse", "bin", "sparse", 3, "train", "bin-grid-b"),
        SceneBlueprint("bin-dense", "bin", "dense", 6, "train", "bin-grid-c"),
        SceneBlueprint("bin-sparse-val", "bin", "sparse", 4, "validation", "bin-grid-d"),
        SceneBlueprint("shelf-single", "shelf", "single", 1, "train", "shelf-single-a"),
        SceneBlueprint("shelf-sparse", "shelf", "sparse", 3, "train", "shelf-grid-b"),
        SceneBlueprint("shelf-dense", "shelf", "dense", 6, "train", "shelf-grid-c"),
        SceneBlueprint("shelf-sparse-val", "shelf", "sparse", 4, "validation", "shelf-grid-d"),
    )


def _json_value(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return _json_value(dataclasses.asdict(value))
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(_json_value(value), sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _hash(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(_json_value(payload), handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        Path(temporary_name).replace(path)
    except Exception:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def _look_at(position: np.ndarray, target: np.ndarray) -> np.ndarray:
    backward = position - target
    backward /= np.linalg.norm(backward)
    right = np.cross(np.array([0.0, 0.0, 1.0]), backward)
    right_norm = float(np.linalg.norm(right))
    if right_norm <= np.finfo(np.float64).eps:
        right = np.array([1.0, 0.0, 0.0])
    else:
        right /= right_norm
    up = np.cross(backward, right)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = np.column_stack([right, up, backward])
    transform[:3, 3] = position
    return transform


def _cameras(width: int, height: int, target_height: float) -> list[CameraSpec]:
    intrinsics = np.array(
        [[120.0, 0.0, width / 2.0], [0.0, 120.0, height / 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    target = np.array([0.0, 0.0, target_height], dtype=np.float64)
    return [
        CameraSpec(
            camera_id="cam_top",
            intrinsics=intrinsics.copy(),
            T_world_camera=_look_at(np.array([0.0, 0.0, target_height + 0.6]), target),
        ),
        CameraSpec(
            camera_id="cam_oblique",
            intrinsics=intrinsics.copy(),
            T_world_camera=_look_at(np.array([0.38, -0.38, target_height + 0.45]), target),
        ),
    ]


def _inertia_box(mass: float, extents: np.ndarray) -> tuple[float, float, float]:
    x, y, z = extents
    return (
        float(mass * (y * y + z * z) / 12.0),
        float(mass * (x * x + z * z) / 12.0),
        float(mass * (x * x + y * y) / 12.0),
    )


def _save_recipe_target(recipe: SceneGraspRecipe, object_id: str, output_dir: Path) -> tuple[Path, ObjectManifestSpec]:
    geom = recipe.target_geoms[0]
    extents = 2.0 * np.asarray(geom.size, dtype=np.float64)
    mesh = trimesh.creation.box(extents=extents)
    mesh_bytes, manifest = create_object_asset(
        object_id=object_id,
        family="primitive",
        shape_type="box",
        mesh=mesh,
        collision_geoms=recipe.target_geoms,
        params={"extents": extents.tolist(), "release_recipe": recipe.recipe_id},
        mass=recipe.target_object_mass,
        inertia=_inertia_box(recipe.target_object_mass, extents),
    )
    return save_object_asset(mesh_bytes, manifest, output_dir), manifest


def _save_generated_object(
    *, object_id: str, split: str, seed: int, output_dir: Path
) -> tuple[Path, ObjectManifestSpec]:
    rng = get_generator(seed, "scene-release-object", object_id)
    if split == "validation":
        mesh, geoms, params, mass, inertia = generate_superquadric(
            rng, scale_range=(0.018, 0.028), n_eta=16, n_omega=16
        )
        family, shape_type = "superquadric", "superquadric"
    else:
        mesh, geoms, params, mass, inertia = generate_box(rng, size_range=(0.035, 0.055), density=350.0)
        family, shape_type = "primitive", "box"
    mesh_bytes, manifest = create_object_asset(
        object_id=object_id,
        family=family,
        shape_type=shape_type,
        mesh=mesh,
        collision_geoms=geoms,
        params=params,
        mass=mass,
        inertia=inertia,
    )
    return save_object_asset(mesh_bytes, manifest, output_dir), manifest


def _positions(count: int, *, base_height: float, environment: str) -> list[tuple[float, float, float]]:
    table_xy = [
        (0.0, 0.0),
        (0.30, 0.0),
        (-0.30, 0.0),
        (0.0, 0.24),
        (0.0, -0.24),
        (0.28, 0.22),
        (-0.28, -0.22),
        (0.28, -0.22),
        (-0.28, 0.22),
        (0.15, 0.15),
    ]
    compact_xy = [
        (0.0, 0.0),
        (0.12, 0.0),
        (-0.12, 0.0),
        (0.0, 0.10),
        (0.0, -0.10),
        (0.12, 0.10),
        (-0.12, -0.10),
        (0.12, -0.10),
        (-0.12, 0.10),
        (0.06, 0.06),
    ]
    xy = table_xy if environment == "table" else compact_xy
    return [(x, y, base_height) for x, y in xy[:count]]


def _scene_spec_payload(spec: SceneSpec, root: Path) -> dict[str, Any]:
    return {
        "scene_id": spec.scene_id,
        "source_dataset": spec.source_dataset,
        "source_version": spec.source_version,
        "source_split": spec.source_split,
        "environment": spec.environment,
        "objects": [
            {
                "object_id": item.object_id,
                "asset_ref": Path(item.asset_ref).resolve().relative_to(root).as_posix(),
                "T_world_object": item.T_world_object,
                "scale": item.scale,
                "mass": item.mass,
                "friction": item.friction,
            }
            for item in spec.objects
        ],
        "supports": [
            {
                "support_id": item.support_id,
                "geom_type": item.geom_type,
                "params": item.params,
                "T_world_support": item.T_world_support,
            }
            for item in spec.supports
        ],
        "cameras": [
            {
                "camera_id": item.camera_id,
                "intrinsics": item.intrinsics,
                "distortion": item.distortion,
                "T_world_camera": item.T_world_camera,
            }
            for item in spec.cameras
        ],
        "gravity": spec.gravity,
        "timestep": spec.timestep,
        "solver_profile": spec.solver_profile,
        "settle_seed": spec.settle_seed,
        "source_record_hash": spec.source_record_hash,
        "license_record": spec.license_record,
        "redistributable": spec.redistributable,
    }


def _scene_state_from_spec(spec: SceneSpec) -> dict[str, dict[str, np.ndarray]]:
    return {
        item.object_id: {
            "pos": np.asarray(item.T_world_object[:3, 3], dtype=np.float64),
            "quat": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
        }
        for item in spec.objects
    }


def _lineage_hash(parent: str, stage: str, state_hash: str) -> str:
    return _hash({"parent": parent, "stage": stage, "state_hash": state_hash})


def _state_records(
    spec: SceneSpec,
    spec_payload: dict[str, Any],
    rollout: SceneRolloutResult | None,
) -> list[dict[str, Any]]:
    initial_state = _scene_state_from_spec(spec)
    states: list[tuple[str, Any, str]] = [
        ("initial", initial_state, hash_scene_state(initial_state)),
        ("settled", initial_state, hash_scene_state(initial_state)),
    ]
    if rollout is not None:
        states = [(stage, state, rollout.state_hashes[stage]) for stage, state in rollout.stage_states.items()]
        states.insert(1, ("settled", states[0][1], states[0][2]))
    records: list[dict[str, Any]] = []
    parent = "0" * 64
    for stage, state, state_hash in states:
        lineage = _lineage_hash(parent, stage, state_hash)
        record = {
            "record_type": "scene_state",
            "scene_id": spec.scene_id,
            "stage": stage,
            "state_hash": state_hash,
            "lineage_hash": lineage,
            "object_poses": state,
        }
        if stage in {"initial", "settled"}:
            record["scene_spec"] = spec_payload
        records.append(record)
        parent = lineage
    return records


def _approach_path(recipe: SceneGraspRecipe) -> np.ndarray:
    direction = np.asarray(recipe.rollout_kwargs["pregrasp_direction"], dtype=np.float64)
    direction /= np.linalg.norm(direction)
    target = np.eye(4, dtype=np.float64)
    target[:3, :3] = np.asarray(recipe.rollout_kwargs["palm_rot"], dtype=np.float64)
    target[:3, 3] = np.asarray(recipe.rollout_kwargs["palm_pos"], dtype=np.float64)
    pregrasp = target.copy()
    pregrasp[:3, 3] += float(recipe.rollout_kwargs["pregrasp_distance"]) * direction
    return np.stack([pregrasp, target])


def _rollout_objects(
    spec: SceneSpec,
    manifests: dict[str, ObjectManifestSpec],
    target_id: str,
) -> list[RolloutSceneObject]:
    return [
        RolloutSceneObject(
            object_id=item.object_id,
            collision_geoms=manifests[item.object_id].collision_geoms,
            pos=tuple(float(value) for value in item.T_world_object[:3, 3]),
            mass=float(manifests[item.object_id].mass),
        )
        for item in spec.objects
        if item.object_id != target_id
    ]


def _positive_record(
    blueprint: SceneBlueprint,
    recipe: SceneGraspRecipe,
    target_id: str,
    rollout: SceneRolloutResult,
    source_hash: str,
    qa_records: list[dict[str, Any]],
) -> dict[str, Any]:
    validation = rollout.validation
    metrics = dict(validation.trajectory_metrics)
    active_indices = [0, len(recipe.robot_spec.fingertip_links) - 1]
    expected = np.asarray(recipe.rollout_kwargs["expected_fingertip_positions"], dtype=np.float64)
    object_pos = np.asarray(recipe.target_object_pos, dtype=np.float64)
    palm = np.eye(4, dtype=np.float64)
    palm[:3, :3] = np.asarray(recipe.rollout_kwargs["palm_rot"], dtype=np.float64)
    palm[:3, 3] = np.asarray(recipe.rollout_kwargs["palm_pos"], dtype=np.float64)
    return {
        "record_type": "grasp",
        "scene_id": blueprint.scene_id,
        "target_object_id": target_id,
        "robot_profile": recipe.robot_profile,
        "candidate_id": f"{blueprint.scene_id}::{recipe.recipe_id}",
        "source_class": "native_measured_release",
        "dynamic_valid": bool(validation.passed),
        "label_stage": "dynamic_valid" if validation.passed else "rejected",
        "failure_reason": "none" if validation.passed else validation.failure_stage,
        "contact_opportunity": expected[active_indices] - object_pos,
        "contact_opportunity_normals": np.array([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        "q_command": [recipe.rollout_kwargs["joint_targets"][name] for name in recipe.robot_spec.actuated_joint_names],
        "palm_T_command": palm,
        "active_fingers": [index in active_indices for index in range(len(expected))],
        "approach_path": _approach_path(recipe),
        "static_certificate": {
            "passed": bool(
                validation.passed
                and metrics.get("active_contact_sustained") == 1.0
                and float(metrics.get("max_penetration", float("inf"))) <= 0.002
            ),
            "source": "measured_rollout_contact_window",
            "max_penetration": metrics.get("max_penetration"),
            "max_cone_violation": metrics.get("max_cone_violation"),
        },
        "swept_clearance_metrics": {
            "passed": metrics.get("swept_clearance_passed") == 1.0,
            "reason": metrics.get("clearance_reason", "none"),
        },
        "dynamic_trajectory_evidence": {
            **metrics,
            "per_finger_loads": validation.per_finger_loads,
        },
        "target_motion": {"lift": metrics.get("measured_target_lift", 0.0)},
        "non_target_motion": metrics.get("non_target_motion", {}),
        "scene_state_hashes": dict(rollout.state_hashes),
        "protocol_hash": recipe.protocol_hash,
        "recipe_hash": recipe.recipe_hash,
        "source_hash": source_hash,
        "qa_stage_evidence": qa_records,
    }


def _rejected_record(
    scene_id: str,
    candidate_id: str,
    failure_reason: str,
    evidence: dict[str, Any],
) -> dict[str, Any]:
    return {
        "record_type": "grasp",
        "scene_id": scene_id,
        "candidate_id": candidate_id,
        "dynamic_valid": False,
        "label_stage": "rejected",
        "failure_reason": failure_reason,
        "source_class": "native_measured_release",
        "rejection_evidence": evidence,
    }


def _measured_negative_records(
    spec: SceneSpec,
    manifests: dict[str, ObjectManifestSpec],
    recipe: SceneGraspRecipe,
    source_hash: str,
) -> list[dict[str, Any]]:
    """Generate collision and non-target-disturbance negatives from real rollouts."""
    target_id = spec.objects[0].object_id
    blocked = RolloutSceneObject(
        object_id="blocked-approach",
        collision_geoms=(SubGeomSpec(type="box", size=(0.03, 0.03, 0.03), pos=(0.0, 0.0, 0.0)),),
        pos=(0.0, 0.0, float(recipe.target_object_pos[2]) + 0.09),
        mass=0.03,
    )
    blocked_result = run_scene_grasp_rollout(
        recipe.hand_xml_path,
        recipe.target_geoms,
        recipe.robot_spec.fingertip_links,
        target_object_id=target_id,
        non_target_objects=[blocked],
        protocol_hash=recipe.protocol_hash,
        recipe_hash=recipe.recipe_hash,
        source_hash=source_hash,
        rollout_kwargs=recipe.rollout_kwargs,
    )
    if blocked_result.validation.passed:
        raise ConfigError("blocked-approach negative unexpectedly passed")

    disturbed_objects = _rollout_objects(spec, manifests, target_id)
    if not disturbed_objects:
        raise ConfigError("disturbance negative requires a sparse positive scene")
    lifted = disturbed_objects[0]
    disturbed_objects[0] = dataclasses.replace(lifted, pos=(lifted.pos[0], lifted.pos[1], lifted.pos[2] + 0.002))
    disturbed_result = run_scene_grasp_rollout(
        recipe.hand_xml_path,
        recipe.target_geoms,
        recipe.robot_spec.fingertip_links,
        target_object_id=target_id,
        non_target_objects=disturbed_objects,
        protocol_hash=recipe.protocol_hash,
        recipe_hash=recipe.recipe_hash,
        source_hash=source_hash,
        scene_validator=SceneDynamicValidator(
            displacement_threshold=0.0001,
            rotation_threshold=0.0001,
            impulse_threshold=0.0001,
        ),
        rollout_kwargs=recipe.rollout_kwargs,
    )
    if disturbed_result.validation.failure_stage != "non_target_disturbed":
        raise ConfigError(
            "non-target disturbance negative did not reach measured disturbance gate: "
            f"{disturbed_result.validation.failure_stage}"
        )
    return [
        _rejected_record(
            spec.scene_id,
            f"{spec.scene_id}::blocked-approach",
            "collision",
            dict(blocked_result.validation.trajectory_metrics),
        ),
        _rejected_record(
            spec.scene_id,
            f"{spec.scene_id}::non-target-disturbed",
            "non_target_disturbance",
            dict(disturbed_result.validation.trajectory_metrics),
        ),
    ]


def _git_identity(repo_root: Path) -> tuple[str, bool]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain", "--untracked-files=no"],
                cwd=repo_root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
        return commit, dirty
    except (OSError, subprocess.CalledProcessError):
        return "unknown", True


def generate_scene_tiny(
    output_root: str | Path,
    *,
    seed: int = 3301,
    scene_limit: int = 12,
    frame_limit: int = 2,
    worker_count: int = 1,
    width: int = 160,
    height: int = 120,
    dry_run: bool = False,
    resume: bool = False,
) -> dict[str, Any]:
    """Generate a bounded release; no recursive source scan or parallel workers."""
    if worker_count != 1:
        raise ConfigError("QDGrasp-Scene-Tiny requires worker_count=1")
    if scene_limit <= 0 or scene_limit > 12:
        raise ConfigError("scene_limit must be in [1, 12]")
    if frame_limit <= 0 or frame_limit > 2:
        raise ConfigError("frame_limit must be in [1, 2]")
    selected = release_blueprints()[:scene_limit]
    summary = {
        "dataset_id": DATASET_ID,
        "scene_count": len(selected),
        "scene_ids": [item.scene_id for item in selected],
        "frame_limit": frame_limit,
        "worker_count": worker_count,
        "full_root_scan": False,
        "source_copy": False,
    }
    if dry_run:
        return summary

    root = Path(output_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    assets_dir = root / "assets" / "objects"
    repo_root = Path(__file__).resolve().parents[2]
    generator_commit, generator_dirty = _git_identity(repo_root)

    scene_spec_hashes: dict[str, str] = {}
    object_hashes: dict[str, str] = {}
    camera_hashes: dict[str, str] = {}
    environment_hashes: dict[str, str] = {}
    observation_records: dict[str, list[dict[str, Any]]] = {"train": [], "validation": []}
    state_records: dict[str, list[dict[str, Any]]] = {"train": [], "validation": []}
    grasp_records: dict[str, list[dict[str, Any]]] = {"train": [], "validation": []}
    qa_artifacts: dict[str, str] = {}
    observation_artifacts: dict[str, str] = {}
    family_by_split: dict[str, set[str]] = {"train": set(), "validation": set()}
    robot_coverage: set[str] = set()
    negative_coverage: set[str] = set()
    completed_count = 0
    checkpoint_path = root / "generation_checkpoint.json"
    if resume:
        if not checkpoint_path.is_file():
            raise ConfigError("resume requested but generation_checkpoint.json is missing")
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        expected_config = {
            "seed": seed,
            "scene_limit": scene_limit,
            "frame_limit": frame_limit,
            "worker_count": worker_count,
            "width": width,
            "height": height,
        }
        if checkpoint.get("config") != expected_config:
            raise ConfigError("resume checkpoint configuration does not match invocation")
        completed_count = int(checkpoint.get("completed_count", 0))
        expected_ids = [item.scene_id for item in selected[:completed_count]]
        if checkpoint.get("completed_scene_ids") != expected_ids:
            raise ConfigError("resume checkpoint scene order is invalid")
        scene_spec_hashes.update(checkpoint["scene_spec_hashes"])
        object_hashes.update(checkpoint["object_hashes"])
        camera_hashes.update(checkpoint["camera_hashes"])
        environment_hashes.update(checkpoint["environment_hashes"])
        observation_records = checkpoint["observation_records"]
        state_records = checkpoint["state_records"]
        grasp_records = checkpoint["grasp_records"]
        qa_artifacts.update(checkpoint["qa_artifacts"])
        observation_artifacts.update(checkpoint["observation_artifacts"])
        family_by_split = {split: set(values) for split, values in checkpoint["family_by_split"].items()}
        robot_coverage.update(checkpoint["robot_coverage"])
        negative_coverage.update(checkpoint["negative_coverage"])
        for reference, digest in {**qa_artifacts, **observation_artifacts}.items():
            artifact = root / reference
            if not artifact.is_file() or _file_hash(artifact) != digest:
                raise ConfigError(f"resume artifact missing or corrupt: {reference}")

    for scene_index in range(completed_count, len(selected)):
        blueprint = selected[scene_index]
        recipe = build_release_grasp_recipe(blueprint.robot_profile) if blueprint.robot_profile else None
        manifests: dict[str, ObjectManifestSpec] = {}
        objects: list[SceneObjectSpec] = []
        base_height = float(recipe.target_object_pos[2]) if recipe else 0.03
        positions = _positions(
            blueprint.object_count,
            base_height=base_height,
            environment=blueprint.environment,
        )
        for object_index in range(blueprint.object_count):
            object_id = f"{blueprint.scene_id}-object-{object_index:02d}"
            if object_index == 0 and recipe is not None:
                manifest_path, manifest = _save_recipe_target(recipe, object_id, assets_dir)
                position = recipe.target_object_pos
            else:
                manifest_path, manifest = _save_generated_object(
                    object_id=object_id,
                    split=blueprint.split,
                    seed=seed,
                    output_dir=assets_dir,
                )
                geom_height = max(float(geom.pos[2] + geom.size[-1]) for geom in manifest.collision_geoms)
                x, y, _ = positions[object_index]
                support_surface = 0.61 if blueprint.environment == "shelf" else 0.0
                position = (x, y, support_surface + geom_height)
            transform = np.eye(4, dtype=np.float64)
            transform[:3, 3] = position
            manifests[object_id] = manifest
            family_by_split[blueprint.split].add(manifest.family)
            object_hashes[object_id] = _hash(
                {"manifest": manifest.model_dump(mode="json"), "manifest_file": _file_hash(manifest_path)}
            )
            objects.append(
                SceneObjectSpec(
                    object_id=object_id,
                    asset_ref=str(manifest_path),
                    T_world_object=transform,
                    mass=float(manifest.mass),
                )
            )
        supports = get_environment(blueprint.environment)
        target_height = float(objects[0].T_world_object[2, 3])
        spec = SceneSpec(
            scene_id=blueprint.scene_id,
            source_dataset="qdgrasp-native",
            source_version=GENERATOR_VERSION,
            source_split=blueprint.split,
            environment=blueprint.environment,
            objects=objects,
            supports=supports,
            cameras=_cameras(width, height, target_height),
            settle_seed=seed + scene_index,
            source_record_hash=_hash(dataclasses.asdict(blueprint)),
            license_record="CC0-1.0",
            redistributable=True,
        )
        payload = _scene_spec_payload(spec, root)
        spec_hash = _hash(payload)
        scene_spec_hashes[spec.scene_id] = spec_hash
        environment_hashes[blueprint.environment] = _hash([dataclasses.asdict(item) for item in supports])
        for camera in spec.cameras:
            camera_hashes[f"{spec.scene_id}/{camera.camera_id}"] = _hash(
                {"intrinsics": camera.intrinsics, "T_world_camera": camera.T_world_camera}
            )

        model = build_scene_mujoco_model(spec, dynamic_objects=False)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        per_scene_observations: list[dict[str, Any]] = []
        for camera in spec.cameras[:frame_limit]:
            observation = build_scene_observation(
                spec,
                model,
                data,
                camera.camera_id,
                "settled",
                root,
                width=width,
                height=height,
            )
            record = scene_observation_record(observation)
            per_scene_observations.append(record)
            observation_records[blueprint.split].append(record)
            for reference in (
                observation.rgb_ref,
                observation.depth_ref,
                observation.point_cloud_ref,
                observation.instance_mask_ref,
            ):
                if reference:
                    observation_artifacts[reference] = _file_hash(root / reference)

        rollout: SceneRolloutResult | None = None
        if recipe is not None:
            target_id = objects[0].object_id
            active = np.zeros(len(recipe.robot_spec.fingertip_links), dtype=bool)
            active[[0, len(active) - 1]] = True
            path = _approach_path(recipe)
            qa_records: list[dict[str, Any]] = []

            def observe(
                stage: str,
                stage_model: mujoco.MjModel,
                stage_data: mujoco.MjData,
                scene_id: str = blueprint.scene_id,
                observed_recipe: SceneGraspRecipe = recipe,
                observed_target_id: str = target_id,
                observed_active: np.ndarray = active,
                observed_path: np.ndarray = path,
                observed_records: list[dict[str, Any]] = qa_records,
            ) -> None:
                record = capture_stage_evidence(
                    stage_model,
                    stage_data,
                    root,
                    scene_id=scene_id,
                    robot_profile=Path(observed_recipe.robot_profile).stem,
                    stage="pregrasp" if stage == "initial" else stage,
                    target_object_id=observed_target_id,
                    fingertip_body_names=observed_recipe.robot_spec.fingertip_links,
                    active_fingers=observed_active,
                    approach_path=observed_path,
                    failure_reason="none",
                    width=max(width, 240),
                    height=max(height, 180),
                )
                observed_records.append(record)
                qa_artifacts[record["image_ref"]] = record["image_sha256"]

            rollout = run_scene_grasp_rollout(
                recipe.hand_xml_path,
                recipe.target_geoms,
                recipe.robot_spec.fingertip_links,
                target_object_id=target_id,
                non_target_objects=_rollout_objects(spec, manifests, target_id),
                protocol_hash=recipe.protocol_hash,
                recipe_hash=recipe.recipe_hash,
                source_hash=spec_hash,
                rollout_kwargs=recipe.rollout_kwargs,
                evidence_stage_observer=observe,
            )
            if not rollout.validation.passed:
                raise ConfigError(
                    f"release positive failed for {blueprint.scene_id}: "
                    f"{rollout.validation.failure_stage} {rollout.validation.trajectory_metrics}"
                )
            grasp_records[blueprint.split].append(
                _positive_record(blueprint, recipe, target_id, rollout, spec_hash, qa_records)
            )
            robot_coverage.add(recipe.robot_profile)
            if scene_index == 0:
                grasp_records[blueprint.split].extend(_measured_negative_records(spec, manifests, recipe, spec_hash))
                negative_coverage.update({"collision", "non_target_disturbance"})

        state_records[blueprint.split].extend(_state_records(spec, payload, rollout))
        minimum_visibility = min(
            (value for record in per_scene_observations for value in record.get("visibility_by_object", {}).values()),
            default=0.0,
        )
        if minimum_visibility < 0.02:
            grasp_records[blueprint.split].append(
                _rejected_record(
                    blueprint.scene_id,
                    f"{blueprint.scene_id}::occlusion-negative",
                    "occlusion",
                    {
                        "source": "rendered_instance_visibility",
                        "minimum_visibility": minimum_visibility,
                        "threshold": 0.02,
                    },
                )
            )
            negative_coverage.add("occlusion")
        _atomic_json(
            checkpoint_path,
            {
                "config": {
                    "seed": seed,
                    "scene_limit": scene_limit,
                    "frame_limit": frame_limit,
                    "worker_count": worker_count,
                    "width": width,
                    "height": height,
                },
                "completed_count": scene_index + 1,
                "completed_scene_ids": [item.scene_id for item in selected[: scene_index + 1]],
                "scene_spec_hashes": scene_spec_hashes,
                "object_hashes": object_hashes,
                "camera_hashes": camera_hashes,
                "environment_hashes": environment_hashes,
                "observation_records": observation_records,
                "state_records": state_records,
                "grasp_records": grasp_records,
                "qa_artifacts": qa_artifacts,
                "observation_artifacts": observation_artifacts,
                "family_by_split": {split: sorted(values) for split, values in family_by_split.items()},
                "robot_coverage": sorted(robot_coverage),
                "negative_coverage": sorted(negative_coverage),
            },
        )

    required_negatives = {"collision", "occlusion", "non_target_disturbance"}
    if not required_negatives.issubset(negative_coverage):
        raise ConfigError(f"release negative coverage incomplete: {sorted(required_negatives - negative_coverage)}")

    shard_metadata: list[SceneShardMetadata] = []
    for split in ("train", "validation"):
        for record_type, records in (
            ("scene_state", state_records[split]),
            ("observation", observation_records[split]),
            ("grasp", grasp_records[split]),
        ):
            if not records:
                continue
            filename = f"shards/{split}-{record_type}.jsonl"
            digest = write_scene_shard(records, root / filename, record_type=record_type)
            shard_metadata.append(
                SceneShardMetadata(
                    filename=filename,
                    sha256=digest,
                    num_records=len(records),
                    record_type=record_type,
                    split=split,
                )
            )

    splits = {
        split: [item.scene_id for item in selected if item.split == split]
        for split in ("train", "validation")
        if any(item.split == split for item in selected)
    }
    split_hashes = {split: _hash(scene_ids) for split, scene_ids in splits.items()}
    robot_hashes = {
        profile: _hash(
            {
                "config": (repo_root / "qdgrasp" / "presets" / "robots" / profile).read_bytes().hex(),
                "asset": _file_hash(
                    resolve_robot_asset(build_release_grasp_recipe(profile).robot_spec.config.source_asset)
                ),
            }
        )
        for profile in sorted(robot_coverage)
    }
    artifact_hashes = {**observation_artifacts, **qa_artifacts}
    manifest = SceneDatasetManifest(
        dataset_id=DATASET_ID,
        generator_version=GENERATOR_VERSION,
        generator_commit=generator_commit,
        generator_worktree_dirty=generator_dirty,
        seed=seed,
        splits=splits,
        scene_spec_hashes=scene_spec_hashes,
        camera_calibration_hashes=camera_hashes,
        environment_hashes=environment_hashes,
        object_asset_hashes=object_hashes,
        robot_profile_hashes=robot_hashes,
        split_hashes=split_hashes,
        release_artifact_hashes=artifact_hashes,
        source_licenses={"qdgrasp-native": "CC0-1.0"},
        shards=tuple(shard_metadata),
        success_criteria={
            "minimum_target_lift": 0.025,
            "maximum_penetration": 0.002,
            "minimum_camera_views": 2.0,
        },
        coverage={
            "scene_count": len(selected),
            "environment_counts": {
                environment: sum(item.environment == environment for item in selected)
                for environment in ("table", "bin", "shelf")
            },
            "clutter_tier_counts": {
                tier: sum(item.clutter_tier == tier for item in selected) for tier in ("single", "sparse", "dense")
            },
            "object_families_by_split": {
                split: sorted(families) for split, families in family_by_split.items() if split in splits
            },
            "scene_templates": {item.scene_id: item.template_id for item in selected},
            "robot_profiles": sorted(robot_coverage),
            "negative_classes": sorted(negative_coverage),
            "qa_stage_images": len(qa_artifacts),
        },
        resource_policy={
            "scene_limit": scene_limit,
            "frame_limit": frame_limit,
            "worker_count": worker_count,
            "full_root_scan": False,
            "source_copy": False,
            "checkpoint_file": "generation_checkpoint.json",
        },
        release_blocked=scene_limit != 12 or frame_limit != 2,
    )
    save_scene_manifest(manifest, root / "scene_manifest.json")
    save_scene_manifest(manifest, root / "dataset_manifest.json")
    counts = audit_scene_dataset(root)
    _atomic_json(root / "generation_report.json", {**summary, "counts": counts})
    return {**summary, "counts": counts, "release_blocked": manifest.release_blocked}
