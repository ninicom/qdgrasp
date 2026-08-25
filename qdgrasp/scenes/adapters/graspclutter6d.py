"""Adapter for GraspClutter6D's official BOP-compatible layout."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from qdgrasp.config.schema import ConfigError
from qdgrasp.scenes.adapters._common import (
    calibration_hash,
    record_hash,
    require_transform,
    sha256_file,
    source_manifest,
    source_manifest_or_none,
)
from qdgrasp.scenes.contracts import (
    CameraSpec,
    ExternalGraspSet,
    SceneIndex,
    SceneObjectSpec,
    SceneObservation,
    SceneSpec,
    SourceDatasetInfo,
    SourceEvidence,
)

DATASET_ID = "graspclutter6d"
SPLIT_FILES = {
    "grasp_train": "grasp_train_scene_ids.json",
    "grasp_test": "grasp_test_scene_ids.json",
    "ycbv_train": "ycbv_train_scene_ids.json",
    "ycbv_test": "ycbv_test_scene_ids.json",
}


def _load_json(path: Path) -> Any:
    if not path.is_file():
        raise ConfigError(f"source JSON not found: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ConfigError(f"invalid source JSON {path}: {exc}") from exc


def _frame_transform(camera_record: dict[str, Any], frame_id: str) -> tuple[np.ndarray, np.ndarray]:
    try:
        intrinsics = np.asarray(camera_record["cam_K"], dtype=np.float64).reshape(3, 3)
        rotation = np.asarray(camera_record["cam_R_w2c"], dtype=np.float64).reshape(3, 3)
        translation = np.asarray(camera_record["cam_t_w2c"], dtype=np.float64) / 1000.0
    except (KeyError, ValueError) as exc:
        raise ConfigError(f"camera frame {frame_id} lacks BOP world extrinsics/intrinsics") from exc
    world_to_camera = np.eye(4)
    world_to_camera[:3, :3] = rotation
    world_to_camera[:3, 3] = translation
    world_camera = require_transform(np.linalg.inv(world_to_camera), f"frame {frame_id} T_world_camera")
    if not np.all(np.isfinite(intrinsics)):
        raise ConfigError(f"frame {frame_id} has non-finite intrinsics")
    return intrinsics, world_camera


def _object_pose(instance: dict[str, Any], world_camera: np.ndarray, label: str) -> np.ndarray:
    try:
        rotation = np.asarray(instance["cam_R_m2c"], dtype=np.float64).reshape(3, 3)
        translation = np.asarray(instance["cam_t_m2c"], dtype=np.float64) / 1000.0
    except (KeyError, ValueError) as exc:
        raise ConfigError(f"invalid BOP object pose: {label}") from exc
    camera_object = np.eye(4)
    camera_object[:3, :3] = rotation
    camera_object[:3, 3] = translation
    return require_transform(world_camera @ camera_object, label)


def _scene_paths(root: Path, scene_key: str) -> dict[str, Path]:
    scene = root / "scenes" / scene_key
    return {
        "source_manifest": root / "source_manifest.json",
        "camera": scene / "scene_camera.json",
        "ground_truth": scene / "scene_gt.json",
        "ground_truth_info": scene / "scene_gt_info.json",
    }


def _normalize_frame_records(records: dict[str, Any]) -> dict[str, Any]:
    try:
        return {f"{int(frame_id):06d}": value for frame_id, value in records.items()}
    except ValueError as exc:
        raise ConfigError("BOP frame keys must be numeric") from exc


def _split_for_scene(root: Path, scene_key: str) -> str:
    for split, filename in SPLIT_FILES.items():
        path = root / "split_info" / filename
        if not path.is_file():
            continue
        values = _load_json(path)
        normalized = {f"{int(value):06d}" for value in values}
        if scene_key in normalized:
            return split
    return "custom"


class GraspClutter6DAdapter:
    def probe(self, root: str) -> SourceDatasetInfo:
        root_path = Path(root).resolve()
        manifest = source_manifest_or_none(root_path, DATASET_ID)
        is_valid = bool(
            manifest
            and (root_path / "scenes").is_dir()
            and (root_path / "models_obj_m").is_dir()
            and (root_path / "split_info").is_dir()
            and (root_path / "grasp_label").is_dir()
        )
        num_scenes = (
            sum(1 for path in (root_path / "scenes").glob("[0-9][0-9][0-9][0-9][0-9][0-9]") if path.is_dir())
            if is_valid
            else 0
        )
        return SourceDatasetInfo(
            dataset_id=DATASET_ID,
            version=manifest["version"] if manifest else "unknown",
            is_valid=is_valid,
            num_scenes=num_scenes,
            license_type=manifest["license"] if manifest else "unknown",
            redistributable=False,
        )

    def index(self, root: str, split: str, limit: int | None = None) -> SceneIndex:
        root_path = Path(root).resolve()
        if source_manifest_or_none(root_path, DATASET_ID) is None:
            return SceneIndex(DATASET_ID, split, [])
        if split not in SPLIT_FILES:
            raise ConfigError(f"unknown GraspClutter6D split: {split}")
        values = _load_json(root_path / "split_info" / SPLIT_FILES[split])
        available = {path.name for path in (root_path / "scenes").iterdir() if path.is_dir()}
        scenes = sorted(f"{int(value):06d}" for value in values if f"{int(value):06d}" in available)
        return SceneIndex(DATASET_ID, split, scenes[:limit] if limit is not None else scenes)

    def load_scene(self, root: str, scene_key: str) -> SceneSpec:
        root_path = Path(root).resolve()
        manifest = source_manifest(root_path, DATASET_ID)
        paths = _scene_paths(root_path, scene_key)
        cameras = _normalize_frame_records(_load_json(paths["camera"]))
        ground_truth = _normalize_frame_records(_load_json(paths["ground_truth"]))
        common_frames = sorted(set(cameras) & set(ground_truth), key=int)
        if not common_frames:
            raise ConfigError(f"scene has no common camera/ground-truth frames: {scene_key}")
        first_frame = common_frames[0]
        _, world_camera = _frame_transform(cameras[first_frame], first_frame)
        counts: dict[int, int] = {}
        objects = []
        for instance_index, instance in enumerate(ground_truth[first_frame]):
            try:
                object_id = int(instance["obj_id"])
            except (KeyError, ValueError) as exc:
                raise ConfigError(f"invalid object ID in scene {scene_key}") from exc
            duplicate = counts.get(object_id, 0)
            counts[object_id] = duplicate + 1
            base_id = f"obj_{object_id:06d}"
            canonical_id = base_id if duplicate == 0 else f"{base_id}_inst_{duplicate:03d}"
            asset = root_path / "models_obj_m" / f"{base_id}.obj"
            if not asset.is_file():
                raise ConfigError(f"GraspClutter6D object asset not found: {asset}")
            objects.append(
                SceneObjectSpec(
                    object_id=canonical_id,
                    asset_ref=str(asset.resolve()),
                    T_world_object=_object_pose(instance, world_camera, f"{scene_key}:{first_frame}:{instance_index}"),
                )
            )
        camera_specs = []
        for frame_id in common_frames:
            frame_intrinsics, frame_transform = _frame_transform(cameras[frame_id], frame_id)
            camera_specs.append(
                CameraSpec(
                    camera_id=f"frame_{int(frame_id):06d}",
                    intrinsics=frame_intrinsics,
                    T_world_camera=frame_transform,
                )
            )
        evidence_files = [paths["source_manifest"], paths["camera"], paths["ground_truth"]]
        evidence_files.extend(Path(item.asset_ref) for item in objects)
        return SceneSpec(
            scene_id=scene_key,
            source_dataset=DATASET_ID,
            source_version=manifest["version"],
            source_split=_split_for_scene(root_path, scene_key),
            environment=str(manifest.get("environment", "custom")),
            objects=objects,
            supports=[],
            cameras=camera_specs,
            source_record_hash=record_hash(root_path, evidence_files),
            license_record=manifest["license"],
            redistributable=False,
        )

    def load_observation(self, root: str, scene_key: str, camera_key: str, frame_key: str) -> SceneObservation:
        root_path = Path(root).resolve()
        source_manifest(root_path, DATASET_ID)
        try:
            frame_id = f"{int(frame_key):06d}"
        except ValueError as exc:
            raise ConfigError(f"frame key must be numeric: {frame_key}") from exc
        scene_dir = root_path / "scenes" / scene_key
        camera_records = _normalize_frame_records(_load_json(scene_dir / "scene_camera.json"))
        ground_truth = _normalize_frame_records(_load_json(scene_dir / "scene_gt.json"))
        if frame_id not in camera_records or frame_id not in ground_truth:
            raise ConfigError(f"frame {frame_id} is absent from scene {scene_key}")
        expected_camera_key = f"frame_{frame_id}"
        if camera_key != expected_camera_key:
            raise ConfigError(f"camera/frame mismatch: expected camera_key={expected_camera_key}, got {camera_key}")
        intrinsics, world_camera = _frame_transform(camera_records[frame_id], frame_id)
        refs = {
            "rgb": scene_dir / "rgb" / f"{frame_id}.png",
            "depth": scene_dir / "depth" / f"{frame_id}.png",
            "label": scene_dir / "label" / f"{frame_id}.png",
        }
        missing = [path for path in refs.values() if not path.is_file()]
        if missing:
            raise ConfigError(f"observation source files missing: {[str(path) for path in missing]}")
        visibility: dict[str, float] = {}
        try:
            from PIL import Image
        except ImportError as exc:
            raise ConfigError("Pillow is required to load source masks") from exc
        for instance_index, instance in enumerate(ground_truth[frame_id]):
            mask_path = scene_dir / "visible_mask" / f"{frame_id}_{instance_index:06d}.png"
            if not mask_path.is_file():
                raise ConfigError(f"visible instance mask not found: {mask_path}")
            mask = np.asarray(Image.open(mask_path))
            if mask.ndim != 2 or mask.size == 0:
                raise ConfigError(f"invalid visible instance mask: {mask_path}")
            object_id = int(instance["obj_id"])
            duplicate = sum(
                int(previous["obj_id"]) == object_id for previous in ground_truth[frame_id][:instance_index]
            )
            base_id = f"obj_{object_id:06d}"
            canonical_id = base_id if duplicate == 0 else f"{base_id}_inst_{duplicate:03d}"
            visibility[canonical_id] = float(np.count_nonzero(mask) / mask.size)
        return SceneObservation(
            scene_id=scene_key,
            camera_id=camera_key,
            frame_id=frame_id,
            timestamp=float(int(frame_id)),
            T_world_camera=world_camera,
            calibration_hash=calibration_hash(intrinsics, world_camera),
            rgb_ref=str(refs["rgb"].resolve()),
            depth_ref=str(refs["depth"].resolve()),
            instance_mask_ref=str(refs["label"].resolve()),
            visibility_by_object=visibility,
        )

    def load_external_grasps(self, root: str, scene_key: str) -> ExternalGraspSet:
        root_path = Path(root).resolve()
        source_manifest(root_path, DATASET_ID)
        scene = self.load_scene(root, scene_key)
        records = []
        for object_id in sorted({item.object_id.split("_inst_")[0] for item in scene.objects}):
            path = root_path / "grasp_label" / f"{object_id}_labels.npz"
            if not path.is_file():
                raise ConfigError(f"external grasp label not found: {path}")
            records.append(
                {
                    "target_object_id": object_id,
                    "label_stage": "external_label",
                    "source_hand": "parallel_jaw",
                    "source_file": str(path.resolve()),
                    "source_sha256": sha256_file(path),
                }
            )
        return ExternalGraspSet(
            scene_id=scene_key,
            gripper_type="parallel_jaw",
            grasps=records,
            source_provenance=f"{DATASET_ID}:grasp_label",
        )

    def audit(self, root: str, scene_key: str) -> SourceEvidence:
        root_path = Path(root).resolve()
        try:
            paths = _scene_paths(root_path, scene_key)
            scene = self.load_scene(root, scene_key)
            frame_key = scene.cameras[0].camera_id.removeprefix("frame_")
            self.load_observation(root, scene_key, scene.cameras[0].camera_id, frame_key)
            grasps = self.load_external_grasps(root, scene_key)
            evidence_files = list(paths.values())
            evidence_files.extend(Path(item.asset_ref) for item in scene.objects)
            evidence_files.extend(Path(item["source_file"]) for item in grasps.grasps)
            missing = sorted(str(path) for path in evidence_files if not path.is_file())
            if missing:
                return SourceEvidence(scene_key, "", False, missing)
            return SourceEvidence(scene_key, record_hash(root_path, evidence_files), True, [])
        except (ConfigError, OSError, ValueError) as exc:
            return SourceEvidence(scene_key, "", False, [str(exc)])
