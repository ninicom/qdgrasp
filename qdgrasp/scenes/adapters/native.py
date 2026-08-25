"""Adapter for verified native QDGrasp scene releases."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from qdgrasp.config.schema import ConfigError
from qdgrasp.dataset.scene_manifest import SceneDatasetManifest, load_scene_manifest
from qdgrasp.dataset.scene_shards import read_scene_shard
from qdgrasp.scenes.adapters._common import calibration_hash, record_hash, require_transform
from qdgrasp.scenes.contracts import (
    CameraSpec,
    ExternalGraspSet,
    SceneIndex,
    SceneObjectSpec,
    SceneObservation,
    SceneSpec,
    SourceDatasetInfo,
    SourceEvidence,
    SupportGeometrySpec,
)

DATASET_ID = "qdgrasp-native"


def _manifest(root: Path) -> SceneDatasetManifest:
    manifest = load_scene_manifest(root / "scene_manifest.json")
    if manifest.invalidated:
        raise ConfigError(f"native scene dataset is invalidated: {manifest.invalidation_reason}")
    return manifest


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _resolve_ref(root: Path, value: str, label: str) -> str:
    path = Path(value)
    resolved = (root / path).resolve() if not path.is_absolute() else path.resolve()
    if not resolved.is_relative_to(root) or not resolved.is_file():
        raise ConfigError(f"native {label} is missing or outside dataset root: {value}")
    return str(resolved)


def _records(
    root: Path,
    manifest: SceneDatasetManifest,
    *,
    scene_id: str,
    record_type: str,
) -> list[dict[str, Any]]:
    split = next((name for name, scenes in manifest.splits.items() if scene_id in scenes), None)
    if split is None:
        raise ConfigError(f"native scene is absent from manifest splits: {scene_id}")
    result: list[dict[str, Any]] = []
    for shard in manifest.shards:
        if shard.split != split or shard.record_type != record_type:
            continue
        records = read_scene_shard(
            root / shard.filename,
            record_type=shard.record_type,
            expected_sha256=shard.sha256,
            expected_records=shard.num_records,
        )
        result.extend(record for record in records if record["scene_id"] == scene_id)
    return result


def _parse_scene_spec(root: Path, payload: dict[str, Any]) -> SceneSpec:
    try:
        objects = [
            SceneObjectSpec(
                object_id=item["object_id"],
                asset_ref=_resolve_ref(root, item["asset_ref"], "object asset"),
                T_world_object=require_transform(item["T_world_object"], f"object {item['object_id']}"),
                scale=float(item.get("scale", 1.0)),
                mass=float(item["mass"]) if item.get("mass") is not None else None,
                friction=tuple(item["friction"]) if item.get("friction") is not None else None,
            )
            for item in payload["objects"]
        ]
        supports = [
            SupportGeometrySpec(
                support_id=item["support_id"],
                geom_type=item["geom_type"],
                params=item["params"],
                T_world_support=require_transform(item["T_world_support"], f"support {item['support_id']}"),
            )
            for item in payload.get("supports", [])
        ]
        cameras = [
            CameraSpec(
                camera_id=item["camera_id"],
                intrinsics=np.asarray(item["intrinsics"], dtype=np.float64),
                distortion=(
                    np.asarray(item["distortion"], dtype=np.float64) if item.get("distortion") is not None else None
                ),
                T_world_camera=require_transform(item["T_world_camera"], f"camera {item['camera_id']}"),
            )
            for item in payload.get("cameras", [])
        ]
        return SceneSpec(
            scene_id=payload["scene_id"],
            source_dataset=payload["source_dataset"],
            source_version=payload["source_version"],
            source_split=payload["source_split"],
            environment=payload["environment"],
            objects=objects,
            supports=supports,
            cameras=cameras,
            gravity=tuple(payload.get("gravity", [0.0, 0.0, -9.81])),
            timestep=float(payload.get("timestep", 0.002)),
            solver_profile=payload.get("solver_profile", "default"),
            settle_seed=int(payload.get("settle_seed", 0)),
            source_record_hash=payload.get("source_record_hash"),
            license_record=payload.get("license_record", "CC0-1.0"),
            redistributable=bool(payload.get("redistributable", True)),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ConfigError(f"invalid native SceneSpec payload: {exc}") from exc


class NativeAdapter:
    def probe(self, root: str) -> SourceDatasetInfo:
        root_path = Path(root).resolve()
        try:
            manifest = _manifest(root_path)
        except ConfigError:
            return SourceDatasetInfo(DATASET_ID, "unknown", False, 0, "unknown", False)
        scene_count = len({scene for scenes in manifest.splits.values() for scene in scenes})
        return SourceDatasetInfo(
            DATASET_ID,
            manifest.generator_version,
            True,
            scene_count,
            ";".join(sorted(set(manifest.source_licenses.values()))),
            True,
        )

    def index(self, root: str, split: str, limit: int | None = None) -> SceneIndex:
        manifest = _manifest(Path(root).resolve())
        if split not in manifest.splits:
            raise ConfigError(f"unknown native scene split: {split}")
        scenes = list(manifest.splits[split])
        return SceneIndex(DATASET_ID, split, scenes[:limit] if limit is not None else scenes)

    def load_scene(self, root: str, scene_key: str) -> SceneSpec:
        root_path = Path(root).resolve()
        manifest = _manifest(root_path)
        records = _records(root_path, manifest, scene_id=scene_key, record_type="scene_state")
        initial = next(
            (record for record in records if record.get("stage") in {"initial", "settled"}),
            None,
        )
        if initial is None or not isinstance(initial.get("scene_spec"), dict):
            raise ConfigError(f"native scene lacks initial/settled SceneSpec record: {scene_key}")
        payload = initial["scene_spec"]
        expected_hash = manifest.scene_spec_hashes.get(scene_key)
        if expected_hash != _canonical_hash(payload):
            raise ConfigError(f"native SceneSpec hash mismatch for {scene_key}")
        scene = _parse_scene_spec(root_path, payload)
        if scene.scene_id != scene_key:
            raise ConfigError(f"native SceneSpec scene ID mismatch: {scene.scene_id} != {scene_key}")
        return scene

    def load_observation(self, root: str, scene_key: str, camera_key: str, frame_key: str) -> SceneObservation:
        root_path = Path(root).resolve()
        manifest = _manifest(root_path)
        records = _records(root_path, manifest, scene_id=scene_key, record_type="observation")
        record = next(
            (
                item
                for item in records
                if item.get("camera_id") == camera_key and str(item.get("frame_id")) == str(frame_key)
            ),
            None,
        )
        if record is None:
            raise ConfigError(
                f"native observation not found: scene={scene_key}, camera={camera_key}, frame={frame_key}"
            )
        scene = self.load_scene(root, scene_key)
        camera = next((item for item in scene.cameras if item.camera_id == camera_key), None)
        if camera is None:
            raise ConfigError(f"native observation camera absent from SceneSpec: {camera_key}")
        transform = require_transform(record["T_world_camera"], f"observation {camera_key}")
        expected_calibration = calibration_hash(camera.intrinsics, transform)
        if record["calibration_hash"] != expected_calibration:
            raise ConfigError(f"native observation calibration hash mismatch: {camera_key}/{frame_key}")

        def optional_ref(name: str) -> str | None:
            return _resolve_ref(root_path, record[name], name) if record.get(name) else None

        return SceneObservation(
            scene_id=scene_key,
            camera_id=camera_key,
            frame_id=str(frame_key),
            timestamp=float(record["timestamp"]),
            T_world_camera=transform,
            calibration_hash=record["calibration_hash"],
            rgb_ref=optional_ref("rgb_ref"),
            depth_ref=optional_ref("depth_ref"),
            point_cloud_ref=optional_ref("point_cloud_ref"),
            instance_mask_ref=optional_ref("instance_mask_ref"),
            normal_ref=optional_ref("normal_ref"),
            visibility_by_object={
                str(key): float(value) for key, value in record.get("visibility_by_object", {}).items()
            },
        )

    def load_external_grasps(self, root: str, scene_key: str) -> ExternalGraspSet:
        self.load_scene(root, scene_key)
        return ExternalGraspSet(
            scene_id=scene_key,
            gripper_type="none",
            grasps=[],
            source_provenance="qdgrasp-native:no-external-labels",
        )

    def audit(self, root: str, scene_key: str) -> SourceEvidence:
        root_path = Path(root).resolve()
        try:
            manifest = _manifest(root_path)
            scene = self.load_scene(root, scene_key)
            record_types = set()
            files = [root_path / "scene_manifest.json"]
            files.extend(Path(item.asset_ref) for item in scene.objects)
            split = next(name for name, scenes in manifest.splits.items() if scene_key in scenes)
            for shard in manifest.shards:
                if shard.split != split:
                    continue
                records = _records(root_path, manifest, scene_id=scene_key, record_type=shard.record_type)
                if records:
                    record_types.add(shard.record_type)
                    files.append(root_path / shard.filename)
                if shard.record_type == "observation":
                    for record in records:
                        observation = self.load_observation(
                            root,
                            scene_key,
                            str(record["camera_id"]),
                            str(record["frame_id"]),
                        )
                        for reference in (
                            observation.rgb_ref,
                            observation.depth_ref,
                            observation.point_cloud_ref,
                            observation.instance_mask_ref,
                            observation.normal_ref,
                        ):
                            if reference is not None:
                                files.append(Path(reference))
            missing_types = sorted({"scene_state", "observation", "grasp"} - record_types)
            if missing_types:
                return SourceEvidence(scene_key, "", False, [f"missing record types: {missing_types}"])
            return SourceEvidence(scene_key, record_hash(root_path, files), True, [])
        except (ConfigError, OSError, ValueError, StopIteration) as exc:
            return SourceEvidence(scene_key, "", False, [str(exc)])
