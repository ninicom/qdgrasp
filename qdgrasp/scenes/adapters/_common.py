"""Shared fail-closed utilities for source scene adapters."""

from __future__ import annotations

import hashlib
import json
import xml.etree.ElementTree as ET
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from qdgrasp.config.schema import ConfigError
from qdgrasp.scenes.contracts import (
    CameraSpec,
    SceneObjectSpec,
    SceneObservation,
    SceneSpec,
    SupportGeometrySpec,
)


def source_manifest(root: Path, expected_dataset_id: str) -> dict[str, Any]:
    path = root / "source_manifest.json"
    if not path.is_file():
        raise ConfigError(f"source manifest not found: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ConfigError(f"invalid source manifest {path}: {exc}") from exc
    required = {"dataset_id", "version", "license", "source_url"}
    missing = sorted(required - set(payload))
    if missing:
        raise ConfigError(f"source manifest missing fields: {missing}")
    if payload["dataset_id"] != expected_dataset_id:
        raise ConfigError(
            f"source manifest dataset mismatch: expected {expected_dataset_id}, got {payload['dataset_id']}"
        )
    if not all(isinstance(payload[name], str) and payload[name] for name in required):
        raise ConfigError("source manifest identity fields must be non-empty strings")
    return payload


def source_manifest_or_none(root: Path, expected_dataset_id: str) -> dict[str, Any] | None:
    try:
        return source_manifest(root, expected_dataset_id)
    except ConfigError:
        return None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def record_hash(root: Path, paths: Iterable[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted({item.resolve() for item in paths}, key=lambda item: item.as_posix()):
        if not path.is_file() or not path.is_relative_to(root.resolve()):
            raise ConfigError(f"source evidence file missing or outside root: {path}")
        relative = path.relative_to(root.resolve()).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest()


def require_transform(value: Any, label: str) -> np.ndarray:
    transform = np.asarray(value, dtype=np.float64)
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        raise ConfigError(f"{label} must be a finite 4x4 transform")
    rotation = transform[:3, :3]
    if not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], atol=1e-8):
        raise ConfigError(f"{label} has invalid homogeneous row")
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5) or not np.isclose(
        np.linalg.det(rotation), 1.0, atol=1e-5
    ):
        raise ConfigError(f"{label} has invalid rotation")
    return transform


def calibration_hash(intrinsics: np.ndarray, transform: np.ndarray) -> str:
    payload = {
        "T_world_camera": np.asarray(transform, dtype=np.float64).tolist(),
        "intrinsics": np.asarray(intrinsics, dtype=np.float64).tolist(),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def table_support() -> SupportGeometrySpec:
    transform = np.eye(4)
    transform[2, 3] = -0.025
    return SupportGeometrySpec(
        support_id="table_surface",
        geom_type="box",
        params={"size": [1.0, 1.0, 0.05], "friction": [1.0, 0.005, 0.0001]},
        T_world_support=transform,
    )


def _parse_graspnet_annotation(path: Path) -> list[tuple[int, np.ndarray]]:
    if not path.is_file():
        raise ConfigError(f"scene annotation not found: {path}")
    try:
        root = ET.parse(path).getroot()
    except Exception as exc:
        raise ConfigError(f"invalid scene annotation {path}: {exc}") from exc
    result: list[tuple[int, np.ndarray]] = []
    for instance in root.findall("obj"):
        try:
            object_id = int(instance.findtext("obj_id", ""))
            position = np.fromstring(instance.findtext("pos_in_world", ""), sep=" ")
            quaternion = np.fromstring(instance.findtext("ori_in_world", ""), sep=" ")
        except ValueError as exc:
            raise ConfigError(f"invalid object annotation in {path}: {exc}") from exc
        if (
            position.shape != (3,)
            or quaternion.shape != (4,)
            or not np.all(np.isfinite(np.concatenate([position, quaternion])))
        ):
            raise ConfigError(f"invalid object pose in {path} for object {object_id}")
        quaternion_norm = float(np.linalg.norm(quaternion))
        if quaternion_norm <= np.finfo(np.float64).eps:
            raise ConfigError(f"invalid zero-norm quaternion in {path} for object {object_id}")
        quaternion /= quaternion_norm
        transform = np.eye(4)
        transform[:3, :3] = Rotation.from_quat([quaternion[1], quaternion[2], quaternion[3], quaternion[0]]).as_matrix()
        transform[:3, 3] = position
        result.append((object_id, require_transform(transform, f"object {object_id} pose")))
    if not result:
        raise ConfigError(f"scene annotation contains no objects: {path}")
    return result


def select_graspnet_camera(scene_dir: Path, preferred: str | None = None) -> str:
    candidates = [preferred] if preferred else ["kinect", "realsense"]
    candidates += [path.name for path in sorted(scene_dir.iterdir()) if path.is_dir()]
    for candidate in candidates:
        if candidate and (scene_dir / candidate / "camera_poses.npy").is_file():
            return candidate
    raise ConfigError(f"scene contains no supported camera directory: {scene_dir}")


def graspnet_frame_files(root: Path, scene_key: str, camera: str, frame_id: int) -> dict[str, Path]:
    scene_dir = root / "scenes" / scene_key
    camera_dir = scene_dir / camera
    frame = f"{frame_id:04d}"
    return {
        "source_manifest": root / "source_manifest.json",
        "object_ids": scene_dir / "object_id_list.txt",
        "camera_poses": camera_dir / "camera_poses.npy",
        "alignment": camera_dir / "cam0_wrt_table.npy",
        "intrinsics": camera_dir / "camK.npy",
        "annotation": camera_dir / "annotations" / f"{frame}.xml",
        "rgb": camera_dir / "rgb" / f"{frame}.png",
        "depth": camera_dir / "depth" / f"{frame}.png",
        "label": camera_dir / "label" / f"{frame}.png",
    }


def load_graspnet_scene(
    root: Path,
    scene_key: str,
    *,
    dataset_id: str,
    source_split: str,
    model_root: str,
    model_filename: str,
) -> SceneSpec:
    manifest = source_manifest(root, dataset_id)
    scene_dir = root / "scenes" / scene_key
    if not scene_dir.is_dir():
        raise ConfigError(f"scene directory not found: {scene_dir}")
    camera = select_graspnet_camera(scene_dir)
    files = graspnet_frame_files(root, scene_key, camera, 0)
    required = [files[name] for name in ("object_ids", "camera_poses", "alignment", "intrinsics", "annotation")]
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise ConfigError(f"scene source files missing: {[str(path) for path in missing]}")
    camera_poses = np.load(files["camera_poses"], allow_pickle=False)
    if camera_poses.ndim != 3 or camera_poses.shape[1:] != (4, 4) or len(camera_poses) == 0:
        raise ConfigError(f"invalid camera_poses array: {files['camera_poses']}")
    alignment = require_transform(np.load(files["alignment"], allow_pickle=False), "cam0_wrt_table")
    world_camera = require_transform(alignment @ camera_poses[0], "T_world_camera")
    intrinsics = np.asarray(np.load(files["intrinsics"], allow_pickle=False), dtype=np.float64)
    if intrinsics.shape != (3, 3) or not np.all(np.isfinite(intrinsics)):
        raise ConfigError(f"invalid camera intrinsics: {files['intrinsics']}")
    annotations = _parse_graspnet_annotation(files["annotation"])
    try:
        declared_object_ids = {int(value) for value in files["object_ids"].read_text(encoding="utf-8").split()}
    except ValueError as exc:
        raise ConfigError(f"invalid object ID list: {files['object_ids']}") from exc
    annotated_object_ids = {object_id for object_id, _ in annotations}
    if annotated_object_ids != declared_object_ids:
        raise ConfigError(
            "scene annotation/object ID list mismatch: "
            f"annotation={sorted(annotated_object_ids)}, declared={sorted(declared_object_ids)}"
        )
    objects: list[SceneObjectSpec] = []
    counts: dict[int, int] = {}
    for object_id, camera_object in annotations:
        instance_index = counts.get(object_id, 0)
        counts[object_id] = instance_index + 1
        code = f"{object_id:03d}"
        canonical_id = code if instance_index == 0 else f"{code}_inst_{instance_index:03d}"
        asset = root / model_root / code / model_filename
        if not asset.is_file():
            raise ConfigError(f"object asset not found: {asset}")
        objects.append(
            SceneObjectSpec(
                object_id=canonical_id,
                asset_ref=str(asset.resolve()),
                T_world_object=require_transform(alignment @ camera_object, f"world pose {canonical_id}"),
            )
        )
    return SceneSpec(
        scene_id=scene_key,
        source_dataset=dataset_id,
        source_version=manifest["version"],
        source_split=source_split,
        environment="table",
        objects=objects,
        supports=[table_support()],
        cameras=[CameraSpec(camera_id=camera, intrinsics=intrinsics, T_world_camera=world_camera)],
        source_record_hash=record_hash(root, [*required, root / "source_manifest.json"]),
        license_record=manifest["license"],
        redistributable=False,
    )


def _mask_visibility(path: Path, object_prefix: str = "") -> dict[str, float]:
    if not path.is_file():
        raise ConfigError(f"instance/semantic label not found: {path}")
    try:
        from PIL import Image
    except ImportError as exc:
        raise ConfigError("Pillow is required to load source masks") from exc
    mask = np.asarray(Image.open(path))
    if mask.ndim != 2 or mask.size == 0:
        raise ConfigError(f"invalid source mask: {path}")
    labels, counts = np.unique(mask, return_counts=True)
    total = float(mask.size)
    return {
        f"{object_prefix}{int(label) - 1:03d}": float(count / total)
        for label, count in zip(labels, counts)
        if int(label) > 0
    }


def load_graspnet_observation(
    root: Path,
    scene_key: str,
    camera_key: str,
    frame_key: str,
    *,
    dataset_id: str,
) -> SceneObservation:
    source_manifest(root, dataset_id)
    try:
        frame_id = int(frame_key)
    except ValueError as exc:
        raise ConfigError(f"frame key must be numeric: {frame_key}") from exc
    scene_dir = root / "scenes" / scene_key
    camera = select_graspnet_camera(scene_dir, camera_key)
    files = graspnet_frame_files(root, scene_key, camera, frame_id)
    required = list(files.values())
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise ConfigError(f"observation source files missing: {[str(path) for path in missing]}")
    poses = np.load(files["camera_poses"], allow_pickle=False)
    if frame_id < 0 or frame_id >= len(poses):
        raise ConfigError(f"frame {frame_id} is outside camera pose range")
    alignment = require_transform(np.load(files["alignment"], allow_pickle=False), "cam0_wrt_table")
    world_camera = require_transform(alignment @ poses[frame_id], "T_world_camera")
    intrinsics = np.asarray(np.load(files["intrinsics"], allow_pickle=False), dtype=np.float64)
    if intrinsics.shape != (3, 3):
        raise ConfigError(f"invalid camera intrinsics: {files['intrinsics']}")
    return SceneObservation(
        scene_id=scene_key,
        camera_id=camera,
        frame_id=f"{frame_id:04d}",
        timestamp=float(frame_id),
        T_world_camera=world_camera,
        calibration_hash=calibration_hash(intrinsics, world_camera),
        rgb_ref=str(files["rgb"].resolve()),
        depth_ref=str(files["depth"].resolve()),
        instance_mask_ref=str(files["label"].resolve()),
        visibility_by_object=_mask_visibility(files["label"]),
    )


def split_scene_directories(
    root: Path, split: str, ranges: dict[str, range], *, explicit_prefix: str = "scene_"
) -> list[str]:
    if split not in ranges:
        raise ConfigError(f"unknown source split: {split}")
    available = {path.name for path in (root / "scenes").glob(f"{explicit_prefix}*") if path.is_dir()}
    return [f"{explicit_prefix}{index:04d}" for index in ranges[split] if f"{explicit_prefix}{index:04d}" in available]
