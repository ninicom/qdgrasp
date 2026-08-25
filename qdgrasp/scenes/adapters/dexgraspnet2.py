"""Adapter for the official DexGraspNet 2.0 scene and LEAP-label layout."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from qdgrasp.config.schema import ConfigError
from qdgrasp.scenes.adapters._common import (
    graspnet_frame_files,
    load_graspnet_observation,
    load_graspnet_scene,
    record_hash,
    select_graspnet_camera,
    sha256_file,
    source_manifest,
    source_manifest_or_none,
)
from qdgrasp.scenes.contracts import (
    ExternalGraspSet,
    SceneIndex,
    SceneObservation,
    SceneSpec,
    SourceDatasetInfo,
    SourceEvidence,
)

DATASET_ID = "dexgraspnet2"


def _default_split(scene_key: str) -> str:
    try:
        index = int(scene_key.rsplit("_", 1)[-1])
    except ValueError:
        return "external_test"
    if 0 <= index < 90:
        return "train"
    if 90 <= index < 100:
        return "val"
    if 100 <= index < 190:
        return "test"
    if 1000 <= index < 8500:
        return "synthetic_train"
    return "custom"


class DexGraspNet2Adapter:
    def probe(self, root: str) -> SourceDatasetInfo:
        root_path = Path(root).resolve()
        manifest = source_manifest_or_none(root_path, DATASET_ID)
        is_valid = bool(
            manifest
            and (root_path / "scenes").is_dir()
            and (root_path / "meshdata").is_dir()
            and (root_path / "dex_grasps_new").is_dir()
        )
        num_scenes = sum(1 for path in (root_path / "scenes").glob("scene_*") if path.is_dir()) if is_valid else 0
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
        manifest = source_manifest_or_none(root_path, DATASET_ID)
        if manifest is None:
            return SceneIndex(DATASET_ID, split, [])
        available = sorted(path.name for path in (root_path / "scenes").glob("scene_*") if path.is_dir())
        declared_splits = manifest.get("splits")
        if declared_splits is not None:
            if not isinstance(declared_splits, dict) or split not in declared_splits:
                raise ConfigError(f"unknown DexGraspNet2 split: {split}")
            declared = set(declared_splits[split])
            scenes = [scene for scene in available if scene in declared]
        else:
            allowed = {"train", "val", "test", "synthetic_train", "custom", "external_test"}
            if split not in allowed:
                raise ConfigError(f"unknown DexGraspNet2 split: {split}")
            scenes = [scene for scene in available if _default_split(scene) == split]
        return SceneIndex(DATASET_ID, split, scenes[:limit] if limit is not None else scenes)

    def load_scene(self, root: str, scene_key: str) -> SceneSpec:
        return load_graspnet_scene(
            Path(root).resolve(),
            scene_key,
            dataset_id=DATASET_ID,
            source_split=_default_split(scene_key),
            model_root="meshdata",
            model_filename="simplified.obj",
        )

    def load_observation(self, root: str, scene_key: str, camera_key: str, frame_key: str) -> SceneObservation:
        return load_graspnet_observation(
            Path(root).resolve(),
            scene_key,
            camera_key,
            frame_key,
            dataset_id=DATASET_ID,
        )

    def load_external_grasps(self, root: str, scene_key: str) -> ExternalGraspSet:
        root_path = Path(root).resolve()
        source_manifest(root_path, DATASET_ID)
        grasp_root = root_path / "dex_grasps_new" / scene_key
        if not grasp_root.is_dir():
            raise ConfigError(f"DexGraspNet2 scene grasp directory not found: {grasp_root}")
        records = []
        hands: set[str] = set()
        for path in sorted(grasp_root.glob("*/*.npz")):
            hand = path.parent.name
            hands.add(hand)
            try:
                with np.load(path, allow_pickle=False) as payload:
                    fields = sorted(payload.files)
                    count = len(payload["point"]) if "point" in payload else 0
            except Exception as exc:
                raise ConfigError(f"invalid DexGraspNet2 grasp file {path}: {exc}") from exc
            if not {"point", "translation", "rotation"}.issubset(fields) or count <= 0:
                raise ConfigError(f"DexGraspNet2 grasp file lacks required arrays: {path}")
            records.append(
                {
                    "target_object_id": path.stem,
                    "label_stage": "external_label",
                    "source_hand": hand,
                    "source_validation": "upstream_simulation_unreplayed",
                    "source_file": str(path.resolve()),
                    "source_sha256": sha256_file(path),
                    "num_grasps": count,
                    "fields": fields,
                }
            )
        if not records:
            raise ConfigError(f"DexGraspNet2 scene contains no external grasp labels: {grasp_root}")
        return ExternalGraspSet(
            scene_id=scene_key,
            gripper_type="+".join(sorted(hands)),
            grasps=records,
            source_provenance=f"{DATASET_ID}:dex_grasps_new",
        )

    def audit(self, root: str, scene_key: str) -> SourceEvidence:
        root_path = Path(root).resolve()
        try:
            scene_dir = root_path / "scenes" / scene_key
            camera = select_graspnet_camera(scene_dir)
            files = graspnet_frame_files(root_path, scene_key, camera, 0)
            scene = self.load_scene(root, scene_key)
            self.load_observation(root, scene_key, camera, "0")
            grasps = self.load_external_grasps(root, scene_key)
            evidence_files = [*files.values(), *(Path(item.asset_ref) for item in scene.objects)]
            evidence_files.extend(Path(item["source_file"]) for item in grasps.grasps)
            missing = sorted(str(path) for path in evidence_files if not path.is_file())
            if missing:
                return SourceEvidence(scene_key, "", False, missing)
            return SourceEvidence(
                scene_key,
                record_hash(root_path, evidence_files),
                True,
                [],
            )
        except (ConfigError, OSError, ValueError) as exc:
            return SourceEvidence(scene_key, "", False, [str(exc)])
