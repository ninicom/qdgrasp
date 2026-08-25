"""Adapter for the official GraspNet-1Billion on-disk layout."""

from __future__ import annotations

from pathlib import Path

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
    split_scene_directories,
)
from qdgrasp.scenes.contracts import (
    ExternalGraspSet,
    SceneIndex,
    SceneObservation,
    SceneSpec,
    SourceDatasetInfo,
    SourceEvidence,
)

DATASET_ID = "graspnet-1billion"
SPLITS = {
    "train": range(100),
    "test_seen": range(100, 130),
    "test_similar": range(130, 160),
    "test_novel": range(160, 190),
}


class GraspNet1BillionAdapter:
    def probe(self, root: str) -> SourceDatasetInfo:
        root_path = Path(root).resolve()
        manifest = source_manifest_or_none(root_path, DATASET_ID)
        is_valid = bool(
            manifest
            and (root_path / "scenes").is_dir()
            and (root_path / "models").is_dir()
            and (root_path / "grasp_label").is_dir()
        )
        num_scenes = (
            sum(1 for path in (root_path / "scenes").glob("scene_[0-9][0-9][0-9][0-9]") if path.is_dir())
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
        scenes = split_scene_directories(root_path, split, SPLITS)
        return SceneIndex(DATASET_ID, split, scenes[:limit] if limit is not None else scenes)

    def load_scene(self, root: str, scene_key: str) -> SceneSpec:
        split = next((name for name, values in SPLITS.items() if int(scene_key[-4:]) in values), "custom")
        return load_graspnet_scene(
            Path(root).resolve(),
            scene_key,
            dataset_id=DATASET_ID,
            source_split=split,
            model_root="models",
            model_filename="nontextured.ply",
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
                    "source_file": str(path),
                    "source_sha256": sha256_file(path),
                }
            )
        return ExternalGraspSet(
            scene_id=scene_key,
            gripper_type="parallel_jaw",
            grasps=records,
            source_provenance=f"{DATASET_ID}:object-grasp-labels",
        )

    def audit(self, root: str, scene_key: str) -> SourceEvidence:
        root_path = Path(root).resolve()
        try:
            scene_dir = root_path / "scenes" / scene_key
            camera = select_graspnet_camera(scene_dir)
            files = graspnet_frame_files(root_path, scene_key, camera, 0)
            scene = self.load_scene(root, scene_key)
            self.load_observation(root, scene_key, camera, "0")
            evidence_files = list(files.values())
            for item in scene.objects:
                evidence_files.append(Path(item.asset_ref))
                evidence_files.append(root_path / "grasp_label" / f"{item.object_id.split('_inst_')[0]}_labels.npz")
            missing = sorted(str(path) for path in evidence_files if not path.is_file())
            if missing:
                return SourceEvidence(scene_key, "", False, missing)
            return SourceEvidence(
                scene_key=scene_key,
                record_hash=record_hash(root_path, evidence_files),
                is_complete=True,
                missing_files=[],
            )
        except (ConfigError, OSError, ValueError) as exc:
            return SourceEvidence(scene_key, "", False, [str(exc)])
