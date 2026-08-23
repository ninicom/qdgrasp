from typing import Optional
import json
from pathlib import Path
import numpy as np

from qdgrasp.scenes.contracts import (
    SceneAdapter,
    SceneSpec,
    SceneObservation,
    SourceDatasetInfo,
    SceneIndex,
    SourceEvidence,
    ExternalGraspSet,
)


class NativeAdapter(SceneAdapter):
    """
    Adapter for natively generated QDGrasp scenes (e.g., procedural clutter).
    This adapter doesn't read from an external third-party dataset, but rather
    reads QDGrasp's own canonical scene shards.
    """

    def probe(self, root: str) -> SourceDatasetInfo:
        # Native datasets are expected to have a dataset_manifest.json
        # or scene_manifest.json in the root.
        manifest_path = Path(root) / "scene_manifest.json"
        is_valid = manifest_path.exists()
        num_scenes = 0
        version = "unknown"
        if is_valid:
            try:
                with open(manifest_path, "r") as f:
                    data = json.load(f)
                    num_scenes = data.get("num_scenes", 0)
                    version = data.get("version", "1.0")
            except Exception:
                is_valid = False

        return SourceDatasetInfo(
            dataset_id="qdgrasp-native",
            version=version,
            is_valid=is_valid,
            num_scenes=num_scenes,
            license_type="CC0-1.0",  # Default native license
            redistributable=True,
        )

    def index(self, root: str, split: str, limit: Optional[int] = None) -> SceneIndex:
        manifest_path = Path(root) / "scene_manifest.json"
        if not manifest_path.exists():
            return SceneIndex("qdgrasp-native", split, [])

        with open(manifest_path, "r") as f:
            data = json.load(f)

        scenes = data.get("scenes", {}).get(split, [])
        if limit is not None:
            scenes = scenes[:limit]

        return SceneIndex("qdgrasp-native", split, scenes)

    def load_scene(self, root: str, scene_key: str) -> SceneSpec:
        scene_path = Path(root) / "scenes" / f"{scene_key}.json"
        if not scene_path.exists():
            raise FileNotFoundError(f"Native scene {scene_key} not found at {scene_path}")

        # TODO: Parse full SceneSpec from native JSON format
        # For now, return a mock or empty
        raise NotImplementedError("Full native scene parsing not yet implemented")

    def load_observation(self, root: str, scene_key: str, camera_key: str, frame_key: str) -> SceneObservation:
        raise NotImplementedError("Native observation loading not yet implemented")

    def load_external_grasps(self, root: str, scene_key: str) -> ExternalGraspSet:
        # Native scenes don't have "external" grasps, they have QDGrasp grasps.
        return ExternalGraspSet(scene_id=scene_key, gripper_type="unknown", grasps=[], source_provenance="native")

    def audit(self, root: str, scene_key: str) -> SourceEvidence:
        return SourceEvidence(scene_key=scene_key, record_hash="", is_complete=True, missing_files=[])
