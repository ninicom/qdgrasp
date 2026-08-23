from typing import Optional
from pathlib import Path

from qdgrasp.scenes.contracts import (
    SceneAdapter,
    SceneSpec,
    SceneObservation,
    SourceDatasetInfo,
    SceneIndex,
    SourceEvidence,
    ExternalGraspSet,
)


class DexGraspNet2Adapter(SceneAdapter):
    """
    Adapter for DexGraspNet 2.0 (synthetic cluttered scenes).
    https://github.com/PKU-EPIC/DexGraspNet2
    """

    def probe(self, root: str) -> SourceDatasetInfo:
        root_path = Path(root)
        is_valid = (
            (root_path / "scenes").is_dir() and
            (root_path / "grasps").is_dir()
        )

        num_scenes = 0
        if is_valid:
            # DGN2 has ~8270 scenes
            num_scenes = len(list((root_path / "scenes").glob("*.json")))

        return SourceDatasetInfo(
            dataset_id="dexgraspnet2",
            version="1.0",
            is_valid=is_valid,
            num_scenes=num_scenes,
            license_type="CC BY-NC 4.0",
            redistributable=False,
        )

    def index(self, root: str, split: str, limit: Optional[int] = None) -> SceneIndex:
        # Typically splits are provided in a train/val text file.
        # For this skeleton, we just mock it.
        scenes = []
        if limit is not None:
            scenes = scenes[:limit]

        return SceneIndex("dexgraspnet2", split, scenes)

    def load_scene(self, root: str, scene_key: str) -> SceneSpec:
        raise NotImplementedError("DexGraspNet2 scene loading not yet implemented")

    def load_observation(self, root: str, scene_key: str, camera_key: str, frame_key: str) -> SceneObservation:
        raise NotImplementedError("DexGraspNet2 observation loading not yet implemented")

    def load_external_grasps(self, root: str, scene_key: str) -> ExternalGraspSet:
        raise NotImplementedError("DexGraspNet2 grasp loading not yet implemented")

    def audit(self, root: str, scene_key: str) -> SourceEvidence:
        return SourceEvidence(scene_key=scene_key, record_hash="", is_complete=True, missing_files=[])
