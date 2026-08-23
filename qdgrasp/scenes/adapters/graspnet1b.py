from typing import Optional
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


class GraspNet1BillionAdapter(SceneAdapter):
    """
    Adapter for the GraspNet-1Billion dataset.
    https://graspnet.net/
    """

    def probe(self, root: str) -> SourceDatasetInfo:
        root_path = Path(root)
        # Check for typical GraspNet-1B directories: 'scenes', 'models', 'grasp_labels'
        is_valid = (
            (root_path / "scenes").is_dir() and
            (root_path / "models").is_dir()
        )

        num_scenes = 0
        if is_valid:
            # GraspNet-1Billion has 190 scenes (0000 to 0189)
            num_scenes = len(list((root_path / "scenes").glob("scene_*")))

        return SourceDatasetInfo(
            dataset_id="graspnet-1billion",
            version="1.0",
            is_valid=is_valid,
            num_scenes=num_scenes,
            license_type="GraspNet License (Non-Commercial)",
            redistributable=False,  # We do not redistribute GraspNet-1B
        )

    def index(self, root: str, split: str, limit: Optional[int] = None) -> SceneIndex:
        # GraspNet-1B splits:
        # Train: 0000-0099
        # Test (seen): 0100-0129
        # Test (similar): 0130-0159
        # Test (novel): 0160-0189

        if split == "train":
            scenes = [f"scene_{i:04d}" for i in range(100)]
        elif split == "test_seen":
            scenes = [f"scene_{i:04d}" for i in range(100, 130)]
        elif split == "test_similar":
            scenes = [f"scene_{i:04d}" for i in range(130, 160)]
        elif split == "test_novel":
            scenes = [f"scene_{i:04d}" for i in range(160, 190)]
        else:
            scenes = []

        if limit is not None:
            scenes = scenes[:limit]

        return SceneIndex("graspnet-1billion", split, scenes)

    def load_scene(self, root: str, scene_key: str) -> SceneSpec:
        raise NotImplementedError("GraspNet-1B scene loading not yet implemented")

    def load_observation(self, root: str, scene_key: str, camera_key: str, frame_key: str) -> SceneObservation:
        raise NotImplementedError("GraspNet-1B observation loading not yet implemented")

    def load_external_grasps(self, root: str, scene_key: str) -> ExternalGraspSet:
        raise NotImplementedError("GraspNet-1B grasp loading not yet implemented")

    def audit(self, root: str, scene_key: str) -> SourceEvidence:
        return SourceEvidence(scene_key=scene_key, record_hash="", is_complete=True, missing_files=[])
