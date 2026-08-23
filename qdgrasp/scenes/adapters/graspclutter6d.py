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


class GraspClutter6DAdapter(SceneAdapter):
    """
    Adapter for GraspClutter6D dataset.
    https://sites.google.com/view/graspclutter6d/dataset
    """

    def probe(self, root: str) -> SourceDatasetInfo:
        root_path = Path(root)
        is_valid = (
            (root_path / "scenes").is_dir()
        )

        num_scenes = 0
        if is_valid:
            num_scenes = len(list((root_path / "scenes").glob("scene_*")))

        return SourceDatasetInfo(
            dataset_id="graspclutter6d",
            version="1.0",
            is_valid=is_valid,
            num_scenes=num_scenes,
            license_type="Non-Commercial",
            redistributable=False,
        )

    def index(self, root: str, split: str, limit: Optional[int] = None) -> SceneIndex:
        scenes = []
        if limit is not None:
            scenes = scenes[:limit]

        return SceneIndex("graspclutter6d", split, scenes)

    def load_scene(self, root: str, scene_key: str) -> SceneSpec:
        raise NotImplementedError("GraspClutter6D scene loading not yet implemented")

    def load_observation(self, root: str, scene_key: str, camera_key: str, frame_key: str) -> SceneObservation:
        raise NotImplementedError("GraspClutter6D observation loading not yet implemented")

    def load_external_grasps(self, root: str, scene_key: str) -> ExternalGraspSet:
        raise NotImplementedError("GraspClutter6D grasp loading not yet implemented")

    def audit(self, root: str, scene_key: str) -> SourceEvidence:
        return SourceEvidence(scene_key=scene_key, record_hash="", is_complete=True, missing_files=[])
