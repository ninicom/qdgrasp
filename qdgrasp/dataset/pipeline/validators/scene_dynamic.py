import numpy as np
from typing import Dict, Any, List

from qdgrasp.dataset.pipeline.contracts import DynamicValidation

class SceneDynamicValidator:
    def __init__(self, displacement_threshold: float = 0.05, rotation_threshold: float = 0.2):
        self.displacement_threshold = displacement_threshold
        self.rotation_threshold = rotation_threshold

    def validate(
        self,
        target_object_id: str,
        initial_scene_state: Dict[str, Dict[str, np.ndarray]],
        final_scene_state: Dict[str, Dict[str, np.ndarray]],
        target_lifted: bool
    ) -> DynamicValidation:
        """
        Validates the dynamic rollout in a multi-object scene.

        Args:
            target_object_id: ID of the object we intended to grasp.
            initial_scene_state: Mapping from object ID to its initial pose {"pos": [3], "quat": [4]}.
            final_scene_state: Mapping from object ID to its final pose after perturbation.
            target_lifted: Boolean from the base validator indicating if target survived lift/perturb.

        Returns:
            DynamicValidation result with scene-aware failure stages.
        """
        if not target_lifted:
            return DynamicValidation(
                trajectory_metrics={},
                per_finger_loads=np.zeros((1, 6)),
                failure_stage="target_not_lifted",
                passed=False
            )

        # Check non-target disturbances
        for obj_id, init_pose in initial_scene_state.items():
            if obj_id == target_object_id:
                continue

            if obj_id not in final_scene_state:
                continue # or raise error

            final_pose = final_scene_state[obj_id]

            # Position displacement
            disp = np.linalg.norm(final_pose["pos"] - init_pose["pos"])
            if disp > self.displacement_threshold:
                return DynamicValidation(
                    trajectory_metrics={"disturbed_object": obj_id, "displacement": disp},
                    per_finger_loads=np.zeros((1, 6)),
                    failure_stage="non_target_disturbed",
                    passed=False
                )

            # Note: Rotation distance would be checked here

        return DynamicValidation(
            trajectory_metrics={"lift_achieved": 0.1},
            per_finger_loads=np.zeros((1, 6)),
            failure_stage="none",
            passed=True
        )
