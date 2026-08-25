from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np


@dataclass(frozen=True)
class SceneObjectSpec:
    object_id: str
    asset_ref: str
    T_world_object: np.ndarray  # [4, 4] homogeneous transform
    scale: float = 1.0
    mass: float | None = None
    friction: tuple[float, ...] | None = None


@dataclass(frozen=True)
class SupportGeometrySpec:
    support_id: str
    geom_type: str  # e.g., "plane", "box"
    params: dict[str, Any]
    T_world_support: np.ndarray  # [4, 4] homogeneous transform


@dataclass(frozen=True)
class CameraSpec:
    camera_id: str
    intrinsics: np.ndarray  # [3, 3] matrix
    distortion: np.ndarray | None = None
    T_world_camera: np.ndarray = field(default_factory=lambda: np.eye(4))


@dataclass(frozen=True)
class SceneSpec:
    scene_id: str
    source_dataset: str
    source_version: str
    source_split: str

    environment: str  # "table", "bin", "shelf", or "custom"
    objects: list[SceneObjectSpec]
    supports: list[SupportGeometrySpec] = field(default_factory=list)
    cameras: list[CameraSpec] = field(default_factory=list)

    gravity: tuple[float, float, float] = (0.0, 0.0, -9.81)
    timestep: float = 0.002
    solver_profile: str = "default"
    settle_seed: int = 0

    source_record_hash: str | None = None
    license_record: str | None = None
    redistributable: bool = False


@dataclass(frozen=True)
class SceneObservation:
    scene_id: str
    camera_id: str
    frame_id: str
    timestamp: float

    T_world_camera: np.ndarray  # [4, 4] matrix
    calibration_hash: str

    rgb_ref: str | None = None
    depth_ref: str | None = None
    point_cloud_ref: str | None = None
    point_cloud_frame: str | None = None
    instance_mask_ref: str | None = None
    normal_ref: str | None = None

    visibility_by_object: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class SceneGraspOutcome:
    scene_id: str
    target_object_id: str
    robot_profile: str
    candidate_id: str

    contact_opportunity: np.ndarray  # Shape [K, 3] points in object frame
    contact_opportunity_normals: np.ndarray  # Shape [K, 3] normals in object frame

    q_command: np.ndarray
    palm_T_command: np.ndarray
    active_fingers: np.ndarray

    approach_path: np.ndarray | None = None
    swept_clearance_metrics: dict[str, float] = field(default_factory=dict)

    static_certificate: dict[str, Any] = field(default_factory=dict)
    dynamic_trajectory_evidence: dict[str, Any] = field(default_factory=dict)

    target_motion: dict[str, float] = field(default_factory=dict)
    non_target_motion: dict[str, dict[str, float]] = field(default_factory=dict)
    scene_state_hashes: dict[str, str] = field(default_factory=dict)

    label_stage: str = "initial"
    failure_reason: str = "none"
    recipe_hash: str = ""
    protocol_hash: str = ""
    source_hash: str = ""


@dataclass(frozen=True)
class SourceDatasetInfo:
    dataset_id: str
    version: str
    is_valid: bool
    num_scenes: int
    license_type: str
    redistributable: bool


@dataclass(frozen=True)
class SceneIndex:
    dataset_id: str
    split: str
    scene_keys: list[str]


@dataclass(frozen=True)
class SourceEvidence:
    scene_key: str
    record_hash: str
    is_complete: bool
    missing_files: list[str]


@dataclass(frozen=True)
class ExternalGraspSet:
    scene_id: str
    gripper_type: str
    grasps: list[dict[str, Any]]
    source_provenance: str


@runtime_checkable
class SceneAdapter(Protocol):
    """
    Interface for third-party dataset adapters to load scene specifications and observations.
    """

    def probe(self, root: str) -> SourceDatasetInfo: ...

    def index(self, root: str, split: str, limit: int | None = None) -> SceneIndex: ...

    def load_scene(self, root: str, scene_key: str) -> SceneSpec: ...

    def load_observation(self, root: str, scene_key: str, camera_key: str, frame_key: str) -> SceneObservation: ...

    def load_external_grasps(self, root: str, scene_key: str) -> ExternalGraspSet: ...

    def audit(self, root: str, scene_key: str) -> SourceEvidence: ...
