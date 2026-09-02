import dataclasses
from typing import Any

import numpy as np


@dataclasses.dataclass(frozen=True)
class ContactProposal:
    """Output of a ProposalStrategy (e.g. surface_fixed, region_opposition, wrench_guided)."""
    target_points: np.ndarray  # [K, 3] points or region centers
    face_ids: np.ndarray       # [K]
    inward_normals: np.ndarray # [K, 3]
    finger_ids: np.ndarray     # [K]
    active_fingers: np.ndarray | None = None  # [K] bool task membership
    opposition_pairs: np.ndarray | None = None  # [P, 2] finger indices
    candidate_id: str = ""     # stable content identity
    region_points: np.ndarray | None = None  # [K, R, 3] exact surface samples
    region_face_ids: np.ndarray | None = None  # [K, R]
    region_normals: np.ndarray | None = None  # [K, R, 3]
    force_hints: np.ndarray | None = None # [K, 3]
    provenance: str = ""       # Identity of the strategy

@dataclasses.dataclass(frozen=True)
class KinematicSolution:
    """Output of a KinematicSolver (e.g. fixed_contact_dls, region_dls)."""
    q: np.ndarray               # [B, J] joint configuration
    palm_pos: np.ndarray        # [B, 3] palm pose used by FK
    palm_rot: np.ndarray        # [B, 3, 3] palm pose used by FK
    achieved_contacts: np.ndarray # [B, K, 3]
    achieved_normals: np.ndarray  # [B, K, 3]
    position_residuals: np.ndarray # [B, K] per-finger position error
    normal_residuals: np.ndarray   # [B, K] per-finger normal angular error
    converged: np.ndarray       # [B] boolean mask
    reason: np.ndarray          # [B] string or enum for failure reason
    iterations: np.ndarray | None = None  # [B] solver iterations consumed
    surface_contacts: np.ndarray | None = None  # nearest object points [B,K,3]
    surface_normals: np.ndarray | None = None  # inward object normals [B,K,3]
    surface_distances: np.ndarray | None = None  # tip-to-surface distance [B,K]
    solver_metrics: dict[str, np.ndarray] | None = None  # per-candidate solver telemetry
    palm_hypothesis_id: str | None = None
    palm_hypothesis_metrics: dict[str, float] | None = None


@dataclasses.dataclass(frozen=True)
class CollisionAdmission:
    """Exact compiled-scene collision evidence for one contact pose."""
    passed: bool
    reason: str
    contact_pairs: tuple[dict[str, Any], ...]
    max_penetration: float
    min_hand_floor_clearance: float

@dataclasses.dataclass(frozen=True)
class StaticCertificate:
    """Output of a StaticCertifier (e.g. contact_force, grasp_wrench)."""
    force_solution: np.ndarray  # [K, 3] per-finger solved force
    cone_residual: float        # max deviation from friction cone
    object_wrench: np.ndarray   # [6] net wrench on object
    quality_margin: float       # minimum distance to wrench space boundary or equivalent
    passed: bool                # True if strictly force closure

@dataclasses.dataclass(frozen=True)
class DynamicValidation:
    """Output of a DynamicValidator (e.g. mujoco_rollout)."""
    trajectory_metrics: dict[str, Any]
    per_finger_loads: np.ndarray # [K, 6] per-finger measured load (force, torque)
    failure_stage: str           # e.g., 'lift', 'perturbation', 'none'
    passed: bool                 # True if survived dynamic disturbance

@dataclasses.dataclass(frozen=True)
class PipelineOutcome:
    """Combines evidence of all stages without losing intermediate rejections."""
    proposal_valid: bool
    ik_valid: bool
    collision_valid: bool
    static_force_valid: bool
    dynamic_valid: bool
    failure_stage: str
    failure_reason: str
    recipe_id: str = ""

    proposal: ContactProposal | None = None
    kinematics: KinematicSolution | None = None
    collision_admission: CollisionAdmission | None = None
    static_certificate: StaticCertificate | None = None
    dynamic_validation: DynamicValidation | None = None

# Allowlist Registry
ALLOWED_RECIPES = {
    "surface_fixed_v1": {
        "proposal": "surface_fixed",
        "solver": "fixed_contact_dls"
    },
    "region_opposition_v1": {
        "proposal": "region_opposition",
        "solver": "region_dls"
    },
    "wrench_guided_v1": {
        "proposal": "wrench_guided",
        "solver": "region_dls"
    }
}

class RegistryError(Exception):
    pass

def get_recipe(recipe_id: str) -> dict[str, str]:
    if recipe_id not in ALLOWED_RECIPES:
        raise RegistryError(f"Recipe {recipe_id} is not in the allowlist.")
    return ALLOWED_RECIPES[recipe_id]
