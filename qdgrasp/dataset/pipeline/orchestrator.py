from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
from pathlib import Path
import numpy as np
import trimesh
import torch

from qdgrasp.robot.spec import RobotSpec
from qdgrasp.objects.schema import SubGeomSpec
from qdgrasp.dataset.pipeline.contracts import (
    ContactProposal,
    KinematicSolution,
    StaticCertificate,
    DynamicValidation,
    PipelineOutcome,
    get_recipe,
)
from qdgrasp.dataset.pipeline.proposals.surface_fixed import generate_surface_fixed_proposal
from qdgrasp.dataset.pipeline.proposals.region_opposition import generate_region_opposition_proposal
from qdgrasp.dataset.pipeline.proposals.wrench_guided import generate_wrench_guided_proposal
from qdgrasp.dataset.pipeline.solvers.fixed_contact_dls import solve_dls_ik_batch
from qdgrasp.dataset.pipeline.solvers.region_dls import solve_region_dls_ik_batch
from qdgrasp.dataset.pipeline.certifiers.contact_force import certify_force_closure
from qdgrasp.dataset.pipeline.certifiers.grasp_wrench import compute_grasp_wrench_space_quality
from qdgrasp.dataset.pipeline.filter import filter_grasp_candidate
from qdgrasp.dataset.pipeline.validators.mujoco_rollout import validate_grasp_rollout


from qdgrasp.dataset.pipeline.sample import _detect_hand_approach_axis, _construct_rotation_from_vectors


def sample_surface_fixed_proposals(spec: RobotSpec, mesh: trimesh.Trimesh, rng: np.random.Generator, num_candidates: int = 16) -> List[ContactProposal]:
    num_fingers = len(spec.fingertip_links)
    finger_ids = np.arange(num_fingers)
    return [generate_surface_fixed_proposal(mesh, num_fingers, rng, finger_ids) for _ in range(num_candidates)]


def sample_region_opposition_proposals(spec: RobotSpec, mesh: trimesh.Trimesh, rng: np.random.Generator, num_candidates: int = 16) -> List[ContactProposal]:
    num_fingers = len(spec.fingertip_links)
    finger_ids = np.arange(num_fingers)
    thumb_idx = num_fingers - 1
    for idx, name in enumerate(spec.fingertip_links):
        if "th" in name.lower() or "thumb" in name.lower():
            thumb_idx = idx
            break
    return [
        generate_region_opposition_proposal(mesh, num_fingers, rng, finger_ids, thumb_index=thumb_idx)
        for _ in range(num_candidates)
    ]


def sample_wrench_guided_proposals(spec: RobotSpec, mesh: trimesh.Trimesh, rng: np.random.Generator, num_candidates: int = 16) -> List[ContactProposal]:
    num_fingers = len(spec.fingertip_links)
    finger_ids = np.arange(num_fingers)
    thumb_idx = num_fingers - 1
    for idx, name in enumerate(spec.fingertip_links):
        if "th" in name.lower() or "thumb" in name.lower():
            thumb_idx = idx
            break
    return [
        generate_wrench_guided_proposal(mesh, num_fingers, rng, finger_ids, thumb_index=thumb_idx)
        for _ in range(num_candidates)
    ]


PROPOSAL_REGISTRY = {
    "surface_fixed": sample_surface_fixed_proposals,
    "region_opposition": sample_region_opposition_proposals,
    "wrench_guided": sample_wrench_guided_proposals,
}

SOLVER_REGISTRY = {
    "fixed_contact_dls": solve_dls_ik_batch,
    "region_dls": solve_region_dls_ik_batch,
}


def run_pipeline_chunk(
    recipe_id: str,
    spec: RobotSpec,
    mesh: trimesh.Trimesh,
    collision_geoms: Sequence[SubGeomSpec],
    hand_xml_path: Optional[str | Path],
    rng: np.random.Generator,
    num_candidates: int = 16,
    object_mass: float = 0.1,
    object_pos: Tuple[float, float, float] = (0.0, 0.0, 0.05),
    run_dynamic: bool = True,
) -> Tuple[List[PipelineOutcome], Dict[str, int]]:
    """
    Runs an end-to-end staged pipeline chunk:
      1. Proposal Strategy (based on recipe)
      2. Batched Kinematic Solver (based on recipe)
      3. Collision Filtering
      4. Static Certification
      5. Dynamic Simulation Rollout

    Returns a list of PipelineOutcome objects and a reason accounting dictionary.
    """
    recipe = get_recipe(recipe_id)
    proposal_name = recipe["proposal"]
    solver_name = recipe["solver"]

    proposal_fn = PROPOSAL_REGISTRY[proposal_name]
    solver_fn = SOLVER_REGISTRY[solver_name]

    reasons_accounting = {
        "proposal_rejected": 0,
        "ik_rejected": 0,
        "collision_rejected": 0,
        "static_force_rejected": 0,
        "dynamic_rejected": 0,
        "accepted": 0,
    }

    outcomes: List[PipelineOutcome] = []

    # STAGE 1: PROPOSAL
    proposals = proposal_fn(spec, mesh, rng, num_candidates=num_candidates)
    if not proposals or len(proposals) == 0:
        reasons_accounting["proposal_rejected"] += num_candidates
        for _ in range(num_candidates):
            outcomes.append(
                PipelineOutcome(
                    proposal_valid=False,
                    ik_valid=False,
                    collision_valid=False,
                    static_force_valid=False,
                    dynamic_valid=False,
                    failure_stage="proposal",
                    failure_reason="empty_proposals",
                )
            )
        return outcomes, reasons_accounting

    # Compute approach palm poses and candidate arrays
    B = len(proposals)
    palm_poses = []
    palm_rots = []
    target_contacts = []
    target_normals = []

    # Calculate hand reach & approach axis
    q_zero = torch.zeros(1, len(spec.actuated_joint_names), dtype=torch.float32)
    zero_pos = torch.zeros(1, 3, dtype=torch.float32)
    zero_rot = torch.eye(3, dtype=torch.float32).unsqueeze(0)
    tip_offsets = spec.fingertip_positions(zero_pos, zero_rot, q_zero)[0].numpy()
    if len(tip_offsets) > 0:
        hand_axis_unnorm = np.mean(tip_offsets, axis=0)
        hand_reach = float(np.linalg.norm(hand_axis_unnorm))
        hand_axis = hand_axis_unnorm / max(1e-5, hand_reach)
    else:
        hand_axis = np.array([0.0, 0.0, 1.0])
        hand_reach = 0.10

    for prop in proposals:
        avg_normal = np.mean(prop.inward_normals, axis=0)
        norm_val = np.linalg.norm(avg_normal)
        if norm_val < 1e-6:
            target_approach = np.array([0.0, 0.0, -1.0])
        else:
            target_approach = -avg_normal / norm_val

        center = np.mean(prop.target_points, axis=0)
        standoff = hand_reach * 0.78
        palm_pos = center - target_approach * standoff

        R_palm = _construct_rotation_from_vectors(hand_axis, target_approach)

        palm_poses.append(palm_pos)
        palm_rots.append(R_palm)
        target_contacts.append(prop.target_points)
        target_normals.append(prop.inward_normals)

    palm_pos_b = np.array(palm_poses, dtype=np.float32)
    palm_rot_b = np.array(palm_rots, dtype=np.float32)
    target_contacts_b = np.array(target_contacts, dtype=np.float32)
    target_normals_b = np.array(target_normals, dtype=np.float32)

    # STAGE 2: KINEMATICS (BATCHED DLS-IK)
    kinematic_solutions = solver_fn(
        spec,
        palm_pos_b,
        palm_rot_b,
        target_contacts_b,
        target_normals_b,
        max_iter=50,
        pos_tolerance=0.02,
        normal_tolerance_dot=0.5,
    )

    # PROCESS EACH CANDIDATE THROUGH THE REMAINING STAGES
    for i in range(B):
        prop = proposals[i]
        q_i = kinematic_solutions.q[i]
        converged_i = bool(kinematic_solutions.converged[i])
        ik_reason_i = str(kinematic_solutions.reason[i])

        single_kinematics = KinematicSolution(
            q=q_i,
            achieved_contacts=kinematic_solutions.achieved_contacts[i],
            achieved_normals=kinematic_solutions.achieved_normals[i],
            position_residuals=kinematic_solutions.position_residuals[i],
            normal_residuals=kinematic_solutions.normal_residuals[i],
            converged=converged_i,
            reason=ik_reason_i,
        )

        if not converged_i:
            reasons_accounting["ik_rejected"] += 1
            outcomes.append(
                PipelineOutcome(
                    proposal_valid=True,
                    ik_valid=False,
                    collision_valid=False,
                    static_force_valid=False,
                    dynamic_valid=False,
                    failure_stage="ik",
                    failure_reason=ik_reason_i,
                    proposal=prop,
                    kinematics=single_kinematics,
                )
            )
            continue

        # STAGE 3: COLLISION FILTER
        filter_res = filter_grasp_candidate(
            spec,
            palm_pos_b[i],
            palm_rot_b[i],
            q_i,
            mesh,
        )

        if not filter_res.valid:
            reasons_accounting["collision_rejected"] += 1
            outcomes.append(
                PipelineOutcome(
                    proposal_valid=True,
                    ik_valid=True,
                    collision_valid=False,
                    static_force_valid=False,
                    dynamic_valid=False,
                    failure_stage="collision",
                    failure_reason=filter_res.reason,
                    proposal=prop,
                    kinematics=single_kinematics,
                )
            )
            continue

        # STAGE 4: STATIC CERTIFIER (Force Closure / GWS)
        static_cert = certify_force_closure(
            target_points=single_kinematics.achieved_contacts,
            inward_normals=single_kinematics.achieved_normals,
            centroid=mesh.centroid,
            mass=object_mass,
        )

        if not static_cert.passed:
            reasons_accounting["static_force_rejected"] += 1
            outcomes.append(
                PipelineOutcome(
                    proposal_valid=True,
                    ik_valid=True,
                    collision_valid=True,
                    static_force_valid=False,
                    dynamic_valid=False,
                    failure_stage="static_force",
                    failure_reason="force_closure_failed",
                    proposal=prop,
                    kinematics=single_kinematics,
                    static_certificate=static_cert,
                )
            )
            continue

        # STAGE 5: DYNAMIC VALIDATION
        if not run_dynamic or hand_xml_path is None:
            reasons_accounting["accepted"] += 1
            outcomes.append(
                PipelineOutcome(
                    proposal_valid=True,
                    ik_valid=True,
                    collision_valid=True,
                    static_force_valid=True,
                    dynamic_valid=True,
                    failure_stage="none",
                    failure_reason="passed",
                    proposal=prop,
                    kinematics=single_kinematics,
                    static_certificate=static_cert,
                )
            )
            continue

        j_targets = {
            j_name: float(q_i[j_idx])
            for j_idx, j_name in enumerate(spec.actuated_joint_names)
        }

        palm_pos_world = palm_pos_b[i] + np.array(object_pos, dtype=np.float32)
        dyn_val = validate_grasp_rollout(
            hand_xml_path=hand_xml_path,
            collision_geoms=collision_geoms,
            fingertip_body_names=spec.fingertip_links,
            palm_pos=tuple(palm_pos_world.tolist()),
            palm_rot=palm_rot_b[i],
            joint_targets=j_targets,
            object_pos=object_pos,
            object_mass=object_mass,
        )

        if not dyn_val.passed:
            reasons_accounting["dynamic_rejected"] += 1
            outcomes.append(
                PipelineOutcome(
                    proposal_valid=True,
                    ik_valid=True,
                    collision_valid=True,
                    static_force_valid=True,
                    dynamic_valid=False,
                    failure_stage=f"dynamic_{dyn_val.failure_stage}",
                    failure_reason=f"rollout_failed_{dyn_val.failure_stage}",
                    proposal=prop,
                    kinematics=single_kinematics,
                    static_certificate=static_cert,
                    dynamic_validation=dyn_val,
                )
            )
        else:
            reasons_accounting["accepted"] += 1
            outcomes.append(
                PipelineOutcome(
                    proposal_valid=True,
                    ik_valid=True,
                    collision_valid=True,
                    static_force_valid=True,
                    dynamic_valid=True,
                    failure_stage="none",
                    failure_reason="passed",
                    proposal=prop,
                    kinematics=single_kinematics,
                    static_certificate=static_cert,
                    dynamic_validation=dyn_val,
                )
            )

    return outcomes, reasons_accounting
