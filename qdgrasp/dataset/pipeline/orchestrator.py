import dataclasses
from typing import Any, Dict, List, Optional, Sequence, Tuple
from pathlib import Path
import numpy as np
import trimesh
import torch

from qdgrasp.robot.spec import RobotSpec
from qdgrasp.objects.schema import SubGeomSpec
from qdgrasp.dataset.pipeline.contracts import (
    ContactProposal,
    KinematicSolution,
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
from qdgrasp.dataset.pipeline.filter import filter_grasp_candidate
from qdgrasp.dataset.pipeline.validators.mujoco_rollout import validate_grasp_rollout


def _fit_palm_pose(
    source_tips: np.ndarray,
    target_tips: np.ndarray,
    source_directions: np.ndarray,
    target_normals: np.ndarray,
    direction_weight: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Fit a proper rigid transform to both tip positions and contact directions."""
    source_center = np.mean(source_tips, axis=0)
    target_center = np.mean(target_tips, axis=0)
    source_centered = source_tips - source_center
    target_centered = target_tips - target_center
    source_scale = max(float(np.linalg.norm(source_centered)), 1e-8)
    target_scale = max(float(np.linalg.norm(target_centered)), 1e-8)
    covariance = (source_centered / source_scale).T @ (target_centered / target_scale)
    covariance += direction_weight * source_directions.T @ target_normals
    u, _, vt = np.linalg.svd(covariance)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0.0:
        vt[-1] *= -1.0
        rotation = vt.T @ u.T
    translation = target_center - rotation @ source_center
    return translation.astype(np.float64), rotation.astype(np.float64)


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
    proposals: List[ContactProposal] = []
    for _ in range(num_candidates):
        try:
            proposals.append(
                generate_region_opposition_proposal(
                    mesh, num_fingers, rng, finger_ids, thumb_index=thumb_idx
                )
            )
        except ValueError:
            continue
    return proposals


def sample_wrench_guided_proposals(spec: RobotSpec, mesh: trimesh.Trimesh, rng: np.random.Generator, num_candidates: int = 16) -> List[ContactProposal]:
    num_fingers = len(spec.fingertip_links)
    finger_ids = np.arange(num_fingers)
    thumb_idx = num_fingers - 1
    for idx, name in enumerate(spec.fingertip_links):
        if "th" in name.lower() or "thumb" in name.lower():
            thumb_idx = idx
            break
    proposals: List[ContactProposal] = []
    for _ in range(num_candidates):
        try:
            proposals.append(
                generate_wrench_guided_proposal(
                    mesh, num_fingers, rng, finger_ids, thumb_index=thumb_idx
                )
            )
        except ValueError:
            continue
    return proposals


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
    object_pos: Optional[Tuple[float, float, float]] = None,
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
    if object_pos is None:
        object_pos = (0.0, 0.0, max(0.0, -float(mesh.bounds[0, 2])))
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
        "dynamic_skipped": 0,
        "accepted": 0,
    }

    outcomes: List[PipelineOutcome] = []

    # STAGE 1: PROPOSAL
    proposals = proposal_fn(spec, mesh, rng, num_candidates=num_candidates)
    missing_proposals = max(0, num_candidates - len(proposals))
    if missing_proposals:
        reasons_accounting["proposal_rejected"] += missing_proposals
        for _ in range(missing_proposals):
            outcomes.append(
                PipelineOutcome(
                    proposal_valid=False,
                    ik_valid=False,
                    collision_valid=False,
                    static_force_valid=False,
                    dynamic_valid=False,
                    failure_stage="proposal",
                    failure_reason="empty_proposals",
                    recipe_id=recipe_id,
                )
            )
    if not proposals:
        return outcomes, reasons_accounting

    # Compute approach palm poses and candidate arrays
    B = len(proposals)
    palm_poses = []
    palm_rots = []
    target_contacts = []
    target_normals = []

    # Fit the complete fingertip constellation at the midpoint pose to each
    # proposal.  A single averaged surface normal cannot initialize an opposing
    # grasp whose normals intentionally cancel.
    zero_pos = torch.zeros(1, 3, dtype=torch.float32)
    zero_rot = torch.eye(3, dtype=torch.float32).unsqueeze(0)
    seed_data = []
    for fraction in (0.15, 0.4, 0.65, 0.9):
        q_seed = torch.tensor(
            [[
                spec.joint_limits[name][0]
                + fraction * (spec.joint_limits[name][1] - spec.joint_limits[name][0])
                for name in spec.actuated_joint_names
            ]],
            dtype=torch.float32,
        )
        tips = spec.fingertip_positions(zero_pos, zero_rot, q_seed)[0].numpy()
        directions = spec.fingertip_contact_directions(zero_pos, zero_rot, q_seed)[0].numpy()
        seed_data.append((q_seed[0].numpy(), tips, directions))

    initial_q = []
    for prop in proposals:
        best = None
        for q_seed, tip_offsets, tip_directions_np in seed_data:
            selected_points = prop.target_points
            selected_normals = prop.inward_normals
            if prop.region_points is not None and prop.region_normals is not None:
                for _ in range(5):
                    palm_pos, R_palm = _fit_palm_pose(
                        tip_offsets, selected_points, tip_directions_np, selected_normals
                    )
                    transformed_tips = (R_palm @ tip_offsets.T).T + palm_pos
                    nearest = np.argmin(
                        np.sum(
                            (prop.region_points - transformed_tips[:, None, :]) ** 2,
                            axis=-1,
                        ),
                        axis=1,
                    )
                    finger_index = np.arange(len(spec.fingertip_links))
                    selected_points = prop.region_points[finger_index, nearest]
                    selected_normals = prop.region_normals[finger_index, nearest]
            palm_pos, R_palm = _fit_palm_pose(
                tip_offsets, selected_points, tip_directions_np, selected_normals
            )
            transformed_tips = (R_palm @ tip_offsets.T).T + palm_pos
            geometric_score = float(
                np.max(np.linalg.norm(transformed_tips - selected_points, axis=1))
            )
            score = geometric_score
            candidate = (
                score, q_seed, palm_pos, R_palm, selected_points, selected_normals
            )
            if best is None or score < best[0]:
                best = candidate

        assert best is not None
        _, q_seed, palm_pos, R_palm, selected_points, selected_normals = best

        palm_poses.append(palm_pos)
        palm_rots.append(R_palm)
        initial_q.append(q_seed)
        target_contacts.append(selected_points)
        target_normals.append(selected_normals)

    palm_pos_b = np.array(palm_poses, dtype=np.float32)
    palm_rot_b = np.array(palm_rots, dtype=np.float32)
    target_contacts_b = np.array(target_contacts, dtype=np.float32)
    target_normals_b = np.array(target_normals, dtype=np.float32)

    # STAGE 2: KINEMATICS (BATCHED DLS-IK)
    solver_kwargs: Dict[str, Any] = {
        "init_q": np.asarray(initial_q, dtype=np.float32),
        "max_iter": 80,
        "pos_tolerance": 0.005,
        "normal_tolerance_dot": 0.866,
    }
    if solver_name == "region_dls":
        solver_kwargs["region_points"] = np.stack(
            [prop.region_points for prop in proposals], axis=0
        )
        solver_kwargs["region_normals"] = np.stack(
            [prop.region_normals for prop in proposals], axis=0
        )
    kinematic_solutions = solver_fn(
        spec,
        palm_pos_b,
        palm_rot_b,
        target_contacts_b,
        target_normals_b,
        **solver_kwargs,
    )

    # PROCESS EACH CANDIDATE THROUGH THE REMAINING STAGES
    for i in range(B):
        prop = proposals[i]
        q_i = kinematic_solutions.q[i]
        converged_i = bool(kinematic_solutions.converged[i])
        ik_reason_i = str(kinematic_solutions.reason[i])

        single_kinematics = KinematicSolution(
            q=q_i,
            palm_pos=kinematic_solutions.palm_pos[i],
            palm_rot=kinematic_solutions.palm_rot[i],
            achieved_contacts=kinematic_solutions.achieved_contacts[i],
            achieved_normals=kinematic_solutions.achieved_normals[i],
            position_residuals=kinematic_solutions.position_residuals[i],
            normal_residuals=kinematic_solutions.normal_residuals[i],
            converged=converged_i,
            reason=ik_reason_i,
            iterations=(
                None
                if kinematic_solutions.iterations is None
                else np.asarray(kinematic_solutions.iterations[i])
            ),
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
                    recipe_id=recipe_id,
                    proposal=prop,
                    kinematics=single_kinematics,
                )
            )
            continue

        surface_contacts, surface_distances, surface_face_ids = (
            trimesh.proximity.closest_point_naive(mesh, single_kinematics.achieved_contacts)
        )
        surface_normals = -mesh.face_normals[surface_face_ids]
        if np.any(surface_distances > 0.005):
            reasons_accounting["ik_rejected"] += 1
            outcomes.append(
                PipelineOutcome(
                    proposal_valid=True,
                    ik_valid=False,
                    collision_valid=False,
                    static_force_valid=False,
                    dynamic_valid=False,
                    failure_stage="ik",
                    failure_reason="fingertip_not_on_surface",
                    recipe_id=recipe_id,
                    proposal=prop,
                    kinematics=single_kinematics,
                )
            )
            continue
        single_kinematics = dataclasses.replace(
            single_kinematics,
            surface_contacts=surface_contacts,
            surface_normals=surface_normals,
            surface_distances=surface_distances,
        )

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
                    recipe_id=recipe_id,
                    proposal=prop,
                    kinematics=single_kinematics,
                )
            )
            continue

        # STAGE 4: STATIC CERTIFIER (Force Closure / GWS)
        static_cert = certify_force_closure(
            target_points=single_kinematics.surface_contacts,
            inward_normals=single_kinematics.surface_normals,
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
                    recipe_id=recipe_id,
                    proposal=prop,
                    kinematics=single_kinematics,
                    static_certificate=static_cert,
                )
            )
            continue

        # STAGE 5: DYNAMIC VALIDATION
        if not run_dynamic or hand_xml_path is None:
            reasons_accounting["dynamic_skipped"] += 1
            outcomes.append(
                PipelineOutcome(
                    proposal_valid=True,
                    ik_valid=True,
                    collision_valid=True,
                    static_force_valid=True,
                    dynamic_valid=False,
                    failure_stage="dynamic",
                    failure_reason="dynamic_skipped",
                    recipe_id=recipe_id,
                    proposal=prop,
                    kinematics=single_kinematics,
                    static_certificate=static_cert,
                )
            )
            continue

        # Solve explicit actuator-control poses around the nominal on-surface
        # contact. q_open is a valid initial condition; q_squeeze commands a
        # small inward preload through actuators. Neither replaces the nominal
        # contact evidence stored in the dataset.
        command_axes = np.asarray(single_kinematics.achieved_normals, dtype=np.float32)
        command_targets = np.stack(
            [
                single_kinematics.achieved_contacts - 0.004 * command_axes,
                single_kinematics.achieved_contacts + 0.003 * command_axes,
            ],
            axis=0,
        )
        command_solution = solve_dls_ik_batch(
            spec,
            np.repeat(palm_pos_b[i : i + 1], 2, axis=0),
            np.repeat(palm_rot_b[i : i + 1], 2, axis=0),
            command_targets,
            np.repeat(command_axes[None, :, :], 2, axis=0),
            init_q=np.repeat(q_i[None, :], 2, axis=0),
            max_iter=35,
            pos_tolerance=0.0007,
            normal_tolerance_dot=0.8,
            require_normal_alignment=False,
        )
        if not bool(np.all(command_solution.converged)):
            reasons_accounting["dynamic_rejected"] += 1
            dyn_val = DynamicValidation(
                trajectory_metrics={
                    "open_command_max_error": float(
                        np.max(command_solution.position_residuals[0])
                    ),
                    "squeeze_command_max_error": float(
                        np.max(command_solution.position_residuals[1])
                    ),
                },
                per_finger_loads=np.zeros((len(spec.fingertip_links), 6)),
                failure_stage="command_ik",
                passed=False,
            )
            outcomes.append(
                PipelineOutcome(
                    proposal_valid=True,
                    ik_valid=True,
                    collision_valid=True,
                    static_force_valid=True,
                    dynamic_valid=False,
                    failure_stage="dynamic_command_ik",
                    failure_reason="rollout_command_ik_failed",
                    recipe_id=recipe_id,
                    proposal=prop,
                    kinematics=single_kinematics,
                    static_certificate=static_cert,
                    dynamic_validation=dyn_val,
                )
            )
            continue

        initial_targets = spec.expand_mimic_joint_targets(
            {
                name: float(command_solution.q[0, index])
                for index, name in enumerate(spec.actuated_joint_names)
            }
        )
        squeeze_targets = spec.expand_mimic_joint_targets(
            {
                name: float(command_solution.q[1, index])
                for index, name in enumerate(spec.actuated_joint_names)
            }
        )

        palm_pos_world = palm_pos_b[i] + np.array(object_pos, dtype=np.float32)
        expected_tips_world = single_kinematics.achieved_contacts + np.asarray(
            object_pos, dtype=np.float64
        )
        dyn_val = validate_grasp_rollout(
            hand_xml_path=hand_xml_path,
            collision_geoms=collision_geoms,
            fingertip_body_names=spec.fingertip_links,
            palm_pos=tuple(palm_pos_world.tolist()),
            palm_rot=palm_rot_b[i],
            joint_targets=squeeze_targets,
            initial_joint_targets=initial_targets,
            object_pos=object_pos,
            object_mass=object_mass,
            expected_fingertip_positions=expected_tips_world,
            fingertip_local_offsets=np.stack(
                [spec.fingertip_contact_offsets[name] for name in spec.fingertip_links],
                axis=0,
            ),
            pregrasp_distance=0.0,
            squeeze_steps=300,
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
                    recipe_id=recipe_id,
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
                    recipe_id=recipe_id,
                    proposal=prop,
                    kinematics=single_kinematics,
                    static_certificate=static_cert,
                    dynamic_validation=dyn_val,
                )
            )

    return outcomes, reasons_accounting
