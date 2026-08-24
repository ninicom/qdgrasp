import dataclasses
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
import trimesh

from qdgrasp.dataset.pipeline.certifiers.contact_force import certify_force_closure
from qdgrasp.dataset.pipeline.contracts import (
    ContactProposal,
    KinematicSolution,
    PipelineOutcome,
    get_recipe,
)
from qdgrasp.dataset.pipeline.filter import filter_grasp_candidate
from qdgrasp.dataset.pipeline.palm_hypotheses import (
    PalmHypothesisError,
    best_palm_hypothesis,
    generate_palm_hypotheses,
)
from qdgrasp.dataset.pipeline.proposals.identity import normalize_active_fingers
from qdgrasp.dataset.pipeline.proposals.region_opposition import generate_region_opposition_proposal
from qdgrasp.dataset.pipeline.proposals.surface_fixed import generate_surface_fixed_proposal
from qdgrasp.dataset.pipeline.proposals.width_mapper import WidthMapper
from qdgrasp.dataset.pipeline.proposals.wrench_guided import generate_wrench_guided_proposal
from qdgrasp.dataset.pipeline.solvers.fixed_contact_dls import solve_dls_ik_batch
from qdgrasp.dataset.pipeline.solvers.joint_palm_dls import solve_joint_palm_dls_batch
from qdgrasp.dataset.pipeline.solvers.region_dls import solve_region_dls_ik_batch
from qdgrasp.dataset.pipeline.validators.collision_admission import (
    admit_mujoco_collision_pose,
)
from qdgrasp.dataset.pipeline.validators.mujoco_rollout import validate_grasp_rollout
from qdgrasp.objects.schema import SubGeomSpec
from qdgrasp.robot.spec import RobotSpec


def _fit_palm_pose(
    source_tips: np.ndarray,
    target_tips: np.ndarray,
    source_directions: np.ndarray,
    target_normals: np.ndarray,
    direction_weight: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
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


def sample_surface_fixed_proposals(
    spec: RobotSpec, mesh: trimesh.Trimesh, rng: np.random.Generator, num_candidates: int = 16
) -> list[ContactProposal]:
    num_fingers = len(spec.fingertip_links)
    finger_ids = np.arange(num_fingers)
    return [generate_surface_fixed_proposal(mesh, num_fingers, rng, finger_ids) for _ in range(num_candidates)]


def _opposing_finger_schedule(num_fingers: int, thumb_idx: int) -> list[int]:
    """Give region and wrench-guided recipes the same morphology ordering."""
    return [index for index in range(num_fingers) if index != thumb_idx]


def sample_region_opposition_proposals(
    spec: RobotSpec, mesh: trimesh.Trimesh, rng: np.random.Generator, num_candidates: int = 16
) -> list[ContactProposal]:
    num_fingers = len(spec.fingertip_links)
    finger_ids = np.arange(num_fingers)
    thumb_idx = num_fingers - 1
    for idx, name in enumerate(spec.fingertip_links):
        if "th" in name.lower() or "thumb" in name.lower():
            thumb_idx = idx
            break
    proposals: list[ContactProposal] = []
    opposing_fingers = _opposing_finger_schedule(num_fingers, thumb_idx)
    for candidate_index in range(num_candidates):
        try:
            proposals.append(
                generate_region_opposition_proposal(
                    mesh,
                    num_fingers,
                    rng,
                    finger_ids,
                    thumb_index=thumb_idx,
                    opposing_finger_index=opposing_fingers[candidate_index % len(opposing_fingers)],
                )
            )
        except ValueError:
            continue
    return proposals


def sample_wrench_guided_proposals(
    spec: RobotSpec, mesh: trimesh.Trimesh, rng: np.random.Generator, num_candidates: int = 16
) -> list[ContactProposal]:
    num_fingers = len(spec.fingertip_links)
    finger_ids = np.arange(num_fingers)
    thumb_idx = num_fingers - 1
    for idx, name in enumerate(spec.fingertip_links):
        if "th" in name.lower() or "thumb" in name.lower():
            thumb_idx = idx
            break
    proposals: list[ContactProposal] = []
    opposing_fingers = _opposing_finger_schedule(num_fingers, thumb_idx)
    for candidate_index in range(num_candidates):
        try:
            proposals.append(
                generate_wrench_guided_proposal(
                    mesh,
                    num_fingers,
                    rng,
                    finger_ids,
                    thumb_index=thumb_idx,
                    opposing_finger_index=opposing_fingers[candidate_index % len(opposing_fingers)],
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
    hand_xml_path: str | Path | None,
    rng: np.random.Generator,
    num_candidates: int = 16,
    object_mass: float = 0.1,
    object_pos: tuple[float, float, float] | None = None,
    run_dynamic: bool = True,
) -> tuple[list[PipelineOutcome], dict[str, int]]:
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

    outcomes: list[PipelineOutcome] = []

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
    palm_poses = []
    palm_rots = []
    target_contacts = []
    target_normals = []
    active_finger_masks = []
    prepared_proposals: list[ContactProposal] = []
    palm_hypothesis_ids: list[str] = []
    palm_hypothesis_metrics: list[dict[str, float]] = []

    # Fit the complete fingertip constellation at the midpoint pose to each
    # proposal.  A single averaged surface normal cannot initialize an opposing
    # grasp whose normals intentionally cancel.
    zero_pos = torch.zeros(1, 3, dtype=torch.float32)
    zero_rot = torch.eye(3, dtype=torch.float32).unsqueeze(0)
    mapper = WidthMapper(spec)
    canonical_seed = mapper.get_canonical_open_qpos()
    target_grasp_width = float(np.min(mesh.extents))
    fractional_seeds = [
        np.asarray(
            [
                spec.joint_limits[name][0] + fraction * (spec.joint_limits[name][1] - spec.joint_limits[name][0])
                for name in spec.actuated_joint_names
            ],
            dtype=np.float32,
        )
        for fraction in (0.4, 0.65)
    ]
    opposition_seed_cache: dict[tuple[bool, ...], np.ndarray] = {}

    initial_q = []
    for prop in proposals:
        active_fingers = normalize_active_fingers(prop.active_fingers, len(spec.fingertip_links))
        active_key = tuple(bool(value) for value in active_fingers)
        if active_key not in opposition_seed_cache:
            opposition_seed_cache[active_key] = mapper.map_width_to_opposition_qpos(
                target_grasp_width,
                active_fingers=active_fingers,
            )
        seed_data = []
        for seed_q in (
            canonical_seed,
            opposition_seed_cache[active_key],
            *fractional_seeds,
        ):
            q_seed = torch.from_numpy(np.asarray(seed_q, dtype=np.float32)[None])
            tips = spec.fingertip_positions(zero_pos, zero_rot, q_seed)[0].numpy()
            directions = spec.fingertip_contact_directions(zero_pos, zero_rot, q_seed)[0].numpy()
            seed_data.append((q_seed[0].numpy(), tips, directions))
        best = None
        for seed_index, (q_seed, tip_offsets, tip_directions_np) in enumerate(seed_data):
            selected_points = prop.target_points
            selected_normals = prop.inward_normals
            try:
                if prop.region_points is not None and prop.region_normals is not None:
                    for refine_index in range(3):
                        hypothesis = best_palm_hypothesis(
                            source_tips=tip_offsets,
                            source_directions=tip_directions_np,
                            target_tips=selected_points,
                            target_normals=selected_normals,
                            active_fingers=active_fingers,
                            opposition_pairs=prop.opposition_pairs,
                            object_centroid=mesh.centroid,
                            floor_z=-float(object_pos[2]),
                            hypothesis_prefix=(f"{prop.candidate_id}:seed{seed_index}:region{refine_index}"),
                        )
                        palm_pos, R_palm = hypothesis.palm_pos, hypothesis.palm_rot
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
                final_hypotheses = generate_palm_hypotheses(
                    source_tips=tip_offsets,
                    source_directions=tip_directions_np,
                    target_tips=selected_points,
                    target_normals=selected_normals,
                    active_fingers=active_fingers,
                    opposition_pairs=prop.opposition_pairs,
                    object_centroid=mesh.centroid,
                    floor_z=-float(object_pos[2]),
                    hypothesis_prefix=f"{prop.candidate_id}:seed{seed_index}:final",
                )
            except PalmHypothesisError:
                continue
            for hypothesis in final_hypotheses:
                palm_pos, R_palm = hypothesis.palm_pos, hypothesis.palm_rot
                hypothesis_collision = None
                if hand_xml_path is not None:
                    seed_joint_targets = spec.expand_mimic_joint_targets(
                        {name: float(q_seed[index]) for index, name in enumerate(spec.actuated_joint_names)}
                    )

                    def check_hypothesis_collision(
                        candidate_palm_pos,
                        *,
                        candidate_active_fingers=active_fingers,
                        candidate_palm_rot=R_palm,
                        candidate_joint_targets=seed_joint_targets,
                    ):
                        return admit_mujoco_collision_pose(
                            hand_xml_path,
                            collision_geoms,
                            palm_body_name=spec.palm_link,
                            fingertip_body_names=spec.fingertip_links,
                            active_fingers=candidate_active_fingers,
                            palm_pos=candidate_palm_pos + np.asarray(object_pos, dtype=np.float64),
                            palm_rot=candidate_palm_rot,
                            joint_targets=candidate_joint_targets,
                            object_pos=object_pos,
                            object_mass=object_mass,
                        )

                    hypothesis_collision = check_hypothesis_collision(palm_pos)
                    if (
                        hypothesis_collision.reason == "hand_floor_contact"
                        and hypothesis_collision.min_hand_floor_clearance >= -0.01
                    ):
                        floor_lift = -hypothesis_collision.min_hand_floor_clearance + 0.001
                        palm_pos = palm_pos + np.array([0.0, 0.0, floor_lift])
                        transformed_tips = (R_palm @ tip_offsets.T).T + palm_pos
                        lifted_error = np.linalg.norm(
                            transformed_tips[active_fingers] - selected_points[active_fingers],
                            axis=1,
                        )
                        hypothesis = dataclasses.replace(
                            hypothesis,
                            palm_pos=palm_pos,
                            max_position_error=float(np.max(lifted_error)),
                            floor_clearance=hypothesis.floor_clearance + floor_lift,
                        )
                        hypothesis_collision = check_hypothesis_collision(palm_pos)
                    # Active fingertip overlap is a refinable seed condition;
                    # every non-tip/object, self, and floor collision remains
                    # a hard initializer rejection, and final admission below
                    # applies the strict penetration limit again after IK.
                    if (
                        not hypothesis_collision.passed
                        and hypothesis_collision.reason != "active_tip_excessive_penetration"
                    ):
                        continue
                score = hypothesis.sort_key
                candidate = (
                    score,
                    hypothesis,
                    hypothesis_collision,
                    q_seed,
                    palm_pos,
                    R_palm,
                    selected_points,
                    selected_normals,
                )
                if best is None or score < best[0]:
                    best = candidate

        if best is None:
            reasons_accounting["proposal_rejected"] += 1
            outcomes.append(
                PipelineOutcome(
                    proposal_valid=False,
                    ik_valid=False,
                    collision_valid=False,
                    static_force_valid=False,
                    dynamic_valid=False,
                    failure_stage="proposal",
                    failure_reason="palm_hypothesis_unavailable",
                    recipe_id=recipe_id,
                    proposal=prop,
                )
            )
            continue
        (
            _,
            selected_hypothesis,
            selected_hypothesis_collision,
            q_seed,
            palm_pos,
            R_palm,
            selected_points,
            selected_normals,
        ) = best

        prepared_proposals.append(prop)
        palm_poses.append(palm_pos)
        palm_rots.append(R_palm)
        initial_q.append(q_seed)
        # Proposal points remain exact mesh samples.  The configured FK point
        # is a fingertip contact anchor, while the compiled rounded fingertip
        # geom has a support surface outside that anchor for oblique normals.
        # Keep the anchor one half-position-tolerance outside the mesh; final
        # surface projection and exact MuJoCo admission remain authoritative.
        target_contacts.append(selected_points - 0.0005 * selected_normals)
        target_normals.append(selected_normals)
        active_finger_masks.append(active_fingers)
        palm_hypothesis_ids.append(selected_hypothesis.hypothesis_id)
        palm_hypothesis_metrics.append(
            {
                "max_position_error": selected_hypothesis.max_position_error,
                "min_normal_alignment": selected_hypothesis.min_normal_alignment,
                "floor_clearance": selected_hypothesis.floor_clearance,
                "exact_min_hand_floor_clearance": (
                    float("nan")
                    if selected_hypothesis_collision is None
                    else selected_hypothesis_collision.min_hand_floor_clearance
                ),
                "exact_max_penetration": (
                    float("nan")
                    if selected_hypothesis_collision is None
                    else selected_hypothesis_collision.max_penetration
                ),
            }
        )

    proposals = prepared_proposals
    B = len(proposals)
    if B == 0:
        return outcomes, reasons_accounting

    palm_pos_b = np.array(palm_poses, dtype=np.float32)
    palm_rot_b = np.array(palm_rots, dtype=np.float32)
    target_contacts_b = np.array(target_contacts, dtype=np.float32)
    target_normals_b = np.array(target_normals, dtype=np.float32)
    active_fingers_b = np.array(active_finger_masks, dtype=bool)

    # STAGE 2: KINEMATICS (BATCHED DLS-IK)
    solver_kwargs: dict[str, Any] = {
        "init_q": np.asarray(initial_q, dtype=np.float32),
        "max_iter": 40,
        "pos_tolerance": 0.001,
        "normal_tolerance_dot": 0.866,
        "active_fingers": active_fingers_b,
    }
    if solver_name == "region_dls":
        solver_kwargs["region_points"] = np.stack([prop.region_points for prop in proposals], axis=0)
        solver_kwargs["region_normals"] = np.stack([prop.region_normals for prop in proposals], axis=0)
    first_kinematic_pass = solver_fn(
        spec,
        palm_pos_b,
        palm_rot_b,
        target_contacts_b,
        target_normals_b,
        **solver_kwargs,
    )

    # Keep the pinned total budget at 80 iterations, but make the second half
    # the plan's actual joint+palm trust-region solve.  The previous point-only
    # Kabsch correction could improve positions while rotating configured
    # contact axes away from their target normals.
    kinematic_solutions = solve_joint_palm_dls_batch(
        spec,
        palm_pos_b,
        palm_rot_b,
        target_contacts_b,
        target_normals_b,
        init_q=np.asarray(first_kinematic_pass.q, dtype=np.float32),
        active_fingers=active_fingers_b,
        max_iter=40,
        pos_tolerance=0.001,
        normal_tolerance_dot=0.866,
        floor_z=-float(object_pos[2]),
    )
    refinement_metrics: list[dict[str, float]] = []
    for candidate_index in range(B):
        translation = float(np.linalg.norm(kinematic_solutions.palm_pos[candidate_index] - palm_pos_b[candidate_index]))
        relative_rotation = kinematic_solutions.palm_rot[candidate_index] @ palm_rot_b[candidate_index].T
        rotation_cosine = np.clip((np.trace(relative_rotation) - 1.0) * 0.5, -1.0, 1.0)
        rotation = float(np.arccos(rotation_cosine))
        refinement_metrics.append(
            {
                "requested_translation": translation,
                "applied_translation": translation,
                "requested_rotation": rotation,
                "applied_rotation": rotation,
                "floor_clearance": float(kinematic_solutions.palm_pos[candidate_index, 2] + object_pos[2]),
                "floor_rejected": 0.0,
            }
        )
    combined_metrics: dict[str, np.ndarray] = {}
    if kinematic_solutions.solver_metrics is not None:
        combined_metrics = {
            name: np.asarray(values).copy() for name, values in kinematic_solutions.solver_metrics.items()
        }
    if first_kinematic_pass.solver_metrics is not None:
        second_observed_linearization = (
            np.asarray(kinematic_solutions.solver_metrics["accepted_steps"])
            + np.asarray(kinematic_solutions.solver_metrics["rejected_steps"])
        ) > 0
        for name in ("accepted_steps", "rejected_steps", "limit_clipped_steps"):
            combined_metrics[name] = np.asarray(first_kinematic_pass.solver_metrics[name]) + np.asarray(
                kinematic_solutions.solver_metrics[name]
            )
        combined_metrics["initial_cost"] = np.asarray(first_kinematic_pass.solver_metrics["initial_cost"])
        # A second pass can satisfy convergence at its initial-state check and
        # therefore never build a Jacobian.  Preserve the last actually
        # observed linearization instead of reporting the solver's zero-filled
        # allocation as rank/gradient evidence.
        for name in (
            "jacobian_rank",
            "jacobian_condition",
            "gradient_norm",
            "raw_step_norm",
            "projected_step_norm",
            "final_damping",
        ):
            combined_metrics[name] = np.where(
                second_observed_linearization,
                np.asarray(kinematic_solutions.solver_metrics[name]),
                np.asarray(first_kinematic_pass.solver_metrics[name]),
            )
        combined_metrics["finite"] = np.asarray(first_kinematic_pass.solver_metrics["finite"], dtype=bool) & np.asarray(
            kinematic_solutions.solver_metrics["finite"], dtype=bool
        )
    combined_metrics["pose_refinement_applied"] = np.asarray(
        [metrics["applied_translation"] > 0.0 or metrics["applied_rotation"] > 0.0 for metrics in refinement_metrics],
        dtype=bool,
    )
    combined_metrics["pose_refinement_translation"] = np.asarray(
        [metrics["applied_translation"] for metrics in refinement_metrics],
        dtype=np.float64,
    )
    combined_metrics["pose_refinement_rotation"] = np.asarray(
        [metrics["applied_rotation"] for metrics in refinement_metrics],
        dtype=np.float64,
    )
    kinematic_solutions = dataclasses.replace(
        kinematic_solutions,
        iterations=(np.asarray(first_kinematic_pass.iterations) + np.asarray(kinematic_solutions.iterations)),
        solver_metrics=combined_metrics,
    )
    palm_pos_b = np.asarray(kinematic_solutions.palm_pos, dtype=np.float32)
    palm_rot_b = np.asarray(kinematic_solutions.palm_rot, dtype=np.float32)

    # PROCESS EACH CANDIDATE THROUGH THE REMAINING STAGES
    for i in range(B):
        prop = proposals[i]
        q_i = kinematic_solutions.q[i]
        active_i = active_fingers_b[i]
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
                None if kinematic_solutions.iterations is None else np.asarray(kinematic_solutions.iterations[i])
            ),
            solver_metrics=(
                None
                if kinematic_solutions.solver_metrics is None
                else {name: np.asarray(values[i]) for name, values in kinematic_solutions.solver_metrics.items()}
            ),
            palm_hypothesis_id=palm_hypothesis_ids[i],
            palm_hypothesis_metrics={
                **palm_hypothesis_metrics[i],
                **{f"refinement_{name}": float(value) for name, value in refinement_metrics[i].items()},
            },
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

        surface_contacts, surface_distances, surface_face_ids = trimesh.proximity.closest_point_naive(
            mesh, single_kinematics.achieved_contacts
        )
        surface_normals = -mesh.face_normals[surface_face_ids]
        if np.any(surface_distances[active_i] > 0.001):
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

        if hand_xml_path is None:
            reasons_accounting["collision_rejected"] += 1
            outcomes.append(
                PipelineOutcome(
                    proposal_valid=True,
                    ik_valid=True,
                    collision_valid=False,
                    static_force_valid=False,
                    dynamic_valid=False,
                    failure_stage="collision",
                    failure_reason="exact_collision_model_unavailable",
                    recipe_id=recipe_id,
                    proposal=prop,
                    kinematics=single_kinematics,
                )
            )
            continue

        collision_joint_targets = spec.expand_mimic_joint_targets(
            {name: float(q_i[index]) for index, name in enumerate(spec.actuated_joint_names)}
        )
        collision_admission = admit_mujoco_collision_pose(
            hand_xml_path,
            collision_geoms,
            palm_body_name=spec.palm_link,
            fingertip_body_names=spec.fingertip_links,
            active_fingers=active_i,
            palm_pos=palm_pos_b[i] + np.asarray(object_pos, dtype=np.float32),
            palm_rot=palm_rot_b[i],
            joint_targets=collision_joint_targets,
            object_pos=object_pos,
            object_mass=object_mass,
        )
        if not collision_admission.passed:
            reasons_accounting["collision_rejected"] += 1
            outcomes.append(
                PipelineOutcome(
                    proposal_valid=True,
                    ik_valid=True,
                    collision_valid=False,
                    static_force_valid=False,
                    dynamic_valid=False,
                    failure_stage="collision",
                    failure_reason=collision_admission.reason,
                    recipe_id=recipe_id,
                    proposal=prop,
                    kinematics=single_kinematics,
                    collision_admission=collision_admission,
                )
            )
            continue

        # STAGE 4: STATIC CERTIFIER (Force Closure / GWS)
        static_cert = certify_force_closure(
            target_points=single_kinematics.surface_contacts[active_i],
            inward_normals=single_kinematics.surface_normals[active_i],
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
                    collision_admission=collision_admission,
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
                    collision_admission=collision_admission,
                    static_certificate=static_cert,
                )
            )
            continue

        # P3.2.1-08: command the active contact displacement directly through
        # the compiled transmission task planner.  A second joint-space IK
        # gate here would reintroduce the global-q feasibility decision that
        # the task-space command contract replaced.
        command_axes = np.asarray(single_kinematics.achieved_normals, dtype=np.float32)
        desired_preload_displacement = np.zeros_like(single_kinematics.achieved_contacts, dtype=np.float64)
        desired_preload_displacement[active_i] = 0.0015 * command_axes[active_i]
        initial_targets = collision_joint_targets
        squeeze_targets = collision_joint_targets

        palm_pos_world = palm_pos_b[i] + np.array(object_pos, dtype=np.float32)
        expected_tips_world = single_kinematics.achieved_contacts + np.asarray(object_pos, dtype=np.float64)
        dyn_val = validate_grasp_rollout(
            hand_xml_path=hand_xml_path,
            collision_geoms=collision_geoms,
            fingertip_body_names=spec.fingertip_links,
            palm_pos=tuple(palm_pos_world.tolist()),
            palm_rot=palm_rot_b[i],
            joint_targets=squeeze_targets,
            initial_joint_targets=initial_targets,
            contact_joint_targets=collision_joint_targets,
            active_fingers=active_i,
            desired_fingertip_displacement=desired_preload_displacement,
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
                    collision_admission=collision_admission,
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
                    collision_admission=collision_admission,
                    static_certificate=static_cert,
                    dynamic_validation=dyn_val,
                )
            )

    return outcomes, reasons_accounting
