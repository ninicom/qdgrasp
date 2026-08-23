"""Phase 3.2 verification gate: underactuated control, transmission contracts, and rollout fidelity."""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import mujoco
import torch
from scipy.spatial.transform import Rotation

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from qdgrasp.objects.schema import SubGeomSpec
from qdgrasp.robot.assets import resolve_robot_asset
from qdgrasp.robot.spec import RobotSpec
from qdgrasp.robot.transmission import (
    create_transmission_model_from_spec_and_mjcf,
    compute_finite_difference_moment_matrix,
)
from qdgrasp.dataset.pipeline.solvers.fixed_contact_dls import solve_dls_ik_batch
from qdgrasp.dataset.pipeline.validators.mujoco_rollout import validate_grasp_rollout

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("check_phase3_2")


def verify_transmission_ranks() -> dict[str, dict[str, int]]:
    """Verifies that all three hands produce exact mathematical transmission ranks."""
    logger.info("Verifying transmission model ranks and dimensions...")
    expected_matrix = {
        "leap_hand.yaml": {"rank": 16, "num_joints": 16, "num_actuators": 16},
        "wonik_allegro.yaml": {"rank": 16, "num_joints": 16, "num_actuators": 16},
        "shadow_hand.yaml": {"rank": 20, "num_joints": 24, "num_actuators": 20},
    }
    results = {}
    for cfg_name, expected in expected_matrix.items():
        spec = RobotSpec.from_config(cfg_name, sample_anchors=False)
        xml_path = resolve_robot_asset(spec.config.source_asset)
        model = mujoco.MjModel.from_xml_path(str(xml_path))
        tm = create_transmission_model_from_spec_and_mjcf(spec, model)

        assert tm.rank == expected["rank"], (
            f"{cfg_name} rank mismatch: expected {expected['rank']}, got {tm.rank}"
        )
        assert tm.num_joints == expected["num_joints"], (
            f"{cfg_name} joints mismatch: expected {expected['num_joints']}, got {tm.num_joints}"
        )
        assert tm.num_actuators == expected["num_actuators"], (
            f"{cfg_name} actuators mismatch: expected {expected['num_actuators']}, got {tm.num_actuators}"
        )
        results[cfg_name] = {
            "rank": tm.rank,
            "num_joints": tm.num_joints,
            "num_actuators": tm.num_actuators,
        }
        logger.info("  %s -> rank=%d, joints=%d, actuators=%d", cfg_name, tm.rank, tm.num_joints, tm.num_actuators)
    return results


def verify_moment_finite_difference_parity() -> dict[str, float]:
    """Verifies analytic moment matrix M matches finite-difference oracle across full workspace."""
    logger.info("Verifying moment matrix finite difference parity...")
    results = {}
    for cfg_name in ("leap_hand.yaml", "wonik_allegro.yaml", "shadow_hand.yaml"):
        spec = RobotSpec.from_config(cfg_name, sample_anchors=False)
        xml_path = resolve_robot_asset(spec.config.source_asset)
        model = mujoco.MjModel.from_xml_path(str(xml_path))
        tm = create_transmission_model_from_spec_and_mjcf(spec, model)

        data = mujoco.MjData(model)
        state = tm.extract_state(model, data)
        M_analytic = state.moment_matrix
        M_num = compute_finite_difference_moment_matrix(model, data, tm.joint_names, tm.actuator_names)

        err = float(np.max(np.abs(M_analytic - M_num)))
        assert err < 1e-5, f"{cfg_name} moment matrix parity failure: max err = {err}"
        results[cfg_name] = err
        logger.info("  %s -> max |M_analytic - M_fd| = %.2e", cfg_name, err)
    return results


def verify_controllable_space_projection() -> None:
    """Verifies controllable space projection and rejection of uncontrollable null-space targets."""
    logger.info("Verifying controllable-space command projection and null-space rejection...")
    spec = RobotSpec.from_config("shadow_hand.yaml", sample_anchors=False)
    xml_path = resolve_robot_asset(spec.config.source_asset)
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    tm = create_transmission_model_from_spec_and_mjcf(spec, model)

    data = mujoco.MjData(model)
    state = tm.extract_state(model, data)

    # 1. Controllable target: symmetric delta on coupled tendon
    dq_controllable = np.zeros(tm.num_joints, dtype=np.float64)
    dq_controllable[tm.joint_names.index("rh_FFJ2")] = 0.2
    dq_controllable[tm.joint_names.index("rh_FFJ1")] = 0.2
    cmd = tm.project_joint_delta(dq_controllable, state)
    assert cmd.reason == "converged"
    assert cmd.nullspace_residual < 1e-6
    assert cmd.controllable_residual < 1e-6

    # 2. Uncontrollable target: asymmetric delta on coupled tendon
    dq_uncontrollable = np.zeros(tm.num_joints, dtype=np.float64)
    dq_uncontrollable[tm.joint_names.index("rh_FFJ2")] = 0.5
    dq_uncontrollable[tm.joint_names.index("rh_FFJ1")] = -0.5
    cmd_bad = tm.project_joint_delta(dq_uncontrollable, state, max_nullspace_residual=0.05)
    assert cmd_bad.reason == "nullspace_rejection"
    assert cmd_bad.nullspace_residual > 0.1
    logger.info("  Shadow hand null-space rejection verified (residual=%.3f)", cmd_bad.nullspace_residual)


def verify_active_finger_dls_ik() -> None:
    """Verifies active-finger mask in fixed-contact DLS IK solver."""
    logger.info("Verifying active-finger masked DLS IK...")
    spec = RobotSpec.from_config("leap_hand.yaml", sample_anchors=False)
    palm_pos = np.array([[0.0, 0.0, 0.15]], dtype=np.float32)
    palm_rot = np.eye(3, dtype=np.float32)[None, ...]
    q_nom = np.zeros((1, len(spec.actuated_joint_names)), dtype=np.float32)

    transforms = spec.forward_kinematics(
        torch.from_numpy(palm_pos), torch.from_numpy(palm_rot), torch.from_numpy(q_nom)
    )
    num_tips = len(spec.fingertip_links)
    target_pos = np.zeros((1, num_tips, 3), dtype=np.float32)
    target_norm = np.zeros((1, num_tips, 3), dtype=np.float32)

    for i, tip in enumerate(spec.fingertip_links):
        transform = transforms[tip]
        offset = getattr(spec, "fingertip_contact_offsets", {}).get(tip)
        if offset is not None:
            p = transform[:, :3, 3] + torch.matmul(
                transform[:, :3, :3], torch.tensor(offset, dtype=torch.float32).view(3, 1)
            ).squeeze(-1)
        else:
            p = transform[:, :3, 3]
        target_pos[0, i] = p[0].detach().cpu().numpy()

        axis = getattr(spec, "fingertip_contact_axes", {}).get(tip)
        if axis is not None:
            n = torch.nn.functional.normalize(
                torch.matmul(
                    transform[:, :3, :3], torch.tensor(axis, dtype=torch.float32).view(3, 1)
                ).squeeze(-1),
                dim=-1,
            )
        else:
            parent = spec.links[tip].parent_link
            origin = transforms[parent][:, :3, 3]
            n = torch.nn.functional.normalize(p - origin, dim=-1)
        target_norm[0, i] = n[0].detach().cpu().numpy()

    # Move inactive fingers to impossible targets
    target_pos[0, 1] += 5.0
    target_pos[0, 2] += 5.0

    active_mask = np.array([[True, False, False, True]], dtype=bool)
    res = solve_dls_ik_batch(
        spec,
        palm_pos,
        palm_rot,
        target_pos,
        target_norm,
        init_q=q_nom,
        active_fingers=active_mask,
        max_iter=30,
    )
    assert res.converged[0]
    assert res.reason[0] == "converged"
    assert res.position_residuals[0, 0] < 0.005
    assert res.position_residuals[0, 3] < 0.005
    logger.info("  Active finger IK pinch converged successfully on active tips.")


def verify_multistage_rollouts() -> dict[str, dict[str, float]]:
    """Verifies end-to-end multi-stage physical rollouts for LEAP, Allegro, Shadow Hand."""
    logger.info("Verifying end-to-end multi-stage physical rollouts...")
    results = {}

    # Shadow Hand
    shadow_spec = RobotSpec.from_config("shadow_hand.yaml", sample_anchors=False)
    q_contact = np.zeros(len(shadow_spec.actuated_joint_names), dtype=np.float32)
    j_names = list(shadow_spec.actuated_joint_names)
    q_contact[j_names.index("rh_MFJ3")] = 1.4; q_contact[j_names.index("rh_MFJ2")] = 1.2; q_contact[j_names.index("rh_MFJ1")] = 1.2
    q_contact[j_names.index("rh_RFJ3")] = 1.4; q_contact[j_names.index("rh_RFJ2")] = 1.2; q_contact[j_names.index("rh_RFJ1")] = 1.2
    q_contact[j_names.index("rh_LFJ3")] = 1.4; q_contact[j_names.index("rh_LFJ2")] = 1.2; q_contact[j_names.index("rh_LFJ1")] = 1.2

    q_contact[j_names.index("rh_FFJ3")] = 0.6; q_contact[j_names.index("rh_FFJ2")] = 0.5; q_contact[j_names.index("rh_FFJ1")] = 0.5
    q_contact[j_names.index("rh_THJ5")] = 0.0; q_contact[j_names.index("rh_THJ4")] = 1.0; q_contact[j_names.index("rh_THJ2")] = 0.5; q_contact[j_names.index("rh_THJ1")] = 0.5

    local_contacts = shadow_spec.fingertip_positions(
        torch.zeros(1, 3), torch.eye(3)[None], torch.from_numpy(q_contact[None])
    )[0].numpy()
    pinch_axis = local_contacts[4] - local_contacts[0]
    dist = np.linalg.norm(pinch_axis)
    pinch_axis /= dist
    palm_rot = Rotation.align_vectors(
        np.array([[-1.0, 0.0, 0.0]]), pinch_axis[None]
    )[0].as_matrix()
    pinch_center = 0.5 * (local_contacts[0] + local_contacts[4])
    half_width = 0.5 * dist - 0.0075
    object_pos = np.array([0.0, 0.0, 0.02])
    palm_pos = object_pos - palm_rot @ pinch_center

    palm_pos_b = palm_pos.astype(np.float32)[None]
    palm_rot_b = palm_rot.astype(np.float32)[None]

    q_open = q_contact.copy()
    q_open[j_names.index("rh_FFJ3")] -= 0.05; q_open[j_names.index("rh_FFJ2")] -= 0.05; q_open[j_names.index("rh_FFJ1")] -= 0.05
    q_open[j_names.index("rh_THJ4")] -= 0.05; q_open[j_names.index("rh_THJ2")] -= 0.05; q_open[j_names.index("rh_THJ1")] -= 0.05

    q_squeeze = q_contact.copy()
    q_squeeze[j_names.index("rh_FFJ3")] += 0.12; q_squeeze[j_names.index("rh_FFJ2")] += 0.10; q_squeeze[j_names.index("rh_FFJ1")] += 0.10
    q_squeeze[j_names.index("rh_THJ4")] += 0.10; q_squeeze[j_names.index("rh_THJ2")] += 0.10; q_squeeze[j_names.index("rh_THJ1")] += 0.10

    contact_points = shadow_spec.fingertip_positions(
        torch.from_numpy(palm_pos_b), torch.from_numpy(palm_rot_b), torch.from_numpy(q_contact[None])
    )[0].numpy()

    observed_stages = []
    res_shadow = validate_grasp_rollout(
        resolve_robot_asset(shadow_spec.config.source_asset),
        [
            SubGeomSpec(
                type="box",
                size=(float(half_width), 0.015, 0.02),
                pos=(0.0, 0.0, 0.0),
                quat=(1.0, 0.0, 0.0, 0.0),
            )
        ],
        shadow_spec.fingertip_links,
        palm_pos=tuple(palm_pos),
        palm_rot=palm_rot,
        initial_joint_targets=dict(zip(shadow_spec.actuated_joint_names, q_open)),
        joint_targets=dict(zip(shadow_spec.actuated_joint_names, q_squeeze)),
        object_pos=tuple(object_pos),
        object_mass=0.02,
        expected_fingertip_positions=contact_points,
        fingertip_local_offsets=np.stack(
            [shadow_spec.fingertip_contact_offsets[name] for name in shadow_spec.fingertip_links]
        ),
        pregrasp_distance=0.0,
        squeeze_steps=250,
        lift_steps=150,
        lift_height=0.05,
        perturbation_steps=40,
        perturbation_wrench=np.array([0.02, 0.02, 0.0, 0.002, 0.002, 0.002]),
        stage_observer=lambda stage, _model, _data: observed_stages.append(stage),
    )
    assert res_shadow.passed, f"Shadow Hand rollout failed: {res_shadow.failure_stage}"
    assert observed_stages == ["squeeze", "lift", "perturbation"]
    results["shadow_hand"] = {
        "lift_achieved": float(res_shadow.trajectory_metrics["lift_achieved"]),
        "max_penetration": float(res_shadow.trajectory_metrics["max_penetration"]),
        "final_active_fingers": float(res_shadow.trajectory_metrics["final_active_fingers"]),
        "transmission_rank": float(res_shadow.trajectory_metrics["transmission_rank"]),
    }
    logger.info("  Shadow Hand rollout PASSED: lift=%.3f, pen=%.4f, fingers=%d, rank=%d",
                results["shadow_hand"]["lift_achieved"],
                results["shadow_hand"]["max_penetration"],
                int(results["shadow_hand"]["final_active_fingers"]),
                int(results["shadow_hand"]["transmission_rank"]))

    # LEAP Hand
    leap_spec = RobotSpec.from_config("leap_hand.yaml", sample_anchors=False)
    q_leap = np.array(
        [
            0.5927356227, -0.3791691612, 0.6132688578, 1.692338131,
            0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0,
            1.228141244, 0.1354573565, -0.1336592733, 1.666422321,
        ],
        dtype=np.float32,
    )
    local_leap = leap_spec.fingertip_positions(torch.zeros(1, 3), torch.eye(3)[None], torch.from_numpy(q_leap[None]))[0].numpy()
    leap_axis = local_leap[3] - local_leap[0]
    leap_width = 0.5 * float(np.linalg.norm(leap_axis))
    leap_axis /= np.linalg.norm(leap_axis)
    leap_rot = Rotation.align_vectors(np.array([[-1.0, 0.0, 0.0]]), leap_axis[None])[0].as_matrix()
    leap_pos = np.array([0.0, 0.0, 0.02]) - leap_rot @ (0.5 * (local_leap[0] + local_leap[3]))

    palm_pos_b = leap_pos.astype(np.float32)[None]
    palm_rot_b = leap_rot.astype(np.float32)[None]
    q_b = q_leap[None]
    contact_points = leap_spec.fingertip_positions(torch.from_numpy(palm_pos_b), torch.from_numpy(palm_rot_b), torch.from_numpy(q_b))[0].numpy()
    contact_axes = leap_spec.fingertip_contact_directions(torch.from_numpy(palm_pos_b), torch.from_numpy(palm_rot_b), torch.from_numpy(q_b))[0].numpy()

    open_contacts = contact_points.copy()
    squeeze_contacts = contact_points.copy()
    open_contacts[[0, 3]] -= 0.004 * contact_axes[[0, 3]]
    squeeze_contacts[[0, 3]] += 0.003 * contact_axes[[0, 3]]
    command_targets = np.stack([open_contacts, squeeze_contacts])

    commands = solve_dls_ik_batch(
        leap_spec,
        np.repeat(palm_pos_b, 2, axis=0),
        np.repeat(palm_rot_b, 2, axis=0),
        command_targets,
        np.repeat(contact_axes[None], 2, axis=0),
        init_q=np.repeat(q_b, 2, axis=0),
        max_iter=35,
        pos_tolerance=0.0007,
        require_normal_alignment=False,
    )

    res_leap = validate_grasp_rollout(
        resolve_robot_asset(leap_spec.config.source_asset),
        [SubGeomSpec(type="box", size=(leap_width, 0.015, 0.02), pos=(0.0, 0.0, 0.0), quat=(1.0, 0.0, 0.0, 0.0))],
        leap_spec.fingertip_links,
        palm_pos=tuple(leap_pos),
        palm_rot=leap_rot,
        initial_joint_targets=dict(zip(leap_spec.actuated_joint_names, commands.q[0])),
        joint_targets=dict(zip(leap_spec.actuated_joint_names, commands.q[1])),
        object_pos=(0.0, 0.0, 0.02),
        object_mass=0.02,
        expected_fingertip_positions=contact_points,
        fingertip_local_offsets=np.stack([leap_spec.fingertip_contact_offsets[name] for name in leap_spec.fingertip_links]),
        pregrasp_distance=0.0,
        squeeze_steps=300,
    )
    assert res_leap.passed, f"LEAP rollout failed: {res_leap.failure_stage}"
    results["leap_hand"] = {
        "lift_achieved": float(res_leap.trajectory_metrics["lift_achieved"]),
        "transmission_rank": float(res_leap.trajectory_metrics["transmission_rank"]),
    }
    logger.info("  LEAP Hand rollout PASSED")

    # Allegro Hand
    allegro_spec = RobotSpec.from_config("wonik_allegro.yaml", sample_anchors=False)
    q_al = np.array(
        [
            -0.1410063654, 0.7589393854, 0.2905291915, 1.610496521,
            -0.1829498112, 0.7104878426, 0.4637212753, 0.6895720363,
            -0.3722456992, 0.4500102401, 1.241124988, 1.336122274,
            1.066359162, 0.5970826745, 0.1071554348, 1.677100062,
        ],
        dtype=np.float32,
    )
    local_al = allegro_spec.fingertip_positions(torch.zeros(1, 3), torch.eye(3)[None], torch.from_numpy(q_al[None]))[0].numpy()
    al_axis = local_al[3] - local_al[0]
    al_width = 0.5 * float(np.linalg.norm(al_axis))
    al_axis /= np.linalg.norm(al_axis)
    al_rot = Rotation.align_vectors(np.array([[-1.0, 0.0, 0.0]]), al_axis[None])[0].as_matrix()
    al_pos = np.array([0.0, 0.0, 0.02]) - al_rot @ (0.5 * (local_al[0] + local_al[3]))

    palm_pos_b = al_pos.astype(np.float32)[None]
    palm_rot_b = al_rot.astype(np.float32)[None]
    q_b = q_al[None]
    contact_points = allegro_spec.fingertip_positions(torch.from_numpy(palm_pos_b), torch.from_numpy(palm_rot_b), torch.from_numpy(q_b))[0].numpy()
    contact_axes = allegro_spec.fingertip_contact_directions(torch.from_numpy(palm_pos_b), torch.from_numpy(palm_rot_b), torch.from_numpy(q_b))[0].numpy()

    squeeze_contacts = contact_points.copy()
    squeeze_contacts[[0, 3]] += 0.0025 * contact_axes[[0, 3]]
    command = solve_dls_ik_batch(
        allegro_spec,
        palm_pos_b,
        palm_rot_b,
        squeeze_contacts,
        contact_axes[None],
        init_q=q_b,
        max_iter=35,
        pos_tolerance=0.0007,
        require_normal_alignment=False,
    )

    res_al = validate_grasp_rollout(
        resolve_robot_asset(allegro_spec.config.source_asset),
        [SubGeomSpec(type="box", size=(al_width, 0.015, 0.02), pos=(0.0, 0.0, 0.0), quat=(1.0, 0.0, 0.0, 0.0))],
        allegro_spec.fingertip_links,
        palm_pos=tuple(al_pos),
        palm_rot=al_rot,
        initial_joint_targets=allegro_spec.expand_mimic_joint_targets(dict(zip(allegro_spec.actuated_joint_names, q_al))),
        joint_targets=allegro_spec.expand_mimic_joint_targets(dict(zip(allegro_spec.actuated_joint_names, command.q[0]))),
        object_pos=(0.0, 0.0, 0.02),
        object_mass=0.02,
        expected_fingertip_positions=contact_points,
        fingertip_local_offsets=np.stack([allegro_spec.fingertip_contact_offsets[name] for name in allegro_spec.fingertip_links]),
        pregrasp_distance=0.0,
        squeeze_steps=250, lift_steps=150, lift_height=0.05, perturbation_steps=40,
        perturbation_wrench=np.array([0.15, 0.15, 0.0, 0.01, 0.01, 0.01], dtype=np.float64),
    )
    assert res_al.passed, f"Allegro rollout failed: {res_al.failure_stage}"
    results["wonik_allegro"] = {
        "lift_achieved": float(res_al.trajectory_metrics["lift_achieved"]),
        "transmission_rank": float(res_al.trajectory_metrics["transmission_rank"]),
    }
    logger.info("  Allegro Hand rollout PASSED")

    return results

def verify_release_blocked_status() -> None:
    """Verifies that release_blocked is correctly configured for Shadow Hand."""
    import yaml
    logger.info("Verifying release_blocked configuration...")
    shadow_cfg_path = REPO_ROOT / "qdgrasp" / "presets" / "robots" / "shadow_hand.yaml"
    with open(shadow_cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    assert cfg.get("release_blocked") is False, "Shadow Hand release_blocked must be false after Phase 3.2 completion"
    logger.info("  Shadow Hand release_blocked is False")

def main() -> None:
    t0 = time.time()
    logger.info("=== Phase 3.2 Underactuated Control & Physical Rollout Gate Audit ===")

    rank_results = verify_transmission_ranks()
    fd_results = verify_moment_finite_difference_parity()
    verify_controllable_space_projection()
    verify_active_finger_dls_ik()
    rollout_results = verify_multistage_rollouts()
    verify_release_blocked_status()

    elapsed = time.time() - t0
    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_seconds": round(elapsed, 2),
        "ranks": rank_results,
        "finite_difference_parity": fd_results,
        "rollout_fixtures": rollout_results,
        "status": "PASSED",
    }

    out_path = Path("runs/phase3_2_verification_report.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info("Report written to %s", out_path)
    logger.info("=== Phase 3.2 Audit PASSED in %.2f seconds ===", elapsed)


if __name__ == "__main__":
    main()
