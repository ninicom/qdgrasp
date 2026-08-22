#!/usr/bin/env python3
"""Fail-closed CPU gate script for Phase 2 (Robot Layer)."""

from __future__ import annotations

import json
import logging
import subprocess
import sys
import tempfile
from pathlib import Path

import mujoco
import numpy as np
import torch
import yaml

from qdgrasp.config import ConfigError, dump_document, load_robot_config, parse_document
from qdgrasp.config.schema import ROBOT_SCHEMA_V1
from qdgrasp.robot.graph import HandGraph
from qdgrasp.robot.meshes import load_mesh, resolve_mesh_path
from qdgrasp.robot.mjcf import parse_mjcf
from qdgrasp.robot.normalize import normalize_urdf, sha256_file
from qdgrasp.robot.provenance import validate_profile_for_release
from qdgrasp.robot.schema import ROBOT_SCHEMA_V2, RobotConfigV2
from qdgrasp.robot.spec import RobotSpec
from qdgrasp.robot.urdf import parse_urdf
from qdgrasp.sim.fixtures import evaluate_grasp_fixture
from qdgrasp.sim.mujoco import MujocoSim


def check_robot_assets_lock(problems: list[str], root: Path) -> None:
    checker = root / "scripts" / "check_robot_assets.py"
    res = subprocess.run(
        [sys.executable, str(checker), "--source-root", str(root / ".references" / "robot-assets")],
        cwd=root,
        capture_output=True,
        text=True,
    )
    if res.returncode != 0:
        problems.append(f"check_robot_assets.py failed: {res.stderr.strip() or res.stdout.strip()}")


def check_schema_round_trip_and_migration(problems: list[str]) -> None:
    # 1. v1 document still parses cleanly
    v1_cfg = load_robot_config("dummy-hand.yaml")
    if v1_cfg.schema_version != ROBOT_SCHEMA_V1:
        problems.append(f"dummy-hand.yaml schema is not {ROBOT_SCHEMA_V1}")

    # 2. v2 documents round-trip through YAML with matching content hash
    for name in ("leap_hand.yaml", "wonik_allegro.yaml", "shadow_hand.yaml"):
        cfg = load_robot_config(name)
        if not isinstance(cfg, RobotConfigV2) or cfg.schema_version != ROBOT_SCHEMA_V2:
            problems.append(f"{name} is not RobotConfigV2 with schema {ROBOT_SCHEMA_V2}")
            continue
        dumped = dump_document(cfg)
        reparsed = parse_document(yaml.safe_load(dumped), RobotConfigV2, origin=name)
        if reparsed != cfg:
            problems.append(f"{name} failed YAML round-trip equality")
        if reparsed.content_hash() != cfg.content_hash():
            problems.append(f"{name} failed YAML round-trip content_hash match")

    # 3. Unknown/dead keys are rejected
    bad_doc = {
        "schema": "qdgrasp/robot/v2",
        "name": "bad",
        "format": "mjcf",
        "source_asset": "dummy.xml",
        "palm_link": "palm",
        "joints": ["j1"],
        "joint_limits": {"j1": [0.0, 1.0]},
        "dead_key_unknown": True,
    }
    try:
        parse_document(bad_doc, RobotConfigV2, origin="gate_test")
    except ConfigError:
        pass
    else:
        problems.append("RobotConfigV2 accepted an unknown/dead key")


def check_urdf_and_mjcf_parsing(problems: list[str], root: Path) -> None:
    # LEAP URDF: 17 links, 16 movable joints
    leap_urdf_path = root / ".references/robot-assets/leap-hand-sim/assets/leap_hand/robot.urdf"
    if leap_urdf_path.is_file():
        m_leap_u = parse_urdf(leap_urdf_path)
        if len(m_leap_u.links) != 17 or len(m_leap_u.movable_joints) != 16:
            problems.append(
                f"LEAP URDF expected 17 links / 16 movable joints, got {len(m_leap_u.links)} links / {len(m_leap_u.movable_joints)} joints"
            )

    # Allegro ROS2 URDF: 22 links, 16 movable joints
    allegro_urdf_path = (
        root
        / ".references/robot-assets/wonik-allegro-ros2/src/allegro_hand_controllers/urdf/allegro_hand_description_right_A.urdf"
    )
    if allegro_urdf_path.is_file():
        m_allegro_u = parse_urdf(allegro_urdf_path)
        if len(m_allegro_u.links) != 22 or len(m_allegro_u.movable_joints) != 16:
            problems.append(
                f"Allegro URDF expected 22 links / 16 movable joints, got {len(m_allegro_u.links)} links / {len(m_allegro_u.movable_joints)} joints"
            )

    # dex-urdf Shadow URDF (parser fixture): 33 links, 24 movable joints
    shadow_urdf_path = root / ".references/robot-assets/dex-urdf/robots/hands/shadow_hand/shadow_hand_right.urdf"
    if shadow_urdf_path.is_file():
        m_shadow_u = parse_urdf(shadow_urdf_path)
        if len(m_shadow_u.links) != 33 or len(m_shadow_u.movable_joints) != 24:
            problems.append(
                f"dex-urdf Shadow URDF expected 33 links / 24 movable joints, got {len(m_shadow_u.links)} links / {len(m_shadow_u.movable_joints)} joints"
            )

    # Menagerie MJCFs
    # LEAP: nq 16, nu 16
    m_leap_m = parse_mjcf(root / ".references/robot-assets/mujoco-menagerie/leap_hand/right_hand.xml")
    if m_leap_m.nq != 16 or m_leap_m.nu != 16:
        problems.append(f"LEAP MJCF expected nq=16, nu=16, got nq={m_leap_m.nq}, nu={m_leap_m.nu}")

    # Allegro: nq 16, nu 16
    m_allegro_m = parse_mjcf(root / ".references/robot-assets/mujoco-menagerie/wonik_allegro/right_hand.xml")
    if m_allegro_m.nq != 16 or m_allegro_m.nu != 16:
        problems.append(f"Allegro MJCF expected nq=16, nu=16, got nq={m_allegro_m.nq}, nu={m_allegro_m.nu}")

    # Shadow: nq 24, nu 20
    m_shadow_m = parse_mjcf(root / ".references/robot-assets/mujoco-menagerie/shadow_hand/right_hand.xml")
    if m_shadow_m.nq != 24 or m_shadow_m.nu != 20:
        problems.append(f"Shadow MJCF expected nq=24, nu=20, got nq={m_shadow_m.nq}, nu={m_shadow_m.nu}")


def check_mesh_resolution(problems: list[str], root: Path) -> None:
    # For all 3 hands, build RobotSpec and verify all referenced meshes load cleanly
    for name in ("leap_hand.yaml", "wonik_allegro.yaml", "shadow_hand.yaml"):
        spec = RobotSpec.from_config(name, sample_anchors=False)
        for link in spec.links.values():
            for m_path in link.mesh_paths:
                if not m_path.is_file():
                    problems.append(f"{name}: missing mesh file {m_path}")
                else:
                    try:
                        mesh_obj = load_mesh(m_path)
                        if len(mesh_obj.vertices) == 0:
                            problems.append(f"{name}: empty mesh vertices in {m_path}")
                    except Exception as exc:
                        problems.append(f"{name}: failed to load mesh {m_path}: {exc}")


def check_normalization_reproducibility(problems: list[str], root: Path) -> None:
    src = (
        root
        / ".references/robot-assets/wonik-allegro-ros2/src/allegro_hand_controllers/urdf/allegro_hand_description_right_A.urdf"
    )
    if not src.is_file():
        problems.append(f"source Allegro URDF not found: {src}")
        return

    src_hash_before = sha256_file(src)
    mesh_dir = (
        root / ".references/robot-assets/wonik-allegro-ros2/src/allegro_hand_controllers/meshes"
    ).resolve()

    with tempfile.TemporaryDirectory(prefix="qdgrasp-norm-gate-") as tmpdir:
        out1 = Path(tmpdir) / "run1" / "allegro.normalized.urdf"
        out2 = Path(tmpdir) / "run2" / "allegro.normalized.urdf"

        m1 = normalize_urdf(
            src,
            out1,
            package_replacements={"package://allegro_hand_controllers/meshes": str(mesh_dir)},
            sanitize_inertias=True,
        )
        m2 = normalize_urdf(
            src,
            out2,
            package_replacements={"package://allegro_hand_controllers/meshes": str(mesh_dir)},
            sanitize_inertias=True,
        )

        if m1["output_sha256"] != m2["output_sha256"]:
            problems.append(
                f"normalization transform is not deterministic: {m1['output_sha256']} vs {m2['output_sha256']}"
            )
        if not m1["modified"]:
            problems.append("normalization manifest missing modified=True flag")

        # Check raw source file was untouched
        src_hash_after = sha256_file(src)
        if src_hash_before != src_hash_after:
            problems.append("normalization mutated the raw source asset in .references!")

        # Verify normalized URDF passes MuJoCo forward
        try:
            sim = MujocoSim(out1)
            sim.forward()
        except Exception as exc:
            problems.append(f"normalized Allegro URDF failed MuJoCo forward: {exc}")


def check_semantic_link_negative_tests(problems: list[str]) -> None:
    # 1. Reject missing palm_link
    try:
        RobotConfigV2.model_validate(
            {
                "schema": "qdgrasp/robot/v2",
                "name": "bad_palm",
                "format": "mjcf",
                "source_asset": ".references/robot-assets/mujoco-menagerie/leap_hand/right_hand.xml",
                "palm_link": "non_existent_palm_link",
                "joints": ["if_mcp"],
                "joint_limits": {"if_mcp": [-0.314, 2.23]},
            }
        )
    except ConfigError:
        pass

    try:
        bad_cfg = RobotConfigV2(
            schema="qdgrasp/robot/v2",
            name="bad_palm",
            format="mjcf",
            source_asset=".references/robot-assets/mujoco-menagerie/leap_hand/right_hand.xml",
            palm_link="non_existent_palm_link",
            joints=("if_mcp",),
            joint_limits={"if_mcp": (-0.314, 2.23)},
        )
        RobotSpec.from_config(bad_cfg, sample_anchors=False)
    except ConfigError:
        pass
    else:
        problems.append("RobotSpec silently accepted a non-existent palm link without raising ConfigError")


MJCF_PROFILES = {
    "leap_hand.yaml": ".references/robot-assets/mujoco-menagerie/leap_hand/right_hand.xml",
    "wonik_allegro.yaml": ".references/robot-assets/mujoco-menagerie/wonik_allegro/right_hand.xml",
    "shadow_hand.yaml": ".references/robot-assets/mujoco-menagerie/shadow_hand/right_hand.xml",
}
FK_GROUND_TRUTH_ATOL = 1e-4


def check_forward_kinematics_ground_truth(problems: list[str], root: Path) -> None:
    """Compare RobotSpec FK against MuJoCo ``mj_forward`` for every shared body.

    Batch-versus-single agreement cannot detect a wrong FK because both sides run
    the same code, so the gate needs an independent oracle.  MuJoCo is that
    oracle: the profile's own joint values are written into ``qpos``, the palm
    pose that ``mj_forward`` produces is fed to ``forward_kinematics``, and every
    body -- not only the palm's descendants -- must land in the same place.
    """

    for profile, mjcf_relative in MJCF_PROFILES.items():
        spec = RobotSpec.from_config(profile, sample_anchors=False)
        config = spec.config
        model = mujoco.MjModel.from_xml_path(str(root / mjcf_relative))
        data = mujoco.MjData(model)

        generator = np.random.default_rng(20260822)
        angles: dict[str, float] = {}
        for joint_name in config.joints:
            lower, upper = config.joint_limits[joint_name]
            angles[joint_name] = float(generator.uniform(lower, upper))

        # Resolve the profile's declared coupling and write it into MuJoCo too, so
        # both sides describe the same full configuration.  Whether the declared
        # ratio matches the hand's physical tendon is a separate question; this
        # check is about FK arithmetic.
        full_angles = dict(angles)
        for mimic_name, mimic in config.mimic_joints.items():
            full_angles[mimic_name] = angles.get(mimic.target_joint, 0.0) * mimic.multiplier + mimic.offset

        for joint_name, value in full_angles.items():
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if joint_id < 0:
                problems.append(f"{profile}: declared joint '{joint_name}' is absent from the MJCF")
                continue
            data.qpos[model.jnt_qposadr[joint_id]] = value
        mujoco.mj_forward(model, data)

        palm_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, config.palm_link)
        if palm_id < 0:
            problems.append(f"{profile}: palm link '{config.palm_link}' is absent from the MJCF")
            continue

        transforms = spec.forward_kinematics(
            torch.tensor(data.xpos[palm_id], dtype=torch.float32).unsqueeze(0),
            torch.tensor(data.xmat[palm_id].reshape(3, 3), dtype=torch.float32).unsqueeze(0),
            torch.tensor([[angles[name] for name in config.joints]], dtype=torch.float32),
        )

        compared = 0
        # Track position and rotation independently: a rotation-only error would
        # be invisible if it were only reported for the worst-position link.
        worst = {"position": ("", 0.0), "rotation": ("", 0.0)}
        for link_name, transform in transforms.items():
            body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, link_name)
            if body_id < 0:
                continue
            compared += 1
            errors = {
                "position": float(np.abs(transform[0, :3, 3].numpy() - data.xpos[body_id]).max()),
                "rotation": float(
                    np.abs(transform[0, :3, :3].numpy() - data.xmat[body_id].reshape(3, 3)).max()
                ),
            }
            for kind, error in errors.items():
                if error > worst[kind][1]:
                    worst[kind] = (link_name, error)

        if compared < len(spec.links):
            problems.append(
                f"{profile}: only {compared}/{len(spec.links)} links could be compared against MuJoCo"
            )
        for kind, (link_name, error) in worst.items():
            if error > FK_GROUND_TRUTH_ATOL:
                problems.append(
                    f"{profile}: FK {kind} disagrees with mj_forward at '{link_name}' "
                    f"({error:.6f}, atol {FK_GROUND_TRUTH_ATOL})"
                )


def check_forward_kinematics_and_batch_parity(problems: list[str]) -> None:
    for name in ("leap_hand.yaml", "wonik_allegro.yaml", "shadow_hand.yaml"):
        spec = RobotSpec.from_config(name, sample_anchors=False)
        B = 6
        J = len(spec.actuated_joint_names)
        palm_pos = torch.randn(B, 3, dtype=torch.float32)
        palm_rot = torch.eye(3, dtype=torch.float32).unsqueeze(0).expand(B, 3, 3)
        joints = torch.randn(B, J, dtype=torch.float32) * 0.1

        t_batch = spec.forward_kinematics(palm_pos, palm_rot, joints)
        tips_batch = spec.fingertip_positions(palm_pos, palm_rot, joints)

        if tips_batch.shape != (B, len(spec.fingertip_links), 3):
            problems.append(
                f"{name}: fingertip positions shape {tips_batch.shape} != ({B}, {len(spec.fingertip_links)}, 3)"
            )

        for b in range(B):
            t_single = spec.forward_kinematics(palm_pos[b : b + 1], palm_rot[b : b + 1], joints[b : b + 1])
            tips_single = spec.fingertip_positions(palm_pos[b : b + 1], palm_rot[b : b + 1], joints[b : b + 1])
            for link_name, mat in t_single.items():
                diff = (t_batch[link_name][b : b + 1] - mat).abs().max().item()
                if diff > 1e-4:
                    problems.append(f"{name} link {link_name} batch vs single FK diff {diff} > 1e-4")
            diff_tips = (tips_batch[b : b + 1] - tips_single).abs().max().item()
            if diff_tips > 1e-4:
                problems.append(f"{name} fingertip batch vs single diff {diff_tips} > 1e-4")


def check_hand_graph_memory_scaling(problems: list[str]) -> None:
    # Build HandGraph for hands of different sizes (LEAP 18 nodes, Allegro 22 nodes, Shadow 26 nodes)
    spec_leap = RobotSpec.from_config("leap_hand.yaml", sample_anchors=False)
    g_leap = spec_leap.to_hand_graph()

    spec_shadow = RobotSpec.from_config("shadow_hand.yaml", sample_anchors=False)
    g_shadow = spec_shadow.to_hand_graph()

    mem_leap = g_leap.memory_bytes()
    mem_shadow = g_shadow.memory_bytes()

    ratio_nodes = g_shadow.num_nodes / g_leap.num_nodes  # 26 / 18 = 1.44
    ratio_mem = mem_shadow / mem_leap  # Should scale ~ 1.44x, NOT (1.44)^2 = 2.08x

    if ratio_mem > ratio_nodes * 1.5:
        problems.append(
            f"HandGraph memory scaling is superlinear: node ratio {ratio_nodes:.2f} vs mem ratio {ratio_mem:.2f} (possible NxN dense expansion)"
        )


def check_mujoco_and_fixtures(problems: list[str], root: Path) -> None:
    # 1. mj_forward on all 3 MJCF models
    for name, subpath in (
        ("leap_hand", "mujoco-menagerie/leap_hand/right_hand.xml"),
        ("wonik_allegro", "mujoco-menagerie/wonik_allegro/right_hand.xml"),
        ("shadow_hand", "mujoco-menagerie/shadow_hand/right_hand.xml"),
    ):
        xml_path = root / ".references/robot-assets" / subpath
        try:
            sim = MujocoSim(xml_path)
            sim.forward()
        except Exception as exc:
            problems.append(f"{name} MuJoCo forward pass failed: {exc}")

    # 2. evaluate_grasp_fixture repeatability test
    leap_xml = root / ".references/robot-assets/mujoco-menagerie/leap_hand/right_hand.xml"
    res1 = evaluate_grasp_fixture(leap_xml, seed=42)
    res2 = evaluate_grasp_fixture(leap_xml, seed=42)

    if res1.metrics != res2.metrics or res1.success != res2.success:
        problems.append("evaluate_grasp_fixture is not deterministic across identical seeds")


def check_provenance_and_release_enforcement(problems: list[str]) -> None:
    # Published profiles must pass release validation
    for name in ("leap_hand.yaml", "wonik_allegro.yaml", "shadow_hand.yaml"):
        cfg = load_robot_config(name)
        try:
            validate_profile_for_release(cfg)
        except Exception as exc:
            problems.append(f"published profile {name} failed release validation: {exc}")

    # Blocked profile must be rejected
    blocked_cfg = RobotConfigV2(
        schema="qdgrasp/robot/v2",
        name="barrett_fixture",
        format="urdf",
        source_asset=".references/robot-assets/dex-urdf/robots/hands/barrett_hand/bhand_model.urdf",
        palm_link="base_link",
        joints=("j1",),
        joint_limits={"j1": (-1.0, 1.0)},
        release_blocked=True,
        provenance={"restriction_reason": "blocked_pending_rightsholder_license"},
    )
    try:
        validate_profile_for_release(blocked_cfg)
    except ConfigError:
        pass
    else:
        problems.append("validate_profile_for_release failed to reject a release_blocked=True profile")


def main() -> int:
    logging.basicConfig(level=logging.WARNING)
    root = Path(__file__).resolve().parents[1]
    problems: list[str] = []

    check_robot_assets_lock(problems, root)
    check_schema_round_trip_and_migration(problems)
    check_urdf_and_mjcf_parsing(problems, root)
    check_mesh_resolution(problems, root)
    check_normalization_reproducibility(problems, root)
    check_semantic_link_negative_tests(problems)
    check_forward_kinematics_ground_truth(problems, root)
    check_forward_kinematics_and_batch_parity(problems)
    check_hand_graph_memory_scaling(problems)
    check_mujoco_and_fixtures(problems, root)
    check_provenance_and_release_enforcement(problems)

    print(f"Phase 2 CPU Robot Layer: {'PASS' if not problems else 'FAIL'}")
    for problem in problems:
        print(f"- {problem}")
    return 0 if not problems else 1


if __name__ == "__main__":
    raise SystemExit(main())
