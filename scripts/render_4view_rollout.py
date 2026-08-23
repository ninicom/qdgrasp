"""Multi-angle (4-view) grasp rollout recorder with DexGraspNet 2.0 WidthMapper integration.

Renders 4 synchronized virtual camera perspectives:
  1. Isometric View (45 deg)
  2. Front View (0 deg)
  3. Side Profile View (90 deg)
  4. Top-Down View (-85 deg)

Outputs categorized videos into distinct `pass/` and `fail/` directories with dynamic HUD overlay.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Sequence

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from qdgrasp.objects.schema import SubGeomSpec
from qdgrasp.robot.spec import RobotSpec, resolve_robot_asset
from qdgrasp.dataset.pipeline.validators.mujoco_rollout import build_rollout_scene_model
from qdgrasp.dataset.pipeline.proposals.width_mapper import WidthMapper, compute_canonical_grasp_frame

try:
    import imageio
    HAS_IMAGEIO = True
except ImportError:
    HAS_IMAGEIO = False

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

try:
    from PIL import Image, ImageDraw
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


def smoothstep(t: float) -> float:
    """Standard smoothstep polynomial: 3*t^2 - 2*t^3 for t in [0, 1]."""
    t_c = np.clip(t, 0.0, 1.0)
    return float(3.0 * t_c**2 - 2.0 * t_c**3)


def compute_root_pose_for_target_palm(
    model: mujoco.MjModel,
    palm_body_name: str,
    p_palm_target: np.ndarray,
    R_palm_target: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute (p_root, quat_root, R_root) so the palm body is exactly at (p_palm_target, R_palm_target)."""
    palm_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, palm_body_name)
    if palm_id < 0:
        palm_id = 0

    d_temp = mujoco.MjData(model)
    d_temp.qpos[0:3] = [0.0, 0.0, 0.0]
    d_temp.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    mujoco.mj_kinematics(model, d_temp)

    p_palm_in_root = d_temp.xpos[palm_id].copy()
    R_palm_in_root = d_temp.xmat[palm_id].reshape(3, 3).copy()

    R_root = R_palm_target @ R_palm_in_root.T
    p_root = p_palm_target - R_root @ p_palm_in_root

    rot_obj = Rotation.from_matrix(R_root)
    q_xyzw = rot_obj.as_quat()
    quat_root = np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]], dtype=np.float64)
    return p_root, quat_root, R_root


class MultiViewGraspRenderer:
    """Renders 4 synchronized camera viewports in MuJoCo for dexterous grasp analysis."""

    def __init__(
        self,
        model: mujoco.MjModel,
        width: int = 480,
        height: int = 360,
    ) -> None:
        self.model = model
        self.data = mujoco.MjData(model)
        self.width = width
        self.height = height
        self.renderer = mujoco.Renderer(model, height=height, width=width)

        # 4 Virtual Cameras
        self.camera_configs = [
            {"name": "Isometric (45°)", "azimuth": 45.0, "elevation": -22.0, "distance": 0.42, "lookat": [0.0, 0.0, 0.07]},
            {"name": "Front View (0°)", "azimuth": 0.0, "elevation": -12.0, "distance": 0.40, "lookat": [0.0, 0.0, 0.07]},
            {"name": "Side Profile (90°)", "azimuth": 90.0, "elevation": -12.0, "distance": 0.40, "lookat": [0.0, 0.0, 0.07]},
            {"name": "Top-Down (-85°)", "azimuth": 0.0, "elevation": -85.0, "distance": 0.42, "lookat": [0.0, 0.0, 0.05]},
        ]

    def render_4views(self) -> list[np.ndarray]:
        """Render synchronized images from all 4 virtual cameras."""
        camera = mujoco.MjvCamera()
        frames = []
        for cfg in self.camera_configs:
            camera.type = mujoco.mjtCamera.mjCAMERA_FREE
            camera.azimuth = cfg["azimuth"]
            camera.elevation = cfg["elevation"]
            camera.distance = cfg["distance"]
            camera.lookat = np.array(cfg["lookat"], dtype=np.float64)

            self.renderer.update_scene(self.data, camera=camera)
            frame = self.renderer.render()
            frames.append(frame.copy())
        return frames

    def create_2x2_grid(
        self,
        views: list[np.ndarray],
        header_title: str = "",
        sub_info: str = "",
        phase_badge: str = "",
    ) -> np.ndarray:
        """Compose 4 views into a 2x2 grid with top HUD banner."""
        top_row = np.hstack([views[0], views[1]])
        bot_row = np.hstack([views[2], views[3]])
        grid = np.vstack([top_row, bot_row])

        if HAS_PIL:
            composite_img = Image.fromarray(grid)
            draw = ImageDraw.Draw(composite_img)

            # Top HUD banner
            banner_h = 44
            draw.rectangle([(0, 0), (grid.shape[1], banner_h)], fill=(20, 20, 26))

            # Viewport divider lines
            mid_x = grid.shape[1] // 2
            mid_y = grid.shape[0] // 2
            draw.line([(mid_x, banner_h), (mid_x, grid.shape[0])], fill=(60, 60, 75), width=2)
            draw.line([(0, mid_y), (grid.shape[1], mid_y)], fill=(60, 60, 75), width=2)

            # Header text
            draw.text((16, 12), header_title, fill=(240, 240, 245))
            if sub_info:
                draw.text((mid_x - 80, 12), sub_info, fill=(160, 160, 180))

            # Dynamic Phase Badge
            badge_color = (34, 197, 94) if "PASS" in phase_badge or "SUCCESS" in phase_badge else (
                (239, 68, 68) if "FAIL" in phase_badge or "SLIP" in phase_badge else (59, 130, 246)
            )
            draw.rectangle([(grid.shape[1] - 220, 8), (grid.shape[1] - 16, 36)], fill=badge_color)
            draw.text((grid.shape[1] - 210, 14), phase_badge, fill=(255, 255, 255))

            # Viewport corner labels
            labels = ["ISOMETRIC (45°)", "FRONT (0°)", "SIDE (90°)", "TOP-DOWN (-85°)"]
            positions = [
                (12, banner_h + 8),
                (mid_x + 12, banner_h + 8),
                (12, mid_y + 8),
                (mid_x + 12, mid_y + 8),
            ]
            for pos, lbl in zip(positions, labels):
                draw.rectangle([(pos[0] - 4, pos[1] - 2), (pos[0] + len(lbl) * 8 + 8, pos[1] + 16)], fill=(0, 0, 0, 180))
                draw.text(pos, lbl, fill=(220, 220, 230))

            return np.array(composite_img)

        return grid


def record_grasp_rollout_video(
    hand_xml_path: str | Path,
    collision_geoms: Sequence[SubGeomSpec],
    q_target: np.ndarray,
    palm_pos_target: np.ndarray,
    palm_rot_target: np.ndarray,
    palm_link_name: str,
    output_video_path: str | Path,
    robot_name: str = "leap_hand",
    object_name: str = "prim_box_01",
    fps: int = 30,
    subsample: int = 8,
    kp_gain: float = 8.0,
    friction: float = 1.0,
) -> bool:
    """Run full physical pinch & lift simulation and record a 4-view MP4 video."""
    out_p = Path(output_video_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    model = build_rollout_scene_model(
        hand_xml_path=hand_xml_path,
        collision_geoms=collision_geoms,
        object_pos=(0.0, 0.0, 0.05),
        object_mass=0.08,
    )
    # Configure actuator stiffness and friction
    for i in range(model.nu):
        model.actuator_gainprm[i, 0] = kp_gain
        model.actuator_biasprm[i, 1] = -kp_gain

    for i in range(model.ngeom):
        if "object_subgeom" in (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or ""):
            model.geom_friction[i, 0] = friction

    renderer = MultiViewGraspRenderer(model, width=480, height=360)
    data = renderer.data

    # Setup initial state
    mujoco.mj_resetData(model, data)
    obj_jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "object_freejoint")
    obj_qpos_adr = model.jnt_qposadr[obj_jnt_id]
    mocap_id = model.body("hand_mocap").mocapid[0]

    # Standoff pose: Retracted 8cm away from object center along approach vector
    approach_vec = palm_pos_target - np.array([0.0, 0.0, 0.05])
    app_norm = np.linalg.norm(approach_vec)
    if app_norm < 1e-4:
        approach_vec = np.array([0.0, 0.0, 1.0])
        app_norm = 1.0
    standoff_dir = approach_vec / app_norm
    palm_pos_standoff = palm_pos_target + 0.08 * standoff_dir

    p_root_standoff, quat_root_standoff, _ = compute_root_pose_for_target_palm(
        model, palm_link_name, palm_pos_standoff, palm_rot_target
    )
    p_root_target, quat_root_target, _ = compute_root_pose_for_target_palm(
        model, palm_link_name, palm_pos_target, palm_rot_target
    )

    # Spawn hand in collision-free standoff pose at t=0
    data.mocap_pos[mocap_id] = p_root_standoff
    data.mocap_quat[mocap_id] = quat_root_standoff
    data.qpos[0:3] = p_root_standoff
    data.qpos[3:7] = quat_root_standoff
    data.qpos[obj_qpos_adr : obj_qpos_adr + 3] = [0.0, 0.0, 0.05]
    data.qpos[obj_qpos_adr + 3 : obj_qpos_adr + 7] = [1.0, 0.0, 0.0, 0.0]

    for i in range(model.nu):
        data.ctrl[i] = 0.0

    mujoco.mj_forward(model, data)

    frames = []
    total_sim_time = 1.8
    dt = model.opt.timestep
    total_steps = int(total_sim_time / dt)

    init_obj_z = 0.05
    lift_height = 0.10
    floor_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    final_success = False

    for step in range(total_steps):
        t = step * dt

        if t < 0.35:
            # Phase 1: Approach
            alpha = smoothstep(t / 0.35)
            p_cur = (1 - alpha) * p_root_standoff + alpha * p_root_target
            data.mocap_pos[mocap_id] = p_cur
            data.mocap_quat[mocap_id] = quat_root_target
            for i in range(model.nu):
                data.ctrl[i] = 0.0
            phase = "PHASE 1: APPROACH"

        elif t < 0.85:
            # Phase 2: Finger Pinch / Width Closure
            data.mocap_pos[mocap_id] = p_root_target
            data.mocap_quat[mocap_id] = quat_root_target
            alpha = smoothstep((t - 0.35) / 0.50)
            for i in range(min(len(q_target), model.nu)):
                data.ctrl[i] = alpha * q_target[i]
            phase = "PHASE 2: WIDTH CLOSURE"

        elif t < 1.45:
            # Phase 3: Zero-Support Lift
            if floor_geom_id >= 0 and model.geom_pos[floor_geom_id][2] > -5.0:
                model.geom_pos[floor_geom_id][2] = -10.0

            alpha = smoothstep((t - 0.85) / 0.60)
            p_cur = p_root_target.copy()
            p_cur[2] += alpha * lift_height
            data.mocap_pos[mocap_id] = p_cur
            for i in range(min(len(q_target), model.nu)):
                data.ctrl[i] = q_target[i]
            phase = "PHASE 3: ZERO-SUPPORT LIFT"

        else:
            # Phase 4: Hold & Evaluation
            p_cur = p_root_target.copy()
            p_cur[2] += lift_height
            data.mocap_pos[mocap_id] = p_cur
            for i in range(min(len(q_target), model.nu)):
                data.ctrl[i] = q_target[i]

            cur_obj_z = data.qpos[obj_qpos_adr + 2]
            final_success = bool(cur_obj_z > (init_obj_z + 0.04))
            phase = "PHASE 4: CERTIFIED PASS" if final_success else "PHASE 4: FAILED / SLIPPED"

        mujoco.mj_step(model, data)

        if step % subsample == 0:
            views = renderer.render_4views()
            header = f"ROBOT: {robot_name.upper()}  |  TARGET: {object_name}"
            sub = f"Time: {t:4.2f}s | Step: {step:04d}"
            frame = renderer.create_2x2_grid(
                views,
                header_title=header,
                sub_info=sub,
                phase_badge=phase,
            )
            frames.append(frame)

    if len(frames) == 0:
        return False, False

    # Encode video
    encoded = False
    if HAS_IMAGEIO:
        try:
            imageio.mimwrite(str(out_p), frames, fps=fps, quality=8, macro_block_size=1)
            encoded = True
        except Exception:
            pass

    if not encoded and HAS_CV2:
        try:
            h, w = frames[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(out_p), fourcc, float(fps), (w, h))
            for f in frames:
                writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
            writer.release()
            encoded = True
        except Exception:
            pass

    return encoded, final_success


def run_kaggle_video_suite(
    output_dir: str | Path = "/kaggle/working/videos",
    robot_assets_root: str | Path | None = None,
) -> list[dict[str, Any]]:
    """
    Execute comprehensive multi-robot 4-view grasp video suite across passing and failing scenarios,
    saving videos into distinct `pass/` and `fail/` subdirectories.
    """
    if robot_assets_root:
        os.environ["QDGRASP_ROBOT_ASSETS_ROOT"] = str(robot_assets_root)

    out_base = Path(output_dir)
    pass_dir = out_base / "pass"
    fail_dir = out_base / "fail"
    pass_dir.mkdir(parents=True, exist_ok=True)
    fail_dir.mkdir(parents=True, exist_ok=True)

    # Scenarios covering passing and intentional failure stress cases
    scenarios = [
        # --- PASSING SCENARIOS ---
        {
            "id": "pass_01_leap_box",
            "category": "pass",
            "robot": "leap_hand.yaml",
            "palm_link": "palm",
            "robot_name": "LEAP Hand (4 DoF)",
            "object_name": "prim_box_01",
            "geoms": [SubGeomSpec(type="box", size=(0.025, 0.025, 0.025), pos=(0.0, 0.0, 0.0), quat=(1.0, 0.0, 0.0, 0.0))],
            "target_width": 0.05,
            "palm_pos": np.array([-0.09, 0.0, 0.065]),
            "palm_rot": np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]]),
            "friction": 1.0,
        },
        {
            "id": "pass_02_leap_dumbbell",
            "category": "pass",
            "robot": "leap_hand.yaml",
            "palm_link": "palm",
            "robot_name": "LEAP Hand (4 DoF)",
            "object_name": "comp_dumbbell_01",
            "geoms": [
                SubGeomSpec(type="cylinder", size=(0.015, 0.035), pos=(0.0, 0.0, 0.0), quat=(1.0, 0.0, 0.0, 0.0)),
                SubGeomSpec(type="sphere", size=(0.022,), pos=(0.0, 0.0, 0.04), quat=(1.0, 0.0, 0.0, 0.0)),
                SubGeomSpec(type="sphere", size=(0.022,), pos=(0.0, 0.0, -0.04), quat=(1.0, 0.0, 0.0, 0.0)),
            ],
            "target_width": 0.03,
            "palm_pos": np.array([-0.09, 0.0, 0.065]),
            "palm_rot": np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]]),
            "friction": 1.0,
        },
        {
            "id": "pass_03_allegro_cylinder",
            "category": "pass",
            "robot": "wonik_allegro.yaml",
            "palm_link": "palm",
            "robot_name": "Wonik Allegro (4 DoF)",
            "object_name": "prim_cylinder_01",
            "geoms": [SubGeomSpec(type="cylinder", size=(0.018, 0.04), pos=(0.0, 0.0, 0.0), quat=(1.0, 0.0, 0.0, 0.0))],
            "target_width": 0.036,
            "palm_pos": np.array([-0.09, 0.0, 0.065]),
            "palm_rot": np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]]),
            "friction": 1.0,
        },
        {
            "id": "pass_04_shadow_t_shape",
            "category": "pass",
            "robot": "shadow_hand.yaml",
            "palm_link": "rh_palm",
            "robot_name": "Shadow Hand (5 DoF)",
            "object_name": "comp_t_shape_01",
            "geoms": [
                SubGeomSpec(type="box", size=(0.015, 0.015, 0.04), pos=(0.0, 0.0, 0.0), quat=(1.0, 0.0, 0.0, 0.0)),
                SubGeomSpec(type="box", size=(0.035, 0.015, 0.015), pos=(0.0, 0.0, 0.03), quat=(1.0, 0.0, 0.0, 0.0)),
            ],
            "target_width": 0.03,
            "palm_pos": np.array([-0.06, 0.0, 0.065]),
            "palm_rot": np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
            "friction": 1.0,
        },

        # --- FAILING / STRESS SCENARIOS ---
        {
            "id": "fail_01_oversized_box",
            "category": "fail",
            "robot": "leap_hand.yaml",
            "palm_link": "palm",
            "robot_name": "LEAP Hand (Oversized Width)",
            "object_name": "prim_box_huge",
            "geoms": [SubGeomSpec(type="box", size=(0.07, 0.07, 0.03), pos=(0.0, 0.0, 0.0), quat=(1.0, 0.0, 0.0, 0.0))],
            "target_width": 0.14,  # Exceeds maximum reachable hand aperture
            "palm_pos": np.array([-0.12, 0.0, 0.065]),
            "palm_rot": np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]]),
            "friction": 1.0,
        },
        {
            "id": "fail_02_low_friction_slip",
            "category": "fail",
            "robot": "wonik_allegro.yaml",
            "palm_link": "palm",
            "robot_name": "Wonik Allegro (Low Friction)",
            "object_name": "prim_sphere_slick",
            "geoms": [SubGeomSpec(type="sphere", size=(0.026,), pos=(0.0, 0.0, 0.0), quat=(1.0, 0.0, 0.0, 0.0))],
            "target_width": 0.052,
            "palm_pos": np.array([-0.09, 0.0, 0.065]),
            "palm_rot": np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]]),
            "friction": 0.02,  # Ultra-low friction causes slippage under gravity
        },
    ]

    results = []
    print(f"Starting DexGraspNet 2.0 Multi-View Grasp Video Suite (6 scenarios)...")
    for sc in scenarios:
        spec = RobotSpec.from_config(sc["robot"], sample_anchors=False)
        xml_path = resolve_robot_asset(spec.config.source_asset)

        # Compute joint targets using WidthMapper
        width_mapper = WidthMapper(spec)
        q_target, _ = width_mapper.map_width_to_qpos(sc["target_width"])

        # Decide destination folder based on scenario category
        dest_folder = pass_dir if sc["category"] == "pass" else fail_dir
        out_vid = dest_folder / f"{sc['id']}.mp4"

        print(f"  Rendering [{sc['id']}] ({sc['category'].upper()}) -> {sc['robot_name']} on {sc['object_name']}...")
        encoded, actual_success = record_grasp_rollout_video(
            hand_xml_path=xml_path,
            collision_geoms=sc["geoms"],
            q_target=q_target,
            palm_pos_target=sc["palm_pos"],
            palm_rot_target=sc["palm_rot"],
            palm_link_name=sc["palm_link"],
            output_video_path=out_vid,
            robot_name=sc["robot_name"],
            object_name=sc["object_name"],
            fps=30,
            subsample=6,
            kp_gain=8.0,
            friction=sc.get("friction", 1.0),
        )

        file_size = out_vid.stat().st_size if out_vid.exists() else 0
        status_label = "PASS" if actual_success else "FAIL"
        print(f"  -> [{status_label}] File: {out_vid.name} | Size: {file_size:,} bytes | Dir: {dest_folder.name}/")

        results.append({
            "scenario": sc["id"],
            "category": sc["category"],
            "actual_outcome": status_label,
            "robot": sc["robot_name"],
            "object": sc["object_name"],
            "video_path": str(out_vid),
            "file_size": file_size,
            "status": "SUCCESS" if encoded and file_size > 0 else "FAILED",
        })

    return results
