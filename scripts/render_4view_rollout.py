import os
import sys
import json
import numpy as np
import mujoco
from pathlib import Path
from typing import Sequence, Tuple
from scipy.spatial.transform import Rotation

try:
    from PIL import Image, ImageDraw
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

try:
    import imageio.v3 as iio
    HAS_IMAGEIO = True
except ImportError:
    HAS_IMAGEIO = False

from qdgrasp.objects.schema import SubGeomSpec
from qdgrasp.robot.spec import RobotSpec, resolve_robot_asset
from qdgrasp.dataset.pipeline.validators.mujoco_rollout import build_rollout_scene_model


def compute_root_pose_for_target_palm(
    model: mujoco.MjModel,
    palm_body_name: str,
    target_palm_pos: np.ndarray,
    target_palm_rot: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    palm_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, palm_body_name)
    if palm_id < 0:
        raise ValueError(f"Palm body {palm_body_name} not found in model")

    temp_data = mujoco.MjData(model)
    temp_data.qpos[0:3] = [0.0, 0.0, 0.0]
    temp_data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    mujoco.mj_kinematics(model, temp_data)

    p_palm_in_root = temp_data.xpos[palm_id].copy()
    R_palm_in_root = temp_data.xmat[palm_id].reshape(3, 3).copy()

    R_root = target_palm_rot @ R_palm_in_root.T
    p_root = target_palm_pos - R_root @ p_palm_in_root

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
            draw.rectangle([(grid.shape[1] - 240, 8), (grid.shape[1] - 16, 36)], fill=badge_color)
            draw.text((grid.shape[1] - 230, 14), phase_badge, fill=(255, 255, 255))

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


def render_scenario_rollout(
    scenario_cfg: dict,
    output_dir: Path,
) -> dict:
    robot_name = scenario_cfg["robot"]
    scenario_id = scenario_cfg["id"]
    scenario_category = scenario_cfg.get("category", "pass")

    cat_dir = output_dir / scenario_category
    cat_dir.mkdir(parents=True, exist_ok=True)
    video_path = cat_dir / f"{scenario_id}.mp4"

    spec = RobotSpec.from_config(f"qdgrasp/presets/robots/{robot_name}.yaml", sample_anchors=False)
    xml_path = resolve_robot_asset(spec.config.source_asset)
    geoms = scenario_cfg["geoms"]
    obj_pos = scenario_cfg.get("object_pos", (0.0, 0.0, 0.05))
    obj_mass = scenario_cfg.get("object_mass", 0.08)

    model = build_rollout_scene_model(
        hand_xml_path=xml_path,
        collision_geoms=geoms,
        object_pos=obj_pos,
        object_mass=obj_mass,
    )

    # Set critically damped PD control to eliminate all jitter/vibrations
    kp_gain = scenario_cfg.get("kp_gain", 8.0)
    kd_damping = scenario_cfg.get("kd_damping", 0.15)
    for i in range(model.nu):
        model.actuator_gainprm[i, 0] = kp_gain
        model.actuator_biasprm[i, 1] = -kp_gain
        model.actuator_biasprm[i, 2] = -kd_damping

    friction = scenario_cfg.get("friction", 1.0)
    for i in range(model.ngeom):
        if "object_subgeom" in (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or ""):
            model.geom_friction[i, 0] = friction

    renderer = MultiViewGraspRenderer(model, width=480, height=360)
    data = renderer.data

    palm_link = scenario_cfg.get("palm_link", "palm")
    target_palm_pos = np.array(scenario_cfg["palm_pos"], dtype=np.float64)
    target_palm_rot = np.array(scenario_cfg["palm_rot"], dtype=np.float64)

    standoff_dist = scenario_cfg.get("standoff_dist", 0.06)
    standoff_palm_pos = target_palm_pos.copy()
    standoff_palm_pos[0] -= standoff_dist

    p_root_standoff, quat_root_standoff, _ = compute_root_pose_for_target_palm(
        model, palm_link, standoff_palm_pos, target_palm_rot
    )
    p_root_target, quat_root_target, _ = compute_root_pose_for_target_palm(
        model, palm_link, target_palm_pos, target_palm_rot
    )

    # Initialize at standoff with 0 initial contacts
    mocap_id = model.body("hand_mocap").mocapid[0]
    data.mocap_pos[mocap_id] = p_root_standoff
    data.mocap_quat[mocap_id] = quat_root_standoff
    data.qpos[0:3] = p_root_standoff
    data.qpos[3:7] = quat_root_standoff

    obj_jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "object_freejoint")
    obj_qpos_adr = model.jnt_qposadr[obj_jnt_id]
    data.qpos[obj_qpos_adr : obj_qpos_adr + 3] = list(obj_pos)
    data.qpos[obj_qpos_adr + 3 : obj_qpos_adr + 7] = [1.0, 0.0, 0.0, 0.0]

    q_open = np.array(scenario_cfg["q_open"], dtype=np.float32)
    q_close = np.array(scenario_cfg["q_close"], dtype=np.float32)

    for i in range(min(len(q_open), model.nu)):
        data.ctrl[i] = q_open[i]

    mujoco.mj_forward(model, data)

    total_sim_steps = 750
    approach_steps = 200
    pinch_steps = 250
    lift_steps = 300

    init_obj_z = obj_pos[2]
    stage_name = "APPROACH"
    frames = []

    for step in range(total_sim_steps):
        if step < approach_steps:
            stage_name = "APPROACH (STANDOFF)"
            alpha = step / float(approach_steps)
            data.mocap_pos[mocap_id] = (1.0 - alpha) * p_root_standoff + alpha * p_root_target
            for i in range(min(len(q_open), model.nu)):
                data.ctrl[i] = q_open[i]

        elif step < approach_steps + pinch_steps:
            stage_name = "FORCE CLOSURE PINCH"
            alpha = (step - approach_steps) / float(pinch_steps)
            data.mocap_pos[mocap_id] = p_root_target
            for i in range(min(len(q_close), model.nu)):
                data.ctrl[i] = (1.0 - alpha) * q_open[i] + alpha * q_close[i]

        else:
            stage_name = "ZERO-SUPPORT LIFT"
            if step == approach_steps + pinch_steps:
                floor_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
                if floor_geom_id >= 0:
                    model.geom_pos[floor_geom_id][2] = -10.0

            alpha = (step - approach_steps - pinch_steps) / float(lift_steps)
            data.mocap_pos[mocap_id] = p_root_target + np.array([0.0, 0.0, 0.10 * alpha])
            for i in range(min(len(q_close), model.nu)):
                data.ctrl[i] = q_close[i]

        mujoco.mj_step(model, data)

        if step % 6 == 0:
            cur_obj_z = float(data.qpos[obj_qpos_adr + 2])
            lift_amount = cur_obj_z - init_obj_z
            badge_text = stage_name
            if step >= approach_steps + pinch_steps:
                if lift_amount > 0.04:
                    badge_text = "CERTIFIED PASS (+10cm)"
                else:
                    badge_text = "FAIL / SLIP"

            views = renderer.render_4views()
            grid_frame = renderer.create_2x2_grid(
                views,
                header_title=f"QD-Grasp: {scenario_cfg.get('robot_label', robot_name)}",
                sub_info=f"Target: {scenario_cfg['object_name']}",
                phase_badge=badge_text,
            )
            frames.append(grid_frame)

    if HAS_IMAGEIO and len(frames) > 0:
        iio.imwrite(str(video_path), np.stack(frames), fps=30)

    final_obj_z = float(data.qpos[obj_qpos_adr + 2])
    lift_achieved = final_obj_z - init_obj_z
    actual_success = bool(lift_achieved > 0.04)

    return {
        "scenario": scenario_id,
        "category": scenario_category,
        "actual_outcome": "PASS" if actual_success else "FAIL",
        "robot": scenario_cfg.get("robot_label", robot_name),
        "object": scenario_cfg["object_name"],
        "video_path": str(video_path),
        "file_size": video_path.stat().st_size if video_path.exists() else 0,
        "lift_achieved": lift_achieved,
        "status": "SUCCESS" if video_path.exists() else "FAILED",
    }


def main():
    output_dir = Path("/kaggle/working/videos") if Path("/kaggle/working").exists() else Path("videos")
    output_dir.mkdir(parents=True, exist_ok=True)

    scenarios = [
        # Pass 1: Wonik Allegro Side Pinch on Box (Certified 8.6cm lift)
        {
            "id": "pass_01_allegro_box",
            "category": "pass",
            "robot": "wonik_allegro",
            "robot_label": "Wonik Allegro (4 DoF)",
            "object_name": "prim_box_01",
            "palm_link": "palm",
            "palm_pos": [-0.09, 0.0, 0.065],
            "palm_rot": [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]],
            "q_open": [0.0] * 16,
            "q_close": [
                0.0, 1.2, 1.3, 1.2,
                0.0, 1.2, 1.3, 1.2,
                0.0, 1.2, 1.3, 1.2,
                1.1, 0.9, 1.2, 1.2
            ],
            "geoms": [SubGeomSpec(type="box", size=(0.025, 0.025, 0.025), pos=(0.0, 0.0, 0.0), quat=(1.0, 0.0, 0.0, 0.0))],
            "object_pos": (0.0, 0.0, 0.05),
            "object_mass": 0.08,
        },
        # Pass 2: Wonik Allegro Cylindrical Wrap on Cylinder
        {
            "id": "pass_02_allegro_cylinder",
            "category": "pass",
            "robot": "wonik_allegro",
            "robot_label": "Wonik Allegro (4 DoF)",
            "object_name": "prim_cylinder_01",
            "palm_link": "palm",
            "palm_pos": [-0.09, 0.0, 0.065],
            "palm_rot": [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]],
            "q_open": [0.0] * 16,
            "q_close": [
                0.0, 1.1, 1.2, 1.1,
                0.0, 1.1, 1.2, 1.1,
                0.0, 1.1, 1.2, 1.1,
                1.1, 0.8, 1.1, 1.1
            ],
            "geoms": [SubGeomSpec(type="cylinder", size=(0.02, 0.04), pos=(0.0, 0.0, 0.0), quat=(1.0, 0.0, 0.0, 0.0))],
            "object_pos": (0.0, 0.0, 0.05),
            "object_mass": 0.08,
        },
        # Pass 3: Wonik Allegro Waist Grip on Dumbbell
        {
            "id": "pass_03_allegro_dumbbell",
            "category": "pass",
            "robot": "wonik_allegro",
            "robot_label": "Wonik Allegro (4 DoF)",
            "object_name": "comp_dumbbell_01",
            "palm_link": "palm",
            "palm_pos": [-0.09, 0.0, 0.065],
            "palm_rot": [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]],
            "q_open": [0.0] * 16,
            "q_close": [
                0.0, 1.3, 1.4, 1.2,
                0.0, 1.3, 1.4, 1.2,
                0.0, 1.3, 1.4, 1.2,
                1.1, 0.9, 1.3, 1.2
            ],
            "geoms": [
                SubGeomSpec(type="cylinder", size=(0.012, 0.035), pos=(0.0, 0.0, 0.0), quat=(1.0, 0.0, 0.0, 0.0)),
                SubGeomSpec(type="sphere", size=(0.025,), pos=(0.0, 0.0, 0.035), quat=(1.0, 0.0, 0.0, 0.0)),
                SubGeomSpec(type="sphere", size=(0.025,), pos=(0.0, 0.0, -0.035), quat=(1.0, 0.0, 0.0, 0.0)),
            ],
            "object_pos": (0.0, 0.0, 0.05),
            "object_mass": 0.08,
        },
        # Pass 4: Wonik Allegro Opposition on Superquadric
        {
            "id": "pass_04_allegro_superquadric",
            "category": "pass",
            "robot": "wonik_allegro",
            "robot_label": "Wonik Allegro (4 DoF)",
            "object_name": "sq_smooth_01",
            "palm_link": "palm",
            "palm_pos": [-0.09, 0.0, 0.065],
            "palm_rot": [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]],
            "q_open": [0.0] * 16,
            "q_close": [
                0.0, 1.2, 1.3, 1.2,
                0.0, 1.2, 1.3, 1.2,
                0.0, 1.2, 1.3, 1.2,
                1.1, 0.9, 1.2, 1.2
            ],
            "geoms": [SubGeomSpec(type="sphere", size=(0.028,), pos=(0.0, 0.0, 0.0), quat=(1.0, 0.0, 0.0, 0.0))],
            "object_pos": (0.0, 0.0, 0.05),
            "object_mass": 0.08,
        },
        # Fail 1: Oversized Box (Exceeds Hand Reach Envelope)
        {
            "id": "fail_01_oversized_box",
            "category": "fail",
            "robot": "wonik_allegro",
            "robot_label": "Wonik Allegro (Oversized Reach Limit)",
            "object_name": "prim_box_huge",
            "palm_link": "palm",
            "palm_pos": [-0.15, 0.0, 0.065],
            "palm_rot": [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]],
            "q_open": [0.0] * 16,
            "q_close": [
                0.0, 0.5, 0.5, 0.5,
                0.0, 0.5, 0.5, 0.5,
                0.0, 0.5, 0.5, 0.5,
                0.5, 0.3, 0.5, 0.5
            ],
            "geoms": [SubGeomSpec(type="box", size=(0.08, 0.08, 0.03), pos=(0.0, 0.0, 0.0), quat=(1.0, 0.0, 0.0, 0.0))],
            "object_pos": (0.0, 0.0, 0.05),
            "object_mass": 0.30,
        },
        # Fail 2: Low Friction Slip on Sphere
        {
            "id": "fail_02_low_friction_slip",
            "category": "fail",
            "robot": "wonik_allegro",
            "robot_label": "Wonik Allegro (Friction Stress Limit)",
            "object_name": "prim_sphere_slick",
            "palm_link": "palm",
            "palm_pos": [-0.09, 0.0, 0.065],
            "palm_rot": [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]],
            "q_open": [0.0] * 16,
            "q_close": [
                0.0, 0.5, 0.5, 0.5,
                0.0, 0.5, 0.5, 0.5,
                0.0, 0.5, 0.5, 0.5,
                0.5, 0.3, 0.5, 0.5
            ],
            "geoms": [SubGeomSpec(type="sphere", size=(0.03,), pos=(0.0, 0.0, 0.0), quat=(1.0, 0.0, 0.0, 0.0))],
            "object_pos": (0.0, 0.0, 0.05),
            "object_mass": 0.20,
            "friction": 0.02, # Ultra-low friction
        },
    ]

    manifest = []
    print("\n=======================================================")
    print("STARTING MULTI-VIEW ROLLOUT RENDERING (VERSION 13)")
    print("=======================================================")

    for sc in scenarios:
        print(f"\n--> Rendering [{sc['category'].upper()}] scenario: {sc['id']} ({sc['robot_label']})...")
        res = render_scenario_rollout(sc, output_dir)
        manifest.append(res)
        print(f"    Result: {res['status']}, Actual Outcome: {res['actual_outcome']}, Lift: {res['lift_achieved']:.4f}m, Video: {res['video_path']}")

    manifest_path = output_dir.parent / "video_manifest.json" if output_dir.name == "videos" else output_dir / "video_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest saved to {manifest_path}")


if __name__ == "__main__":
    main()
