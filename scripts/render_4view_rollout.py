"""4-Camera Multi-View Grasp Rollout Renderer.

Renders synchronized 2x2 multi-angle videos (Isometric, Front, Side, Top-down)
for dexterous hand-object grasp rollouts in MuJoCo physics simulation.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Sequence, Tuple, Any
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

try:
    import imageio
    HAS_IMAGEIO = True
except ImportError:
    HAS_IMAGEIO = False

try:
    from PIL import Image, ImageDraw, ImageFont
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

import subprocess
import shutil

from qdgrasp.objects.schema import SubGeomSpec
from qdgrasp.dataset.pipeline.validators.mujoco_rollout import build_rollout_scene_model, smoothstep
from qdgrasp.robot.spec import RobotSpec
from qdgrasp.robot.assets import resolve_robot_asset


class MultiViewGraspRenderer:
    """Renderer that captures 4 synchronized camera viewpoints in MuJoCo and exports 2x2 grid MP4."""

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

        # Configure 4 virtual cameras
        # 1. Isometric (45 deg)
        self.cam_iso = mujoco.MjvCamera()
        self.cam_iso.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.cam_iso.azimuth = 45.0
        self.cam_iso.elevation = -25.0
        self.cam_iso.distance = 0.35

        # 2. Front View (0 deg)
        self.cam_front = mujoco.MjvCamera()
        self.cam_front.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.cam_front.azimuth = 0.0
        self.cam_front.elevation = -15.0
        self.cam_front.distance = 0.35

        # 3. Side / Profile View (90 deg)
        self.cam_side = mujoco.MjvCamera()
        self.cam_side.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.cam_side.azimuth = 90.0
        self.cam_side.elevation = -15.0
        self.cam_side.distance = 0.35

        # 4. Top-Down View (-85 deg)
        self.cam_top = mujoco.MjvCamera()
        self.cam_top.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.cam_top.azimuth = 0.0
        self.cam_top.elevation = -85.0
        self.cam_top.distance = 0.35

    def _update_camera_lookat(self, target_pos: np.ndarray) -> None:
        for cam in (self.cam_iso, self.cam_front, self.cam_side, self.cam_top):
            cam.lookat[:] = target_pos

    def render_4view_frame(
        self,
        robot_name: str,
        object_name: str,
        phase_label: str,
        sim_time: float,
    ) -> np.ndarray:
        """Render 4 camera views and stitch into a 2x2 grid with HUD annotations."""
        obj_bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "target_object")
        if obj_bid >= 0:
            target_pos = self.data.xpos[obj_bid].copy()
        else:
            target_pos = np.array([0.0, 0.0, 0.05])
        self._update_camera_lookat(target_pos)

        # Render 4 views
        self.renderer.update_scene(self.data, camera=self.cam_iso)
        img_iso = self.renderer.render()

        self.renderer.update_scene(self.data, camera=self.cam_front)
        img_front = self.renderer.render()

        self.renderer.update_scene(self.data, camera=self.cam_side)
        img_side = self.renderer.render()

        self.renderer.update_scene(self.data, camera=self.cam_top)
        img_top = self.renderer.render()

        # Stitch 2x2
        top_row = np.hstack([img_iso, img_front])
        bot_row = np.hstack([img_side, img_top])
        grid = np.vstack([top_row, bot_row])  # (2*H, 2*W, 3)

        # Draw HUD using PIL
        if HAS_PIL:
            header_h = 44
            grid_img = Image.fromarray(grid)
            composite_img = Image.new("RGB", (grid.shape[1], grid.shape[0] + header_h), (25, 25, 28))
            composite_img.paste(grid_img, (0, header_h))
            draw = ImageDraw.Draw(composite_img)

            # Header text
            title_text = f"QDGrasp Multi-View Rollout | Robot: {robot_name.upper()} | Object: {object_name} | t={sim_time:.2f}s"
            draw.text((16, 12), title_text, fill=(255, 255, 255))

            status_color = (80, 240, 100) if "PASS" in phase_label or "LIFT" in phase_label else (255, 200, 60)
            draw.text((grid.shape[1] - 280, 12), f"[{phase_label}]", fill=status_color)

            # Viewport Badges
            w, h = self.width, self.height
            badges = [
                ("Cam 1: Isometric (45 deg)", 12, header_h + 12),
                ("Cam 2: Front View (0 deg)", w + 12, header_h + 12),
                ("Cam 3: Side Profile (90 deg)", 12, header_h + h + 12),
                ("Cam 4: Top-Down (-85 deg)", w + 12, header_h + h + 12),
            ]
            for text, x, y in badges:
                draw.rectangle([x - 4, y - 2, x + len(text) * 8 + 4, y + 16], fill=(30, 30, 30, 200))
                draw.text((x, y), text, fill=(230, 230, 230))

            return np.array(composite_img)

        return grid


def record_grasp_rollout_video(
    hand_xml_path: str | Path,
    collision_geoms: Sequence[SubGeomSpec],
    q_target: np.ndarray,
    palm_pos_target: np.ndarray,
    palm_rot_target: np.ndarray,
    output_video_path: str | Path,
    robot_name: str = "leap_hand",
    object_name: str = "prim_box_01",
    fps: int = 30,
    subsample: int = 8,
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
    renderer = MultiViewGraspRenderer(model, width=480, height=360)
    data = renderer.data

    # Setup initial state
    mujoco.mj_resetData(model, data)

    # Identify actuated joints
    hand_root = model.body(0)
    actuator_names = [model.actuator(i).name for i in range(model.nu)]

    # Object free joint
    obj_jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "object_freejoint")
    obj_qpos_adr = model.jnt_qposadr[obj_jnt_id]

    # Pre-grasp offset (approach 5cm along approach vector)
    R_target = Rotation.from_matrix(palm_rot_target)
    approach_vec = R_target.as_matrix()[:, 0]  # x-axis approach
    palm_pos_pre = palm_pos_target - 0.05 * approach_vec

    # Mocap body
    mocap_id = model.body("hand_mocap").mocapid[0]
    data.mocap_pos[mocap_id] = palm_pos_pre
    data.mocap_quat[mocap_id] = np.array([R_target.as_quat()[3], R_target.as_quat()[0], R_target.as_quat()[1], R_target.as_quat()[2]])

    # Hand free joint
    data.qpos[0:3] = palm_pos_pre
    data.qpos[3:7] = data.mocap_quat[mocap_id]
    data.qpos[obj_qpos_adr : obj_qpos_adr + 3] = [0.0, 0.0, 0.05]
    data.qpos[obj_qpos_adr + 3 : obj_qpos_adr + 7] = [1.0, 0.0, 0.0, 0.0]

    frames = []

    # Simulation Phases:
    # Phase 1: Approach (0 to 0.4s)
    # Phase 2: Close / Pinch Fingers (0.4 to 0.9s)
    # Phase 3: Zero-Support Lift Upward (0.9 to 1.5s)
    # Phase 4: Hold & Certify (1.5 to 1.8s)

    total_sim_time = 1.8
    dt = model.opt.timestep
    total_steps = int(total_sim_time / dt)

    init_obj_z = 0.05
    lift_height = 0.08  # 8cm lift

    for step in range(total_steps):
        t = step * dt

        # Stage control
        if t < 0.4:
            # Approach
            alpha = smoothstep(t / 0.4)
            data.mocap_pos[mocap_id] = (1 - alpha) * palm_pos_pre + alpha * palm_pos_target
            phase = "PHASE 1: APPROACH"
            # Hand open
            for i in range(min(len(q_target), model.nu)):
                data.ctrl[i] = 0.0
        elif t < 0.9:
            # Pinch / close fingers
            data.mocap_pos[mocap_id] = palm_pos_target
            alpha = smoothstep((t - 0.4) / 0.5)
            phase = "PHASE 2: FINGER PINCH"
            for i in range(min(len(q_target), model.nu)):
                data.ctrl[i] = alpha * q_target[i]
        elif t < 1.5:
            # Lift upward (remove floor support, raise palm by 8cm)
            alpha = smoothstep((t - 0.9) / 0.6)
            lifted_pos = palm_pos_target.copy()
            lifted_pos[2] += alpha * lift_height
            data.mocap_pos[mocap_id] = lifted_pos
            phase = "PHASE 3: ZERO-SUPPORT LIFT"
            for i in range(min(len(q_target), model.nu)):
                data.ctrl[i] = q_target[i]
        else:
            # Hold
            lifted_pos = palm_pos_target.copy()
            lifted_pos[2] += lift_height
            data.mocap_pos[mocap_id] = lifted_pos
            cur_obj_z = data.qpos[obj_qpos_adr + 2]
            success = cur_obj_z > (init_obj_z + 0.04)
            phase = "PHASE 4: CERTIFIED PASS" if success else "PHASE 4: SLIPPED"
            for i in range(min(len(q_target), model.nu)):
                data.ctrl[i] = q_target[i]

        mujoco.mj_step(model, data)

        # Capture frame at desired subsample rate (~30 FPS)
        if step % subsample == 0:
            frame = renderer.render_4view_frame(
                robot_name=robot_name,
                object_name=object_name,
                phase_label=phase,
                sim_time=t,
            )
            frames.append(frame)

    # Save video with fallback priority: imageio -> cv2 -> ffmpeg CLI -> PIL GIF
    if len(frames) == 0:
        return False

    # 1. imageio (mp4)
    if HAS_IMAGEIO:
        try:
            imageio.mimwrite(str(out_p), frames, fps=fps, quality=8, macro_block_size=1)
            return True
        except Exception:
            pass

    # 2. OpenCV VideoWriter (mp4)
    if HAS_CV2:
        try:
            h, w = frames[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(out_p), fourcc, float(fps), (w, h))
            for f in frames:
                writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
            writer.release()
            return True
        except Exception:
            pass

    # 3. ffmpeg subprocess (mp4)
    ffmpeg_bin = shutil.which("ffmpeg")
    if ffmpeg_bin:
        try:
            h, w = frames[0].shape[:2]
            cmd = [
                ffmpeg_bin,
                "-y",
                "-f", "rawvideo",
                "-vcodec", "rawvideo",
                "-s", f"{w}x{h}",
                "-pix_fmt", "rgb24",
                "-r", str(fps),
                "-i", "-",
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                "-crf", "22",
                str(out_p),
            ]
            proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            for f in frames:
                proc.stdin.write(f.tobytes())
            proc.stdin.close()
            proc.wait(timeout=30)
            return out_p.exists() and out_p.stat().st_size > 0
        except Exception:
            pass

    # 4. Fallback: Save as animated GIF or webp if MP4 failed
    if HAS_PIL:
        try:
            pil_frames = [Image.fromarray(f) for f in frames]
            gif_p = out_p.with_suffix(".gif")
            pil_frames[0].save(
                str(gif_p),
                save_all=True,
                append_images=pil_frames[1:],
                duration=int(1000 / fps),
                loop=0,
            )
            return gif_p.exists()
        except Exception:
            pass

    return False


def run_kaggle_video_suite(
    output_dir: str | Path = "/kaggle/working/videos",
    robot_assets_root: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Execute the complete multi-robot 4-view grasp video suite across 3 hands and diverse shapes."""
    if robot_assets_root:
        os.environ["QDGRASP_ROBOT_ASSETS_ROOT"] = str(robot_assets_root)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 6 Representative Test Scenarios
    scenarios = [
        {
            "id": "01_leap_hand_box",
            "robot": "leap_hand.yaml",
            "robot_name": "LEAP Hand (4 DoF)",
            "object_name": "prim_box_01",
            "geoms": [SubGeomSpec(type="box", size=(0.025, 0.025, 0.025), pos=(0.0, 0.0, 0.0), quat=(1.0, 0.0, 0.0, 0.0))],
            "palm_pos": np.array([-0.075, 0.0, 0.05]),
            "palm_rot": np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
            "q_target": np.array([0.0, 0.85, 0.85, 0.0,  0.0, 0.85, 0.85, 0.0,  0.0, 0.85, 0.85, 0.0,  0.85, 0.85, 0.0, 0.0], dtype=np.float32),
        },
        {
            "id": "02_leap_hand_dumbbell",
            "robot": "leap_hand.yaml",
            "robot_name": "LEAP Hand (4 DoF)",
            "object_name": "comp_dumbbell_01",
            "geoms": [
                SubGeomSpec(type="cylinder", size=(0.015, 0.035), pos=(0.0, 0.0, 0.0), quat=(1.0, 0.0, 0.0, 0.0)),
                SubGeomSpec(type="sphere", size=(0.022,), pos=(0.0, 0.0, 0.04), quat=(1.0, 0.0, 0.0, 0.0)),
                SubGeomSpec(type="sphere", size=(0.022,), pos=(0.0, 0.0, -0.04), quat=(1.0, 0.0, 0.0, 0.0)),
            ],
            "palm_pos": np.array([-0.072, 0.0, 0.05]),
            "palm_rot": np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
            "q_target": np.array([0.0, 0.9, 0.9, 0.0,  0.0, 0.9, 0.9, 0.0,  0.0, 0.9, 0.9, 0.0,  0.9, 0.9, 0.0, 0.0], dtype=np.float32),
        },
        {
            "id": "03_wonik_allegro_superquadric",
            "robot": "wonik_allegro.yaml",
            "robot_name": "Wonik Allegro (4 DoF)",
            "object_name": "sq_04",
            "geoms": [SubGeomSpec(type="box", size=(0.022, 0.022, 0.03), pos=(0.0, 0.0, 0.0), quat=(1.0, 0.0, 0.0, 0.0))],
            "palm_pos": np.array([-0.065, 0.0, 0.05]),
            "palm_rot": np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
            "q_target": np.array([0.0, 0.8, 0.8, 0.8,  0.0, 0.8, 0.8, 0.8,  0.0, 0.8, 0.8, 0.8,  0.8, 0.8, 0.8, 0.0], dtype=np.float32),
        },
        {
            "id": "04_wonik_allegro_cylinder",
            "robot": "wonik_allegro.yaml",
            "robot_name": "Wonik Allegro (4 DoF)",
            "object_name": "prim_cylinder_01",
            "geoms": [SubGeomSpec(type="cylinder", size=(0.018, 0.04), pos=(0.0, 0.0, 0.0), quat=(1.0, 0.0, 0.0, 0.0))],
            "palm_pos": np.array([-0.062, 0.0, 0.05]),
            "palm_rot": np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
            "q_target": np.array([0.0, 0.85, 0.85, 0.85,  0.0, 0.85, 0.85, 0.85,  0.0, 0.85, 0.85, 0.85,  0.85, 0.85, 0.85, 0.0], dtype=np.float32),
        },
        {
            "id": "05_shadow_hand_t_shape",
            "robot": "shadow_hand.yaml",
            "robot_name": "Shadow Hand (5 DoF)",
            "object_name": "comp_t_shape_01",
            "geoms": [
                SubGeomSpec(type="box", size=(0.015, 0.015, 0.04), pos=(0.0, 0.0, 0.0), quat=(1.0, 0.0, 0.0, 0.0)),
                SubGeomSpec(type="box", size=(0.035, 0.015, 0.015), pos=(0.0, 0.0, 0.03), quat=(1.0, 0.0, 0.0, 0.0)),
            ],
            "palm_pos": np.array([0.0, -0.105, 0.05]),
            "palm_rot": np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
            "q_target": np.full(20, 0.75, dtype=np.float32),
        },
        {
            "id": "06_shadow_hand_superquadric",
            "robot": "shadow_hand.yaml",
            "robot_name": "Shadow Hand (5 DoF)",
            "object_name": "sq_01",
            "geoms": [SubGeomSpec(type="sphere", size=(0.024,), pos=(0.0, 0.0, 0.0), quat=(1.0, 0.0, 0.0, 0.0))],
            "palm_pos": np.array([0.0, -0.102, 0.05]),
            "palm_rot": np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
            "q_target": np.full(20, 0.8, dtype=np.float32),
        },
    ]

    results = []
    print(f"Starting 4-View Grasp Video Suite (6 scenarios)...")
    for sc in scenarios:
        spec = RobotSpec.from_config(sc["robot"], sample_anchors=False)
        xml_path = resolve_robot_asset(spec.config.source_asset)
        out_vid = out_dir / f"{sc['id']}.mp4"

        print(f"  Rendering [{sc['id']}] -> {sc['robot_name']} on {sc['object_name']}...")
        ok = record_grasp_rollout_video(
            hand_xml_path=xml_path,
            collision_geoms=sc["geoms"],
            q_target=sc["q_target"],
            palm_pos_target=sc["palm_pos"],
            palm_rot_target=sc["palm_rot"],
            output_video_path=out_vid,
            robot_name=sc["robot_name"],
            object_name=sc["object_name"],
            fps=30,
            subsample=6,
        )
        file_size = out_vid.stat().st_size if out_vid.exists() else 0
        status = "SUCCESS" if ok and file_size > 0 else "FAILED"
        print(f"  -> {status} | Size: {file_size:,} bytes | Path: {out_vid}")
        results.append({
            "scenario": sc["id"],
            "robot": sc["robot_name"],
            "object": sc["object_name"],
            "video_path": str(out_vid),
            "file_size": file_size,
            "status": status,
        })

    return results
