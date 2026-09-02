"""Render the bounded LEAP known-positive Phase 3.1 rollout.

The images are captured from the exact MuJoCo states observed by the dynamic
validator.  This script does not search poses, regenerate data, or run an
ablation.
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import zlib
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np
import torch
from scipy.spatial.transform import Rotation

from qdgrasp.dataset.pipeline.solvers.fixed_contact_dls import solve_dls_ik_batch
from qdgrasp.dataset.pipeline.validators.mujoco_rollout import validate_grasp_rollout
from qdgrasp.objects.schema import SubGeomSpec
from qdgrasp.robot.spec import RobotSpec, resolve_robot_asset

Q_CONTACT = np.array(
    [
        0.5927356227,
        -0.3791691612,
        0.6132688578,
        1.692338131,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.228141244,
        0.1354573565,
        -0.1336592733,
        1.666422321,
    ],
    dtype=np.float32,
)


def _write_rgb_png(path: Path, image: np.ndarray) -> None:
    """Write an RGB uint8 image with only the Python standard library."""
    rgb = np.ascontiguousarray(image, dtype=np.uint8)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"expected HxWx3 RGB image, got {rgb.shape}")
    height, width, _ = rgb.shape

    def chunk(kind: bytes, payload: bytes) -> bytes:
        return (
            struct.pack(">I", len(payload))
            + kind
            + payload
            + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)
        )

    scanlines = b"".join(b"\x00" + row.tobytes() for row in rgb)
    png = b"\x89PNG\r\n\x1a\n"
    png += chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
    png += chunk(b"IDAT", zlib.compress(scanlines, level=6))
    png += chunk(b"IEND", b"")
    path.write_bytes(png)


def _fixture() -> tuple[
    RobotSpec,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    float,
]:
    spec = RobotSpec.from_config("leap_hand.yaml", sample_anchors=False)
    local_contacts = spec.fingertip_positions(
        torch.zeros(1, 3),
        torch.eye(3)[None],
        torch.from_numpy(Q_CONTACT[None]),
    )[0].numpy()
    pinch_axis = local_contacts[3] - local_contacts[0]
    pinch_axis /= np.linalg.norm(pinch_axis)
    palm_rotation, _ = Rotation.align_vectors(
        np.array([[-1.0, 0.0, 0.0]]), pinch_axis[None]
    )
    palm_rot = palm_rotation.as_matrix()
    pinch_center = 0.5 * (local_contacts[0] + local_contacts[3])
    half_width = 0.5 * np.linalg.norm(local_contacts[3] - local_contacts[0])
    object_pos = np.array([0.0, 0.0, 0.02])
    palm_pos = object_pos - palm_rot @ pinch_center

    palm_pos_b = palm_pos.astype(np.float32)[None]
    palm_rot_b = palm_rot.astype(np.float32)[None]
    q_b = Q_CONTACT[None]
    contact_points = spec.fingertip_positions(
        torch.from_numpy(palm_pos_b),
        torch.from_numpy(palm_rot_b),
        torch.from_numpy(q_b),
    )[0].numpy()
    contact_axes = spec.fingertip_contact_directions(
        torch.from_numpy(palm_pos_b),
        torch.from_numpy(palm_rot_b),
        torch.from_numpy(q_b),
    )[0].numpy()

    open_contacts = contact_points.copy()
    squeeze_contacts = contact_points.copy()
    open_contacts[[0, 3]] -= 0.004 * contact_axes[[0, 3]]
    squeeze_contacts[[0, 3]] += 0.003 * contact_axes[[0, 3]]
    commands = solve_dls_ik_batch(
        spec,
        np.repeat(palm_pos_b, 2, axis=0),
        np.repeat(palm_rot_b, 2, axis=0),
        np.stack([open_contacts, squeeze_contacts]),
        np.repeat(contact_axes[None], 2, axis=0),
        init_q=np.repeat(q_b, 2, axis=0),
        max_iter=35,
        pos_tolerance=0.0007,
        normal_tolerance_dot=0.8,
        require_normal_alignment=False,
    )
    if not np.all(commands.converged):
        raise RuntimeError("known-positive command IK no longer converges")
    return (
        spec,
        palm_pos,
        palm_rot,
        object_pos,
        contact_points,
        commands.q,
        float(half_width),
    )


def render_rollout(output_dir: Path) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    spec, palm_pos, palm_rot, object_pos, contact_points, commands, half_width = (
        _fixture()
    )
    saved: list[str] = []
    renderer: mujoco.Renderer | None = None
    clean_option = mujoco.MjvOption()
    contact_option = mujoco.MjvOption()
    contact_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True

    def observe(stage: str, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        nonlocal renderer
        if renderer is None:
            renderer = mujoco.Renderer(model, height=480, width=640)
        object_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, "target_object"
        )
        lookat = np.asarray(data.xpos[object_id], dtype=np.float64).copy()
        for view, azimuth, elevation, distance, option in (
            ("overview", 88.0, -18.0, 0.40, clean_option),
            ("contact", 88.0, -8.0, 0.13, contact_option),
        ):
            camera = mujoco.MjvCamera()
            camera.type = mujoco.mjtCamera.mjCAMERA_FREE
            camera.lookat[:] = lookat
            camera.azimuth = azimuth
            camera.elevation = elevation
            camera.distance = distance
            renderer.update_scene(data, camera=camera, scene_option=option)
            frame = renderer.render()
            path = output_dir / f"leap_pinch_{stage}_{view}.png"
            _write_rgb_png(path, frame)
            saved.append(str(path.resolve()))

    result = validate_grasp_rollout(
        resolve_robot_asset(spec.config.source_asset),
        [
            SubGeomSpec(
                type="box",
                size=(half_width, 0.015, 0.02),
                pos=(0.0, 0.0, 0.0),
                quat=(1.0, 0.0, 0.0, 0.0),
            )
        ],
        spec.fingertip_links,
        palm_pos=tuple(palm_pos),
        palm_rot=palm_rot,
        initial_joint_targets=spec.expand_mimic_joint_targets(
            dict(zip(spec.actuated_joint_names, commands[0]))
        ),
        joint_targets=spec.expand_mimic_joint_targets(
            dict(zip(spec.actuated_joint_names, commands[1]))
        ),
        object_pos=tuple(object_pos),
        object_mass=0.02,
        expected_fingertip_positions=contact_points,
        fingertip_local_offsets=np.stack(
            [spec.fingertip_contact_offsets[name] for name in spec.fingertip_links]
        ),
        pregrasp_distance=0.0,
        squeeze_steps=300,
        stage_observer=observe,
    )
    if renderer is not None:
        renderer.close()
    if not result.passed:
        raise RuntimeError(
            f"known-positive rollout failed at {result.failure_stage}: "
            f"{result.trajectory_metrics}"
        )
    return {
        "passed": result.passed,
        "failure_stage": result.failure_stage,
        "metrics": result.trajectory_metrics,
        "images": saved,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("docs/reports/evidence/phase3_1_pose"),
    )
    args = parser.parse_args()
    print(json.dumps(render_rollout(args.output_dir), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
