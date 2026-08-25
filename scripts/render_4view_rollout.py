"""Render measured Phase 3.3 release grasps from their immutable records.

This script intentionally has no hand-authored palm pose, joint target, expected
PASS category, or standalone lift predicate. It reloads the three admitted
``QDGrasp-Scene-Tiny`` positives, verifies their identities against the release
manifest, and replays the same fail-closed physical validator that created the
labels. Video output is evidence for those bounded release controls, not a claim
that the general object-level generator has converged.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
from PIL import Image, ImageDraw

from qdgrasp.config.schema import ConfigError
from qdgrasp.dataset.pipeline.validators.mujoco_rollout import RolloutSceneObject
from qdgrasp.dataset.pipeline.validators.scene_rollout import run_scene_grasp_rollout
from qdgrasp.dataset.scene_manifest import load_scene_manifest
from qdgrasp.dataset.scene_shards import read_scene_shard
from qdgrasp.objects.manifest import load_object_asset
from qdgrasp.scenes.adapters import get_adapter
from qdgrasp.scenes.release_recipes import SceneGraspRecipe, build_release_grasp_recipe

REPO_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_ROBOTS = {"leap_hand.yaml", "wonik_allegro.yaml", "shadow_hand.yaml"}
REQUIRED_STAGES = ("initial", "squeeze", "lift", "perturbation")


def _load_positive_records(dataset_root: Path) -> list[dict[str, Any]]:
    manifest = load_scene_manifest(dataset_root / "scene_manifest.json")
    if manifest.release_blocked or manifest.invalidated:
        raise ConfigError("scene video suite requires an admitted, non-invalidated release")
    records: list[dict[str, Any]] = []
    for shard in manifest.shards:
        if shard.record_type != "grasp":
            continue
        records.extend(
            record
            for record in read_scene_shard(
                dataset_root / shard.filename,
                record_type="grasp",
                expected_sha256=shard.sha256,
                expected_records=shard.num_records,
            )
            if bool(record.get("dynamic_valid"))
        )
    records.sort(key=lambda item: str(item["candidate_id"]))
    robots = {str(item.get("robot_profile")) for item in records}
    if len(records) != 3 or robots != EXPECTED_ROBOTS:
        raise ConfigError(
            "scene video suite requires exactly one admitted positive for LEAP, Allegro, and Shadow; "
            f"got records={len(records)}, robots={sorted(robots)}"
        )
    return records


def _assert_close(label: str, actual: Any, expected: Any, *, atol: float = 1e-9) -> None:
    left = np.asarray(actual, dtype=np.float64)
    right = np.asarray(expected, dtype=np.float64)
    if left.shape != right.shape or not np.allclose(left, right, rtol=0.0, atol=atol):
        raise ConfigError(f"release video identity mismatch for {label}")


def _assert_evidence_equal(label: str, actual: Any, expected: Any) -> None:
    if isinstance(expected, Mapping):
        if not isinstance(actual, Mapping) or set(actual) != set(expected):
            raise ConfigError(f"release video identity mismatch for {label} keys")
        for key in sorted(expected):
            _assert_evidence_equal(f"{label}.{key}", actual[key], expected[key])
        return
    if isinstance(expected, (Sequence, np.ndarray)) and not isinstance(expected, (str, bytes)):
        if (
            not isinstance(actual, (Sequence, np.ndarray))
            or isinstance(actual, (str, bytes))
            or len(actual) != len(expected)
        ):
            raise ConfigError(f"release video identity mismatch for {label} shape")
        for index, (left, right) in enumerate(zip(actual, expected, strict=True)):
            _assert_evidence_equal(f"{label}[{index}]", left, right)
        return
    if isinstance(expected, (bool, int, float)) and isinstance(actual, (bool, int, float)):
        _assert_close(label, actual, expected, atol=1e-8)
        return
    if actual != expected:
        raise ConfigError(f"release video identity mismatch for {label}")


def _verify_record_identity(
    record: Mapping[str, Any],
    recipe: SceneGraspRecipe,
    *,
    source_hash: str,
) -> None:
    if record.get("source_class") != "native_measured_release":
        raise ConfigError("video candidate is not an admitted native measured release")
    if record.get("failure_reason") != "none" or record.get("label_stage") != "dynamic_valid":
        raise ConfigError("video candidate does not carry a passing dynamic label")
    if record.get("recipe_hash") != recipe.recipe_hash:
        raise ConfigError("video candidate recipe hash does not match executable recipe")
    if record.get("protocol_hash") != recipe.protocol_hash:
        raise ConfigError("video candidate protocol hash does not match executable protocol")
    if record.get("source_hash") != source_hash:
        raise ConfigError("video candidate source hash does not match release scene")
    if not str(record.get("candidate_id", "")).endswith(f"::{recipe.recipe_id}"):
        raise ConfigError("video candidate ID does not match executable recipe")
    command = [recipe.rollout_kwargs["joint_targets"][name] for name in recipe.robot_spec.actuated_joint_names]
    _assert_close("q_command", record.get("q_command"), command, atol=1e-7)
    palm = np.eye(4, dtype=np.float64)
    palm[:3, :3] = np.asarray(recipe.rollout_kwargs["palm_rot"], dtype=np.float64)
    palm[:3, 3] = np.asarray(recipe.rollout_kwargs["palm_pos"], dtype=np.float64)
    _assert_close("palm_T_command", record.get("palm_T_command"), palm)


def _scene_objects(dataset_root: Path, scene_id: str, target_id: str) -> tuple[Any, list[RolloutSceneObject]]:
    scene = get_adapter("native").load_scene(str(dataset_root), scene_id)
    target = next((item for item in scene.objects if item.object_id == target_id), None)
    if target is None:
        raise ConfigError(f"video target is absent from release scene: {scene_id}/{target_id}")
    non_targets: list[RolloutSceneObject] = []
    for item in scene.objects:
        _, manifest = load_object_asset(Path(item.asset_ref))
        if item.object_id == target_id:
            continue
        non_targets.append(
            RolloutSceneObject(
                object_id=item.object_id,
                collision_geoms=manifest.collision_geoms,
                pos=tuple(float(value) for value in item.T_world_object[:3, 3]),
                mass=float(item.mass if item.mass is not None else manifest.mass),
            )
        )
    return target, non_targets


class MultiViewGraspRenderer:
    """Render synchronized diagnostic views from the exact validator ``MjData``."""

    def __init__(self, model: mujoco.MjModel, *, width: int = 320, height: int = 240) -> None:
        self.model = model
        self.width = width
        self.height = height
        self.renderer = mujoco.Renderer(model, height=height, width=width)
        self.camera_configs = (
            ("ISOMETRIC", 135.0, -28.0),
            ("FRONT", 90.0, -15.0),
            ("SIDE", 180.0, -15.0),
            ("TOP", 90.0, -82.0),
        )

    def close(self) -> None:
        self.renderer.close()

    def render_grid(self, data: mujoco.MjData, *, title: str, stage: str) -> np.ndarray:
        views: list[np.ndarray] = []
        target_body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "target_object")
        lookat = (
            np.asarray(data.xpos[target_body], dtype=np.float64)
            if target_body >= 0
            else np.array([0.0, 0.0, 0.08])
        )
        for label, azimuth, elevation in self.camera_configs:
            camera = mujoco.MjvCamera()
            camera.type = mujoco.mjtCamera.mjCAMERA_FREE
            camera.azimuth = azimuth
            camera.elevation = elevation
            camera.distance = 0.55
            camera.lookat[:] = lookat
            self.renderer.update_scene(data, camera=camera)
            frame = Image.fromarray(np.asarray(self.renderer.render()).copy())
            draw = ImageDraw.Draw(frame)
            draw.rectangle((0, 0, 100, 20), fill=(0, 0, 0))
            draw.text((5, 5), label, fill=(255, 255, 255))
            views.append(np.asarray(frame))
        grid = np.vstack((np.hstack((views[0], views[1])), np.hstack((views[2], views[3]))))
        image = Image.fromarray(grid)
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 0, image.width, 44), fill=(16, 16, 22))
        draw.text((10, 7), title, fill=(245, 245, 245))
        draw.text((10, 25), f"MEASURED VALIDATOR ROLLOUT | stage={stage}", fill=(120, 210, 255))
        return np.asarray(image)


class _RolloutRecorder:
    def __init__(self, candidate_id: str, *, frame_stride: int, width: int, height: int) -> None:
        if frame_stride <= 0:
            raise ConfigError("frame_stride must be positive")
        self.candidate_id = candidate_id
        self.frame_stride = frame_stride
        self.width = width
        self.height = height
        self.frames: list[np.ndarray] = []
        self._renderer: MultiViewGraspRenderer | None = None
        self._model_identity: int | None = None
        self._step = 0

    def _capture(self, stage: str, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        if not all(np.all(np.isfinite(values)) for values in (data.qpos, data.qvel, data.qacc)):
            raise ConfigError(f"non-finite MuJoCo state while rendering {self.candidate_id}/{stage}")
        warning_count = sum(int(warning.number) for warning in data.warning)
        if warning_count:
            raise ConfigError(f"MuJoCo warning while rendering {self.candidate_id}/{stage}")
        if self._renderer is None:
            self._renderer = MultiViewGraspRenderer(model, width=self.width, height=self.height)
            self._model_identity = id(model)
        elif self._model_identity != id(model):
            raise ConfigError("video recorder received multiple MuJoCo models")
        self.frames.append(self._renderer.render_grid(data, title=self.candidate_id, stage=stage))

    def observe_stage(self, stage: str, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        # Stage observers are emitted at validator endpoints. Always capture them
        # so the PASS overlay cannot land on a stride-sampled pre-final frame.
        self._capture(stage, model, data)

    def observe_step(self, stage: str, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        self._step += 1
        if self._step % self.frame_stride == 0:
            self._capture(stage, model, data)

    def finish(self, verdict: str, metrics: Mapping[str, Any]) -> list[np.ndarray]:
        if not self.frames:
            raise ConfigError(f"no frames captured for {self.candidate_id}")
        color = (20, 135, 70) if verdict == "PASS" else (180, 35, 35)
        final = Image.fromarray(self.frames[-1].copy())
        draw = ImageDraw.Draw(final)
        draw.rectangle((0, final.height - 42, final.width, final.height), fill=color)
        draw.text(
            (10, final.height - 34),
            (
                f"{verdict} | lift={float(metrics.get('measured_target_lift', 0.0)):.4f}m "
                f"| fingers={int(float(metrics.get('final_active_fingers', 0.0)))} "
                f"| palm={int(float(metrics.get('has_palm_contact', 0.0)))} "
                f"| floor={int(float(metrics.get('floor_support', 0.0)))}"
            ),
            fill=(255, 255, 255),
        )
        self.frames.extend([np.asarray(final)] * 20)
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
        return self.frames


def _write_video(path: Path, frames: Sequence[np.ndarray], *, fps: int) -> None:
    try:
        import imageio.v3 as iio
    except ImportError as exc:
        raise ConfigError("video rendering requires imageio[ffmpeg]") from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(str(path), np.stack(frames), fps=fps)
    if not path.is_file() or path.stat().st_size == 0:
        raise ConfigError(f"video writer produced no artifact: {path}")


def render_release_record(
    record: Mapping[str, Any],
    *,
    dataset_root: Path,
    output_dir: Path,
    frame_stride: int = 10,
    fps: int = 24,
    width: int = 320,
    height: int = 240,
) -> dict[str, Any]:
    manifest = load_scene_manifest(dataset_root / "scene_manifest.json")
    scene_id = str(record["scene_id"])
    source_hash = manifest.scene_spec_hashes.get(scene_id)
    if source_hash is None:
        raise ConfigError(f"video scene is absent from release manifest: {scene_id}")
    recipe = build_release_grasp_recipe(str(record["robot_profile"]))
    _verify_record_identity(record, recipe, source_hash=source_hash)
    target, non_targets = _scene_objects(dataset_root, scene_id, str(record["target_object_id"]))
    _, target_manifest = load_object_asset(Path(target.asset_ref))
    expected_target = [item.model_dump(mode="json") for item in recipe.target_geoms]
    actual_target = [item.model_dump(mode="json") for item in target_manifest.collision_geoms]
    if actual_target != expected_target:
        raise ConfigError("release video target geometry does not match executable recipe")

    recorder = _RolloutRecorder(
        str(record["candidate_id"]),
        frame_stride=frame_stride,
        width=width,
        height=height,
    )
    rollout = run_scene_grasp_rollout(
        recipe.hand_xml_path,
        recipe.target_geoms,
        recipe.robot_spec.fingertip_links,
        target_object_id=str(record["target_object_id"]),
        non_target_objects=non_targets,
        protocol_hash=recipe.protocol_hash,
        recipe_hash=recipe.recipe_hash,
        source_hash=source_hash,
        rollout_kwargs=recipe.rollout_kwargs,
        evidence_stage_observer=recorder.observe_stage,
        evidence_step_observer=recorder.observe_step,
    )
    if not rollout.validation.passed:
        raise ConfigError(
            "admitted release grasp no longer passes measured replay: "
            f"{record['candidate_id']}::{rollout.validation.failure_stage}"
        )
    if dict(rollout.state_hashes) != dict(record["scene_state_hashes"]):
        raise ConfigError(f"release video state hashes drifted for {record['candidate_id']}")
    evidence = dict(rollout.validation.trajectory_metrics)
    evidence["per_finger_loads"] = rollout.validation.per_finger_loads
    recorded_evidence = dict(record["dynamic_trajectory_evidence"])
    _assert_evidence_equal("dynamic_trajectory_evidence", evidence, recorded_evidence)
    if tuple(evidence.get("validated_stages", ())) != REQUIRED_STAGES:
        raise ConfigError(f"release video replay lacks required stages for {record['candidate_id']}")

    frames = recorder.finish("PASS", evidence)
    filename = f"{scene_id}--{Path(str(record['robot_profile'])).stem}.mp4"
    video_path = output_dir / "pass" / filename
    _write_video(video_path, frames, fps=fps)
    return {
        "scenario": str(record["candidate_id"]),
        "category": "pass",
        "actual_outcome": "PASS",
        "failure_stage": "none",
        "scene_id": scene_id,
        "target_object_id": str(record["target_object_id"]),
        "robot_profile": str(record["robot_profile"]),
        "source_class": str(record["source_class"]),
        "video_path": str(video_path),
        "file_size": video_path.stat().st_size,
        "frame_count": len(frames),
        "measured_target_lift": float(evidence["measured_target_lift"]),
        "final_active_fingers": int(float(evidence["final_active_fingers"])),
        "has_palm_contact": bool(float(evidence["has_palm_contact"])),
        "floor_support": bool(float(evidence["floor_support"])),
        "protocol_hash": recipe.protocol_hash,
        "recipe_hash": recipe.recipe_hash,
        "source_hash": source_hash,
        "scene_state_hashes": dict(rollout.state_hashes),
        "status": "SUCCESS",
    }


def run_kaggle_video_suite(
    output_dir: str | Path = "/kaggle/working/videos",
    robot_assets_root: str | None = None,
    dataset_root: str | Path | None = None,
    *,
    frame_stride: int = 10,
    fps: int = 24,
) -> list[dict[str, Any]]:
    """Replay and render the exact three admitted Phase 3.3 release controls."""
    if robot_assets_root:
        os.environ["QDGRASP_ROBOT_ASSETS_ROOT"] = str(robot_assets_root)
    root = Path(dataset_root or REPO_ROOT / "datasets" / "qdgrasp-scene-tiny").resolve()
    output = Path(output_dir).resolve()
    records = _load_positive_records(root)
    results = [
        render_release_record(
            record,
            dataset_root=root,
            output_dir=output,
            frame_stride=frame_stride,
            fps=fps,
        )
        for record in records
    ]
    if any(item["actual_outcome"] != "PASS" or item["category"] != "pass" for item in results):
        raise ConfigError("release video manifest contains a non-passing admitted positive")
    return results


def main() -> None:
    output_dir = Path("/kaggle/working/videos") if Path("/kaggle/working").exists() else Path("videos")
    results = run_kaggle_video_suite(output_dir=output_dir)
    manifest_path = output_dir.parent / "video_manifest.json"
    manifest_path.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Measured release-control manifest saved to {manifest_path}")


if __name__ == "__main__":
    main()
