"""Compile canonical scene contracts into MuJoCo models."""

from __future__ import annotations

from html import escape
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from qdgrasp.config.schema import ConfigError
from qdgrasp.objects.manifest import load_object_asset
from qdgrasp.objects.schema import ObjectManifestSpec
from qdgrasp.scenes.contracts import SceneObjectSpec, SceneSpec


def _transform_parts(transform: np.ndarray, label: str) -> tuple[np.ndarray, np.ndarray]:
    value = np.asarray(transform, dtype=np.float64)
    if value.shape != (4, 4) or not np.all(np.isfinite(value)):
        raise ConfigError(f"{label} transform must be finite and 4x4")
    rotation = value[:3, :3]
    if not np.allclose(value[3], [0.0, 0.0, 0.0, 1.0], atol=1e-8):
        raise ConfigError(f"{label} transform has invalid homogeneous row")
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6) or not np.isclose(
        np.linalg.det(rotation), 1.0, atol=1e-6
    ):
        raise ConfigError(f"{label} transform has invalid rotation")
    quat_xyzw = Rotation.from_matrix(rotation).as_quat()
    return value[:3, 3], np.array(
        [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]]
    )


def _numbers(values) -> str:
    return " ".join(f"{float(value):.17g}" for value in values)


def _object_manifest(scene_object: SceneObjectSpec) -> ObjectManifestSpec:
    asset = Path(scene_object.asset_ref).expanduser().resolve()
    candidates = [asset]
    if asset.suffix.lower() == ".obj":
        candidates.insert(0, asset.with_name(f"{asset.stem}.manifest.json"))
    manifest_path = next(
        (candidate for candidate in candidates if candidate.name.endswith(".manifest.json")),
        None,
    )
    if manifest_path is None or not manifest_path.is_file():
        raise ConfigError(
            f"scene object asset_ref must resolve to an object manifest or paired OBJ: {asset}"
        )
    _, manifest = load_object_asset(manifest_path)
    if manifest.object_id != scene_object.object_id:
        raise ConfigError(
            f"scene object ID {scene_object.object_id} does not match manifest {manifest.object_id}"
        )
    return manifest


def build_scene_mujoco_model(
    spec: SceneSpec, *, include_objects: bool = True, dynamic_objects: bool = True
) -> mujoco.MjModel:
    """Compile supports, cameras, and verified convex object assets."""
    if not math_is_positive_finite(spec.timestep):
        raise ConfigError("scene timestep must be finite and positive")
    object_ids = [scene_object.object_id for scene_object in spec.objects]
    if len(object_ids) != len(set(object_ids)):
        raise ConfigError("scene object IDs must be unique")
    xml = [
        '<mujoco model="qdgrasp_scene">',
        f'  <option timestep="{spec.timestep:.17g}" gravity="{_numbers(spec.gravity)}"/>',
        "  <worldbody>",
    ]
    for support in spec.supports:
        if support.geom_type != "box":
            raise ConfigError(f"unsupported support geom type: {support.geom_type}")
        pos, quat = _transform_parts(support.T_world_support, f"support {support.support_id}")
        size = np.asarray(support.params.get("size", []), dtype=np.float64)
        if size.shape != (3,) or not np.all(np.isfinite(size)) or np.any(size <= 0.0):
            raise ConfigError(f"support {support.support_id} size must contain three positive values")
        friction = support.params.get("friction", [1.0, 0.005, 0.0001])
        xml.extend(
            [
                f'    <body name="{escape(support.support_id)}" pos="{_numbers(pos)}" quat="{_numbers(quat)}">',
                f'      <geom name="{escape(support.support_id)}::geom" type="box" size="{_numbers(size / 2.0)}" friction="{_numbers(friction)}"/>',
                "    </body>",
            ]
        )
    if include_objects:
        for scene_object in spec.objects:
            if not math_is_positive_finite(scene_object.scale):
                raise ConfigError(f"object {scene_object.object_id} scale must be finite and positive")
            manifest = _object_manifest(scene_object)
            pos, quat = _transform_parts(
                scene_object.T_world_object, f"object {scene_object.object_id}"
            )
            mass = (
                float(scene_object.mass)
                if scene_object.mass is not None
                else manifest.mass * scene_object.scale**3
            )
            if not math_is_positive_finite(mass):
                raise ConfigError(f"object {scene_object.object_id} mass must be finite and positive")
            friction = scene_object.friction or (1.0, 0.005, 0.0001)
            xml.append(
                f'    <body name="{escape(scene_object.object_id)}" pos="{_numbers(pos)}" quat="{_numbers(quat)}">'
            )
            if dynamic_objects:
                xml.append(f'      <freejoint name="{escape(scene_object.object_id)}::freejoint"/>')
            geom_mass = mass / len(manifest.collision_geoms)
            for index, geom in enumerate(manifest.collision_geoms):
                xml.append(
                    f'      <geom name="{escape(scene_object.object_id)}::geom::{index}" '
                    f'type="{geom.type}" size="{_numbers(np.asarray(geom.size) * scene_object.scale)}" '
                    f'pos="{_numbers(np.asarray(geom.pos) * scene_object.scale)}" '
                    f'quat="{_numbers(geom.quat)}" mass="{geom_mass:.17g}" '
                    f'friction="{_numbers(friction)}" condim="4"/>'
                )
            xml.append("    </body>")
    for camera in spec.cameras:
        pos, quat = _transform_parts(camera.T_world_camera, f"camera {camera.camera_id}")
        xml.append(
            f'    <camera name="{escape(camera.camera_id)}" pos="{_numbers(pos)}" '
            f'quat="{_numbers(quat)}" mode="fixed" fovy="45"/>'
        )
    xml.extend(["  </worldbody>", "</mujoco>"])
    try:
        return mujoco.MjModel.from_xml_string("\n".join(xml))
    except Exception as exc:
        raise ConfigError(f"failed to compile scene {spec.scene_id}: {exc}") from exc


def math_is_positive_finite(value: float) -> bool:
    return bool(np.isfinite(value) and value > 0.0)


def build_base_mujoco_model(spec: SceneSpec) -> mujoco.MjModel:
    """Compile only environment supports and cameras (legacy helper)."""
    return build_scene_mujoco_model(spec, include_objects=False)
