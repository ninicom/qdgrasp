"""Compile a hand into a resolved scene (P3.5-10 support).

``qdgrasp.scenes.builders.base`` compiles a scene on its own; this puts a hand in
it.  The hand keeps the mocap-weld control that Phase 3 validated -- a free root
welded to a mocap body, so commanding the palm means moving a kinematic target
and letting the constraint solver carry the hand there -- and the scene objects
keep their free joints, so nothing about how they move changes by adding a hand.

The two active profiles of ``ADR-0008`` are supported.  Shadow is not loaded
here; it stays ``paused_by_ADR-0008``.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence

import mujoco
import numpy as np

from qdgrasp.config.schema import ConfigError
from qdgrasp.objects.manifest import load_object_asset
from qdgrasp.robot.spec import RobotSpec, resolve_robot_asset
from qdgrasp.scenes.contracts import SceneSpec

#: Robot profiles the RL environments accept.  Shadow is deliberately absent.
ACTIVE_ROBOT_PROFILES: tuple[str, ...] = ("leap_hand.yaml", "wonik_allegro.yaml")

MOCAP_BODY = "hand_mocap"

_WELD_SOLREF = (0.01, 1.0)
_WELD_SOLIMP = (0.99, 0.999, 0.001, 0.5, 2)


@dataclasses.dataclass(frozen=True)
class HandSceneIndices:
    """MuJoCo ids the acquisition environment reads every step."""

    palm_body: int
    root_body: int
    root_qpos_adr: int
    mocap_index: int
    hand_joint_ids: tuple[int, ...]
    hand_qpos_adr: tuple[int, ...]
    hand_dof_adr: tuple[int, ...]
    actuator_ids: tuple[int, ...]
    fingertip_bodies: tuple[int, ...]
    hand_geoms: tuple[int, ...]
    object_bodies: dict[str, int]
    object_geoms: dict[str, tuple[int, ...]]
    support_geoms: tuple[int, ...]


def _object_manifest_for(asset_ref: str):
    from pathlib import Path

    path = Path(asset_ref)
    manifest_path = path if path.name.endswith(".manifest.json") else path.with_name(f"{path.stem}.manifest.json")
    if not manifest_path.is_file():
        raise ConfigError(f"scene object asset_ref must resolve to an object manifest: {manifest_path}")
    _, manifest = load_object_asset(manifest_path)
    return manifest


def build_hand_scene_model(spec: SceneSpec, robot_profile: str) -> tuple[mujoco.MjModel, RobotSpec]:
    """Compile the scene's supports and objects together with one hand."""

    if robot_profile not in ACTIVE_ROBOT_PROFILES:
        raise ConfigError(
            f"robot profile {robot_profile!r} is not in the active corpus {ACTIVE_ROBOT_PROFILES}; "
            "Shadow remains paused_by_ADR-0008"
        )
    robot = RobotSpec.from_config(robot_profile, sample_anchors=False)
    hand_path = str(resolve_robot_asset(robot.config.source_asset))
    model_spec = mujoco.MjSpec.from_file(hand_path)
    if len(model_spec.worldbody.bodies) == 0:
        raise ConfigError(f"no bodies found in hand model: {hand_path}")

    hand_root = model_spec.worldbody.bodies[0]
    if not any(joint.type == mujoco.mjtJoint.mjJNT_FREE for joint in hand_root.joints):
        hand_root.add_freejoint(name="hand_freejoint")
    model_spec.worldbody.add_body(name=MOCAP_BODY, mocap=True)
    weld = model_spec.add_equality(
        type=mujoco.mjtEq.mjEQ_WELD,
        name="mocap_weld",
        name1=MOCAP_BODY,
        name2=hand_root.name,
        objtype=mujoco.mjtObj.mjOBJ_BODY,
        solref=list(_WELD_SOLREF),
        solimp=list(_WELD_SOLIMP),
    )
    weld.data = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float64)

    for support in spec.supports:
        if support.geom_type != "box":
            raise ConfigError(f"unsupported support geom type: {support.geom_type}")
        size = np.asarray(support.params.get("size", []), dtype=np.float64)
        if size.shape != (3,) or np.any(size <= 0.0):
            raise ConfigError(f"support {support.support_id} size must be three positive numbers")
        transform = np.asarray(support.T_world_support, dtype=np.float64)
        model_spec.worldbody.add_geom(
            name=support.support_id,
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=(size / 2.0).tolist(),
            pos=transform[:3, 3].tolist(),
            quat=_quat_from_matrix(transform[:3, :3]),
            condim=4,
            friction=[float(value) for value in support.params.get("friction", [1.0, 0.005, 0.0001])],
            rgba=[0.75, 0.72, 0.66, 1.0],
        )

    for scene_object in spec.objects:
        manifest = _object_manifest_for(scene_object.asset_ref)
        transform = np.asarray(scene_object.T_world_object, dtype=np.float64)
        body = model_spec.worldbody.add_body(
            name=scene_object.object_id,
            pos=transform[:3, 3].tolist(),
            quat=_quat_from_matrix(transform[:3, :3]),
        )
        body.add_freejoint(name=f"{scene_object.object_id}::freejoint")
        mass = (
            float(scene_object.mass) if scene_object.mass is not None else float(manifest.mass) * scene_object.scale**3
        )
        friction = list(scene_object.friction or (1.0, 0.005, 0.0001))
        share = mass / len(manifest.collision_geoms)
        for index, geom in enumerate(manifest.collision_geoms):
            body.add_geom(
                name=f"{scene_object.object_id}::geom::{index}",
                type=getattr(mujoco.mjtGeom, f"mjGEOM_{geom.type.upper()}"),
                size=(np.asarray(geom.size, dtype=np.float64) * scene_object.scale).tolist(),
                pos=(np.asarray(geom.pos, dtype=np.float64) * scene_object.scale).tolist(),
                quat=[float(value) for value in geom.quat],
                mass=share,
                condim=4,
                friction=friction,
                rgba=[0.82, 0.31, 0.29, 1.0],
            )

    try:
        compiled = model_spec.compile()
    except Exception as error:
        raise ConfigError(f"failed to compile hand scene for {robot_profile}: {error}") from error
    return compiled, robot


def _quat_from_matrix(rotation: np.ndarray) -> list[float]:
    from scipy.spatial.transform import Rotation

    quat_xyzw = Rotation.from_matrix(np.asarray(rotation, dtype=np.float64)).as_quat()
    return [float(quat_xyzw[3]), float(quat_xyzw[0]), float(quat_xyzw[1]), float(quat_xyzw[2])]


def resolve_hand_scene_indices(model: mujoco.MjModel, robot: RobotSpec, spec: SceneSpec) -> HandSceneIndices:
    """Resolve every id the step loop needs, failing loudly if one is missing."""

    def body(name: str) -> int:
        index = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if index < 0:
            raise ConfigError(f"body {name!r} is absent from the compiled hand scene")
        return index

    palm_candidates = [
        index
        for index in range(1, model.nbody)
        if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, index) or "").lower().endswith("palm")
    ]
    if not palm_candidates:
        raise ConfigError("could not identify the palm body in the compiled hand scene")
    palm_body = palm_candidates[0]
    root_body = palm_body
    while int(model.body_parentid[root_body]) != 0:
        root_body = int(model.body_parentid[root_body])
    root_joint = int(model.body_jntadr[root_body])
    if root_joint < 0 or int(model.jnt_type[root_joint]) != int(mujoco.mjtJoint.mjJNT_FREE):
        raise ConfigError("hand root must carry a free joint for mocap-weld control")

    # Actuators are matched by what they *drive*, not by what they are called.
    # LEAP names them `<joint>_act` and Allegro names them `ffa0` for joint
    # `ffj0`; a name convention that works for one hand silently excludes the
    # other, while the transmission target is the same fact in both models.
    joint_to_actuator: dict[int, int] = {}
    for actuator_id in range(model.nu):
        if int(model.actuator_trntype[actuator_id]) != int(mujoco.mjtTrn.mjTRN_JOINT):
            continue
        joint_to_actuator.setdefault(int(model.actuator_trnid[actuator_id, 0]), actuator_id)

    joint_ids: list[int] = []
    actuator_ids: list[int] = []
    for name in robot.actuated_joint_names:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            raise ConfigError(f"hand joint {name!r} is absent from the compiled hand scene")
        joint_ids.append(joint_id)
        actuator_id = joint_to_actuator.get(joint_id, -1)
        if actuator_id < 0:
            raise ConfigError(
                f"no joint-transmission actuator drives hand joint {name!r}; tendon-driven joints are not "
                "supported by the v1 RL action contract"
            )
        actuator_ids.append(actuator_id)

    object_bodies = {item.object_id: body(item.object_id) for item in spec.objects}
    support_names = {item.support_id for item in spec.supports}
    object_geoms: dict[str, tuple[int, ...]] = {name: () for name in object_bodies}
    support_geoms: list[int] = []
    hand_geoms: list[int] = []
    for geom_id in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
        owner = name.split("::")[0]
        if owner in object_geoms:
            object_geoms[owner] = (*object_geoms[owner], geom_id)
        elif name in support_names:
            support_geoms.append(geom_id)
        else:
            hand_geoms.append(geom_id)

    mocap_index = int(model.body_mocapid[body(MOCAP_BODY)])
    if mocap_index < 0:
        raise ConfigError("mocap body has no mocap index")

    return HandSceneIndices(
        palm_body=palm_body,
        root_body=root_body,
        root_qpos_adr=int(model.jnt_qposadr[root_joint]),
        mocap_index=mocap_index,
        hand_joint_ids=tuple(joint_ids),
        hand_qpos_adr=tuple(int(model.jnt_qposadr[index]) for index in joint_ids),
        hand_dof_adr=tuple(int(model.jnt_dofadr[index]) for index in joint_ids),
        actuator_ids=tuple(actuator_ids),
        fingertip_bodies=tuple(body(name) for name in robot.fingertip_links),
        hand_geoms=tuple(hand_geoms),
        object_bodies=object_bodies,
        object_geoms={name: tuple(values) for name, values in object_geoms.items()},
        support_geoms=tuple(support_geoms),
    )


def object_free_joint_addresses(model: mujoco.MjModel, body_id: int) -> tuple[int, int]:
    joint_id = int(model.body_jntadr[body_id])
    if joint_id < 0 or int(model.jnt_type[joint_id]) != int(mujoco.mjtJoint.mjJNT_FREE):
        raise ConfigError("scene object must carry a free joint")
    return int(model.jnt_qposadr[joint_id]), int(model.jnt_dofadr[joint_id])


def geom_owner_map(indices: HandSceneIndices) -> dict[int, str]:
    """Map every geom id onto ``hand``, ``support`` or its object id."""

    owners: dict[int, str] = {}
    for geom_id in indices.hand_geoms:
        owners[geom_id] = "hand"
    for geom_id in indices.support_geoms:
        owners[geom_id] = "support"
    for object_id, geom_ids in indices.object_geoms.items():
        for geom_id in geom_ids:
            owners[geom_id] = object_id
    return owners


def fingertip_offsets(robot: RobotSpec) -> np.ndarray:
    return np.stack([robot.fingertip_contact_offsets[name] for name in robot.fingertip_links])


def joint_limits(model: mujoco.MjModel, joint_ids: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
    lower = np.array([model.jnt_range[index, 0] for index in joint_ids], dtype=np.float64)
    upper = np.array([model.jnt_range[index, 1] for index in joint_ids], dtype=np.float64)
    return lower, upper
