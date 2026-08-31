"""The one scene the MVP environment runs: hand, table, single cuboid target.

Compiled once per object variant and reused across episodes.  Mass and friction
are re-stamped on the compiled model at reset (both are runtime-writable and
cheap); the extents are not, because changing a box's ``geom_size`` after
compilation leaves the body inertia describing the old box.

The hand is driven the way the validated Phase 3 rollout drives it: a free root
welded to a mocap body, so commanding the palm means moving a kinematic target
and letting the constraint solver carry the hand there.  Nothing in this module
can write the target's pose during an episode.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from pathlib import Path

import mujoco
import numpy as np

from qdgrasp.config.schema import ConfigError

#: Name of the free body the policy is trying to acquire.
TARGET_BODY = "target_object"
TARGET_GEOM = "target_object_geom"
TABLE_GEOM = "table_top"
MOCAP_BODY = "hand_mocap"

#: Table top sits at z = 0, so a resting target's centre is at its half height.
TABLE_HALF_EXTENTS = (0.30, 0.30, 0.02)

#: Solver settings copied from the validated rollout: a weld stiffer than this
#: turns a small initialisation mismatch into an unstable root acceleration.
_WELD_SOLREF = (0.01, 1.0)
_WELD_SOLIMP = (0.99, 0.999, 0.001, 0.5, 2)


@dataclasses.dataclass(frozen=True)
class SceneIndices:
    """Resolved MuJoCo ids the environment reads every step."""

    target_body: int
    target_geom: int
    target_qpos_adr: int
    target_qvel_adr: int
    table_geom: int
    mocap_index: int
    palm_body: int
    root_body: int
    root_qpos_adr: int
    hand_joint_ids: tuple[int, ...]
    hand_qpos_adr: tuple[int, ...]
    hand_dof_adr: tuple[int, ...]
    actuator_ids: tuple[int, ...]
    fingertip_bodies: tuple[int, ...]
    #: Synergy/contact group index per geom; ``-1`` for geoms that belong to no
    #: finger group (palm, table, target).
    geom_group: np.ndarray
    hand_geoms: tuple[int, ...]


def build_mvp_scene(
    hand_xml_path: str | Path,
    half_extents: Sequence[float],
    *,
    reference_mass: float,
    table_half_extents: Sequence[float] = TABLE_HALF_EXTENTS,
) -> mujoco.MjModel:
    """Compile hand + table + one cuboid target into a single model."""

    hand_path = Path(hand_xml_path).resolve()
    if not hand_path.is_file():
        raise ConfigError(f"hand XML file not found: {hand_path}")
    extents = np.asarray(half_extents, dtype=np.float64)
    if extents.shape != (3,) or not np.all(np.isfinite(extents)) or np.any(extents <= 0.0):
        raise ConfigError(f"target half extents must be three positive numbers, got {half_extents!r}")
    if not np.isfinite(reference_mass) or reference_mass <= 0.0:
        raise ConfigError("reference mass must be finite and positive")

    spec = mujoco.MjSpec.from_file(str(hand_path))
    if len(spec.worldbody.bodies) == 0:
        raise ConfigError(f"no bodies found in hand model: {hand_path}")

    hand_root = spec.worldbody.bodies[0]
    if not any(joint.type == mujoco.mjtJoint.mjJNT_FREE for joint in hand_root.joints):
        hand_root.add_freejoint(name="hand_freejoint")

    spec.worldbody.add_body(name=MOCAP_BODY, mocap=True)
    weld = spec.add_equality(
        type=mujoco.mjtEq.mjEQ_WELD,
        name="mocap_weld",
        name1=MOCAP_BODY,
        name2=hand_root.name,
        objtype=mujoco.mjtObj.mjOBJ_BODY,
        solref=list(_WELD_SOLREF),
        solimp=list(_WELD_SOLIMP),
    )
    # Declare the intended identity relative pose explicitly; otherwise the weld
    # inherits the bodies' compile-time offset and is violated at step zero.
    weld.data = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float64)

    target = spec.worldbody.add_body(name=TARGET_BODY, pos=[0.0, 0.0, float(extents[2])])
    target.add_freejoint(name=f"{TARGET_BODY}::freejoint")
    target.add_geom(
        name=TARGET_GEOM,
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[float(value) for value in extents],
        mass=float(reference_mass),
        condim=4,
        friction=[1.0, 0.005, 0.0001],
        rgba=[0.82, 0.31, 0.29, 1.0],
    )

    table = np.asarray(table_half_extents, dtype=np.float64)
    spec.worldbody.add_geom(
        name=TABLE_GEOM,
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[float(value) for value in table],
        pos=[0.0, 0.0, -float(table[2])],
        condim=4,
        friction=[1.0, 0.005, 0.0001],
        rgba=[0.75, 0.72, 0.66, 1.0],
    )

    try:
        return spec.compile()
    except Exception as exc:  # pragma: no cover - compilation failure is fatal
        raise ConfigError(f"failed to compile MVP scene for {hand_path}: {exc}") from exc


def _body_group(name: str, group_prefixes: Sequence[str]) -> int:
    for index, prefix in enumerate(group_prefixes):
        if name.startswith(prefix):
            return index
    return -1


def resolve_indices(
    model: mujoco.MjModel,
    joint_names: Sequence[str],
    fingertip_links: Sequence[str],
    group_prefixes: Sequence[str],
) -> SceneIndices:
    """Resolve every id the step loop needs, failing loudly if one is missing."""

    def body(name: str) -> int:
        index = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if index < 0:
            raise ConfigError(f"body '{name}' is absent from the compiled MVP scene")
        return index

    def geom(name: str) -> int:
        index = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        if index < 0:
            raise ConfigError(f"geom '{name}' is absent from the compiled MVP scene")
        return index

    target_body = body(TARGET_BODY)
    target_joint = int(model.body_jntadr[target_body])
    if target_joint < 0 or int(model.jnt_type[target_joint]) != int(mujoco.mjtJoint.mjJNT_FREE):
        raise ConfigError("target body must carry a free joint")

    palm_candidates = [
        index
        for index in range(1, model.nbody)
        if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, index) or "").lower().endswith("palm")
    ]
    if not palm_candidates:
        raise ConfigError("could not identify the palm body in the compiled MVP scene")
    palm_body = palm_candidates[0]
    root_body = palm_body
    while int(model.body_parentid[root_body]) != 0:
        root_body = int(model.body_parentid[root_body])
    root_joint = int(model.body_jntadr[root_body])
    if root_joint < 0 or int(model.jnt_type[root_joint]) != int(mujoco.mjtJoint.mjJNT_FREE):
        raise ConfigError("hand root must carry a free joint for mocap-weld control")

    joint_ids: list[int] = []
    for name in joint_names:
        index = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if index < 0:
            raise ConfigError(f"hand joint '{name}' is absent from the compiled MVP scene")
        joint_ids.append(index)

    actuator_ids: list[int] = []
    for name in joint_names:
        index = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{name}_act")
        if index < 0:
            index = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        if index < 0:
            raise ConfigError(f"no actuator found for hand joint '{name}'")
        actuator_ids.append(index)

    geom_group = np.full(model.ngeom, -1, dtype=np.int32)
    hand_geoms: list[int] = []
    target_geom = geom(TARGET_GEOM)
    table_geom = geom(TABLE_GEOM)
    for geom_id in range(model.ngeom):
        if geom_id in (target_geom, table_geom):
            continue
        body_id = int(model.geom_bodyid[geom_id])
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
        geom_group[geom_id] = _body_group(name, group_prefixes)
        hand_geoms.append(geom_id)

    mocap_index = int(model.body_mocapid[body(MOCAP_BODY)])
    if mocap_index < 0:
        raise ConfigError("mocap body has no mocap index")

    return SceneIndices(
        target_body=target_body,
        target_geom=target_geom,
        target_qpos_adr=int(model.jnt_qposadr[target_joint]),
        target_qvel_adr=int(model.jnt_dofadr[target_joint]),
        table_geom=table_geom,
        mocap_index=mocap_index,
        palm_body=palm_body,
        root_body=root_body,
        root_qpos_adr=int(model.jnt_qposadr[root_joint]),
        hand_joint_ids=tuple(joint_ids),
        hand_qpos_adr=tuple(int(model.jnt_qposadr[index]) for index in joint_ids),
        hand_dof_adr=tuple(int(model.jnt_dofadr[index]) for index in joint_ids),
        actuator_ids=tuple(actuator_ids),
        fingertip_bodies=tuple(body(name) for name in fingertip_links),
        geom_group=geom_group,
        hand_geoms=tuple(hand_geoms),
    )
