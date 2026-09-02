"""Deterministic MuJoCo evaluation fixtures for grasp, squeeze and lift."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from ..config.schema import ConfigError


@dataclass(frozen=True)
class FixtureResult:
    """Outcome and physical metrics of a grasp simulation fixture."""

    success: bool
    stable_lift: bool
    contact_count: int
    max_penetration: float
    lift_height: float
    metrics: dict[str, float]


GEOM_TYPES: dict[str, mujoco.mjtGeom] = {
    "box": mujoco.mjtGeom.mjGEOM_BOX,
    "sphere": mujoco.mjtGeom.mjGEOM_SPHERE,
    "cylinder": mujoco.mjtGeom.mjGEOM_CYLINDER,
}


def build_evaluation_model(
    hand_xml_path: str | Path,
    *,
    object_type: str = "box",
    object_size: tuple[float, ...] = (0.03, 0.03, 0.03),
    object_pos: tuple[float, float, float] = (0.0, 0.0, 0.05),
    object_mass: float = 0.1,
    timestep: float = 0.002,
    gravity: tuple[float, float, float] = (0.0, 0.0, -9.81),
) -> mujoco.MjModel:
    """Compile a scene holding the hand plus a graspable object.

    Built through :class:`mujoco.MjSpec` rather than by splicing an ``<include>``
    into a scene string: a string-built scene has no model directory, so the
    hand's own ``meshdir`` is not applied and every mesh reference fails to
    resolve.  ``MjSpec.from_file`` loads the hand exactly as MuJoCo would on its
    own, and the object is added to the resulting spec.
    """

    hand_p = Path(hand_xml_path).resolve()
    if not hand_p.is_file():
        raise ConfigError(f"hand XML file not found: {hand_p}")
    if object_type not in GEOM_TYPES:
        raise ConfigError(f"unsupported object type '{object_type}'; expected one of {sorted(GEOM_TYPES)}")

    try:
        spec = mujoco.MjSpec.from_file(str(hand_p))
    except Exception as exc:
        raise ConfigError(f"failed to load hand model {hand_p}: {exc}") from exc

    spec.option.timestep = float(timestep)
    spec.option.gravity = list(gravity)

    body = spec.worldbody.add_body(name="target_object", pos=list(object_pos))
    body.add_freejoint(name="object_freejoint")
    body.add_geom(
        name="object_geom",
        type=GEOM_TYPES[object_type],
        size=list(object_size) + [0.0] * (3 - len(object_size)),
        mass=float(object_mass),
        condim=4,
        friction=[1.0, 0.005, 0.0001],
        rgba=[0.8, 0.3, 0.3, 1.0],
    )
    spec.worldbody.add_geom(
        name="floor",
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=[1.0, 1.0, 0.1],
        pos=[0.0, 0.0, -0.1],
        rgba=[0.9, 0.9, 0.9, 1.0],
    )

    try:
        return spec.compile()
    except Exception as exc:
        raise ConfigError(f"failed to compile grasp fixture scene for {hand_p}: {exc}") from exc


def evaluate_grasp_fixture(
    hand_xml_path: str | Path,
    *,
    joint_targets: Mapping[str, float] | None = None,
    palm_pos: tuple[float, float, float] = (0.0, 0.0, 0.1),
    object_pos: tuple[float, float, float] = (0.0, 0.0, 0.05),
    object_size: tuple[float, ...] = (0.025, 0.025, 0.025),
    squeeze_steps: int = 150,
    lift_steps: int = 150,
    lift_height: float = 0.05,
    seed: int = 0,
) -> FixtureResult:
    """Run deterministic grasp -> squeeze -> lift evaluation in MuJoCo.

    1. Grasp stage: sets hand joints and object pose.
    2. Squeeze stage: drives joint actuators to the requested targets.
    3. Lift stage: translates the hand root upward and checks if the object follows.
    """
    np.random.seed(seed)
    # A direct-hand fallback would silently remove the target object and make a
    # grasp fixture pass without testing a grasp, so scene construction fails
    # closed instead.
    model = build_evaluation_model(
        hand_xml_path,
        object_pos=object_pos,
        object_size=object_size,
    )

    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)

    palm_candidates = [
        b_id
        for b_id in range(1, model.nbody)
        if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b_id) or "").lower().endswith("palm")
    ]
    palm_id = palm_candidates[0] if palm_candidates else -1
    if palm_id < 0:
        raise ConfigError("grasp fixture could not identify a palm body")
    root_id = palm_id
    while int(model.body_parentid[root_id]) != 0:
        root_id = int(model.body_parentid[root_id])
    mujoco.mj_forward(model, data)
    palm_delta = np.asarray(palm_pos, dtype=np.float64) - data.xpos[palm_id]
    model.body_pos[root_id] += palm_delta
    mujoco.mj_forward(model, data)

    # Set initial joint positions
    if joint_targets:
        for j_name, val in joint_targets.items():
            j_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j_name)
            if j_id < 0:
                raise ConfigError(f"joint target '{j_name}' not found in hand model")
            data.qpos[model.jnt_qposadr[j_id]] = float(val)

    target_by_joint: dict[int, float] = {}
    for j_id in range(model.njnt):
        if model.jnt_type[j_id] not in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
            continue
        if joint_targets:
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j_id)
            if name in joint_targets:
                target_by_joint[j_id] = float(joint_targets[name])
        if j_id not in target_by_joint and model.jnt_limited[j_id]:
            lo, hi = model.jnt_range[j_id]
            target_by_joint[j_id] = float((lo + hi) * 0.5)

    object_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "object_geom")
    floor_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")

    def hand_object_contacts() -> int:
        """Count contacts between the hand and the object only.

        The raw contact count also includes the object resting on the floor, so
        it stays positive even when the hand never touches anything.
        """

        total = 0
        for index in range(int(data.ncon)):
            contact = data.contact[index]
            geoms = (int(contact.geom1), int(contact.geom2))
            if object_geom_id not in geoms or floor_geom_id in geoms:
                continue
            total += 1
        return total

    mujoco.mj_forward(model, data)

    # Squeeze phase
    for _ in range(squeeze_steps):
        for a_id in range(model.nu):
            # Position actuators target a joint directly.  Tendon/general
            # actuators are not guessed at; their declared control midpoint is
            # the only deterministic control available from this fixture API.
            if model.actuator_trntype[a_id] == mujoco.mjtTrn.mjTRN_JOINT:
                j_id = int(model.actuator_trnid[a_id, 0])
                if j_id in target_by_joint:
                    data.ctrl[a_id] = target_by_joint[j_id]
                    continue
            if model.actuator_ctrllimited[a_id]:
                lo, hi = model.actuator_ctrlrange[a_id]
                data.ctrl[a_id] = (lo + hi) * 0.5
        mujoco.mj_step(model, data)

    squeeze_contacts = hand_object_contacts()

    # Lift phase: observe target object displacement if present
    obj_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target_object")
    if obj_id < 0:
        raise ConfigError("grasp fixture scene has no target_object body")
    init_z = float(data.xpos[obj_id][2])
    requested_lift_height = float(lift_height)
    for _ in range(lift_steps):
        if lift_steps > 0:
            model.body_pos[root_id][2] += requested_lift_height / lift_steps
            mujoco.mj_forward(model, data)
        mujoco.mj_step(model, data)

    final_z = float(data.xpos[obj_id][2])
    observed_lift_height = final_z - init_z

    # Compute penetration / distance summary
    max_penetration = 0.0
    for c_id in range(data.ncon):
        dist = float(data.contact[c_id].dist)
        if dist < 0:
            max_penetration = max(max_penetration, abs(dist))

    # Stable if object maintained contact and did not penetrate excessively
    final_contacts = hand_object_contacts()
    success = squeeze_contacts > 0 and final_contacts > 0 and max_penetration < 0.05
    stable_lift = success and (observed_lift_height >= 0.5 * requested_lift_height)

    metrics = {
        "squeeze_contacts": float(squeeze_contacts),
        "final_contacts": float(final_contacts),
        "lift_height": float(observed_lift_height),
        "max_penetration": float(max_penetration),
    }

    return FixtureResult(
        success=success,
        stable_lift=stable_lift,
        contact_count=squeeze_contacts,
        max_penetration=max_penetration,
        lift_height=observed_lift_height,
        metrics=metrics,
    )
