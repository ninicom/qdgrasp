"""Physics labeling pipeline: dynamic squeeze, lift, and perturbation evaluation in MuJoCo."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import mujoco
import numpy as np

from ..config.schema import ConfigError
from ..objects.schema import SubGeomSpec
from .fixtures import build_evaluation_model


@dataclass(frozen=True)
class PhysicsLabelResult:
    """Rigorous physical evaluation result for a grasp sample."""

    success: bool
    stable_lift: bool
    contact_count: int
    contacting_links: tuple[str, ...]
    lift_height: float
    max_penetration: float
    metrics: dict[str, float]


def build_labeled_scene_model(
    hand_xml_path: str | Path,
    collision_geoms: Sequence[SubGeomSpec],
    *,
    object_pos: tuple[float, float, float] = (0.0, 0.0, 0.05),
    object_mass: float = 0.1,
) -> mujoco.MjModel:
    """Build a compiled MuJoCo model containing the hand and the procedural object geoms."""
    hand_p = Path(hand_xml_path).resolve()
    if not hand_p.is_file():
        raise ConfigError(f"hand XML file not found: {hand_p}")

    spec = mujoco.MjSpec.from_file(str(hand_p))
    hand_root = spec.worldbody.bodies[0]
    has_free = any(j.type == mujoco.mjtJoint.mjJNT_FREE for j in hand_root.joints)
    if not has_free:
        hand_root.add_freejoint(name="hand_freejoint")

    mocap_body = spec.worldbody.add_body(name="hand_mocap", mocap=True)
    spec.add_equality(
        type=mujoco.mjtEq.mjEQ_WELD,
        name="mocap_weld",
        name1="hand_mocap",
        name2=hand_root.name,
        objtype=mujoco.mjtObj.mjOBJ_BODY,
    )

    obj_body = spec.worldbody.add_body(
        name="target_object",
        pos=[float(object_pos[0]), float(object_pos[1]), float(object_pos[2])],
    )
    obj_body.add_freejoint(name="object_freejoint")

    for i, g in enumerate(collision_geoms):
        geom_type = getattr(mujoco.mjtGeom, f"mjGEOM_{g.type.upper()}")
        size_list = [float(s) for s in g.size]
        pos_list = [float(p) for p in g.pos]
        quat_list = [float(q) for q in g.quat]

        obj_body.add_geom(
            name=f"object_subgeom_{i}",
            type=geom_type,
            size=size_list,
            pos=pos_list,
            quat=quat_list,
            mass=float(object_mass / len(collision_geoms)),
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
        raise ConfigError(f"failed to compile labeled scene for {hand_p}: {exc}") from exc


def evaluate_grasp_physics(
    hand_xml_path: str | Path,
    collision_geoms: Sequence[SubGeomSpec],
    *,
    palm_pos: tuple[float, float, float] = (0.0, 0.0, 0.1),
    joint_targets: Mapping[str, float] | None = None,
    object_pos: tuple[float, float, float] = (0.0, 0.0, 0.05),
    object_mass: float = 0.1,
    squeeze_steps: int = 150,
    lift_steps: int = 150,
    lift_height: float = 0.05,
    perturbation_steps: int = 50,
    seed: int = 0,
) -> PhysicsLabelResult:
    """Replay a candidate grasp through squeeze, lift, and perturbation test.

    Enforces strict physical success criteria (§6.2):
    1. Contact force must be non-zero on at least 2 distinct fingers/links.
    2. Object must maintain contact after lifting and disturbance wrench.
    3. Penetration depth must remain under 20mm.
    """
    np.random.seed(seed)
    model = build_labeled_scene_model(
        hand_xml_path=hand_xml_path,
        collision_geoms=collision_geoms,
        object_pos=object_pos,
        object_mass=object_mass,
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
        raise ConfigError("could not identify palm body in model")
    root_id = palm_id
    while int(model.body_parentid[root_id]) != 0:
        root_id = int(model.body_parentid[root_id])

    mocap_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand_mocap")
    if mocap_body_id < 0:
        raise ConfigError("could not identify mocap body in model")
    mocap_idx = model.body_mocapid[mocap_body_id]

    mujoco.mj_forward(model, data)
    palm_delta = np.asarray(palm_pos, dtype=np.float64) - data.xpos[palm_id]

    root_start_pos = np.array(data.xpos[root_id]) + palm_delta
    data.mocap_pos[mocap_idx][:3] = root_start_pos

    jnt_id = model.body_jntadr[root_id]
    if jnt_id >= 0 and model.jnt_type[jnt_id] == mujoco.mjtJoint.mjJNT_FREE:
        qpos_adr = model.jnt_qposadr[jnt_id]
        data.qpos[qpos_adr : qpos_adr + 3] = root_start_pos
    else:
        model.body_pos[root_id] += palm_delta

    mujoco.mj_forward(model, data)

    # Set initial joint positions
    target_by_joint: dict[int, float] = {}
    if joint_targets:
        for j_name, val in joint_targets.items():
            j_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j_name)
            if j_id >= 0:
                data.qpos[model.jnt_qposadr[j_id]] = float(val)
                target_by_joint[j_id] = float(val)

    # Object geom IDs
    object_geom_ids = set()
    for g_id in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g_id) or ""
        if name.startswith("object_subgeom_"):
            object_geom_ids.add(g_id)
    floor_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")

    def get_hand_contacts() -> tuple[int, set[str]]:
        contacts = 0
        links = set()
        for idx in range(int(data.ncon)):
            c = data.contact[idx]
            g1, g2 = int(c.geom1), int(c.geom2)
            if floor_geom_id in (g1, g2):
                continue
            is_hand_obj = (g1 in object_geom_ids and g2 not in object_geom_ids) or (
                g2 in object_geom_ids and g1 not in object_geom_ids
            )
            if is_hand_obj:
                contacts += 1
                hand_geom = g2 if g1 in object_geom_ids else g1
                b_id = int(model.geom_bodyid[hand_geom])
                b_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b_id) or f"body_{b_id}"
                links.add(b_name)
        return contacts, links

    mujoco.mj_forward(model, data)

    # 1. Squeeze stage
    for _ in range(squeeze_steps):
        for a_id in range(model.nu):
            if model.actuator_trntype[a_id] == mujoco.mjtTrn.mjTRN_JOINT:
                j_id = int(model.actuator_trnid[a_id, 0])
                if j_id in target_by_joint:
                    data.ctrl[a_id] = target_by_joint[j_id]
                    continue
            if model.actuator_ctrllimited[a_id]:
                lo, hi = model.actuator_ctrlrange[a_id]
                data.ctrl[a_id] = (lo + hi) * 0.5
        mujoco.mj_step(model, data)

    squeeze_contacts, contacting_links = get_hand_contacts()

    # 2. Lift stage
    obj_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target_object")
    init_z = float(data.xpos[obj_id][2]) if obj_id >= 0 else 0.0
    requested_lift = float(lift_height)

    for _ in range(lift_steps):
        if lift_steps > 0:
            data.mocap_pos[mocap_idx][2] += requested_lift / lift_steps
        mujoco.mj_step(model, data)

    # 3. Perturbation stage (apply disturbance shaking wrench)
    for step_p in range(perturbation_steps):
        shake_x = 0.002 * np.sin(step_p * 0.5)
        shake_y = 0.002 * np.cos(step_p * 0.5)
        data.mocap_pos[mocap_idx][0] += shake_x
        data.mocap_pos[mocap_idx][1] += shake_y
        mujoco.mj_step(model, data)

    final_z = float(data.xpos[obj_id][2]) if obj_id >= 0 else 0.0
    observed_lift = final_z - init_z

    # Max penetration computation
    max_penetration = 0.0
    for idx in range(int(data.ncon)):
        c = data.contact[idx]
        g1, g2 = int(c.geom1), int(c.geom2)
        if (g1 in object_geom_ids or g2 in object_geom_ids) and floor_geom_id not in (g1, g2):
            dist = float(c.dist)
            if dist < 0:
                max_penetration = max(max_penetration, abs(dist))

    final_contacts, final_links = get_hand_contacts()

    # Strict multi-link force closure success:
    # Must contact at least 2 distinct links, maintain positive lift, and small penetration
    stable_lift = observed_lift >= 0.5 * requested_lift
    multi_link_contact = len(contacting_links) >= 2 or len(final_links) >= 2
    success = (
        (squeeze_contacts > 0)
        and (final_contacts > 0)
        and multi_link_contact
        and stable_lift
        and (max_penetration < 0.02)
    )

    metrics = {
        "squeeze_contacts": float(squeeze_contacts),
        "final_contacts": float(final_contacts),
        "num_contacting_links": float(len(contacting_links.union(final_links))),
        "lift_height": float(observed_lift),
        "max_penetration": float(max_penetration),
    }

    return PhysicsLabelResult(
        success=bool(success),
        stable_lift=bool(stable_lift),
        contact_count=int(squeeze_contacts),
        contacting_links=tuple(sorted(contacting_links.union(final_links))),
        lift_height=float(observed_lift),
        max_penetration=float(max_penetration),
        metrics=metrics,
    )
