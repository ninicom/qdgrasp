"""Settle certification and immutable scene snapshots (P3.5-08).

A scene is ``settled`` only when *every* dynamic object simultaneously satisfies
the velocity, kinetic-energy and pose-delta thresholds for the pinned number of
consecutive steps.  "Simultaneously" and "consecutive" are the load-bearing
words: an object that goes quiet while another is still rolling has not settled
the scene, and a single quiet step is not a settled state.

Failure is classified, never collapsed into a boolean.  ``settle_timeout`` and
``fell_off_support`` are different problems with different fixes, and a caller
that only learns "not settled" cannot tell them apart.  The classes are checked
in a fixed precedence so the same physics always yields the same label.

The output is a :class:`SceneSnapshot`: exact poses and velocities, a contact
summary, the hashes that identify the world it came from, and a reduced settle
trace.  It is a record, not a handle -- replaying it must not carry solver warm
state from the episode that produced it.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from collections.abc import Sequence
from enum import Enum
from typing import Any

import mujoco
import numpy as np

from qdgrasp.scenes.contracts import SceneSpec
from qdgrasp.scenes.virtual_drop import SettleThresholds, SpawnRegion

SCENE_SNAPSHOT_SCHEMA_V1 = "qdgrasp/scene-snapshot/v1"


class SettleOutcome(str, Enum):
    """Every verdict the certifier is allowed to return."""

    SETTLED = "settled"
    INITIAL_OVERLAP = "initial_overlap"
    NON_FINITE_STATE = "non_finite_state"
    SOLVER_WARNING = "solver_warning"
    ESCAPED_SPAWN_REGION = "escaped_spawn_region"
    FELL_OFF_SUPPORT = "fell_off_support"
    EXCESSIVE_PENETRATION = "excessive_penetration"
    SETTLE_TIMEOUT = "settle_timeout"
    BACKEND_DIVERGENCE = "backend_divergence"


#: Precedence for classification.  A step can trip several conditions at once --
#: a diverging solve tends to trip all of them -- so the label is chosen from
#: this order rather than from whichever check happens to be written first.
#: Causes come before their consequences: a non-finite state explains the
#: penetration that follows it, not the other way round.
OUTCOME_PRECEDENCE: tuple[SettleOutcome, ...] = (
    SettleOutcome.NON_FINITE_STATE,
    SettleOutcome.SOLVER_WARNING,
    SettleOutcome.EXCESSIVE_PENETRATION,
    SettleOutcome.FELL_OFF_SUPPORT,
    SettleOutcome.ESCAPED_SPAWN_REGION,
    SettleOutcome.SETTLE_TIMEOUT,
)


def classify_settle_failure(triggered: set[SettleOutcome]) -> SettleOutcome:
    """Pick one label for a step that tripped one or more failure conditions."""

    for outcome in OUTCOME_PRECEDENCE:
        if outcome in triggered:
            return outcome
    raise ValueError(f"no classifiable failure in {sorted(item.value for item in triggered)}")


@dataclasses.dataclass(frozen=True)
class ObjectState:
    """One dynamic object's exact state at snapshot time."""

    object_id: str
    position_m: tuple[float, float, float]
    quaternion_wxyz: tuple[float, float, float, float]
    linear_velocity_mps: tuple[float, float, float]
    angular_velocity_radps: tuple[float, float, float]

    def to_document(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class SceneSnapshot:
    """The immutable result of one settle attempt."""

    scene_id: str
    outcome: SettleOutcome
    steps: int
    objects: tuple[ObjectState, ...]
    contact_summary: dict[str, Any]
    settle_trace: tuple[dict[str, float], ...]
    scene_hash: str
    thresholds_hash: str
    backend: str
    telemetry: dict[str, float]
    schema_version: str = SCENE_SNAPSHOT_SCHEMA_V1

    @property
    def settled(self) -> bool:
        return self.outcome is SettleOutcome.SETTLED

    def to_document(self) -> dict[str, Any]:
        return {
            "schema": self.schema_version,
            "scene_id": self.scene_id,
            "outcome": self.outcome.value,
            "steps": self.steps,
            "objects": [state.to_document() for state in self.objects],
            "contact_summary": self.contact_summary,
            "settle_trace": [dict(entry) for entry in self.settle_trace],
            "scene_hash": self.scene_hash,
            "thresholds_hash": self.thresholds_hash,
            "backend": self.backend,
            "telemetry": self.telemetry,
        }

    def content_hash(self) -> str:
        payload = json.dumps(self.to_document(), sort_keys=True, separators=(",", ":"), allow_nan=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @classmethod
    def from_document(cls, document: dict[str, Any]) -> SceneSnapshot:
        if document.get("schema") != SCENE_SNAPSHOT_SCHEMA_V1:
            raise ValueError(f"unsupported scene snapshot schema: {document.get('schema')!r}")
        return cls(
            scene_id=document["scene_id"],
            outcome=SettleOutcome(document["outcome"]),
            steps=int(document["steps"]),
            objects=tuple(
                ObjectState(
                    object_id=item["object_id"],
                    position_m=tuple(item["position_m"]),
                    quaternion_wxyz=tuple(item["quaternion_wxyz"]),
                    linear_velocity_mps=tuple(item["linear_velocity_mps"]),
                    angular_velocity_radps=tuple(item["angular_velocity_radps"]),
                )
                for item in document["objects"]
            ),
            contact_summary=document["contact_summary"],
            settle_trace=tuple(document["settle_trace"]),
            scene_hash=document["scene_hash"],
            thresholds_hash=document["thresholds_hash"],
            backend=document["backend"],
            telemetry=document["telemetry"],
        )


def _thresholds_hash(thresholds: SettleThresholds) -> str:
    payload = json.dumps(dataclasses.asdict(thresholds), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _scene_hash(spec: SceneSpec) -> str:
    payload = {
        "scene_id": spec.scene_id,
        "environment": spec.environment,
        "gravity": list(spec.gravity),
        "timestep": spec.timestep,
        "solver_profile": spec.solver_profile,
        "settle_seed": spec.settle_seed,
        "source_record_hash": spec.source_record_hash,
        "objects": [
            {
                "object_id": item.object_id,
                "asset_ref": item.asset_ref,
                "scale": item.scale,
                "mass": item.mass,
                "transform": np.asarray(item.T_world_object, dtype=np.float64).round(12).tolist(),
            }
            for item in spec.objects
        ],
        "supports": [
            {
                "support_id": item.support_id,
                "geom_type": item.geom_type,
                "params": item.params,
                "transform": np.asarray(item.T_world_support, dtype=np.float64).round(12).tolist(),
            }
            for item in spec.supports
        ],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


@dataclasses.dataclass
class _ObjectIndex:
    object_id: str
    body_id: int
    qpos_adr: int
    qvel_adr: int
    mass: float
    inertia: np.ndarray


def _index_objects(model: mujoco.MjModel, object_ids: Sequence[str]) -> list[_ObjectIndex]:
    indices: list[_ObjectIndex] = []
    for object_id in object_ids:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, object_id)
        if body_id < 0:
            raise ValueError(f"scene object {object_id!r} is absent from the compiled model")
        joint_id = int(model.body_jntadr[body_id])
        if joint_id < 0 or int(model.jnt_type[joint_id]) != int(mujoco.mjtJoint.mjJNT_FREE):
            raise ValueError(f"scene object {object_id!r} must carry a free joint to be dynamic")
        indices.append(
            _ObjectIndex(
                object_id=object_id,
                body_id=body_id,
                qpos_adr=int(model.jnt_qposadr[joint_id]),
                qvel_adr=int(model.jnt_dofadr[joint_id]),
                mass=float(model.body_mass[body_id]),
                inertia=np.array(model.body_inertia[body_id], dtype=np.float64),
            )
        )
    return indices


def _kinetic_energy(data: mujoco.MjData, indices: Sequence[_ObjectIndex]) -> float:
    total = 0.0
    for item in indices:
        linear = data.qvel[item.qvel_adr : item.qvel_adr + 3]
        angular = data.qvel[item.qvel_adr + 3 : item.qvel_adr + 6]
        total += 0.5 * item.mass * float(linear @ linear)
        total += 0.5 * float(angular @ (item.inertia * angular))
    return total


def _max_penetration(data: mujoco.MjData) -> float:
    depth = 0.0
    for index in range(int(data.ncon)):
        depth = max(depth, float(-data.contact[index].dist))
    return depth


def certify_settle(
    spec: SceneSpec,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    thresholds: SettleThresholds,
    *,
    spawn_region: SpawnRegion | None = None,
    support_z_m: float = 0.0,
    fall_margin_m: float = 0.25,
    trace_every: int = 25,
    backend: str = "mujoco-cpu",
) -> SceneSnapshot:
    """Step until the scene settles or fails, and classify which happened.

    The caller supplies an already-compiled model and a data whose initial state
    is the placement.  Nothing here writes a pose: the only thing that moves the
    objects is ``mj_step``.
    """

    thresholds.validate()
    object_ids = [item.object_id for item in spec.objects]
    indices = _index_objects(model, object_ids)
    mujoco.mj_forward(model, data)

    initial_penetration = _max_penetration(data)
    telemetry: dict[str, float] = {"initial_penetration_m": initial_penetration}
    if initial_penetration > thresholds.max_penetration_m:
        return _snapshot(
            spec, model, data, indices, SettleOutcome.INITIAL_OVERLAP, 0, (), telemetry, thresholds, backend
        )

    previous_pose = _poses(data, indices)
    stable_steps = 0
    trace: list[dict[str, float]] = []
    outcome = SettleOutcome.SETTLE_TIMEOUT
    steps = 0
    max_penetration = initial_penetration
    # MuJoCo accumulates warning counts for the lifetime of the data, so the
    # question "did this step warn" is a difference against a baseline, not a
    # read of the counter.
    warning_baseline = np.array(data.warning.number, dtype=np.int64).copy()

    for step in range(1, thresholds.timeout_steps + 1):
        steps = step
        mujoco.mj_step(model, data)
        triggered: set[SettleOutcome] = set()

        if not np.all(np.isfinite(data.qpos)) or not np.all(np.isfinite(data.qvel)):
            triggered.add(SettleOutcome.NON_FINITE_STATE)
        warnings_now = np.array(data.warning.number, dtype=np.int64)
        new_warnings = int(np.sum(np.maximum(warnings_now - warning_baseline, 0)))
        if new_warnings:
            telemetry["solver_warning_count"] = float(new_warnings)
            triggered.add(SettleOutcome.SOLVER_WARNING)
        warning_baseline = warnings_now.copy()

        penetration = _max_penetration(data)
        max_penetration = max(max_penetration, penetration)
        # Two different budgets, because they detect two different things: this
        # one catches tunnelling and solver blow-up at any instant, while the
        # settled-state budget is enforced inside the quiet condition below.
        if penetration > thresholds.max_transient_penetration_m:
            telemetry["transient_penetration_m"] = penetration
            triggered.add(SettleOutcome.EXCESSIVE_PENETRATION)

        positions = np.stack([np.array(data.xpos[item.body_id], dtype=np.float64) for item in indices])
        if np.any(positions[:, 2] < support_z_m - fall_margin_m):
            telemetry["lowest_object_z_m"] = float(positions[:, 2].min())
            triggered.add(SettleOutcome.FELL_OFF_SUPPORT)
        if spawn_region is not None and not all(
            spawn_region.contains_xy(position, margin=fall_margin_m) for position in positions
        ):
            triggered.add(SettleOutcome.ESCAPED_SPAWN_REGION)

        if triggered:
            outcome = classify_settle_failure(triggered)
            break

        linear = max(float(np.linalg.norm(data.qvel[item.qvel_adr : item.qvel_adr + 3])) for item in indices)
        angular = max(float(np.linalg.norm(data.qvel[item.qvel_adr + 3 : item.qvel_adr + 6])) for item in indices)
        energy = _kinetic_energy(data, indices)
        pose = _poses(data, indices)
        translation_delta = float(np.max(np.linalg.norm(pose[:, :3] - previous_pose[:, :3], axis=1)))
        rotation_delta = float(np.max(np.abs(pose[:, 3:] - previous_pose[:, 3:])))
        previous_pose = pose

        quiet = (
            linear <= thresholds.linear_velocity_mps
            and angular <= thresholds.angular_velocity_radps
            and energy <= thresholds.kinetic_energy_j
            and translation_delta <= thresholds.pose_delta_m
            and rotation_delta <= thresholds.pose_delta_rad
            # A state that is still interpenetrating is not a settled state,
            # however quiet it has gone.
            and penetration <= thresholds.max_penetration_m
        )
        stable_steps = stable_steps + 1 if quiet else 0

        if step % trace_every == 0 or quiet:
            trace.append(
                {
                    "step": float(step),
                    "max_linear_mps": linear,
                    "max_angular_radps": angular,
                    "kinetic_energy_j": energy,
                    "max_penetration_m": penetration,
                    "stable_steps": float(stable_steps),
                }
            )
        if stable_steps >= thresholds.consecutive_steps:
            outcome = SettleOutcome.SETTLED
            break

    telemetry["max_penetration_m"] = max_penetration
    telemetry["stable_steps"] = float(stable_steps)
    telemetry["steps"] = float(steps)
    return _snapshot(spec, model, data, indices, outcome, steps, tuple(trace), telemetry, thresholds, backend)


def _poses(data: mujoco.MjData, indices: Sequence[_ObjectIndex]) -> np.ndarray:
    rows = []
    for item in indices:
        rows.append(np.array(data.qpos[item.qpos_adr : item.qpos_adr + 7], dtype=np.float64))
    return np.stack(rows)


def _contact_summary(model: mujoco.MjModel, data: mujoco.MjData) -> dict[str, Any]:
    pairs: dict[str, int] = {}
    for index in range(int(data.ncon)):
        contact = data.contact[index]
        name1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom1)) or f"geom{contact.geom1}"
        name2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom2)) or f"geom{contact.geom2}"
        key = "|".join(sorted((name1, name2)))
        pairs[key] = pairs.get(key, 0) + 1
    return {"contact_count": int(data.ncon), "pairs": dict(sorted(pairs.items()))}


def _snapshot(
    spec: SceneSpec,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    indices: Sequence[_ObjectIndex],
    outcome: SettleOutcome,
    steps: int,
    trace: tuple[dict[str, float], ...],
    telemetry: dict[str, float],
    thresholds: SettleThresholds,
    backend: str,
) -> SceneSnapshot:
    states = []
    for item in indices:
        qpos = np.array(data.qpos[item.qpos_adr : item.qpos_adr + 7], dtype=np.float64)
        qvel = np.array(data.qvel[item.qvel_adr : item.qvel_adr + 6], dtype=np.float64)
        states.append(
            ObjectState(
                object_id=item.object_id,
                position_m=tuple(float(value) for value in qpos[:3]),  # type: ignore[arg-type]
                quaternion_wxyz=tuple(float(value) for value in qpos[3:7]),  # type: ignore[arg-type]
                linear_velocity_mps=tuple(float(value) for value in qvel[:3]),  # type: ignore[arg-type]
                angular_velocity_radps=tuple(float(value) for value in qvel[3:6]),  # type: ignore[arg-type]
            )
        )
    return SceneSnapshot(
        scene_id=spec.scene_id,
        outcome=outcome,
        steps=steps,
        objects=tuple(states),
        contact_summary=_contact_summary(model, data),
        settle_trace=trace,
        scene_hash=_scene_hash(spec),
        thresholds_hash=_thresholds_hash(thresholds),
        backend=backend,
        telemetry={key: float(value) for key, value in telemetry.items()},
    )


def replay_snapshot(model: mujoco.MjModel, snapshot: SceneSnapshot) -> mujoco.MjData:
    """Rebuild a data at the snapshot's state, with no warm solver state.

    A fresh ``MjData`` is allocated rather than reset in place: reusing the one
    that produced the snapshot would carry its contact and constraint warm start
    into a replay that is supposed to be independent of it.
    """

    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    for state in snapshot.objects:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, state.object_id)
        if body_id < 0:
            raise ValueError(f"snapshot object {state.object_id!r} is absent from this model")
        joint_id = int(model.body_jntadr[body_id])
        qpos_adr = int(model.jnt_qposadr[joint_id])
        qvel_adr = int(model.jnt_dofadr[joint_id])
        data.qpos[qpos_adr : qpos_adr + 3] = state.position_m
        data.qpos[qpos_adr + 3 : qpos_adr + 7] = state.quaternion_wxyz
        data.qvel[qvel_adr : qvel_adr + 3] = state.linear_velocity_mps
        data.qvel[qvel_adr + 3 : qvel_adr + 6] = state.angular_velocity_radps
    mujoco.mj_forward(model, data)
    return data
