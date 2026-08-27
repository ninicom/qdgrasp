"""Everything needed to replay one candidate, and nothing that depends on the
search that found it (G06).

A GPU search ranks candidates; only a CPU replay admits one. That only means
something if the CPU can replay the *same* candidate, and v1 could not: the GPU
exported a :class:`~qdgrasp.dataset.dynamic_contracts.DynamicGraspRequest`, which
says which scene and which seed but not which controls were actually applied.
The CPU then regenerated a candidate from the seed and confirmed whatever it
happened to produce (blocker B-04).

A capsule closes that gap. It carries the compiled model's identity, the exact
initial state, the exact control tensor with its dtype, the integrator settings
and the strategy parameters -- enough for a reviewer with the repository and
nothing else to reproduce the rollout and get the same answer. It is hashed
canonically, so changing one control value or one byte of initial state produces
a different capsule and invalidates any certificate that named the old one.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from qdgrasp.dataset.dynamic_contracts import (
    REPLAY_CAPSULE_SCHEMA_V1,
    ContractViolation,
    canonical_hash,
    sequence_hash,
)


class CapsuleError(ValueError):
    """A capsule is malformed, or does not match the model it is replayed on."""


def _array(values: Any, *, field: str, rank: int) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != rank:
        raise CapsuleError(f"{field} must have rank {rank}, got {array.ndim}")
    if array.size and not np.all(np.isfinite(array)):
        raise CapsuleError(f"{field} contains non-finite values")
    return array


@dataclasses.dataclass(frozen=True)
class ModelIdentity:
    """Which compiled model this capsule was produced against.

    A replay that reproduces the numbers on a *different* model has reproduced
    nothing, so the identity travels with the capsule and is checked on hydration.
    """

    robot_profile: str
    scene_signature: str
    model_sha256: str
    timestep_s: float
    integrator: int
    solver: int
    cone: int
    nq: int
    nv: int
    nu: int
    ngeom: int
    nbody: int

    def __post_init__(self) -> None:
        if not (np.isfinite(self.timestep_s) and self.timestep_s > 0.0):
            raise CapsuleError(f"timestep_s must be finite and positive, got {self.timestep_s}")
        if len(self.model_sha256) != 64:
            raise CapsuleError(f"model_sha256 must be a sha256 digest, got {self.model_sha256!r}")

    @classmethod
    def from_model(
        cls,
        model: mujoco.MjModel,
        *,
        robot_profile: str,
        scene_signature: str,
        model_sha256: str,
    ) -> ModelIdentity:
        return cls(
            robot_profile=robot_profile,
            scene_signature=scene_signature,
            model_sha256=model_sha256,
            timestep_s=float(model.opt.timestep),
            integrator=int(model.opt.integrator),
            solver=int(model.opt.solver),
            cone=int(model.opt.cone),
            nq=int(model.nq),
            nv=int(model.nv),
            nu=int(model.nu),
            ngeom=int(model.ngeom),
            nbody=int(model.nbody),
        )

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class InitialState:
    """The exact state the rollout started from.

    Mass and friction are in here because they are per-world batched data: two
    worlds can share a compiled model and still be different physics, and a
    replay that used the compiled defaults would be replaying a third thing.
    """

    qpos: np.ndarray  # [nq]
    qvel: np.ndarray  # [nv]
    mocap_pos: np.ndarray  # [nmocap, 3]
    mocap_quat: np.ndarray  # [nmocap, 4]
    body_mass: np.ndarray  # [nbody]
    geom_friction: np.ndarray  # [ngeom, 3]

    def __post_init__(self) -> None:
        object.__setattr__(self, "qpos", _array(self.qpos, field="qpos", rank=1))
        object.__setattr__(self, "qvel", _array(self.qvel, field="qvel", rank=1))
        object.__setattr__(self, "mocap_pos", _array(self.mocap_pos, field="mocap_pos", rank=2))
        object.__setattr__(self, "mocap_quat", _array(self.mocap_quat, field="mocap_quat", rank=2))
        object.__setattr__(self, "body_mass", _array(self.body_mass, field="body_mass", rank=1))
        object.__setattr__(
            self, "geom_friction", _array(self.geom_friction, field="geom_friction", rank=2)
        )
        if self.mocap_pos.shape[0] != self.mocap_quat.shape[0]:
            raise CapsuleError("mocap_pos and mocap_quat disagree on the mocap count")

    @classmethod
    def from_data(cls, model: mujoco.MjModel, data: mujoco.MjData) -> InitialState:
        nmocap = int(model.nmocap)
        return cls(
            qpos=np.array(data.qpos, dtype=np.float64),
            qvel=np.array(data.qvel, dtype=np.float64),
            mocap_pos=np.array(data.mocap_pos, dtype=np.float64).reshape(nmocap, 3),
            mocap_quat=np.array(data.mocap_quat, dtype=np.float64).reshape(nmocap, 4),
            body_mass=np.array(model.body_mass, dtype=np.float64),
            geom_friction=np.array(model.geom_friction, dtype=np.float64).reshape(-1, 3),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "qpos": self.qpos.tolist(),
            "qvel": self.qvel.tolist(),
            "mocap_pos": self.mocap_pos.tolist(),
            "mocap_quat": self.mocap_quat.tolist(),
            "body_mass": self.body_mass.tolist(),
            "geom_friction": self.geom_friction.tolist(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> InitialState:
        return cls(
            qpos=np.asarray(payload["qpos"], dtype=np.float64),
            qvel=np.asarray(payload["qvel"], dtype=np.float64),
            mocap_pos=np.asarray(payload["mocap_pos"], dtype=np.float64).reshape(-1, 3),
            mocap_quat=np.asarray(payload["mocap_quat"], dtype=np.float64).reshape(-1, 4),
            body_mass=np.asarray(payload["body_mass"], dtype=np.float64),
            geom_friction=np.asarray(payload["geom_friction"], dtype=np.float64).reshape(-1, 3),
        )


@dataclasses.dataclass(frozen=True)
class ReplayCapsule:
    """One candidate, replayable from disk by someone who was not there."""

    capsule_id: str
    model: ModelIdentity
    state: InitialState
    control_sequence: np.ndarray  # [T, U]
    control_dtype: str
    seed: int
    strategy_id: str
    strategy_parameters: Mapping[str, float]
    safety_budget_id: str
    safety_budget_hash: str
    schema: str = REPLAY_CAPSULE_SCHEMA_V1

    def __post_init__(self) -> None:
        if self.schema != REPLAY_CAPSULE_SCHEMA_V1:
            raise CapsuleError(f"unknown capsule schema {self.schema!r}")
        if not self.capsule_id.strip():
            raise CapsuleError("capsule_id must be a non-empty reference")
        controls = _array(self.control_sequence, field="control_sequence", rank=2)
        object.__setattr__(self, "control_sequence", controls)
        if controls.shape[1] != self.model.nu:
            raise CapsuleError(
                f"control_sequence is [T, {controls.shape[1]}] but the model has "
                f"{self.model.nu} actuators"
            )
        if self.state.qpos.shape[0] != self.model.nq:
            raise CapsuleError(
                f"qpos has {self.state.qpos.shape[0]} entries but the model has {self.model.nq}"
            )
        if self.state.qvel.shape[0] != self.model.nv:
            raise CapsuleError(
                f"qvel has {self.state.qvel.shape[0]} entries but the model has {self.model.nv}"
            )
        if len(self.safety_budget_hash) != 64:
            raise CapsuleError("safety_budget_hash must be a sha256 digest")
        for name, value in self.strategy_parameters.items():
            if not np.isfinite(float(value)):
                raise CapsuleError(f"strategy parameter {name!r} is not finite")

    @property
    def horizon(self) -> int:
        return int(self.control_sequence.shape[0])

    @property
    def command_sha256(self) -> str:
        return sequence_hash(self.control_sequence.astype(self.control_dtype, copy=False))

    @property
    def capsule_sha256(self) -> str:
        """Canonical hash over everything that changes the rollout."""
        return canonical_hash(
            {
                "schema": self.schema,
                "model": self.model.as_dict(),
                "state": self.state.as_dict(),
                "command_sha256": self.command_sha256,
                "control_dtype": self.control_dtype,
                "control_shape": list(self.control_sequence.shape),
                "seed": int(self.seed),
                "strategy_id": self.strategy_id,
                "strategy_parameters": {k: float(v) for k, v in sorted(self.strategy_parameters.items())},
                "safety_budget_id": self.safety_budget_id,
                "safety_budget_hash": self.safety_budget_hash,
            }
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "capsule_id": self.capsule_id,
            "capsule_sha256": self.capsule_sha256,
            "command_sha256": self.command_sha256,
            "model": self.model.as_dict(),
            "state": self.state.as_dict(),
            "control_sequence": self.control_sequence.tolist(),
            "control_dtype": self.control_dtype,
            "control_shape": list(self.control_sequence.shape),
            "seed": int(self.seed),
            "strategy_id": self.strategy_id,
            "strategy_parameters": {k: float(v) for k, v in sorted(self.strategy_parameters.items())},
            "safety_budget_id": self.safety_budget_id,
            "safety_budget_hash": self.safety_budget_hash,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ReplayCapsule:
        """Rebuild a capsule, and refuse one whose declared hashes do not hold."""
        if payload.get("schema") != REPLAY_CAPSULE_SCHEMA_V1:
            raise CapsuleError(f"unknown capsule schema {payload.get('schema')!r}")
        capsule = cls(
            capsule_id=str(payload["capsule_id"]),
            model=ModelIdentity(**dict(payload["model"])),
            state=InitialState.from_dict(payload["state"]),
            control_sequence=np.asarray(payload["control_sequence"], dtype=np.float64).reshape(
                *payload["control_shape"]
            ),
            control_dtype=str(payload["control_dtype"]),
            seed=int(payload["seed"]),
            strategy_id=str(payload["strategy_id"]),
            strategy_parameters=dict(payload.get("strategy_parameters", {})),
            safety_budget_id=str(payload["safety_budget_id"]),
            safety_budget_hash=str(payload["safety_budget_hash"]),
        )
        declared = payload.get("capsule_sha256")
        if declared is not None and declared != capsule.capsule_sha256:
            raise CapsuleError(
                f"capsule hash {capsule.capsule_sha256} does not match the declared {declared}; "
                "the payload was changed after it was written"
            )
        return capsule

    def write(self, path: str | Path) -> str:
        """Write the capsule deterministically and return its sha256."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        text = json.dumps(self.as_dict(), indent=2, sort_keys=True) + "\n"
        target.write_text(text, encoding="utf-8")
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    @classmethod
    def read(cls, path: str | Path) -> ReplayCapsule:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def capture_capsule(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    capsule_id: str,
    robot_profile: str,
    scene_signature: str,
    model_sha256: str,
    control_sequence: np.ndarray,
    seed: int,
    strategy_id: str,
    strategy_parameters: Mapping[str, float],
    safety_budget_id: str,
    safety_budget_hash: str,
    control_dtype: str = "float64",
) -> ReplayCapsule:
    """Capture the state a rollout is about to start from, plus its commands."""
    return ReplayCapsule(
        capsule_id=capsule_id,
        model=ModelIdentity.from_model(
            model,
            robot_profile=robot_profile,
            scene_signature=scene_signature,
            model_sha256=model_sha256,
        ),
        state=InitialState.from_data(model, data),
        control_sequence=np.asarray(control_sequence, dtype=np.float64),
        control_dtype=control_dtype,
        seed=int(seed),
        strategy_id=strategy_id,
        strategy_parameters=dict(strategy_parameters),
        safety_budget_id=safety_budget_id,
        safety_budget_hash=safety_budget_hash,
    )


def hydrate(model: mujoco.MjModel, data: mujoco.MjData, capsule: ReplayCapsule) -> None:
    """Put a model and data into exactly the state the capsule recorded.

    The model's per-world data -- mass and friction -- is restored too, because
    a replay against the compiled defaults is a replay of a different world.
    """
    if int(model.nq) != capsule.model.nq or int(model.nv) != capsule.model.nv:
        raise CapsuleError(
            f"capsule was captured on a model with nq={capsule.model.nq}, nv={capsule.model.nv}; "
            f"this model has nq={int(model.nq)}, nv={int(model.nv)}"
        )
    if int(model.nu) != capsule.model.nu:
        raise CapsuleError(
            f"capsule expects {capsule.model.nu} actuators, this model has {int(model.nu)}"
        )

    mujoco.mj_resetData(model, data)
    model.body_mass[:] = capsule.state.body_mass
    model.geom_friction[:] = capsule.state.geom_friction
    data.qpos[:] = capsule.state.qpos
    data.qvel[:] = capsule.state.qvel
    if int(model.nmocap):
        data.mocap_pos[:] = capsule.state.mocap_pos
        data.mocap_quat[:] = capsule.state.mocap_quat
    data.ctrl[:] = 0.0
    data.time = 0.0
    mujoco.mj_forward(model, data)


def replay(
    model: mujoco.MjModel,
    capsule: ReplayCapsule,
    *,
    steps_per_control: int = 1,
    observer: Any = None,
) -> mujoco.MjData:
    """Replay a capsule on the CPU oracle from its own recorded commands.

    Nothing is regenerated from the seed: the commands are the ones the capsule
    carries, which is the whole point of it existing (blocker B-04).
    """
    if steps_per_control < 1:
        raise CapsuleError(f"steps_per_control must be >= 1, got {steps_per_control}")
    data = mujoco.MjData(model)
    hydrate(model, data, capsule)

    commands = capsule.control_sequence.astype(capsule.control_dtype, copy=False)
    dt = float(model.opt.timestep)
    step_index = 0
    for row in range(capsule.horizon):
        data.ctrl[:] = commands[row]
        for _ in range(steps_per_control):
            mujoco.mj_step(model, data)
            step_index += 1
            if observer is not None:
                observer.observe(data, time_index=row, dt=dt, simulator_step=step_index)
    return data


def outcome_evidence_hash(capsule: ReplayCapsule, data: mujoco.MjData) -> str:
    """Hash the final state a replay reached, for certificate comparison."""
    return canonical_hash(
        {
            "capsule_sha256": capsule.capsule_sha256,
            "qpos": np.asarray(data.qpos, dtype=np.float64).tolist(),
            "qvel": np.asarray(data.qvel, dtype=np.float64).tolist(),
            "time": float(data.time),
        }
    )


def certificate_matches(capsule: ReplayCapsule, certificate: Any) -> bool:
    """Whether a CPU certificate was issued against exactly this capsule."""
    try:
        return (
            certificate.capsule_sha256 == capsule.capsule_sha256
            and certificate.command_sha256 == capsule.command_sha256
            and certificate.model_sha256 == capsule.model.model_sha256
        )
    except AttributeError as exc:  # pragma: no cover - a typed certificate always has these
        raise ContractViolation(f"not a CPU replay certificate: {exc}") from exc
