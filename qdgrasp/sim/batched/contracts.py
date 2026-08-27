"""Backend contract for batched contact rollout (P3.4-02).

Two implementations are planned: a MuJoCo CPU oracle and an MJWarp CUDA backend.
The protocol exists so search strategies never learn which one they are on, and
so the fail-closed rules below are stated once rather than per backend.

Scenes are bucketed by :class:`SceneSignature`.  Compiling per candidate would
dominate the search cost, so anything that changes model topology moves a scene
to a different bucket; per-object mass, friction and pose stay in batched data.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable

import numpy as np

from qdgrasp.dataset.dynamic_contracts import DynamicGraspRequest


class BackendUnavailableError(RuntimeError):
    """A requested backend cannot run here and must not silently downgrade.

    ``backend=cuda`` without a real NVIDIA device is an error, never a CPU
    fallback: a CPU number reported as CUDA evidence is the failure this guards.
    """


class BackendCapabilityError(RuntimeError):
    """The compiled model uses a feature this backend does not support.

    Raised before a search starts rather than after, so an unsupported tendon or
    weld never produces quietly wrong contact numbers.
    """


class WorldRejected(RuntimeError):
    """A world produced numerically unusable state and is rejected whole.

    NaN/Inf, contact-buffer overflow or a truncated contact stream invalidate
    every step of that world, not just the offending one.
    """


@dataclasses.dataclass(frozen=True)
class SceneSignature:
    """Bucket key for compiled-model reuse.

    Two scenes share a compiled model only if every field here matches.  Object
    poses and masses are deliberately absent: those are batched data.
    """

    robot_profile: str
    environment: str
    geom_type_counts: tuple[tuple[str, int], ...]
    joint_count: int
    support_count: int
    solver_profile: str
    timestep: float

    def __post_init__(self) -> None:
        if not (np.isfinite(self.timestep) and self.timestep > 0.0):
            raise ValueError(f"timestep must be finite and positive, got {self.timestep}")
        counts = [name for name, _ in self.geom_type_counts]
        if counts != sorted(counts):
            raise ValueError("geom_type_counts must be sorted by geom type for a stable key")

    @property
    def bucket_key(self) -> str:
        payload = json.dumps(dataclasses.asdict(self), sort_keys=True, default=str)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclasses.dataclass(frozen=True)
class BackendState:
    """State of every live world after a step or reset."""

    qpos: np.ndarray  # [W, nq]
    qvel: np.ndarray  # [W, nv]
    object_pose: np.ndarray  # [W, O, 7]
    object_velocity: np.ndarray  # [W, O, 6]
    contact_counts: np.ndarray  # [W] contacts currently active
    invalid_worlds: tuple[int, ...] = ()


@dataclasses.dataclass(frozen=True)
class RolloutSummary:
    """What a strategy needs to rank one world without pulling full state."""

    world_index: int
    steps_executed: int
    objective_terms: dict[str, float]
    peak_safety_metrics: dict[str, float]
    cumulative_safety_metrics: dict[str, float]
    hard_reject: bool
    failure_stage: str
    failure_reason: str


@dataclasses.dataclass(frozen=True)
class BackendTiming:
    """Compile and warmup are reported apart from steady state.

    Folding them into throughput is how a slow backend looks fast on a short
    run, so the plan's performance gate reads these separately.
    """

    compile_seconds: float
    warmup_seconds: float
    steady_state_seconds: float
    steps_executed: int
    worlds: int

    @property
    def steps_per_second(self) -> float:
        if self.steady_state_seconds <= 0.0:
            return float("nan")
        return (self.steps_executed * self.worlds) / self.steady_state_seconds


@runtime_checkable
class BatchedContactBackend(Protocol):
    """Uniform surface over the CPU oracle and the CUDA backend."""

    #: Stable identifier written into evidence, e.g. ``"mujoco_cpu"``.
    backend_id: str

    def compile(
        self,
        signature: SceneSignature,
        robot_profile: str,
        batch_capacity: int,
    ) -> None:
        """Build one model for a bucket and size the world pool."""

    def reset(self, requests: Sequence[DynamicGraspRequest]) -> BackendState:
        """Seat one request per world and return the initial state."""

    def step(self, control_batch: np.ndarray, steps: int = 1) -> BackendState:
        """Advance every live world by ``steps`` under ``control_batch`` [W, U]."""

    def observe(self) -> BackendState:
        """Read current state without advancing time."""

    def rollout(self, control_sequences: np.ndarray) -> tuple[RolloutSummary, ...]:
        """Run ``[W, T, U]`` commands to completion and summarise each world."""

    def export_finalists(self, indices: Sequence[int]) -> tuple[DynamicGraspRequest, ...]:
        """Return replayable CPU requests for the given worlds."""

    @property
    def timing(self) -> BackendTiming:
        """Timing of the most recent rollout."""


def validate_control_batch(control_batch: np.ndarray, worlds: int, actuators: int) -> None:
    """Reject a control batch before it reaches the integrator."""
    if control_batch.shape != (worlds, actuators):
        raise ValueError(
            f"control batch must be [{worlds}, {actuators}], got {control_batch.shape}"
        )
    if not np.all(np.isfinite(control_batch)):
        raise WorldRejected("control batch contains non-finite entries")


def validate_control_sequences(
    control_sequences: np.ndarray, worlds: int, actuators: int
) -> int:
    """Reject a rollout command tensor and return its horizon."""
    if control_sequences.ndim != 3:
        raise ValueError(
            f"control sequences must be [W, T, U], got rank {control_sequences.ndim}"
        )
    got_worlds, horizon, got_actuators = control_sequences.shape
    if got_worlds != worlds or got_actuators != actuators:
        raise ValueError(
            f"control sequences must be [{worlds}, T, {actuators}], "
            f"got {control_sequences.shape}"
        )
    if horizon <= 0:
        raise ValueError("control sequences must have a positive horizon")
    if not np.all(np.isfinite(control_sequences)):
        raise WorldRejected("control sequences contain non-finite entries")
    return int(horizon)


def scene_signature_from_spec(spec: Any, robot_profile: str) -> SceneSignature:
    """Derive a bucket key from a :class:`~qdgrasp.scenes.contracts.SceneSpec`."""
    counts: dict[str, int] = {}
    for obj in spec.objects:
        key = getattr(obj, "shape_type", None) or getattr(obj, "asset_kind", "mesh")
        counts[str(key)] = counts.get(str(key), 0) + 1
    return SceneSignature(
        robot_profile=robot_profile,
        environment=spec.environment,
        geom_type_counts=tuple(sorted(counts.items())),
        joint_count=len(spec.objects) * 6,
        support_count=len(spec.supports),
        solver_profile=spec.solver_profile,
        timestep=spec.timestep,
    )
