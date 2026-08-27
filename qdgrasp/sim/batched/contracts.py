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


class BackendStateError(RuntimeError):
    """A backend method was called out of order.

    compile, reset, step/rollout, observe and export form a state machine.
    Calling one of them early used to raise whatever happened to fail first --
    an AttributeError here, an IndexError there -- which made "the backend is
    not ready" indistinguishable from "the backend is broken" (C02.9).
    """


@dataclasses.dataclass(frozen=True)
class SceneSignature:
    """Bucket key for compiled-model reuse.

    Two scenes share a compiled model only if every field here matches.  Object
    poses are deliberately absent: those are batched data.

    v1 hashed seven fields, which meant two models with different actuator
    counts, different contact capacities or a different integrator could land in
    the same bucket and reuse each other's compiled model (blocker B-13). Every
    field below changes what the solver does, so every field is in the key.
    Mass and friction are *not* here: they are per-world data, and a backend
    that cannot vary them per world has to say so at preflight rather than
    quietly share one value.
    """

    robot_profile: str
    environment: str
    geom_type_counts: tuple[tuple[str, int], ...]
    joint_count: int
    support_count: int
    solver_profile: str
    timestep: float

    # -- topology, all of which changes the compiled model -----------------
    robot_asset_sha256: str = ""
    dof_count: int = 0
    actuator_count: int = 0
    tendon_count: int = 0
    equality_count: int = 0
    site_count: int = 0
    mocap_count: int = 0
    body_count: int = 0
    collision_geom_count: int = 0
    non_target_count: int = 0
    #: Contact and constraint capacities. A world that overflows them is
    #: rejected, so two models with different capacities are different worlds.
    contact_capacity: int = 0
    constraint_capacity: int = 0
    #: Integrator and solver options.
    integrator: int = 0
    solver: int = 0
    cone: int = 0
    solver_iterations: int = 0

    def __post_init__(self) -> None:
        if not (np.isfinite(self.timestep) and self.timestep > 0.0):
            raise ValueError(f"timestep must be finite and positive, got {self.timestep}")
        counts = [name for name, _ in self.geom_type_counts]
        if counts != sorted(counts):
            raise ValueError("geom_type_counts must be sorted by geom type for a stable key")

    @classmethod
    def from_model(
        cls,
        model: Any,
        *,
        robot_profile: str,
        environment: str,
        support_count: int,
        solver_profile: str = "default",
        robot_asset_sha256: str = "",
        non_target_count: int = 0,
    ) -> SceneSignature:
        """Derive the whole signature from a compiled model.

        Preferred over building one by hand: a field nobody remembered to fill
        in is exactly how two different models end up sharing a bucket.
        """
        geom_counts: dict[str, int] = {}
        collision_geoms = 0
        for geom in range(int(model.ngeom)):
            name = str(int(model.geom_type[geom]))
            geom_counts[name] = geom_counts.get(name, 0) + 1
            if int(model.geom_contype[geom]) or int(model.geom_conaffinity[geom]):
                collision_geoms += 1
        return cls(
            robot_profile=robot_profile,
            environment=environment,
            geom_type_counts=tuple(sorted(geom_counts.items())),
            joint_count=int(model.njnt),
            support_count=int(support_count),
            solver_profile=solver_profile,
            timestep=float(model.opt.timestep),
            robot_asset_sha256=robot_asset_sha256,
            dof_count=int(model.nv),
            actuator_count=int(model.nu),
            tendon_count=int(model.ntendon),
            equality_count=int(model.neq),
            site_count=int(model.nsite),
            mocap_count=int(model.nmocap),
            body_count=int(model.nbody),
            collision_geom_count=collision_geoms,
            non_target_count=int(non_target_count),
            contact_capacity=int(getattr(model, "nconmax", 0) or 0),
            constraint_capacity=int(getattr(model, "njmax", 0) or 0),
            integrator=int(model.opt.integrator),
            solver=int(model.opt.solver),
            cone=int(model.opt.cone),
            solver_iterations=int(model.opt.iterations),
        )

    @property
    def bucket_key(self) -> str:
        payload = json.dumps(dataclasses.asdict(self), sort_keys=True, default=str)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @property
    def topology_fields(self) -> tuple[str, ...]:
        """Fields that must differ for two scenes to need different models."""
        return tuple(
            field.name for field in dataclasses.fields(self) if field.name != "robot_asset_sha256"
        )


@dataclasses.dataclass(frozen=True)
class BackendState:
    """State of every live world after a step or reset."""

    qpos: np.ndarray  # [W, nq]
    qvel: np.ndarray  # [W, nv]
    object_pose: np.ndarray  # [W, O, 7]
    object_velocity: np.ndarray  # [W, O, 6]
    contact_counts: np.ndarray  # [W] contacts currently active
    invalid_worlds: tuple[int, ...] = ()


#: Schema of the summary both backends emit. The CPU oracle and the CUDA
#: backend have to describe a world the same way, or "parity" compares two
#: different things (blockers B-03, B-14).
ROLLOUT_SUMMARY_SCHEMA_V2 = "qdgrasp/rollout-summary/v2"


@dataclasses.dataclass(frozen=True)
class ContactTelemetry:
    """What the solver reported about contact, including what it could not.

    Overflow and truncation are first-class: a world whose contact buffer filled
    up did not observe fewer contacts, it observed an unknown number of them,
    and ranking it against worlds that did observe theirs is meaningless.
    """

    contact_count: int = 0
    max_contact_count: int = 0
    class_counts: dict[str, int] = dataclasses.field(default_factory=dict)
    buffer_overflow: bool = False
    stream_truncated: bool = False
    unavailable_fields: tuple[str, ...] = ()

    @property
    def observed(self) -> bool:
        return not (self.buffer_overflow or self.stream_truncated or self.unavailable_fields)


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
    contact: ContactTelemetry = dataclasses.field(default_factory=ContactTelemetry)
    schema: str = ROLLOUT_SUMMARY_SCHEMA_V2
    backend_id: str = ""

    def __post_init__(self) -> None:
        if self.schema != ROLLOUT_SUMMARY_SCHEMA_V2:
            raise ValueError(f"unknown rollout summary schema {self.schema!r}")
        for name in ("objective_terms", "peak_safety_metrics", "cumulative_safety_metrics"):
            for key, value in getattr(self, name).items():
                if not np.isfinite(float(value)):
                    raise WorldRejected(f"{name}[{key!r}] is not finite: {value!r}")
        # A world that survived cannot also be a hard reject, and one that was
        # rejected cannot claim reason "none": the two fields are one verdict.
        if self.hard_reject and self.failure_reason == "none":
            raise ValueError("a hard-rejected world must name a failure reason")
        if not self.hard_reject and self.failure_reason != "none":
            raise ValueError(
                f"a surviving world must carry failure_reason 'none', got {self.failure_reason!r}"
            )

    @property
    def survived(self) -> bool:
        return not self.hard_reject


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

    def reset(
        self,
        requests: Sequence[DynamicGraspRequest],
        initial_states: Sequence[Any] | None = None,
    ) -> BackendState:
        """Seat one request per world, hydrating each world's own state."""

    def step(self, control_batch: np.ndarray, steps: int = 1) -> BackendState:
        """Advance every live world by ``steps`` under ``control_batch`` [W, U]."""

    def observe(self) -> BackendState:
        """Read current state without advancing time."""

    def rollout(self, control_sequences: np.ndarray) -> tuple[RolloutSummary, ...]:
        """Run ``[W, T, U]`` commands to completion and summarise each world."""

    def export_finalists(self, indices: Sequence[int]) -> tuple[Any, ...]:
        """Return a :class:`~qdgrasp.dynamic.capsule.ReplayCapsule` per world.

        A request is not a candidate: it names the scene and the seed, not the
        controls that were applied. Exporting one meant the CPU regenerated
        something from the seed and confirmed whatever came out (blocker B-04).
        """

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
