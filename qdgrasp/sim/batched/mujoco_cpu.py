"""MuJoCo CPU oracle backend (P3.4-03).

This backend is the correctness reference, not the fast one.  It runs worlds
sequentially over one compiled model and reads contact state through MuJoCo's
own solver output, so a CUDA backend can be checked against it world by world.

It is explicitly *not* CUDA evidence.  ``backend_id`` says ``mujoco_cpu`` and the
Phase 3.4 gate refuses to close on it alone.
"""

from __future__ import annotations

import time
from collections.abc import Sequence

import mujoco
import numpy as np

from qdgrasp.dataset.dynamic_contracts import DynamicGraspRequest
from qdgrasp.sim.batched.contracts import (
    BackendCapabilityError,
    BackendState,
    BackendTiming,
    RolloutSummary,
    SceneSignature,
    WorldRejected,
    validate_control_batch,
    validate_control_sequences,
)

#: Features whose contact or transmission output this backend reads directly.
#: A model using anything outside this set must be rejected before a search, not
#: silently integrated with wrong numbers.
_SUPPORTED_TRANSMISSIONS = frozenset(
    {
        int(mujoco.mjtTrn.mjTRN_JOINT),
        int(mujoco.mjtTrn.mjTRN_TENDON),
        int(mujoco.mjtTrn.mjTRN_SITE),
    }
)


class MuJoCoCpuBackend:
    """Sequential oracle over a single compiled MuJoCo model."""

    backend_id = "mujoco_cpu"

    def __init__(self, model_source: str | mujoco.MjModel) -> None:
        """Accept a compiled model as well as XML.

        A mesh-based hand cannot survive a round trip through an XML string:
        the serialised model references its assets by relative path and fails
        to reopen them. Passing the compiled model keeps those meshes attached.
        """
        self._model_xml = model_source if isinstance(model_source, str) else None
        self._prebuilt = None if isinstance(model_source, str) else model_source
        self._model: mujoco.MjModel | None = None
        self._worlds: list[mujoco.MjData] = []
        self._requests: tuple[DynamicGraspRequest, ...] = ()
        self._signature: SceneSignature | None = None
        self._object_body_ids: np.ndarray = np.zeros(0, dtype=int)
        self._invalid: set[int] = set()
        self._timing = BackendTiming(0.0, 0.0, 0.0, 0, 0)

    # -- lifecycle ---------------------------------------------------------

    def compile(
        self,
        signature: SceneSignature,
        robot_profile: str,
        batch_capacity: int,
    ) -> None:
        if batch_capacity <= 0:
            raise ValueError(f"batch_capacity must be positive, got {batch_capacity}")
        if signature.robot_profile != robot_profile:
            raise ValueError(
                f"signature robot_profile {signature.robot_profile!r} does not match "
                f"requested {robot_profile!r}"
            )
        started = time.perf_counter()
        model = (
            self._prebuilt
            if self._prebuilt is not None
            else mujoco.MjModel.from_xml_string(self._model_xml)
        )
        self._assert_supported(model)
        model.opt.timestep = signature.timestep
        self._model = model
        self._signature = signature
        self._worlds = [mujoco.MjData(model) for _ in range(batch_capacity)]
        self._object_body_ids = self._free_body_ids(model)
        compile_seconds = time.perf_counter() - started
        self._timing = BackendTiming(compile_seconds, 0.0, 0.0, 0, 0)

    @staticmethod
    def _assert_supported(model: mujoco.MjModel) -> None:
        for actuator in range(int(model.nu)):
            transmission = int(model.actuator_trntype[actuator])
            if transmission not in _SUPPORTED_TRANSMISSIONS:
                raise BackendCapabilityError(
                    f"actuator {actuator} uses transmission {transmission}, which this "
                    "backend does not read correctly; reject before searching"
                )

    @staticmethod
    def _free_body_ids(model: mujoco.MjModel) -> np.ndarray:
        """Bodies carrying a free joint, in model order: the movable objects."""
        ids = []
        for joint in range(int(model.njnt)):
            if int(model.jnt_type[joint]) == int(mujoco.mjtJoint.mjJNT_FREE):
                ids.append(int(model.jnt_bodyid[joint]))
        return np.asarray(ids, dtype=int)

    # -- state -------------------------------------------------------------

    @property
    def model(self) -> mujoco.MjModel:
        if self._model is None:
            raise RuntimeError("backend used before compile()")
        return self._model

    @property
    def num_worlds(self) -> int:
        return len(self._worlds)

    @property
    def num_actuators(self) -> int:
        return int(self.model.nu)

    @property
    def timing(self) -> BackendTiming:
        return self._timing

    def reset(self, requests: Sequence[DynamicGraspRequest]) -> BackendState:
        model = self.model
        if len(requests) > len(self._worlds):
            raise ValueError(
                f"{len(requests)} requests exceed batch capacity {len(self._worlds)}"
            )
        self._requests = tuple(requests)
        self._invalid = set()
        for index in range(len(requests)):
            mujoco.mj_resetData(model, self._worlds[index])
            mujoco.mj_forward(model, self._worlds[index])
        return self.observe()

    def observe(self) -> BackendState:
        model = self.model
        live = len(self._requests)
        qpos = np.zeros((live, int(model.nq)))
        qvel = np.zeros((live, int(model.nv)))
        objects = len(self._object_body_ids)
        object_pose = np.zeros((live, objects, 7))
        object_velocity = np.zeros((live, objects, 6))
        contact_counts = np.zeros(live, dtype=int)
        for index in range(live):
            data = self._worlds[index]
            qpos[index] = data.qpos
            qvel[index] = data.qvel
            contact_counts[index] = int(data.ncon)
            for slot, body_id in enumerate(self._object_body_ids):
                object_pose[index, slot, :3] = data.xpos[body_id]
                object_pose[index, slot, 3:] = data.xquat[body_id]
                object_velocity[index, slot] = data.cvel[body_id]
        return BackendState(
            qpos=qpos,
            qvel=qvel,
            object_pose=object_pose,
            object_velocity=object_velocity,
            contact_counts=contact_counts,
            invalid_worlds=tuple(sorted(self._invalid)),
        )

    # -- integration -------------------------------------------------------

    def step(self, control_batch: np.ndarray, steps: int = 1) -> BackendState:
        model = self.model
        live = len(self._requests)
        validate_control_batch(control_batch, live, self.num_actuators)
        if steps <= 0:
            raise ValueError(f"steps must be positive, got {steps}")
        started = time.perf_counter()
        for index in range(live):
            if index in self._invalid:
                continue
            data = self._worlds[index]
            data.ctrl[:] = control_batch[index]
            for _ in range(steps):
                mujoco.mj_step(model, data)
            self._reject_if_unusable(index, data)
        elapsed = time.perf_counter() - started
        self._timing = BackendTiming(
            self._timing.compile_seconds,
            self._timing.warmup_seconds,
            self._timing.steady_state_seconds + elapsed,
            self._timing.steps_executed + steps,
            live,
        )
        return self.observe()

    def _reject_if_unusable(self, index: int, data: mujoco.MjData) -> None:
        """A world that went non-finite or overflowed contacts is rejected whole."""
        if not (np.all(np.isfinite(data.qpos)) and np.all(np.isfinite(data.qvel))):
            self._invalid.add(index)
            return
        if int(data.ncon) >= int(self.model.nconmax) > 0:
            self._invalid.add(index)

    def rollout(self, control_sequences: np.ndarray) -> tuple[RolloutSummary, ...]:
        live = len(self._requests)
        horizon = validate_control_sequences(
            control_sequences, live, self.num_actuators
        )
        warmup_started = time.perf_counter()
        self.observe()
        warmup = time.perf_counter() - warmup_started

        started = time.perf_counter()
        for tick in range(horizon):
            self.step(control_sequences[:, tick, :], steps=1)
        elapsed = time.perf_counter() - started

        self._timing = BackendTiming(
            self._timing.compile_seconds, warmup, elapsed, horizon, live
        )
        state = self.observe()
        summaries = []
        for index in range(live):
            invalid = index in self._invalid
            summaries.append(
                RolloutSummary(
                    world_index=index,
                    steps_executed=0 if invalid else horizon,
                    objective_terms={},
                    peak_safety_metrics={
                        "max_object_speed_mps": float(
                            np.max(np.abs(state.object_velocity[index]))
                        )
                        if state.object_velocity.shape[1]
                        else 0.0
                    },
                    cumulative_safety_metrics={},
                    hard_reject=invalid,
                    failure_stage="rollout" if invalid else "none",
                    failure_reason="world_rejected" if invalid else "none",
                )
            )
        return tuple(summaries)

    def export_finalists(self, indices: Sequence[int]) -> tuple[DynamicGraspRequest, ...]:
        finalists = []
        for index in indices:
            if not 0 <= index < len(self._requests):
                raise IndexError(f"world {index} is outside the live batch")
            if index in self._invalid:
                raise WorldRejected(
                    f"world {index} was rejected and cannot be exported as a finalist"
                )
            finalists.append(self._requests[index])
        return tuple(finalists)
