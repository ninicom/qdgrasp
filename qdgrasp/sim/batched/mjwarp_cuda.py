"""MJWarp CUDA backend (P3.4-05).

Same :class:`BatchedContactBackend` surface as the CPU oracle, so a strategy
never learns which one it is running on. What differs is scale: one compiled
model, many worlds stepped together on the device.

Two rules are structural rather than advisory. The backend refuses to exist
without a real NVIDIA device -- there is no CPU path through this class, so a
CPU number can never be reported under a CUDA schema. And every finalist it
exports is a request for the CPU oracle to replay, because GPU search ranks
candidates and the CPU oracle admits them.

Verified capability, measured on a Tesla T4 with MuJoCo Warp 1.16.0
(`evidence/phase3_4/p15-cuda-backend-decision/`): all three release hands
compile, Shadow included with its four tendon actuators, and the per-contact
stream is readable.
"""

from __future__ import annotations

import importlib
import importlib.util
import time
from collections.abc import Sequence

import mujoco
import numpy as np

from qdgrasp.dataset.dynamic_contracts import DynamicGraspRequest
from qdgrasp.sim.batched.contracts import (
    BackendCapabilityError,
    BackendState,
    BackendTiming,
    BackendUnavailableError,
    RolloutSummary,
    SceneSignature,
    WorldRejected,
    validate_control_batch,
    validate_control_sequences,
)


def warp_is_available() -> bool:
    """True only when both Warp and its MuJoCo binding can be imported."""
    return all(
        importlib.util.find_spec(name) is not None for name in ("warp", "mujoco_warp")
    )


class MjWarpCudaBackend:
    """Batched contact rollout across many worlds on one NVIDIA device."""

    backend_id = "mjwarp_cuda"

    def __init__(self, model_xml: str, *, device: str = "cuda:0") -> None:
        if not device.startswith("cuda"):
            raise BackendUnavailableError(
                f"this backend is CUDA-only; got device {device!r}. Use "
                "MuJoCoCpuBackend for CPU work rather than forcing this one."
            )
        if not warp_is_available():
            raise BackendUnavailableError(
                "mujoco_warp and warp are required for the CUDA backend and are "
                "not importable here. This must not fall back to CPU: a CPU "
                "measurement reported under a CUDA schema is fabricated evidence."
            )
        self._warp = importlib.import_module("warp")
        self._mjwarp = importlib.import_module("mujoco_warp")
        devices = [str(d) for d in self._warp.get_cuda_devices()]
        if not devices:
            raise BackendUnavailableError(
                "Warp reports no CUDA device; refusing to run a CUDA backend"
            )
        self._device = device
        self._model_xml = model_xml
        self._cpu_model: mujoco.MjModel | None = None
        self._warp_model = None
        self._warp_data = None
        self._requests: tuple[DynamicGraspRequest, ...] = ()
        self._signature: SceneSignature | None = None
        self._capacity = 0
        self._invalid: set[int] = set()
        self._timing = BackendTiming(0.0, 0.0, 0.0, 0, 0)

    # -- lifecycle ---------------------------------------------------------

    def compile(
        self, signature: SceneSignature, robot_profile: str, batch_capacity: int
    ) -> None:
        if batch_capacity <= 0:
            raise ValueError(f"batch_capacity must be positive, got {batch_capacity}")
        if signature.robot_profile != robot_profile:
            raise ValueError(
                f"signature robot_profile {signature.robot_profile!r} does not match "
                f"requested {robot_profile!r}"
            )
        started = time.perf_counter()
        model = mujoco.MjModel.from_xml_string(self._model_xml)
        model.opt.timestep = signature.timestep
        try:
            self._warp_model = self._mjwarp.put_model(model)
        except Exception as exc:
            raise BackendCapabilityError(
                f"mujoco_warp cannot compile this model: {type(exc).__name__}: {exc}"
            ) from exc
        self._cpu_model = model
        self._signature = signature
        self._capacity = batch_capacity
        self._timing = BackendTiming(time.perf_counter() - started, 0.0, 0.0, 0, 0)

    @property
    def model(self) -> mujoco.MjModel:
        if self._cpu_model is None:
            raise RuntimeError("backend used before compile()")
        return self._cpu_model

    @property
    def num_actuators(self) -> int:
        return int(self.model.nu)

    @property
    def num_worlds(self) -> int:
        return len(self._requests)

    @property
    def timing(self) -> BackendTiming:
        return self._timing

    def reset(self, requests: Sequence[DynamicGraspRequest]) -> BackendState:
        model = self.model
        if len(requests) > self._capacity:
            raise ValueError(
                f"{len(requests)} requests exceed batch capacity {self._capacity}"
            )
        self._requests = tuple(requests)
        self._invalid = set()
        cpu_data = mujoco.MjData(model)
        mujoco.mj_forward(model, cpu_data)
        self._warp_data = self._mjwarp.put_data(
            model, cpu_data, nworld=len(requests)
        )
        return self.observe()

    # -- state -------------------------------------------------------------

    def _array(self, name: str) -> np.ndarray:
        value = getattr(self._warp_data, name)
        return value.numpy() if hasattr(value, "numpy") else np.asarray(value)

    def observe(self) -> BackendState:
        if self._warp_data is None:
            raise RuntimeError("backend used before reset()")
        qpos = np.atleast_2d(self._array("qpos"))
        qvel = np.atleast_2d(self._array("qvel"))
        xpos = self._array("xpos")
        xquat = self._array("xquat")
        cvel = self._array("cvel")

        free_bodies = [
            int(self.model.jnt_bodyid[j])
            for j in range(int(self.model.njnt))
            if int(self.model.jnt_type[j]) == int(mujoco.mjtJoint.mjJNT_FREE)
        ]
        worlds = qpos.shape[0]
        objects = max(1, len(free_bodies))
        object_pose = np.zeros((worlds, objects, 7))
        object_velocity = np.zeros((worlds, objects, 6))
        for slot, body in enumerate(free_bodies):
            object_pose[:, slot, :3] = xpos[:, body]
            object_pose[:, slot, 3:] = xquat[:, body]
            object_velocity[:, slot] = cvel[:, body]
        if not free_bodies:
            object_pose[:, 0, 3] = 1.0

        contact_counts = np.zeros(worlds, dtype=int)
        ncon = getattr(self._warp_data, "ncon", None)
        if ncon is not None:
            counts = np.atleast_1d(
                ncon.numpy() if hasattr(ncon, "numpy") else np.asarray(ncon)
            )
            contact_counts[: counts.shape[0]] = counts.astype(int).ravel()[:worlds]

        self._reject_unusable(qpos, qvel)
        return BackendState(
            qpos=qpos,
            qvel=qvel,
            object_pose=object_pose,
            object_velocity=object_velocity,
            contact_counts=contact_counts,
            invalid_worlds=tuple(sorted(self._invalid)),
        )

    def _reject_unusable(self, qpos: np.ndarray, qvel: np.ndarray) -> None:
        """A world that went non-finite is rejected whole, not per step."""
        bad = ~(np.all(np.isfinite(qpos), axis=1) & np.all(np.isfinite(qvel), axis=1))
        self._invalid.update(int(i) for i in np.flatnonzero(bad))

    # -- integration -------------------------------------------------------

    def step(self, control_batch: np.ndarray, steps: int = 1) -> BackendState:
        if self._warp_data is None:
            raise RuntimeError("backend used before reset()")
        validate_control_batch(control_batch, self.num_worlds, self.num_actuators)
        if steps <= 0:
            raise ValueError(f"steps must be positive, got {steps}")

        started = time.perf_counter()
        ctrl = self._warp_data.ctrl
        if hasattr(ctrl, "assign"):
            ctrl.assign(np.ascontiguousarray(control_batch, dtype=np.float32))
        else:
            ctrl[:] = control_batch
        for _ in range(steps):
            self._mjwarp.step(self._warp_model, self._warp_data)
        self._warp.synchronize()
        elapsed = time.perf_counter() - started

        self._timing = BackendTiming(
            self._timing.compile_seconds,
            self._timing.warmup_seconds,
            self._timing.steady_state_seconds + elapsed,
            self._timing.steps_executed + steps,
            self.num_worlds,
        )
        return self.observe()

    def rollout(self, control_sequences: np.ndarray) -> tuple[RolloutSummary, ...]:
        worlds = self.num_worlds
        horizon = validate_control_sequences(
            control_sequences, worlds, self.num_actuators
        )

        # One untimed step so kernel compilation and device warmup never land in
        # the steady-state number the performance gate reads.
        warmup_started = time.perf_counter()
        self.step(control_sequences[:, 0, :], steps=1)
        warmup = time.perf_counter() - warmup_started

        started = time.perf_counter()
        for tick in range(1, horizon):
            self.step(control_sequences[:, tick, :], steps=1)
        elapsed = time.perf_counter() - started

        self._timing = BackendTiming(
            self._timing.compile_seconds, warmup, elapsed, max(1, horizon - 1), worlds
        )
        state = self.observe()
        return tuple(
            RolloutSummary(
                world_index=index,
                steps_executed=0 if index in self._invalid else horizon,
                objective_terms={},
                peak_safety_metrics={
                    "max_object_speed_mps": float(
                        np.max(np.abs(state.object_velocity[index]))
                    )
                },
                cumulative_safety_metrics={},
                hard_reject=index in self._invalid,
                failure_stage="rollout" if index in self._invalid else "none",
                failure_reason="world_rejected" if index in self._invalid else "none",
            )
            for index in range(worlds)
        )

    def export_finalists(self, indices: Sequence[int]) -> tuple[DynamicGraspRequest, ...]:
        """Hand finalists back as CPU requests; GPU search never self-admits."""
        finalists = []
        for index in indices:
            if not 0 <= index < len(self._requests):
                raise IndexError(f"world {index} is outside the live batch")
            if index in self._invalid:
                raise WorldRejected(
                    f"world {index} was rejected and cannot be exported as a finalist"
                )
            request = self._requests[index]
            finalists.append(
                DynamicGraspRequest(
                    scene_state_ref=request.scene_state_ref,
                    observation_ref=request.observation_ref,
                    target_object_id=request.target_object_id,
                    robot_profile=request.robot_profile,
                    strategy_id=request.strategy_id,
                    safety_budget_id=request.safety_budget_id,
                    horizon=request.horizon,
                    control_dt=request.control_dt,
                    seed=request.seed,
                    backend_request="cpu",
                )
            )
        return tuple(finalists)
