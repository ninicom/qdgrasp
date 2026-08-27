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

import hashlib
import importlib
import importlib.util
import json
import time
from collections.abc import Sequence

import mujoco
import numpy as np

from qdgrasp.dataset.dynamic_contracts import DynamicGraspRequest
from qdgrasp.dynamic.capsule import InitialState, ModelIdentity, ReplayCapsule
from qdgrasp.sim.batched.contracts import (
    BackendCapabilityError,
    BackendState,
    BackendStateError,
    BackendTiming,
    BackendUnavailableError,
    ContactTelemetry,
    RolloutSummary,
    SceneSignature,
    WorldRejected,
    validate_control_batch,
    validate_control_sequences,
)

#: Contact fields the safety budget needs. A device build that cannot supply all
#: of them cannot enforce the budget, so the gate refuses it rather than
#: reporting the subset it happens to have (G07, G08.1).
REQUIRED_CONTACT_FIELDS: tuple[str, ...] = (
    "dist",
    "pos",
    "frame",
    "geom",
    "efc_address",
    "includemargin",
)


def _model_digest(model_xml: str | None, model: mujoco.MjModel) -> str:
    """A stable identity for the compiled model, mirroring the CPU oracle."""
    if model_xml is not None:
        return hashlib.sha256(model_xml.encode("utf-8")).hexdigest()
    payload = json.dumps(
        {
            "nq": int(model.nq),
            "nv": int(model.nv),
            "nu": int(model.nu),
            "nbody": int(model.nbody),
            "ngeom": int(model.ngeom),
            "njnt": int(model.njnt),
            "ntendon": int(model.ntendon),
            "neq": int(model.neq),
            "timestep": float(model.opt.timestep),
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def warp_is_available() -> bool:
    """True only when both Warp and its MuJoCo binding can be imported."""
    return all(
        importlib.util.find_spec(name) is not None for name in ("warp", "mujoco_warp")
    )


class MjWarpCudaBackend:
    """Batched contact rollout across many worlds on one NVIDIA device."""

    backend_id = "mjwarp_cuda"

    def __init__(
        self, model_source: str | mujoco.MjModel, *, device: str = "cuda:0"
    ) -> None:
        """Accept a compiled model as well as XML; see the CPU oracle for why."""
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
        self._model_xml = model_source if isinstance(model_source, str) else None
        self._prebuilt = None if isinstance(model_source, str) else model_source
        self._cpu_model: mujoco.MjModel | None = None
        self._warp_model = None
        self._warp_data = None
        self._requests: tuple[DynamicGraspRequest, ...] = ()
        self._signature: SceneSignature | None = None
        self._capacity = 0
        self._invalid: set[int] = set()
        self._invalid_reason: dict[int, str] = {}
        self._initial_states: tuple[InitialState | None, ...] = ()
        self._peak_contacts: np.ndarray = np.zeros(0, dtype=int)
        self._overflowed: set[int] = set()
        self._rolled_out = False
        self._last_commands: np.ndarray | None = None
        self._peak_contact_force: np.ndarray | None = None
        self._model_sha256 = ""
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
        model = (
            self._prebuilt
            if self._prebuilt is not None
            else mujoco.MjModel.from_xml_string(self._model_xml)
        )
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
        self._model_sha256 = _model_digest(self._model_xml, model)
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

    @property
    def batch_capacity(self) -> int:
        """Size of the world pool, which is not the same as the live count."""
        return int(self._capacity)

    def reset(
        self,
        requests: Sequence[DynamicGraspRequest],
        initial_states: Sequence[InitialState | None] | None = None,
    ) -> BackendState:
        """Seat one request per world, hydrating each world's own state.

        ``put_data`` broadcasts one CPU state across every world, so a batch
        that differs in initial state has to be written into the device arrays
        afterwards -- otherwise every world rolls out the same physics and the
        request batch means nothing (blocker B-13).
        """
        model = self.model
        if len(requests) > self._capacity:
            raise ValueError(
                f"{len(requests)} requests exceed batch capacity {self._capacity}"
            )
        if initial_states is not None and len(initial_states) != len(requests):
            raise ValueError(
                f"{len(initial_states)} initial states for {len(requests)} requests"
            )
        states = list(initial_states) if initial_states is not None else [None] * len(requests)
        self._assert_batchable(states)

        self._requests = tuple(requests)
        self._initial_states = tuple(states)
        self._invalid = set()
        self._invalid_reason = {}
        self._peak_contacts = np.zeros(len(requests), dtype=int)
        self._overflowed = set()
        self._rolled_out = False
        self._last_commands = None
        self._peak_contact_force = None

        cpu_data = mujoco.MjData(model)
        shared = next((state for state in states if state is not None), None)
        if shared is not None:
            model.body_mass[:] = shared.body_mass
            model.geom_friction[:] = shared.geom_friction
            cpu_data.qpos[:] = shared.qpos
            cpu_data.qvel[:] = shared.qvel
            if int(model.nmocap):
                cpu_data.mocap_pos[:] = shared.mocap_pos
                cpu_data.mocap_quat[:] = shared.mocap_quat
        mujoco.mj_forward(model, cpu_data)
        self._warp_data = self._mjwarp.put_data(
            model, cpu_data, nworld=len(requests)
        )
        self._hydrate_per_world(states)
        return self.observe()

    def _hydrate_per_world(self, states: Sequence[InitialState | None]) -> None:
        """Write each world's own qpos/qvel/mocap into the device arrays."""
        if all(state is None for state in states):
            return
        qpos = np.atleast_2d(self._array("qpos")).copy()
        qvel = np.atleast_2d(self._array("qvel")).copy()
        for index, state in enumerate(states):
            if state is None:
                continue
            qpos[index] = state.qpos
            qvel[index] = state.qvel
        self._assign("qpos", qpos)
        self._assign("qvel", qvel)
        if int(self.model.nmocap):
            mocap_pos = np.array(self._array("mocap_pos"))
            mocap_quat = np.array(self._array("mocap_quat"))
            for index, state in enumerate(states):
                if state is None:
                    continue
                mocap_pos[index] = state.mocap_pos
                mocap_quat[index] = state.mocap_quat
            self._assign("mocap_pos", mocap_pos)
            self._assign("mocap_quat", mocap_quat)

    def _assign(self, name: str, values: np.ndarray) -> None:
        target = getattr(self._warp_data, name)
        payload = np.ascontiguousarray(values, dtype=np.float32)
        if hasattr(target, "assign"):
            target.assign(payload)
        else:
            target[:] = values

    @staticmethod
    def _assert_batchable(states: Sequence[InitialState | None]) -> None:
        present = [state for state in states if state is not None]
        if len(present) < 2:
            return
        first = present[0]
        for other in present[1:]:
            if not np.array_equal(first.body_mass, other.body_mass) or not np.array_equal(
                first.geom_friction, other.geom_friction
            ):
                raise BackendCapabilityError(
                    "this backend shares one compiled model across worlds, so it cannot "
                    "vary body mass or geom friction per world; bucket those requests "
                    "separately instead of collapsing them onto one value"
                )

    # -- state -------------------------------------------------------------

    def _array(self, name: str) -> np.ndarray:
        value = getattr(self._warp_data, name)
        return value.numpy() if hasattr(value, "numpy") else np.asarray(value)

    def observe(self) -> BackendState:
        if self._warp_data is None:
            raise BackendStateError("observe() before reset(): no world has been seated")
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
        if self._peak_contacts.size >= worlds:
            self._peak_contacts[:worlds] = np.maximum(
                self._peak_contacts[:worlds], contact_counts
            )
        forces = self.read_contact_forces()
        if forces is not None and forces.size:
            # The device stream is flat across worlds; without a per-contact
            # world index the honest reduction is the batch peak, recorded as
            # such rather than attributed to a world it may not belong to.
            peak = float(np.max(forces))
            if self._peak_contact_force is None or self._peak_contact_force.size != worlds:
                self._peak_contact_force = np.zeros(worlds)
            self._peak_contact_force[:] = np.maximum(self._peak_contact_force, peak)

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
        for index in np.flatnonzero(bad):
            self._invalid.add(int(index))
            self._invalid_reason.setdefault(int(index), "non_finite_state")

    # -- integration -------------------------------------------------------

    def step(self, control_batch: np.ndarray, steps: int = 1) -> BackendState:
        if self._warp_data is None:
            raise BackendStateError("step() before reset(): no world has been seated")
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
        self._rolled_out = True
        self._last_commands = np.array(control_sequences, dtype=np.float64)
        return tuple(self._summarise(index, state, horizon) for index in range(worlds))

    def missing_contact_fields(self) -> tuple[str, ...]:
        """Contact fields this build does not expose.

        The safety budget needs the force, the frame, the penetration depth and
        the identity of both geoms. A build that supplies only ``pos`` can count
        contacts but cannot say whether any of them was safe, so the gate refuses
        it rather than reporting the subset it happens to have (G08.1).
        """
        contact = getattr(self._warp_data, "contact", None)
        if contact is None:
            return REQUIRED_CONTACT_FIELDS
        missing = [field for field in REQUIRED_CONTACT_FIELDS if not hasattr(contact, field)]
        if not hasattr(self._warp_data, "efc_force"):
            missing.append("efc_force")
        return tuple(missing)

    def read_contact_forces(self) -> np.ndarray | None:
        """Per-contact normal force, or ``None`` when the build cannot supply it.

        MuJoCo resolves contact force through the constraint solver, so the
        force of a contact is read at its ``efc_address`` in ``efc_force``. A
        contact whose address is negative was not admitted to the constraint
        system and carries no force; that is a real zero, unlike a missing field.
        """
        contact = getattr(self._warp_data, "contact", None)
        if contact is None or self.missing_contact_fields():
            return None
        address = self._as_numpy(contact.efc_address)
        forces = self._as_numpy(self._warp_data.efc_force)
        if address is None or forces is None:
            return None
        flat_address = np.asarray(address).astype(int).ravel()
        flat_forces = np.asarray(forces, dtype=np.float64).ravel()
        out = np.zeros(flat_address.shape[0], dtype=np.float64)
        valid = (flat_address >= 0) & (flat_address < flat_forces.shape[0])
        out[valid] = np.abs(flat_forces[flat_address[valid]])
        return out

    @staticmethod
    def _as_numpy(value: object) -> np.ndarray | None:
        if value is None:
            return None
        if hasattr(value, "numpy"):
            return value.numpy()
        try:
            return np.asarray(value)
        except (TypeError, ValueError):
            return None

    def contact_telemetry(self, world_index: int) -> ContactTelemetry:
        """What the device reported about contact in one world.

        A field the build does not expose lands in ``unavailable_fields`` rather
        than being reported as zero: an unobserved contact quantity is unknown,
        and a world whose contacts were not observed cannot be ranked against
        one whose were (blocker B-03).
        """
        unavailable = self.missing_contact_fields()
        capacity = int(getattr(self.model, "nconmax", 0) or 0)
        count = int(self._peak_contacts[world_index]) if self._peak_contacts.size else 0
        overflow = bool(capacity > 0 and count >= capacity) or world_index in self._overflowed
        return ContactTelemetry(
            contact_count=count,
            max_contact_count=count,
            class_counts={},
            buffer_overflow=overflow,
            stream_truncated=bool(unavailable),
            unavailable_fields=unavailable,
        )

    def _summarise(
        self, index: int, state: BackendState, horizon: int
    ) -> RolloutSummary:
        """Describe one world the same way the CPU oracle does.

        ``hard_reject`` covers more than NaN. A world that overflowed its
        contact buffer, or whose contact fields this build cannot read, did not
        observe fewer contacts -- it observed an unknown number of them, and
        letting it survive to be ranked is how an unsafe world becomes a
        finalist (blocker B-03).
        """
        telemetry = self.contact_telemetry(index)
        invalid = index in self._invalid
        reason = self._invalid_reason.get(index, "non_finite_state") if invalid else "none"
        if not invalid and telemetry.buffer_overflow:
            invalid, reason = True, "contact_buffer_overflow"
        elif not invalid and telemetry.unavailable_fields:
            invalid, reason = True, "truncated_contact_stream"

        peak = {
            "max_object_speed_mps": float(np.max(np.abs(state.object_velocity[index])))
        }
        forces = self._peak_contact_force
        if forces is not None and forces.size:
            peak["peak_normal_force_N"] = float(forces[index]) if forces.size > index else 0.0
        return RolloutSummary(
            world_index=index,
            steps_executed=0 if invalid else horizon,
            objective_terms={"steps": float(0 if invalid else horizon)},
            peak_safety_metrics=peak,
            cumulative_safety_metrics={},
            hard_reject=invalid,
            failure_stage="rollout" if invalid else "none",
            failure_reason=reason,
            contact=telemetry,
            backend_id=self.backend_id,
        )

    def export_finalists(self, indices: Sequence[int]) -> tuple[ReplayCapsule, ...]:
        """Hand finalists back as capsules the CPU oracle can replay exactly.

        v1 handed back a request, which names the scene and the seed but not the
        controls that were applied -- so the CPU regenerated a candidate and
        confirmed whatever that produced (blocker B-04). GPU search still never
        self-admits: the capsule is what the oracle checks, not a verdict.
        """
        if not self._rolled_out or self._last_commands is None:
            raise BackendStateError(
                "export_finalists() before rollout(): there is nothing to be a finalist of"
            )
        finalists = []
        for index in indices:
            if not 0 <= index < len(self._requests):
                raise IndexError(f"world {index} is outside the live batch")
            if index in self._invalid:
                raise WorldRejected(
                    f"world {index} was rejected and cannot be exported as a finalist"
                )
            finalists.append(self._capsule_for(index))
        return tuple(finalists)

    def _capsule_for(self, index: int) -> ReplayCapsule:
        request = self._requests[index]
        state = self._initial_states[index]
        if state is None:
            # The world started from the compiled defaults; read them back from
            # the device rather than assuming what they were.
            qpos = np.atleast_2d(self._array("qpos"))
            qvel = np.atleast_2d(self._array("qvel"))
            nmocap = int(self.model.nmocap)
            state = InitialState(
                qpos=qpos[index],
                qvel=qvel[index],
                mocap_pos=np.zeros((nmocap, 3)),
                mocap_quat=np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (nmocap, 1))
                if nmocap
                else np.zeros((0, 4)),
                body_mass=np.array(self.model.body_mass, dtype=np.float64),
                geom_friction=np.array(self.model.geom_friction, dtype=np.float64).reshape(-1, 3),
            )
        return ReplayCapsule(
            capsule_id=f"capsule:{self.backend_id}#{index}",
            model=ModelIdentity.from_model(
                self.model,
                robot_profile=request.robot_profile,
                scene_signature=self._signature.bucket_key if self._signature else "",
                model_sha256=self._model_sha256,
            ),
            state=state,
            control_sequence=np.asarray(self._last_commands[index], dtype=np.float64),
            control_dtype="float64",
            seed=int(request.seed),
            strategy_id=request.strategy_id,
            strategy_parameters={"control_dt": float(request.control_dt)},
            safety_budget_id=request.safety_budget_id,
            safety_budget_hash="0" * 64,
        )
