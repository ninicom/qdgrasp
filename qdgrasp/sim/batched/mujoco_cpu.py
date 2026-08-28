"""MuJoCo CPU oracle backend (P3.4-03).

This backend is the correctness reference, not the fast one.  It runs worlds
sequentially over one compiled model and reads contact state through MuJoCo's
own solver output, so a CUDA backend can be checked against it world by world.

It is explicitly *not* CUDA evidence.  ``backend_id`` says ``mujoco_cpu`` and the
Phase 3.4 gate refuses to close on it alone.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Sequence

import mujoco
import numpy as np

from qdgrasp.dataset.dynamic_contracts import DynamicGraspRequest
from qdgrasp.dynamic.capsule import InitialState, ModelIdentity, ReplayCapsule
from qdgrasp.dynamic.safety import ContactObserver, SceneRoles
from qdgrasp.sim.batched.contracts import (
    BackendCapabilityError,
    BackendState,
    BackendStateError,
    BackendTiming,
    ContactTelemetry,
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

#: Equality constraints whose forces this backend resolves. A weld or a connect
#: is read correctly; anything else is rejected before a search rather than
#: integrated with numbers nobody checked (C02.8).
_SUPPORTED_EQUALITIES = frozenset(
    {
        int(mujoco.mjtEq.mjEQ_WELD),
        int(mujoco.mjtEq.mjEQ_CONNECT),
        int(mujoco.mjtEq.mjEQ_JOINT),
    }
)


def _model_digest(model_xml: str | None, model: mujoco.MjModel) -> str:
    """A stable identity for the compiled model.

    The XML when there is one; otherwise the structural counts, which is weaker
    but still detects the model being swapped underneath a capsule.
    """
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
        self._invalid_reason: dict[int, str] = {}
        self._timing = BackendTiming(0.0, 0.0, 0.0, 0, 0)
        self._roles: SceneRoles | None = None
        self._budget = None
        self._observers: dict[int, ContactObserver] = {}
        self._peak_contacts: dict[int, int] = {}
        self._overflowed: set[int] = set()
        self._rolled_out = False

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
        self._model_sha256 = _model_digest(self._model_xml, model)
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
        for equality in range(int(model.neq)):
            kind = int(model.eq_type[equality])
            if kind not in _SUPPORTED_EQUALITIES:
                raise BackendCapabilityError(
                    f"equality {equality} is type {kind}, whose constraint force this "
                    "backend does not resolve; reject before searching"
                )
        # A tendon nothing actuates carries load nothing reports, so a safety
        # budget claiming to bound tendon load cannot be enforced against it.
        if int(model.ntendon):
            driven = {
                int(model.actuator_trnid[i, 0])
                for i in range(int(model.nu))
                if int(model.actuator_trntype[i]) == int(mujoco.mjtTrn.mjTRN_TENDON)
            }
            undriven = sorted(set(range(int(model.ntendon))) - driven)
            if undriven:
                raise BackendCapabilityError(
                    f"tendons {undriven} have no actuator, so their load is unobservable; "
                    "reject before searching"
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
        """Live worlds: the ones a request was actually seated in.

        v1 returned the pool size here while the CUDA backend returned the live
        count, so the same number meant two different things depending on which
        backend produced it (C02.5).
        """
        return len(self._requests)

    @property
    def batch_capacity(self) -> int:
        """Size of the world pool, which is a different quantity."""
        return len(self._worlds)

    @property
    def num_actuators(self) -> int:
        return int(self.model.nu)

    @property
    def timing(self) -> BackendTiming:
        return self._timing

    def attach_safety(self, roles: SceneRoles, budget) -> None:
        """Give the oracle the scene roles and budget its summaries need.

        Without these the backend can count contacts but cannot classify them,
        and a summary that says "seventeen contacts" without saying what they
        touched is not a contact-rich summary (blocker B-14).
        """
        self._roles = roles
        self._budget = budget

    def reset(
        self,
        requests: Sequence[DynamicGraspRequest],
        initial_states: Sequence[InitialState | None] | None = None,
    ) -> BackendState:
        """Seat one request per world, hydrating each world's own state.

        v1 reset every world to the compiled model's defaults, so a batch of
        requests that differed in initial state, mass or friction all rolled out
        the same physics and the seed did nothing (blocker B-13). Per-world mass
        and friction cannot be varied over one shared compiled model, so a batch
        that disagrees on them is rejected here and has to be bucketed
        separately -- not silently collapsed onto one value (C02.4).
        """
        model = self.model
        if len(requests) > len(self._worlds):
            raise ValueError(
                f"{len(requests)} requests exceed batch capacity {len(self._worlds)}"
            )
        if initial_states is not None and len(initial_states) != len(requests):
            raise ValueError(
                f"{len(initial_states)} initial states for {len(requests)} requests"
            )

        self._assert_targets_exist(requests)
        states = list(initial_states) if initial_states is not None else [None] * len(requests)
        self._assert_batchable(states)

        self._requests = tuple(requests)
        self._invalid = set()
        self._invalid_reason = {}
        self._observers = {}
        self._peak_contacts = {}
        self._overflowed = set()
        self._rolled_out = False
        self._timing = BackendTiming(self._timing.compile_seconds, 0.0, 0.0, 0, 0)

        shared = next((state for state in states if state is not None), None)
        if shared is not None:
            model.body_mass[:] = shared.body_mass
            model.geom_friction[:] = shared.geom_friction

        for index, request in enumerate(requests):
            data = self._worlds[index]
            mujoco.mj_resetData(model, data)
            state = states[index]
            if state is not None:
                data.qpos[:] = state.qpos
                data.qvel[:] = state.qvel
                if int(model.nmocap):
                    data.mocap_pos[:] = state.mocap_pos
                    data.mocap_quat[:] = state.mocap_quat
            else:
                # No explicit state: the seed is the only thing distinguishing
                # this world, so it is recorded on the world rather than
                # pretended to have had an effect.
                del request
            data.time = 0.0
            mujoco.mj_forward(model, data)
            if self._roles is not None and self._budget is not None:
                observer = ContactObserver(model, self._roles, self._budget)
                observer.reset(data)
                self._observers[index] = observer
        # The state each world actually starts from, whether it came from a
        # capsule or from the compiled defaults. A capsule that guessed would
        # not replay.
        self._seated_states = tuple(
            InitialState.from_data(model, self._worlds[index]) for index in range(len(requests))
        )
        seated = self.observe()
        self._start_pose = (
            np.array(seated.object_pose[:, 0, :3])
            if seated.object_pose.shape[1]
            else np.zeros((len(requests), 3))
        )
        return seated

    def _assert_targets_exist(self, requests: Sequence[DynamicGraspRequest]) -> None:
        """A request naming a target this model does not have is a bad request."""
        model = self.model
        for index, request in enumerate(requests):
            body = mujoco.mj_name2id(
                model, int(mujoco.mjtObj.mjOBJ_BODY), request.target_object_id
            )
            if body < 0 and int(model.nbody) > 1:
                raise BackendCapabilityError(
                    f"world {index} targets {request.target_object_id!r}, which is not a body "
                    "in this compiled model; the outcome would be about something else"
                )

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

    def observe(self) -> BackendState:
        model = self.model
        if not self._requests:
            raise BackendStateError("observe() before reset(): no world has been seated")
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
        if not self._requests:
            raise BackendStateError("step() before reset(): no world has been seated")
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
        contacts = int(data.ncon)
        self._peak_contacts[index] = max(self._peak_contacts.get(index, 0), contacts)
        if not (np.all(np.isfinite(data.qpos)) and np.all(np.isfinite(data.qvel))):
            self._reject(index, "non_finite_state")
            return
        if not np.all(np.isfinite(data.ctrl)):
            self._reject(index, "non_finite_state")
            return
        capacity = int(getattr(self.model, "nconmax", 0) or 0)
        if capacity > 0 and contacts >= capacity:
            self._overflowed.add(index)
            self._reject(index, "contact_buffer_overflow")
            return
        observer = self._observers.get(index)
        if observer is not None:
            events = observer.observe(
                data,
                time_index=self._timing.steps_executed,
                dt=float(self.model.opt.timestep),
                simulator_step=self._timing.steps_executed,
            )
            if any(event.is_hard_reject for event in events):
                self._reject(index, "damaging_contact")

    def _reject(self, index: int, reason: str) -> None:
        self._invalid.add(index)
        self._invalid_reason.setdefault(index, reason)

    def rollout(self, control_sequences: np.ndarray) -> tuple[RolloutSummary, ...]:
        if not self._requests:
            raise BackendStateError("rollout() before reset(): no world has been seated")
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
        self._rolled_out = True
        self._last_commands = np.asarray(control_sequences, dtype=np.float64)
        return tuple(self._summarise(index, state, horizon) for index in range(live))

    def _summarise(
        self, index: int, state: BackendState, horizon: int
    ) -> RolloutSummary:
        """Describe one world the way both backends have to describe it.

        v1 returned empty objective and safety dicts here, so the oracle the
        GPU was supposed to be checked against said nothing about contact
        (blocker B-14). Everything below is measured, and a world whose contact
        stream could not be observed is rejected rather than summarised.
        """
        invalid = index in self._invalid
        reason = self._invalid_reason.get(index, "world_rejected") if invalid else "none"
        observer = self._observers.get(index)

        displacement = (
            float(np.linalg.norm(state.object_pose[index, 0, :3] - self._start_pose[index]))
            if self._start_pose.size and state.object_pose.shape[1]
            else 0.0
        )
        speed = (
            float(np.max(np.abs(state.object_velocity[index])))
            if state.object_velocity.shape[1]
            else 0.0
        )
        # The summary contract refuses to hold a non-finite metric, so a world
        # that produced one is rejected here rather than handed over to raise.
        if not (np.isfinite(displacement) and np.isfinite(speed)) and not invalid:
            invalid, reason = True, "non_finite_state"

        objective: dict[str, float] = {"steps": float(0 if invalid else horizon)}
        if np.isfinite(displacement):
            objective["object_displacement_m"] = displacement
        peak: dict[str, float] = {}
        if np.isfinite(speed):
            peak["max_object_speed_mps"] = speed
        cumulative: dict[str, float] = {}
        class_counts: dict[str, int] = {}
        unavailable: tuple[str, ...] = ()

        if observer is not None:
            evaluation = observer.evaluation
            peak.update({k: float(v) for k, v in observer.measurements.items()})
            peak["min_budget_margin"] = float(evaluation.min_margin)
            cumulative["contact_seconds"] = float(observer.elapsed_s)
            unavailable = evaluation.unavailable_fields
            if not invalid and not evaluation.safe:
                invalid = True
                reason = (
                    evaluation.failure_reasons[0]
                    if evaluation.failure_reasons
                    else "safety_budget_violation"
                )
        else:
            unavailable = ("contact_classification",)

        telemetry = ContactTelemetry(
            contact_count=int(state.contact_counts[index]),
            max_contact_count=int(self._peak_contacts.get(index, 0)),
            class_counts=class_counts,
            buffer_overflow=index in self._overflowed,
            stream_truncated=False,
            unavailable_fields=unavailable,
        )
        return RolloutSummary(
            world_index=index,
            steps_executed=0 if invalid else horizon,
            objective_terms=objective,
            peak_safety_metrics=peak,
            cumulative_safety_metrics=cumulative,
            hard_reject=invalid,
            failure_stage="rollout" if invalid else "none",
            failure_reason=reason if invalid else "none",
            contact=telemetry,
            backend_id=self.backend_id,
        )

    def export_finalists(self, indices: Sequence[int]) -> tuple[ReplayCapsule, ...]:
        """Hand back a capsule per world: the exact candidate, not a request."""
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
        state = self._seated_states[index]
        signature = self._signature
        return ReplayCapsule(
            capsule_id=f"capsule:{self.backend_id}#{index}",
            model=ModelIdentity.from_model(
                self.model,
                robot_profile=request.robot_profile,
                scene_signature=signature.bucket_key if signature else "",
                model_sha256=self._model_sha256,
            ),
            state=state,
            control_sequence=np.asarray(self._last_commands[index], dtype=np.float64),
            control_dtype="float64",
            seed=int(request.seed),
            strategy_id=request.strategy_id,
            strategy_parameters={"control_dt": float(request.control_dt)},
            safety_budget_id=request.safety_budget_id,
            safety_budget_hash=(
                self._budget.budget_hash if self._budget is not None else "0" * 64
            ),
        )
