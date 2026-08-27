"""Contact observation and multi-quantity safety accounting (P3.4-06).

Static grasp asks one question of a contact: does it exist. That is enough when
every non-target contact is a rejection. Phase 3.4 permits table and neighbour
contact, so it has to ask how hard, for how long, how much it slipped and how
much energy went in -- and answer per contact, over time.

Everything here is measured from MuJoCo's own solver output. Nothing is
predicted, and no contact is admitted on geometry alone.

Two defects from the first version are fixed here.

**The window was not a window** (blocker B-02). Impulse was accumulated into a
block that reset the moment it filled, so an impact straddling the boundary was
split in half and each half passed. It is now a real sliding window: increments
are kept with their timestamps and dropped when they age out, so the verdict
does not depend on where the impact happened to land.

**Six declared limits had no sensor** (blocker B-01). Wrist force and torque,
joint or tendon load, and the translation, rotation and velocity a neighbouring
object may pick up were declared, hashed into manifests and never read. All six
are measured here, and a budget whose limits this model cannot measure fails
preflight rather than evaluating to "fine".
"""

from __future__ import annotations

import dataclasses
from collections import deque
from collections.abc import Sequence

import mujoco
import numpy as np

from qdgrasp.dataset.dynamic_contracts import (
    ContactClass,
    ContactEvent,
    ContactPairKind,
    ContactSafetyBudget,
)
from qdgrasp.dynamic.safety_budget import (
    SafetyCoverageError,
    SafetyEvaluation,
    SensorScope,
    evaluate_budget,
    require_full_coverage,
)


@dataclasses.dataclass(frozen=True)
class SceneRoles:
    """Which geoms play which part, by geom id.

    Classification is by identity, not by name substring: a substring such as
    "table" or "obj" is not a reliable label across scene sources.

    ``wrist_body`` is the body the wrist budget is resolved at. It is optional
    only so that micro test scenes without a wrist can still be built; a budget
    that declares a wrist limit against a scene without one fails preflight.
    """

    target_geoms: frozenset[int]
    support_geoms: frozenset[int]
    non_target_geoms: frozenset[int]
    robot_geoms: frozenset[int]
    #: Robot geom pairs that may touch each other, as sorted 2-tuples.
    self_contact_allowlist: frozenset[tuple[int, int]] = frozenset()
    #: Pairs that are never permitted regardless of measured force.
    forbidden_pairs: frozenset[tuple[int, int]] = frozenset()
    #: Body the wrist force and torque budgets are read at.
    wrist_body: int | None = None
    #: Body whose pose is recorded as the palm pose.
    palm_body: int | None = None

    def role_of(self, geom_id: int) -> str:
        if geom_id in self.target_geoms:
            return "target"
        if geom_id in self.support_geoms:
            return "support"
        if geom_id in self.non_target_geoms:
            return "non_target"
        if geom_id in self.robot_geoms:
            return "robot"
        return "unknown"


#: Every ordered pair of roles, and the kind of contact it makes. Written as a
#: table rather than a chain of conditionals so that a missing combination is
#: visible rather than falling through to whatever the last branch happened to
#: be (blocker B-12).
_PAIR_KINDS: dict[frozenset[str], ContactPairKind] = {
    frozenset({"target", "support"}): ContactPairKind.TARGET_SUPPORT,
    frozenset({"robot", "support"}): ContactPairKind.ROBOT_SUPPORT,
    frozenset({"robot", "target"}): ContactPairKind.TARGET_ROBOT,
    frozenset({"non_target", "support"}): ContactPairKind.NON_TARGET_SUPPORT,
    frozenset({"non_target", "robot"}): ContactPairKind.NON_TARGET_ROBOT,
    frozenset({"non_target", "target"}): ContactPairKind.NON_TARGET_TARGET,
    frozenset({"non_target"}): ContactPairKind.NON_TARGET_NON_TARGET,
    frozenset({"robot"}): ContactPairKind.ROBOT_SELF,
    frozenset({"target"}): ContactPairKind.TARGET_TARGET,
    frozenset({"support"}): ContactPairKind.SUPPORT_SUPPORT,
}


def classify_pair(roles: SceneRoles, geom_a: int, geom_b: int) -> ContactPairKind:
    """Say exactly which two roles are touching.

    This is identity, not judgement: whether the contact is permitted is a
    separate question, answered by :func:`classify_contact` once the roles and
    the measured forces are both known.
    """
    pair_roles = frozenset({roles.role_of(geom_a), roles.role_of(geom_b)})
    if "unknown" in pair_roles:
        return ContactPairKind.UNKNOWN
    return _PAIR_KINDS.get(pair_roles, ContactPairKind.UNKNOWN)


def classify_contact(roles: SceneRoles, geom_a: int, geom_b: int) -> ContactClass:
    """Assign a contact class from the roles of its two geoms.

    ``DAMAGING`` is deliberately not produced here: it is a budget verdict, not
    a geometric one, and is applied after the forces are measured.
    """
    pair = (min(geom_a, geom_b), max(geom_a, geom_b))
    if pair in roles.forbidden_pairs:
        return ContactClass.FORBIDDEN

    kind = classify_pair(roles, geom_a, geom_b)
    if kind is ContactPairKind.UNKNOWN:
        # An unclassified geom cannot be waved through: the safety budget is
        # only meaningful over geoms whose role is known.
        return ContactClass.FORBIDDEN
    if kind is ContactPairKind.ROBOT_SELF:
        return (
            ContactClass.SELF_CONTACT_ALLOWED
            if pair in roles.self_contact_allowlist
            else ContactClass.FORBIDDEN
        )
    if kind in (ContactPairKind.TARGET_ROBOT, ContactPairKind.TARGET_TARGET):
        return ContactClass.TARGET_INTENTIONAL
    if kind in (
        ContactPairKind.TARGET_SUPPORT,
        ContactPairKind.ROBOT_SUPPORT,
        ContactPairKind.SUPPORT_SUPPORT,
    ):
        return ContactClass.SUPPORT_ASSISTED
    if kind in (
        ContactPairKind.NON_TARGET_SUPPORT,
        ContactPairKind.NON_TARGET_ROBOT,
        ContactPairKind.NON_TARGET_TARGET,
        ContactPairKind.NON_TARGET_NON_TARGET,
    ):
        return ContactClass.NEIGHBOR_INCIDENTAL
    return ContactClass.FORBIDDEN


def _point_velocity(
    model: mujoco.MjModel, data: mujoco.MjData, body_id: int, point: np.ndarray
) -> np.ndarray:
    """World-frame velocity of a material point on a body."""
    velocity = np.zeros(6)
    mujoco.mj_objectVelocity(
        model, data, int(mujoco.mjtObj.mjOBJ_BODY), int(body_id), velocity, 0
    )
    angular, linear = velocity[:3], velocity[3:]
    lever = np.asarray(point, dtype=np.float64) - data.xpos[body_id]
    return linear + np.cross(angular, lever)


def _quaternion_angle(a: np.ndarray, b: np.ndarray) -> float:
    """Absolute rotation angle between two unit quaternions, in radians."""
    dot = float(np.clip(abs(float(np.dot(a, b))), -1.0, 1.0))
    return float(2.0 * np.arccos(dot))


def budget_margin(
    budget: ContactSafetyBudget,
    *,
    normal_force_N: float,
    tangential_force_N: float,
    normal_impulse_Ns: float,
    tangential_impulse_Ns: float,
    penetration_m: float,
    work_J: float,
    duration_s: float,
) -> float:
    """Smallest fractional headroom across every contact-scope quantity.

    Normalised so quantities in different units are comparable; negative means
    at least one limit is blown, and the magnitude says by how much. This covers
    the seven contact-scope limits only; the wrist, actuation and non-target
    limits are trajectory-scope and are evaluated by
    :meth:`ContactObserver.evaluation`.
    """
    ratios = (
        normal_force_N / budget.peak_normal_force_N,
        tangential_force_N / budget.peak_tangential_force_N,
        normal_impulse_Ns / budget.normal_impulse_Ns,
        tangential_impulse_Ns / budget.tangential_impulse_Ns,
        penetration_m / budget.max_penetration_m,
        work_J / budget.contact_work_J,
        duration_s / budget.contact_duration_s,
    )
    return float(1.0 - max(ratios))


@dataclasses.dataclass
class _Episode:
    """One uninterrupted contact between the same pair of geoms.

    A recontact starts a new episode rather than resuming the old one: duration
    and work are properties of a contact, and inheriting them across a gap would
    make a hand that touched twice look like one that never let go.
    """

    index: int
    duration_s: float = 0.0
    work_J: float = 0.0
    normal_impulse_Ns: float = 0.0
    tangential_impulse_Ns: float = 0.0
    last_step: int = -1
    #: (timestamp, normal impulse increment, tangential impulse increment)
    window: deque[tuple[float, float, float]] = dataclasses.field(default_factory=deque)

    def windowed(self) -> tuple[float, float]:
        normal = sum(entry[1] for entry in self.window)
        tangential = sum(entry[2] for entry in self.window)
        return float(normal), float(tangential)


class ContactObserver:
    """Turns MuJoCo contacts into classified, budget-checked events.

    The observer carries accumulator state, because impulse, work and duration
    are properties of a contact over time rather than of a single step.

    Impulse is judged over a **rolling** window with a documented endpoint
    convention: an increment recorded at time ``t`` is inside the window while
    ``t > now - impulse_window_s``, and the verdict for a step is computed after
    that step's increment has been added. That makes the verdict independent of
    where an impact falls relative to a window boundary, which the previous
    block accumulator was not (blocker B-02).
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        roles: SceneRoles,
        budget: ContactSafetyBudget,
        *,
        enforce_coverage: bool = True,
    ) -> None:
        self._model = model
        self._roles = roles
        self._budget = budget
        self._scopes = self._available_scopes(model, roles)
        if enforce_coverage:
            require_full_coverage(budget, self._scopes)

        self._episodes: dict[tuple[int, int], _Episode] = {}
        self._closed_episodes: dict[tuple[int, int], int] = {}
        self._step = 0
        self._elapsed_s = 0.0

        self._non_target_bodies = sorted(
            {int(model.geom_bodyid[g]) for g in roles.non_target_geoms}
        )
        self._initial_pose: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        self._peak: dict[str, float] = {}

    # -- construction helpers ------------------------------------------

    @staticmethod
    def _available_scopes(model: mujoco.MjModel, roles: SceneRoles) -> frozenset[SensorScope]:
        """Which sensor scopes this compiled model can actually supply."""
        scopes = {SensorScope.CONTACT, SensorScope.NON_TARGET}
        if roles.wrist_body is not None:
            scopes.add(SensorScope.WRIST)
        if int(model.nu) > 0 and ContactObserver._tendons_are_actuated(model):
            scopes.add(SensorScope.ACTUATION)
        return frozenset(scopes)

    @staticmethod
    def _tendons_are_actuated(model: mujoco.MjModel) -> bool:
        """True when every tendon is driven by an actuator we can read.

        A tendon with no actuator carries load nothing reports, so a budget that
        claims to bound tendon load against such a model cannot be enforced.
        """
        if int(model.ntendon) == 0:
            return True
        driven = {
            int(model.actuator_trnid[i, 0])
            for i in range(int(model.nu))
            if int(model.actuator_trntype[i]) == int(mujoco.mjtTrn.mjTRN_TENDON)
        }
        return all(index in driven for index in range(int(model.ntendon)))

    # -- properties ----------------------------------------------------

    @property
    def budget(self) -> ContactSafetyBudget:
        return self._budget

    @property
    def available_scopes(self) -> frozenset[SensorScope]:
        return self._scopes

    @property
    def elapsed_s(self) -> float:
        return self._elapsed_s

    def reset(self, data: mujoco.MjData | None = None) -> None:
        """Clear accumulators and pin the reference pose for non-target motion."""
        self._episodes.clear()
        self._closed_episodes.clear()
        self._peak.clear()
        self._initial_pose.clear()
        self._step = 0
        self._elapsed_s = 0.0
        if data is not None:
            self._capture_initial_pose(data)

    def _capture_initial_pose(self, data: mujoco.MjData) -> None:
        for body in self._non_target_bodies:
            self._initial_pose[body] = (
                np.array(data.xpos[body], dtype=np.float64),
                np.array(data.xquat[body], dtype=np.float64),
            )

    def _record_peak(self, field: str, value: float) -> None:
        if not np.isfinite(value):
            return
        self._peak[field] = max(self._peak.get(field, 0.0), float(value))

    # -- observation ---------------------------------------------------

    def observe(
        self,
        data: mujoco.MjData,
        *,
        time_index: int,
        dt: float,
        simulator_step: int = -1,
        accumulate: bool = True,
    ) -> tuple[ContactEvent, ...]:
        """Read every active contact for this step and accumulate its history.

        ``dt`` must be the **simulator** timestep, not a control period: the
        observer integrates impulse and work with it, so a control period feeds
        it an interval several times longer than the one that was simulated.

        ``accumulate=False`` reads the current contacts without advancing any
        accumulator. A controller that peeks at contacts before stepping needs
        that, and the previous version instead charged the same interval twice
        -- once on the peek and once on the record.
        """
        if dt <= 0.0:
            raise ValueError(f"dt must be positive, got {dt}")
        if not self._initial_pose and self._non_target_bodies:
            self._capture_initial_pose(data)

        if accumulate:
            self._step += 1
            self._elapsed_s += float(dt)
        now = self._elapsed_s

        readings = self._read_contacts(data)
        if accumulate:
            self._accumulate(readings, dt=dt, now=now)
            self._close_stale_episodes(present=set(readings))
            self._measure_scene(data)

        return self._build_events(
            readings, time_index=time_index, simulator_step=simulator_step, dt=float(dt)
        )

    def _read_contacts(self, data: mujoco.MjData) -> dict[tuple[int, int], list[dict[str, object]]]:
        """Group this step's contact points by geom pair.

        Grouping matters: several contact points on one geom pair are one
        contact for the purposes of duration, and charging each of them a full
        timestep would make a flat resting hand look like it had been holding on
        several times longer than it had.
        """
        readings: dict[tuple[int, int], list[dict[str, object]]] = {}
        wrench = np.zeros(6)
        for index in range(int(data.ncon)):
            contact = data.contact[index]
            geom_a, geom_b = int(contact.geom1), int(contact.geom2)
            key = (min(geom_a, geom_b), max(geom_a, geom_b))

            mujoco.mj_contactForce(self._model, data, index, wrench)
            # ``np.asarray`` on a MuJoCo field returns a view into the live
            # solver buffer, so a stored event would keep changing as the
            # rollout advanced -- and end up holding whatever occupied that
            # contact slot at the end. These are copies on purpose.
            frame = np.array(contact.frame, dtype=np.float64).reshape(3, 3)
            point = np.array(contact.pos, dtype=np.float64)

            normal_force = abs(float(wrench[0]))
            tangential_force = float(np.linalg.norm(wrench[1:3]))
            # MuJoCo reports a negative distance while the geoms interpenetrate.
            penetration = max(0.0, -float(contact.dist))

            body_a = int(self._model.geom_bodyid[geom_a])
            body_b = int(self._model.geom_bodyid[geom_b])
            relative = _point_velocity(self._model, data, body_b, point) - _point_velocity(
                self._model, data, body_a, point
            )
            normal_axis = frame[0]
            tangential_velocity = relative - normal_axis * float(relative @ normal_axis)
            slip_rate = float(np.linalg.norm(tangential_velocity))

            readings.setdefault(key, []).append(
                {
                    "geom_a": geom_a,
                    "geom_b": geom_b,
                    "body_a": body_a,
                    "body_b": body_b,
                    "point": point,
                    "frame": frame,
                    "normal_force": normal_force,
                    "tangential_force": tangential_force,
                    "penetration": penetration,
                    "relative": relative,
                    "slip_rate": slip_rate,
                }
            )
        return readings

    def _accumulate(
        self,
        readings: dict[tuple[int, int], list[dict[str, object]]],
        *,
        dt: float,
        now: float,
    ) -> None:
        window_s = float(self._budget.impulse_window_s)
        for key, points in readings.items():
            episode = self._episodes.get(key)
            if episode is None:
                episode = _Episode(index=self._closed_episodes.get(key, 0))
                self._episodes[key] = episode

            pair_normal = sum(float(p["normal_force"]) for p in points)
            pair_tangential = sum(float(p["tangential_force"]) for p in points)
            pair_work = sum(
                float(p["tangential_force"]) * float(p["slip_rate"]) * dt for p in points
            )

            normal_increment = pair_normal * dt
            tangential_increment = pair_tangential * dt
            episode.normal_impulse_Ns += normal_increment
            episode.tangential_impulse_Ns += tangential_increment
            episode.work_J += pair_work
            # Duration is charged once per step per pair, however many contact
            # points the solver produced for it.
            episode.duration_s += dt
            episode.last_step = self._step

            episode.window.append((now, normal_increment, tangential_increment))
            cutoff = now - window_s
            while episode.window and episode.window[0][0] <= cutoff:
                episode.window.popleft()

            for point in points:
                self._record_peak("peak_normal_force_N", float(point["normal_force"]))
                self._record_peak("peak_tangential_force_N", float(point["tangential_force"]))
                self._record_peak("max_penetration_m", float(point["penetration"]))

            windowed_normal, windowed_tangential = episode.windowed()
            self._record_peak("normal_impulse_Ns", windowed_normal)
            self._record_peak("tangential_impulse_Ns", windowed_tangential)
            self._record_peak("contact_duration_s", episode.duration_s)
            self._record_peak("contact_work_J", episode.work_J)

    def _close_stale_episodes(self, *, present: set[tuple[int, int]]) -> None:
        for key in [k for k in self._episodes if k not in present]:
            episode = self._episodes.pop(key)
            self._closed_episodes[key] = episode.index + 1

    def _measure_scene(self, data: mujoco.MjData) -> None:
        """Read the six trajectory-scope limits v1 declared but never measured."""
        if SensorScope.WRIST in self._scopes and self._roles.wrist_body is not None:
            wrench = np.asarray(data.cfrc_ext[int(self._roles.wrist_body)], dtype=np.float64)
            self._record_peak("max_wrist_torque_Nm", float(np.linalg.norm(wrench[:3])))
            self._record_peak("max_wrist_force_N", float(np.linalg.norm(wrench[3:])))

        if SensorScope.ACTUATION in self._scopes and int(self._model.nu) > 0:
            loads = np.abs(np.asarray(data.actuator_force, dtype=np.float64))
            if loads.size:
                self._record_peak("max_joint_or_tendon_load", float(np.max(loads)))

        # With no neighbouring objects these are genuinely zero, which is a
        # measurement; they are only unavailable when a sensor is missing.
        self._peak.setdefault("max_non_target_translation_m", 0.0)
        self._peak.setdefault("max_non_target_rotation_rad", 0.0)
        self._peak.setdefault("max_non_target_velocity_mps", 0.0)
        velocity = np.zeros(6)
        for body in self._non_target_bodies:
            start_pos, start_quat = self._initial_pose.get(
                body, (np.array(data.xpos[body]), np.array(data.xquat[body]))
            )
            self._record_peak(
                "max_non_target_translation_m",
                float(np.linalg.norm(np.asarray(data.xpos[body]) - start_pos)),
            )
            self._record_peak(
                "max_non_target_rotation_rad",
                _quaternion_angle(np.asarray(data.xquat[body], dtype=np.float64), start_quat),
            )
            mujoco.mj_objectVelocity(
                self._model, data, int(mujoco.mjtObj.mjOBJ_BODY), int(body), velocity, 0
            )
            self._record_peak("max_non_target_velocity_mps", float(np.linalg.norm(velocity[3:])))

    def _build_events(
        self,
        readings: dict[tuple[int, int], list[dict[str, object]]],
        *,
        time_index: int,
        simulator_step: int,
        dt: float,
    ) -> tuple[ContactEvent, ...]:
        events: list[ContactEvent] = []
        for key, points in readings.items():
            episode = self._episodes.get(key)
            windowed_normal, windowed_tangential = (
                episode.windowed() if episode is not None else (0.0, 0.0)
            )
            duration = episode.duration_s if episode is not None else 0.0
            work = episode.work_J if episode is not None else 0.0
            cumulative_normal = episode.normal_impulse_Ns if episode is not None else 0.0
            cumulative_tangential = episode.tangential_impulse_Ns if episode is not None else 0.0
            episode_index = episode.index if episode is not None else 0

            for point in points:
                geom_a = int(point["geom_a"])
                geom_b = int(point["geom_b"])
                pair_kind = classify_pair(self._roles, geom_a, geom_b)
                contact_class = classify_contact(self._roles, geom_a, geom_b)
                margin = budget_margin(
                    self._budget,
                    normal_force_N=float(point["normal_force"]),
                    tangential_force_N=float(point["tangential_force"]),
                    normal_impulse_Ns=windowed_normal,
                    tangential_impulse_Ns=windowed_tangential,
                    penetration_m=float(point["penetration"]),
                    work_J=work,
                    duration_s=duration,
                )
                # A permitted contact that blows the budget becomes damaging.
                # The class is a safety verdict once forces are known, not just
                # geometry.
                if margin < 0.0 and contact_class is not ContactClass.FORBIDDEN:
                    contact_class = ContactClass.DAMAGING

                relative = np.asarray(point["relative"], dtype=np.float64)
                events.append(
                    ContactEvent(
                        time_index=time_index,
                        contact_class=contact_class,
                        geom_a=self._geom_name(geom_a),
                        geom_b=self._geom_name(geom_b),
                        body_a=self._body_name(int(point["body_a"])),
                        body_b=self._body_name(int(point["body_b"])),
                        point=np.asarray(point["point"], dtype=np.float64),
                        frame=np.asarray(point["frame"], dtype=np.float64),
                        normal_force_N=float(point["normal_force"]),
                        tangential_force_N=float(point["tangential_force"]),
                        normal_impulse_Ns=cumulative_normal,
                        tangential_impulse_Ns=cumulative_tangential,
                        penetration_m=float(point["penetration"]),
                        relative_velocity_mps=float(np.linalg.norm(relative)),
                        slip_m=float(point["slip_rate"]) * dt,
                        work_J=work,
                        budget_margin=margin,
                        duration_s=duration,
                        link_class=self._roles.role_of(geom_a)
                        if geom_a in self._roles.robot_geoms
                        else self._roles.role_of(geom_b),
                        pair_kind=pair_kind,
                        simulator_step=int(simulator_step),
                        episode_index=int(episode_index),
                    )
                )
        return tuple(events)

    # -- verdict -------------------------------------------------------

    @property
    def measurements(self) -> dict[str, float]:
        """Every limit's measured value, by budget field name."""
        return dict(self._peak)

    @property
    def evaluation(self) -> SafetyEvaluation:
        """Check every declared limit against what was actually measured."""
        return evaluate_budget(self._budget, self._peak)

    def _geom_name(self, geom_id: int) -> str:
        name = mujoco.mj_id2name(self._model, int(mujoco.mjtObj.mjOBJ_GEOM), geom_id)
        return name if name else f"geom_{geom_id}"

    def _body_name(self, body_id: int) -> str:
        name = mujoco.mj_id2name(self._model, int(mujoco.mjtObj.mjOBJ_BODY), body_id)
        return name if name else f"body_{body_id}"


def summarise_safety(
    events: Sequence[ContactEvent],
) -> tuple[dict[str, float], dict[str, float]]:
    """Peak and cumulative safety metrics over a trajectory's contact stream."""
    if not events:
        return ({}, {})
    peak = {
        "peak_normal_force_N": max(e.normal_force_N for e in events),
        "peak_tangential_force_N": max(e.tangential_force_N for e in events),
        "max_penetration_m": max(e.penetration_m for e in events),
        "min_budget_margin": min(e.budget_margin for e in events),
    }
    # Impulse, work and duration already accumulate inside the observer, so the
    # cumulative figure is the final value per contact episode, not a sum of
    # steps. Episodes are keyed separately so a recontact is not folded into the
    # episode before it.
    latest: dict[tuple[str, str, int], ContactEvent] = {}
    for event in events:
        latest[(event.geom_a, event.geom_b, event.episode_index)] = event
    cumulative = {
        "normal_impulse_Ns": sum(e.normal_impulse_Ns for e in latest.values()),
        "tangential_impulse_Ns": sum(e.tangential_impulse_Ns for e in latest.values()),
        "contact_work_J": sum(e.work_J for e in latest.values()),
        "total_slip_m": sum(e.slip_m for e in events),
    }
    return (peak, cumulative)


__all__ = [
    "ContactObserver",
    "ContactPairKind",
    "SafetyCoverageError",
    "SafetyEvaluation",
    "SceneRoles",
    "budget_margin",
    "classify_contact",
    "classify_pair",
    "summarise_safety",
]
