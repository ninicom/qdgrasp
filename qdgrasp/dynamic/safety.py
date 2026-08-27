"""Contact observation and multi-quantity safety accounting (P3.4-06).

Static grasp asks one question of a contact: does it exist. That is enough when
every non-target contact is a rejection. Phase 3.4 permits table and neighbour
contact, so it has to ask how hard, for how long, how much it slipped and how
much energy went in -- and answer per contact, over time.

Everything here is measured from MuJoCo's own solver output. Nothing is
predicted, and no contact is admitted on geometry alone.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence

import mujoco
import numpy as np

from qdgrasp.dataset.dynamic_contracts import (
    ContactClass,
    ContactEvent,
    ContactSafetyBudget,
)


@dataclasses.dataclass(frozen=True)
class SceneRoles:
    """Which geoms play which part, by geom id.

    Classification is by identity, not by name substring: a substring such as
    "table" or "obj" is not a reliable label across scene sources.
    """

    target_geoms: frozenset[int]
    support_geoms: frozenset[int]
    non_target_geoms: frozenset[int]
    robot_geoms: frozenset[int]
    #: Robot geom pairs that may touch each other, as sorted 2-tuples.
    self_contact_allowlist: frozenset[tuple[int, int]] = frozenset()
    #: Pairs that are never permitted regardless of measured force.
    forbidden_pairs: frozenset[tuple[int, int]] = frozenset()

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


def classify_contact(roles: SceneRoles, geom_a: int, geom_b: int) -> ContactClass:
    """Assign a contact class from the roles of its two geoms.

    ``DAMAGING`` is deliberately not produced here: it is a budget verdict, not
    a geometric one, and is applied after the forces are measured.
    """
    pair = (min(geom_a, geom_b), max(geom_a, geom_b))
    if pair in roles.forbidden_pairs:
        return ContactClass.FORBIDDEN

    role_a, role_b = roles.role_of(geom_a), roles.role_of(geom_b)
    pair_roles = {role_a, role_b}

    if "unknown" in pair_roles:
        # An unclassified geom cannot be waved through: the safety budget is
        # only meaningful over geoms whose role is known.
        return ContactClass.FORBIDDEN

    if pair_roles == {"robot"}:
        return (
            ContactClass.SELF_CONTACT_ALLOWED
            if pair in roles.self_contact_allowlist
            else ContactClass.FORBIDDEN
        )
    if pair_roles == {"robot", "target"}:
        return ContactClass.TARGET_INTENTIONAL
    if "support" in pair_roles:
        return ContactClass.SUPPORT_ASSISTED
    if "non_target" in pair_roles:
        return ContactClass.NEIGHBOR_INCIDENTAL
    if pair_roles == {"target"}:
        return ContactClass.TARGET_INTENTIONAL
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
    """Smallest fractional headroom across every quantity of the budget.

    Normalised so quantities in different units are comparable; negative means
    at least one limit is blown, and the magnitude says by how much.
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


class ContactObserver:
    """Turns MuJoCo contacts into classified, budget-checked events.

    The observer carries accumulator state, because impulse, work and duration
    are properties of a contact over time rather than of a single step.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        roles: SceneRoles,
        budget: ContactSafetyBudget,
    ) -> None:
        self._model = model
        self._roles = roles
        self._budget = budget
        self._accumulated: dict[tuple[int, int], dict[str, float]] = {}

    @property
    def budget(self) -> ContactSafetyBudget:
        return self._budget

    def reset(self) -> None:
        self._accumulated.clear()

    def observe(
        self, data: mujoco.MjData, *, time_index: int, dt: float
    ) -> tuple[ContactEvent, ...]:
        """Read every active contact for this step and accumulate its history."""
        if dt <= 0.0:
            raise ValueError(f"dt must be positive, got {dt}")

        events: list[ContactEvent] = []
        wrench = np.zeros(6)
        for index in range(int(data.ncon)):
            contact = data.contact[index]
            geom_a, geom_b = int(contact.geom1), int(contact.geom2)
            key = (min(geom_a, geom_b), max(geom_a, geom_b))

            mujoco.mj_contactForce(self._model, data, index, wrench)
            frame = np.asarray(contact.frame, dtype=np.float64).reshape(3, 3)
            point = np.asarray(contact.pos, dtype=np.float64)

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

            state = self._accumulated.setdefault(
                key,
                {"normal_impulse": 0.0, "tangential_impulse": 0.0, "work": 0.0, "duration": 0.0},
            )
            state["normal_impulse"] += normal_force * dt
            state["tangential_impulse"] += tangential_force * dt
            # Frictional work: only the tangential force does work through slip.
            state["work"] += tangential_force * slip_rate * dt
            state["duration"] += dt

            contact_class = classify_contact(self._roles, geom_a, geom_b)
            margin = budget_margin(
                self._budget,
                normal_force_N=normal_force,
                tangential_force_N=tangential_force,
                normal_impulse_Ns=state["normal_impulse"],
                tangential_impulse_Ns=state["tangential_impulse"],
                penetration_m=penetration,
                work_J=state["work"],
                duration_s=state["duration"],
            )
            # A permitted contact that blows the budget becomes damaging. The
            # class is a safety verdict once forces are known, not just geometry.
            if margin < 0.0 and contact_class is not ContactClass.FORBIDDEN:
                contact_class = ContactClass.DAMAGING

            events.append(
                ContactEvent(
                    time_index=time_index,
                    contact_class=contact_class,
                    geom_a=self._geom_name(geom_a),
                    geom_b=self._geom_name(geom_b),
                    body_a=self._body_name(body_a),
                    body_b=self._body_name(body_b),
                    point=point,
                    frame=frame,
                    normal_force_N=normal_force,
                    tangential_force_N=tangential_force,
                    normal_impulse_Ns=state["normal_impulse"],
                    tangential_impulse_Ns=state["tangential_impulse"],
                    penetration_m=penetration,
                    relative_velocity_mps=float(np.linalg.norm(relative)),
                    slip_m=slip_rate * dt,
                    work_J=state["work"],
                    budget_margin=margin,
                    duration_s=state["duration"],
                    link_class=self._roles.role_of(geom_a)
                    if geom_a in self._roles.robot_geoms
                    else self._roles.role_of(geom_b),
                )
            )
        return tuple(events)

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
    # cumulative figure is the final value per contact pair, not a sum of steps.
    latest: dict[tuple[str, str], ContactEvent] = {}
    for event in events:
        latest[(event.geom_a, event.geom_b)] = event
    cumulative = {
        "normal_impulse_Ns": sum(e.normal_impulse_Ns for e in latest.values()),
        "tangential_impulse_Ns": sum(e.tangential_impulse_Ns for e in latest.values()),
        "contact_work_J": sum(e.work_J for e in latest.values()),
        "total_slip_m": sum(e.slip_m for e in events),
    }
    return (peak, cumulative)
