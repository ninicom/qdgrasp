"""Static resistance: how much of the declared disturbance a frozen grasp holds.

WRK-R1. The criterion this replaces compared a grasp-wrench-space margin, built
from unit primitive wrenches with torques scaled by ``1/characteristic_length``,
against the norm of a raw wrench mixing newtons with newton-metres. Those two
numbers do not share units, so their ordering carried no physical meaning and
neither did any pairing derived from it.

What follows is dimensionless by construction. It asks the only question a
frozen analysis can answer honestly: with these contacts, this friction and this
force budget, what multiple of the declared disturbance can the grasp balance
while holding the object against gravity? That multiple is ``alpha``. A grasp
resists the disturbance it will actually meet when ``alpha >= 1``.

The force bound matters as much as the cone. Without a cap on normal force, an
LP will happily squeeze arbitrarily hard and certify anything, so a resistance
factor computed without ``force_limits`` is not a physical certificate and this
module refuses to produce one.
"""

from __future__ import annotations

import dataclasses

import numpy as np

#: Sides of the linearised friction pyramid. Matches the force-closure certifier
#: so the two see the same cone.
CONE_EDGES: int = 8

#: Equilibrium residual tolerance, in the normalised units of the LP.
EQUILIBRIUM_TOLERANCE: float = 1e-6


@dataclasses.dataclass(frozen=True)
class ResistanceCertificate:
    """The frozen grasp's answer, with the numbers it was reached from."""

    alpha: float
    passed: bool
    contacts: int
    characteristic_length_m: float
    disturbance_wrench: tuple[float, float, float, float, float, float]
    force_limits_N: tuple[float, ...]
    equilibrium_residual: float
    status: str
    reason: str = ""


def _cone_edges(normal: np.ndarray, mu: float) -> np.ndarray:
    """Unit edge directions of the linearised friction cone about ``normal``."""
    normal = normal / np.linalg.norm(normal)
    helper = np.array([1.0, 0.0, 0.0]) if abs(normal[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    t1 = np.cross(normal, helper)
    t1 /= np.linalg.norm(t1)
    t2 = np.cross(normal, t1)
    angles = np.linspace(0.0, 2.0 * np.pi, CONE_EDGES, endpoint=False)
    return np.array(
        [normal + mu * (np.cos(a) * t1 + np.sin(a) * t2) for a in angles],
        dtype=np.float64,
    )


def certify_static_resistance(
    contact_points: np.ndarray,
    inward_normals: np.ndarray,
    centroid: np.ndarray,
    *,
    mass: float,
    mu: float,
    disturbance_wrench: np.ndarray,
    force_limits: np.ndarray | float,
    characteristic_length: float,
    gravity: np.ndarray | None = None,
) -> ResistanceCertificate:
    """Largest multiple of ``disturbance_wrench`` this frozen grasp can balance.

    ``characteristic_length`` divides every torque row, of the contact map and of
    both wrenches alike. That is a row scaling of one linear system, so it leaves
    ``alpha`` unchanged while making the six equations comparable in magnitude --
    and it is what makes ``alpha`` invariant when the whole scene is expressed in
    different length units.
    """
    from scipy.optimize import linprog

    points = np.asarray(contact_points, dtype=np.float64)
    normals = np.asarray(inward_normals, dtype=np.float64)
    centroid = np.asarray(centroid, dtype=np.float64)
    disturbance = np.asarray(disturbance_wrench, dtype=np.float64).reshape(6)
    gravity_vec = (
        np.array([0.0, 0.0, -9.81]) if gravity is None else np.asarray(gravity, dtype=np.float64)
    )

    count = 0 if points.ndim != 2 else int(points.shape[0])
    limits = (
        np.full(count, float(force_limits))
        if np.isscalar(force_limits)
        else np.asarray(force_limits, dtype=np.float64)
    )

    empty = tuple(float(v) for v in disturbance)
    if count == 0:
        return ResistanceCertificate(
            alpha=0.0, passed=False, contacts=0,
            characteristic_length_m=float(characteristic_length),
            disturbance_wrench=empty, force_limits_N=(),
            equilibrium_residual=float("inf"), status="no_contacts",
            reason="a frozen grasp with no contacts resists nothing",
        )
    if not np.all(np.isfinite(limits)) or np.any(limits <= 0.0):
        return ResistanceCertificate(
            alpha=0.0, passed=False, contacts=count,
            characteristic_length_m=float(characteristic_length),
            disturbance_wrench=empty, force_limits_N=tuple(float(v) for v in np.atleast_1d(limits)),
            equilibrium_residual=float("inf"), status="no_force_bound",
            reason=(
                "resistance without a normal-force cap is not a physical "
                "certificate: an unbounded LP squeezes as hard as it likes"
            ),
        )
    if float(np.linalg.norm(disturbance)) <= 0.0:
        return ResistanceCertificate(
            alpha=0.0, passed=False, contacts=count,
            characteristic_length_m=float(characteristic_length),
            disturbance_wrench=empty, force_limits_N=tuple(float(v) for v in limits),
            equilibrium_residual=float("inf"), status="no_disturbance",
            reason="alpha is a multiple of the disturbance; a zero disturbance has no multiple",
        )

    length = float(characteristic_length)
    if not np.isfinite(length) or length <= 0.0:
        length = 1.0

    # Columns: one per cone edge per contact, then alpha.
    columns: list[np.ndarray] = []
    normal_rows: list[np.ndarray] = []
    for index in range(count):
        edges = _cone_edges(normals[index], mu)
        arm = points[index] - centroid
        unit_normal = normals[index] / np.linalg.norm(normals[index])
        for edge in edges:
            torque = np.cross(arm, edge) / length
            columns.append(np.concatenate([edge, torque]))
        row = np.zeros(count * CONE_EDGES)
        row[index * CONE_EDGES : (index + 1) * CONE_EDGES] = edges @ unit_normal
        normal_rows.append(row)

    contact_map = np.array(columns, dtype=np.float64).T  # 6 x (K*E)
    disturbance_column = np.concatenate([disturbance[:3], disturbance[3:] / length])
    weight = np.concatenate([mass * gravity_vec, np.zeros(3)])
    weight_scaled = np.concatenate([weight[:3], weight[3:] / length])

    a_eq = np.hstack([contact_map, disturbance_column.reshape(6, 1)])
    b_eq = -weight_scaled

    a_ub = np.hstack([np.array(normal_rows), np.zeros((count, 1))])
    b_ub = limits

    objective = np.zeros(count * CONE_EDGES + 1)
    objective[-1] = -1.0  # maximise alpha

    bounds = [(0.0, None)] * (count * CONE_EDGES) + [(0.0, None)]
    solution = linprog(objective, A_ub=a_ub, b_ub=b_ub, A_eq=a_eq, b_eq=b_eq, bounds=bounds)

    if not solution.success:
        return ResistanceCertificate(
            alpha=0.0, passed=False, contacts=count,
            characteristic_length_m=length,
            disturbance_wrench=empty, force_limits_N=tuple(float(v) for v in limits),
            equilibrium_residual=float("inf"), status="infeasible",
            reason=f"no equilibrium exists for any positive alpha: {solution.message}",
        )

    alpha = float(solution.x[-1])
    residual = float(np.linalg.norm(a_eq @ solution.x - b_eq))
    return ResistanceCertificate(
        alpha=alpha,
        passed=alpha >= 1.0,
        contacts=count,
        characteristic_length_m=length,
        disturbance_wrench=empty,
        force_limits_N=tuple(float(v) for v in limits),
        equilibrium_residual=residual,
        status="solved",
        reason="" if alpha >= 1.0 else (
            f"resists {alpha:.4f} of the declared disturbance, short of the 1.0 it will meet"
        ),
    )
