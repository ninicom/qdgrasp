"""Every declared safety limit, and the sensor that actually measures it (G01).

The first version of the safety verdict checked seven of the thirteen limits the
budget declares. The other six -- wrist force, wrist torque, joint or tendon
load, and the translation, rotation and velocity a non-target object is allowed
to pick up -- were written down, hashed into manifests, and never measured. A
limit nobody measures is worse than no limit, because it reads as a guarantee
(blocker B-01).

So this module states, for each field, four things that used to be implicit:

* the **sensor** it is read from, by name, so a reviewer can go and look;
* the **unit**, so the number means something;
* the **aggregation** -- peak, windowed, per-contact cumulative or per-object
  cumulative -- because one number cannot stand for four different semantics;
* the **failure reason** it produces, so the ledger reconciles.

A budget field with no spec, or a spec whose sensor is not available on the
compiled model, fails preflight. It does not quietly evaluate to "fine".
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from enum import Enum

import numpy as np

from qdgrasp.dataset.dynamic_contracts import ContactSafetyBudget


class Aggregation(str, Enum):
    """How a measurement is reduced before it meets its threshold.

    These are four different questions. ``peak`` asks how hard, ever;
    ``windowed`` asks how much impulse arrived inside one rolling window;
    ``per_contact_cumulative`` asks how much a single contact episode added up
    to; ``per_object_cumulative`` asks how far one object was pushed in total.
    v1 used one accumulator for several of them at once.
    """

    PEAK = "peak"
    WINDOWED = "windowed"
    PER_CONTACT_CUMULATIVE = "per_contact_cumulative"
    PER_OBJECT_CUMULATIVE = "per_object_cumulative"


class SensorScope(str, Enum):
    """Which part of the scene a sensor reads."""

    CONTACT = "contact"
    WRIST = "wrist"
    ACTUATION = "actuation"
    NON_TARGET = "non_target"


class SafetyCoverageError(RuntimeError):
    """A declared limit has no sensor behind it on this model.

    Raised at preflight rather than at verdict time: discovering mid-rollout
    that a limit was never measured means every result before it is suspect.
    """


@dataclasses.dataclass(frozen=True)
class BudgetFieldSpec:
    """What measures one budget field, in what unit, reduced how."""

    field: str
    sensor: str
    unit: str
    aggregation: Aggregation
    scope: SensorScope
    failure_reason: str
    description: str

    def as_dict(self) -> dict[str, str]:
        return {
            "field": self.field,
            "sensor": self.sensor,
            "unit": self.unit,
            "aggregation": self.aggregation.value,
            "scope": self.scope.value,
            "failure_reason": self.failure_reason,
        }


def _spec(
    field: str,
    sensor: str,
    unit: str,
    aggregation: Aggregation,
    scope: SensorScope,
    failure_reason: str,
    description: str,
) -> tuple[str, BudgetFieldSpec]:
    return field, BudgetFieldSpec(
        field=field,
        sensor=sensor,
        unit=unit,
        aggregation=aggregation,
        scope=scope,
        failure_reason=failure_reason,
        description=description,
    )


#: One entry per limit the budget declares. The coverage test asserts that this
#: mapping is total: adding a limit without adding its sensor fails the build.
SAFETY_FIELD_SPECS: Mapping[str, BudgetFieldSpec] = dict(
    (
        _spec(
            "peak_normal_force_N",
            "mj_contactForce[0]",
            "N",
            Aggregation.PEAK,
            SensorScope.CONTACT,
            "damaging_contact",
            "Largest normal force any single contact ever carried.",
        ),
        _spec(
            "peak_tangential_force_N",
            "mj_contactForce[1:3]",
            "N",
            Aggregation.PEAK,
            SensorScope.CONTACT,
            "damaging_contact",
            "Largest friction force any single contact ever carried.",
        ),
        _spec(
            "normal_impulse_Ns",
            "mj_contactForce[0] integrated over impulse_window_s",
            "N*s",
            Aggregation.WINDOWED,
            SensorScope.CONTACT,
            "damaging_contact",
            "Normal impulse delivered inside one rolling window; catches impacts.",
        ),
        _spec(
            "tangential_impulse_Ns",
            "mj_contactForce[1:3] integrated over impulse_window_s",
            "N*s",
            Aggregation.WINDOWED,
            SensorScope.CONTACT,
            "damaging_contact",
            "Friction impulse delivered inside one rolling window.",
        ),
        _spec(
            "contact_duration_s",
            "simulator dt accumulated per contact episode",
            "s",
            Aggregation.PER_CONTACT_CUMULATIVE,
            SensorScope.CONTACT,
            "damaging_contact",
            "How long one uninterrupted contact episode lasted.",
        ),
        _spec(
            "contact_work_J",
            "tangential force * slip speed * dt per contact episode",
            "J",
            Aggregation.PER_CONTACT_CUMULATIVE,
            SensorScope.CONTACT,
            "damaging_contact",
            "Frictional work one contact episode put into the surfaces.",
        ),
        _spec(
            "max_penetration_m",
            "-mjContact.dist",
            "m",
            Aggregation.PEAK,
            SensorScope.CONTACT,
            "damaging_contact",
            "Deepest interpenetration the solver allowed.",
        ),
        _spec(
            "max_wrist_force_N",
            "mjData.cfrc_ext[wrist_body][3:6]",
            "N",
            Aggregation.PEAK,
            SensorScope.WRIST,
            "damaging_contact",
            "Largest external force resolved at the wrist body.",
        ),
        _spec(
            "max_wrist_torque_Nm",
            "mjData.cfrc_ext[wrist_body][0:3]",
            "N*m",
            Aggregation.PEAK,
            SensorScope.WRIST,
            "damaging_contact",
            "Largest external torque resolved at the wrist body.",
        ),
        _spec(
            "max_joint_or_tendon_load",
            "mjData.actuator_force",
            "N or N*m",
            Aggregation.PEAK,
            SensorScope.ACTUATION,
            "damaging_contact",
            "Largest actuator or tendon load commanded through the transmission.",
        ),
        _spec(
            "max_non_target_translation_m",
            "mjData.xpos[non_target_body] - initial",
            "m",
            Aggregation.PER_OBJECT_CUMULATIVE,
            SensorScope.NON_TARGET,
            "non_target_disturbance",
            "How far a neighbouring object was displaced from where it started.",
        ),
        _spec(
            "max_non_target_rotation_rad",
            "angle(mjData.xquat[non_target_body], initial)",
            "rad",
            Aggregation.PER_OBJECT_CUMULATIVE,
            SensorScope.NON_TARGET,
            "non_target_disturbance",
            "How far a neighbouring object was rotated from where it started.",
        ),
        _spec(
            "max_non_target_velocity_mps",
            "mj_objectVelocity(non_target_body)[3:6]",
            "m/s",
            Aggregation.PEAK,
            SensorScope.NON_TARGET,
            "non_target_disturbance",
            "Fastest a neighbouring object was ever made to move.",
        ),
    )
)


def declared_limit_fields(budget: ContactSafetyBudget) -> tuple[str, ...]:
    """Every limit the budget declares, in declaration order."""
    return budget.limit_fields


def missing_specs(budget: ContactSafetyBudget) -> tuple[str, ...]:
    """Declared limits with no sensor mapping at all."""
    return tuple(field for field in declared_limit_fields(budget) if field not in SAFETY_FIELD_SPECS)


def require_full_coverage(budget: ContactSafetyBudget, available_scopes: frozenset[SensorScope]) -> None:
    """Fail preflight unless every declared limit can actually be measured.

    ``available_scopes`` is what the compiled model and the scene roles can
    supply. A model with no identified wrist body cannot measure wrist load, and
    a budget that declares a wrist limit against it is not enforceable.
    """
    unmapped = missing_specs(budget)
    if unmapped:
        raise SafetyCoverageError(
            f"budget {budget.budget_id!r} declares limits with no sensor mapping: "
            f"{sorted(unmapped)}. A limit nobody measures reads as a guarantee."
        )
    unavailable = sorted(
        {
            field
            for field in declared_limit_fields(budget)
            if SAFETY_FIELD_SPECS[field].scope not in available_scopes
        }
    )
    if unavailable:
        missing_scopes = sorted({SAFETY_FIELD_SPECS[f].scope.value for f in unavailable})
        raise SafetyCoverageError(
            f"budget {budget.budget_id!r} declares {unavailable}, but this model "
            f"provides no sensor for {missing_scopes}. Identify the missing bodies "
            "or use a budget that does not claim those limits."
        )


@dataclasses.dataclass(frozen=True)
class SafetyEvaluation:
    """The safety verdict, with its working shown.

    ``min_margin_field`` names which limit produced the tightest headroom, so a
    result can be argued with rather than only believed.
    """

    measured_fields: tuple[str, ...]
    unavailable_fields: tuple[str, ...]
    violated_fields: tuple[str, ...]
    measurements: Mapping[str, float]
    margins: Mapping[str, float]
    min_margin: float
    min_margin_field: str
    budget_id: str
    budget_hash: str

    @property
    def safe(self) -> bool:
        """A positive is only valid when every limit was measured and cleared."""
        return not self.violated_fields and not self.unavailable_fields

    @property
    def failure_reasons(self) -> tuple[str, ...]:
        return tuple(
            sorted({SAFETY_FIELD_SPECS[field].failure_reason for field in self.violated_fields})
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "budget_id": self.budget_id,
            "budget_hash": self.budget_hash,
            "measured_fields": list(self.measured_fields),
            "unavailable_fields": list(self.unavailable_fields),
            "violated_fields": list(self.violated_fields),
            "measurements": {k: float(v) for k, v in self.measurements.items()},
            "margins": {k: float(v) for k, v in self.margins.items()},
            "min_margin": float(self.min_margin),
            "min_margin_field": self.min_margin_field,
            "safe": self.safe,
        }


def evaluate_budget(
    budget: ContactSafetyBudget,
    measurements: Mapping[str, float],
) -> SafetyEvaluation:
    """Check every declared limit against what was measured.

    A field that was never measured lands in ``unavailable_fields`` and makes
    the evaluation unsafe. It is not treated as zero: an unmeasured quantity is
    unknown, and unknown is not within budget.
    """
    measured: list[str] = []
    unavailable: list[str] = []
    violated: list[str] = []
    margins: dict[str, float] = {}
    values: dict[str, float] = {}

    for field in declared_limit_fields(budget):
        if field not in SAFETY_FIELD_SPECS:
            unavailable.append(field)
            continue
        if field not in measurements:
            unavailable.append(field)
            continue
        value = float(measurements[field])
        limit = float(getattr(budget, field))
        if not np.isfinite(value):
            unavailable.append(field)
            continue
        measured.append(field)
        values[field] = value
        margin = 1.0 - (value / limit)
        margins[field] = margin
        if margin < 0.0:
            violated.append(field)

    if margins:
        min_field = min(margins, key=lambda name: margins[name])
        min_margin = margins[min_field]
    else:
        min_field = ""
        min_margin = float("-inf")

    return SafetyEvaluation(
        measured_fields=tuple(measured),
        unavailable_fields=tuple(sorted(unavailable)),
        violated_fields=tuple(sorted(violated)),
        measurements=values,
        margins=margins,
        min_margin=float(min_margin),
        min_margin_field=min_field,
        budget_id=budget.budget_id,
        budget_hash=budget.budget_hash,
    )


def coverage_matrix(budget: ContactSafetyBudget) -> dict[str, dict[str, str]]:
    """The field-to-sensor table, for the evidence packet."""
    return {
        field: SAFETY_FIELD_SPECS[field].as_dict()
        for field in declared_limit_fields(budget)
        if field in SAFETY_FIELD_SPECS
    }
