"""Corrective track for the 2026-09-01 cross-component audit (``PLAN.md`` §9).

Two things live here: the registry of reproduced failure chains that the
characterization tests key off, and the G0 hard stop that keeps public training
off a corpus which fails its own audit or positive gate.
"""

from __future__ import annotations

from .gate import (
    CORRECTIVE_GATE_SCHEMA,
    CheckResult,
    CorrectiveGateError,
    GateReport,
    assert_public_training_allowed,
    evaluate,
)
from .registry import (
    AUDIT_SESSION,
    CORRECTIVE_REGISTRY_SCHEMA,
    FINDINGS,
    FINDINGS_BY_ID,
    PLAN_REVISION,
    PLANNED_SCHEMA_BUMPS,
    REVISION_RECORD,
    CorrectiveError,
    Finding,
    SchemaBump,
    get,
    open_findings,
    release_is_blocked,
    summary,
)

__all__ = [
    "AUDIT_SESSION",
    "CORRECTIVE_GATE_SCHEMA",
    "CORRECTIVE_REGISTRY_SCHEMA",
    "FINDINGS",
    "FINDINGS_BY_ID",
    "PLANNED_SCHEMA_BUMPS",
    "PLAN_REVISION",
    "REVISION_RECORD",
    "CheckResult",
    "CorrectiveError",
    "CorrectiveGateError",
    "Finding",
    "GateReport",
    "SchemaBump",
    "assert_public_training_allowed",
    "evaluate",
    "get",
    "open_findings",
    "release_is_blocked",
    "summary",
]
