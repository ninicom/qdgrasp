"""The registry, the plan and the tests have to keep saying the same thing.

``PLAN.md`` §9.2 is the normative list of findings; :mod:`qdgrasp.corrective`
mirrors it so tests can key off it; the characterization suite is what actually
reproduces each one.  Any two of those three agreeing is not enough -- a finding
dropped from the registry would silently un-mark its tests, and a test written
against an unregistered id would never flip.
"""

from __future__ import annotations

import importlib
import re
from pathlib import Path

import pytest

from qdgrasp.corrective import registry

REPO_ROOT = Path(__file__).resolve().parents[2]
PLAN = REPO_ROOT / "PLAN.md"
SUITE = Path(__file__).resolve().parent

_PLAN_ROW = re.compile(r"^\|\s*`(COR-\d\d)`\s*\|\s*([^|]+?)\s*\|")
_TEST_MARK = re.compile(r'@characterization\(\s*"(COR-\d\d)"')


def _plan_findings() -> dict[str, str]:
    rows: dict[str, str] = {}
    for line in PLAN.read_text(encoding="utf-8").splitlines():
        match = _PLAN_ROW.match(line)
        if match:
            rows[match.group(1)] = match.group(2)
    return rows


def _tested_findings() -> dict[str, list[str]]:
    found: dict[str, list[str]] = {}
    for path in sorted(SUITE.glob("test_*.py")):
        for finding_id in _TEST_MARK.findall(path.read_text(encoding="utf-8")):
            found.setdefault(finding_id, []).append(path.name)
    return found


def test_the_registry_lists_exactly_what_the_plan_lists() -> None:
    plan = _plan_findings()
    assert plan, "PLAN.md §9.2 has no parseable finding table"
    assert set(plan) == set(registry.FINDINGS_BY_ID), (
        f"plan lists {sorted(plan)}, registry lists {sorted(registry.FINDINGS_BY_ID)}"
    )


def test_the_registry_agrees_with_the_plan_on_severity() -> None:
    plan = _plan_findings()
    mismatched = {
        finding_id: (registry.get(finding_id).severity, severity)
        for finding_id, severity in plan.items()
        if registry.get(finding_id).severity != severity
    }
    assert not mismatched, f"registry/plan severity disagreement: {mismatched}"


def test_every_finding_has_at_least_one_characterization_test() -> None:
    tested = _tested_findings()
    uncovered = sorted(set(registry.FINDINGS_BY_ID) - set(tested))
    assert not uncovered, f"PLAN.md §9.3 requires a characterization test per finding; missing: {uncovered}"


def test_no_test_names_an_unregistered_finding() -> None:
    unknown = sorted(set(_tested_findings()) - set(registry.FINDINGS_BY_ID))
    assert not unknown, f"characterization tests name unregistered findings: {unknown}"


def test_every_finding_closes_at_a_declared_gate_and_pull_request() -> None:
    for finding in registry.FINDINGS:
        assert re.fullmatch(r"G[0-7]", finding.gate), finding.id
        assert re.fullmatch(r"R[1-9]", finding.pull_request), finding.id
        # R1 carries characterization and the hard stop only; §9.11 forbids
        # mixing a semantic fix into it.
        assert finding.pull_request != "R1", f"{finding.id} may not be closed by R1"


@pytest.mark.parametrize("bump", registry.PLANNED_SCHEMA_BUMPS, ids=lambda item: item.constant)
def test_a_planned_schema_bump_pins_the_version_in_the_code(bump: registry.SchemaBump) -> None:
    """The declaration is only useful while it still describes the code."""

    module = importlib.import_module(bump.module)
    actual = getattr(module, bump.constant)
    finding = registry.get(bump.finding)
    expected = bump.current if finding.is_open else bump.planned

    assert actual == expected, (
        f"{bump.module}.{bump.constant} is {actual!r}; with {finding.id} {finding.status} it should be "
        f"{expected!r}. A schema moves when the semantics move, and the registry has to move with it."
    )


def test_release_stays_blocked_while_any_finding_is_open() -> None:
    assert registry.release_is_blocked() == bool(registry.open_findings())
