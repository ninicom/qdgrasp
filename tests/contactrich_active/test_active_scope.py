"""S1 — ADR-0008 is enforced where a workload picks its hands (G05, B-10)."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from qdgrasp.config.active_scope import (
    ACTIVE_HANDS,
    DEFAULT_ROBOT_PROFILES,
    GOVERNING_DECISION,
    KNOWN_HANDS,
    PAUSED_HANDS,
    ScopeViolation,
    historical_reproduction_scope,
    require_release_scope,
    resolve_workload_hands,
)
from qdgrasp.roadmap import audit_active_scope, audit_config_files, audit_runtime_defaults

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_active_corpus_is_the_two_hands_of_adr_0008() -> None:
    assert ACTIVE_HANDS == ("leap_hand", "wonik_allegro")
    assert PAUSED_HANDS == ("shadow_hand",)
    assert set(KNOWN_HANDS) == set(ACTIVE_HANDS) | set(PAUSED_HANDS)


def test_default_workload_resolves_to_the_active_corpus_only() -> None:
    scope = resolve_workload_hands()
    assert scope.hands == ACTIVE_HANDS
    assert scope.non_release is False
    assert scope.coverage == "2/2_active"
    assert scope.three_hand_coverage is False


def test_selecting_a_paused_hand_without_the_flag_is_refused() -> None:
    with pytest.raises(ScopeViolation, match=GOVERNING_DECISION):
        resolve_workload_hands(["leap_hand", "shadow_hand"])


def test_experimental_shadow_requires_a_stated_purpose() -> None:
    with pytest.raises(ScopeViolation, match="diagnostic purpose"):
        resolve_workload_hands(["shadow_hand"], experimental_shadow=True)


def test_experimental_shadow_runs_but_is_never_release() -> None:
    scope = resolve_workload_hands(
        ["shadow_hand"], experimental_shadow=True, purpose="underactuated tendon diagnostic"
    )
    assert scope.hands == ("shadow_hand",)
    assert scope.non_release is True
    assert scope.experimental_shadow is True
    disclosure = scope.as_disclosure()
    assert disclosure["three_hand_coverage"] is False
    assert disclosure["historical_p3_4_state"] == "paused_by_ADR-0008"


def test_unknown_hand_is_refused() -> None:
    with pytest.raises(ScopeViolation, match="unknown hands"):
        resolve_workload_hands(["not_a_hand"])


def test_release_scope_rejects_a_paused_hand() -> None:
    with pytest.raises(ScopeViolation, match="forbids it"):
        require_release_scope(["leap_hand", "wonik_allegro", "shadow_hand"])


def test_release_scope_rejects_a_subset_of_the_active_corpus() -> None:
    with pytest.raises(ScopeViolation, match="missing active hands"):
        require_release_scope(["leap_hand"])


def test_release_scope_accepts_exactly_the_active_corpus() -> None:
    assert require_release_scope(DEFAULT_ROBOT_PROFILES) == ACTIVE_HANDS


def test_historical_reproduction_is_declared_and_non_release() -> None:
    scope = historical_reproduction_scope(
        "QDGrasp-Scene-Tiny", ["leap_hand.yaml", "wonik_allegro.yaml", "shadow_hand.yaml"]
    )
    assert scope.non_release is True
    assert scope.three_hand_coverage is False
    with pytest.raises(ScopeViolation, match="not a declared"):
        historical_reproduction_scope("SomeNewDataset", ["shadow_hand"])


def test_data_config_default_does_not_select_a_paused_hand() -> None:
    from qdgrasp.dataset.schema import DataConfigV2

    default = DataConfigV2.model_fields["robot_profiles"].default
    assert default == DEFAULT_ROBOT_PROFILES
    assert audit_runtime_defaults() == ()


def test_shipped_configs_declare_any_paused_selection() -> None:
    assert audit_config_files(REPO_ROOT) == ()


def test_repository_has_no_undeclared_paused_selection() -> None:
    findings = audit_active_scope(REPO_ROOT)
    assert findings == (), [str(finding) for finding in findings]


def test_scan_reports_a_new_config_that_selects_a_paused_hand(tmp_path: Path) -> None:
    configs = tmp_path / "configs" / "data"
    configs.mkdir(parents=True)
    (configs / "new_workload.yaml").write_text(
        yaml.safe_dump({"schema": "qdgrasp/data/v2", "robot_profiles": ["shadow_hand.yaml"]}),
        encoding="utf-8",
    )
    findings = audit_config_files(tmp_path)
    assert len(findings) == 1
    assert findings[0].hand == "shadow_hand"
    assert "without declaring" in str(findings[0])


def test_shadow_preset_and_assets_are_kept() -> None:
    # The pause is not a deletion: the preset has to stay loadable so another
    # ADR can reverse the decision without an archaeology project.
    assert (REPO_ROOT / "qdgrasp" / "presets" / "robots" / "shadow_hand.yaml").is_file()


def test_contactrich_generator_reads_the_corpus_from_the_registry() -> None:
    source = (REPO_ROOT / "scripts" / "generate_contactrich_tiny.py").read_text(encoding="utf-8")
    assert "resolve_workload_hands()" in source
    assert 'ACTIVE_HANDS = ("leap_hand", "wonik_allegro")' not in source


def test_the_audit_sees_a_selection_a_yaml_scan_never_could(tmp_path):
    """RRV-02: the selection that reopened B-10 lived in a Python list literal.

    A config allowlist cannot reach it, so the audit has to parse Python. This
    builds the shape of the original defect and asserts it is caught.
    """
    from qdgrasp.roadmap.scope_audit import audit_python_entry_points

    (tmp_path / "scripts").mkdir()
    (tmp_path / "qdgrasp").mkdir()
    (tmp_path / "scripts" / "leaky_generator.py").write_text(
        'robot_configs = [\n'
        '    ("leap_hand", "leap_hand.yaml"),\n'
        '    ("shadow_hand", "shadow_hand.yaml"),\n'
        ']\n',
        encoding="utf-8",
    )
    findings = audit_python_entry_points(tmp_path)
    assert any(finding.hand == "shadow_hand" for finding in findings)


def test_a_table_keyed_by_hand_is_not_a_selection(tmp_path):
    """Supporting a paused profile is not choosing it.

    Flagging every lookup table would bury the one case that matters, and a
    gate nobody can read is worse than no gate.
    """
    from qdgrasp.roadmap.scope_audit import audit_python_entry_points

    (tmp_path / "scripts").mkdir()
    (tmp_path / "qdgrasp").mkdir()
    (tmp_path / "scripts" / "table.py").write_text(
        'ROBOT_CONFIGS = {\n'
        '    "leap_hand": "leap_hand.yaml",\n'
        '    "shadow_hand": "shadow_hand.yaml",\n'
        '}\n',
        encoding="utf-8",
    )
    assert audit_python_entry_points(tmp_path) == ()


def test_an_unparseable_file_is_a_finding_not_a_skip(tmp_path):
    """A file the audit cannot read is a file the audit cannot clear."""
    from qdgrasp.roadmap.scope_audit import audit_python_entry_points

    (tmp_path / "scripts").mkdir()
    (tmp_path / "qdgrasp").mkdir()
    (tmp_path / "scripts" / "broken.py").write_text("def (\n", encoding="utf-8")
    findings = audit_python_entry_points(tmp_path)
    assert any(finding.key == "<unparseable>" for finding in findings)


def test_the_repository_has_no_undeclared_paused_selection():
    """WRK-R4 acceptance: zero findings across the whole repository."""
    from qdgrasp.roadmap.scope_audit import audit_active_scope

    findings = audit_active_scope(REPO_ROOT)
    assert findings == (), "\n".join(str(finding) for finding in findings)
