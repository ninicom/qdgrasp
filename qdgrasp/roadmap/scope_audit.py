"""Scan the repository for workloads that select a paused hand (G05).

The registry in :mod:`qdgrasp.config.active_scope` decides what a workload is
allowed to select. This module checks that the repository actually goes through
it: it reads the shipped configuration and the runtime defaults and reports
every place a paused hand is chosen without declaring why.

Two things are allowed to name a paused hand, and both have to say so:

* a config that regenerates an artifact published before ADR-0008, which is
  listed here by path;
* a workload that passes ``experimental_shadow`` with a stated purpose, which
  is a runtime decision this static scan cannot see and does not try to.

Everything else is a finding.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from pathlib import Path

import yaml

from qdgrasp.config.active_scope import GOVERNING_DECISION, hand_of_profile, is_paused

#: Configs that reproduce artifacts published before ADR-0008. Listed by path
#: rather than inferred, so adding one is a reviewable change.
HISTORICAL_CONFIG_ALLOWLIST: frozenset[str] = frozenset(
    {
        "configs/data/dgn_open_tiny.yaml",
    }
)

#: Keys that select hands in a configuration document.
_SELECTION_KEYS = ("robot_profiles", "robot_profile", "hands", "active_hands")


@dataclasses.dataclass(frozen=True)
class ScopeFinding:
    """One place that selects a paused hand without declaring why."""

    source: str
    key: str
    hand: str

    def __str__(self) -> str:
        return (
            f"{self.source}: {self.key} selects paused hand {self.hand!r} without "
            f"declaring a pre-{GOVERNING_DECISION} artifact or experimental_shadow"
        )


def _selected_hands(document: Mapping[str, object]) -> list[tuple[str, str]]:
    found: list[tuple[str, str]] = []
    for key in _SELECTION_KEYS:
        value = document.get(key)
        if isinstance(value, str):
            found.append((key, hand_of_profile(value)))
        elif isinstance(value, Sequence):
            found.extend((key, hand_of_profile(str(item))) for item in value)
    return found


def audit_config_files(repo_root: str | Path) -> tuple[ScopeFinding, ...]:
    """Report paused-hand selections in shipped YAML configuration."""
    root = Path(repo_root)
    findings: list[ScopeFinding] = []
    for path in sorted((root / "configs").rglob("*.yaml")):
        relative = path.relative_to(root).as_posix()
        if relative in HISTORICAL_CONFIG_ALLOWLIST:
            continue
        try:
            document = yaml.safe_load(path.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError):
            continue
        if not isinstance(document, Mapping):
            continue
        findings.extend(
            ScopeFinding(source=relative, key=key, hand=hand)
            for key, hand in _selected_hands(document)
            if is_paused(hand)
        )
    return tuple(findings)


def audit_runtime_defaults() -> tuple[ScopeFinding, ...]:
    """Report paused hands reachable from a default runtime configuration."""
    from qdgrasp.dataset.schema import DataConfigV2

    findings: list[ScopeFinding] = []
    default_profiles = DataConfigV2.model_fields["robot_profiles"].default or ()
    findings.extend(
        ScopeFinding(source="qdgrasp.dataset.schema.DataConfigV2", key="robot_profiles", hand=hand_of_profile(profile))
        for profile in default_profiles
        if is_paused(profile)
    )
    return tuple(findings)


def audit_active_scope(repo_root: str | Path) -> tuple[ScopeFinding, ...]:
    """Every undeclared paused-hand selection this scan can see."""
    return (*audit_runtime_defaults(), *audit_config_files(repo_root))
