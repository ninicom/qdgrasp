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

import ast
import dataclasses
from collections.abc import Mapping, Sequence
from pathlib import Path

import yaml

from qdgrasp.config.active_scope import (
    GOVERNING_DECISION,
    PAUSED_HANDS,
    hand_of_profile,
    is_paused,
)

_KNOWN_PAUSED = frozenset(PAUSED_HANDS)

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
    paths = sorted(
        {*(root / "configs").rglob("*.yaml"), *(root / "configs").rglob("*.yml")}
    )
    for path in paths:
        relative = path.relative_to(root).as_posix()
        if relative in HISTORICAL_CONFIG_ALLOWLIST:
            continue
        try:
            document = yaml.safe_load(path.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError) as error:
            # A config the audit cannot read is a config the audit cannot clear.
            # Skipping it silently is how an unreadable file becomes a pass.
            findings.append(
                ScopeFinding(source=relative, key="<unparseable>", hand=str(error)[:80])
            )
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


#: Python entry points that may name a paused hand, because they exist to
#: reproduce a pre-ADR-0008 artifact and gate it behind an explicit flag.
HISTORICAL_PYTHON_ALLOWLIST: frozenset[str] = frozenset()

#: Trees whose Python is release-reachable and therefore in scope for the audit.
_PYTHON_ROOTS: tuple[str, ...] = ("scripts", "qdgrasp")

#: The guards that turn naming a paused hand into a declared reproduction
#: rather than a default. A file using one of these is asking out loud.
_DECLARED_GUARDS: tuple[str, ...] = (
    "historical_reproduction_scope",
    "experimental_shadow_scope",
    "historical_reproduction",
    "experimental_shadow",
)


#: Names that mean "these are the hands this run will use". A paused hand
#: inside one of these is a selection; the same name in a lookup table keyed by
#: hand is not, which is why this matches assignment targets rather than any
#: occurrence of the string.
_SELECTION_NAMES: tuple[str, ...] = (
    "robot_configs",
    "robot_profiles",
    "robot_profile",
    "profiles",
    "hands",
    "active_hands",
    "default_hands",
    "hand_list",
)


def _is_selection_name(name: str) -> bool:
    return name.lower().lstrip("_") in _SELECTION_NAMES


def _paused_strings(node: ast.AST) -> list[tuple[str, int]]:
    """Paused-hand names among the string literals inside one node.

    A dict keyed by hand is a table of per-hand parameters, not a choice of
    which hands to run: it supports a paused profile without selecting it. Only
    sequences are selections, so only sequences are searched.
    """
    if isinstance(node, ast.Dict):
        return []
    found: list[tuple[str, int]] = []
    for child in ast.walk(node):
        if not isinstance(child, ast.Constant) or not isinstance(child.value, str):
            continue
        candidate = child.value
        if "/" in candidate or " " in candidate:
            continue
        hand = hand_of_profile(candidate)
        if hand in _KNOWN_PAUSED:
            found.append((hand, child.lineno))
    return found


def audit_python_entry_points(repo_root: str | Path) -> tuple[ScopeFinding, ...]:
    """Report Python whose *default selection* names a paused hand.

    A config allowlist does not make a Python generator safe: the selection that
    reopened this finding lived in a list literal in a release generator, where
    no YAML scan could ever reach it. This parses rather than greps, and it
    matches selections rather than mentions -- a table of per-hand parameters
    keyed by name supports a paused profile without choosing it, and flagging
    those would bury the one case that matters.
    """
    root = Path(repo_root)
    findings: list[ScopeFinding] = []
    for tree_name in _PYTHON_ROOTS:
        for path in sorted((root / tree_name).rglob("*.py")):
            relative = path.relative_to(root).as_posix()
            if relative in HISTORICAL_PYTHON_ALLOWLIST:
                continue
            source = path.read_text(encoding="utf-8")
            try:
                module = ast.parse(source)
            except SyntaxError as error:
                findings.append(
                    ScopeFinding(source=relative, key="<unparseable>", hand=str(error)[:80])
                )
                continue
            if any(guard in source for guard in _DECLARED_GUARDS):
                continue

            for node in ast.walk(module):
                # A collection assigned to a selection-shaped name.
                if isinstance(node, ast.Assign):
                    names = [
                        target.id
                        for target in node.targets
                        if isinstance(target, ast.Name)
                    ]
                    if any(_is_selection_name(name) for name in names):
                        findings.extend(
                            ScopeFinding(
                                source=f"{relative}:{line}",
                                key=f"{names[0]} =",
                                hand=hand,
                            )
                            for hand, line in _paused_strings(node.value)
                        )
                # A function parameter whose default selects one.
                elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    args = node.args
                    pairs = list(
                        zip(args.args[len(args.args) - len(args.defaults):], args.defaults)
                    ) + [
                        (arg, default)
                        for arg, default in zip(args.kwonlyargs, args.kw_defaults)
                        if default is not None
                    ]
                    for arg, default in pairs:
                        if not _is_selection_name(arg.arg):
                            continue
                        findings.extend(
                            ScopeFinding(
                                source=f"{relative}:{line}",
                                key=f"{node.name}({arg.arg}=...)",
                                hand=hand,
                            )
                            for hand, line in _paused_strings(default)
                        )
    return tuple(findings)


def audit_active_scope(repo_root: str | Path) -> tuple[ScopeFinding, ...]:
    """Every undeclared paused-hand selection this scan can see."""
    return (
        *audit_runtime_defaults(),
        *audit_config_files(repo_root),
        *audit_python_entry_points(repo_root),
    )
