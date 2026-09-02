"""One release, one declared version, and a mapping between its two notations.

``ROADMAP-MVP-RELEASE-001`` §5 MR-02 asks for a single version source and a
checker that verifies the *mapping* between the distribution version and the
release tag instead of demanding two equal strings.  ``0.1.0a2`` and
``0.1.0-alpha.2`` name the same release and can never be string-equal: PEP 440
and SemVer spell prereleases differently, and packaging normalises one of them.
A gate built on equality would therefore have to be satisfied by weakening one
of the two notations, which is how the project ended up carrying both.
"""

from __future__ import annotations

import importlib.metadata
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import qdgrasp
from qdgrasp.version import (
    RELEASE_VERSION,
    TAG,
    __version__,
    distribution_and_release,
    distribution_version,
    release_version,
    tag_name,
)

RELEASE_GATE = PROJECT_ROOT / "scripts/release_gate.sh"
IDENTITY_CHECKER = PROJECT_ROOT / "scripts/check_version_identity.py"


def declared() -> str:
    return (PROJECT_ROOT / "VERSION").read_text(encoding="utf-8").strip()


# -- the single source ----------------------------------------------------


def test_version_is_declared_in_exactly_one_place() -> None:
    project = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert "version" not in project["project"], "pyproject must not carry a second declaration"
    assert "version" in project["project"]["dynamic"]
    assert project["tool"]["setuptools"]["dynamic"]["version"] == {"file": ["VERSION"]}


def test_every_consumer_reports_the_declared_version() -> None:
    assert __version__ == declared()
    assert qdgrasp.__version__ == declared()
    assert importlib.metadata.version("qdgrasp") == declared()


def test_the_version_file_holds_one_line() -> None:
    raw = (PROJECT_ROOT / "VERSION").read_text(encoding="utf-8")
    assert raw == declared() + "\n"


def test_the_release_target_of_this_plan_is_what_the_tree_declares() -> None:
    assert declared() == "0.1.0a2"
    assert RELEASE_VERSION == "0.1.0-alpha.2"
    assert TAG == "v0.1.0-alpha.2"


# -- the mapping ----------------------------------------------------------


@pytest.mark.parametrize(
    ("distribution", "release"),
    [
        ("0.1.0a1", "0.1.0-alpha.1"),
        ("0.1.0a2", "0.1.0-alpha.2"),
        ("0.2.0b3", "0.2.0-beta.3"),
        ("1.0.0rc1", "1.0.0-rc.1"),
        ("1.2.3", "1.2.3"),
    ],
)
def test_the_two_notations_map_onto_each_other(distribution: str, release: str) -> None:
    assert release_version(distribution) == release
    assert distribution_version(release) == distribution
    assert tag_name(release) == f"v{release}"
    assert distribution_and_release(distribution) == (distribution, release)
    assert distribution_and_release(release) == (distribution, release)


@pytest.mark.parametrize("bad", ["", "1.0", "0.1.0alpha2", "0.1.0-a.2", "v0.1.0-alpha.2", "0.1.0-alpha"])
def test_a_version_in_neither_notation_is_refused(bad: str) -> None:
    with pytest.raises(ValueError):
        distribution_and_release(bad)


def test_the_notations_are_not_equal_which_is_why_the_gate_maps_them() -> None:
    assert RELEASE_VERSION != __version__


# -- the checker ----------------------------------------------------------


def _run(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(IDENTITY_CHECKER), *arguments],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def test_the_identity_checker_passes_on_this_tree() -> None:
    result = _run()
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Version identity: PASS" in result.stdout
    assert TAG in result.stdout


def test_the_identity_checker_accepts_the_release_this_tree_declares() -> None:
    assert _run("--release", RELEASE_VERSION).returncode == 0


def test_the_identity_checker_rejects_a_release_the_tree_does_not_declare() -> None:
    result = _run("--release", "0.1.0-alpha.1")
    assert result.returncode == 1
    assert "maps to distribution 0.1.0a1" in result.stdout


def test_the_identity_checker_rejects_a_malformed_release() -> None:
    result = _run("--release", "not-a-version")
    assert result.returncode == 1
    assert "not a supported release version" in result.stdout


# -- the release script ---------------------------------------------------


def _gate(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(RELEASE_GATE), *arguments],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def test_the_release_gate_delegates_the_version_question_to_the_checker() -> None:
    script = RELEASE_GATE.read_text(encoding="utf-8")
    assert "check_version_identity.py --release" in script
    # The equality test this replaced would have made 0.1.0a2 and
    # 0.1.0-alpha.2 impossible to satisfy at the same time.
    assert 'tr -d \'[:space:]\' < VERSION' not in script


def test_the_release_gate_refuses_a_version_it_cannot_parse() -> None:
    result = _gate("alpha-two")
    assert result.returncode == 2
    assert "Usage" in result.stderr


def test_the_release_gate_refuses_to_run_off_its_own_release_branch() -> None:
    result = _gate(RELEASE_VERSION)
    assert result.returncode == 1
    assert f"release/{RELEASE_VERSION}" in result.stderr
