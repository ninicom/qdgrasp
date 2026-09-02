#!/usr/bin/env python3
"""Fail-closed check that one release has exactly one identity.

``ROADMAP-MVP-RELEASE-001`` §5 MR-02 asks for two things this script provides.
First, a single declared version source: ``VERSION`` at the repository root,
which ``pyproject.toml`` reads for the distribution version and
``qdgrasp/version.py`` reads for ``__version__``.  Second, a checker that
verifies the *mapping* between the distribution version and the release tag
rather than demanding two identical strings -- ``0.1.0a2`` and
``0.1.0-alpha.2`` are the same release written in two notations that cannot be
made equal, because PEP 440 and SemVer disagree and packaging normalises one of
them.

Run with no arguments to check the tree.  Pass ``--release <semver>`` to also
check that a release branch or tag names the version this tree declares.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Imported after the path insert above, which is what makes it importable.
from qdgrasp.version import (
    distribution_version,
    release_version,
    tag_name,
)


def check(release: str | None) -> list[str]:
    problems: list[str] = []

    version_file = ROOT / "VERSION"
    if not version_file.is_file():
        return [f"missing the declared version source: {version_file}"]
    raw = version_file.read_text(encoding="utf-8")
    declared = raw.strip()
    if raw != declared + "\n":
        problems.append("VERSION must hold exactly one line and a trailing newline")

    try:
        declared_release = release_version(declared)
    except ValueError as error:
        return [*problems, str(error)]

    # The mapping is a bijection on what this project publishes, and a checker
    # that only walked it one way would accept a notation it could not produce.
    if distribution_version(declared_release) != declared:
        problems.append(
            f"version mapping does not round-trip: {declared} -> {declared_release} -> "
            f"{distribution_version(declared_release)}"
        )

    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8")).get("project", {})
    if "version" in project:
        problems.append("pyproject.toml declares a static version; VERSION is the single source")
    if "version" not in project.get("dynamic", []):
        problems.append("pyproject.toml must declare a dynamic version")
    dynamic = (
        tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        .get("tool", {})
        .get("setuptools", {})
        .get("dynamic", {})
        .get("version", {})
    )
    if dynamic.get("file") != ["VERSION"]:
        problems.append(f"pyproject.toml must read the version from VERSION, not {dynamic!r}")

    from qdgrasp.version import __version__ as imported

    if imported != declared:
        problems.append(f"qdgrasp.version.__version__ is {imported}, declared is {declared}")

    try:
        installed = importlib.metadata.version("qdgrasp")
    except importlib.metadata.PackageNotFoundError:
        problems.append("qdgrasp is not installed for this interpreter; the release gate needs the built metadata")
    else:
        if installed != declared:
            problems.append(
                f"installed distribution is {installed} but the tree declares {declared}; reinstall before releasing"
            )

    if release is not None:
        try:
            expected = distribution_version(release)
        except ValueError as error:
            problems.append(str(error))
        else:
            if expected != declared:
                problems.append(
                    f"release {release} maps to distribution {expected}, but this tree declares {declared}"
                )

    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", default=None, help="release version in SemVer notation, e.g. 0.1.0-alpha.2")
    args = parser.parse_args(argv)

    problems = check(args.release)
    for problem in problems:
        print(f"FAIL {problem}")
    if problems:
        print("Version identity: FAIL")
        return 1

    declared = (ROOT / "VERSION").read_text(encoding="utf-8").strip()
    print(f"distribution        {declared}")
    print(f"release             {release_version(declared)}")
    print(f"tag                 {tag_name(release_version(declared))}")
    print("Version identity: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
