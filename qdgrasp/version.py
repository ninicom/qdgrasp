"""One declared version, and the mapping between the two forms it takes.

A Python distribution version and a release tag are different notations for the
same release, and this project was carrying both -- ``0.1.0a1`` in the package
metadata, ``0.1.0-alpha.1`` in the tag and the release report -- with nothing
connecting them.  A checker that demanded the two strings be equal would be
wrong, because they cannot be: PEP 440 and SemVer disagree about how to spell a
prerelease, and the packaging ecosystem normalises one of them.

So the contract is a mapping, not an equality.  ``VERSION`` at the repository
root is the single declared source; everything else is derived from it and
checked against it by ``scripts/check_version_identity.py``.
"""

from __future__ import annotations

import re
from pathlib import Path

#: The declared source of truth.  ``pyproject.toml`` reads this same file for
#: its distribution version, so there is exactly one place to edit.
VERSION_FILE = Path(__file__).resolve().parents[1] / "VERSION"

#: PEP 440, restricted to what this project actually publishes.
_DISTRIBUTION = re.compile(r"^(?P<release>\d+\.\d+\.\d+)(?:(?P<kind>a|b|rc)(?P<number>\d+))?$")
#: SemVer, restricted to the same set of prerelease kinds.
_RELEASE = re.compile(r"^(?P<release>\d+\.\d+\.\d+)(?:-(?P<label>alpha|beta|rc)\.(?P<number>\d+))?$")

_LABEL_OF_KIND = {"a": "alpha", "b": "beta", "rc": "rc"}
_KIND_OF_LABEL = {label: kind for kind, label in _LABEL_OF_KIND.items()}


def _declared_version() -> str:
    """The declared version: the VERSION file, or the installed metadata.

    In a source tree the file is authoritative.  In an installed wheel there is
    no file, and the metadata built *from* that file is the same value.
    """

    try:
        return VERSION_FILE.read_text(encoding="utf-8").strip()
    except OSError:  # pragma: no cover - only reachable from an installed wheel
        import importlib.metadata

        return importlib.metadata.version("qdgrasp")


def release_version(distribution: str) -> str:
    """The SemVer release string a PEP 440 distribution version maps to."""

    match = _DISTRIBUTION.match(distribution.strip())
    if match is None:
        raise ValueError(f"not a supported distribution version: {distribution!r}")
    if match["kind"] is None:
        return match["release"]
    return f"{match['release']}-{_LABEL_OF_KIND[match['kind']]}.{int(match['number'])}"


def distribution_version(release: str) -> str:
    """The PEP 440 distribution version a SemVer release string maps to."""

    match = _RELEASE.match(release.strip())
    if match is None:
        raise ValueError(f"not a supported release version: {release!r}")
    if match["label"] is None:
        return match["release"]
    return f"{match['release']}{_KIND_OF_LABEL[match['label']]}{int(match['number'])}"


def tag_name(release: str) -> str:
    """The annotated tag that carries a release, e.g. ``v0.1.0-alpha.2``."""

    return f"v{distribution_and_release(release)[1]}"


def distribution_and_release(version: str) -> tuple[str, str]:
    """Both notations of one version, whichever notation was handed in."""

    text = version.strip()
    if _DISTRIBUTION.match(text):
        return text, release_version(text)
    if _RELEASE.match(text):
        return distribution_version(text), text
    raise ValueError(f"not a supported version in either notation: {version!r}")


#: The distribution version, in PEP 440 form.
__version__ = _declared_version()
#: The same version as a release string and as the tag that carries it.
RELEASE_VERSION = release_version(__version__)
TAG = tag_name(RELEASE_VERSION)

__all__ = [
    "RELEASE_VERSION",
    "TAG",
    "VERSION_FILE",
    "__version__",
    "distribution_and_release",
    "distribution_version",
    "release_version",
    "tag_name",
]
