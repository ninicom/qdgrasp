"""Source provenance for reproducible DGN dataset generation."""

from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

DGN_OPEN_TINY_REQUIRED_GENERATOR_SOURCES = frozenset(
    {
        "scripts/generate_dgn_open_tiny.py",
        "qdgrasp/dataset/manifest.py",
        "qdgrasp/dataset/provenance.py",
        "qdgrasp/dataset/render.py",
        "qdgrasp/dataset/rng.py",
        "qdgrasp/dataset/shards.py",
        "qdgrasp/dataset/split.py",
        "qdgrasp/objects/generate.py",
        "qdgrasp/objects/manifest.py",
        "qdgrasp/objects/schema.py",
        "qdgrasp/robot/provenance.py",
        "qdgrasp/robot/spec.py",
    }
)


def _source_path(module: ModuleType) -> Path | None:
    filename = getattr(module, "__file__", None)
    if not filename:
        return None
    path = Path(filename)
    if path.suffix in {".pyc", ".pyo"}:
        try:
            path = Path(importlib.util.source_from_cache(str(path)))
        except ValueError:
            return None
    return path.resolve() if path.suffix == ".py" else None


def loaded_qdgrasp_source_hashes(
    repo_root: str | Path,
    *,
    entry_script: str = "scripts/generate_dgn_open_tiny.py",
) -> dict[str, str]:
    """Hash the effective transitive import closure of a generator run."""

    root = Path(repo_root).resolve()
    package_root = (root / "qdgrasp").resolve()
    names = {entry_script}
    for module in tuple(sys.modules.values()):
        if not isinstance(module, ModuleType):
            continue
        path = _source_path(module)
        if path is None or not path.is_relative_to(package_root):
            continue
        names.add(path.relative_to(root).as_posix())

    missing = sorted(name for name in names if not (root / name).is_file())
    if missing:
        raise RuntimeError(f"generator provenance source files are missing: {missing}")
    if not DGN_OPEN_TINY_REQUIRED_GENERATOR_SOURCES <= names:
        absent = sorted(DGN_OPEN_TINY_REQUIRED_GENERATOR_SOURCES - names)
        raise RuntimeError(f"generator did not load required corpus sources: {absent}")
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in sorted(names)}
