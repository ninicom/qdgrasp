#!/usr/bin/env python3
"""Run Ruff and Mypy on the explicit QDGrasp release/core boundary."""

from __future__ import annotations

import argparse
import ast
import shutil
import subprocess
import sys
from collections import deque
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = PROJECT_ROOT / "qdgrasp"

PUBLIC_ROOTS = ("qdgrasp/__init__.py", "qdgrasp/cli.py")
GENERATOR_ROOT = "scripts/generate_dgn_open_tiny.py"
EXTRA_CORE_ROOTS = (
    "qdgrasp/dataset/contactrich_active.py",
    "qdgrasp/roadmap/review_packet.py",
)
RUFF_SCRIPT_TARGETS = (
    "scripts/check_dataset_manifest.py",
    "scripts/check_docs.py",
    "scripts/check_phase4.py",
    "scripts/check_phase5_inputs.py",
    "scripts/check_wheel.py",
    "scripts/generate_dgn_open_tiny.py",
    "scripts/phase4_cuda_gate.py",
    "scripts/phase4_review_packet.py",
)
MYPY_TARGETS = (
    "qdgrasp/api/facade.py",
    "qdgrasp/api/protocols.py",
    "qdgrasp/api/results.py",
    "qdgrasp/config/__init__.py",
    "qdgrasp/config/active_scope.py",
    "qdgrasp/config/loader.py",
    "qdgrasp/config/policy.py",
    "qdgrasp/config/registry.py",
    "qdgrasp/config/schema.py",
    "qdgrasp/corrective/__init__.py",
    "qdgrasp/corrective/gate.py",
    "qdgrasp/corrective/registry.py",
    "qdgrasp/dataset/artifact.py",
    "qdgrasp/dataset/artifact_io.py",
    "qdgrasp/dataset/loader.py",
    "qdgrasp/dataset/manifest.py",
    "qdgrasp/dataset/schema.py",
    "qdgrasp/dataset/shards.py",
    "qdgrasp/dataset/split.py",
    "qdgrasp/engine/checkpoint.py",
    "qdgrasp/engine/compatibility.py",
    "qdgrasp/engine/runner.py",
    "qdgrasp/engine/sampling.py",
    "qdgrasp/engine/seeding.py",
    "qdgrasp/models/hand_graph.py",
    "qdgrasp/models/protocol.py",
    "qdgrasp/mvp/config.py",
    "qdgrasp/mvp/evaluate.py",
    "qdgrasp/mvp/policy.py",
    "qdgrasp/mvp/ppo.py",
    "qdgrasp/objects/schema.py",
    "qdgrasp/robot/schema.py",
)
FORBIDDEN_PREFIXES = ("qdgrasp/data/", "qdgrasp/nn/")


class StaticBoundaryError(RuntimeError):
    """The declared active-core boundary is incomplete or unsafe."""


def _module_name(path: Path, root: Path) -> str | None:
    try:
        relative = path.relative_to(root).with_suffix("")
    except ValueError:
        return None
    if not relative.parts or relative.parts[0] != "qdgrasp":
        return None
    parts = relative.parts[:-1] if relative.name == "__init__" else relative.parts
    return ".".join(parts)


def _module_path(module: str, root: Path) -> Path | None:
    if module != "qdgrasp" and not module.startswith("qdgrasp."):
        return None
    relative = Path(*module.split("."))
    package = root / relative / "__init__.py"
    source = (root / relative).with_suffix(".py")
    if package.is_file():
        return package
    if source.is_file():
        return source
    return None


def _absolute_from_import(node: ast.ImportFrom, path: Path, root: Path) -> str | None:
    if node.level == 0:
        return node.module
    current = _module_name(path, root)
    if current is None:
        raise StaticBoundaryError(f"relative import outside package module: {path}")
    package = current if path.name == "__init__.py" else current.rpartition(".")[0]
    parts = package.split(".") if package else []
    climbs = node.level - 1
    if climbs > len(parts) - 1:
        raise StaticBoundaryError(f"relative import escapes qdgrasp package: {path}:{node.lineno}")
    if climbs:
        parts = parts[:-climbs]
    if node.module:
        parts.extend(node.module.split("."))
    return ".".join(parts)


def _package_ancestors(path: Path, root: Path) -> list[Path]:
    ancestors: list[Path] = []
    directory = path.parent
    while directory.is_relative_to(root / "qdgrasp"):
        init = directory / "__init__.py"
        if init.is_file():
            ancestors.append(init)
        if directory == root / "qdgrasp":
            break
        directory = directory.parent
    return ancestors


def import_closure(seeds: list[Path], root: Path = PROJECT_ROOT) -> set[Path]:
    """Return the local qdgrasp import closure for explicit Python roots."""

    queue = deque(path.resolve() for path in seeds)
    seen: set[Path] = set()
    while queue:
        path = queue.popleft()
        if path in seen:
            continue
        if not path.is_file():
            raise StaticBoundaryError(f"active-core source is missing: {path}")
        seen.add(path)
        queue.extend(item.resolve() for item in _package_ancestors(path, root) if item.resolve() not in seen)
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (OSError, UnicodeError, SyntaxError) as error:
            raise StaticBoundaryError(f"cannot parse active-core source {path}: {error}") from error
        for node in ast.walk(tree):
            modules: list[tuple[str, bool]] = []
            if isinstance(node, ast.Import):
                modules.extend((alias.name, True) for alias in node.names if alias.name.startswith("qdgrasp"))
            elif isinstance(node, ast.ImportFrom):
                base = _absolute_from_import(node, path, root)
                if base and (base == "qdgrasp" or base.startswith("qdgrasp.")):
                    modules.append((base, True))
                    modules.extend((f"{base}.{alias.name}", False) for alias in node.names if alias.name != "*")
            for module, required in modules:
                imported = _module_path(module, root)
                if imported is None:
                    if required:
                        raise StaticBoundaryError(f"unresolved active-core import {module!r} in {path}:{node.lineno}")
                    continue
                queue.append(imported.resolve())
    return seen


def active_ruff_targets(root: Path = PROJECT_ROOT) -> tuple[str, ...]:
    seeds = [root / name for name in (*PUBLIC_ROOTS, GENERATOR_ROOT, *EXTRA_CORE_ROOTS)]
    seeds.extend(sorted((root / "qdgrasp/mvp").rglob("*.py")))
    closure = import_closure(seeds, root)
    closure.update((root / name).resolve() for name in RUFF_SCRIPT_TARGETS)
    relative = tuple(sorted(path.relative_to(root).as_posix() for path in closure))
    forbidden = [name for name in relative if name.startswith(FORBIDDEN_PREFIXES)]
    if forbidden:
        raise StaticBoundaryError(f"active core reaches quarantined legacy sources: {forbidden}")
    return relative


def _tool(name: str) -> str:
    sibling = Path(sys.executable).with_name(name)
    if sibling.is_file():
        return str(sibling)
    resolved = shutil.which(name)
    if resolved is None:
        raise StaticBoundaryError(f"required development tool is unavailable: {name}")
    return resolved


def _run(command: list[str], *, root: Path) -> int:
    return subprocess.run(command, cwd=root, check=False).returncode


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--check", choices=("all", "ruff", "mypy"), default="all")
    parser.add_argument("--list", action="store_true", help="print the resolved Ruff boundary")
    args = parser.parse_args(argv)
    root = args.root.resolve()
    try:
        ruff_targets = active_ruff_targets(root)
        missing_mypy = [name for name in MYPY_TARGETS if not (root / name).is_file()]
        if missing_mypy:
            raise StaticBoundaryError(f"Mypy contract targets are missing: {missing_mypy}")
    except StaticBoundaryError as error:
        print(f"Static core gate: FAIL: {error}")
        return 1
    if args.list:
        print("\n".join(ruff_targets))
    if args.check in {"all", "ruff"} and _run([_tool("ruff"), "check", *ruff_targets], root=root):
        return 1
    if args.check in {"all", "mypy"} and _run(
        [_tool("mypy"), "--follow-imports=skip", *MYPY_TARGETS],
        root=root,
    ):
        return 1
    print(f"Static core gate: PASS (Ruff {len(ruff_targets)} files; Mypy {len(MYPY_TARGETS)} contracts)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
