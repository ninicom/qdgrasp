#!/usr/bin/env python3
"""Stdlib-only AST import-graph checker for the vendored ``dexgrasp/`` package.

Walks every ``dexgrasp/**/*.py`` file with :mod:`ast` (module-level and nested
imports alike -- ``ast.walk`` visits the whole tree, including imports inside
function bodies such as ``engine/exporter.py``'s per-format export methods or
``utils/callbacks/base.py``'s deferred integration imports), classifies each
import as internal (``ultralytics.*``/``dexgrasp.*``, or a relative import --
both names are treated as internal during the M1 namespace-rewrite transition,
since content has only been path-renamed so far, not rewritten), stdlib, or
third-party, and verifies every internal import resolves to a real file on
disk.

This checks module-level resolution only (does the imported *module* exist as
a file), not symbol-level resolution (whether the imported *name* is actually
defined inside that module) -- deliberately, to stay stdlib-only and fast.

Also counts literal, case-sensitive ``"ultralytics"`` token occurrences per
file, for tracking progress on the PLAN.md M1 namespace rewrite (not a
pass/fail signal by itself). Case-sensitive on purpose: lowercase hits are
almost always import-statement/package-name residue (the mechanical rewrite
pass's target); capitalized "Ultralytics" in prose/license headers is a
distinct, explicitly-deferred category and is not counted here.
"""

from __future__ import annotations

import argparse
import ast
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PACKAGE_ROOT = PROJECT_ROOT / "dexgrasp"
INTERNAL_PREFIXES = ("ultralytics", "dexgrasp")


@dataclass(frozen=True)
class ImportProblem:
    file: str
    line: int
    message: str


@dataclass
class FileReport:
    path: str
    ultralytics_token_count: int = 0
    internal_import_count: int = 0
    stdlib_import_count: int = 0
    third_party_import_count: int = 0
    unresolved: list[str] = field(default_factory=list)


def module_parts(file_path: Path, package_root: Path) -> tuple[str, ...]:
    """Dotted-name components of ``file_path``'s own module, rooted one level
    above ``package_root`` (so the first component is the package dir name,
    e.g. ``dexgrasp``)."""

    relative = file_path.relative_to(package_root.parent)
    parts = list(relative.with_suffix("").parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return tuple(parts)


def package_parts(file_path: Path, package_root: Path) -> tuple[str, ...]:
    """Dotted-name components of ``file_path``'s containing package (itself,
    for ``__init__.py``; its parent directory otherwise)."""

    parts = module_parts(file_path, package_root)
    if file_path.name == "__init__.py":
        return parts
    return parts[:-1]


def resolve_internal(parts: Sequence[str], package_root: Path) -> bool:
    """Check whether a dotted path with an internal prefix resolves to a real
    file. ``parts[0]`` must already be ``dexgrasp`` or ``ultralytics``."""

    if not parts or parts[0] not in INTERNAL_PREFIXES:
        return False
    rest = parts[1:]
    if not rest:
        return (package_root / "__init__.py").is_file()
    candidate = package_root.joinpath(*rest)
    return candidate.with_suffix(".py").is_file() or (candidate / "__init__.py").is_file()


def classify_absolute(name: str) -> str:
    top = name.split(".", 1)[0]
    if top in INTERNAL_PREFIXES:
        return "internal"
    if top == "__future__" or top in sys.stdlib_module_names:
        return "stdlib"
    return "third_party"


def record_kind(report: FileReport, kind: str) -> None:
    if kind == "internal":
        report.internal_import_count += 1
    elif kind == "stdlib":
        report.stdlib_import_count += 1
    else:
        report.third_party_import_count += 1


def check_file(file_path: Path, package_root: Path) -> tuple[FileReport, list[ImportProblem]]:
    relative_path = str(file_path.relative_to(package_root.parent))
    report = FileReport(path=relative_path)
    problems: list[ImportProblem] = []

    try:
        source = file_path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        return report, [ImportProblem(relative_path, 0, f"không đọc được file: {exc}")]
    report.ultralytics_token_count = source.count("ultralytics")

    try:
        tree = ast.parse(source, filename=str(file_path))
    except SyntaxError as exc:
        problems.append(ImportProblem(relative_path, exc.lineno or 0, f"syntax error: {exc.msg}"))
        return report, problems

    file_package = package_parts(file_path, package_root)

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                kind = classify_absolute(alias.name)
                record_kind(report, kind)
                if kind == "internal" and not resolve_internal(alias.name.split("."), package_root):
                    problems.append(
                        ImportProblem(relative_path, node.lineno, f"import không resolve được: {alias.name}")
                    )
                    report.unresolved.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0:
                module = node.module or ""
                kind = classify_absolute(module) if module else "third_party"
                record_kind(report, kind)
                if kind == "internal" and not resolve_internal(module.split("."), package_root):
                    problems.append(
                        ImportProblem(relative_path, node.lineno, f"import không resolve được: {module}")
                    )
                    report.unresolved.append(module)
                continue

            report.internal_import_count += 1
            truncate = node.level - 1
            if truncate >= len(file_package):
                problems.append(
                    ImportProblem(
                        relative_path,
                        node.lineno,
                        f"relative import vượt quá package root: level={node.level}",
                    )
                )
                continue
            base = file_package[: len(file_package) - truncate] if truncate else file_package
            extra = tuple(node.module.split(".")) if node.module else ()
            full = base + extra
            if not resolve_internal(full, package_root):
                dotted_prefix = "." * node.level
                dotted = f"{dotted_prefix}{node.module or ''}"
                problems.append(
                    ImportProblem(relative_path, node.lineno, f"relative import không resolve được: {dotted}")
                )
                report.unresolved.append(dotted)

    return report, problems


def iter_python_files(package_root: Path) -> list[Path]:
    return sorted(package_root.rglob("*.py"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Kiểm import graph của dexgrasp/ bằng AST, không phụ thuộc third-party."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_PACKAGE_ROOT,
        help="Thư mục package cần kiểm (mặc định dexgrasp/).",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="In thêm bảng tổng hợp per-file (internal/stdlib/third-party count, ultralytics token count).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    package_root = args.root.expanduser().resolve()

    if not package_root.is_dir():
        print(f"ERROR: package root không tồn tại: {package_root}")
        print("Dexgrasp import check failed: 1 lỗi.")
        return 2

    files = iter_python_files(package_root)
    all_problems: list[ImportProblem] = []
    reports: list[FileReport] = []
    total_ultralytics_tokens = 0

    for file_path in files:
        report, problems = check_file(file_path, package_root)
        reports.append(report)
        all_problems.extend(problems)
        total_ultralytics_tokens += report.ultralytics_token_count

    if args.summary:
        for report in reports:
            print(
                f"{report.path}: internal={report.internal_import_count} "
                f"stdlib={report.stdlib_import_count} "
                f"third_party={report.third_party_import_count} "
                f"ultralytics_tokens={report.ultralytics_token_count}"
            )

    if all_problems:
        for problem in all_problems:
            print(f"ERROR: {problem.file}:{problem.line}: {problem.message}")
        print(
            f"Dexgrasp import check failed: {len(all_problems)} lỗi trong "
            f"{len(files)} file."
        )
        return 1

    print(
        f"Dexgrasp import check passed: {len(files)} file, 0 unresolved internal "
        f"import; tổng {total_ultralytics_tokens} lần xuất hiện literal 'ultralytics' "
        f"còn lại (namespace rewrite chưa chạy)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
