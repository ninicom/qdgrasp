#!/usr/bin/env python3
"""Validate the immutable reference-source lock and local read-only checkouts."""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


LOCK_SHA256 = "921e798163abace643c8a825559425e1bbd16d19476e1464436615fa00b193b7"


@dataclass(frozen=True)
class Artifact:
    path: str
    sha256: str


@dataclass(frozen=True)
class Source:
    directory: str
    repository: str
    commit: str
    required: bool
    artifacts: tuple[Artifact, ...] = ()


SOURCES = (
    Source(
        directory="ultralytics",
        repository="https://github.com/ultralytics/ultralytics.git",
        commit="329682a29d27203582ba30e519340f95abccc6a6",
        required=True,
        artifacts=(
            Artifact(
                "ultralytics/cfg/default.yaml",
                "eb5e9ab6825a5d55076f8b38aed00953dec722ed5d5368a6584df35f50f32839",
            ),
            Artifact(
                "ultralytics/cfg/__init__.py",
                "ea9a98b498686f44561e9ce9892aa0e7de52226429496b08a20475ebbfa4eed8",
            ),
            Artifact(
                "ultralytics/engine/model.py",
                "69cb4c9c6f572c50bd35328e5154e6b596ed50dc10d218584da8aed3f7c9b32a",
            ),
            Artifact(
                "ultralytics/engine/trainer.py",
                "d02bfd82d2af38fb58a6fe7903ef0cb9a93633a75b3db778768896d6c57a32e6",
            ),
            Artifact(
                "LICENSE",
                "0d96a4ff68ad6d4b6f1f30f713b18d5184912ba8dd389f86aa7710db079abcb0",
            ),
        ),
    ),
    Source(
        directory="dexgraspnet2",
        repository="https://github.com/PKU-EPIC/DexGraspNet2.git",
        commit="26ecd76121e3c8218ad53db9840cf34f6b81b076",
        required=True,
        artifacts=(
            Artifact(
                "README.md",
                "4919074edb0af357cb180cf4c4c7806273accc5bffd6919fbfe8cfb3afd1ef7f",
            ),
        ),
    ),
    Source(
        directory="graspgenx",
        repository="https://github.com/NVlabs/GraspGenX.git",
        commit="b9429097728cb1c430dd78b92edf17ba318aad03",
        required=False,
    ),
    Source(
        directory="rectified-flow",
        repository="https://github.com/lqiang67/rectified-flow.git",
        commit="14b4925ad90abdadaca1f7b5caba5555b84e810a",
        required=False,
    ),
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_value(source: Path, *args: str) -> tuple[str | None, str | None]:
    try:
        result = subprocess.run(
            ["git", "-C", str(source), *args],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        return None, str(exc)
    if result.returncode:
        return None, result.stderr.strip() or result.stdout.strip()
    return result.stdout.strip(), None


def validate_lock(path: Path) -> tuple[list[str], bool]:
    if not path.is_file():
        return [f"reference lock không tồn tại: {path}"], True
    try:
        actual = file_sha256(path)
    except OSError as exc:
        return [f"không đọc được reference lock: {exc}"], True
    if actual != LOCK_SHA256:
        return [f"reference lock SHA-256 sai: expected={LOCK_SHA256}, actual={actual}"], False
    return [], False


def validate_checkout(source_root: Path, spec: Source) -> list[str]:
    problems: list[str] = []
    checkout = source_root / spec.directory
    if not checkout.is_dir():
        if spec.required:
            problems.append(f"thiếu checkout bắt buộc: {checkout}")
        return problems

    head, error = git_value(checkout, "rev-parse", "HEAD")
    if error:
        problems.append(f"không đọc được HEAD {spec.directory}: {error}")
    elif head != spec.commit:
        problems.append(
            f"HEAD {spec.directory} sai: expected={spec.commit}, actual={head}"
        )

    remote, error = git_value(checkout, "remote", "get-url", "origin")
    if error:
        problems.append(f"không đọc được origin {spec.directory}: {error}")
    elif remote != spec.repository:
        problems.append(
            f"origin {spec.directory} sai: expected={spec.repository}, actual={remote}"
        )

    status, error = git_value(checkout, "status", "--porcelain")
    if error:
        problems.append(f"không đọc được status {spec.directory}: {error}")
    elif status:
        problems.append(f"checkout {spec.directory} không sạch: {status!r}")

    for artifact in spec.artifacts:
        target = checkout / artifact.path
        if not target.is_file():
            problems.append(f"{spec.directory} thiếu artifact: {artifact.path}")
            continue
        actual_hash = file_sha256(target)
        if actual_hash != artifact.sha256:
            problems.append(
                f"{spec.directory}/{artifact.path} SHA-256 sai: "
                f"expected={artifact.sha256}, actual={actual_hash}"
            )
    return problems


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Kiểm tra references.lock.yaml và các checkout read-only."
    )
    parser.add_argument(
        "--lock",
        type=Path,
        default=Path("references.lock.yaml"),
        help="Reference lock được pin bằng content hash.",
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path(".references"),
        help="Thư mục chứa các clone reference.",
    )
    parser.add_argument(
        "--lock-only",
        action="store_true",
        help="Chỉ kiểm immutable lock; phù hợp CI không chứa .references.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    lock = args.lock.expanduser().resolve()
    problems, invocation_error = validate_lock(lock)

    if not args.lock_only:
        source_root = args.source_root.expanduser().resolve()
        if not source_root.is_dir():
            problems.append(f"source root không tồn tại: {source_root}")
            invocation_error = True
        else:
            for spec in SOURCES:
                problems.extend(validate_checkout(source_root, spec))

    if problems:
        for problem in problems:
            print(f"ERROR: {problem}")
        print(f"Reference source check failed: {len(problems)} lỗi.")
        return 2 if invocation_error else 1

    checked = "lock-only" if args.lock_only else "2 required checkouts + optional-present checkouts"
    print(f"Reference source check passed: immutable lock; {checked}; clean pinned HEAD/hash/origin.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
