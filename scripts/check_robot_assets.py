#!/usr/bin/env python3
"""Validate the immutable robot_assets.lock.yaml and local read-only checkouts."""

from __future__ import annotations

import argparse
import hashlib
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

LOCK_SHA256 = "19b4011eec37c2324eb7d9132babff03611946608462bb9f800592646f4b49f9"


@dataclass(frozen=True)
class Artifact:
    path: str
    sha256: str


@dataclass(frozen=True)
class RobotSource:
    directory: str
    repository: str
    commit: str
    required: bool
    artifacts: tuple[Artifact, ...] = ()
    trees: tuple[tuple[str, str], ...] = ()


SOURCES = (
    RobotSource(
        directory="mujoco-menagerie",
        repository="https://github.com/google-deepmind/mujoco_menagerie.git",
        commit="da76818e269b82289eba39808e2fb91d679d6994",
        required=True,
        artifacts=(
            Artifact(
                "shadow_hand/right_hand.xml",
                "ac14d09dd12b62c3ca62233d0ac646101cc718146dda76e26d0b993a91abf399",
            ),
            Artifact(
                "shadow_hand/LICENSE",
                "d42cae9a0f292b7a713597ba26e1e1e23d50df906c488f0d36fd3538af200bc1",
            ),
            Artifact(
                "wonik_allegro/right_hand.xml",
                "cec3efe34be49c8220c23afdae0cfacbce10164a5bc7c34075b6668ee0a862a6",
            ),
            Artifact(
                "wonik_allegro/LICENSE",
                "09c187439ecece8b00330e71eb988d3d30f29373c580dc31ab54200d0aa4deb4",
            ),
            Artifact(
                "leap_hand/right_hand.xml",
                "9d8f25c38bdb60617f378c07b7e71f97306471353c81bf281eb9943f3f2340d8",
            ),
            Artifact(
                "leap_hand/LICENSE",
                "5a91c3d709c2ee3a71f166b46e3208cebd83c0c7a870746c729b31a01f156984",
            ),
        ),
        trees=(
            ("shadow_hand", "8f85903630df9e0738b2df3c3cb0e8b3dd421c63"),
            ("wonik_allegro", "42564d556620bc0c3ba41823bbdf860887f10ff4"),
            ("leap_hand", "efe8f001e555cc08cdd21fdbacc30bfd896241e3"),
        ),
    ),
    RobotSource(
        directory="leap-hand-sim",
        repository="https://github.com/leap-hand/LEAP_Hand_Sim.git",
        commit="150bc3d4b61fd6619193ba5a8ef209f3609ced89",
        required=True,
        artifacts=(
            Artifact(
                "assets/leap_hand/robot.urdf",
                "b3f29881e1c013f8b4eda85e9afecc6b2f4b89b5b49b643183ef6031422ed7f4",
            ),
            Artifact(
                "LICENSE.txt",
                "5a91c3d709c2ee3a71f166b46e3208cebd83c0c7a870746c729b31a01f156984",
            ),
        ),
        trees=(
            ("assets/leap_hand", "57d56e2e67d9c0a38c0037ed26b7f505a4b114c9"),
        ),
    ),
    RobotSource(
        directory="wonik-allegro-ros2",
        repository="https://github.com/Wonikrobotics-git/allegro_hand_ros2_v5.git",
        commit="80bd4a88d2c59b8ad0242ec3730302bde61c84fb",
        required=True,
        artifacts=(
            Artifact(
                "src/allegro_hand_controllers/urdf/allegro_hand_description_right_A.urdf",
                "3c0965dff548fdccab3f0b21eb0eef3575278282f839b29872e50ef2a6f4ba09",
            ),
            Artifact(
                "LICENSE",
                "1fe45dba5ed74a680ad1c68a704fe1e4a257bcafcf964f8ef531affe6f1d3c32",
            ),
        ),
        trees=(
            ("src/allegro_hand_controllers", "feed20ce5e4f912ce3d49dd2c27fd8283c4295ff"),
        ),
    ),
    RobotSource(
        directory="dex-urdf",
        repository="https://github.com/dexsuite/dex-urdf.git",
        commit="f5e7132f22108164577fea4c25ef99b5cc0e1900",
        required=True,
        artifacts=(
            Artifact(
                "robots/hands/shadow_hand/shadow_hand_right.urdf",
                "29a50063cced2fba81dfe3d208ba54499fc12635c9189d0822f7caa2771c36de",
            ),
            Artifact(
                "robots/hands/barrett_hand/bhand_model.urdf",
                "13cbfdb7715e54f4e10f9e1f2c7620db6c5f6b2b05adf9a16cdd795cda620fc1",
            ),
        ),
        trees=(
            ("robots/hands/shadow_hand", "713c63bf9b817dc08e270566d54471777887e25f"),
            ("robots/hands/barrett_hand", "13b890008d3b3f6269e4651f7b2073c92c899818"),
        ),
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
        return [f"robot assets lock không tồn tại: {path}"], True
    try:
        actual = file_sha256(path)
    except OSError as exc:
        return [f"không đọc được robot assets lock: {exc}"], True
    if actual != LOCK_SHA256:
        return [f"robot assets lock SHA-256 sai: expected={LOCK_SHA256}, actual={actual}"], False

    text = path.read_text(encoding="utf-8")
    if "removed_from_active_scope" not in text or "rh56e2" not in text:
        return ["robot assets lock thiếu chỉ thị exclusion RH56E2"], False

    return [], False


def validate_checkout(source_root: Path, spec: RobotSource) -> list[str]:
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

    for subpath, expected_tree in spec.trees:
        tree, error = git_value(checkout, "rev-parse", f"HEAD:{subpath}")
        if error:
            problems.append(f"{spec.directory}:{subpath} không đọc được git tree: {error}")
        elif tree != expected_tree:
            problems.append(
                f"{spec.directory}:{subpath} git tree sai: expected={expected_tree}, actual={tree}"
            )

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
        description="Kiểm tra robot_assets.lock.yaml và các checkout robot read-only."
    )
    parser.add_argument(
        "--lock",
        type=Path,
        default=Path("robot_assets.lock.yaml"),
        help="Robot assets lock được pin bằng content hash.",
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path(".references/robot-assets"),
        help="Thư mục chứa các clone robot asset.",
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
        print(f"Robot assets check failed: {len(problems)} lỗi.")
        return 2 if invocation_error else 1

    checked = "lock-only" if args.lock_only else f"{len(SOURCES)} required checkouts"
    print(f"Robot assets check passed: immutable lock; {checked}; clean pinned HEAD/hash/origin.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
