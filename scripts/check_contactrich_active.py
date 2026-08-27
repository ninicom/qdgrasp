#!/usr/bin/env python3
"""Audit a QDGrasp-ContactRich-Active-Tiny artifact (S11; G09, C05, C06).

Reads the artifact the way a consumer would, and refuses it for the reasons a
consumer would care about: a schema that is not the release one, counts that
disagree with the shards, a hash that does not match, a shard path that escapes
the root, a split with a group on both sides, a negative control that passed.

``--require-release`` is the release gate. Without it the audit still runs and
still reports everything, but a blocked artifact is allowed through so it can be
inspected -- inspection never promotes it.

    python scripts/check_contactrich_active.py datasets/contactrich-active-tiny
    python scripts/check_contactrich_active.py DATASET_ROOT --require-release
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from qdgrasp.dataset.contactrich_active import (
    DatasetRejected,
    dataset_card,
    load,
    verify,
)

PASS_EXIT = 0
FAIL_EXIT = 1
BLOCKED_EXIT = 3
CONFIG_EXIT = 4


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="dataset root containing dataset_manifest.json")
    parser.add_argument(
        "--require-release",
        action="store_true",
        help="refuse an artifact that is release_blocked; this is the release gate",
    )
    parser.add_argument(
        "--write-card",
        type=Path,
        default=None,
        help="write the dataset card next to the artifact",
    )
    args = parser.parse_args()

    if not args.root.is_dir():
        print(json.dumps({"verdict": "CONFIG_ERROR", "error": f"{args.root} is not a directory"}, indent=2))
        return CONFIG_EXIT

    problems = verify(args.root, allow_blocked=not args.require_release)
    manifest_path = args.root / "dataset_manifest.json"
    manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
    )

    result = {
        "dataset_root": str(args.root),
        "dataset_id": manifest.get("dataset_id"),
        "schema": manifest.get("schema"),
        "release_blocked": manifest.get("release_blocked", True),
        "blocked_reasons": manifest.get("blocked_reasons", []),
        "counts": manifest.get("counts", {}),
        "coverage": manifest.get("coverage", {}),
        "scope": manifest.get("scope", {}),
        "problems": problems,
        "require_release": args.require_release,
    }

    if problems:
        result["verdict"] = "FAIL"
    elif manifest.get("release_blocked", True):
        result["verdict"] = "BLOCKED"
    else:
        result["verdict"] = "PASS"

    if args.write_card and manifest:
        args.write_card.parent.mkdir(parents=True, exist_ok=True)
        args.write_card.write_text(dataset_card(manifest), encoding="utf-8")
        result["dataset_card"] = str(args.write_card)

    print(json.dumps(result, indent=2, sort_keys=True))

    if result["verdict"] == "FAIL":
        for problem in problems:
            print(f"  PROBLEM: {problem}", file=sys.stderr)
        return FAIL_EXIT
    if result["verdict"] == "BLOCKED":
        print(
            "ContactRich-Active-Tiny: BLOCKED. The artifact is well-formed and "
            f"inspectable, and it is not cleared for release: {manifest.get('blocked_reasons')}",
            file=sys.stderr,
        )
        return BLOCKED_EXIT

    # A clean, unblocked artifact still has to load through the public path.
    try:
        dataset = load(args.root)
    except DatasetRejected as exc:
        print(f"the public loader refused an artifact the audit passed: {exc}", file=sys.stderr)
        return FAIL_EXIT
    for name, count in dataset.iter_splits():
        loaded = len(dataset.split(name))
        if loaded != count:
            print(
                f"split {name!r} declares {count} samples but loaded {loaded}",
                file=sys.stderr,
            )
            return FAIL_EXIT
    print(
        f"ContactRich-Active-Tiny: PASS for {list(dataset.active_hands)}. "
        "Two active hands; this is not three-hand coverage and does not close P3.4.",
        file=sys.stderr,
    )
    return PASS_EXIT


if __name__ == "__main__":
    raise SystemExit(main())
