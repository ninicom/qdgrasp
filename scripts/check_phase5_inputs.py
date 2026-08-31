#!/usr/bin/env python3
"""Are P5's inputs sufficient to train on at all? (ROADMAP-P5-001 §0.1)

P5 is the first phase whose output is a number an outsider could cite, so the
question "is there enough signal in the data to produce one" has to be answered
before the training loop is written, not after a week of runs.

The measurement is blunt: how many **successful** grasps does the locked
protocol's train split actually contain, per active hand.  A generative model
regressed onto labels that are overwhelmingly failed proposals learns the
proposal distribution, not grasping; and a quality head with three positives
learns the prior.  Neither failure announces itself in a loss curve.

This exits non-zero when the count is below the floor.  That is the point: an
insufficient dataset should stop P5 the way a missing GPU stops P4, loudly and
by name, instead of being discovered in the discussion section.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

#: Positives per hand below which the train split cannot support a claim.
#: Not a statistical threshold -- there is no honest one at this scale -- but a
#: line under which "we trained on it" would be a misuse of the word.
MINIMUM_POSITIVES_PER_HAND = 25


def measure(dataset_root: Path, protocol_path: Path) -> dict[str, Any]:
    from qdgrasp.models.data import iter_active_datasets, load_manifest
    from qdgrasp.models.protocol import check_dataset_agreement, load_protocol

    protocol = load_protocol(protocol_path)
    manifest = load_manifest(dataset_root)
    check_dataset_agreement(protocol, manifest)

    wanted = {"train": set(protocol.train_objects), "val": set(protocol.val_objects)}
    rows: list[dict[str, Any]] = []
    for split, objects in wanted.items():
        for robot, dataset in iter_active_datasets(dataset_root, split=split, verify=False):
            kept = [sample for sample in dataset if sample["object_id"] in objects]
            positives = sum(int(float(sample["success"])) for sample in kept)
            rows.append(
                {
                    "split": split,
                    "robot": robot,
                    "samples": len(kept),
                    "positives": positives,
                    "positive_fraction": (positives / len(kept)) if kept else 0.0,
                    "objects": len({sample["object_id"] for sample in kept}),
                }
            )
    train_rows = [row for row in rows if row["split"] == "train"]
    return {
        "schema": "qdgrasp/phase5-inputs/v1",
        "dataset_id": manifest.get("dataset_id"),
        "protocol": protocol.name,
        "protocol_hash": protocol.protocol_hash,
        "minimum_positives_per_hand": MINIMUM_POSITIVES_PER_HAND,
        "rows": rows,
        "train_positives_total": sum(row["positives"] for row in train_rows),
        "sufficient": all(row["positives"] >= MINIMUM_POSITIVES_PER_HAND for row in train_rows),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=REPO_ROOT / "datasets/dgn-open-tiny")
    parser.add_argument("--protocol", type=Path, default=REPO_ROOT / "configs/phase5/protocol-v1.yaml")
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args(argv)

    report = measure(args.dataset, args.protocol)
    print(f"dataset       {report['dataset_id']}")
    print(f"protocol      {report['protocol']}  ({report['protocol_hash'][:16]}…)")
    print(f"{'split':6s} {'hand':16s} {'samples':>8s} {'positives':>10s} {'fraction':>9s} {'objects':>8s}")
    for row in report["rows"]:
        print(
            f"{row['split']:6s} {row['robot']:16s} {row['samples']:8d} {row['positives']:10d} "
            f"{row['positive_fraction']:9.3f} {row['objects']:8d}"
        )
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"\nwrote {args.json}")

    if report["sufficient"]:
        print(f"\nEvery active hand has at least {MINIMUM_POSITIVES_PER_HAND} positives to train on.")
        return 0
    print(
        f"\nNot enough successful grasps to train on: {report['train_positives_total']} across the whole "
        f"active train split, against a floor of {MINIMUM_POSITIVES_PER_HAND} per hand."
    )
    print(
        "Regressing the generator onto these labels teaches the proposal distribution, not grasping, and a "
        "quality head at this ratio learns the prior. P5-03 onward stay blocked on the data layer until "
        "DGN-Open-Tiny is regenerated with a recipe that yields positives, or a larger corpus replaces it."
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
