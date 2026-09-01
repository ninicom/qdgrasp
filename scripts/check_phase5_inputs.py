#!/usr/bin/env python3
"""Are P5's inputs sufficient to train on at all? (ROADMAP-P5-001 §0.1)

P5 is the first phase whose output is a number an outsider could cite, so the
question "is there enough signal in the data to produce one" has to be answered
before the training loop is written, not after a week of runs.

The measurement is blunt: how many **successful** grasps does the locked
protocol's train split actually contain for each hand the protocol admits to
training.  A generative model
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
    from qdgrasp.config.active_scope import ACTIVE_HANDS
    from qdgrasp.dataset.loader import DgnOpenDataset
    from qdgrasp.models.protocol import load_protocol

    protocol = load_protocol(protocol_path)
    rows: list[dict[str, Any]] = []
    dataset_id = ""
    manifest_hash = ""
    for split in ("train", "val"):
        for robot in ACTIVE_HANDS:
            dataset = DgnOpenDataset(
                dataset_root=dataset_root,
                split=split,
                robot_name=robot,
                protocol_file=protocol_path,
            )
            dataset_id = dataset.manifest_spec.dataset_id
            manifest_hash = dataset.artifact.manifest_hash
            positives = sum(int(float(sample["success"])) for sample in dataset.samples)
            admitted_to_training = split != "train" or robot == protocol.heldout_embodiment.train_hand
            rows.append(
                {
                    "split": split,
                    "robot": robot,
                    "samples": len(dataset),
                    "positives": positives,
                    "positive_fraction": (positives / len(dataset)) if len(dataset) else 0.0,
                    "objects": len({sample["object_id"] for sample in dataset.samples}),
                    "admitted_to_training": admitted_to_training,
                }
            )
    train_rows = [row for row in rows if row["split"] == "train" and row["admitted_to_training"]]
    return {
        "schema": "qdgrasp/phase5-inputs/v1",
        "dataset_id": dataset_id,
        "dataset_manifest_hash": manifest_hash,
        "protocol": protocol.name,
        "protocol_hash": protocol.protocol_hash,
        "minimum_positives_per_hand": MINIMUM_POSITIVES_PER_HAND,
        "rows": rows,
        "train_positives_total": sum(row["positives"] for row in train_rows),
        "sufficient": bool(train_rows) and all(row["positives"] >= MINIMUM_POSITIVES_PER_HAND for row in train_rows),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=REPO_ROOT / "datasets/dgn-open-tiny")
    parser.add_argument("--protocol", type=Path, default=REPO_ROOT / "configs/phase5/protocol-v2.yaml")
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args(argv)

    try:
        report = measure(args.dataset, args.protocol)
    except Exception as error:  # noqa: BLE001 - the gate reports, it does not traceback
        print(f"dataset       {args.dataset}")
        print(f"protocol      {args.protocol}")
        print(
            f"\nThe corpus could not be measured: {type(error).__name__}: {error}\n"
            "A dataset that cannot be verified cannot be counted, so P5 stays blocked on the data layer."
        )
        return 1

    print(f"dataset       {report['dataset_id']}")
    print(f"protocol      {report['protocol']}  ({report['protocol_hash'][:16]}…)")
    print(f"{'split':6s} {'hand':16s} {'samples':>8s} {'positives':>10s} {'fraction':>9s} {'objects':>8s} {'role':>9s}")
    for row in report["rows"]:
        role = "train" if row["split"] == "train" and row["admitted_to_training"] else "evaluate"
        print(
            f"{row['split']:6s} {row['robot']:16s} {row['samples']:8d} {row['positives']:10d} "
            f"{row['positive_fraction']:9.3f} {row['objects']:8d} {role:>9s}"
        )
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"\nwrote {args.json}")

    if report["sufficient"]:
        print(f"\nEvery protocol-admitted training hand has at least {MINIMUM_POSITIVES_PER_HAND} positives.")
        return 0
    print(
        f"\nNot enough successful grasps to train on: {report['train_positives_total']} across the whole "
        f"protocol-admitted train split, against a floor of {MINIMUM_POSITIVES_PER_HAND} per training hand."
    )
    print(
        "Regressing the generator onto these labels teaches the proposal distribution, not grasping, and a "
        "quality head at this ratio learns the prior. P5-03 onward stay blocked on the data layer until "
        "DGN-Open-Tiny is regenerated with a recipe that yields positives, or a larger corpus replaces it."
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
