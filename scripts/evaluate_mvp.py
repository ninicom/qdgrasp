#!/usr/bin/env python3
"""MVP-05: run the locked evaluation on a candidate checkpoint.

The seeds come from the immutable evaluation manifest, the per-episode ledger is
written raw, and the aggregate carries Wilson bounds and failure buckets beside
the headline rate.  ``ROADMAP-MVP-001`` §8 allows one locked-eval run per
candidate per round; running it twice on the same candidate does not make the
first result go away, so the report records the checkpoint hash it measured.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

from qdgrasp.mvp.config import load_mvp_scope
from qdgrasp.mvp.evaluate import evaluate_candidate, format_report, write_report
from qdgrasp.mvp.policy import verify_reload_probe
from qdgrasp.mvp.prior import DEFAULT_PRIOR_PATH, PinchPriorTable

DEFAULT_OUTPUT = Path("runs/mvp/evaluation")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", type=Path, default=None)
    parser.add_argument("--prior", type=Path, default=DEFAULT_PRIOR_PATH)
    parser.add_argument("--checkpoint", type=Path, default=None, help="omit to measure the controller prior alone")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--label", type=str, default=None)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = parser.parse_args(argv)

    scope = load_mvp_scope(args.scope)
    prior = PinchPriorTable.load(args.prior)
    label = args.label or ("controller_prior" if args.checkpoint is None else args.checkpoint.stem)
    ledger_dir = args.out / label

    reload_mismatch = 0
    checkpoint_meta: dict[str, str] = {}
    if args.checkpoint is not None:
        checkpoint_meta = {"checkpoint_sha256": _sha256(args.checkpoint)}
        # The reload gate compares this file against the answers the *training*
        # process recorded inside it.  Comparing a load against another load of
        # the same file would prove nothing.
        if not verify_reload_probe(args.checkpoint):
            reload_mismatch = 1

    report = evaluate_candidate(
        scope,
        prior,
        scope_path=str(args.scope) if args.scope is not None else None,
        prior_path=str(args.prior),
        checkpoint_path=str(args.checkpoint) if args.checkpoint is not None else None,
        workers=args.workers,
        ledger_dir=ledger_dir,
        reload_mismatch=reload_mismatch,
        label=label,
    )
    report.update(checkpoint_meta)
    path = write_report(args.out / f"{label}.json", report)
    print(format_report(report))
    print(f"wrote {path}")
    print(json.dumps({"all_tiers_passed": report["all_tiers_passed"]}))
    return 0 if report["all_tiers_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
