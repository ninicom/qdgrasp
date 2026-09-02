#!/usr/bin/env python3
"""Copy the MVP run artifacts into the committed evidence tree with a manifest.

``runs/`` is ignored by Git, so a result that only exists there is a result
nobody else can check.  This lifts the auditable parts -- reports, per-episode
ledgers, generator ledgers and the checkpoints themselves -- into
``evidence/mvp/`` and writes a sorted SHA-256 manifest over what it copied.

The demonstration tensors are deliberately left behind: they are large, and they
regenerate byte-for-byte from the committed scope, prior and seeds.  Their
ledgers and summaries, which are what an auditor actually reads, do come along.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

DEFAULT_RUNS = Path("runs/mvp")
DEFAULT_OUT = Path("evidence/mvp")

#: Copied verbatim when present.  Anything not listed stays in ``runs/``.
COPY_GLOBS = (
    "demonstrations/index.json",
    "demonstrations/*/ledger.jsonl",
    "demonstrations/*/manifest.json",
    "demonstrations/*/summary.json",
    "policy/training-report.json",
    "policy/bc.pt",
    "policy/ppo.pt",
    "evaluation/*.json",
    "evaluation/*/tier-*.jsonl",
    # The release contract's own artifacts.  A sealed evidence set that lacked
    # them would still pass the experimental gate and could never reach a
    # release verdict, which is the confusion the two gates exist to prevent.
    "contribution.json",
    "closure.json",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=DEFAULT_RUNS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--no-checkpoints",
        action="store_true",
        help="skip the weight files; for a superseded round, the reports and ledgers are the evidence",
    )
    args = parser.parse_args(argv)

    if not args.runs.is_dir():
        print(f"FAIL no run directory at {args.runs}")
        return 1
    if args.out.exists():
        shutil.rmtree(args.out)
    args.out.mkdir(parents=True, exist_ok=True)

    patterns = [p for p in COPY_GLOBS if not (args.no_checkpoints and p.endswith(".pt"))]
    copied: list[Path] = []
    for pattern in patterns:
        for source in sorted(args.runs.glob(pattern)):
            if not source.is_file():
                continue
            destination = args.out / source.relative_to(args.runs)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            copied.append(destination)

    artifacts = [
        {
            "path": str(path.relative_to(args.out)),
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in sorted(copied)
    ]
    manifest = {
        "schema": "qdgrasp/mvp-evidence-manifest/v0",
        "source": str(args.runs),
        "artifacts": artifacts,
    }
    (args.out / "MANIFEST.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    total = sum(int(entry["bytes"]) for entry in artifacts)
    print(f"copied {len(copied)} artifacts ({total / 1e6:.2f} MB) into {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
