#!/usr/bin/env python3
"""Generate the bounded QDGrasp-Scene-Tiny native release."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from qdgrasp.scenes.release import generate_scene_tiny


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=3301)
    parser.add_argument("--scene-limit", type=int, default=12)
    parser.add_argument("--frame-limit", type=int, default=2)
    parser.add_argument("--worker-count", type=int, default=1)
    parser.add_argument("--width", type=int, default=160)
    parser.add_argument("--height", type=int, default=120)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    result = generate_scene_tiny(
        args.output_root,
        seed=args.seed,
        scene_limit=args.scene_limit,
        frame_limit=args.frame_limit,
        worker_count=args.worker_count,
        width=args.width,
        height=args.height,
        dry_run=args.dry_run,
        resume=args.resume,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
