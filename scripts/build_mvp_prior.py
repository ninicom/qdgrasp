#!/usr/bin/env python3
"""Fit and write the LEAP pinch prior table for the locked MVP scope.

The table is committed rather than fitted at import time: the IK solve is slow
enough to matter in a training loop, and a committed artifact makes "the prior
changed" visible in a diff.  Re-running this script on the same scope must
reproduce the same table to solver tolerance, which ``tests/mvp`` asserts.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from qdgrasp.mvp.config import load_mvp_scope
from qdgrasp.mvp.prior import DEFAULT_PRIOR_PATH, build_pinch_prior_table
from qdgrasp.robot.spec import RobotSpec


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", type=Path, default=None, help="MVP scope YAML")
    parser.add_argument("--out", type=Path, default=DEFAULT_PRIOR_PATH, help="output prior JSON")
    args = parser.parse_args(argv)

    scope = load_mvp_scope(args.scope)
    spec = RobotSpec.from_config(scope.robot_profile, sample_anchors=False)
    widths = [variant.half_width for variant in scope.train_variants]
    table = build_pinch_prior_table(spec, widths)
    path = table.save(args.out)
    print(f"scope_hash    {scope.content_hash()}")
    print(f"prior_hash    {table.content_hash()}")
    print(f"wrote         {path}")
    for knot in table.knots:
        print(f"  half_width={knot.half_width:.4f}  contact_residual={knot.contact_residual_m * 1e3:.3f} mm")
    return 0


if __name__ == "__main__":
    sys.exit(main())
