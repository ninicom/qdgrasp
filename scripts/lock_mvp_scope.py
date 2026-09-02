#!/usr/bin/env python3
"""Write the immutable evaluation manifest for the locked MVP scope.

``ROADMAP-MVP-001`` MVP-00 requires the eval protocol to be frozen before the
final tune round.  The manifest is derived from the scope document, so this
script never invents anything; it makes the derivation auditable as a file with
a hash, and re-running it after any scope edit produces a visibly different one.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from qdgrasp.mvp.config import DEFAULT_SCOPE_PATH, load_mvp_scope


def manifest_path_for(scope_path: Path | None) -> Path:
    """Where a scope document's manifest belongs.

    Derived from the scope path rather than defaulted to one file, because a
    locker whose output default belongs to a *different* scope will happily
    overwrite that scope's frozen manifest when someone passes ``--scope`` and
    forgets ``--out``.
    """

    resolved = scope_path if scope_path is not None else DEFAULT_SCOPE_PATH
    return resolved.with_suffix("").with_suffix(".eval-manifest.json")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None, help="default: derived from the scope path")
    parser.add_argument("--check", action="store_true", help="verify the artifact matches the scope, write nothing")
    args = parser.parse_args(argv)

    if args.out is None:
        args.out = manifest_path_for(args.scope)
    scope = load_mvp_scope(args.scope)
    rendered = json.dumps(scope.eval_manifest(), indent=2, sort_keys=True) + "\n"
    if args.check:
        if not args.out.is_file():
            print(f"FAIL missing eval manifest: {args.out}")
            return 1
        if args.out.read_text(encoding="utf-8") != rendered:
            print(f"FAIL eval manifest at {args.out} does not match the scope document")
            return 1
        print(f"OK   eval manifest matches scope {scope.content_hash()}")
        return 0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(rendered, encoding="utf-8")
    print(f"scope_hash          {scope.content_hash()}")
    print(f"eval_manifest_hash  {scope.eval_manifest_hash()}")
    print(f"wrote               {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
