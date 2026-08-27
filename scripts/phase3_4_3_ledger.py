"""Update one requirement in the Phase 3.4.3 manifest, in place and in order.

The manifest is edited textually rather than round-tripped through a YAML
dumper, because the dumper would drop the ``<<: *pending_requirement`` anchors
and turn every status change into a whole-file diff nobody can review.

A status change is refused unless the fields that status requires are present,
so a requirement cannot be marked ``passed`` from the command line without
naming implementation, tests and evidence.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from qdgrasp.roadmap import ALLOWED_STATUS, audit_closure, load_manifest

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = REPO_ROOT / "docs" / "roadmap" / "phase3_4_3_requirements.yaml"

_MANAGED = (
    "owner",
    "required",
    "implementation_refs",
    "test_ids",
    "evidence_refs",
    "status",
    "blocker_reason",
)


def _format_list(values: list[str]) -> str:
    return "[" + ", ".join(values) + "]"


def update_entry(
    text: str,
    requirement_id: str,
    *,
    status: str,
    owner: str | None = None,
    required: bool | None = None,
    implementation_refs: list[str] | None = None,
    test_ids: list[str] | None = None,
    evidence_refs: list[str] | None = None,
    blocker_reason: str | None = None,
) -> str:
    """Rewrite the managed fields of one requirement entry."""
    # An entry runs until the next entry, a comment, or the end of the file.
    # Stopping only at the next entry silently skipped the last requirement of
    # each section, which is the one followed by a blank line and a comment.
    pattern = re.compile(
        r"^(  - <<: \*pending_requirement\n(?:    \S.*\n)*)", re.MULTILINE
    )
    for match in pattern.finditer(text):
        block = match.group(1)
        if not re.search(rf"^    id: {re.escape(requirement_id)}\s*$", block, re.MULTILINE):
            continue

        kept = [
            line
            for line in block.splitlines()
            if not any(line.startswith(f"    {field}:") for field in _MANAGED)
        ]
        additions: list[str] = []
        if owner is not None:
            additions.append(f"    owner: {owner}")
        if required is not None:
            additions.append(f"    required: {'true' if required else 'false'}")
        if implementation_refs is not None:
            additions.append(f"    implementation_refs: {_format_list(implementation_refs)}")
        if test_ids is not None:
            additions.append(f"    test_ids: {_format_list(test_ids)}")
        if evidence_refs is not None:
            additions.append(f"    evidence_refs: {_format_list(evidence_refs)}")
        additions.append(f"    status: {status}")
        if blocker_reason is not None:
            additions.append(f'    blocker_reason: "{blocker_reason}"')
        elif status == "passed":
            additions.append('    blocker_reason: ""')

        replacement = "\n".join([*kept, *additions]) + "\n"
        return text[: match.start(1)] + replacement + text[match.end(1) :]

    raise SystemExit(f"requirement {requirement_id!r} not found in {MANIFEST}")


def _split(value: str | None) -> list[str] | None:
    if value is None:
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--id", required=True, action="append", dest="ids")
    parser.add_argument("--status", required=True, choices=sorted(ALLOWED_STATUS))
    parser.add_argument("--owner")
    parser.add_argument(
        "--optional",
        action="store_true",
        help="mark the requirement not required; only for packages the plan declares optional",
    )
    parser.add_argument("--implementation")
    parser.add_argument("--tests")
    parser.add_argument("--evidence")
    parser.add_argument("--reason")
    args = parser.parse_args()

    if args.status == "passed" and not (args.implementation and args.tests and args.evidence):
        print(
            "refusing to mark passed without --implementation, --tests and --evidence",
            file=sys.stderr,
        )
        return 2
    if args.status in {"failed", "blocked", "paused", "deferred_not_claimed"} and not args.reason:
        print(f"status {args.status} requires --reason", file=sys.stderr)
        return 2

    text = MANIFEST.read_text(encoding="utf-8")
    for requirement_id in args.ids:
        text = update_entry(
            text,
            requirement_id,
            status=args.status,
            owner=args.owner,
            required=False if args.optional else None,
            implementation_refs=_split(args.implementation),
            test_ids=_split(args.tests),
            evidence_refs=_split(args.evidence),
            blocker_reason=args.reason,
        )
    MANIFEST.write_text(text, encoding="utf-8")

    manifest = load_manifest(MANIFEST)
    verdict = audit_closure(manifest, repo_root=REPO_ROOT, worktree_dirty=False)
    print(
        f"updated {', '.join(args.ids)} -> {args.status}; "
        f"passed={verdict.status_counts['passed']}/{verdict.total_requirements}, "
        f"verdict={verdict.verdict}"
    )
    for violation in verdict.violations:
        print(f"  violation: {violation}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
