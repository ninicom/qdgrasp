#!/usr/bin/env python3
"""Assemble the independent-review packet (P3.4.1-09).

The author may prepare the packet and may not sign it. Section 5.2 is explicit:
the reviewer runs in a separate context, does not write or edit artifacts, is not
handed a desired conclusion, and reads only the exact commit plus this packet.

So this writes hashes, a frozen checklist and reproduction commands, and leaves
the verdict field empty. Filling it in here would defeat the point of asking.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]

EVIDENCE_ROOTS = ("evidence/phase3_4", "evidence/phase3_4_1")
SOURCE_ROOTS = (
    "qdgrasp/dynamic",
    "qdgrasp/sim/batched",
    "qdgrasp/dataset/dynamic_contracts.py",
    "qdgrasp/dataset/dynamic_shards.py",
    "scripts/check_phase3_4.py",
    "scripts/phase3_4_backend_spike.py",
    "scripts/phase3_4_cuda_contact_search.py",
    "scripts/phase3_4_ablation.py",
    "scripts/phase3_4_1_shadow_audit.py",
    "scripts/generate_contactrich_tiny.py",
    "tests/dynamic_grasp",
)

#: Frozen from ROADMAP-P3.4-001 section 16 and ROADMAP-P3.4.1-001 section 5.2.
#: Every item is stated so it can be answered no.
CHECKLIST = [
    "The 2.0x speed gate was not changed, and 4.444x is not presented as a full pass while 29 worlds went NaN.",
    "VRAM is measured outside the PyTorch allocator; the earlier 0.0 GiB figure is marked invalid rather than deleted.",
    "nonfinite, overflow and OOM are reported as three separate quantities.",
    "GPU and CPU run the same workload, and every finalist is replayed on the CPU oracle.",
    "The reject denominator is reported, not only the accepted count.",
    "No safety threshold, contact stiffness, actuator gain or force budget was lowered anywhere in Phase 3.4 or 3.4.1.",
    "The Shadow classification is supported by measurement, and no legitimate collision was disabled to reach it.",
    "Impacted P2, P3.2 and P3.3 claims are either untouched or replayed; the Shadow recipe is shared.",
    "Evidence hashes match the exact commit and the worktree is clean.",
    "Every failure is included in the packet, not only the runs that passed.",
    "Findings at severity S0 to S3 are resolved, or the verdict is not a pass.",
]

#: Author-declared disclosures. A reviewer should reject the packet if any value
#: other than the honest one appears here.
DISCLOSURES = {
    "safety_thresholds_changed": "none",
    "objective_weights_changed_after_seeing_results": "none",
    "gates_relaxed_to_pass": "none",
    "budget_changes": (
        "one: contact impulse is judged over a rolling window instead of "
        "accumulated over the whole rollout. Applied identically to all three "
        "hands. Rationale: a cumulative impulse limit rejects every sustained "
        "hold regardless of how gentle it is, so it measured grasp duration "
        "rather than safety. LEAP and Allegro pass under the same budget Shadow "
        "fails."
    ),
    "benchmark_operating_point_changed": (
        "yes: from 64 worlds to 1024. 64 used ~0 GiB of a 14 GiB budget, leaving "
        "the GPU idle. 1024 was declared once before running and both counts are "
        "reported. The batch was not looped upward until a number passed."
    ),
    "known_unresolved": [
        (
            "29-34 of 1024 GPU worlds go non-finite from identical inputs; all "
            "990 survivors also differ from each other, so the divergence is not "
            "confined to the rejected worlds."
        ),
        (
            "shadow_hand has no dynamic positive; option A fixes the safety "
            "violation but the sweep parameterised tendon-coupled joints as "
            "independent."
        ),
        "Data.overflow has never been read on a GPU run.",
    ],
}


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _collect(root: Path) -> dict[str, str]:
    if root.is_file():
        return {str(root.relative_to(REPO_ROOT)): _hash(root)}
    return {
        str(p.relative_to(REPO_ROOT)): _hash(p)
        for p in sorted(root.rglob("*"))
        if p.is_file() and "__pycache__" not in p.parts
    }


def build() -> dict[str, Any]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=REPO_ROOT, check=True, capture_output=True, text=True,
    ).stdout.strip()

    manifest: dict[str, str] = {}
    for group in (*SOURCE_ROOTS, *EVIDENCE_ROOTS):
        manifest.update(_collect(REPO_ROOT / group))

    return {
        "schema": "qdgrasp/review-packet/phase3.4.1/v1",
        "work_package": "P3.4.1-09",
        "commit": commit,
        "worktree_clean": not dirty,
        "dirty_paths": dirty.splitlines(),
        "manifest_sha256": manifest,
        "manifest_entries": len(manifest),
        "checklist": CHECKLIST,
        "author_disclosures": DISCLOSURES,
        "reproduction": [
            "scripts/check_phase3_4.py --backend cpu --profile micro",
            "python -m pytest tests/dynamic_grasp/ -q",
            "scripts/phase3_4_1_shadow_audit.py",
            "scripts/phase3_4_ablation.py",
            "on a real NVIDIA device: scripts/phase3_4_cuda_contact_search.py --device cuda:0",
        ],
        "verdict": {
            "value": None,
            "allowed": ["PASS", "FAIL", "BLOCKED"],
            "reviewer_type": None,
            "allowed_reviewer_types": ["external", "internal_independent"],
            "reviewer_identity": None,
            "signed_at": None,
            "note": (
                "Left empty deliberately. The author of these artifacts cannot "
                "sign this; a verdict written here by the author would satisfy "
                "no part of section 5.2."
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("evidence/phase3_4_1/review-packet"))
    args = parser.parse_args()
    packet = build()
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "MANIFEST.sha256").write_text(
        "\n".join(f"{h}  {p}" for p, h in sorted(packet["manifest_sha256"].items())) + "\n",
        encoding="utf-8",
    )
    trimmed = {k: v for k, v in packet.items() if k != "manifest_sha256"}
    (args.out / "packet.json").write_text(
        json.dumps(trimmed, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({k: trimmed[k] for k in ("commit", "worktree_clean", "manifest_entries")}, indent=2))
    print(f"verdict left empty; {packet['manifest_entries']} artifacts hashed", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
