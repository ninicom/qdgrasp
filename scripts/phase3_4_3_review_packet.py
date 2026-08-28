#!/usr/bin/env python3
"""Assemble the immutable Phase 3.4.3 review packet (S12; G10).

A reviewer cannot check a claim they have to take on trust, so this collects
everything the plan says the review needs -- the exact commit, the environment
locks, the gate ledger, the safety coverage and mutation results, the trajectory
and parity evidence, the dataset manifest and shard hashes, and the known
limitations -- and hashes all of it.

Two things it deliberately does not do. It does not issue a verdict: the author
of a change cannot review it, and a script that stamped PASS here would be the
author signing their own work. And it does not fill in evidence that is missing;
an absent CUDA run is recorded as absent, which is what blocks the release.

    python scripts/phase3_4_3_review_packet.py --out evidence/phase3_4_3/s12
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]

#: The packet's own filename, excluded from the evidence it collects.
PACKET_FILENAME = "review-packet.json"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from qdgrasp.config.active_scope import ACTIVE_HANDS, PAUSED_HANDS
from qdgrasp.roadmap import audit_closure, load_manifest
from qdgrasp.roadmap.review_packet import canonical_digest

PACKET_SCHEMA = "qdgrasp/evidence/phase3.4.3-review-packet/v1"

#: What the reviewer has to be able to read, by path. A packet missing one of
#: these is incomplete rather than smaller.
REQUIRED_ARTIFACTS: tuple[str, ...] = (
    "docs/roadmap/PHASE3_4_3_ACTIVE_GATE_COMPLETION_PLAN.md",
    "docs/roadmap/PHASE3_4_3_EXECUTION_BREAKDOWN.md",
    "docs/roadmap/phase3_4_3_requirements.yaml",
    "docs/decisions/0008-temporary-shadow-hand-pause.md",
    "docs/revisions/REV-20260827-010-mjwarp-upstream-defect.md",
    "environments/environment.lock.yaml",
    "references.lock.yaml",
    "robot_assets.lock.yaml",
)

#: Implementation the review is about. Hashed so a reviewer can tell whether the
#: code they read is the code the evidence came from.
REVIEWED_IMPLEMENTATION: tuple[str, ...] = (
    "qdgrasp/dataset/dynamic_contracts.py",
    "qdgrasp/dataset/dynamic_shards.py",
    "qdgrasp/dataset/contactrich_active.py",
    "qdgrasp/dynamic",
    "qdgrasp/sim/batched",
    "qdgrasp/roadmap",
    "qdgrasp/config/active_scope.py",
    "scripts/check_phase3_4_3.py",
    "scripts/check_phase3_4_3_cuda.py",
    "scripts/check_contactrich_active.py",
    "scripts/generate_contactrich_active_tiny.py",
    "scripts/phase3_4_3_ablation.py",
)

#: What the reviewer is asked to check, from G10. Written into the packet so the
#: scope of the review is part of the record rather than a conversation.
REVIEW_CHECKLIST: tuple[str, ...] = (
    "force and impulse arithmetic, including the rolling window endpoints",
    "that no declared safety limit is left without a sensor",
    "the exact GPU-to-CPU replay lineage, capsule by capsule",
    "manifest counts, splits and shard hashes against what is on disk",
    "the ADR-0008 scope disclosure everywhere it appears",
    "the checker exit codes under mutation",
    "that nothing claims three-hand coverage or a historical P3.4 pass",
)


def _git(*args: str) -> str:
    completed = subprocess.run(
        ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, timeout=120, check=False
    )
    return completed.stdout.strip() if completed.returncode == 0 else ""


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _hash_ref(ref: str) -> dict[str, Any]:
    target = REPO_ROOT / ref
    if target.is_file():
        return {"kind": "file", "sha256": _sha256_file(target)}
    if target.is_dir():
        entries = {
            str(child.relative_to(REPO_ROOT)): _sha256_file(child)
            for child in sorted(target.rglob("*.py"))
            if child.is_file()
        }
        digest = hashlib.sha256(
            json.dumps(entries, sort_keys=True).encode("utf-8")
        ).hexdigest()
        return {"kind": "directory", "files": len(entries), "sha256": digest}
    return {"kind": "missing", "sha256": None}


def collect_evidence(root: Path, *, exclude: Path | None = None) -> dict[str, Any]:
    """Every recorded sub-phase result, by hash, except this packet itself.

    WRK-R5. The output directory has to be excluded or the packet hashes the
    previous packet before overwriting it, which is RRV-05: an attestation whose
    content depends on the attestation it replaces cannot be rebuilt, and a
    reviewer's signature over it binds to nothing stable.
    """
    out: dict[str, Any] = {}
    if not root.is_dir():
        return out
    excluded = exclude.resolve() if exclude is not None else None
    for path in sorted(root.rglob("*.json")):
        if excluded is not None and excluded in path.resolve().parents:
            continue
        # A packet is never evidence about itself, wherever it was written.
        if path.name == PACKET_FILENAME:
            continue
        out[str(path.relative_to(REPO_ROOT))] = _sha256_file(path)
    return out


def build_packet(
    *, dataset_root: Path, cuda_evidence: Path | None, out_dir: Path | None = None
) -> dict[str, Any]:
    manifest = load_manifest(REPO_ROOT / "docs" / "roadmap" / "phase3_4_3_requirements.yaml")
    closure = audit_closure(manifest, repo_root=REPO_ROOT)

    dataset_manifest_path = dataset_root / "dataset_manifest.json"
    dataset_block: dict[str, Any] = {"root": str(dataset_root), "present": dataset_manifest_path.is_file()}
    if dataset_manifest_path.is_file():
        payload = json.loads(dataset_manifest_path.read_text(encoding="utf-8"))
        dataset_block.update(
            {
                "manifest_sha256": _sha256_file(dataset_manifest_path),
                "dataset_id": payload.get("dataset_id"),
                "counts": payload.get("counts"),
                "coverage": payload.get("coverage"),
                "release_blocked": payload.get("release_blocked"),
                "blocked_reasons": payload.get("blocked_reasons"),
                "shard_hashes": {s["path"]: s["sha256"] for s in payload.get("shards", [])},
            }
        )

    cuda_block: dict[str, Any] = {"present": bool(cuda_evidence and cuda_evidence.is_file())}
    if cuda_evidence and cuda_evidence.is_file():
        cuda_payload = json.loads(cuda_evidence.read_text(encoding="utf-8"))
        cuda_block.update(
            {
                "path": str(cuda_evidence),
                "sha256": _sha256_file(cuda_evidence),
                "verdict": cuda_payload.get("verdict"),
                "gpu": cuda_payload.get("cuda_environment", {}).get("gpu_name"),
            }
        )
    else:
        cuda_block["detail"] = (
            "no CUDA gate run is in this packet. The Kaggle T4 notebook has not been "
            "executed against this commit, so the GPU gate is not passed -- it is "
            "not run, which blocks the release."
        )

    missing = [ref for ref in REQUIRED_ARTIFACTS if not (REPO_ROOT / ref).exists()]

    packet = {
        "schema": PACKET_SCHEMA,
        "phase": "3.4.3",
        "assembled_at": datetime.now(UTC).isoformat(),
        "commit": _git("rev-parse", "HEAD"),
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "worktree_dirty": bool(_git("status", "--porcelain")),
        "scope": {
            "active_hands": list(ACTIVE_HANDS),
            "paused_hands": list(PAUSED_HANDS),
            "three_hand_coverage": False,
            "historical_p3_4_state": "paused_by_ADR-0008",
            "verdict_name": "P3.4.3-ACTIVE-PASS",
        },
        "completeness": {
            "manifest_sha256": closure.manifest_sha256,
            "verdict": closure.verdict,
            "total_requirements": closure.total_requirements,
            "status_counts": dict(sorted(closure.status_counts.items())),
            "open_required": list(closure.open_required),
            "violations": list(closure.violations),
        },
        "required_artifacts": {ref: _hash_ref(ref) for ref in REQUIRED_ARTIFACTS},
        "missing_required_artifacts": missing,
        "reviewed_implementation": {ref: _hash_ref(ref) for ref in REVIEWED_IMPLEMENTATION},
        "sub_phase_evidence": collect_evidence(
            REPO_ROOT / "evidence" / "phase3_4_3", exclude=out_dir
        ),
        "dataset": dataset_block,
        "cuda_gate": cuda_block,
        "review_checklist": list(REVIEW_CHECKLIST),
        "known_limitations": [
            "Simulation-only contact. Nothing here is a hardware safety claim.",
            "Two active hands under ADR-0008; not three-hand coverage.",
            (
                "The upstream uninitialised-read defect is unresolved and now "
                "measured, not assumed: on a Kaggle T4, mujoco-warp 3.10.0.3, "
                "3.11.0 and 3.12.0 on warp-lang 1.16.0 each report tens of "
                "thousands of uninitialised reads in _linesearch_iterative_kernel "
                "on the REV-20260827-010 reproducer scene, while racecheck stays "
                "clean. warp-lang 1.16.0 is the newest release, so there is no "
                "newer runtime to move to, and the solver-configuration space "
                "is exhausted too: ls_parallel was removed upstream in 3.9.1, "
                "ls_iterations=1 still leaks 12788 reads so the defect fires on "
                "the first linesearch iteration, and solver_cg is an order of "
                "magnitude worse. The gate itself ran end to end and returned "
                "FAIL on measurement, not absence: parity holds to 5.75e-10 with "
                "nothing in contact and breaks to 8.39mm against a 2mm tolerance "
                "once a contact is involved, and 84 of 1024 LEAP worlds go "
                "non-finite. Speed passes at 5.47x and 14.04x, which does not buy "
                "back correctness. See evidence/phase3_4_3/s10/."
            ),
            (
                "The same three versions report zero errors on a three-geom toy "
                "scene. A compatibility answer is only as good as the scene it "
                "was asked on, and both readings are kept so that is visible."
            ),
            "MPPI (P3.4-10) is deferred_not_claimed and carries no coverage.",
            (
                "The static-vs-dynamic ablation reports no_measured_difference "
                "across four arms, and the fourth shows the paired evidence "
                "section 16.3 asks for cannot exist as specified rather than "
                "being absent from this corpus. Both predicates carry the same "
                "floor -- certify_force_closure needs two contacts and the "
                "dynamic predicate needs min_active_fingers=2 sustained -- so "
                "they cannot disagree in the direction 16.3 assumes, and the "
                "dynamic predicate further requires floor_support_after_lift to "
                "be false, which excludes the environment-supported grasps that "
                "would make a frozen force-closure test fail. A reviewer should "
                "treat this as a contract defect needing amendment, not as an "
                "implementation shortfall. No threshold was moved to avoid it."
            ),
        ],
        "reviewer_instructions": (
            "This packet carries no verdict. The author of a change cannot review "
            "it. A reviewer records their verdict in a separate file with "
            "reviewer, reviewer_verdict, open_findings and packet_sha256, and "
            "scripts/check_phase3_4_3.py reads that file."
        ),
    }
    return packet


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", type=Path, default=REPO_ROOT / "evidence" / "phase3_4_3" / "s12"
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=REPO_ROOT / "datasets" / "contactrich-active-tiny",
    )
    parser.add_argument("--cuda-evidence", type=Path, default=None)
    args = parser.parse_args()

    packet = build_packet(
        dataset_root=args.dataset_root,
        cuda_evidence=args.cuda_evidence,
        out_dir=args.out,
    )
    payload = json.dumps(packet, indent=2, sort_keys=True) + "\n"
    # WRK-R5: the signed digest is over the canonical payload, which carries
    # neither the digest itself nor the assembly timestamp. Rebuilding the same
    # candidate twice therefore yields the same value for a reviewer to sign.
    digest = canonical_digest(packet)
    packet_with_hash = json.dumps(
        {**packet, "packet_sha256": digest}, indent=2, sort_keys=True
    ) + "\n"

    args.out.mkdir(parents=True, exist_ok=True)
    target = args.out / PACKET_FILENAME
    target.write_text(packet_with_hash, encoding="utf-8")

    print(payload)
    print(f"wrote {target.relative_to(REPO_ROOT)}", file=sys.stderr)
    print(f"packet_sha256={digest}", file=sys.stderr)
    if packet["worktree_dirty"]:
        print(
            "worktree is dirty: this packet cannot be reproduced from a commit, "
            "so it is not a release candidate.",
            file=sys.stderr,
        )
        return 1
    if packet["missing_required_artifacts"]:
        print(
            f"missing required artifacts: {packet['missing_required_artifacts']}",
            file=sys.stderr,
        )
        return 1
    print(
        "Packet assembled. It carries no verdict: an independent reviewer has to "
        "issue one, and the author of the change is not one.",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
