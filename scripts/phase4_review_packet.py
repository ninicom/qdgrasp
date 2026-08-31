#!/usr/bin/env python3
"""Assemble the immutable Phase 4 review packet (P4-12).

A reviewer cannot check a claim they have to take on trust, so this collects
what ``ROADMAP-P4-001`` §7 says the review covers -- the tokenizer's injectivity,
the absence of any ``N x N`` tensor, gradient coverage, output validity,
cross-embodiment, the loss accounting and the CUDA gate -- and hashes all of it
against one commit.

Two things it deliberately does not do, both for the same reason.

It does not issue a verdict.  The author of a change cannot review it, and a
script that stamped PASS here would be exactly that.  What it produces is the
material a reviewer signs *against*.

It does not fill in evidence that is missing.  The CUDA gate has not run, so the
packet records it as absent and says which gate that blocks, rather than quietly
shipping a smaller packet that looks complete.

    python scripts/phase4_review_packet.py --out evidence/phase4/review
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
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

PACKET_SCHEMA = "qdgrasp/evidence/phase4-review-packet/v1"
PACKET_FILENAME = "review-packet.json"

#: What the reviewer must be able to read.  A packet missing one of these is
#: incomplete, not smaller, and the packet says so.
REQUIRED_ARTIFACTS: tuple[str, ...] = (
    "docs/roadmap/PHASE4_EXECUTION_PLAN.md",
    "docs/roadmap/PHASE4_REVIEWER_GUIDE.md",
    "docs/sessions/SESSION-20260831-003-phase4-qdgrasp-flow.md",
    "docs/revisions/REV-20260831-003-phase4-architecture.md",
    "qdgrasp/models/tokenizer.py",
    "qdgrasp/models/encoder.py",
    "qdgrasp/models/hand_graph.py",
    "qdgrasp/models/flow.py",
    "qdgrasp/models/losses.py",
    "qdgrasp/models/config.py",
    "qdgrasp/presets/qdgrasp-flow-n.yaml",
    "qdgrasp/presets/qdgrasp-flow-s.yaml",
    "qdgrasp/presets/qdgrasp-flow-m.yaml",
    "scripts/check_phase4.py",
    "scripts/overfit_qdgrasp_flow.py",
    "scripts/phase4_cuda_gate.py",
    "notebooks/phase4_cuda_gate.ipynb",
    "tests/model_flow/test_model_flow.py",
    "tests/model_flow/test_model_config.py",
    "tests/model_flow/test_phase4_gate.py",
    "evidence/phase4/overfit-leap-cpu.json",
    "evidence/phase4/overfit-allegro-cpu.json",
)

#: Evidence the plan requires that this machine cannot produce.  Named here so a
#: reviewer sees the hole rather than having to notice its absence.
KNOWN_ABSENT: tuple[dict[str, str], ...] = (
    {
        "artifact": "CUDA forward/backward, parity and memory evidence",
        "gate": "P4-11",
        "reason": (
            "no NVIDIA GPU on the machine that built this packet. ADR-0006 forbids a CPU run standing in "
            "for a CUDA one, and §7.4/§7.5 require a measured CUDA overfit and CPU/CUDA FP32 parity."
        ),
        "how_to_produce": "run notebooks/phase4_cuda_gate.ipynb on a Kaggle or Colab GPU runtime",
    },
)

#: The review scope of §7, so a reviewer can tick items rather than infer them.
REVIEW_SCOPE: tuple[dict[str, str], ...] = (
    {
        "area": "tokenizer injectivity",
        "where": "qdgrasp/models/tokenizer.py, tests/model_flow/test_model_flow.py",
        "question": "is the token key positional and injective, and is a grid too fine to pack refused rather than aliased?",
    },
    {
        "area": "no N x N",
        "where": "qdgrasp/models/encoder.py, qdgrasp/models/flow.py, test_attention_never_sees_two_long_sides",
        "question": "does any attention call have two long sides, and does the probe that says otherwise have a working negative control?",
    },
    {
        "area": "cross-embodiment",
        "where": "qdgrasp/models/hand_graph.py",
        "question": "do LEAP (18 nodes) and Allegro (22) run through one encoder with no size compiled in, and is node order irrelevant?",
    },
    {
        "area": "output validity",
        "where": "qdgrasp/models/flow.py",
        "question": "is every rotation in SO(3) within 1e-4 and every joint inside its named limit, by construction rather than by clipping afterwards?",
    },
    {
        "area": "loss accounting",
        "where": "qdgrasp/models/losses.py",
        "question": "is the total the sum of the logged terms, is an unknown term refused, and is rotation error geodesic?",
    },
    {
        "area": "gradient coverage",
        "where": "qdgrasp/models/losses.py, evidence/phase4/overfit-leap-cpu.json",
        "question": "does every trainable parameter receive a finite gradient after one backward?",
    },
    {
        "area": "the overfit verdict",
        "where": "scripts/overfit_qdgrasp_flow.py",
        "question": "is the verdict read off pose error rather than the total, and is the reason for that stated where the thresholds are set?",
    },
    {
        "area": "config and registry",
        "where": "qdgrasp/models/config.py, qdgrasp/presets/qdgrasp-flow-*.yaml",
        "question": "is an unknown preset parameter refused rather than ignored, and can a preset change a shape the scale table owns?",
    },
    {
        "area": "CUDA gate",
        "where": "scripts/phase4_cuda_gate.py",
        "question": "ABSENT -- the CUDA gate has not run; see known_absent. Does the harness refuse a CPU run rather than label it?",
    },
    {
        "area": "scope of the claim",
        "where": "docs/sessions/SESSION-20260831-003-phase4-qdgrasp-flow.md",
        "question": "is every P4 number presented as architecture evidence, with no sentence that could be read as a grasping result?",
    },
)


def _git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.SubprocessError):  # pragma: no cover
        return "unknown"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_digest(document: Any) -> str:
    payload = json.dumps(document, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_packet() -> dict[str, Any]:
    artifacts: list[dict[str, Any]] = []
    missing: list[str] = []
    for relative in REQUIRED_ARTIFACTS:
        path = REPO_ROOT / relative
        if not path.is_file():
            missing.append(relative)
            continue
        artifacts.append({"path": relative, "bytes": path.stat().st_size, "sha256": _sha256(path)})

    worktree_clean = _git("status", "--porcelain") == ""
    overfit_reports = sorted((REPO_ROOT / "evidence/phase4").glob("overfit-*-cpu.json"))
    overfits = [json.loads(path.read_text(encoding="utf-8")) for path in overfit_reports]

    packet: dict[str, Any] = {
        "schema": PACKET_SCHEMA,
        "phase": "P4",
        "plan": "ROADMAP-P4-001",
        "created_at": datetime.now(UTC).isoformat(),
        "commit": _git("rev-parse", "HEAD"),
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "worktree_clean": worktree_clean,
        "verdict": None,
        "verdict_note": (
            "This packet carries no verdict. The author of an artifact may not sign its review "
            "(docs/governance/THIRD_PARTY_REVIEW.md); an independent reviewer records the verdict "
            "against the packet digest below."
        ),
        "gate_command": "python scripts/check_phase4.py --profile micro",
        "gate_expected_exit": 1,
        "gate_expected_note": (
            "The gate is expected to exit 1 while P4-11b and P4-12 are open. An exit of 0 without CUDA "
            "evidence and a signed review means the gate was widened, not that the phase finished."
        ),
        "claim": (
            "The architecture trains. That is the whole claim. ROADMAP-P4-001 §7 forbids citing any number "
            "here as a grasping result, and the reviewer is asked to check that no document does."
        ),
        "review_scope": list(REVIEW_SCOPE),
        "known_absent": list(KNOWN_ABSENT),
        "overfit_summary": [
            {
                "robot": report.get("robot"),
                "device": report.get("device"),
                "converged": report.get("converged"),
                "gradient_coverage": report.get("gradient_coverage"),
                "pose_thresholds": report.get("pose_thresholds"),
                "pose_thresholds_met": report.get("pose_thresholds_met"),
                "last": report.get("last"),
            }
            for report in overfits
        ],
        "artifacts": artifacts,
        "missing_artifacts": missing,
        "complete": worktree_clean and not missing,
    }
    packet["packet_digest"] = _canonical_digest({key: value for key, value in packet.items() if key != "created_at"})
    return packet


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "evidence/phase4/review")
    args = parser.parse_args(argv)

    packet = build_packet()
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / PACKET_FILENAME
    path.write_text(json.dumps(packet, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"commit          {packet['commit']}")
    print(f"worktree clean  {packet['worktree_clean']}")
    print(f"artifacts       {len(packet['artifacts'])} hashed")
    if packet["missing_artifacts"]:
        print(f"MISSING         {packet['missing_artifacts']}")
    print(f"known absent    {[item['gate'] for item in packet['known_absent']]}")
    print(f"packet digest   {packet['packet_digest']}")
    print(f"wrote           {path}")
    if not packet["complete"]:
        print(
            "\nPacket is incomplete. A reviewer may still read it, but the missing items must be "
            "produced before a verdict can cover them."
        )
        return 1
    print("\nPacket is complete and carries no verdict, which is the point: an independent reviewer signs it.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
