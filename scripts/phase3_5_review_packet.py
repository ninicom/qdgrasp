#!/usr/bin/env python3
"""Assemble the immutable Phase 3.5 review packet (P3.5-18).

A reviewer cannot check a claim they have to take on trust, so this collects
what ``ROADMAP-P3.5-001`` §13.9 says the review covers -- asset transforms and
units, the CoACD surface and its security posture, settle semantics, the Gym
API, reward accounting, backend parity and cloud evidence -- and hashes all of
it against one commit.

Two things it deliberately does not do, both for the same reason.

It does not issue a verdict.  The author of a change cannot review it, and a
script that stamped PASS here would be exactly that.  What it produces is the
material a reviewer signs *against*.

It does not fill in evidence that is missing.  The GPU spike has not run, so the
packet records it as absent and says which gate that blocks, rather than
quietly shipping a smaller packet that looks complete.

    python scripts/phase3_5_review_packet.py --out evidence/phase3_5/review
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

PACKET_SCHEMA = "qdgrasp/evidence/phase3.5-review-packet/v1"
PACKET_FILENAME = "review-packet.json"

#: What the reviewer must be able to read.  A packet missing one of these is
#: incomplete, not smaller, and the packet says so.
REQUIRED_ARTIFACTS: tuple[str, ...] = (
    "docs/roadmap/PHASE3_5_ASSET_SCENE_RL_READINESS_PLAN.md",
    "docs/sessions/SESSION-20260831-002-phase3-5-ingest-and-rl-contracts.md",
    "docs/revisions/REV-20260831-002-phase3-5-execution.md",
    "qdgrasp/objects/ingest.py",
    "qdgrasp/objects/coacd.py",
    "qdgrasp/objects/manifest_v2.py",
    "qdgrasp/scenes/resolver.py",
    "qdgrasp/scenes/serialize.py",
    "qdgrasp/scenes/virtual_drop.py",
    "qdgrasp/scenes/settle.py",
    "qdgrasp/rl/contracts.py",
    "qdgrasp/rl/randomization.py",
    "qdgrasp/rl/envs/hand_scene.py",
    "qdgrasp/rl/envs/object_settle.py",
    "qdgrasp/rl/envs/dex_acquire.py",
    "qdgrasp/rl/tasks/scripted.py",
    "qdgrasp/rl/tasks/grasp_prior.py",
    "scripts/check_phase3_5.py",
    "scripts/phase3_5_gpu_rl_readiness.py",
    "scripts/generate_rl_env_tiny.py",
    "notebooks/phase3_5_rl_readiness.ipynb",
    "datasets/qdgrasp-rl-env-tiny/dataset_manifest.json",
    "datasets/qdgrasp-rl-env-tiny/artifact_hashes.json",
)

#: Evidence the plan requires that this machine cannot produce.  Named here so a
#: reviewer sees the hole rather than having to notice its absence.
KNOWN_ABSENT: tuple[dict[str, str], ...] = (
    {
        "artifact": "GPU backend spike evidence",
        "gate": "P3.5-15",
        "reason": (
            "no NVIDIA GPU on the machine that built this packet. ADR-0006 forbids a CPU run standing in "
            "for a CUDA one, and §7 forbids a backend decision without measured two-hand parity."
        ),
        "how_to_produce": "run notebooks/phase3_5_rl_readiness.ipynb on a Kaggle or Colab GPU runtime",
    },
    {
        "artifact": "CoACD output-class parity against the Stage 0 artifacts",
        "gate": "P3.5-03/04 test matrix",
        "reason": (
            "neither CoACD nor ManifoldPlus is installed, and neither was added as a dependency. The "
            "profile's parameters are pinned and tested; the parts it produces are not."
        ),
        "how_to_produce": "install the audited CoACD wheel and ManifoldPlus, then re-run the profile parity fixture",
    },
)

#: The review scope of §13.9, so a reviewer can tick items rather than infer them.
REVIEW_SCOPE: tuple[dict[str, str], ...] = (
    {
        "area": "asset transforms and units",
        "where": "qdgrasp/objects/ingest.py, tests/assets_ingest/test_ingest.py",
        "question": "is the unit scale applied exactly once, and is every transform recorded with the raw input hash?",
    },
    {
        "area": "CoACD surface and security",
        "where": "qdgrasp/objects/coacd.py, tests/assets_ingest/test_coacd_api.py",
        "question": "no network, no implicit writes, every official parameter typed and in the config hash?",
    },
    {
        "area": "settle semantics",
        "where": "qdgrasp/scenes/settle.py, tests/rl_env/test_scene_and_envs.py",
        "question": "does 'settled' require every object quiet simultaneously for the pinned consecutive steps?",
    },
    {
        "area": "scene resolution",
        "where": "qdgrasp/scenes/resolver.py",
        "question": "does a broken scene reference fail rather than silently become a generated scene?",
    },
    {
        "area": "Gym API",
        "where": "qdgrasp/rl/contracts.py, qdgrasp/rl/envs/",
        "question": "are terminated and truncated separated, and does every terminated step name one reason?",
    },
    {
        "area": "reward accounting",
        "where": "qdgrasp/rl/contracts.py",
        "question": "is the total the sum of logged terms, and can no positive term pay for a safety barrier?",
    },
    {
        "area": "backend parity",
        "where": "scripts/phase3_5_gpu_rl_readiness.py",
        "question": "ABSENT -- the GPU spike has not run; see known_absent",
    },
    {
        "area": "cloud evidence",
        "where": "notebooks/phase3_5_rl_readiness.ipynb",
        "question": "does the notebook pin an immutable commit and refuse to label a CPU run as GPU evidence?",
    },
    {
        "area": "artifact",
        "where": "datasets/qdgrasp-rl-env-tiny/",
        "question": "do the positive, negative and random cases each behave as their class requires, with hashes?",
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
    gate = REPO_ROOT / "datasets/qdgrasp-rl-env-tiny/dataset_manifest.json"
    tiny_summary = json.loads(gate.read_text(encoding="utf-8")).get("summary", {}) if gate.is_file() else {}

    packet: dict[str, Any] = {
        "schema": PACKET_SCHEMA,
        "phase": "P3.5",
        "plan": "ROADMAP-P3.5-001",
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
        "gate_command": "python scripts/check_phase3_5.py --profile micro",
        "gate_expected_exit": 1,
        "gate_expected_note": (
            "The gate is expected to exit 1 while P3.5-15 and P3.5-18 are open. An exit of 0 without GPU "
            "evidence and a signed review means the gate was widened, not that the phase finished."
        ),
        "review_scope": list(REVIEW_SCOPE),
        "known_absent": list(KNOWN_ABSENT),
        "rl_env_tiny_summary": tiny_summary,
        "artifacts": artifacts,
        "missing_artifacts": missing,
        "complete": worktree_clean and not missing,
    }
    packet["packet_digest"] = _canonical_digest({key: value for key, value in packet.items() if key != "created_at"})
    return packet


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "evidence/phase3_5/review")
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
