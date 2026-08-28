"""WRK-R5: a packet that does not hash itself, and rebuilds to the same digest.

RRV-05: the packet walked the evidence tree, met the previous packet, hashed it,
and then overwrote it. An attestation whose content depends on the attestation it
replaces cannot be rebuilt, so a reviewer's signature over it binds to nothing.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from qdgrasp.roadmap.review_packet import (
    EXCLUDED_FIELDS,
    canonical_digest,
    canonical_packet_digest,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
CUDA_EVIDENCE = "evidence/phase3_4_3/s10/kaggle-run-v8/cuda-gate.json"


def _build(out_dir: Path) -> dict:
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/phase3_4_3_review_packet.py",
            "--out",
            str(out_dir),
            "--cuda-evidence",
            CUDA_EVIDENCE,
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    target = out_dir / "review-packet.json"
    assert target.is_file(), completed.stderr[-2000:]
    return json.loads(target.read_text(encoding="utf-8"))


def test_the_digest_excludes_itself_and_the_timestamp():
    assert "packet_sha256" in EXCLUDED_FIELDS
    assert "assembled_at" in EXCLUDED_FIELDS


def test_the_canonical_digest_ignores_a_changed_timestamp():
    base = {"phase": "3.4.3", "commit": "a" * 40, "assembled_at": "2026-01-01"}
    later = {**base, "assembled_at": "2026-12-31"}
    assert canonical_digest(base) == canonical_digest(later)


def test_a_stale_declared_digest_does_not_change_the_canonical_one():
    base = {"phase": "3.4.3", "commit": "a" * 40}
    assert canonical_digest(base) == canonical_digest({**base, "packet_sha256": "stale"})


def test_the_packet_does_not_hash_itself(tmp_path):
    packet = _build(tmp_path / "one")
    references = [
        name for name in packet["sub_phase_evidence"] if "review-packet" in name
    ]
    assert references == [], references


def test_rebuilding_the_same_candidate_gives_the_same_digest(tmp_path):
    first = _build(tmp_path / "one")
    second = _build(tmp_path / "two")
    assert canonical_digest(first) == canonical_digest(second)


def test_the_declared_digest_is_the_canonical_one(tmp_path):
    out = tmp_path / "one"
    packet = _build(out)
    assert packet["packet_sha256"] == canonical_packet_digest(out / "review-packet.json")
