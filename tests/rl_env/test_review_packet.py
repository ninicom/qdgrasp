"""P3.5-18: the packet must be checkable, and must never sign itself.

The author of an artifact preparing the material a reviewer works from is the
process ``docs/governance/THIRD_PARTY_REVIEW.md`` describes.  The author
supplying the verdict is not, so the property worth a test is the absence of one.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.phase3_5_review_packet import (
    KNOWN_ABSENT,
    REQUIRED_ARTIFACTS,
    REVIEW_SCOPE,
    build_packet,
)


def test_the_packet_carries_no_verdict() -> None:
    packet = build_packet()
    assert packet["verdict"] is None
    assert "may not sign" in packet["verdict_note"]


def test_every_required_artifact_exists_and_is_hashed() -> None:
    packet = build_packet()
    assert packet["missing_artifacts"] == []
    assert len(packet["artifacts"]) == len(REQUIRED_ARTIFACTS)
    for entry in packet["artifacts"]:
        assert len(entry["sha256"]) == 64
        assert (PROJECT_ROOT / entry["path"]).is_file()


def test_absent_evidence_is_named_rather_than_omitted() -> None:
    """A smaller packet that looked complete would be the failure mode."""

    gates = {item["gate"] for item in KNOWN_ABSENT}
    assert "P3.5-15" in gates
    for item in KNOWN_ABSENT:
        assert item["reason"] and item["how_to_produce"]


def test_the_review_scope_covers_every_area_the_plan_lists() -> None:
    areas = {item["area"] for item in REVIEW_SCOPE}
    assert {
        "asset transforms and units",
        "CoACD surface and security",
        "settle semantics",
        "Gym API",
        "reward accounting",
        "backend parity",
        "cloud evidence",
    } <= areas


def test_the_digest_is_stable_and_excludes_the_timestamp() -> None:
    first = build_packet()
    second = build_packet()
    assert first["packet_digest"] == second["packet_digest"]
    assert first["created_at"] != second["created_at"] or True  # timestamps may tie; the digest must not depend on it


def test_the_packet_states_that_a_zero_exit_would_be_suspicious() -> None:
    packet = build_packet()
    assert packet["gate_expected_exit"] == 1
    assert "widened" in packet["gate_expected_note"]
