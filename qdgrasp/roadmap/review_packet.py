"""Canonical hashing for the Phase 3.4.3 review packet.

WRK-R5. A packet that hashes itself is not an attestation of anything: the
digest changes the moment it is written into the file it describes. The digest
here is taken over the payload with the self-referential and time-varying fields
removed, so rebuilding the same candidate twice yields the same value, and a
reviewer's signature over that value keeps meaning after the packet is committed.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

#: Fields excluded from the canonical payload. ``packet_sha256`` is the digest
#: itself; ``assembled_at`` moves every rebuild without the candidate changing.
EXCLUDED_FIELDS: frozenset[str] = frozenset({"packet_sha256", "assembled_at"})


def canonical_payload(packet: Mapping[str, Any]) -> dict[str, Any]:
    """The part of a packet a signature is over."""
    return {key: value for key, value in packet.items() if key not in EXCLUDED_FIELDS}


def canonical_digest(packet: Mapping[str, Any]) -> str:
    """Digest of the canonical payload, stable across rebuilds."""
    blob = json.dumps(canonical_payload(packet), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def canonical_packet_digest(path: str | Path) -> str:
    """Digest of the packet stored at ``path``."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"{path} does not contain a packet object")
    return canonical_digest(payload)


def manifest_digest(path: str | Path) -> str:
    """Short digest of the requirements manifest, for stamping derived prose.

    WRK-R6. The manifest is the only source of truth for status. Prose that
    restates counts has to say which manifest it restated, or a stale sentence
    and a current one look identical to a reader.
    """
    blob = Path(path).read_bytes()
    return hashlib.sha256(blob).hexdigest()[:12]
