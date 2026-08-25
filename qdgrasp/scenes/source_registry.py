"""Pinned external scene-source provenance and redistribution policy."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from qdgrasp.config.schema import ConfigError

SOURCE_IDS = ("graspnet1b", "dexgraspnet2", "graspclutter6d")
_REQUIRED = {
    "dataset_id",
    "version",
    "source_url",
    "evidence_url",
    "license",
    "verified_date",
    "expected_layout",
    "redistributable",
}


def load_source_records() -> dict[str, dict[str, Any]]:
    root = Path(__file__).resolve().parent / "sources"
    records: dict[str, dict[str, Any]] = {}
    for dataset_id in SOURCE_IDS:
        path = root / f"{dataset_id}.json"
        if not path.is_file():
            raise ConfigError(f"external source record missing: {path}")
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise ConfigError(f"invalid external source record {path}: {exc}") from exc
        missing = sorted(_REQUIRED - set(payload))
        if missing or payload.get("dataset_id") != dataset_id:
            raise ConfigError(f"external source record identity incomplete for {dataset_id}: missing={missing}")
        string_fields = _REQUIRED - {"redistributable"}
        if any(not isinstance(payload[field], str) or not payload[field] for field in string_fields):
            raise ConfigError(f"external source record has empty identity fields: {dataset_id}")
        if payload["redistributable"] is not False:
            raise ConfigError(f"external source must remain non-redistributable until independent review: {dataset_id}")
        records[dataset_id] = payload
    return records
