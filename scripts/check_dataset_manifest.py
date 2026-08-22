"""Cryptographic audit and integrity validation for dataset releases."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from qdgrasp.config.schema import ConfigError
from qdgrasp.dataset.manifest import load_dataset_manifest
from qdgrasp.objects.manifest import load_object_asset


def audit_dataset_manifest(dataset_root: str | Path) -> dict[str, object]:
    """Perform a complete cryptographic audit of all shards and procedural objects."""
    root = Path(dataset_root).resolve()
    manifest_path = root / "dataset_manifest.json"
    if not manifest_path.is_file():
        raise ConfigError(f"dataset manifest missing: {manifest_path}")

    manifest = load_dataset_manifest(manifest_path)

    if manifest.license != "CC0-1.0":
        raise ConfigError(f"unauthorized dataset license: expected 'CC0-1.0', got '{manifest.license}'")
    if manifest.release_blocked:
        raise ConfigError("dataset manifest has release_blocked=True")

    # Verify disjoint splits
    train_objs = set(manifest.splits.get("train", []))
    val_objs = set(manifest.splits.get("val", []))
    if not train_objs:
        raise ConfigError("empty train split")
    if not val_objs:
        raise ConfigError("empty val split")
    overlap = train_objs & val_objs
    if overlap:
        raise ConfigError(f"split leakage between train and val: {sorted(overlap)}")

    # Verify object assets
    obj_dir = root / "objects"
    if not obj_dir.is_dir():
        raise ConfigError(f"objects directory missing: {obj_dir}")

    total_objects = 0
    for obj_id in train_objs | val_objs:
        man_p = obj_dir / f"{obj_id}.manifest.json"
        load_object_asset(man_p)
        total_objects += 1

    # Verify shard SHA-256 integrity
    total_samples = 0
    total_positives = 0
    for shard in manifest.shards:
        shard_p = root / shard.filename
        if not shard_p.is_file():
            raise ConfigError(f"shard file missing: {shard_p}")

        data = shard_p.read_bytes()
        actual_sha = hashlib.sha256(data).hexdigest()
        if actual_sha != shard.sha256:
            raise ConfigError(
                f"cryptographic integrity mismatch on {shard.filename}: expected {shard.sha256}, got {actual_sha}"
            )
        total_samples += shard.num_samples
        total_positives += shard.positive_samples

    return {
        "dataset_id": manifest.dataset_id,
        "shards": len(manifest.shards),
        "total_objects": total_objects,
        "total_samples": total_samples,
        "total_positives": total_positives,
        "status": "PASS",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit dataset release integrity.")
    parser.add_argument("--root", default="datasets/dgn-open-tiny", help="Path to dataset root.")
    args = parser.parse_args()

    try:
        summary = audit_dataset_manifest(args.root)
        print(json.dumps(summary, indent=2, sort_keys=True))
    except Exception as exc:
        print(f"Audit FAIL: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
