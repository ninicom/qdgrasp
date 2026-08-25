#!/usr/bin/env python3
"""Acceptance gate for Phase 3.3 scene-grasp data and release evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from qdgrasp.config.schema import ConfigError
from qdgrasp.dataset.scene_loader import audit_scene_dataset
from qdgrasp.dataset.scene_manifest import load_scene_manifest
from qdgrasp.dataset.scene_shards import read_scene_shard
from qdgrasp.scenes.adapters import get_adapter
from qdgrasp.scenes.release import DATASET_ID, generate_scene_tiny
from qdgrasp.scenes.source_registry import load_source_records

REPO_ROOT = Path(__file__).resolve().parents[1]
REAL_ROOTS = {
    "graspnet1b": "QDGRASP_GRASPNET1B_ROOT",
    "dexgraspnet2": "QDGRASP_DEXGRASPNET2_ROOT",
    "graspclutter6d": "QDGRASP_GRASPCLUTTER6D_ROOT",
}


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _bounded_tree_hashes(root: Path) -> dict[str, str]:
    return {path.relative_to(root).as_posix(): _file_hash(path) for path in sorted(root.rglob("*")) if path.is_file()}


def _records(root: Path, record_type: str) -> list[dict[str, Any]]:
    manifest = load_scene_manifest(root / "scene_manifest.json")
    records: list[dict[str, Any]] = []
    for shard in manifest.shards:
        if shard.record_type != record_type:
            continue
        records.extend(
            read_scene_shard(
                root / shard.filename,
                record_type=shard.record_type,
                expected_sha256=shard.sha256,
                expected_records=shard.num_records,
            )
        )
    return records


def audit_release(root: Path, *, require_full: bool) -> dict[str, Any]:
    root = root.resolve()
    manifest_path = root / "scene_manifest.json"
    dataset_manifest_path = root / "dataset_manifest.json"
    if not dataset_manifest_path.is_file() or manifest_path.read_bytes() != dataset_manifest_path.read_bytes():
        raise ConfigError("scene_manifest.json and dataset_manifest.json must be byte-identical")
    manifest = load_scene_manifest(manifest_path)
    if manifest.dataset_id != DATASET_ID:
        raise ConfigError(f"unexpected scene release dataset ID: {manifest.dataset_id}")
    if require_full and manifest.release_blocked:
        raise ConfigError("full QDGrasp-Scene-Tiny release is still blocked")
    counts = audit_scene_dataset(root)

    for reference, expected_hash in manifest.release_artifact_hashes.items():
        path = (root / reference).resolve()
        if not path.is_relative_to(root) or not path.is_file() or _file_hash(path) != expected_hash:
            raise ConfigError(f"release artifact integrity mismatch: {reference}")
    expected_split_hashes = {
        split: hashlib.sha256(json.dumps(scene_ids, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
        for split, scene_ids in manifest.splits.items()
    }
    if manifest.split_hashes != expected_split_hashes:
        raise ConfigError("release split hashes are inconsistent")

    positives = [record for record in _records(root, "grasp") if record.get("dynamic_valid")]
    negatives = [record for record in _records(root, "grasp") if not record.get("dynamic_valid")]
    if any(record.get("source_class") == "test_fixture_only" for record in positives):
        raise ConfigError("release contains a fabricated fixture positive")
    negative_classes = {record.get("failure_reason") for record in negatives}
    if not {"collision", "occlusion", "non_target_disturbance"}.issubset(negative_classes):
        raise ConfigError("release negative coverage is incomplete")

    families = manifest.coverage.get("object_families_by_split", {})
    family_sets = [set(values) for values in families.values()]
    if len(family_sets) > 1 and set.intersection(*family_sets):
        raise ConfigError("object family leakage detected across splits")
    templates = manifest.coverage.get("scene_templates", {})
    if len(set(templates.values())) != len(templates):
        raise ConfigError("scene template leakage detected")

    if require_full:
        if sum(len(scene_ids) for scene_ids in manifest.splits.values()) != 12:
            raise ConfigError("full release must contain exactly 12 scenes")
        if manifest.coverage.get("environment_counts") != {"bin": 4, "shelf": 4, "table": 4}:
            raise ConfigError("full release environment coverage is incomplete")
        if set(manifest.coverage.get("clutter_tier_counts", {})) != {"single", "sparse", "dense"}:
            raise ConfigError("full release clutter tiers are incomplete")
        if set(manifest.coverage.get("robot_profiles", [])) != {
            "leap_hand.yaml",
            "wonik_allegro.yaml",
            "shadow_hand.yaml",
        }:
            raise ConfigError("full release robot coverage is incomplete")
        observations = _records(root, "observation")
        views: dict[str, set[str]] = {}
        for record in observations:
            views.setdefault(record["scene_id"], set()).add(record["camera_id"])
        if any(len(camera_ids) < 2 for camera_ids in views.values()) or len(views) != 12:
            raise ConfigError("full release requires two camera views per scene")
        if len(positives) != 3 or manifest.coverage.get("qa_stage_images") != 12:
            raise ConfigError("full release requires three positives and twelve stage QA images")
        for record in positives:
            stages = {item["stage"] for item in record.get("qa_stage_evidence", [])}
            if stages != {"pregrasp", "squeeze", "lift", "perturbation"}:
                raise ConfigError(f"positive QA stage coverage incomplete: {record['candidate_id']}")
    return {
        "counts": counts,
        "positive_count": len(positives),
        "negative_classes": sorted(negative_classes),
        "artifact_count": len(manifest.release_artifact_hashes),
    }


def _run_pytest() -> None:
    env = dict(os.environ)
    env.update(
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        PYTHONHASHSEED="0",
    )
    command = [
        str(REPO_ROOT / ".venv/bin/python"),
        "-m",
        "pytest",
        "-q",
        "tests/scenes",
        "tests/pipeline/test_release_recipes.py",
        "tests/pipeline/test_scene_rollout.py",
        "tests/test_scene_dataset.py",
    ]
    subprocess.run(command, cwd=REPO_ROOT, env=env, check=True, timeout=120)


def _real_source_smoke() -> dict[str, str]:
    results: dict[str, str] = {}
    for adapter_name, variable in REAL_ROOTS.items():
        value = os.environ.get(variable)
        if not value:
            results[adapter_name] = "not_configured"
            continue
        adapter = get_adapter(adapter_name)
        root = str(Path(value).resolve())
        info = adapter.probe(root)
        if not info.is_valid:
            raise ConfigError(f"configured external root failed probe: {adapter_name}")
        index = adapter.index(root, "train", limit=1)
        if len(index.scene_keys) != 1:
            raise ConfigError(f"configured external root has no bounded train scene: {adapter_name}")
        evidence = adapter.audit(root, index.scene_keys[0])
        if not evidence.is_complete:
            raise ConfigError(f"configured external root failed audit: {adapter_name}: {evidence.missing_files}")
        results[adapter_name] = "pass"
    return results


def run_micro_gate() -> dict[str, Any]:
    source_records = load_source_records()
    dry_run = generate_scene_tiny(Path("unused"), scene_limit=1, frame_limit=1, worker_count=1, dry_run=True)
    if dry_run["full_root_scan"] or dry_run["source_copy"]:
        raise ConfigError("resource dry-run reports unsafe source behavior")
    _run_pytest()
    with (
        tempfile.TemporaryDirectory(prefix="qdgrasp-p33-a-") as first_dir,
        tempfile.TemporaryDirectory(prefix="qdgrasp-p33-b-") as second_dir,
    ):
        first = Path(first_dir)
        second = Path(second_dir)
        generate_scene_tiny(first, scene_limit=1, frame_limit=1, worker_count=1)
        generate_scene_tiny(second, scene_limit=1, frame_limit=1, worker_count=1)
        first_hashes = _bounded_tree_hashes(first)
        second_hashes = _bounded_tree_hashes(second)
        if first_hashes != second_hashes:
            changed = sorted(set(first_hashes) ^ set(second_hashes))
            changed += sorted(
                name for name in set(first_hashes) & set(second_hashes) if first_hashes[name] != second_hashes[name]
            )
            raise ConfigError(f"micro regenerate parity failed: {changed[:10]}")
        release_audit = audit_release(first, require_full=False)
        native_evidence = get_adapter("native").audit(str(first), "table-leap-sparse")
        if not native_evidence.is_complete:
            raise ConfigError(f"native release adapter audit failed: {native_evidence.missing_files}")
    return {
        "profile": "micro",
        "source_records": sorted(source_records),
        "real_source_smoke": _real_source_smoke(),
        "regenerate_parity": "pass",
        "release_audit": release_audit,
        "resource_policy": dry_run,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("micro", "release"), default="micro")
    parser.add_argument("--dataset-root", type=Path)
    args = parser.parse_args()
    try:
        if args.profile == "micro":
            result = run_micro_gate()
        else:
            if args.dataset_root is None:
                raise ConfigError("--dataset-root is required for profile=release")
            result = {
                "profile": "release",
                "source_records": sorted(load_source_records()),
                "real_source_smoke": _real_source_smoke(),
                "release_audit": audit_release(args.dataset_root, require_full=True),
            }
    except (ConfigError, OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        print(f"Phase 3.3 gate failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
