"""Cryptographic audit and integrity validation for dataset releases."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

from qdgrasp.config.loader import load_robot_config
from qdgrasp.config.schema import ConfigError
from qdgrasp.dataset import DatasetArtifact
from qdgrasp.dataset.manifest import DATASET_MANIFEST_SCHEMA_V3
from qdgrasp.dataset.shards import read_shard_file
from qdgrasp.objects.manifest import load_object_asset


def audit_dataset_manifest(dataset_root: str | Path) -> dict[str, object]:
    """Perform a complete cryptographic audit of all shards and procedural objects."""
    root = Path(dataset_root).resolve()
    artifact = DatasetArtifact.open_verified(root)
    manifest_path = artifact.manifest_path
    manifest = artifact.manifest

    if manifest.schema_version != DATASET_MANIFEST_SCHEMA_V3:
        raise ConfigError(f"release manifest must use {DATASET_MANIFEST_SCHEMA_V3}, got {manifest.schema_version}")
    if manifest.recipe_id == "legacy" or manifest.proposal_module == "legacy":
        raise ConfigError("dataset manifest lacks recipe/module provenance")
    if not manifest.generator_source_hashes:
        raise ConfigError("dataset manifest lacks generator source hashes")

    if manifest.license != "CC0-1.0":
        raise ConfigError(f"unauthorized dataset license: expected 'CC0-1.0', got '{manifest.license}'")
    if manifest.release_blocked:
        raise ConfigError("dataset manifest has release_blocked=True")
    if manifest.invalidated:
        raise ConfigError(
            f"dataset manifest is marked invalidated: {manifest.invalidation_reason or 'no reason recorded'}"
        )
    if manifest.generator_commit == "legacy" or manifest.generator_worktree_dirty:
        raise ConfigError("dataset was not generated from a recorded clean commit")
    if manifest.success_criteria.get("min_contacts", 0.0) < 2.0:
        raise ConfigError("success criteria permit fewer than two fingers")
    if manifest.success_criteria.get("max_penetration", float("inf")) > 0.002:
        raise ConfigError("success criteria permit excessive penetration")

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

    for robot_name, expected_hash in manifest.robot_profile_hashes.items():
        profile = load_robot_config(f"robots/{robot_name}.yaml")
        actual_hash = profile.content_hash()
        if actual_hash != expected_hash:
            raise ConfigError(f"robot profile provenance mismatch for {robot_name}")

    # Verify object assets
    obj_dir = root / "objects"
    if not obj_dir.is_dir():
        raise ConfigError(f"objects directory missing: {obj_dir}")

    total_objects = 0
    for obj_id in train_objs | val_objs:
        man_p = obj_dir / f"{obj_id}.manifest.json"
        load_object_asset(man_p)
        expected_object_hash = manifest.object_manifest_hashes.get(obj_id)
        actual_object_hash = hashlib.sha256(man_p.read_bytes()).hexdigest()
        if expected_object_hash != actual_object_hash:
            raise ConfigError(f"object manifest provenance mismatch for {obj_id}")
        total_objects += 1

    # Verify shard SHA-256 integrity
    total_samples = 0
    total_positives = 0
    observed_pairs: set[tuple[str, str]] = set()
    for shard in manifest.shards:
        shard_relative = Path(shard.filename)
        if shard_relative.is_absolute() or ".." in shard_relative.parts:
            raise ConfigError(f"unsafe shard path: {shard.filename}")
        pair = (shard.split, shard.robot_name)
        if pair in observed_pairs:
            raise ConfigError(f"duplicate shard for split/robot pair: {pair}")
        observed_pairs.add(pair)
        if shard.split not in manifest.splits:
            raise ConfigError(f"unknown split on {shard.filename}: {shard.split}")
        if shard.robot_name not in manifest.robot_profile_hashes:
            raise ConfigError(f"unknown robot on {shard.filename}: {shard.robot_name}")
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
        samples = read_shard_file(shard_p, expected_sha256=shard.sha256)
        if len(samples) != shard.num_samples:
            raise ConfigError(
                f"sample count mismatch on {shard.filename}: manifest={shard.num_samples}, actual={len(samples)}"
            )
        actual_positives = 0
        for sample_index, sample in enumerate(samples):
            required = {
                "success",
                "dynamic_valid",
                "static_force_valid",
                "collision_valid",
                "ik_valid",
                "proposal_valid",
                "recipe_id",
                "frame",
                "proposal_module",
                "solver_module",
                "certifier_version",
                "dynamic_protocol_version",
                "success_schema_version",
                "object_id",
                "robot_name",
                "failure_stage",
                "failure_reason",
                "kinematics_valid",
                "pose_target_valid",
                "joint_target_valid",
                "fk_target_valid",
            }
            missing = sorted(required - set(sample))
            if missing:
                raise ConfigError(f"sample {sample_index} in {shard.filename} lacks {missing}")
            success = bool(float(sample["success"]) > 0.5)
            dynamic_valid = bool(sample["dynamic_valid"])
            if success != dynamic_valid:
                raise ConfigError(f"sample {sample_index} in {shard.filename} has success/dynamic mismatch")
            if success and not all(
                bool(sample[field])
                for field in (
                    "proposal_valid",
                    "ik_valid",
                    "collision_valid",
                    "static_force_valid",
                    "dynamic_valid",
                )
            ):
                raise ConfigError(f"positive sample {sample_index} in {shard.filename} skipped a stage")
            stage_flags = [
                bool(sample[field])
                for field in (
                    "proposal_valid",
                    "ik_valid",
                    "collision_valid",
                    "static_force_valid",
                    "dynamic_valid",
                )
            ]
            if any(stage_flags[index] and not stage_flags[index - 1] for index in range(1, 5)):
                raise ConfigError(f"sample {sample_index} in {shard.filename} has non-monotonic stage flags")
            expected_failure_stage = next(
                (
                    stage
                    for stage, valid in zip(
                        ("proposal", "ik", "collision", "static_force", "dynamic"),
                        stage_flags,
                    )
                    if not valid
                ),
                "none",
            )
            actual_failure_stage = str(sample["failure_stage"])
            if expected_failure_stage == "dynamic":
                stage_matches = actual_failure_stage.startswith("dynamic")
            else:
                stage_matches = actual_failure_stage == expected_failure_stage
            if not stage_matches:
                raise ConfigError(f"sample {sample_index} in {shard.filename} has inconsistent failure stage")
            if sample["robot_name"] != shard.robot_name or sample["object_id"] not in manifest.splits[shard.split]:
                raise ConfigError(f"sample {sample_index} in {shard.filename} has split/robot drift")
            sample_provenance = (
                sample["recipe_id"],
                sample["proposal_module"],
                sample["solver_module"],
                sample["certifier_version"],
                sample["dynamic_protocol_version"],
            )
            manifest_provenance = (
                manifest.recipe_id,
                manifest.proposal_module,
                manifest.solver_module,
                manifest.certifier_version,
                manifest.dynamic_protocol_version,
            )
            if (
                sample_provenance != manifest_provenance
                or sample["frame"] != "object"
                or sample["success_schema_version"] != "dynamic-only-v1"
            ):
                raise ConfigError(f"sample {sample_index} in {shard.filename} has provenance/frame drift")
            actual_positives += int(success)
        if actual_positives != shard.positive_samples:
            raise ConfigError(
                f"positive count mismatch on {shard.filename}: "
                f"manifest={shard.positive_samples}, actual={actual_positives}"
            )
        if shard.recipe_id != manifest.recipe_id:
            raise ConfigError(f"recipe mismatch on {shard.filename}")
        total_positives += actual_positives
        if shard.positive_samples == 0:
            raise ConfigError(f"shard {shard.filename} has 0 positive samples")
        if shard.positive_samples == shard.num_samples:
            raise ConfigError(f"shard {shard.filename} has 0 negative samples")

    expected_pairs = {
        (split_name, robot_name) for split_name in manifest.splits for robot_name in manifest.robot_profile_hashes
    }
    if observed_pairs != expected_pairs:
        raise ConfigError(
            "split/robot shard coverage mismatch: "
            f"missing={sorted(expected_pairs - observed_pairs)}, "
            f"extra={sorted(observed_pairs - expected_pairs)}"
        )

    if total_positives == 0:
        raise ConfigError("total_positives is 0 across the entire dataset")

    repo_root = root.parent.parent
    try:
        in_repo = (
            subprocess.run(
                ["git", "rev-parse", "--is-inside-work-tree"],
                cwd=str(repo_root),
                capture_output=True,
                text=True,
                check=False,
            ).returncode
            == 0
        )
    except FileNotFoundError:
        in_repo = False
    for source_name, expected_hash in manifest.generator_source_hashes.items():
        source_relative = Path(source_name)
        if source_relative.is_absolute() or ".." in source_relative.parts:
            raise ConfigError(f"unsafe generator source path: {source_name}")
        source_path = repo_root / source_relative
        if not source_path.is_file():
            raise ConfigError(f"generator source missing: {source_name}")
        actual_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()
        if actual_hash != expected_hash:
            raise ConfigError(f"generator source drift: {source_name}")

    if in_repo:
        commit_check = subprocess.run(
            ["git", "cat-file", "-e", f"{manifest.generator_commit}^{{commit}}"],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            check=False,
        )
        if commit_check.returncode != 0:
            raise ConfigError(f"unknown generator commit: {manifest.generator_commit}")

    expected_release_relative = {Path("dataset_manifest.json")}
    expected_release_relative.update(Path(shard.filename) for shard in manifest.shards)
    for obj_id in train_objs | val_objs:
        expected_release_relative.add(Path("objects") / f"{obj_id}.manifest.json")
        expected_release_relative.add(Path("objects") / f"{obj_id}.obj")
    actual_release_relative = {Path("dataset_manifest.json")}
    actual_release_relative.update(path.relative_to(root) for path in (root / "shards").glob("*.pt"))
    actual_release_relative.update(
        path.relative_to(root) for pattern in ("*.manifest.json", "*.obj") for path in (root / "objects").glob(pattern)
    )
    stale = actual_release_relative - expected_release_relative
    missing = expected_release_relative - actual_release_relative
    if stale or missing:
        raise ConfigError(
            f"release file set mismatch: stale={sorted(map(str, stale))}, missing={sorted(map(str, missing))}"
        )

    # Check ignore and tracking status for every released artifact.
    release_files = [manifest_path]
    release_files.extend(root / shard.filename for shard in manifest.shards)
    release_files.extend(obj_dir / f"{obj_id}.manifest.json" for obj_id in train_objs | val_objs)
    release_files.extend(obj_dir / f"{obj_id}.obj" for obj_id in train_objs | val_objs)
    try:
        for release_file in release_files if in_repo else ():
            relative = release_file.relative_to(repo_root)
            res_ignore = subprocess.run(
                ["git", "check-ignore", "--no-index", "-q", str(relative)],
                cwd=str(repo_root),
                capture_output=True,
                text=True,
                check=False,
            )
            if res_ignore.returncode == 0:
                raise ConfigError(f"release artifact is ignored: {relative}")
            res_tracked = subprocess.run(
                ["git", "ls-files", "--error-unmatch", str(relative)],
                cwd=str(repo_root),
                capture_output=True,
                text=True,
                check=False,
            )
            if res_tracked.returncode != 0:
                raise ConfigError(f"release artifact is not tracked: {relative}")
    except FileNotFoundError:
        pass

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
    except Exception as exc:  # noqa: BLE001 - CLI gate reports one fail-closed verdict
        print(f"Audit FAIL: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
