"""Verified lazy loader for scene dataset shards."""

from __future__ import annotations

from pathlib import Path

from qdgrasp.config.schema import ConfigError
from qdgrasp.dataset.scene_manifest import SceneDatasetManifest, load_scene_manifest
from qdgrasp.dataset.scene_shards import SceneRecordType, read_scene_shard


def audit_scene_dataset(
    dataset_root: str | Path, *, manifest_file: str = "scene_manifest.json"
) -> dict[str, int]:
    """Verify all shards and positive cross-record references in a release."""
    root = Path(dataset_root).resolve()
    manifest = load_scene_manifest(root / manifest_file)
    state_hashes: dict[str, set[str]] = {}
    coverage: dict[str, set[str]] = {
        scene_id: set() for scenes in manifest.splits.values() for scene_id in scenes
    }
    positives: list[dict[str, object]] = []
    counts = {"scene_state": 0, "observation": 0, "grasp": 0, "positive_grasp": 0}
    split_by_scene = {
        scene_id: split for split, scenes in manifest.splits.items() for scene_id in scenes
    }
    for shard in manifest.shards:
        shard_path = (root / shard.filename).resolve()
        if not shard_path.is_relative_to(root):
            raise ConfigError(f"scene shard escapes dataset root: {shard.filename}")
        records = read_scene_shard(
            shard_path,
            record_type=shard.record_type,
            expected_sha256=shard.sha256,
            expected_records=shard.num_records,
        )
        counts[shard.record_type] += len(records)
        for record in records:
            scene_id = str(record["scene_id"])
            if split_by_scene.get(scene_id) != shard.split:
                raise ConfigError(
                    f"scene record split mismatch for {scene_id}: shard={shard.split}"
                )
            coverage[scene_id].add(shard.record_type)
            if shard.record_type == "scene_state":
                state_hashes.setdefault(scene_id, set()).add(str(record["state_hash"]))
            elif shard.record_type == "grasp" and bool(record.get("dynamic_valid")):
                positives.append(record)
                counts["positive_grasp"] += 1

    for record in positives:
        scene_id = str(record["scene_id"])
        missing = sorted(set(record["scene_state_hashes"].values()) - state_hashes.get(scene_id, set()))
        if missing:
            raise ConfigError(
                f"positive grasp references absent scene-state hashes for {scene_id}: {missing}"
            )
    if not manifest.release_blocked:
        required = {"scene_state", "observation", "grasp"}
        incomplete = {
            scene_id: sorted(required - record_types)
            for scene_id, record_types in coverage.items()
            if not required.issubset(record_types)
        }
        if incomplete:
            raise ConfigError(f"unblocked scene release has incomplete coverage: {incomplete}")
    return counts


class SceneDataset:
    def __init__(
        self,
        dataset_root: str | Path,
        *,
        split: str,
        record_type: SceneRecordType | None = None,
        manifest_file: str = "scene_manifest.json",
        allow_incomplete: bool = False,
    ) -> None:
        self.root = Path(dataset_root).resolve()
        self.manifest: SceneDatasetManifest = load_scene_manifest(self.root / manifest_file)
        if self.manifest.invalidated:
            raise ConfigError(
                f"scene dataset is invalidated: {self.manifest.invalidation_reason}"
            )
        if self.manifest.release_blocked and not allow_incomplete:
            raise ConfigError("scene dataset release is blocked")
        if split not in self.manifest.splits:
            raise ConfigError(f"unknown scene dataset split: {split}")
        allowed_scene_ids = set(self.manifest.splits[split])
        self.records: list[dict[str, object]] = []
        for shard in self.manifest.shards:
            if shard.split != split or (record_type is not None and shard.record_type != record_type):
                continue
            shard_path = (self.root / shard.filename).resolve()
            if not shard_path.is_relative_to(self.root):
                raise ConfigError(f"scene shard escapes dataset root: {shard.filename}")
            records = read_scene_shard(
                shard_path,
                record_type=shard.record_type,
                expected_sha256=shard.sha256,
                expected_records=shard.num_records,
            )
            unexpected = sorted(
                {str(record["scene_id"]) for record in records} - allowed_scene_ids
            )
            if unexpected:
                raise ConfigError(
                    f"scene shard contains IDs outside split {split}: {unexpected}"
                )
            self.records.extend(records)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, object]:
        return self.records[index]
