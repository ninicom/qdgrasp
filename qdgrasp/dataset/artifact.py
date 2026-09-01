"""One verified entry point for DGN dataset artifacts.

``DatasetArtifact.open_verified`` validates the manifest, its provenance, every
referenced file and every training sample before returning anything a loader can
iterate.  Audit scripts and runtime loaders therefore cannot disagree about
what a valid corpus means.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import torch

from ..config.loader import load_robot_config
from ..config.schema import ConfigError
from ..objects.schema import ObjectManifestSpec
from .artifact_io import resolve_contained_regular_file, validate_relative_artifact_path
from .manifest import DATASET_MANIFEST_SCHEMA_V2, DatasetManifestSpec, ShardMetadata, load_dataset_manifest
from .shards import read_shard_file

_TENSOR_FIELDS = (
    "points",
    "palm_pos",
    "palm_rot",
    "joint_angles",
    "fingertip_positions",
    "success",
    "quality",
)
_VALIDITY_FIELDS = (
    "proposal_valid",
    "ik_valid",
    "collision_valid",
    "static_force_valid",
    "dynamic_valid",
)
_PROVENANCE_FIELDS = (
    "frame",
    "recipe_id",
    "proposal_module",
    "solver_module",
    "certifier_version",
    "dynamic_protocol_version",
    "success_schema_version",
    "failure_stage",
    "failure_reason",
)
_REQUIRED_SAMPLE_FIELDS = frozenset(
    (*_TENSOR_FIELDS, *_VALIDITY_FIELDS, *_PROVENANCE_FIELDS, "object_id", "robot_name")
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_hash(path: Path, expected: str, *, label: str) -> None:
    actual = _sha256(path)
    if actual != expected:
        raise ConfigError(f"{label} integrity mismatch: expected {expected}, got {actual}")


@dataclass(frozen=True)
class RobotSampleContract:
    """The ordered target dimensions bound by one robot profile hash."""

    profile_hash: str
    joint_names: tuple[str, ...]
    fingertip_links: tuple[str, ...]


@dataclass(frozen=True)
class VerifiedShard:
    """A shard whose bytes and samples satisfy its manifest entry."""

    metadata: ShardMetadata
    path: Path
    samples: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class DatasetArtifact:
    """A fail-closed dataset release ready for runtime consumption."""

    root: Path
    manifest_path: Path
    manifest: DatasetManifestSpec
    manifest_hash: str
    shards: tuple[VerifiedShard, ...]
    robot_contracts: Mapping[str, RobotSampleContract]

    @classmethod
    def open_verified(
        cls,
        root: str | Path,
        *,
        manifest_file: str = "dataset_manifest.json",
    ) -> "DatasetArtifact":
        """Open and fully verify a release artifact before exposing samples."""

        try:
            resolved_root = Path(root).resolve(strict=True)
        except OSError as exc:
            raise ConfigError(f"dataset root does not exist: {root}") from exc
        if not resolved_root.is_dir():
            raise ConfigError(f"dataset root is not a directory: {resolved_root}")

        manifest_path = resolve_contained_regular_file(resolved_root, manifest_file)
        manifest = load_dataset_manifest(manifest_path)
        cls._verify_release_contract(resolved_root, manifest)
        robot_contracts = cls._verify_robot_profiles(manifest)
        cls._verify_objects(resolved_root, manifest)
        cls._verify_generator_sources(resolved_root, manifest)

        verified_shards: list[VerifiedShard] = []
        for metadata in manifest.shards:
            path = resolve_contained_regular_file(resolved_root, metadata.filename)
            samples = read_shard_file(path, expected_sha256=metadata.sha256)
            if len(samples) != metadata.num_samples:
                raise ConfigError(
                    f"sample count mismatch on {metadata.filename}: "
                    f"manifest={metadata.num_samples}, actual={len(samples)}"
                )
            contract = robot_contracts[metadata.robot_name]
            positives = 0
            validated: list[dict[str, Any]] = []
            for index, sample in enumerate(samples):
                cls._verify_sample(sample, index=index, shard=metadata, manifest=manifest, robot=contract)
                positives += int(float(sample["success"]) > 0.5)
                validated.append(sample)
            if positives != metadata.positive_samples:
                raise ConfigError(
                    f"positive count mismatch on {metadata.filename}: "
                    f"manifest={metadata.positive_samples}, actual={positives}"
                )
            verified_shards.append(VerifiedShard(metadata, path, tuple(validated)))

        expected_pairs = {
            (split_name, robot_name)
            for split_name in manifest.splits
            for robot_name in manifest.robot_profile_hashes
        }
        observed_pairs = {(shard.metadata.split, shard.metadata.robot_name) for shard in verified_shards}
        if observed_pairs != expected_pairs:
            raise ConfigError(
                "split/robot shard coverage mismatch: "
                f"missing={sorted(expected_pairs - observed_pairs)}, "
                f"extra={sorted(observed_pairs - expected_pairs)}"
            )

        return cls(
            root=resolved_root,
            manifest_path=manifest_path,
            manifest=manifest,
            manifest_hash=_sha256(manifest_path),
            shards=tuple(verified_shards),
            robot_contracts=MappingProxyType(robot_contracts),
        )

    @staticmethod
    def _verify_release_contract(root: Path, manifest: DatasetManifestSpec) -> None:
        if manifest.schema_version != DATASET_MANIFEST_SCHEMA_V2:
            raise ConfigError(
                f"unsupported dataset schema {manifest.schema_version!r}; expected {DATASET_MANIFEST_SCHEMA_V2!r}"
            )
        if manifest.release_blocked:
            raise ConfigError("dataset manifest has release_blocked=True")
        if manifest.invalidated:
            reason = manifest.invalidation_reason or "no reason recorded"
            raise ConfigError(f"dataset manifest is invalidated: {reason}")
        if manifest.generator_worktree_dirty:
            raise ConfigError("dataset was generated from a dirty worktree")
        if manifest.generator_commit == "legacy":
            raise ConfigError("dataset lacks a recorded generator commit")
        if manifest.license != "CC0-1.0":
            raise ConfigError(f"unauthorized dataset license {manifest.license!r}")
        if any(
            value == "legacy"
            for value in (
                manifest.recipe_id,
                manifest.proposal_module,
                manifest.solver_module,
                manifest.certifier_version,
                manifest.dynamic_protocol_version,
            )
        ):
            raise ConfigError("dataset lacks generator recipe/module provenance")
        if not manifest.generator_source_hashes:
            raise ConfigError("dataset lacks generator source hashes")
        if not manifest.object_manifest_hashes:
            raise ConfigError("dataset lacks object manifest hashes")

        train = set(manifest.splits.get("train", ()))
        val = set(manifest.splits.get("val", ()))
        if not train or not val:
            raise ConfigError("dataset release needs non-empty train and val splits")
        overlap = sorted(train & val)
        if overlap:
            raise ConfigError(f"split leakage between train and val: {overlap}")
        if manifest.success_criteria.get("min_contacts", 0.0) < 2.0:
            raise ConfigError("success criteria permit fewer than two fingers")
        if manifest.success_criteria.get("max_penetration", float("inf")) > 0.002:
            raise ConfigError("success criteria permit excessive penetration")
        if not root.is_dir():  # Defensive: keeps this method safe when reused directly.
            raise ConfigError(f"dataset root is not a directory: {root}")

    @staticmethod
    def _verify_robot_profiles(manifest: DatasetManifestSpec) -> dict[str, RobotSampleContract]:
        contracts: dict[str, RobotSampleContract] = {}
        for robot_name, expected_hash in manifest.robot_profile_hashes.items():
            if not robot_name or robot_name in {".", ".."} or "/" in robot_name or "\\" in robot_name:
                raise ConfigError(f"unsafe robot name {robot_name!r}")
            profile = load_robot_config(f"robots/{robot_name}.yaml")
            actual_hash = profile.content_hash()
            if actual_hash != expected_hash:
                raise ConfigError(
                    f"robot profile provenance mismatch for {robot_name}: "
                    f"expected {expected_hash}, got {actual_hash}"
                )
            contracts[robot_name] = RobotSampleContract(
                profile_hash=actual_hash,
                joint_names=tuple(profile.joints),
                fingertip_links=tuple(getattr(profile, "fingertip_links", ())),
            )
        if not contracts:
            raise ConfigError("dataset declares no robot profiles")
        return contracts

    @staticmethod
    def _verify_objects(root: Path, manifest: DatasetManifestSpec) -> None:
        object_ids = set().union(*(set(ids) for ids in manifest.splits.values()))
        declared = set(manifest.object_manifest_hashes)
        if declared != object_ids:
            raise ConfigError(
                "object manifest coverage mismatch: "
                f"missing={sorted(object_ids - declared)}, extra={sorted(declared - object_ids)}"
            )
        for object_id in sorted(object_ids):
            relative = validate_relative_artifact_path(f"objects/{object_id}.manifest.json")
            path = resolve_contained_regular_file(root, relative)
            _require_hash(path, manifest.object_manifest_hashes[object_id], label=f"object manifest {object_id}")
            try:
                document = json.loads(path.read_text(encoding="utf-8"))
                object_manifest = ObjectManifestSpec.model_validate(document)
            except Exception as exc:
                raise ConfigError(f"invalid object manifest {object_id}: {exc}") from exc
            if object_manifest.schema_version != "qdgrasp/object-manifest/v1":
                raise ConfigError(f"unsupported object manifest schema for {object_id}")
            if object_manifest.object_id != object_id:
                raise ConfigError(f"object manifest identity mismatch for {object_id}")
            mesh_relative = validate_relative_artifact_path(f"objects/{object_manifest.mesh_filename}")
            mesh_path = resolve_contained_regular_file(root, mesh_relative)
            _require_hash(mesh_path, object_manifest.mesh_sha256, label=f"object mesh {object_id}")

    @staticmethod
    def _verify_generator_sources(root: Path, manifest: DatasetManifestSpec) -> None:
        # Dataset releases live at <project>/datasets/<dataset>.  This convention
        # is part of v2 because the manifest records project-relative source paths.
        project_root = root.parent.parent.resolve()
        for source_name, expected_hash in manifest.generator_source_hashes.items():
            source_path = resolve_contained_regular_file(project_root, source_name)
            _require_hash(source_path, expected_hash, label=f"generator source {source_name}")

    @staticmethod
    def _verify_sample(
        sample: Any,
        *,
        index: int,
        shard: ShardMetadata,
        manifest: DatasetManifestSpec,
        robot: RobotSampleContract,
    ) -> None:
        label = f"{shard.filename}[{index}]"
        if not isinstance(sample, dict):
            raise ConfigError(f"{label} is not a sample mapping")
        missing = sorted(_REQUIRED_SAMPLE_FIELDS - set(sample))
        if missing:
            raise ConfigError(f"{label} lacks required sample fields {missing}")

        for field in _TENSOR_FIELDS:
            value = sample[field]
            if not isinstance(value, torch.Tensor):
                raise ConfigError(f"{label}.{field} must be a torch.Tensor")
            if value.dtype != torch.float32:
                raise ConfigError(f"{label}.{field} must use torch.float32, got {value.dtype}")
            if not bool(torch.isfinite(value).all()):
                raise ConfigError(f"{label}.{field} contains NaN or Inf")

        expected_shapes = {
            "palm_pos": (3,),
            "palm_rot": (3, 3),
            "joint_angles": (len(robot.joint_names),),
            "fingertip_positions": (len(robot.fingertip_links), 3),
            "success": (),
            "quality": (),
        }
        points = sample["points"]
        if points.ndim != 2 or points.shape[0] < 1 or points.shape[1] != 3:
            raise ConfigError(f"{label}.points must have shape [N, 3] with N >= 1")
        for field, shape in expected_shapes.items():
            if tuple(sample[field].shape) != shape:
                raise ConfigError(
                    f"{label}.{field} has shape {tuple(sample[field].shape)}, expected {shape}"
                )

        robot_name = sample["robot_name"]
        object_id = sample["object_id"]
        if not isinstance(robot_name, str) or robot_name != shard.robot_name:
            raise ConfigError(f"{label} robot identity does not match shard {shard.robot_name!r}")
        if not isinstance(object_id, str) or object_id not in manifest.splits[shard.split]:
            raise ConfigError(f"{label} object {object_id!r} is not in split {shard.split!r}")

        for field in _VALIDITY_FIELDS:
            if not isinstance(sample[field], bool):
                raise ConfigError(f"{label}.{field} must be bool")
        flags = [bool(sample[field]) for field in _VALIDITY_FIELDS]
        if any(flags[position] and not flags[position - 1] for position in range(1, len(flags))):
            raise ConfigError(f"{label} has non-monotonic pipeline validity flags")

        success = float(sample["success"])
        if success not in {0.0, 1.0}:
            raise ConfigError(f"{label}.success must be binary, got {success}")
        if bool(success) != bool(sample["dynamic_valid"]):
            raise ConfigError(f"{label} has success/dynamic_valid mismatch")
        if float(sample["quality"]) < 0.0:
            raise ConfigError(f"{label}.quality must be non-negative")

        expected_provenance = {
            "frame": "object",
            "recipe_id": manifest.recipe_id,
            "proposal_module": manifest.proposal_module,
            "solver_module": manifest.solver_module,
            "certifier_version": manifest.certifier_version,
            "dynamic_protocol_version": manifest.dynamic_protocol_version,
            "success_schema_version": "dynamic-only-v1",
        }
        for field, expected in expected_provenance.items():
            if sample[field] != expected:
                raise ConfigError(f"{label}.{field}={sample[field]!r}, expected {expected!r}")
        if shard.recipe_id != manifest.recipe_id:
            raise ConfigError(f"{shard.filename} recipe does not match dataset manifest")
        if not isinstance(sample["failure_stage"], str) or not isinstance(sample["failure_reason"], str):
            raise ConfigError(f"{label} failure metadata must be strings")

    def samples(
        self,
        *,
        split: str,
        robot_name: str | None = None,
    ) -> tuple[dict[str, Any], ...]:
        """Return verified samples for one split/optional robot in manifest order."""

        if split not in self.manifest.splits:
            raise ConfigError(f"unknown dataset split {split!r}; available={sorted(self.manifest.splits)}")
        selected: list[dict[str, Any]] = []
        for shard in self.shards:
            if shard.metadata.split != split:
                continue
            if robot_name is not None and shard.metadata.robot_name != robot_name:
                continue
            selected.extend(shard.samples)
        if robot_name is not None and robot_name not in self.robot_contracts:
            raise ConfigError(f"dataset does not declare robot {robot_name!r}")
        return tuple(selected)

    def sample_count(self, *, split: str, robot_name: str | None = None) -> int:
        """Count samples without exposing a second manifest interpretation."""

        return len(self.samples(split=split, robot_name=robot_name))


__all__ = ("DatasetArtifact", "RobotSampleContract", "VerifiedShard")
