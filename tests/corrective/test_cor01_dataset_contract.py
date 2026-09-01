"""COR-01: three loaders, three contracts, and a facade that gates on none.

The historical audit, Phase 5 adapter and public loader each had their own idea
of what a valid sample was, so a corpus could pass one and reach training
through another.  They now share ``DatasetArtifact.open_verified``.

G1's target is one verified entry point, ``DatasetArtifact.open_verified()``,
used by the audit, the gate, the facade and the Runner alike.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from _corrective_support import characterization, manifest_document, sample, write_manifest, write_shard

from qdgrasp.config.schema import ConfigError


@characterization("COR-01", note="the public loader ignores release_blocked")
def test_the_public_loader_refuses_a_release_blocked_corpus(tmp_path: Path) -> None:
    """The audit refuses it; the loader the facade uses never asks."""

    root = tmp_path / "corpus"
    digest = write_shard(root / "shards" / "train.pt", [sample()])
    write_manifest(
        root,
        manifest_document(filename="shards/train.pt", sha256=digest, release_blocked=True),
    )

    from qdgrasp.dataset.loader import DgnOpenDataset

    with pytest.raises(ConfigError):
        DgnOpenDataset(dataset_root=root, split="train", point_count=64)


@characterization("COR-01", note="the public loader accepts a sample the Phase 5 adapter refuses")
def test_every_loader_refuses_the_same_incomplete_sample(tmp_path: Path) -> None:
    """A sample missing its joint targets must not be openable by either path."""

    incomplete = sample()
    del incomplete["joint_angles"]
    root = tmp_path / "corpus"
    digest = write_shard(root / "shards" / "train.pt", [incomplete])
    write_manifest(root, manifest_document(filename="shards/train.pt", sha256=digest))

    from qdgrasp.dataset import DatasetArtifact

    with pytest.raises(ConfigError):
        DatasetArtifact.open_verified(root)

    from qdgrasp.dataset.loader import DgnOpenDataset

    with pytest.raises(ConfigError):
        DgnOpenDataset(dataset_root=root, split="train", point_count=64)


@characterization("COR-01", note="there is no single verified entry point yet")
def test_a_verified_dataset_artifact_is_the_only_entry_point() -> None:
    """G1: one object opens a corpus, and it verifies before it returns one."""

    from qdgrasp import dataset as dataset_package

    assert hasattr(dataset_package, "DatasetArtifact"), (
        "PLAN.md §9.4 asks for DatasetArtifact.open_verified() as the single entry point for the audit, "
        "the gate, the facade and the Runner"
    )
    assert hasattr(dataset_package.DatasetArtifact, "open_verified")
