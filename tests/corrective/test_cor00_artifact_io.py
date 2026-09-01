"""COR-00: an untrusted artifact has an execution path into this process.

The chain the audit reproduced has two halves and both are load-bearing.  The
manifest decides which file is opened, and nothing constrains that decision to
the dataset root; the loader then hands the bytes to a pickle that may call
whatever it names.  Either half alone is a bug.  Together they are: read a
manifest, run code.

A SHA-256 in the same manifest does not close this.  It proves the bytes have
not changed since whoever wrote the manifest measured them, which is a statement
about drift, not about who wrote them.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from _corrective_support import (
    characterization,
    manifest_document,
    payload_with_reducer,
    sample,
    write_manifest,
    write_reducer_payload,
    write_shard,
)

from qdgrasp.config.schema import ConfigError


@characterization("COR-00", note="absolute path in a manifest entry")
def test_a_manifest_may_not_name_a_shard_outside_the_dataset_root(tmp_path: Path) -> None:
    """``root / "/etc/x"`` is ``/etc/x``: an absolute entry silently wins."""

    outside = tmp_path / "outside" / "shard.pt"
    digest = write_shard(outside, [sample()])
    root = tmp_path / "corpus"
    write_manifest(root, manifest_document(filename=str(outside), sha256=digest))

    from qdgrasp.dataset.loader import DgnOpenDataset

    with pytest.raises(ConfigError):
        DgnOpenDataset(dataset_root=root, split="train", point_count=64)


@characterization("COR-00", note="parent traversal in a manifest entry")
def test_a_manifest_may_not_traverse_out_of_the_dataset_root(tmp_path: Path) -> None:
    outside = tmp_path / "outside" / "shard.pt"
    digest = write_shard(outside, [sample()])
    root = tmp_path / "corpus"
    write_manifest(root, manifest_document(filename="../outside/shard.pt", sha256=digest))

    from qdgrasp.dataset.loader import DgnOpenDataset

    with pytest.raises(ConfigError):
        DgnOpenDataset(dataset_root=root, split="train", point_count=64)


@characterization("COR-00", note="string-prefix containment accepts a sibling directory")
def test_containment_is_by_path_ancestry_not_by_string_prefix(tmp_path: Path) -> None:
    """``/x/data-evil`` starts with ``/x/data`` and is not inside it."""

    from qdgrasp.dataset.contactrich_active import DatasetRejected, _safe_path

    root = tmp_path / "data"
    root.mkdir()
    (tmp_path / "data-evil").mkdir()

    with pytest.raises(DatasetRejected):
        _safe_path(root, "../data-evil/shard.pt")


@characterization("COR-00", note="shard load reaches torch.load(weights_only=False)")
def test_loading_a_shard_does_not_execute_a_pickle_reducer(tmp_path: Path) -> None:
    marker = tmp_path / "reducer-ran"
    shard = write_reducer_payload(tmp_path / "shard.pt", marker)

    from qdgrasp.dataset.shards import read_shard_file

    refusal: Exception | None = None
    try:
        read_shard_file(shard)
    except Exception as error:  # noqa: BLE001 - any refusal is acceptable, silence is not
        refusal = error

    assert not marker.exists(), "loading a dataset shard executed code named by the shard"
    assert refusal is not None, "a shard carrying a pickle reducer must be refused, not loaded quietly"


@characterization("COR-00", note="MVP checkpoint load reaches torch.load(weights_only=False)")
def test_loading_an_mvp_checkpoint_does_not_execute_a_pickle_reducer(tmp_path: Path) -> None:
    from qdgrasp.mvp.policy import POLICY_SCHEMA_V0, load_checkpoint

    marker = tmp_path / "reducer-ran"
    payload = payload_with_reducer(
        {
            "schema": POLICY_SCHEMA_V0,
            "stage": "characterization",
            "fingerprint": {},
            "architecture": {"observation_dim": 4, "action_dim": 2, "hidden": [8]},
            "normalizer": {"dimension": 4, "count": 1.0, "mean": [0.0] * 4, "m2": [1.0] * 4},
            "state_dict": {},
            "optimizer_state": None,
            "reload_probe": {},
        },
        marker,
    )
    path = tmp_path / "checkpoint.pt"
    torch.save(payload, path)

    refusal: Exception | None = None
    try:
        load_checkpoint(path)
    except Exception as error:  # noqa: BLE001
        refusal = error

    assert not marker.exists(), "loading a policy checkpoint executed code named by the checkpoint"
    assert refusal is not None, "a checkpoint carrying a pickle reducer must be refused, not loaded quietly"


@characterization("COR-00", note="the Phase 5 input gate reads shards with verify=False")
def test_the_phase5_input_gate_verifies_shard_bytes() -> None:
    """A gate that skips verification cannot report on the data it measured."""

    import inspect

    from qdgrasp.corrective.gate import _load_script

    module = _load_script("check_phase5_inputs.py")
    assert module is not None
    source = inspect.getsource(module.measure)
    assert "verify=False" not in source, (
        "the positive gate reads the corpus without checking it against its own manifest, so its counts "
        "describe whatever bytes are on disk rather than the dataset the manifest names"
    )
