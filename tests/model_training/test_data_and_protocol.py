"""P5-01/02: the adapter's refusals and the protocol's leakage rules.

Every test here is a way a training run could look valid while measuring
something else: a shard that drifted from its manifest, a paused hand entering
by default, a batch that silently evaluates two hands against one graph, an
object sitting in both sides of a split, a protocol edited after the fact.
"""

from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path

import pytest
import torch
import yaml

from qdgrasp.config.active_scope import ACTIVE_HANDS
from qdgrasp.config.schema import ConfigError
from qdgrasp.dataset import DatasetArtifact
from qdgrasp.dataset.loader import DgnOpenDataset
from qdgrasp.engine.sampling import BatchIdentityError, collate_samples, iterate_batches
from qdgrasp.models.protocol import (
    ABLATION_REGISTRY,
    SELECTION_REGISTRY,
    ProtocolError,
    check_dataset_agreement,
    load_protocol,
    parse_protocol,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET = REPO_ROOT / "datasets/dgn-open-tiny"
PROTOCOL = REPO_ROOT / "configs/phase5/protocol-v2.yaml"


@pytest.fixture(scope="module")
def artifact() -> DatasetArtifact:
    return DatasetArtifact.open_verified(DATASET)


@pytest.fixture(scope="module")
def manifest(artifact: DatasetArtifact) -> dict:
    return artifact.manifest.model_dump(by_alias=True)


@pytest.fixture(scope="module")
def protocol_document() -> dict:
    return yaml.safe_load(PROTOCOL.read_text(encoding="utf-8"))


# -- P5-01 adapter ---------------------------------------------------------


def test_the_shipped_shards_match_their_manifest_hashes(artifact: DatasetArtifact) -> None:
    """The dataset and its manifest must describe the same bytes."""

    assert artifact.shards
    for shard in artifact.shards:
        assert len(shard.samples) == shard.metadata.num_samples


def _copy_verified_corpus(destination_project: Path) -> Path:
    destination = destination_project / "datasets" / DATASET.name
    shutil.copytree(DATASET, destination)
    manifest = json.loads((destination / "dataset_manifest.json").read_text(encoding="utf-8"))
    for source_name in manifest["generator_source_hashes"]:
        target = destination_project / source_name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO_ROOT / source_name, target)
    return destination


def test_a_shard_that_drifted_from_the_manifest_is_refused(tmp_path: Path) -> None:
    root = _copy_verified_corpus(tmp_path)
    manifest = json.loads((root / "dataset_manifest.json").read_text(encoding="utf-8"))
    shard = next(item for item in manifest["shards"] if item["split"] == "val")
    torch.save([], root / shard["filename"])
    with pytest.raises(ConfigError, match="integrity mismatch"):
        DatasetArtifact.open_verified(root)


def test_a_paused_hand_is_not_loaded_by_a_default_workload(manifest) -> None:
    """ADR-0008: an active release contains no Shadow shard."""

    assert not any(entry["robot_name"] == "shadow_hand" for entry in manifest["shards"])
    assert set(manifest["robot_profile_hashes"]) == set(ACTIVE_HANDS)
    with pytest.raises(ConfigError, match="allowlist"):
        DgnOpenDataset(DATASET, split="train", robot_name="shadow_hand")


def test_an_invalidated_dataset_is_refused(manifest, tmp_path: Path) -> None:
    broken = copy.deepcopy(manifest)
    broken["invalidated"] = True
    broken["invalidation_reason"] = "regenerated under a different certifier"
    (tmp_path / "dataset_manifest.json").write_text(json.dumps(broken), encoding="utf-8")
    with pytest.raises(ConfigError, match="invalidated"):
        DatasetArtifact.open_verified(tmp_path)


def test_a_batch_may_not_mix_hands() -> None:
    leap = DgnOpenDataset(DATASET, split="val", robot_name="leap_hand")
    allegro = DgnOpenDataset(DATASET, split="val", robot_name="wonik_allegro")
    with pytest.raises(BatchIdentityError, match="may not mix robots"):
        collate_samples([leap[0], allegro[0]])


def test_collate_produces_exactly_what_the_loss_consumes() -> None:
    dataset = DgnOpenDataset(DATASET, split="val", robot_name="leap_hand")
    batch = collate_samples([dataset[index] for index in range(4)])
    assert {
        "points",
        "point_mask",
        "palm_pos",
        "palm_rot",
        "joint_angles",
        "fingertip_positions",
        "success",
        "kinematics_valid",
        "pose_target_valid",
        "joint_target_valid",
        "fk_target_valid",
        "robot_name",
        "robot_profile_hash",
        "joint_names",
    } <= set(batch)
    assert batch["points"].shape[0] == 4
    assert batch["success"].shape == (4,)
    for field in ("points", "palm_pos", "palm_rot", "joint_angles", "fingertip_positions", "success"):
        assert batch[field].dtype == torch.float32
    for field in ("point_mask", "kinematics_valid", "pose_target_valid", "joint_target_valid", "fk_target_valid"):
        assert batch[field].dtype == torch.bool


def test_batches_cover_every_sample_exactly_once() -> None:
    dataset = DgnOpenDataset(DATASET, split="val", robot_name="leap_hand")
    seen = sum(batch["points"].shape[0] for batch in iterate_batches(dataset, 5))
    assert seen == len(dataset)


def test_an_empty_batch_is_refused() -> None:
    with pytest.raises(BatchIdentityError, match="empty batch"):
        collate_samples([])


def test_the_dataset_reports_how_few_positives_it_has() -> None:
    """The number that decides whether any of P5 is meaningful."""

    datasets = {robot: DgnOpenDataset(DATASET, split="train", robot_name=robot) for robot in ACTIVE_HANDS}
    fractions = {
        robot: sum(float(sample["success"]) for sample in dataset.samples) / len(dataset)
        for robot, dataset in datasets.items()
    }
    assert set(fractions) == set(ACTIVE_HANDS)
    # This asserts the corpus is what it is, not that it is good enough: every
    # active hand is far below anything trainable, and check_phase5_inputs.py
    # turns that into a gate.
    assert all(0.0 < value < 0.1 for value in fractions.values()), fractions


# -- P5-02 protocol --------------------------------------------------------


def _bind_family(document: dict, object_id: str, family: str) -> dict:
    """Add an object's declared family, as ``qdgrasp/protocol/v2`` requires."""

    document["object_families"][object_id] = family
    return document


def _unbind_family(document: dict, object_id: str) -> dict:
    document["object_families"].pop(object_id, None)
    return document


def test_the_shipped_protocol_is_valid_and_matches_the_dataset(manifest) -> None:
    protocol = load_protocol(PROTOCOL)
    check_dataset_agreement(protocol, manifest)
    assert protocol.selection in SELECTION_REGISTRY
    assert "baseline" in protocol.ablations
    assert len(protocol.protocol_hash) == 64


def test_the_hash_is_stable_over_key_order_and_sensitive_to_values(protocol_document) -> None:
    base = parse_protocol(protocol_document)
    reordered = parse_protocol(dict(reversed(list(protocol_document.items()))))
    assert base.protocol_hash == reordered.protocol_hash

    changed = copy.deepcopy(protocol_document)
    changed["seeds"] = [0, 1, 2, 3]
    assert parse_protocol(changed).protocol_hash != base.protocol_hash


def test_an_object_in_both_splits_is_leakage(protocol_document) -> None:
    document = copy.deepcopy(protocol_document)
    document["splits"]["train_objects"].append(document["splits"]["val_objects"][1])
    with pytest.raises(ProtocolError, match="leakage"):
        parse_protocol(document)


def test_a_held_out_family_with_a_member_in_train_is_refused(protocol_document) -> None:
    # A member the protocol does not already place, so the refusal is about the
    # family rather than about the object being on both sides.
    document = copy.deepcopy(protocol_document)
    document["splits"]["train_objects"].append("comp_extra_01")
    _bind_family(document, "comp_extra_01", "compound")
    with pytest.raises(ProtocolError, match="held-out family"):
        parse_protocol(document)


def test_a_held_out_family_nothing_measures_is_refused(protocol_document) -> None:
    document = copy.deepcopy(protocol_document)
    held_out = document["splits"]["heldout_family"]
    # Val holds only the held-out family, so something else has to stay behind:
    # an empty val is refused for its own reason and would hide this one.
    document["splits"]["val_objects"].append("sq_extra_01")
    _bind_family(document, "sq_extra_01", "superquadric")

    dropped = [
        object_id
        for object_id in document["splits"]["val_objects"]
        if document["object_families"][object_id] == held_out
    ]
    document["splits"]["val_objects"] = [
        object_id for object_id in document["splits"]["val_objects"] if object_id not in dropped
    ]
    for object_id in dropped:
        _unbind_family(document, object_id)
    with pytest.raises(ProtocolError, match="no member in val"):
        parse_protocol(document)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda d: d["splits"]["heldout_embodiment"].update(test_hand="shadow_hand"), "not in the active corpus"),
        (lambda d: d["splits"]["heldout_embodiment"].update(test_hand="leap_hand"), "different hands"),
        (lambda d: d.update(seeds=[]), "at least one seed"),
        (lambda d: d.update(seeds=[0, 0, 1]), "repeat"),
        (lambda d: d.update(ablations=["baseline", "made_up"]), "unknown ablation"),
        (lambda d: d.update(ablations=["no_graph"]), "must include 'baseline'"),
        (lambda d: d.update(metrics=["success", "vibes"]), "unknown metric"),
        (lambda d: d.update(selection="total_loss"), "unknown selection rule"),
        (lambda d: d.update(schema="qdgrasp/protocol/v1"), "unsupported protocol schema"),
        (lambda d: d["object_families"].pop("sq_04"), "must exactly cover"),
    ],
)
def test_a_protocol_that_cannot_be_measured_is_refused(protocol_document, mutate, message) -> None:
    document = copy.deepcopy(protocol_document)
    mutate(document)
    with pytest.raises(ProtocolError, match=message):
        parse_protocol(document)


def test_total_loss_is_not_a_selectable_rule() -> None:
    """P4 measured why: the flow term's floor makes the total a poor signal."""

    assert "total_loss" not in SELECTION_REGISTRY
    assert "baseline" in ABLATION_REGISTRY


def test_a_protocol_naming_objects_the_dataset_lacks_is_refused(manifest, protocol_document) -> None:
    document = copy.deepcopy(protocol_document)
    document["splits"]["val_objects"].append("prim_torus_99")
    _bind_family(document, "prim_torus_99", "primitive")
    with pytest.raises(ProtocolError, match="does not contain"):
        check_dataset_agreement(parse_protocol(document), manifest)
