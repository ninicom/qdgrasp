"""COR-02: the protocol describes a split the trainer never sees.

The protocol document holds out the ``comp`` family and names an exact object
list per split.  The corpus on disk is split by a different rule, and the public
dataset opens the corpus rather than the protocol -- so a run can present itself
as trained under ``phase5-dgn-open-tiny-v1`` while having trained on objects the
protocol excludes.

The split's own claim is the second half.  ``create_object_family_splits``
promises "no family or shape leakage" and then stratifies *within* each shape,
which is the opposite operation: it guarantees every shape appears on both
sides.  Stratification is a legitimate split; it is just not a hold-out, and the
difference is the entire generalisation claim.
"""

from __future__ import annotations

import json
from pathlib import Path

from _corrective_support import characterization

REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET = REPO_ROOT / "datasets/dgn-open-tiny"
PROTOCOL = REPO_ROOT / "configs/phase5/protocol-v1.yaml"

ACTIVE_HANDS = ("leap_hand", "wonik_allegro")


def _object_manifest(object_id: str) -> dict:
    return json.loads((DATASET / "objects" / f"{object_id}.manifest.json").read_text(encoding="utf-8"))


def _make_object(object_id: str, family: str, shape_type: str):
    from qdgrasp.objects.schema import ObjectManifestSpec, SubGeomSpec

    return ObjectManifestSpec(
        object_id=object_id,
        family=family,
        shape_type=shape_type,
        params={},
        mesh_filename=f"{object_id}.obj",
        mesh_sha256="0" * 64,
        mass=0.1,
        inertia=(1e-4, 1e-4, 1e-4),
        bounding_box=(-0.01, -0.01, -0.01, 0.01, 0.01, 0.01),
        collision_geoms=(SubGeomSpec(type="box", size=(0.01, 0.01, 0.01)),),
    )


@characterization("COR-02", note="the physical train split carries objects the protocol excludes")
def test_the_train_split_holds_no_object_outside_the_protocol() -> None:
    from qdgrasp.models.data import FlowDataset
    from qdgrasp.models.protocol import load_protocol

    protocol = load_protocol(PROTOCOL)
    allowed = set(protocol.train_objects)
    leaked: dict[str, list[str]] = {}
    for hand in ACTIVE_HANDS:
        dataset = FlowDataset(DATASET, split="train", robot=hand)
        outside = sorted(set(dataset.object_ids) - allowed)
        if outside:
            leaked[hand] = outside

    assert not leaked, (
        f"objects outside the locked protocol reach the public train split: {leaked}. "
        "A run that trains on these is not a run under this protocol."
    )


@characterization("COR-02", note="the held-out family is present in the physical train split")
def test_the_held_out_family_never_appears_in_train() -> None:
    from qdgrasp.models.data import FlowDataset
    from qdgrasp.models.protocol import load_protocol

    protocol = load_protocol(PROTOCOL)
    present: dict[str, int] = {}
    for hand in ACTIVE_HANDS:
        dataset = FlowDataset(DATASET, split="train", robot=hand)
        count = sum(1 for item in dataset if str(item["object_id"]).startswith(f"{protocol.heldout_family}_"))
        if count:
            present[hand] = count

    assert not present, (
        f"held-out family {protocol.heldout_family!r} appears in the train split: {present} samples. "
        "The generalisation the protocol claims to measure was trained on."
    )


@characterization(
    "COR-02",
    note="stratification inside a shape is described as a family hold-out",
    satisfied_by="R3",
)
def test_the_splitter_does_not_call_stratification_a_family_hold_out() -> None:
    from qdgrasp.dataset.split import create_object_family_splits

    objects = [_make_object(f"comp_t_shape_{index:02d}", "compound", "t_shape") for index in range(1, 5)]
    objects += [_make_object(f"prim_box_{index:02d}", "primitive", "box") for index in range(1, 5)]
    families = {item.object_id: item.family for item in objects}

    splits = create_object_family_splits(objects, val_fraction=0.25, seed=0)
    train_families = {families[object_id] for object_id in splits["train"]}
    val_families = {families[object_id] for object_id in splits["val"]}

    assert not (train_families & val_families), (
        f"families {sorted(train_families & val_families)} appear on both sides of a split documented as "
        "having no family leakage; stratification is a valid split but it is not a hold-out"
    )


@characterization("COR-02", note="family is inferred from the object id prefix")
def test_family_is_read_from_the_object_manifest_not_from_the_id() -> None:
    """``prim_box_01`` is family ``primitive``; the prefix says ``prim``."""

    from qdgrasp.models.protocol import load_protocol

    protocol = load_protocol(PROTOCOL)
    disagreements = {}
    for object_id in list(protocol.train_objects) + list(protocol.val_objects):
        declared = _object_manifest(object_id)["family"]
        if protocol.family_of(object_id) != declared:
            disagreements[object_id] = (protocol.family_of(object_id), declared)

    assert not disagreements, (
        f"the protocol derives the family from the object id while the hashed object manifest declares "
        f"another: {disagreements}. Renaming an object would then move it between families."
    )


@characterization("COR-02", note="no materialised protocol view exists", satisfied_by="R3")
def test_a_protocol_view_is_materialised_before_the_runner() -> None:
    from qdgrasp.models import protocol as protocol_module

    assert hasattr(protocol_module, "ProtocolDatasetView"), (
        "PLAN.md §9.5 asks for a ProtocolDatasetView materialised on exactly (split, robot, object_id), "
        "carrying dataset_manifest_hash, protocol_hash and dataset_view_hash"
    )


@characterization("COR-02", note="the held-out hand is trainable through the public path")
def test_the_held_out_hand_has_no_train_samples() -> None:
    from qdgrasp.models.data import FlowDataset
    from qdgrasp.models.protocol import load_protocol

    protocol = load_protocol(PROTOCOL)
    test_hand = protocol.heldout_embodiment.test_hand
    dataset = FlowDataset(DATASET, split="train", robot=test_hand)

    assert len(dataset) == 0, (
        f"{test_hand} is the held-out embodiment yet its train split holds {len(dataset)} samples; "
        "count(train, held-out hand) must be 0 for a cross-embodiment claim to mean anything"
    )
