"""COR-09: a bundle is accepted on shapes and rebuilt on a guessed schema.

``load_public_bundle`` checks the robot hash and then whether the tensors fit.
Everything that decides what those tensors *mean* -- the model configuration and
the declared preprocessing -- is carried in the manifest and never compared, so
a bundle can be loaded into a model that reads its inputs differently and the
result will look like a normal prediction.

``QDGrasp.from_bundle`` then parses the embedded robot profile as ``robot/v1``
regardless of what it says.  Both active hands ship as ``robot/v2``, so the one
path that is supposed to rebuild a released model from its own bundle cannot
rebuild a bundle either active hand produced.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from _corrective_support import characterization, refuses

from qdgrasp.config import ModelConfig, parse_document

REPO_ROOT = Path(__file__).resolve().parents[2]


@characterization("COR-09", note="the declared preprocessing is not part of the load gate")
def test_a_bundle_with_different_preprocessing_is_refused(tmp_path: Path) -> None:
    from qdgrasp.api import QDGrasp
    from qdgrasp.engine.checkpoint import MANIFEST_FILE, _canonical_hash

    grasper = QDGrasp()
    info = grasper.save_bundle(tmp_path / "bundle")

    manifest = json.loads((info.directory / MANIFEST_FILE).read_text(encoding="utf-8"))
    manifest["preprocess"]["units"] = "millimeters"
    hashes = dict(manifest["hashes"])
    hashes.pop("bundle")
    probe = dict(manifest)
    probe["hashes"] = hashes
    manifest["hashes"]["bundle"] = _canonical_hash(probe)
    (info.directory / MANIFEST_FILE).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    refuses(
        lambda: QDGrasp().load_weights(info.directory),
        because=(
            "a bundle declaring millimetre inputs loaded into a model that reads metres; the declared "
            "preprocessing is carried in the manifest and never compared"
        ),
    )


@characterization("COR-09", note="from_bundle parses the embedded profile as robot/v1")
def test_a_bundle_round_trips_the_robot_schema_it_was_written_with(tmp_path: Path) -> None:
    from qdgrasp.api import QDGrasp

    grasper = QDGrasp("qdgrasp-dummy-n.yaml", robot="leap_hand.yaml")
    assert grasper.robot_config.schema_version == "qdgrasp/robot/v2"
    info = grasper.save_bundle(tmp_path / "bundle")

    rebuilt = QDGrasp.from_bundle(info.directory)
    assert rebuilt.robot_config.content_hash() == grasper.robot_config.content_hash()


@characterization("COR-09", note="a setting that does not change a shape is not compared")
def test_a_bundle_whose_flow_steps_differ_is_refused(tmp_path: Path) -> None:
    """``flow_steps`` changes what the weights mean and no tensor's shape."""

    from qdgrasp.api import QDGrasp

    trained = QDGrasp("qdgrasp-flow-n.yaml", robot="leap_hand.yaml")
    info = trained.save_bundle(tmp_path / "bundle")

    document = trained.model_config.to_document()
    document["params"]["flow_steps"] = trained.model_config.params["flow_steps"] + 1
    rebuilt = QDGrasp("qdgrasp-flow-n.yaml", robot="leap_hand.yaml")
    rebuilt.model_config = parse_document(document, ModelConfig, origin="test")

    refuses(
        lambda: rebuilt.load_weights(info.directory),
        because=(
            "a bundle integrated over five Euler steps loaded into a model that will take six; every tensor "
            "fits and the model is a different one"
        ),
    )


@characterization("COR-09", note="architecture semantics were implied by the installed code")
def test_a_bundle_produced_under_other_architecture_semantics_is_refused(tmp_path: Path) -> None:
    """``PLAN.md`` §9.6 asks for the joint parameterization to be written down.

    A configuration says how wide the model is, not what its numbers mean.  The
    same document under the old ``tanh``-clamped joints and the new ``atanh``
    latent produces the same tensor shapes and a different hand pose, so the
    bundle records the semantics and the loader compares them.
    """

    from qdgrasp.api import QDGrasp
    from qdgrasp.engine.checkpoint import MANIFEST_FILE, _canonical_hash

    trained = QDGrasp("qdgrasp-flow-n.yaml", robot="leap_hand.yaml")
    info = trained.save_bundle(tmp_path / "bundle")
    assert info.manifest["semantics"]["joint_parameterization"] == "atanh-normalized-limits/v1"

    manifest = json.loads((info.directory / MANIFEST_FILE).read_text(encoding="utf-8"))
    manifest["semantics"]["joint_parameterization"] = "tanh-clamped-limits/v0"
    hashes = dict(manifest["hashes"])
    hashes["semantics"] = _canonical_hash(manifest["semantics"])
    hashes.pop("bundle")
    probe = dict(manifest)
    probe["hashes"] = hashes
    hashes["bundle"] = _canonical_hash(probe)
    manifest["hashes"] = hashes
    (info.directory / MANIFEST_FILE).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    refuses(
        lambda: QDGrasp("qdgrasp-flow-n.yaml", robot="leap_hand.yaml").load_weights(info.directory),
        because=(
            "weights produced under the superseded joint parameterization loaded into a model that reads "
            "the latent differently; every tensor fits and the joints they describe do not"
        ),
    )


@characterization("COR-09", note="one robot hash serves both roles")
def test_a_bundle_separates_the_training_hand_from_the_runtime_hand(tmp_path: Path) -> None:
    """A stored bundle names the hand it was trained on; a transfer names both.

    One ``robot_hash`` cannot answer both questions, and an exact-match gate on
    it is wrong in both directions: it forbids the cross-embodiment inference the
    protocol exists to measure, and it lets an artifact made for one hand be
    reported under another.
    """

    from qdgrasp.api import QDGrasp
    from qdgrasp.engine.compatibility import CompatibilityError
    from qdgrasp.models.protocol import load_protocol

    leap = QDGrasp("qdgrasp-dummy-n.yaml", robot="leap_hand.yaml")
    info = leap.save_bundle(tmp_path / "bundle")
    assert info.manifest["hashes"]["training_robot_config"] == leap.robot_config.content_hash()
    assert "robot_config" not in info.manifest["hashes"], "the ambiguous single hash must be gone"

    protocol = load_protocol(REPO_ROOT / "configs" / "phase5" / "protocol-v2.yaml")

    # Without a protocol that declares the pairing, similar kinematics are not a
    # permission -- both active hands have sixteen joints.
    with pytest.raises(CompatibilityError):
        leap.bind_to("wonik_allegro.yaml")

    binding = leap.bind_to("wonik_allegro.yaml", protocol=protocol)
    assert binding.is_transfer
    assert binding.training_robot == "leap_hand"
    assert binding.runtime_robot == "wonik_allegro"
    assert binding.protocol_hash == protocol.protocol_hash

    allegro = QDGrasp("qdgrasp-dummy-n.yaml", robot="wonik_allegro.yaml")
    refuses(
        lambda: allegro.load_weights(info.directory),
        because="LEAP weights loaded into an Allegro model with no binding, on matching tensor shapes",
    )
    allegro.load_weights(info.directory, binding=binding)

    result = allegro.predict(torch.randn(32, 3))
    assert result.training_robot_hash == leap.robot_config.content_hash()
    assert result.runtime_robot_hash == allegro.robot_config.content_hash()
    assert result.training_robot_hash != result.runtime_robot_hash


@characterization("COR-09", note="the paused hand has no permitted pairing")
def test_an_undeclared_pairing_is_refused_however_similar_the_hands_are() -> None:
    from qdgrasp.api import QDGrasp
    from qdgrasp.engine.compatibility import CompatibilityError
    from qdgrasp.models.protocol import load_protocol

    leap = QDGrasp("qdgrasp-dummy-n.yaml", robot="leap_hand.yaml")
    protocol = load_protocol(REPO_ROOT / "configs" / "phase5" / "protocol-v2.yaml")

    with pytest.raises(CompatibilityError, match="declares"):
        leap.bind_to("shadow_hand.yaml", protocol=protocol)
