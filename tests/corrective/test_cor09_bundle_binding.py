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

from _corrective_support import characterization, refuses


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


@characterization("COR-09", note="one robot hash serves both roles")
def test_a_bundle_separates_the_training_hand_from_the_runtime_hand(tmp_path: Path) -> None:
    """Held-out inference needs both recorded; one field cannot hold two answers."""

    from qdgrasp.api import QDGrasp

    grasper = QDGrasp("qdgrasp-dummy-n.yaml", robot="leap_hand.yaml")
    info = grasper.save_bundle(tmp_path / "bundle")

    hashes = info.manifest["hashes"]
    missing = [key for key in ("training_robot_config", "runtime_robot_config") if key not in hashes]
    assert not missing, (
        f"the bundle records {missing} nowhere; with a single robot hash, the exact-match gate either forbids "
        "the cross-embodiment inference the protocol requires or waves through a mislabelled artifact"
    )
