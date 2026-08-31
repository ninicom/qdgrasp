"""The evaluation protocol, locked by hash before the first run (P5-02).

``ROADMAP-P5-001`` §1.2: a protocol chosen after seeing results is not a
protocol.  So splits, seeds, ablations, metrics and the selection rule live in
one document, that document hashes to a value, and every artifact carries the
value.  Changing the protocol changes the hash, and a result produced under the
old one can no longer be presented as if it were produced under the new one.

Leakage is a hard error, never a warning.  A warning about a leaked object is
read once, by the person who already knows, and then never again.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from qdgrasp.config.active_scope import ACTIVE_HANDS, is_active

PROTOCOL_SCHEMA = "qdgrasp/protocol/v1"

#: Ablations P5 may run.  A free-text name would let two runs disagree about
#: what they compared while both looking valid.
ABLATION_REGISTRY: tuple[str, ...] = (
    "baseline",
    "no_graph",
    "direct_only",
    "no_fk_consistency",
    "no_quality_guidance",
)

#: Metrics P5 may report, from §3.3 of the plan.
METRIC_REGISTRY: tuple[str, ...] = (
    "success",
    "collision",
    "penetration",
    "diversity",
    "coverage",
    "latency",
)

#: Selection rules.  ``total_loss`` is deliberately absent: P4 measured that the
#: flow term has an irreducible floor, so the total mixes a constant into the
#: signal.  Naming it here would make the mistake configurable.
SELECTION_REGISTRY: tuple[str, ...] = (
    "pose_then_physics",
    "physics_success",
)


class ProtocolError(ValueError):
    """The protocol document describes something that cannot be measured."""


@dataclasses.dataclass(frozen=True)
class HeldOutEmbodiment:
    train_hand: str
    test_hand: str


@dataclasses.dataclass(frozen=True)
class Protocol:
    """One locked evaluation protocol."""

    schema: str
    name: str
    dataset_id: str
    train_objects: tuple[str, ...]
    val_objects: tuple[str, ...]
    heldout_family: str
    heldout_embodiment: HeldOutEmbodiment
    seeds: tuple[int, ...]
    ablations: tuple[str, ...]
    metrics: tuple[str, ...]
    selection: str

    def to_document(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "name": self.name,
            "dataset_id": self.dataset_id,
            "splits": {
                "train_objects": list(self.train_objects),
                "val_objects": list(self.val_objects),
                "heldout_family": self.heldout_family,
                "heldout_embodiment": dataclasses.asdict(self.heldout_embodiment),
            },
            "seeds": list(self.seeds),
            "ablations": list(self.ablations),
            "metrics": list(self.metrics),
            "selection": self.selection,
        }

    @property
    def protocol_hash(self) -> str:
        """Stable over key order, sensitive to every value."""

        payload = json.dumps(self.to_document(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def family_of(self, object_id: str) -> str:
        """``comp_t_shape_01`` -> ``comp``.  The corpus names families by prefix."""

        return object_id.split("_", 1)[0]

    def validate(self) -> None:
        if self.schema != PROTOCOL_SCHEMA:
            raise ProtocolError(f"unsupported protocol schema {self.schema!r}; expected {PROTOCOL_SCHEMA!r}")
        if not self.train_objects or not self.val_objects:
            raise ProtocolError("a protocol needs both a train and a val split")

        overlap = sorted(set(self.train_objects) & set(self.val_objects))
        if overlap:
            raise ProtocolError(f"objects {overlap} appear in both train and val; that is leakage, not a split")

        leaked_family = sorted(
            object_id for object_id in self.train_objects if self.family_of(object_id) == self.heldout_family
        )
        if leaked_family:
            raise ProtocolError(
                f"held-out family {self.heldout_family!r} still has {leaked_family} in train; "
                "a family is held out only when none of its members is trained on"
            )
        if not any(self.family_of(object_id) == self.heldout_family for object_id in self.val_objects):
            raise ProtocolError(
                f"held-out family {self.heldout_family!r} has no member in val either, so nothing measures it"
            )

        embodiment = self.heldout_embodiment
        for hand in (embodiment.train_hand, embodiment.test_hand):
            if not is_active(hand):
                raise ProtocolError(f"hand {hand!r} is not in the active corpus {list(ACTIVE_HANDS)}")
        if embodiment.train_hand == embodiment.test_hand:
            raise ProtocolError("held-out embodiment must train and test on different hands")

        if not self.seeds:
            raise ProtocolError("a protocol needs at least one seed; a single unnamed run is not reproducible")
        if len(set(self.seeds)) != len(self.seeds):
            raise ProtocolError(f"seeds {list(self.seeds)} repeat; a repeated seed is one run reported twice")

        for name, registry, label in (
            (self.ablations, ABLATION_REGISTRY, "ablation"),
            (self.metrics, METRIC_REGISTRY, "metric"),
            ((self.selection,), SELECTION_REGISTRY, "selection rule"),
        ):
            unknown = sorted(set(name) - set(registry))
            if unknown:
                raise ProtocolError(f"unknown {label}(s) {unknown}; registered: {list(registry)}")
        if "baseline" not in self.ablations:
            raise ProtocolError(
                "ablations must include 'baseline'; an ablation with nothing to differ from measures nothing"
            )


def parse_protocol(document: dict[str, Any]) -> Protocol:
    """Build and validate a protocol from a parsed YAML mapping."""

    try:
        splits = document["splits"]
        embodiment = splits["heldout_embodiment"]
        protocol = Protocol(
            schema=document["schema"],
            name=document["name"],
            dataset_id=document["dataset_id"],
            train_objects=tuple(splits["train_objects"]),
            val_objects=tuple(splits["val_objects"]),
            heldout_family=splits["heldout_family"],
            heldout_embodiment=HeldOutEmbodiment(
                train_hand=embodiment["train_hand"], test_hand=embodiment["test_hand"]
            ),
            seeds=tuple(int(seed) for seed in document["seeds"]),
            ablations=tuple(document["ablations"]),
            metrics=tuple(document["metrics"]),
            selection=document["selection"],
        )
    except KeyError as error:
        raise ProtocolError(f"protocol document is missing {error}") from None
    protocol.validate()
    return protocol


def load_protocol(path: str | Path) -> Protocol:
    document = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ProtocolError(f"{path} does not contain a protocol mapping")
    return parse_protocol(document)


def check_dataset_agreement(protocol: Protocol, manifest: dict[str, Any]) -> None:
    """The protocol must describe the dataset it will actually be run on."""

    if manifest.get("dataset_id") != protocol.dataset_id:
        raise ProtocolError(
            f"protocol names dataset {protocol.dataset_id!r}, manifest says {manifest.get('dataset_id')!r}"
        )
    known = set(manifest.get("object_manifest_hashes", {}))
    unknown = sorted((set(protocol.train_objects) | set(protocol.val_objects)) - known)
    if unknown:
        raise ProtocolError(f"protocol names objects {unknown} that the dataset does not contain")
