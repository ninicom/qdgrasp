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
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

from qdgrasp.config.active_scope import ACTIVE_HANDS, is_active

PROTOCOL_SCHEMA_V1 = "qdgrasp/protocol/v1"
PROTOCOL_SCHEMA_V2 = "qdgrasp/protocol/v2"
PROTOCOL_SCHEMA = PROTOCOL_SCHEMA_V2

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
    object_families: tuple[tuple[str, str], ...]
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
            "object_families": dict(self.object_families),
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
        """Return family membership locked from the hashed object manifests."""

        families = dict(self.object_families)
        try:
            return families[object_id]
        except KeyError:
            raise ProtocolError(
                f"object {object_id!r} has no family binding in protocol {self.name!r}; "
                "family membership must come from a hashed object manifest, never an id prefix"
            ) from None

    def validate(self) -> None:
        if self.schema != PROTOCOL_SCHEMA:
            migration = (
                f"; {PROTOCOL_SCHEMA_V1!r} inferred family membership from object-id prefixes and is "
                "historical-only"
                if self.schema == PROTOCOL_SCHEMA_V1
                else ""
            )
            raise ProtocolError(f"unsupported protocol schema {self.schema!r}; expected {PROTOCOL_SCHEMA!r}{migration}")
        if not self.train_objects or not self.val_objects:
            raise ProtocolError("a protocol needs both a train and a val split")

        selected_objects = set(self.train_objects) | set(self.val_objects)
        family_objects = {object_id for object_id, _family in self.object_families}
        missing_families = sorted(selected_objects - family_objects)
        extra_families = sorted(family_objects - selected_objects)
        if missing_families or extra_families:
            raise ProtocolError(
                "object_families must exactly cover the protocol object matrix; "
                f"missing={missing_families}, extra={extra_families}"
            )

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

    # The schema decides which keys are even expected, so it is read first: a
    # v1 document is superseded, not malformed, and saying "missing
    # 'object_families'" would send the reader to fix the wrong thing.
    schema = document.get("schema")
    if schema != PROTOCOL_SCHEMA:
        migration = (
            f"; {PROTOCOL_SCHEMA_V1!r} inferred family membership from object-id prefixes and is "
            "historical-only"
            if schema == PROTOCOL_SCHEMA_V1
            else ""
        )
        raise ProtocolError(f"unsupported protocol schema {schema!r}; expected {PROTOCOL_SCHEMA!r}{migration}")

    try:
        splits = document["splits"]
        embodiment = splits["heldout_embodiment"]
        protocol = Protocol(
            schema=document["schema"],
            name=document["name"],
            dataset_id=document["dataset_id"],
            object_families=tuple(
                sorted((str(object_id), str(family)) for object_id, family in document["object_families"].items())
            ),
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


def _canonical_hash(document: Mapping[str, Any]) -> str:
    payload = json.dumps(dict(document), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def verify_object_family_bindings(protocol: Protocol, dataset_root: str | Path, manifest: Mapping[str, Any]) -> None:
    """Verify protocol family bindings against the immutable object manifests.

    The dataset manifest already hashes each object manifest.  We verify those
    bytes before reading ``family`` so neither an object rename nor a stale
    side-file can silently move an object across a generalisation boundary.
    """

    root = Path(dataset_root).resolve()
    declared_hashes = manifest.get("object_manifest_hashes", {})
    if not isinstance(declared_hashes, Mapping):
        raise ProtocolError("dataset manifest object_manifest_hashes must be a mapping")
    for object_id, expected_family in protocol.object_families:
        if not object_id or any(part in {"", ".", ".."} for part in Path(object_id).parts):
            raise ProtocolError(f"unsafe object id in protocol: {object_id!r}")
        path = (root / "objects" / f"{object_id}.manifest.json").resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ProtocolError(f"object manifest for {object_id!r} is missing or escapes {root}")
        expected_hash = declared_hashes.get(object_id)
        actual_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual_hash != expected_hash:
            raise ProtocolError(
                f"object manifest for {object_id!r} hashes to {actual_hash}, dataset declares {expected_hash}"
            )
        document = json.loads(path.read_text(encoding="utf-8"))
        actual_family = document.get("family")
        if actual_family != expected_family:
            raise ProtocolError(
                f"protocol binds {object_id!r} to family {expected_family!r}, hashed object manifest says "
                f"{actual_family!r}"
            )


class ProtocolDatasetView(Sequence[dict[str, Any]]):
    """Materialised, identity-bearing view consumed by a training Runner.

    A physical corpus may retain samples used by another experiment.  This view
    selects the exact ``(split, robot, object_id)`` matrix of one locked
    protocol, records every exclusion, refuses missing objects, and hashes the
    selection together with the dataset and protocol identities.  Thus filtering
    is explicit provenance rather than an unreported list comprehension in a
    gate script.
    """

    def __init__(
        self,
        source: Sequence[dict[str, Any]],
        *,
        protocol: Protocol,
        split: str,
        robot: str,
        manifest: Mapping[str, Any],
        dataset_root: str | Path | None = None,
        dataset_manifest_hash: str | None = None,
    ) -> None:
        if split not in {"train", "val", "test"}:
            raise ProtocolError(f"unsupported protocol split {split!r}; expected train, val or test")
        check_dataset_agreement(protocol, dict(manifest))
        if dataset_root is not None:
            verify_object_family_bindings(protocol, dataset_root, manifest)

        self.protocol = protocol
        self.split = split
        self.robot = robot
        self.dataset_manifest_hash = dataset_manifest_hash or _canonical_hash(manifest)
        self.protocol_hash = protocol.protocol_hash

        heldout = protocol.heldout_embodiment
        robot_admitted = split != "train" or robot == heldout.train_hand
        wanted = set(protocol.train_objects if split == "train" else protocol.val_objects)
        selected: list[dict[str, Any]] = []
        selected_indices: list[int] = []
        excluded: dict[str, int] = {"wrong_robot": 0, "outside_object_matrix": 0, "heldout_train_hand": 0}
        seen_objects: set[str] = set()

        for index, sample in enumerate(source):
            sample_robot = str(sample.get("robot_name", ""))
            object_id = str(sample.get("object_id", ""))
            if sample_robot != robot:
                excluded["wrong_robot"] += 1
                continue
            if not robot_admitted:
                excluded["heldout_train_hand"] += 1
                continue
            if object_id not in wanted:
                excluded["outside_object_matrix"] += 1
                continue
            # ``family_of`` is a lookup into the manifest-derived mapping, not a
            # naming convention.  Calling it here also refuses an unbound id.
            protocol.family_of(object_id)
            selected.append(sample)
            selected_indices.append(index)
            seen_objects.add(object_id)

        if robot_admitted:
            missing = sorted(wanted - seen_objects)
            if missing:
                raise ProtocolError(
                    f"protocol view ({split}, {robot}) is missing objects {missing}; a sample may not be "
                    "silently absent from the declared matrix"
                )

        self.samples = tuple(selected)
        self.excluded_counts = {key: value for key, value in excluded.items() if value}
        identity = {
            "dataset_manifest_hash": self.dataset_manifest_hash,
            "protocol_hash": self.protocol_hash,
            "split": split,
            "robot": robot,
            "selected_source_indices": selected_indices,
            "selected_objects": sorted(seen_objects),
            "excluded_counts": self.excluded_counts,
        }
        self.dataset_view_hash = _canonical_hash(identity)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.samples[index]

    @property
    def object_ids(self) -> tuple[str, ...]:
        return tuple(sorted({str(sample["object_id"]) for sample in self.samples}))

    @property
    def positive_samples(self) -> int:
        return sum(int(float(sample["success"])) for sample in self.samples)

    @property
    def positive_fraction(self) -> float:
        return self.positive_samples / len(self.samples) if self.samples else 0.0

    def manifest(self) -> dict[str, Any]:
        """Identity document persisted into run, resume and public bundle."""

        return {
            "dataset_id": self.protocol.dataset_id,
            "dataset_manifest_hash": self.dataset_manifest_hash,
            "protocol": self.protocol.name,
            "protocol_hash": self.protocol_hash,
            "dataset_view_hash": self.dataset_view_hash,
            "split": self.split,
            "robot_name": self.robot,
            "samples": len(self.samples),
            "positive_samples": self.positive_samples,
            "objects": list(self.object_ids),
            "excluded_counts": self.excluded_counts,
        }


__all__ = [
    "ABLATION_REGISTRY",
    "METRIC_REGISTRY",
    "PROTOCOL_SCHEMA",
    "PROTOCOL_SCHEMA_V1",
    "PROTOCOL_SCHEMA_V2",
    "SELECTION_REGISTRY",
    "HeldOutEmbodiment",
    "Protocol",
    "ProtocolDatasetView",
    "ProtocolError",
    "check_dataset_agreement",
    "load_protocol",
    "parse_protocol",
    "verify_object_family_bindings",
]
