"""DGN-Open-Tiny → QDGrasp-Flow batches (P5-01).

The shards already carry the fields the loss needs, so this is deliberately a
thin adapter rather than a second dataset layer.  What it adds is the three
refusals that a thin adapter is exactly the wrong place to leave out:

*A shard whose bytes do not match the manifest is not loaded.*  A dataset whose
content drifted from its own hashes is a silent relabelling, and every number
downstream would be about something nobody can name.

*A paused hand is not loaded by default.*  ``ADR-0008`` keeps Shadow's shards in
the dataset for reproducing history; a default workload picking them up would
put a paused hand into a release artifact without anyone choosing it.

*A batch never mixes robots.*  The model takes one ``HandGraph`` per forward
pass, so a mixed batch would silently evaluate every sample against whichever
hand happened to come first.  Grouping by robot is the honest collation.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from collections.abc import Iterator, Sequence
from pathlib import Path

import torch

from qdgrasp.config.active_scope import ACTIVE_HANDS, PAUSED_HANDS, is_paused

#: Fields a sample must carry to reach the loss.
REQUIRED_FIELDS: tuple[str, ...] = (
    "points",
    "palm_pos",
    "palm_rot",
    "joint_angles",
    "fingertip_positions",
    "success",
)

#: Everything the collated batch carries into ``forward_and_loss``.
BATCH_FIELDS: tuple[str, ...] = REQUIRED_FIELDS


class DatasetError(ValueError):
    """The adapter refuses to hand on data it cannot stand behind."""


@dataclasses.dataclass(frozen=True)
class ShardRef:
    """One shard as the manifest describes it."""

    filename: str
    robot_name: str
    split: str
    sha256: str
    num_samples: int
    positive_samples: int


def load_manifest(root: Path) -> dict:
    path = Path(root) / "dataset_manifest.json"
    if not path.is_file():
        raise DatasetError(f"no dataset manifest at {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("invalidated"):
        raise DatasetError(
            f"dataset {manifest.get('dataset_id')} is marked invalidated: {manifest.get('invalidation_reason')!r}"
        )
    return manifest


def shard_refs(
    manifest: dict,
    *,
    split: str | None = None,
    robots: Sequence[str] | None = None,
    allow_paused: bool = False,
) -> list[ShardRef]:
    """Shards for a split and a set of hands, refusing paused hands by default."""

    wanted = tuple(robots) if robots is not None else ACTIVE_HANDS
    paused = sorted({name for name in wanted if is_paused(name)})
    if paused and not allow_paused:
        raise DatasetError(
            f"{paused} are paused by ADR-0008 and may not be selected by a default workload; "
            f"the active corpus is {list(ACTIVE_HANDS)}. Pass allow_paused=True only for a declared "
            "diagnostic whose output is marked non_release."
        )
    refs = [
        ShardRef(
            filename=entry["filename"],
            robot_name=entry["robot_name"],
            split=entry["split"],
            sha256=entry["sha256"],
            num_samples=int(entry["num_samples"]),
            positive_samples=int(entry["positive_samples"]),
        )
        for entry in manifest.get("shards", [])
        if (split is None or entry["split"] == split) and entry["robot_name"] in wanted
    ]
    if not refs:
        raise DatasetError(f"no shard for split={split!r} and robots={list(wanted)}")
    return refs


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_shard(root: Path, ref: ShardRef, *, verify: bool = True) -> list[dict]:
    """Load one shard, checking its bytes against the manifest first."""

    path = Path(root) / ref.filename
    if not path.is_file():
        raise DatasetError(f"shard {ref.filename} is named by the manifest but missing from {root}")
    if verify:
        actual = _sha256(path)
        if actual != ref.sha256:
            raise DatasetError(
                f"shard {ref.filename} hashes to {actual}, the manifest says {ref.sha256}; "
                "the dataset and its manifest describe different data"
            )
    samples = torch.load(path, map_location="cpu", weights_only=True)
    if len(samples) != ref.num_samples:
        raise DatasetError(f"shard {ref.filename} holds {len(samples)} samples, the manifest says {ref.num_samples}")
    for index, sample in enumerate(samples):
        missing = [field for field in REQUIRED_FIELDS if field not in sample]
        if missing:
            raise DatasetError(f"{ref.filename}[{index}] is missing {missing}")
    return samples


def collate(samples: Sequence[dict]) -> dict[str, torch.Tensor]:
    """Stack samples into one batch, refusing a batch that mixes hands."""

    if not samples:
        raise DatasetError("cannot collate an empty batch")
    robots = {sample.get("robot_name") for sample in samples}
    if len(robots) > 1:
        raise DatasetError(
            f"a batch may not mix robots {sorted(robots)}: the model takes one HandGraph per forward pass, "
            "so a mixed batch would evaluate every sample against whichever hand came first"
        )
    counts = {tuple(sample["points"].shape) for sample in samples}
    if len(counts) > 1:
        raise DatasetError(f"samples carry different point-cloud shapes {sorted(counts)}")

    batch = {}
    for field in BATCH_FIELDS:
        values = [sample[field] for sample in samples]
        stacked = torch.stack([value if value.dim() else value.reshape(()) for value in values])
        batch[field] = stacked.float()
    return batch


class FlowDataset(Sequence[dict]):
    """Every sample of one split for one hand, in manifest order.

    Map-style on purpose: the engine's resume contract needs a position it can
    write into a checkpoint, and an iterator has none.
    """

    def __init__(self, root: str | Path, *, split: str, robot: str, verify: bool = True) -> None:
        self.root = Path(root)
        self.split = split
        self.robot = robot
        manifest = load_manifest(self.root)
        self.dataset_id = manifest.get("dataset_id", "")
        refs = shard_refs(manifest, split=split, robots=[robot])
        self.samples: list[dict] = []
        for ref in refs:
            self.samples.extend(load_shard(self.root, ref, verify=verify))
        self.positive_samples = sum(ref.positive_samples for ref in refs)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:  # type: ignore[override]
        return self.samples[index]

    @property
    def object_ids(self) -> tuple[str, ...]:
        return tuple(sorted({sample["object_id"] for sample in self.samples}))

    @property
    def positive_fraction(self) -> float:
        """Share of samples labelled success.

        Read this before reading any training curve.  ``DGN-Open-Tiny`` is a
        pipeline fixture, not a balanced corpus -- its train split for LEAP holds
        one positive in forty-four -- so a quality head trained on it learns the
        prior, and that is a fact about the data rather than about the model.
        """

        if not self.samples:
            return 0.0
        return float(sum(float(sample["success"]) for sample in self.samples) / len(self.samples))

    def batches(self, batch_size: int, *, generator: torch.Generator | None = None, shuffle: bool = False):
        order = (
            torch.randperm(len(self.samples), generator=generator).tolist()
            if shuffle
            else list(range(len(self.samples)))
        )
        for start in range(0, len(order), batch_size):
            yield collate([self.samples[index] for index in order[start : start + batch_size]])


def iter_active_datasets(root: str | Path, *, split: str, verify: bool = True) -> Iterator[tuple[str, FlowDataset]]:
    """One dataset per active hand, in canonical order."""

    for robot in ACTIVE_HANDS:
        yield robot, FlowDataset(root, split=split, robot=robot, verify=verify)


__all__ = [
    "ACTIVE_HANDS",
    "BATCH_FIELDS",
    "PAUSED_HANDS",
    "REQUIRED_FIELDS",
    "DatasetError",
    "FlowDataset",
    "ShardRef",
    "collate",
    "iter_active_datasets",
    "load_manifest",
    "load_shard",
    "shard_refs",
]
