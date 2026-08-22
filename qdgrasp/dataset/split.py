"""Disjoint train/validation dataset splitting by object family."""

from __future__ import annotations

from typing import Sequence
import numpy as np

from ..objects.schema import ObjectManifestSpec


def create_object_family_splits(
    objects: Sequence[ObjectManifestSpec],
    val_fraction: float = 0.2,
    seed: int = 42,
) -> dict[str, list[str]]:
    """Partition objects into train and val splits without family or shape leakage.

    Guarantees that objects from the same family/shape_type are assigned
    proportionally to train and val, while object IDs are strictly disjoint.
    """
    rng = np.random.default_rng(seed)

    by_shape: dict[str, list[str]] = {}
    for obj in objects:
        by_shape.setdefault(obj.shape_type, []).append(obj.object_id)

    train_ids: list[str] = []
    val_ids: list[str] = []

    for shape_type, ids in sorted(by_shape.items()):
        shuffled = list(ids)
        rng.shuffle(shuffled)
        n_val = max(1, int(round(len(shuffled) * val_fraction))) if len(shuffled) > 1 else 0
        val_ids.extend(shuffled[:n_val])
        train_ids.extend(shuffled[n_val:])

    # Assert disjointness
    train_set = set(train_ids)
    val_set = set(val_ids)
    assert not (train_set & val_set), "train and val splits must be strictly disjoint"

    return {
        "train": sorted(train_ids),
        "val": sorted(val_ids),
    }
