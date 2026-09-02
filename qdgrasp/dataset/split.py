"""Disjoint train/validation dataset splitting by object family.

The unit of assignment is a *family*.  Earlier versions shuffled members inside
each ``shape_type`` while claiming a family hold-out; that guaranteed the exact
leakage the split was meant to prevent.  This module now keeps every family on
one side of the boundary and lets a locked protocol name the held-out families
explicitly.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ..objects.schema import ObjectManifestSpec


def create_object_family_splits(
    objects: Sequence[ObjectManifestSpec],
    val_fraction: float = 0.2,
    seed: int = 42,
    *,
    val_families: Sequence[str] | None = None,
) -> dict[str, list[str]]:
    """Partition objects into train and validation without family leakage.

    When ``val_families`` is supplied it is the locked hold-out declaration and
    every other family is assigned to train.  Otherwise whole families are
    shuffled deterministically and selected until the requested validation
    fraction is reached as closely as a group-wise split permits.  A family is
    never divided merely to make the fraction look exact.
    """
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be strictly between 0 and 1")
    if not objects:
        raise ValueError("cannot split an empty object collection")

    rng = np.random.default_rng(seed)

    by_family: dict[str, list[str]] = {}
    for obj in objects:
        by_family.setdefault(obj.family, []).append(obj.object_id)

    known_families = set(by_family)
    if val_families is not None:
        selected = set(val_families)
        unknown = sorted(selected - known_families)
        if unknown:
            raise ValueError(f"validation families are not present in the object collection: {unknown}")
        if not selected or selected == known_families:
            raise ValueError("val_families must leave at least one family on each side")
    else:
        order = sorted(known_families)
        rng.shuffle(order)
        target = max(1, round(len(objects) * val_fraction))
        selected = set()
        selected_count = 0
        # Pick the next whole group when it improves (or initially establishes)
        # the distance to the requested sample count.  At least one group is
        # selected, and at least one is kept for train.
        for family in order:
            if len(selected) >= len(order) - 1:
                break
            group_count = len(by_family[family])
            before = abs(target - selected_count)
            after = abs(target - (selected_count + group_count))
            if not selected or after <= before:
                selected.add(family)
                selected_count += group_count

    val_ids = sorted(object_id for family in selected for object_id in by_family[family])
    train_ids = sorted(
        object_id for family, object_ids in by_family.items() if family not in selected for object_id in object_ids
    )

    train_families = {obj.family for obj in objects if obj.object_id in set(train_ids)}
    validation_families = {obj.family for obj in objects if obj.object_id in set(val_ids)}
    if train_families & validation_families:  # defensive invariant, not input validation
        raise RuntimeError("family-wise split produced leakage")

    return {
        "train": train_ids,
        "val": val_ids,
    }
