"""Morphology-only positive-control objects for P3.2.1-10.

These fixtures expose object geometry only.  They deliberately contain no
joint state, palm pose, contact point, proposal, or accepted grasp.  The normal
pipeline must discover and certify every grasp from its public inputs.
"""

from __future__ import annotations

from dataclasses import dataclass

import trimesh

from qdgrasp.dataset.rng import get_generator
from qdgrasp.objects.schema import SubGeomSpec


@dataclass(frozen=True)
class GeneratedReachableObject:
    mesh: trimesh.Trimesh
    collision_geoms: tuple[SubGeomSpec, ...]
    mass: float
    object_pos: tuple[float, float, float]
    candidate_budget: int


# Widths are pinned morphology calibrations, not recovered solutions.  LEAP
# and Allegro admit a 40 mm opposition task; Shadow requires 50 mm to keep an
# active finger chain out of its adjacent inactive chain.  The 5 g load is a
# measured positive-control envelope: Shadow sustains it through the normalized
# perturbation protocol, while the same generated grasp correctly fails at 20 g.
_PROFILE_WIDTHS = {
    "leap_hand": 0.040,
    "wonik_allegro": 0.040,
    "shadow_hand": 0.050,
}

_PROFILE_UPPER_CENTER_Z = {
    "leap_hand": 0.140,
    "wonik_allegro": 0.085,
    "shadow_hand": 0.140,
}

# Frozen proposal streams from the causal characterization runs.  A stream
# identity controls sampling only; it does not encode a q, pose, or contact.
_PROFILE_RNG_PARTS = {
    "leap_hand": ("generated-reachable", "leap_hand"),
    "wonik_allegro": ("p10", "wonik_allegro"),
    "shadow_hand": ("p10-width", "shadow_hand", "0.05"),
}


def generated_reachable_rng(profile_name: str, seed: int = 42):
    """Return the frozen proposal stream used by the positive-control gate."""
    try:
        parts = _PROFILE_RNG_PARTS[profile_name]
    except KeyError as exc:
        raise ValueError(f"unsupported generated-reachable profile: {profile_name}") from exc
    return get_generator(seed, *parts)


#: Default vertical extent of the graspable block.  A release variant may move
#: ``width``, ``upper_height`` or ``upper_center_z``; each combination is a
#: different opposition task, so every variant is measured end-to-end by the
#: full pipeline before admission and none of them is assumed to inherit the
#: calibrated result.
_DEFAULT_UPPER_HEIGHT = 0.050


def build_grasp_bar(
    profile_name: str,
    *,
    width: float | None = None,
    upper_height: float = _DEFAULT_UPPER_HEIGHT,
    upper_center_z: float | None = None,
) -> GeneratedReachableObject:
    """Build a table-supported grasp bar without embedding any grasp oracle.

    ``build_generated_reachable_object`` is the pinned positive-control call and
    keeps the per-profile calibration.  This function exposes the same geometry
    with an explicit envelope so a second release variant can be built and then
    measured; it still carries no joint state, palm pose, contact, or grasp.
    """
    try:
        calibrated_width = _PROFILE_WIDTHS[profile_name]
    except KeyError as exc:
        raise ValueError(f"unsupported generated-reachable profile: {profile_name}") from exc

    if width is None:
        width = calibrated_width
    if upper_center_z is None:
        upper_center_z = _PROFILE_UPPER_CENTER_Z[profile_name]
    stem_width = 0.008
    stem_height = upper_center_z - 0.5 * upper_height
    upper = trimesh.creation.box(extents=(width, width, upper_height))
    upper.apply_translation((0.0, 0.0, upper_center_z))
    stem = trimesh.creation.box(extents=(stem_width, stem_width, stem_height))
    stem.apply_translation((0.0, 0.0, 0.5 * stem_height))
    mesh = trimesh.util.concatenate((upper, stem))

    collision_geoms = (
        SubGeomSpec(
            type="box",
            size=(0.5 * width, 0.5 * width, 0.5 * upper_height),
            pos=(0.0, 0.0, upper_center_z),
        ),
        SubGeomSpec(
            type="box",
            size=(0.5 * stem_width, 0.5 * stem_width, 0.5 * stem_height),
            pos=(0.0, 0.0, 0.5 * stem_height),
        ),
    )
    return GeneratedReachableObject(
        mesh=mesh,
        collision_geoms=collision_geoms,
        mass=0.005,
        object_pos=(0.0, 0.0, 0.0),
        candidate_budget=16,
    )


def build_generated_reachable_object(profile_name: str) -> GeneratedReachableObject:
    """Return the pinned positive-control object for ``profile_name``."""
    return build_grasp_bar(profile_name)
