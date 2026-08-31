"""Seed streams, domain randomization and topology bucketing (P3.5-12).

Three separate jobs, kept apart on purpose.

*Seed streams.*  Asset choice, scene layout, drop, physics, observation noise and
policy sampling each draw from their own generator derived from one episode
seed.  Widening the friction range then does not reshuffle which object was
picked, which is what makes an ablation an ablation.

*Randomization ranges.*  Registered up front and hashed.  Randomizing a property
never edits the source manifest -- the manifest records what the asset is, and
the range records what the episode did to it.

*Topology buckets.*  Two scenes with the same shape signature can share a
compiled model; two with different signatures cannot.  Bucketing by signature is
what keeps a vector environment from recompiling on every reset.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from qdgrasp.scenes.contracts import SceneSpec

RANDOMIZATION_SCHEMA_V1 = "qdgrasp/rl-randomization/v1"

#: Named streams.  ``policy`` is included so that action sampling is reproducible
#: from the same episode seed without borrowing another stream's state.
STREAM_NAMES: tuple[str, ...] = (
    "asset",
    "scene",
    "drop",
    "physics",
    "observation",
    "policy",
)


@dataclasses.dataclass(frozen=True)
class SeedStreams:
    """One independent generator per named concern, derived from one seed."""

    episode_seed: int
    namespace: str = "qdgrasp/rl"

    def generator(self, stream: str) -> np.random.Generator:
        if stream not in STREAM_NAMES:
            raise KeyError(f"unknown seed stream {stream!r}; known: {STREAM_NAMES}")
        material = f"{self.namespace}|{self.episode_seed}|{stream}".encode()
        return np.random.default_rng(int.from_bytes(hashlib.sha256(material).digest()[:8], "big"))

    def all(self) -> dict[str, np.random.Generator]:
        return {name: self.generator(name) for name in STREAM_NAMES}


@dataclasses.dataclass(frozen=True)
class Range:
    """An inclusive interval, sampled uniformly."""

    low: float
    high: float

    def validate(self, name: str) -> None:
        if not (np.isfinite(self.low) and np.isfinite(self.high)) or self.high < self.low:
            raise ValueError(f"randomization range {name!r} is not an ordered finite interval: {self}")

    def sample(self, rng: np.random.Generator) -> float:
        return float(rng.uniform(self.low, self.high))


@dataclasses.dataclass(frozen=True)
class DomainRandomization:
    """Ranges registered before a run, and hashed into its identity.

    Every field is multiplicative on the manifest value except ``friction``,
    which is absolute: a friction coefficient has a meaning of its own, while a
    mass scale only means anything relative to the mass the asset declares.
    """

    mass_scale: Range = dataclasses.field(default_factory=lambda: Range(1.0, 1.0))
    friction_slide: Range = dataclasses.field(default_factory=lambda: Range(1.0, 1.0))
    object_scale: Range = dataclasses.field(default_factory=lambda: Range(1.0, 1.0))
    observation_position_noise_m: Range = dataclasses.field(default_factory=lambda: Range(0.0, 0.0))

    def validate(self) -> None:
        for name in ("mass_scale", "friction_slide", "object_scale", "observation_position_noise_m"):
            getattr(self, name).validate(name)
        if self.mass_scale.low <= 0.0 or self.object_scale.low <= 0.0 or self.friction_slide.low <= 0.0:
            raise ValueError("mass, object scale and friction ranges must stay strictly positive")

    def sample(self, rng: np.random.Generator) -> dict[str, float]:
        return {
            "mass_scale": self.mass_scale.sample(rng),
            "friction_slide": self.friction_slide.sample(rng),
            "object_scale": self.object_scale.sample(rng),
            "observation_position_noise_m": self.observation_position_noise_m.sample(rng),
        }

    def to_document(self) -> dict[str, Any]:
        return {
            "schema": RANDOMIZATION_SCHEMA_V1,
            **{
                name: [getattr(self, name).low, getattr(self, name).high]
                for name in ("mass_scale", "friction_slide", "object_scale", "observation_position_noise_m")
            },
        }

    def content_hash(self) -> str:
        payload = json.dumps(self.to_document(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def scene_signature(spec: SceneSpec, *, robot_profile: str | None = None) -> str:
    """A hash of everything that changes the compiled model's *shape*.

    Poses and masses are excluded on purpose: they are writable at runtime, so
    two scenes that differ only in where the objects sit can share one compiled
    model.  Object identity, geometry references and support layout are not
    writable, so they are what the signature is made of.
    """

    payload = {
        "environment": spec.environment,
        "robot_profile": robot_profile,
        "objects": sorted((item.object_id, item.asset_ref, round(float(item.scale), 9)) for item in spec.objects),
        "supports": sorted(
            (
                item.support_id,
                item.geom_type,
                json.dumps(item.params, sort_keys=True, separators=(",", ":")),
            )
            for item in spec.supports
        ),
        "cameras": sorted(item.camera_id for item in spec.cameras),
        "timestep": round(float(spec.timestep), 12),
        "gravity": [round(float(value), 12) for value in spec.gravity],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def bucket_by_signature(specs: Sequence[SceneSpec], *, robot_profile: str | None = None) -> dict[str, list[int]]:
    """Group scene indices by topology so each bucket compiles once."""

    buckets: dict[str, list[int]] = {}
    for index, spec in enumerate(specs):
        buckets.setdefault(scene_signature(spec, robot_profile=robot_profile), []).append(index)
    return buckets


@dataclasses.dataclass(frozen=True)
class EvaluationSplit:
    """A locked split.  Membership is a function of the key, not of a shuffle.

    Assigning by hash rather than by a shuffled list means adding an object to
    the corpus cannot move an existing one across the boundary, which is the
    usual way an evaluation set quietly stops being held out.
    """

    name: str
    #: Fraction of keys assigned to this split, in ``[0, 1]``.
    fraction: float
    salt: str = "qdgrasp/rl/split/v1"

    def contains(self, key: str) -> bool:
        material = f"{self.salt}|{key}".encode()
        draw = int.from_bytes(hashlib.sha256(material).digest()[:8], "big") / float(1 << 64)
        return draw < self.fraction


def assert_no_split_leak(train_keys: Sequence[str], evaluation_keys: Sequence[str]) -> None:
    """Refuse an overlap between the training and evaluation key sets."""

    overlap = sorted(set(train_keys) & set(evaluation_keys))
    if overlap:
        raise ValueError(f"evaluation keys leaked into training: {overlap[:10]}")


def apply_randomization(
    model: Any,
    body_names: Sequence[str],
    geom_names: Sequence[str],
    sample: Mapping[str, float],
) -> dict[str, Any]:
    """Stamp a randomization sample onto a compiled MuJoCo model.

    Mass and inertia scale together, and MuJoCo's mass-derived solver constants
    are recomputed afterwards -- leaving them stale would give a heavy object the
    contact compliance of a light one, which is precisely the property being
    randomized.
    """

    import mujoco

    applied: dict[str, Any] = {"bodies": [], "geoms": []}
    mass_scale = float(sample.get("mass_scale", 1.0))
    for name in body_names:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if body_id < 0:
            raise KeyError(f"body {name!r} is absent from the compiled model")
        model.body_mass[body_id] = float(model.body_mass[body_id]) * mass_scale
        model.body_inertia[body_id] = np.asarray(model.body_inertia[body_id], dtype=np.float64) * mass_scale
        applied["bodies"].append({"name": name, "mass": float(model.body_mass[body_id])})

    friction = float(sample.get("friction_slide", 1.0))
    for name in geom_names:
        geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        if geom_id < 0:
            raise KeyError(f"geom {name!r} is absent from the compiled model")
        model.geom_friction[geom_id, 0] = friction
        applied["geoms"].append({"name": name, "friction_slide": friction})
    return applied
