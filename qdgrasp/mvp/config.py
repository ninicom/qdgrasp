"""Locked scope, randomization and evaluation protocol for the Grasp Policy MVP.

``ROADMAP-MVP-001`` §2 requires that the object set, the size/mass/friction
ranges, the palm spawn box and every randomization range live in one versioned
YAML, and that the ranges are not edited once the evaluation seeds are locked.
This module is that contract: a strict, frozen document with a content hash, so
"the config changed" is a measurable statement rather than a claim.

The seed manifests are *derived* from the document rather than stored in it.  A
seed list of eight hundred integers pasted into YAML is a list nobody audits; a
deterministic derivation from ``(mvp_id, tier, index)`` is one line of code and
one test.  What must stay immutable is the derivation and the tier sizes, and
both are covered by :meth:`MvpScopeConfig.eval_manifest_hash`.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

MVP_SCOPE_SCHEMA_V0 = "qdgrasp/mvp-scope/v0"

#: Tiers of the acceptance gate (``ROADMAP-MVP-001`` §7).
EvalTier = Literal["A", "B", "C"]

#: Splits an episode sampler can be asked for.  ``dev`` drives hyperparameter
#: choice, ``train`` feeds the expert recorder and the PPO rollouts, and the
#: three locked tiers are only ever run on a finished candidate.
EpisodeSplit = Literal["train", "dev", "eval_a", "eval_b", "eval_c"]

_TIER_OF_SPLIT: dict[str, str] = {"eval_a": "A", "eval_b": "B", "eval_c": "C"}


class _Doc(BaseModel):
    """Strict, frozen and hashable base for every MVP configuration block."""

    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)


class ObjectVariant(_Doc):
    """One member of the locked cuboid family.

    ``half_width`` is measured along the pinch axis, which is the only extent
    the controller prior is parameterised by.  ``half_depth`` varies too, so a
    held-out variant differs in a dimension the prior does not model as well as
    in one it does.
    """

    variant_id: str
    half_width: float = Field(gt=0.0)
    half_depth: float = Field(gt=0.0)
    half_height: float = Field(gt=0.0)
    #: ``train`` variants may appear in demonstrations, replay buffers and the
    #: normalization fit; ``heldout`` variants may not (``ROADMAP-MVP-001`` §5).
    membership: Literal["train", "heldout"]

    @property
    def half_extents(self) -> tuple[float, float, float]:
        return (self.half_width, self.half_depth, self.half_height)


class RandomizationRanges(_Doc):
    """Per-episode domain randomization, all inclusive intervals."""

    #: Target centre offset from the table origin, metres.
    position_x: tuple[float, float]
    position_y: tuple[float, float]
    #: Target yaw about the table normal, radians.
    yaw: tuple[float, float]
    #: Uniform density of the cuboid, kg/m^3.  Mass follows from the extents.
    density: tuple[float, float]
    #: Tangential friction coefficient of the target's contact pairs.
    friction_slide: tuple[float, float]
    #: Height above the settled resting height the target is released from.
    drop_height: tuple[float, float]

    @model_validator(mode="after")
    def _ordered(self) -> RandomizationRanges:
        for name in (
            "position_x",
            "position_y",
            "yaw",
            "density",
            "friction_slide",
            "drop_height",
        ):
            low, high = getattr(self, name)
            if not low <= high:
                raise ValueError(f"randomization range '{name}' is inverted: {low} > {high}")
        if self.density[0] <= 0.0:
            raise ValueError("density range must be strictly positive")
        if self.friction_slide[0] <= 0.0:
            raise ValueError("friction_slide range must be strictly positive")
        if self.drop_height[0] < 0.0:
            raise ValueError("drop_height range must be non-negative")
        return self


class EpisodeSpec(_Doc):
    """Timing of one episode, expressed in control steps at ``control_hz``."""

    control_hz: float = Field(gt=0.0)
    #: Physics steps executed per control step.  ``control_hz * substeps`` must
    #: equal the compiled model's integration rate, which the environment
    #: asserts against the model rather than assuming.
    physics_substeps: int = Field(gt=0)
    settle_steps: int = Field(gt=0)
    approach_steps: int = Field(gt=0)
    enclose_steps: int = Field(gt=0)
    lift_steps: int = Field(gt=0)
    retain_steps: int = Field(gt=0)

    @property
    def control_dt(self) -> float:
        return 1.0 / float(self.control_hz)

    @property
    def max_steps(self) -> int:
        return self.approach_steps + self.enclose_steps + self.lift_steps + self.retain_steps


class SuccessSpec(_Doc):
    """The measured success predicate of ``ROADMAP-MVP-001`` §4."""

    lift_height_m: float = Field(gt=0.0)
    retain_duration_s: float = Field(gt=0.0)
    min_finger_groups: int = Field(ge=1)
    max_penetration_m: float = Field(gt=0.0)
    max_contact_force_n: float = Field(gt=0.0)
    max_contact_impulse_ns: float = Field(gt=0.0)
    #: A target still touching a support geom is support-assisted, and a
    #: support-assisted target has not been acquired.
    support_clearance_m: float = Field(gt=0.0)
    #: A target whose pose jumps by more than this between two consecutive
    #: control steps is a simulator artefact, not a grasp.
    max_pose_jump_m: float = Field(gt=0.0)


class ActionSpec(_Doc):
    """Bounds on the eight-dimensional residual (``ROADMAP-MVP-001`` §3.2)."""

    delta_xyz_m: float = Field(gt=0.0)
    delta_rot_rad: float = Field(gt=0.0)
    synergy_rad: float = Field(gt=0.0)
    #: Half-extent of the box the commanded palm target is clamped into,
    #: centred on the prior's own commanded target.
    workspace_radius_m: float = Field(gt=0.0)
    #: First-order low-pass coefficient applied to the residual before it
    #: reaches the palm target.  ``1.0`` passes the action through unchanged.
    #:
    #: This is not a tuning knob, it is a measured correction.  With the raw
    #: per-step residual, injecting N(0, 0.15) noise on the applied action --
    #: about a millimetre of palm target -- lost the target in 42 of 42
    #: rollouts, because an independent draw every 20 ms is a 50 Hz disturbance
    #: dragged through a mocap weld, not a control input.  A constant residual
    #: three times larger was harmless.  What the interface cannot tolerate is
    #: high-frequency variation, which is precisely what a learned policy and a
    #: PPO exploration draw both produce, so the filter belongs in the
    #: interface rather than in a penalty term asking the policy to be smooth.
    residual_low_pass: float = Field(gt=0.0, le=1.0, default=1.0)

    @property
    def dimension(self) -> int:
        return 8

    def scale_vector(self) -> tuple[float, ...]:
        """Per-dimension scale mapping a unit action onto physical units."""

        return (
            self.delta_xyz_m,
            self.delta_xyz_m,
            self.delta_xyz_m,
            self.delta_rot_rad,
            self.delta_rot_rad,
            self.delta_rot_rad,
            self.synergy_rad,
            self.synergy_rad,
        )


class ControllerSpec(_Doc):
    """The controller prior's own parameters (``ROADMAP-MVP-001`` §3.1).

    The grip regulator exists because an open-loop squeeze depth cannot serve a
    whole size family: the same commanded interference that grips a narrow box
    at four newtons drives twenty through a wide one, where the fingers are near
    full extension and far stiffer.  Regulating grip force closes that gap in the
    controller, which is where §3.1 says it belongs -- not by widening the safety
    budget, and not by asking PPO to paper over it.
    """

    #: Height above the grasp pose the hand starts the approach from.
    pregrasp_height_m: float = Field(gt=0.0)
    #: Vertical travel of the lift phase.  Exceeds the success gate so a
    #: successful lift does not sit exactly on the threshold.
    lift_travel_m: float = Field(gt=0.0)
    #: Closure fraction held during the approach.  Negative values extrapolate
    #: *past* the fitted open posture, spreading the fingers wider than the
    #: grasp needs so a descending fingertip clears the target's side instead of
    #: scraping it.  Without the spread a wide target and a grippy contact jam
    #: the finger on the way down, which is where the measured force spikes came
    #: from -- a geometry problem, not a force-budget problem.
    approach_closure: float = Field(le=0.0)
    #: Normal force per finger group the regulator aims for, newtons.
    grip_force_target_n: float = Field(gt=0.0)
    #: Proportional gain from force error onto the closure fraction, 1/(N*s).
    grip_gain: float = Field(gt=0.0)
    #: Bounds on the closure fraction the regulator may command.  The upper
    #: bound is one: the regulator only ever backs off the fitted squeeze.
    closure_min: float = Field(ge=0.0, le=1.0)
    #: Largest closure change one control step may make.
    closure_rate_limit: float = Field(gt=0.0)

    @model_validator(mode="after")
    def _sane(self) -> ControllerSpec:
        if self.lift_travel_m >= self.pregrasp_height_m:
            raise ValueError("lift travel must stay below the pregrasp height")
        return self


class RewardSpec(_Doc):
    """Weights of the shaping terms.  They never enter the success verdict."""

    approach_progress: float
    target_contact: float
    enclosure: float
    lift_progress: float
    retain_bonus: float
    penetration_penalty: float
    excess_force_penalty: float
    action_rate_penalty: float
    drop_penalty: float
    timeout_penalty: float


class EvalTierSpec(_Doc):
    """One acceptance tier: its domain, its sample size and its gate."""

    tier: EvalTier
    episodes: int = Field(gt=0)
    membership: Literal["train", "heldout"]
    randomized: bool
    min_success_rate: float = Field(ge=0.0, le=1.0)
    #: Wilson 95% lower bound the tier must also clear, when the plan sets one.
    min_wilson_lower_bound: float | None = Field(default=None, ge=0.0, le=1.0)


class MvpScopeConfig(_Doc):
    """The whole locked scope of ``QDGrasp-DexAcquire-MVP-v0``."""

    schema_version: Literal[MVP_SCOPE_SCHEMA_V0] = Field(alias="schema")
    mvp_id: str
    environment_id: str
    artifact_id: str
    release_class: Literal["experimental_non_release"]
    robot_profile: str
    objects: tuple[ObjectVariant, ...]
    randomization: RandomizationRanges
    episode: EpisodeSpec
    success: SuccessSpec
    controller: ControllerSpec
    action: ActionSpec
    reward: RewardSpec
    eval_tiers: tuple[EvalTierSpec, ...]
    #: Root of the deterministic seed derivation.  Changing it invalidates every
    #: locked seed list, which is exactly why it lives under the content hash.
    seed_root: str

    @model_validator(mode="after")
    def _consistent(self) -> MvpScopeConfig:
        ids = [variant.variant_id for variant in self.objects]
        if len(ids) != len(set(ids)):
            raise ValueError("object variant IDs must be unique")
        if not self.train_variants:
            raise ValueError("at least one train object variant is required")
        if not self.heldout_variants:
            raise ValueError("at least one held-out object variant is required")
        tiers = [tier.tier for tier in self.eval_tiers]
        if sorted(tiers) != ["A", "B", "C"]:
            raise ValueError("eval_tiers must define exactly tiers A, B and C")
        retain_steps_required = self.success.retain_duration_s * self.episode.control_hz
        if self.episode.retain_steps < retain_steps_required:
            raise ValueError(
                "retain phase is shorter than the retain duration the success "
                f"predicate demands ({self.episode.retain_steps} < {retain_steps_required:g} steps)"
            )
        return self

    # -- derived views ----------------------------------------------------

    @property
    def train_variants(self) -> tuple[ObjectVariant, ...]:
        return tuple(v for v in self.objects if v.membership == "train")

    @property
    def heldout_variants(self) -> tuple[ObjectVariant, ...]:
        return tuple(v for v in self.objects if v.membership == "heldout")

    def variant(self, variant_id: str) -> ObjectVariant:
        for candidate in self.objects:
            if candidate.variant_id == variant_id:
                return candidate
        raise KeyError(f"unknown object variant: {variant_id}")

    def tier(self, name: EvalTier) -> EvalTierSpec:
        for candidate in self.eval_tiers:
            if candidate.tier == name:
                return candidate
        raise KeyError(f"unknown evaluation tier: {name}")

    def variants_for_split(self, split: EpisodeSplit) -> tuple[ObjectVariant, ...]:
        """Which object variants a split is allowed to draw from.

        Held-out variants appear in exactly one split.  A bug that leaked them
        into ``train`` would not raise anywhere else, so the mapping is written
        once, here, and asserted by test.
        """

        if split in ("train", "dev", "eval_a", "eval_b"):
            return self.train_variants
        if split == "eval_c":
            return self.heldout_variants
        raise KeyError(f"unknown split: {split}")

    # -- hashing and seeds ------------------------------------------------

    def to_document(self) -> dict[str, Any]:
        return self.model_dump(by_alias=True, mode="json")

    def content_hash(self) -> str:
        payload = json.dumps(self.to_document(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def episode_seed(self, split: EpisodeSplit, index: int) -> int:
        """Deterministic 63-bit seed for one episode of one split.

        Derived, not stored: the manifest is reproducible from the config alone,
        and two splits can never collide because the split name is hashed in.
        """

        if index < 0:
            raise ValueError("episode index must be non-negative")
        material = f"{self.seed_root}|{self.mvp_id}|{split}|{index}".encode()
        return int.from_bytes(hashlib.sha256(material).digest()[:8], "big") >> 1

    def locked_seeds(self, tier: EvalTier) -> tuple[int, ...]:
        spec = self.tier(tier)
        split: EpisodeSplit = f"eval_{tier.lower()}"  # type: ignore[assignment]
        return tuple(self.episode_seed(split, index) for index in range(spec.episodes))

    def eval_manifest(self) -> dict[str, Any]:
        """The immutable evaluation manifest: tiers, domains and exact seeds."""

        return {
            "schema": "qdgrasp/mvp-eval-manifest/v0",
            "mvp_id": self.mvp_id,
            "environment_id": self.environment_id,
            "scope_hash": self.content_hash(),
            "tiers": [
                {
                    "tier": spec.tier,
                    "episodes": spec.episodes,
                    "membership": spec.membership,
                    "randomized": spec.randomized,
                    "min_success_rate": spec.min_success_rate,
                    "min_wilson_lower_bound": spec.min_wilson_lower_bound,
                    "variant_ids": [
                        v.variant_id
                        for v in self.variants_for_split(f"eval_{spec.tier.lower()}")  # type: ignore[arg-type]
                    ],
                    "seeds": list(self.locked_seeds(spec.tier)),
                }
                for spec in sorted(self.eval_tiers, key=lambda item: item.tier)
            ],
        }

    def eval_manifest_hash(self) -> str:
        payload = json.dumps(self.eval_manifest(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


#: Repository-relative home of the locked scope documents.
MVP_CONFIG_DIR = Path("configs/mvp")

#: The one document this MVP runs on.
DEFAULT_SCOPE_PATH = MVP_CONFIG_DIR / "dexacquire-mvp-v0.yaml"


def load_mvp_scope(path: str | Path | None = None) -> MvpScopeConfig:
    """Load and validate a locked MVP scope document."""

    resolved = Path(path) if path is not None else DEFAULT_SCOPE_PATH
    if not resolved.is_file():
        raise FileNotFoundError(f"MVP scope document not found: {resolved}")
    document = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise TypeError(f"MVP scope document must be a mapping: {resolved}")
    return MvpScopeConfig.model_validate(document)


def tier_of_split(split: EpisodeSplit) -> EvalTier | None:
    """The acceptance tier a split belongs to, or ``None`` for train/dev."""

    tier = _TIER_OF_SPLIT.get(split)
    return tier  # type: ignore[return-value]
