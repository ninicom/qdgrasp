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
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

MVP_SCOPE_SCHEMA_V0 = "qdgrasp/mvp-scope/v0"
MVP_SCOPE_SCHEMA_V1 = "qdgrasp/mvp-scope/v1"

#: Release class of the artifacts a scope produces, as a value a checker can
#: read rather than a sentence in a report.  ``experimental_non_release`` is the
#: v0 class and may never open a release gate; ``release_candidate`` is the only
#: class that may.  The distinction is the whole point of
#: ``ROADMAP-MVP-RELEASE-001`` §5 MR-02: an experimental gate that happens to
#: pass is not a release gate that passed.
EXPERIMENTAL_RELEASE_CLASS = "experimental_non_release"
RELEASE_CANDIDATE_CLASS = "release_candidate"
RELEASE_CLASSES: tuple[str, ...] = (EXPERIMENTAL_RELEASE_CLASS, RELEASE_CANDIDATE_CLASS)

#: Tiers of the acceptance gate (``ROADMAP-MVP-001`` §7).  ``D`` is the
#: challenge tier added by the release contract: the domain where the
#: controller prior is not saturated, and therefore the only tier on which a
#: learned residual can be shown to contribute anything.
EvalTier = Literal["A", "B", "C", "D"]

#: Splits an episode sampler can be asked for.  ``dev`` drives hyperparameter
#: choice, ``train`` feeds the expert recorder and the PPO rollouts, and the
#: locked tiers are only ever run on a finished candidate.
EpisodeSplit = Literal["train", "dev", "eval_a", "eval_b", "eval_c", "eval_d"]

_TIER_OF_SPLIT: dict[str, str] = {"eval_a": "A", "eval_b": "B", "eval_c": "C", "eval_d": "D"}

#: Fields scope v1 introduces.  A v0 document has to keep hashing exactly as it
#: did before those fields existed -- the committed evaluation manifest and
#: three rounds of published evidence are pinned to that hash -- so they are
#: omitted from a v0 document rather than serialized as nulls.
_V1_SCOPE_FIELDS = ("challenge", "release")
_V1_TIER_FIELDS = ("min_uplift_pp", "min_paired_ci_lower", "challenge_domain")


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
    """One acceptance tier: its domain, its sample size and its gate.

    A tier is gated either on a *level* -- an absolute success rate the
    candidate must reach -- or on an *uplift* over the controller prior.  The
    challenge tier is the second kind: on a domain where the prior is not
    saturated there is no meaningful absolute floor to set in advance, and
    setting one anyway would be a threshold invented to be cleared.
    """

    tier: EvalTier
    episodes: int = Field(gt=0)
    membership: Literal["train", "heldout"]
    randomized: bool
    #: Absolute floor.  ``None`` only on a tier gated on uplift instead.
    min_success_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    #: Wilson 95% lower bound the tier must also clear, when the plan sets one.
    min_wilson_lower_bound: float | None = Field(default=None, ge=0.0, le=1.0)
    #: Percentage points the candidate must beat the controller prior by, on
    #: the same seeds, and the paired 95% CI lower bound that must exceed it.
    min_uplift_pp: float | None = Field(default=None, ge=0.0)
    min_paired_ci_lower: float | None = Field(default=None)
    #: A challenge tier draws its domain from a separately locked challenge
    #: document rather than from the scope's own randomization block, because
    #: that domain is calibrated after this scope is frozen.
    challenge_domain: bool = False

    @model_validator(mode="after")
    def _gated(self) -> EvalTierSpec:
        if self.min_success_rate is None and self.min_uplift_pp is None:
            raise ValueError(f"eval tier {self.tier} declares no gate at all")
        if (self.min_uplift_pp is None) != (self.min_paired_ci_lower is None):
            raise ValueError(
                f"eval tier {self.tier} must declare an uplift gate and its paired CI floor together"
            )
        if self.min_wilson_lower_bound is not None and self.min_success_rate is None:
            raise ValueError(f"eval tier {self.tier} sets a Wilson floor without a success floor")
        if self.challenge_domain and self.min_uplift_pp is None:
            raise ValueError(f"eval tier {self.tier} is a challenge tier without an uplift gate")
        return self


#: Axes a challenge domain may vary.  ``ROADMAP-MVP-RELEASE-001`` §5 MR-03
#: names these and forbids widening the search to raw meshes, clutter, another
#: hand or another observation, so the permitted set is a closed literal rather
#: than free text a later session could quietly extend.
ChallengeAxis = Literal[
    "half_width",
    "half_depth",
    "half_height",
    "yaw",
    "friction_slide",
    "density",
    "position_x",
    "position_y",
]


class ChallengeSpec(_Doc):
    """What a valid Tier D domain must be, written down before one is chosen.

    The domain itself is calibrated in MR-03 and lives in its own locked
    document, because this scope is frozen before that calibration starts.
    What is frozen *here* is the rule the calibration has to satisfy: which
    axes may move, how unsaturated the prior has to be for the tier to be able
    to show anything, how many failures must be available to measure, and how
    many development configurations may be tried before the answer is `NO-GO`.
    """

    axes: tuple[ChallengeAxis, ...]
    #: The controller prior must land inside this band on the challenge domain.
    #: Below it the domain is broken rather than hard; above it the prior is
    #: saturated and there is no headroom left for a residual to occupy.
    prior_success_band: tuple[float, float]
    #: Measurable failures the prior must produce, so an uplift has something
    #: to be an uplift over.
    min_prior_failures: int = Field(gt=0)
    #: Development configurations allowed before the attempt is abandoned.
    max_development_configurations: int = Field(gt=0)
    #: Seed root of the development-only exploration.  It is deliberately not
    #: the scope's ``seed_root``: nothing explored during calibration may share
    #: a seed with the tier that later judges the candidate.
    development_seed_root: str
    #: Repository path the locked challenge domain will be written to.
    domain_document: str

    @model_validator(mode="after")
    def _ordered(self) -> ChallengeSpec:
        if not self.axes:
            raise ValueError("a challenge domain must be allowed to vary at least one axis")
        if len(set(self.axes)) != len(self.axes):
            raise ValueError("challenge axes must be unique")
        low, high = self.prior_success_band
        if not 0.0 <= low < high <= 1.0:
            raise ValueError(f"prior success band is not an ordered interval in [0, 1]: {self.prior_success_band}")
        return self


class AblationSpec(_Doc):
    """How the residual proves it is the thing doing the work.

    A candidate can beat the prior for reasons that have nothing to do with
    what it learned -- a different clamp, a different filter state, luck on a
    seed set.  The ablation is the control: run the exact candidate trajectory
    contract with the learned residual switched off, and the improvement has to
    disappear.  A residual that has collapsed to zero also passes "no
    regression" tests, so its magnitude and saturation are reported rather than
    assumed.
    """

    #: The disabled-residual run is mandatory, not an option a session may skip.
    require_disabled_residual_run: bool
    #: With the residual off, the uplift that remains must be no larger than
    #: this.  Anything more means the improvement was not the model's.
    max_disabled_uplift_pp: float = Field(ge=0.0)
    #: A residual whose typical magnitude is below this has degenerated to the
    #: prior and is not a learned contribution.
    min_residual_magnitude: float = Field(gt=0.0)
    #: Fraction of commanded residual components allowed to sit on their bound.
    max_saturation_rate: float = Field(ge=0.0, le=1.0)


class SafetySpec(_Doc):
    """Counters that must be exactly zero, on every tier, in the locked run.

    They are written as bounds rather than assumed, so that "we did not widen
    the safety budget to rescue a success rate" is a checkable statement about
    a frozen document.
    """

    max_safety_violation: Literal[0]
    max_invalid_state: Literal[0]
    max_checkpoint_reload_mismatch: Literal[0]


class ReleaseCriteria(_Doc):
    """Selection, comparison, ablation and safety, locked before any training.

    Every number here has to exist before the run that it judges, otherwise it
    is a threshold chosen after seeing the result.  The paired comparison is
    specified down to the resample count and the seed for the same reason: a
    confidence interval is only evidence if it could not have been re-rolled.
    """

    #: Candidates may only be selected on train/dev/challenge-development
    #: evidence.  The locked tiers are read once, afterwards, by MR-05.
    candidate_evidence: Literal["development_only"]
    #: The v0 tolerance -- PPO promoted while up to two points below BC -- is
    #: not a release rule.  PPO is promoted only if it is at least as good as
    #: BC on every regression tier.
    ppo_promotion: Literal["at_least_bc_on_every_regression_tier"]
    #: Tiers on which the candidate may not lose paired successes to the prior.
    regression_tiers: tuple[EvalTier, ...]
    #: The tier the contribution claim rests on.
    contribution_tier: EvalTier
    paired_confidence: float = Field(gt=0.5, lt=1.0)
    paired_method: Literal["paired_bootstrap_percentile"]
    paired_resamples: int = Field(gt=0)
    paired_seed: int = Field(ge=0)
    ablation: AblationSpec
    safety: SafetySpec

    @model_validator(mode="after")
    def _disjoint(self) -> ReleaseCriteria:
        if not self.regression_tiers:
            raise ValueError("at least one regression tier is required")
        if len(set(self.regression_tiers)) != len(self.regression_tiers):
            raise ValueError("regression tiers must be unique")
        if self.contribution_tier in self.regression_tiers:
            raise ValueError(
                f"tier {self.contribution_tier} cannot be both the contribution tier and a regression tier"
            )
        return self


class MvpScopeConfig(_Doc):
    """The whole locked scope of one MVP version.

    Two schemas are readable.  ``v0`` is the experimental scope the first three
    evidence rounds were produced under, and it is frozen: it is loaded so that
    old artifacts can still be checked, never extended.  ``v1`` is the release
    contract -- a challenge tier, a written-down selection and comparison rule,
    an ablation and a safety budget -- and it is the only schema whose
    ``release_class`` may open a release gate.
    """

    schema_version: Literal["qdgrasp/mvp-scope/v0", "qdgrasp/mvp-scope/v1"] = Field(alias="schema")
    mvp_id: str
    environment_id: str
    artifact_id: str
    release_class: Literal["experimental_non_release", "release_candidate"]
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
    #: Release contract.  Absent on v0, required on v1.
    challenge: ChallengeSpec | None = None
    release: ReleaseCriteria | None = None

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
        if len(tiers) != len(set(tiers)):
            raise ValueError("eval tiers must be unique")
        if self.schema_version == MVP_SCOPE_SCHEMA_V0:
            self._validate_v0(tiers)
        else:
            self._validate_v1(tiers)
        retain_steps_required = self.success.retain_duration_s * self.episode.control_hz
        if self.episode.retain_steps < retain_steps_required:
            raise ValueError(
                "retain phase is shorter than the retain duration the success "
                f"predicate demands ({self.episode.retain_steps} < {retain_steps_required:g} steps)"
            )
        return self

    def _validate_v0(self, tiers: Sequence[str]) -> None:
        """v0 is closed: the release fields did not exist when it was frozen."""

        if self.release_class != EXPERIMENTAL_RELEASE_CLASS:
            raise ValueError(f"scope v0 must declare release_class '{EXPERIMENTAL_RELEASE_CLASS}'")
        if self.challenge is not None or self.release is not None:
            raise ValueError("scope v0 cannot carry a challenge or release contract; use scope v1")
        if sorted(tiers) != ["A", "B", "C"]:
            raise ValueError("scope v0 eval_tiers must define exactly tiers A, B and C")
        for spec in self.eval_tiers:
            if spec.min_success_rate is None:
                raise ValueError(f"scope v0 tier {spec.tier} must declare an absolute min_success_rate")
            if spec.min_uplift_pp is not None or spec.challenge_domain:
                raise ValueError(f"scope v0 tier {spec.tier} cannot declare a challenge or uplift gate")

    def _validate_v1(self, tiers: Sequence[str]) -> None:
        """v1 must carry the whole release contract, or it is not one."""

        if self.release_class != RELEASE_CANDIDATE_CLASS:
            raise ValueError(f"scope v1 must declare release_class '{RELEASE_CANDIDATE_CLASS}'")
        if self.challenge is None or self.release is None:
            raise ValueError("scope v1 requires both a challenge and a release contract")
        if sorted(tiers) != ["A", "B", "C", "D"]:
            raise ValueError("scope v1 eval_tiers must define exactly tiers A, B, C and D")
        challenge_tiers = [spec.tier for spec in self.eval_tiers if spec.challenge_domain]
        if challenge_tiers != [self.release.contribution_tier]:
            raise ValueError(
                "exactly one tier may draw the challenge domain, and it must be the declared "
                f"contribution tier: challenge={challenge_tiers}, contribution={self.release.contribution_tier}"
            )
        declared = set(self.release.regression_tiers) | {self.release.contribution_tier}
        if declared != set(tiers):
            raise ValueError(
                f"release criteria must classify every tier: tiers={sorted(tiers)}, classified={sorted(declared)}"
            )
        for spec in self.eval_tiers:
            if spec.tier in self.release.regression_tiers and spec.min_success_rate is None:
                raise ValueError(f"regression tier {spec.tier} must declare an absolute min_success_rate")

    # -- derived views ----------------------------------------------------

    @property
    def is_release_candidate(self) -> bool:
        """Whether artifacts produced under this scope may open a release gate."""

        return self.release_class == RELEASE_CANDIDATE_CLASS

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

        if split in ("train", "dev"):
            return self.train_variants
        tier = tier_of_split(split)
        if tier is None:
            raise KeyError(f"unknown split: {split}")
        membership = self.tier(tier).membership
        return self.train_variants if membership == "train" else self.heldout_variants

    # -- hashing and seeds ------------------------------------------------

    def to_document(self) -> dict[str, Any]:
        document = self.model_dump(by_alias=True, mode="json")
        if self.schema_version != MVP_SCOPE_SCHEMA_V0:
            return document
        for field in _V1_SCOPE_FIELDS:
            document.pop(field, None)
        for tier in document["eval_tiers"]:
            for field in _V1_TIER_FIELDS:
                tier.pop(field, None)
        return document

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

        ordered = sorted(self.eval_tiers, key=lambda item: item.tier)
        entries: list[dict[str, Any]] = [
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
            for spec in ordered
        ]
        document: dict[str, Any] = {
            "schema": EVAL_MANIFEST_SCHEMA[self.schema_version],
            "mvp_id": self.mvp_id,
            "environment_id": self.environment_id,
            "scope_hash": self.content_hash(),
            "tiers": entries,
        }
        if self.schema_version == MVP_SCOPE_SCHEMA_V0:
            return document
        # A v1 manifest carries the uplift gate beside the level gate, so a
        # reader cannot mistake a challenge tier for one with no threshold.
        for entry, spec in zip(entries, ordered, strict=True):
            entry["min_uplift_pp"] = spec.min_uplift_pp
            entry["min_paired_ci_lower"] = spec.min_paired_ci_lower
            entry["challenge_domain"] = spec.challenge_domain
        document["release_class"] = self.release_class
        return document

    def eval_manifest_hash(self) -> str:
        payload = json.dumps(self.eval_manifest(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


#: The eval manifest schema each scope schema emits.  A v1 manifest is not a
#: v0 manifest with extra keys: the tier rows carry the uplift gate, so a
#: consumer that only understands v0 must refuse it rather than ignore them.
EVAL_MANIFEST_SCHEMA = {
    MVP_SCOPE_SCHEMA_V0: "qdgrasp/mvp-eval-manifest/v0",
    MVP_SCOPE_SCHEMA_V1: "qdgrasp/mvp-eval-manifest/v1",
}

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
