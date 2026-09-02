"""The locked challenge domain Tier D is drawn from.

``ROADMAP-MVP-RELEASE-001`` §5 MR-02 freezes the scope before MR-03 calibrates
this domain, so the two cannot live in the same document.  What the scope holds
is the rule -- which axes may move, how unsaturated the controller prior has to
be, how many failures must be measurable.  What this document holds is the one
domain that was found to satisfy that rule, and its hash is what binds a Tier D
result to the domain it was actually measured on.

A challenge domain narrows the scope; it never widens it.  Every declared range
must sit inside the scope's own randomization, and every declared axis must be
one the scope authorised.  A "harder domain" that reached outside those bounds
would be a different world, and a tier measured in it could not be compared to
tiers A, B and C at all.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from qdgrasp.mvp.config import ChallengeAxis, MvpScopeConfig, ObjectVariant
from qdgrasp.mvp.contracts import CHALLENGE_DOMAIN_SCHEMA

#: Axes that narrow a per-episode randomization range.
CONTINUOUS_AXES: tuple[str, ...] = ("position_x", "position_y", "yaw", "density", "friction_slide")
#: Axes that narrow the object set, by selecting the variants whose extents fall
#: inside the declared interval.  The scope's object family stays locked: this
#: chooses among its members, it does not invent new ones.
EXTENT_AXES: tuple[str, ...] = ("half_width", "half_depth", "half_height")


class ChallengeDomain(BaseModel):
    """One challenge domain, frozen and hashed."""

    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)

    schema_version: str = Field(alias="schema")
    #: The scope this domain narrows.  A domain is meaningless without it.
    scope_hash: str
    #: Which development configuration this was, for the audit trail.
    configuration_id: str
    #: Inclusive intervals, keyed by axis.  An axis that is absent keeps the
    #: scope's own range.
    axes: dict[ChallengeAxis, tuple[float, float]]

    @model_validator(mode="after")
    def _ordered(self) -> ChallengeDomain:
        if self.schema_version != CHALLENGE_DOMAIN_SCHEMA:
            raise ValueError(f"unsupported challenge domain schema: {self.schema_version!r}")
        if not self.axes:
            raise ValueError("a challenge domain must narrow at least one axis")
        for axis, (low, high) in self.axes.items():
            if not low <= high:
                raise ValueError(f"challenge axis {axis} is inverted: {low} > {high}")
        return self

    def validate_against(self, scope: MvpScopeConfig) -> None:
        """Refuse a domain that is not a narrowing of this exact scope."""

        if scope.challenge is None:
            raise ValueError("this scope declares no challenge contract")
        if self.scope_hash != scope.content_hash():
            raise ValueError(
                f"challenge domain is bound to another scope: domain={self.scope_hash}, scope={scope.content_hash()}"
            )
        unauthorised = sorted(set(self.axes) - set(scope.challenge.axes))
        if unauthorised:
            raise ValueError(f"challenge domain moves axes the scope did not authorise: {unauthorised}")
        for axis in self.axes:
            if axis in CONTINUOUS_AXES:
                low, high = self.axes[axis]
                base_low, base_high = getattr(scope.randomization, axis)
                if low < base_low or high > base_high:
                    raise ValueError(
                        f"challenge axis {axis} reaches outside the locked scope: "
                        f"[{low}, {high}] not within [{base_low}, {base_high}]"
                    )
        if not self.variants(scope):
            raise ValueError("the challenge domain's extent bounds select no object variant")

    def variants(self, scope: MvpScopeConfig) -> tuple[ObjectVariant, ...]:
        """The train variants whose extents fall inside the declared bounds."""

        selected = []
        for variant in scope.train_variants:
            if all(
                self.axes[axis][0] <= getattr(variant, axis) <= self.axes[axis][1]
                for axis in EXTENT_AXES
                if axis in self.axes
            ):
                selected.append(variant)
        return tuple(selected)

    def range_for(self, axis: str, scope: MvpScopeConfig) -> tuple[float, float]:
        """The interval one continuous axis is drawn from under this domain."""

        if axis in self.axes:
            return self.axes[axis]  # type: ignore[index]
        return tuple(getattr(scope.randomization, axis))  # type: ignore[return-value]

    def to_document(self) -> dict[str, Any]:
        return self.model_dump(by_alias=True, mode="json")

    def content_hash(self) -> str:
        payload = json.dumps(self.to_document(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def challenge_development_seeds(scope: MvpScopeConfig, count: int) -> list[int]:
    """Seeds for calibrating and selecting on the challenge domain.

    Derived from the scope's ``challenge.development_seed_root``, which is
    deliberately not the root Tier D draws from: nothing explored during
    calibration, and nothing a candidate was selected on, may appear in the
    tier that later judges that candidate.
    """

    if scope.challenge is None:
        raise ValueError("this scope declares no challenge contract")
    if count < 0:
        raise ValueError("episode count must be non-negative")
    root = scope.challenge.development_seed_root
    return [
        int.from_bytes(
            hashlib.sha256(f"{root}|{scope.mvp_id}|challenge_dev|{index}".encode()).digest()[:8],
            "big",
        )
        >> 1
        for index in range(count)
    ]


def load_challenge_domain(path: str | Path, scope: MvpScopeConfig | None = None) -> ChallengeDomain:
    """Load a challenge domain, and check it narrows ``scope`` when given."""

    resolved = Path(path)
    if not resolved.is_file():
        raise FileNotFoundError(f"challenge domain document not found: {resolved}")
    document = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise TypeError(f"challenge domain document must be a mapping: {resolved}")
    domain = ChallengeDomain.model_validate(document)
    if scope is not None:
        domain.validate_against(scope)
    return domain
