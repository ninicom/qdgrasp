"""Which hands a workload is allowed to select (ADR-0008, G05).

ADR-0008 pauses the Shadow Hand from the active corpus. The pause is not a
deletion: the preset, the asset provenance and the compatibility tests stay, so
the decision can be reversed by another ADR rather than by an archaeology
project. What it does forbid is a *default* workload quietly selecting Shadow
and a release artifact then carrying it.

The failure this module exists to prevent (blocker B-10) is that the hand tuple
was written out by hand in a dozen places, so "the active corpus" meant whatever
each file happened to say. There is one list here, and every default resolves
through it.

Shadow can still be run, but only deliberately: an explicit ``experimental_shadow``
flag plus a stated diagnostic purpose, and whatever comes out is marked
``non_release`` and can never be folded into a release artifact.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterable, Sequence

#: The decision that governs this module. Written into every scope it resolves.
GOVERNING_DECISION = "ADR-0008"

#: Hands in the active corpus. Default workloads resolve to exactly these.
ACTIVE_HANDS: tuple[str, ...] = ("leap_hand", "wonik_allegro")

#: Hands kept, but out of every default workload, gate and release artifact.
PAUSED_HANDS: tuple[str, ...] = ("shadow_hand",)

#: Every hand the repository still carries a preset and assets for.
KNOWN_HANDS: tuple[str, ...] = (*ACTIVE_HANDS, *PAUSED_HANDS)

#: Robot profile filenames for the active corpus, in canonical order.
DEFAULT_ROBOT_PROFILES: tuple[str, ...] = tuple(f"{hand}.yaml" for hand in ACTIVE_HANDS)

#: Profile filenames of paused hands, for auditing a configuration.
PAUSED_ROBOT_PROFILES: tuple[str, ...] = tuple(f"{hand}.yaml" for hand in PAUSED_HANDS)


#: Artifacts that were generated and released **before** ADR-0008, whose
#: definitions are frozen because regenerating them has to reproduce the bytes
#: that were published. They may name a paused hand, but only by declaring the
#: artifact they reproduce, and nothing they produce is release evidence for the
#: active scope. This is an exemption for reproducing history, not a loophole
#: for new workloads: an undeclared default selecting a paused hand is still a
#: violation.
HISTORICAL_THREE_HAND_ARTIFACTS: frozenset[str] = frozenset(
    {
        "QDGrasp-Scene-Tiny",
        "phase3-2-recipe-ablation",
    }
)


class ScopeViolation(ValueError):
    """A workload tried to select a paused hand without saying so out loud."""


def hand_of_profile(profile: str) -> str:
    """``"leap_hand.yaml"`` -> ``"leap_hand"``; unknown names pass through."""
    return profile.removesuffix(".yaml")


def profile_of_hand(hand: str) -> str:
    return hand if hand.endswith(".yaml") else f"{hand}.yaml"


def is_paused(name: str) -> bool:
    return hand_of_profile(name) in PAUSED_HANDS


def is_active(name: str) -> bool:
    return hand_of_profile(name) in ACTIVE_HANDS


@dataclasses.dataclass(frozen=True)
class WorkloadScope:
    """The resolved hand selection for one workload, and what it may be used for."""

    hands: tuple[str, ...]
    experimental_shadow: bool
    purpose: str
    non_release: bool
    governing_decision: str = GOVERNING_DECISION

    @property
    def robot_profiles(self) -> tuple[str, ...]:
        return tuple(profile_of_hand(hand) for hand in self.hands)

    @property
    def coverage(self) -> str:
        active = [hand for hand in self.hands if is_active(hand)]
        return f"{len(active)}/{len(ACTIVE_HANDS)}_active"

    @property
    def three_hand_coverage(self) -> bool:
        # Kept as an explicit property so that nothing has to infer it from a
        # count: a two-hand artifact must never be read as the three-hand one.
        return False

    def as_disclosure(self) -> dict[str, object]:
        """The scope block every manifest and evidence record has to carry."""
        return {
            "active_hands": list(ACTIVE_HANDS),
            "paused_hands": list(PAUSED_HANDS),
            "selected_hands": list(self.hands),
            "coverage": self.coverage,
            "three_hand_coverage": False,
            "historical_p3_4_state": "paused_by_ADR-0008",
            "governing_decision": self.governing_decision,
            "experimental_shadow": self.experimental_shadow,
            "non_release": self.non_release,
            "purpose": self.purpose,
        }


def resolve_workload_hands(
    requested: Sequence[str] | None = None,
    *,
    experimental_shadow: bool = False,
    purpose: str = "",
) -> WorkloadScope:
    """Resolve the hands a workload runs on, or refuse to.

    ``requested=None`` is the default path and always yields the active corpus.
    A paused hand may only appear when ``experimental_shadow`` is set *and* a
    purpose is given, and the result is then marked ``non_release``.
    """
    if requested is None:
        selected = tuple(ACTIVE_HANDS)
    else:
        selected = tuple(hand_of_profile(str(name)) for name in requested)

    unknown = [hand for hand in selected if hand not in KNOWN_HANDS]
    if unknown:
        raise ScopeViolation(f"unknown hands {sorted(unknown)}; known hands are {list(KNOWN_HANDS)}")

    paused_selected = [hand for hand in selected if hand in PAUSED_HANDS]

    if experimental_shadow:
        if not purpose.strip():
            raise ScopeViolation(
                "experimental_shadow requires an explicit diagnostic purpose; a "
                "paused hand is not run by accident"
            )
        if requested is None:
            selected = tuple(KNOWN_HANDS)
            paused_selected = list(PAUSED_HANDS)
    elif paused_selected:
        raise ScopeViolation(
            f"{GOVERNING_DECISION} pauses {sorted(paused_selected)}; selecting it needs "
            "experimental_shadow=True and a stated diagnostic purpose, and the "
            "result is non-release"
        )

    return WorkloadScope(
        hands=selected,
        experimental_shadow=bool(experimental_shadow),
        purpose=purpose.strip(),
        non_release=bool(paused_selected),
    )


def require_release_scope(hands: Iterable[str]) -> tuple[str, ...]:
    """Assert a release artifact covers the active corpus and nothing paused."""
    selected = tuple(hand_of_profile(str(name)) for name in hands)
    paused_selected = sorted({hand for hand in selected if hand in PAUSED_HANDS})
    if paused_selected:
        raise ScopeViolation(
            f"release artifact carries paused hands {paused_selected}; "
            f"{GOVERNING_DECISION} forbids it while the pause holds"
        )
    missing = [hand for hand in ACTIVE_HANDS if hand not in selected]
    if missing:
        raise ScopeViolation(
            f"release artifact is missing active hands {missing}; the active scope is "
            f"{len(ACTIVE_HANDS)}/{len(ACTIVE_HANDS)}, not a subset of it"
        )
    return selected


def historical_reproduction_scope(artifact_id: str, hands: Sequence[str]) -> WorkloadScope:
    """Scope for regenerating a pre-ADR-0008 artifact exactly as published.

    The artifact has to be one of the declared historical ones. Anything it
    produces is ``non_release``: reproducing a three-hand artifact from before
    the pause does not give the pause an exception, and does not create new
    three-hand coverage.
    """
    if artifact_id not in HISTORICAL_THREE_HAND_ARTIFACTS:
        raise ScopeViolation(
            f"{artifact_id!r} is not a declared pre-{GOVERNING_DECISION} artifact; "
            f"declared artifacts are {sorted(HISTORICAL_THREE_HAND_ARTIFACTS)}"
        )
    selected = tuple(hand_of_profile(str(name)) for name in hands)
    unknown = [hand for hand in selected if hand not in KNOWN_HANDS]
    if unknown:
        raise ScopeViolation(f"unknown hands {sorted(unknown)} in historical artifact {artifact_id!r}")
    return WorkloadScope(
        hands=selected,
        experimental_shadow=False,
        purpose=f"historical reproduction of {artifact_id}",
        non_release=True,
    )


def audit_selection(profiles: Iterable[str], *, source: str) -> tuple[str, ...]:
    """Report paused hands found in a configuration, without raising.

    Used by gates that need to list every offending default at once rather than
    stop at the first one.
    """
    return tuple(
        f"{source}: selects paused hand {hand_of_profile(str(profile))!r} "
        f"without experimental_shadow ({GOVERNING_DECISION})"
        for profile in profiles
        if is_paused(str(profile))
    )
