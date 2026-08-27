"""Which of a hand's own links are allowed to touch each other (C03.2).

A dexterous hand touches itself constantly, so self-contact cannot simply be
forbidden. The first version solved that by allowing the Cartesian product of
every robot geom with every other one, which is not a policy: it says yes to a
fingertip driven through the back of the palm exactly as readily as to two
fingers meeting in a pinch (blocker B-12).

The policy here is derived from the robot profile's own kinematic tree and
versioned, so it can be reviewed and hashed into a manifest:

* any link may touch the palm or base -- that is what grasping looks like;
* links on **different** fingers may touch -- that is what a pinch looks like;
* within one finger, only a link and its immediate parent may touch -- a finger
  folding onto a link three joints away has gone through itself, and a solver
  that permits it is reporting a penetration, not a grasp.

Anything else stays unlisted, which makes it a forbidden contact rather than a
silently accepted one.
"""

from __future__ import annotations

import dataclasses
import itertools
from collections.abc import Mapping

import mujoco

from qdgrasp.dataset.dynamic_contracts import canonical_hash

SELF_CONTACT_POLICY_SCHEMA = "qdgrasp/self-contact-policy/v1"


class SelfContactPolicyError(ValueError):
    """The profile does not describe a tree this policy can be derived from."""


@dataclasses.dataclass(frozen=True)
class SelfContactPolicy:
    """Link pairs this robot profile permits to touch, and why."""

    robot_profile: str
    schema: str
    allowed_link_pairs: frozenset[tuple[str, str]]
    finger_links: Mapping[str, tuple[str, ...]]
    root_links: tuple[str, ...]

    @property
    def policy_hash(self) -> str:
        return canonical_hash(
            {
                "schema": self.schema,
                "robot_profile": self.robot_profile,
                "allowed_link_pairs": sorted(self.allowed_link_pairs),
                "root_links": sorted(self.root_links),
            }
        )

    def permits(self, link_a: str, link_b: str) -> bool:
        return (min(link_a, link_b), max(link_a, link_b)) in self.allowed_link_pairs

    def as_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "robot_profile": self.robot_profile,
            "policy_hash": self.policy_hash,
            "allowed_link_pair_count": len(self.allowed_link_pairs),
            "fingers": {name: list(links) for name, links in self.finger_links.items()},
            "root_links": list(self.root_links),
        }


def _finger_chain(links: Mapping[str, object], tip: str, roots: frozenset[str]) -> tuple[str, ...]:
    """Walk from a fingertip up to the palm, exclusive of the palm."""
    chain: list[str] = []
    current: str | None = tip
    seen: set[str] = set()
    while current is not None and current not in roots:
        if current in seen:
            raise SelfContactPolicyError(f"link chain from {tip!r} contains a cycle")
        seen.add(current)
        chain.append(current)
        spec = links.get(current)
        current = getattr(spec, "parent_link", None) if spec is not None else None
    return tuple(reversed(chain))


def build_self_contact_policy(spec: object, *, robot_profile: str) -> SelfContactPolicy:
    """Derive the permitted self-contact pairs from a robot profile.

    ``spec`` is a :class:`~qdgrasp.robot.spec.RobotSpec`; it is taken loosely so
    that this module does not drag the whole robot package into the contact
    observer's import graph.
    """
    links: Mapping[str, object] = getattr(spec, "links", {})
    if not links:
        raise SelfContactPolicyError(f"{robot_profile}: profile declares no links")

    palm = getattr(spec, "palm_link", None)
    base = getattr(spec, "base_link", None)
    wrist = getattr(spec, "wrist_link", None)
    roots = frozenset(name for name in (palm, base, wrist) if name)
    if not roots:
        raise SelfContactPolicyError(f"{robot_profile}: profile declares no palm or base link")

    fingers: dict[str, tuple[str, ...]] = {}
    for tip in getattr(spec, "fingertip_links", ()):  # type: ignore[arg-type]
        chain = _finger_chain(links, str(tip), roots)
        if chain:
            fingers[str(tip)] = chain

    assigned = {link for chain in fingers.values() for link in chain}
    # Links that belong to no finger chain are structural: mounts, covers, the
    # palm itself. They behave like the palm for the purposes of this policy.
    structural = tuple(sorted(roots | {name for name in links if name not in assigned}))

    allowed: set[tuple[str, str]] = set()

    def allow(a: str, b: str) -> None:
        if a != b:
            allowed.add((min(a, b), max(a, b)))

    # Any link may touch a structural link.
    for link in links:
        for root in structural:
            allow(str(link), root)
    for a, b in itertools.combinations(structural, 2):
        allow(a, b)

    # Links on different fingers may touch.
    for (tip_a, chain_a), (tip_b, chain_b) in itertools.combinations(fingers.items(), 2):
        del tip_a, tip_b
        for link_a in chain_a:
            for link_b in chain_b:
                allow(link_a, link_b)

    # Within one finger, only immediate parent and child.
    for chain in fingers.values():
        for parent, child in itertools.pairwise(chain):
            allow(parent, child)

    return SelfContactPolicy(
        robot_profile=robot_profile,
        schema=SELF_CONTACT_POLICY_SCHEMA,
        allowed_link_pairs=frozenset(allowed),
        finger_links=dict(fingers),
        root_links=structural,
    )


def resolve_geom_allowlist(
    model: mujoco.MjModel,
    policy: SelfContactPolicy,
    robot_geoms: frozenset[int],
) -> frozenset[tuple[int, int]]:
    """Expand a link-pair policy into the geom pairs of a compiled model.

    Geoms on the same body are always allowed: MuJoCo does not report contacts
    between them, and refusing them would describe a contact that cannot happen.
    """
    body_of: dict[int, str] = {}
    for geom in robot_geoms:
        body_id = int(model.geom_bodyid[int(geom)])
        name = mujoco.mj_id2name(model, int(mujoco.mjtObj.mjOBJ_BODY), body_id)
        body_of[int(geom)] = name if name else f"body_{body_id}"

    pairs: set[tuple[int, int]] = set()
    for geom_a, geom_b in itertools.combinations(sorted(robot_geoms), 2):
        link_a, link_b = body_of[geom_a], body_of[geom_b]
        if link_a == link_b or policy.permits(link_a, link_b):
            pairs.add((min(geom_a, geom_b), max(geom_a, geom_b)))
    return frozenset(pairs)


def policy_coverage(
    model: mujoco.MjModel,
    policy: SelfContactPolicy,
    robot_geoms: frozenset[int],
) -> dict[str, float]:
    """How much of the Cartesian product this policy actually permits.

    Reported so that "we have a policy" can be checked rather than asserted: a
    policy that admits everything is the bug it was meant to fix.
    """
    total = len(robot_geoms) * (len(robot_geoms) - 1) // 2
    allowed = len(resolve_geom_allowlist(model, policy, robot_geoms))
    return {
        "robot_geoms": float(len(robot_geoms)),
        "candidate_pairs": float(total),
        "allowed_pairs": float(allowed),
        "allowed_fraction": float(allowed / total) if total else 0.0,
    }
