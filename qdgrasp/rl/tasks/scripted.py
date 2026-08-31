"""Scripted fixtures for the acquisition environments (P3.5-11).

A scripted controller is not a policy and is not a baseline.  It exists so the
environment has a fixture whose outcome class is *expected*, which is what makes
a regression detectable: if the descend/close/lift script stops reaching the
class it reached before, something in the asset, scene, settle or control path
changed, and the test says so without anyone having to train anything.

The script drives the same bounded action the policy does.  It has no privileged
channel into the simulator, so whatever it achieves is achievable through the
action space -- which is the property worth asserting about an environment that
is about to be handed to a learner.

The descent ends on observed contact rather than on a step count.  A fixed count
is calibrated to one hand's geometry and drives the other into the target: with
an open-loop descent, LEAP acquires and lifts while Allegro sits five millimetres
inside the box for six consecutive control steps and trips the penetration
barrier.  Ending a phase when the physics says its precondition is met is the
same rule Phase 3.4's primitives use, and it is what makes one fixture valid for
two hands.

What this fixture does **not** do is complete an acquire.  Both active hands run
the full episode without tripping a barrier, and neither ends up holding the
target.  A seating phase -- continuing to descend past first contact so the
fingers end up alongside the object rather than on top of it -- was measured and
made things worse on both hands, so it is not in the spec.  The reason is
structural rather than a matter of tuning: descend-and-close does not enclose a
box.

That is what :mod:`qdgrasp.rl.tasks.grasp_prior` is for.  It fits an opposed
pinch to the target's measured width, places the palm on the target's own frame,
and drives the result through this same bounded action -- and both hands then
acquire and hold.  This module stays because the two fixtures answer different
questions: the open-loop one asks whether the environment is *stable* under a
dumb controller, and the prior-driven one asks whether it can be *solved* at all.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import Any

import numpy as np

from qdgrasp.rl.contracts import TerminalReason


@dataclasses.dataclass(frozen=True)
class ScriptedAcquireSpec:
    """Phase lengths, in control steps, and how hard the script squeezes."""

    #: Upper bound on the descent; contact normally ends it sooner.
    descend_steps: int = 60
    close_steps: int = 40
    lift_steps: int = 30
    hold_steps: int = 40
    #: Fraction of the palm translation limit used while descending.  Full rate
    #: is 0.4 m/s at this control period, which the safety budget correctly
    #: reads as an impact rather than an approach.
    descend_rate: float = 0.2
    #: Fraction of the joint delta limit used while closing.
    close_rate: float = 0.1
    #: Fraction of the palm translation limit used while lifting.
    lift_rate: float = 1.0

    @property
    def total_steps(self) -> int:
        return self.descend_steps + self.close_steps + self.lift_steps + self.hold_steps


class ScriptedAcquirePolicy:
    """Descend onto the target, close the fingers, lift, then hold.

    The phase is a function of the step index alone.  It reads the observation
    only to know how many joints there are, because a script that adapted to the
    state would be a controller, and a controller is what P5 is for.
    """

    def __init__(
        self,
        action_dimension: int,
        spec: ScriptedAcquireSpec | None = None,
        schema: Any | None = None,
    ) -> None:
        self.spec = spec or ScriptedAcquireSpec()
        self.action_dimension = action_dimension
        self.joint_dimension = action_dimension - 6
        if self.joint_dimension <= 0:
            raise ValueError("a scripted acquire needs a palm command and at least one joint")
        self._contact_slice = None if schema is None else schema.offset_of("fingertip_contact")
        self._step = 0
        self._descend_ended_at: int | None = None

    def reset(self) -> None:
        self._step = 0
        self._descend_ended_at = None

    def phase(self, step: int | None = None) -> str:
        index = self._step if step is None else step
        spec = self.spec
        descend_end = spec.descend_steps if self._descend_ended_at is None else self._descend_ended_at
        if index < descend_end:
            return "descend"
        if index < descend_end + spec.close_steps:
            return "close"
        if index < descend_end + spec.close_steps + spec.lift_steps:
            return "lift"
        return "hold"

    def __call__(self, observation: np.ndarray) -> np.ndarray:
        if (
            self._descend_ended_at is None
            and self._contact_slice is not None
            and self._step > 0
            and float(np.max(np.asarray(observation)[self._contact_slice])) > 0.0
        ):
            # First touch ends the descent.  Continuing to drive down past
            # contact is how an open-loop script buries a finger in the target.
            self._descend_ended_at = self._step
        action = np.zeros(self.action_dimension, dtype=np.float64)
        phase = self.phase()
        if phase == "descend":
            action[2] = -self.spec.descend_rate
        elif phase == "close":
            action[6:] = self.spec.close_rate
        elif phase == "lift":
            action[2] = self.spec.lift_rate
            action[6:] = self.spec.close_rate * 0.2
        else:
            action[6:] = self.spec.close_rate * 0.2
        self._step += 1
        return np.clip(action, -1.0, 1.0)


def run_scripted_episode(
    env: Any,
    *,
    seed: int,
    spec: ScriptedAcquireSpec | None = None,
) -> dict[str, Any]:
    """Run one scripted episode and report the measured outcome class."""

    observation, reset_info = env.reset(seed=seed)
    policy = ScriptedAcquirePolicy(env.action_space().shape[0], spec, schema=getattr(env, "schema", None))
    reward_total = 0.0
    steps = 0
    info: Mapping[str, Any] = {}
    terminated = truncated = False
    max_lift = 0.0
    finite = True
    while True:
        observation, reward, terminated, truncated, info = env.step(policy(observation))
        finite = finite and bool(np.all(np.isfinite(observation)))
        reward_total += float(reward)
        max_lift = max(max_lift, float(info.get("lift_m", 0.0)))
        steps += 1
        if terminated or truncated:
            break
    reason = info.get("terminal_reason", TerminalReason.NONE)
    return {
        "seed": seed,
        "steps": steps,
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "terminal_reason": reason.value if isinstance(reason, TerminalReason) else str(reason),
        "success": bool(info.get("success", False)),
        "max_lift_m": max_lift,
        "reward_total": reward_total,
        "observations_finite": finite,
        "scene_source": reset_info.get("scene_source"),
        "robot_profile": reset_info.get("robot_profile"),
        "observation_schema_hash": reset_info.get("observation_schema_hash"),
        "descend_ended_at": policy._descend_ended_at,
    }


def random_policy_probe(env: Any, *, seed: int, steps: int = 40) -> dict[str, Any]:
    """Run a uniform-random policy and report whether the environment stayed sane.

    The point is not the reward.  It is that a random policy must not produce a
    non-finite observation, must not be able to score a success, and must not
    crash the simulator -- three properties an environment has to have before a
    learner is pointed at it.
    """

    observation, _ = env.reset(seed=seed)
    rng = np.random.default_rng(seed ^ 0xA11CE)
    dimension = env.action_space().shape[0]
    finite = True
    successes = 0
    taken = 0
    for _ in range(steps):
        observation, _reward, terminated, truncated, info = env.step(rng.uniform(-1.0, 1.0, size=dimension))
        finite = finite and bool(np.all(np.isfinite(observation)))
        successes += int(bool(info.get("success", False)))
        taken += 1
        if terminated or truncated:
            break
    return {"steps": taken, "observations_finite": finite, "successes": successes}
