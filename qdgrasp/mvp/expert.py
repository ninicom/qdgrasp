"""MVP-02: an expert that beats the prior, and the ledger that admits what it cost.

The controller prior is already good, which creates a problem for behaviour
cloning: cloning a zero residual teaches a policy to output zero, and a policy
that outputs zero has learned nothing the prior did not already know.

So the expert here is a small per-episode search over the residual, not the
prior itself.  The residual is piecewise constant over two segments, before and
after the hand closes, which is enough for the search to say things like "this
target is heavy and slippery, grip harder through the lift" without pretending
to be a planner.

The expert is **minimum-intervention**: the zero residual is tried first, and if
the prior alone acquires the target then zero *is* the demonstration and the
search stops.  Only when the prior fails does the search look for a residual
that rescues the episode.

That rule was not the first design, and the reason for it is measured.  Round
one ranked every successful candidate by margin and cloned the winner, which
labelled 97% of transitions with a large arbitrary constant.  Behaviour cloning
fitted it perfectly -- train loss 0.0014, validation 0.0021 -- by learning the
shortcut the label handed it: the expert's residual is constant within a
segment and the previous action is in the observation, so `a_t = a_{t-1}` fits
almost everything.  At rollout time that shortcut is an unanchored random walk;
measured autocorrelation between consecutive actions was 0.98-0.997, the actions
drifted to +/-0.6, and locked-eval success collapsed to 37/34.7/23.5% against a
prior that scores 100/89.7/88.5%.  A label that is mostly zero cannot teach that
shortcut, because following it reproduces the prior.

Demonstrations are also collected under injected action noise.  Round two fixed
the labels and still measured 6.7% on dev, because the remaining failure was not
the labels at all: every demonstrated state lay exactly on the expert's own
trajectory, so the first millimetre of error put the policy somewhere the data
never went, and its output grew from 0.005 at the first control step to 0.9 by
the fortieth.  Rolling the accepted residual out again with noise on the applied
action, while labelling each state with the noise-free residual, covers a tube
around the expert path and teaches the policy what to do after a small mistake --
including not to copy its own previous action, which under noise no longer
predicts the label.

Only physically successful episodes become demonstrations, noisy ones included:
a noisy rollout that loses the target is discarded rather than labelled.
Everything the search tried and failed is still counted, in the generator
ledger, because a demonstration set whose rejection rate is invisible is a
demonstration set nobody can calibrate.
"""

from __future__ import annotations

import dataclasses
import json
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from qdgrasp.mvp.config import EpisodeSplit, MvpScopeConfig
from qdgrasp.mvp.env import DexAcquireMvpEnv, EpisodeResult

DEMONSTRATION_SCHEMA_V0 = "qdgrasp/mvp-demonstrations/v0"

#: The two segments the searched residual is constant over: everything up to
#: and including the closing schedule, then the lift and the hold.
SEGMENT_NAMES: tuple[str, ...] = ("pregrasp", "carry")


@dataclasses.dataclass(frozen=True)
class ExpertSearchSpec:
    """How hard the expert is allowed to look for a better residual."""

    candidates: int = 12
    #: Standard deviation of the candidate residual, in unit-action space.
    sigma: float = 0.35
    #: Candidates are drawn per dimension; the palm-rotation entries are kept
    #: small because a large rotation residual mostly just misses the target.
    rotation_scale: float = 0.3
    #: Extra rollouts of the accepted residual under injected action noise.
    noise_rollouts: int = 3
    #: Standard deviation of that noise, in unit-action space.  It has to be
    #: comparable to the errors a cloned policy actually makes, or the tube it
    #: covers is too thin to catch them.
    noise_sigma: float = 0.15

    def sample(self, rng: np.random.Generator, action_dim: int) -> np.ndarray:
        """Draw the candidate set, with the zero (prior) candidate first."""

        draws = rng.normal(0.0, self.sigma, size=(self.candidates - 1, len(SEGMENT_NAMES), action_dim))
        draws[:, :, 3:6] *= self.rotation_scale
        candidates = np.concatenate([np.zeros((1, len(SEGMENT_NAMES), action_dim)), draws], axis=0)
        return np.clip(candidates, -1.0, 1.0)


@dataclasses.dataclass
class ExpertEpisode:
    """One accepted demonstration: what was seen, what was done, what happened.

    ``actions`` is what the *expert* intended at each state, which under noise
    injection is not the action that was applied.  Cloning the applied action
    would teach the policy to reproduce the noise.
    """

    observations: np.ndarray  # [T, D] float32
    actions: np.ndarray  # [T, A] float32
    result: EpisodeResult
    candidate_index: int
    score: tuple[float, ...]

    def __len__(self) -> int:
        return int(self.observations.shape[0])


def _segment_of(step_index: int, scope: MvpScopeConfig) -> int:
    """Which residual segment a control step belongs to."""

    closed = scope.episode.approach_steps + scope.episode.enclose_steps
    return 0 if step_index < closed else 1


def _score(result: EpisodeResult, scope: MvpScopeConfig) -> tuple[float, ...]:
    """Rank candidates by margin, not by whether they scraped through.

    Two candidates that both succeed are not equally good demonstrations: the
    one that held longer, lifted higher and pressed less is the one worth
    cloning.  Force enters negated, so a candidate cannot buy hold time by
    crushing the target.
    """

    return (
        1.0 if result.success else 0.0,
        float(result.hold_steps),
        float(result.max_lift_m - scope.success.lift_height_m),
        -float(result.max_contact_force_n),
    )


def search_expert_episode(
    env: DexAcquireMvpEnv,
    seed: int,
    split: EpisodeSplit,
    spec: ExpertSearchSpec,
) -> tuple[ExpertEpisode | None, dict[str, Any]]:
    """Search residual candidates on one episode and return the best success.

    The second element is the per-episode ledger row, written whether or not a
    demonstration came out of it.
    """

    scope = env.scope
    action_dim = scope.action.dimension
    rng = np.random.default_rng(seed ^ 0x5EED)
    candidates = spec.sample(rng, action_dim)
    setup = env.sample_setup(seed, split)

    best: ExpertEpisode | None = None
    best_score: tuple[float, ...] | None = None
    prior_succeeded = False
    outcomes: Counter[str] = Counter()
    for index, candidate in enumerate(candidates):
        if index > 0 and prior_succeeded:
            # Minimum intervention: the prior already acquired the target, so
            # the demonstration is "do nothing" and the remaining candidates
            # would only be looking for a prettier way to do the same thing.
            break
        observations: list[np.ndarray] = []
        actions: list[np.ndarray] = []
        observation = env.reset(seed, split, setup=setup)
        while not env.done:
            action = candidate[_segment_of(env.step_index, scope)]
            observations.append(observation)
            actions.append(action)
            observation, _, _, _ = env.step(action)
        result = env.result
        assert result is not None
        outcomes[result.failure_bucket] += 1
        if index == 0:
            # Candidate zero is the prior itself, so the ledger gets the
            # prior's own verdict without paying for a second rollout.
            prior_succeeded = bool(result.success)
        if not result.success:
            continue
        score = _score(result, scope)
        if best_score is None or score > best_score:
            best_score = score
            best = ExpertEpisode(
                observations=np.stack(observations).astype(np.float32),
                actions=np.stack(actions).astype(np.float32),
                result=result,
                candidate_index=index,
                score=score,
            )

    noisy_attempted = 0
    noisy_accepted = 0
    if best is not None and spec.noise_rollouts > 0:
        extra_observations: list[np.ndarray] = [best.observations]
        extra_actions: list[np.ndarray] = [best.actions]
        candidate = candidates[best.candidate_index]
        for trial in range(spec.noise_rollouts):
            noisy_attempted += 1
            noise_rng = np.random.default_rng((seed ^ 0xDA27) + trial)
            observations = []
            actions = []
            observation = env.reset(seed, split, setup=setup)
            while not env.done:
                intended = candidate[_segment_of(env.step_index, scope)]
                applied = np.clip(intended + noise_rng.normal(0.0, spec.noise_sigma, action_dim), -1.0, 1.0)
                observations.append(observation)
                actions.append(intended)
                observation, _, _, _ = env.step(applied)
            result = env.result
            assert result is not None
            outcomes[f"noisy_{result.failure_bucket}"] += 1
            if not result.success:
                continue
            noisy_accepted += 1
            extra_observations.append(np.stack(observations).astype(np.float32))
            extra_actions.append(np.stack(actions).astype(np.float32))
        best = dataclasses.replace(
            best,
            observations=np.concatenate(extra_observations),
            actions=np.concatenate(extra_actions),
        )

    row = {
        "seed": int(seed),
        "split": split,
        "variant_id": setup.variant_id,
        "mass": setup.mass,
        "friction_slide": setup.friction_slide,
        "candidates": int(spec.candidates),
        "candidate_outcomes": dict(sorted(outcomes.items())),
        "accepted": best is not None,
        "accepted_candidate": None if best is None else int(best.candidate_index),
        "prior_candidate_succeeded": prior_succeeded,
        "noisy_rollouts_attempted": noisy_attempted,
        "noisy_rollouts_accepted": noisy_accepted,
    }
    return best, row


@dataclasses.dataclass
class DemonstrationSet:
    """Flattened demonstrations plus the ledger that explains their provenance."""

    observations: np.ndarray  # [N, D]
    actions: np.ndarray  # [N, A]
    episode_index: np.ndarray  # [N]
    variant_ids: tuple[str, ...]
    seeds: tuple[int, ...]
    ledger: list[dict[str, Any]]

    @property
    def episodes(self) -> int:
        return len(self.seeds)

    def summary(self) -> dict[str, Any]:
        attempted = len(self.ledger)
        accepted = sum(1 for row in self.ledger if row["accepted"])
        prior_only = sum(1 for row in self.ledger if row.get("prior_candidate_succeeded"))
        improved = sum(1 for row in self.ledger if row["accepted"] and not row.get("prior_candidate_succeeded"))
        return {
            "schema": DEMONSTRATION_SCHEMA_V0,
            "episodes_attempted": attempted,
            "episodes_accepted": accepted,
            "acceptance_rate": accepted / attempted if attempted else 0.0,
            "prior_alone_succeeded": prior_only,
            "search_rescued": improved,
            "transitions": int(self.observations.shape[0]),
            "variants": dict(sorted(Counter(self.variant_ids).items())),
            "noisy_rollouts_attempted": sum(int(row.get("noisy_rollouts_attempted", 0)) for row in self.ledger),
            "noisy_rollouts_accepted": sum(int(row.get("noisy_rollouts_accepted", 0)) for row in self.ledger),
            "non_zero_residual_fraction": float(
                np.mean(np.any(np.abs(self.actions) > 1e-6, axis=1)) if self.actions.size else 0.0
            ),
        }

    def save(self, directory: str | Path) -> Path:
        target = Path(directory)
        target.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            target / "demonstrations.npz",
            observations=self.observations,
            actions=self.actions,
            episode_index=self.episode_index,
        )
        (target / "ledger.jsonl").write_text(
            "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in self.ledger),
            encoding="utf-8",
        )
        (target / "summary.json").write_text(
            json.dumps(self.summary(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return target

    @classmethod
    def load(cls, directory: str | Path) -> DemonstrationSet:
        source = Path(directory)
        arrays = np.load(source / "demonstrations.npz")
        ledger = [
            json.loads(line)
            for line in (source / "ledger.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        accepted = [row for row in ledger if row["accepted"]]
        return cls(
            observations=arrays["observations"],
            actions=arrays["actions"],
            episode_index=arrays["episode_index"],
            variant_ids=tuple(row["variant_id"] for row in accepted),
            seeds=tuple(int(row["seed"]) for row in accepted),
            ledger=ledger,
        )


def collect_demonstrations(
    env: DexAcquireMvpEnv,
    seeds: Sequence[int],
    split: EpisodeSplit,
    spec: ExpertSearchSpec | None = None,
) -> DemonstrationSet:
    """Run the expert search over a list of seeds and assemble the dataset."""

    search = spec or ExpertSearchSpec()
    observations: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    episode_index: list[np.ndarray] = []
    variant_ids: list[str] = []
    accepted_seeds: list[int] = []
    ledger: list[dict[str, Any]] = []
    for seed in seeds:
        episode, row = search_expert_episode(env, seed, split, search)
        ledger.append(row)
        if episode is None:
            continue
        index = len(accepted_seeds)
        observations.append(episode.observations)
        actions.append(episode.actions)
        episode_index.append(np.full(len(episode), index, dtype=np.int64))
        variant_ids.append(episode.result.setup.variant_id)
        accepted_seeds.append(int(seed))
    if not observations:
        raise RuntimeError("the expert search accepted no episodes; the controller prior is broken")
    return DemonstrationSet(
        observations=np.concatenate(observations),
        actions=np.concatenate(actions),
        episode_index=np.concatenate(episode_index),
        variant_ids=tuple(variant_ids),
        seeds=tuple(accepted_seeds),
        ledger=ledger,
    )
