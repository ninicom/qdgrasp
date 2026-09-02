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
import hashlib
import json
import math
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from qdgrasp.mvp.config import EpisodeSplit, MvpScopeConfig
from qdgrasp.mvp.env import DexAcquireMvpEnv, EpisodeResult

DEMONSTRATION_SCHEMA_V0 = "qdgrasp/mvp-demonstrations/v0"
DEMONSTRATION_SCHEMA_V1 = "qdgrasp/mvp-demonstrations/v1"
DEMONSTRATION_SCHEMA = DEMONSTRATION_SCHEMA_V1
DEMONSTRATION_MANIFEST_SCHEMA = "qdgrasp/mvp-demonstration-content/v1"
DEMONSTRATION_INDEX_SCHEMA = "qdgrasp/mvp-demonstration-index/v1"

#: The two segments the searched residual is constant over: everything up to
#: and including the closing schedule, then the lift and the hold.
SEGMENT_NAMES: tuple[str, ...] = ("pregrasp", "carry")

#: Where the two synergy (finger closure) entries sit in the action vector.
#: The layout is fixed by ``ActionSpec.scale_vector``: three translation, three
#: rotation, two synergy.
SYNERGY_ACTION_SLICE = slice(6, 8)


@dataclasses.dataclass(frozen=True)
class ExpertSearchSpec:
    """How hard the expert is allowed to look for a better residual.

    The search is part random and part directed, and the directed part exists
    because the random part could not find the answer.  Sampling isotropic
    noise in a sixteen-dimensional space almost never produces "hold the grip
    harder once the hand has closed", and that is the correction the measured
    failures call for: on the challenge domain the prior loses targets to
    ``contact_loss`` and ``drop`` under low friction and high mass, and
    ``scripts/probe_mvp_residual.py`` shows a constant closing residual
    recovering a quarter of them with no safety violation and peak forces
    around half the budget.

    So the grip candidates are tried first, the random draws keep the search
    able to find what the structure did not anticipate, and the zero candidate
    stays first of all so a working grasp is still left alone.  Nothing here
    changes what counts as a success: candidates are still admitted only by
    the measured predicate, and still ranked by a score that negates force so
    a candidate cannot buy hold time by crushing the target.
    """

    candidates: int = 12
    #: Standard deviation of the random candidate residual, in unit-action space.
    sigma: float = 0.35
    #: Candidates are drawn per dimension; the palm-rotation entries are kept
    #: small because a large rotation residual mostly just misses the target.
    rotation_scale: float = 0.3
    #: Closing residuals applied to the carry segment only, as (first group,
    #: second group) on the prior's own open-to-squeeze direction.  The approach
    #: is left alone on purpose: it spreads the fingers wider than the grasp
    #: needs so a descending fingertip clears the target's side, and closing
    #: during it is the jam the controller prior was shaped to avoid.
    grip_candidates: tuple[tuple[float, float], ...] = (
        (0.0, 0.75),
        (0.25, 0.75),
        (0.5, 0.5),
        (1.0, 0.0),
        (0.0, 0.40),
        (0.40, 0.0),
    )
    #: How many of the best-scoring successes are re-run under noise before
    #: one is kept, when no earlier candidate has already been accepted.
    robustness_finalists: int = 3
    #: Noisy rollouts an earlier candidate must survive to be taken without
    #: looking at the later ones.
    #:
    #: This is what makes the expert *consistent*, and consistency is the
    #: property a clone actually needs.  Ranking every success by margin makes
    #: the expert an oracle that picks a different residual per episode --
    #: measured, half the rescued episodes were carried by scattered random
    #: draws -- and cloning inconsistent labels for similar states returns
    #: their mean, which is small and points nowhere.  Taking the earliest
    #: candidate that robustly works means the structured closing correction
    #: is used wherever it works, so the same state maps to the same answer.
    accept_first_robust: bool = True
    #: Extra rollouts of the accepted residual under injected action noise.
    noise_rollouts: int = 3
    #: Standard deviation of that noise, in unit-action space.  It has to be
    #: comparable to the errors a cloned policy actually makes, or the tube it
    #: covers is too thin to catch them.
    noise_sigma: float = 0.15

    def to_document(self) -> dict[str, Any]:
        """Return the exact generator configuration carried by the index."""

        return dataclasses.asdict(self)

    def sample(self, rng: np.random.Generator, action_dim: int) -> np.ndarray:
        """Draw the candidate set: zero first, then grip, then random."""

        # ``candidates`` is the whole search budget, zero candidate included:
        # the grip set is drawn from it rather than added on top, so a spec
        # that asks for four rollouts pays for four.
        used = self.grip_candidates[: max(0, self.candidates - 1)]
        grip = np.zeros((len(used), len(SEGMENT_NAMES), action_dim))
        for index, pair in enumerate(used):
            grip[index, 1, SYNERGY_ACTION_SLICE] = pair
        random_count = max(0, self.candidates - 1 - len(used))
        draws = rng.normal(0.0, self.sigma, size=(random_count, len(SEGMENT_NAMES), action_dim))
        draws[:, :, 3:6] *= self.rotation_scale
        candidates = np.concatenate([np.zeros((1, len(SEGMENT_NAMES), action_dim)), grip, draws], axis=0)
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
    *,
    challenged: bool | None = None,
) -> tuple[ExpertEpisode | None, dict[str, Any]]:
    """Search residual candidates on one episode and return the best success.

    The second element is the per-episode ledger row, written whether or not a
    demonstration came out of it.
    """

    scope = env.scope
    action_dim = scope.action.dimension
    rng = np.random.default_rng(seed ^ 0x5EED)
    candidates = spec.sample(rng, action_dim)
    setup = env.sample_setup(seed, split, challenged=challenged)

    successes: list[ExpertEpisode] = []
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
        successes.append(
            ExpertEpisode(
                observations=np.stack(observations).astype(np.float32),
                actions=np.stack(actions).astype(np.float32),
                result=result,
                candidate_index=index,
                score=_score(result, scope),
            )
        )

    if prior_succeeded:
        # The prior needs no run-off: "do nothing" is the demonstration, and
        # there is nothing to be robust about.
        finalists = successes[:1]
    elif spec.accept_first_robust:
        # Candidate order is the preference order: zero, then the structured
        # closing corrections, then the random draws.
        finalists = successes
    else:
        successes.sort(key=lambda episode: episode.score, reverse=True)
        finalists = successes[: max(1, spec.robustness_finalists)]

    required = math.ceil(spec.noise_rollouts / 2) if spec.accept_first_robust else spec.noise_rollouts + 1
    best: ExpertEpisode | None = None
    best_rank: tuple[int, tuple[float, ...]] | None = None
    noisy_attempted = 0
    noisy_accepted = 0
    winning_rollouts: list[tuple[np.ndarray, np.ndarray]] = []
    for finalist in finalists:
        candidate = candidates[finalist.candidate_index]
        survived: list[tuple[np.ndarray, np.ndarray]] = []
        for trial in range(spec.noise_rollouts):
            noisy_attempted += 1
            noise_rng = np.random.default_rng((seed ^ 0xDA27) + trial + 0x100 * finalist.candidate_index)
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
            if result.success:
                survived.append(
                    (np.stack(observations).astype(np.float32), np.stack(actions).astype(np.float32))
                )
        # Robustness first, margin only to break ties: the demonstration kept
        # is the one that still works when the action is wrong, because a
        # cloned policy's action is always a little wrong.
        rank = (len(survived), finalist.score)
        if best_rank is None or rank > best_rank:
            best_rank = rank
            best = finalist
            winning_rollouts = survived
        if len(survived) >= required:
            # Good enough, and earlier in the preference order than anything
            # left: stop rather than shop for a prettier label.
            break
    if best is not None:
        noisy_accepted = len(winning_rollouts)
        best = dataclasses.replace(
            best,
            observations=np.concatenate([best.observations, *(item[0] for item in winning_rollouts)]),
            actions=np.concatenate([best.actions, *(item[1] for item in winning_rollouts)]),
        )

    row = {
        "seed": int(seed),
        "split": split,
        # Whether this episode was drawn from the challenge domain.  A
        # demonstration set that mixes the two has to say which is which, or
        # its acceptance rate cannot be read.
        "challenged": bool(setup.challenged),
        "variant_id": setup.variant_id,
        "mass": setup.mass,
        "friction_slide": setup.friction_slide,
        "candidates": int(spec.candidates),
        "candidate_outcomes": dict(sorted(outcomes.items())),
        "accepted": best is not None,
        "accepted_candidate": None if best is None else int(best.candidate_index),
        "finalists": len(finalists),
        "noisy_survivals": noisy_accepted,
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

    @staticmethod
    def _array_record(array: np.ndarray) -> dict[str, Any]:
        value = np.ascontiguousarray(array)
        return {
            "dtype": value.dtype.str,
            "shape": list(value.shape),
            "sha256": hashlib.sha256(value.tobytes(order="C")).hexdigest(),
        }

    @staticmethod
    def _canonical_hash(document: Any) -> str:
        encoded = json.dumps(
            document,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def content_manifest(self) -> dict[str, Any]:
        """Describe and hash every training-relevant demonstration field.

        The compressed ``.npz`` bytes are deliberately not the identity: zip
        metadata can change while the arrays do not.  Dtype, shape and raw
        contiguous bytes are hashed instead, together with the ordered ledger
        and the accepted episode identities reconstructed from it.
        """

        body = {
            "schema": DEMONSTRATION_MANIFEST_SCHEMA,
            "dataset_schema": DEMONSTRATION_SCHEMA,
            "arrays": {
                "actions": self._array_record(self.actions),
                "episode_index": self._array_record(self.episode_index),
                "observations": self._array_record(self.observations),
            },
            "ledger": {
                "rows": len(self.ledger),
                "sha256": self._canonical_hash(self.ledger),
            },
            "accepted_identity": {
                "episodes": self.episodes,
                "seeds_sha256": self._canonical_hash(list(self.seeds)),
                "variant_ids_sha256": self._canonical_hash(list(self.variant_ids)),
            },
        }
        return {**body, "content_hash": self._canonical_hash(body)}

    def content_hash(self) -> str:
        """Return the digest that binds arrays, ledger and episode identity."""

        return str(self.content_manifest()["content_hash"])

    def summary(self) -> dict[str, Any]:
        attempted = len(self.ledger)
        accepted = sum(1 for row in self.ledger if row["accepted"])
        prior_only = sum(1 for row in self.ledger if row.get("prior_candidate_succeeded"))
        improved = sum(1 for row in self.ledger if row["accepted"] and not row.get("prior_candidate_succeeded"))
        return {
            "schema": DEMONSTRATION_SCHEMA,
            "content_hash": self.content_hash(),
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
        (target / "manifest.json").write_text(
            json.dumps(self.content_manifest(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return target

    @classmethod
    def load(cls, directory: str | Path) -> DemonstrationSet:
        source = Path(directory)
        with np.load(source / "demonstrations.npz", allow_pickle=False) as archive:
            expected_arrays = {"observations", "actions", "episode_index"}
            if set(archive.files) != expected_arrays:
                raise ValueError(
                    f"{source}: demonstration array set mismatch: "
                    f"stored={sorted(archive.files)}, expected={sorted(expected_arrays)}"
                )
            observations = np.asarray(archive["observations"])
            actions = np.asarray(archive["actions"])
            episode_index = np.asarray(archive["episode_index"])
        ledger = [
            json.loads(line)
            for line in (source / "ledger.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        accepted = [row for row in ledger if row["accepted"]]
        dataset = cls(
            observations=observations,
            actions=actions,
            episode_index=episode_index,
            variant_ids=tuple(row["variant_id"] for row in accepted),
            seeds=tuple(int(row["seed"]) for row in accepted),
            ledger=ledger,
        )
        manifest_path = source / "manifest.json"
        try:
            stored_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise ValueError(f"{source}: missing or invalid demonstration content manifest") from error
        expected_manifest = dataset.content_manifest()
        if stored_manifest != expected_manifest:
            raise ValueError(
                f"{source}: demonstration content manifest mismatch: "
                f"stored={stored_manifest.get('content_hash')!r}, "
                f"actual={expected_manifest['content_hash']!r}"
            )
        summary_path = source / "summary.json"
        try:
            stored_summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise ValueError(f"{source}: missing or invalid demonstration summary") from error
        if stored_summary != dataset.summary():
            raise ValueError(f"{source}: demonstration summary does not match its content")
        return dataset


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
