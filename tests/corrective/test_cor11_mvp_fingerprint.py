"""COR-11: the guard exists, the worker does not use it, the report signs anyway.

``load_policy`` refuses a checkpoint whose fingerprint disagrees with the world
it is being loaded into -- when it is given a fingerprint.  ``_worker_init``
calls it without one.  The episodes then run against whatever environment the
worker built, and ``evaluate_candidate`` stamps the *current* fingerprint onto
the report, so an artifact trained in another world comes out labelled as if it
had been evaluated in this one.

Separately, the actor is documented as a bounded Gaussian residual and is a raw
Normal around a squashed mean.  Its samples leave ``[-1, 1]``, the environment
clips them, and the log-probability the update uses is the density of a value
that was never executed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from _corrective_support import characterization, refuses


@characterization("COR-11", note="the evaluation worker loads a checkpoint without its fingerprint")
def test_the_evaluation_worker_refuses_a_foreign_checkpoint(tmp_path: Path, monkeypatch) -> None:
    from qdgrasp.mvp import evaluate as evaluate_module
    from qdgrasp.mvp.policy import ResidualActorCritic, RunningNormalizer, save_checkpoint
    from qdgrasp.mvp.prior import DEFAULT_PRIOR_PATH

    network = ResidualActorCritic(observation_dim=4, action_dim=2, hidden=(8,))
    checkpoint = save_checkpoint(
        tmp_path / "foreign.pt",
        network,
        RunningNormalizer(dimension=4),
        fingerprint={"scope": "a-world-that-is-not-this-one"},
        stage="characterization",
    )

    class _StubEnvironment:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

    monkeypatch.setattr(evaluate_module, "DexAcquireMvpEnv", _StubEnvironment)
    evaluate_module._WORKER.clear()

    refuses(
        lambda: evaluate_module._worker_init(None, str(DEFAULT_PRIOR_PATH), str(checkpoint)),
        because=(
            "the worker built a policy from a checkpoint trained against another environment; the guard has "
            "to run before an environment, an episode, a ledger or a report exists"
        ),
    )
    assert evaluate_module._WORKER.get("policy") is None


@characterization("COR-11", note="no verification runs before the report is written")
def test_the_evaluator_verifies_the_checkpoint_before_it_reports() -> None:
    from qdgrasp.mvp import evaluate as evaluate_module

    assert hasattr(evaluate_module, "verify_checkpoint_fingerprint"), (
        "PLAN.md §9.9 asks for the expected fingerprint to be computed once and validated before the first "
        "episode, and for the report to carry stored, effective and verdict rather than a fresh stamp"
    )


@characterization("COR-11", note="the actor is a raw Normal described as bounded")
def test_sampled_actions_stay_inside_the_bounds_they_are_documented_to_have() -> None:
    from qdgrasp.mvp.policy import LOG_STD_BOUNDS, ResidualActorCritic

    torch.manual_seed(0)
    network = ResidualActorCritic(observation_dim=6, action_dim=4, hidden=(16,))
    with torch.no_grad():
        network.log_std.fill_(LOG_STD_BOUNDS[1])
        distribution = network.distribution(torch.zeros(512, 6))
        actions = distribution.sample()

    outside = int((actions.abs() > 1.0).sum())
    assert outside == 0, (
        f"{outside} of {actions.numel()} sampled actions fall outside [-1, 1] for a distribution documented "
        "as bounded; the environment clips them afterwards, so the log-probability PPO uses belongs to an "
        "action that was never executed"
    )
