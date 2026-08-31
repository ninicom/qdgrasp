"""Reinforcement-learning environment contracts and environments (P3.5-09..12).

The environments here are the RL-readiness surface Phase 3.5 delivers: an asset
becomes a scene, the scene drops and settles, and the settled scene becomes an
environment with ``reset``/``step``.  What they are *not* is a claim that a
policy trained in them is any good -- that is P5's business.
"""

from qdgrasp.rl.contracts import (
    RL_CONTRACT_SCHEMA_V1,
    BoxSpace,
    ObservationField,
    ObservationSchema,
    RewardBreakdown,
    RlActionSpec,
    RlEnvironment,
    StepResult,
    TerminalReason,
    to_gymnasium_space,
)

__all__ = (
    "RL_CONTRACT_SCHEMA_V1",
    "BoxSpace",
    "ObservationField",
    "ObservationSchema",
    "RewardBreakdown",
    "RlActionSpec",
    "RlEnvironment",
    "StepResult",
    "TerminalReason",
    "to_gymnasium_space",
)
