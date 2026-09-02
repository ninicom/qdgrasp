"""Versioned report contracts shared by the MVP producers and gates."""

from __future__ import annotations

TRAINING_REPORT_SCHEMA_V1 = "qdgrasp/mvp-training-report/v1"
TRAINING_REPORT_SCHEMA = TRAINING_REPORT_SCHEMA_V1

#: The release contract's two new reports.  They exist only under scope v1:
#: the experimental gate has nothing to say about contribution or ablation, and
#: an artifact set that lacks them cannot reach a release verdict.
CONTRIBUTION_REPORT_SCHEMA_V1 = "qdgrasp/mvp-contribution-report/v1"
CONTRIBUTION_REPORT_SCHEMA = CONTRIBUTION_REPORT_SCHEMA_V1

ABLATION_REPORT_SCHEMA_V1 = "qdgrasp/mvp-ablation-report/v1"
ABLATION_REPORT_SCHEMA = ABLATION_REPORT_SCHEMA_V1

#: The locked challenge domain Tier D is drawn from, written in MR-03.
CHALLENGE_DOMAIN_SCHEMA_V1 = "qdgrasp/mvp-challenge-domain/v1"
CHALLENGE_DOMAIN_SCHEMA = CHALLENGE_DOMAIN_SCHEMA_V1

__all__ = [
    "ABLATION_REPORT_SCHEMA",
    "ABLATION_REPORT_SCHEMA_V1",
    "CHALLENGE_DOMAIN_SCHEMA",
    "CHALLENGE_DOMAIN_SCHEMA_V1",
    "CONTRIBUTION_REPORT_SCHEMA",
    "CONTRIBUTION_REPORT_SCHEMA_V1",
    "TRAINING_REPORT_SCHEMA",
    "TRAINING_REPORT_SCHEMA_V1",
]
