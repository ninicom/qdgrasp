"""Provenance tracking and release-blocked policy for robot profiles."""

from __future__ import annotations

from typing import Any

from ..config.schema import ConfigError
from .schema import RobotConfigV2


def validate_profile_for_release(profile: RobotConfigV2) -> None:
    """Enforce that release-blocked profiles are never published in a public release."""
    if profile.release_blocked:
        raise ConfigError(
            f"robot profile '{profile.name}' is marked release_blocked=True "
            f"(provenance reason: {profile.provenance.get('restriction_reason', 'unverified license')}); "
            "it is restricted to local research fixtures and must not be included in public distribution."
        )

    # Check required provenance fields for published profiles
    prov = profile.provenance
    if not prov:
        raise ConfigError(f"robot profile '{profile.name}' is missing provenance metadata")

    required_keys = ("license", "source_repository", "source_commit")
    missing = [k for k in required_keys if k not in prov]
    if missing:
        raise ConfigError(f"robot profile '{profile.name}' provenance missing required fields: {missing}")


def get_profile_provenance(profile: RobotConfigV2) -> dict[str, Any]:
    """Extract complete provenance summary for a robot profile."""
    return {
        "profile_name": profile.name,
        "schema": profile.schema_version,
        "format": profile.format,
        "source_asset": profile.source_asset,
        "release_blocked": profile.release_blocked,
        "num_actuated_joints": len(profile.joints),
        "fingertip_links": list(profile.fingertip_links),
        "provenance": dict(profile.provenance),
    }
