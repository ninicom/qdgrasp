"""Temporary Grasp Policy MVP vertical slice (``ROADMAP-MVP-001``).

Everything under this package is scoped to one hand (LEAP), one table, one
target and privileged state observation.  Artifacts it produces carry
``release_class: experimental_non_release`` and must never be cited as closure
evidence for P3.4.3, P3.5, P4 or P5.
"""

from qdgrasp.mvp.config import (
    MVP_SCOPE_SCHEMA_V0,
    ControllerSpec,
    MvpScopeConfig,
    ObjectVariant,
    load_mvp_scope,
)

__all__ = [
    "MVP_SCOPE_SCHEMA_V0",
    "ControllerSpec",
    "MvpScopeConfig",
    "ObjectVariant",
    "load_mvp_scope",
]
