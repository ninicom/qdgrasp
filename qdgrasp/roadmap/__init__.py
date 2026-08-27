"""Machine-readable roadmap ledgers (P3.4.3 C08).

A roadmap requirement is only closed when code, tests and evidence all point at
it. This package reads that claim from a manifest and refuses to compute a pass
verdict from prose.
"""

from .requirements import (
    ALLOWED_STATUS,
    ClosureVerdict,
    ManifestError,
    Requirement,
    RequirementsManifest,
    audit_closure,
    load_manifest,
)
from .scope_audit import (
    HISTORICAL_CONFIG_ALLOWLIST,
    ScopeFinding,
    audit_active_scope,
    audit_config_files,
    audit_runtime_defaults,
)

__all__ = (
    "ALLOWED_STATUS",
    "HISTORICAL_CONFIG_ALLOWLIST",
    "ClosureVerdict",
    "ManifestError",
    "Requirement",
    "RequirementsManifest",
    "ScopeFinding",
    "audit_active_scope",
    "audit_closure",
    "audit_config_files",
    "audit_runtime_defaults",
    "load_manifest",
)
