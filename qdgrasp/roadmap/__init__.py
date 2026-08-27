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

__all__ = (
    "ALLOWED_STATUS",
    "ClosureVerdict",
    "ManifestError",
    "Requirement",
    "RequirementsManifest",
    "audit_closure",
    "load_manifest",
)
