"""Public QDGrasp API: façade, results and the protocols they satisfy.

Importing this package registers the built-in model/dataset builders in the
configuration allowlist.
"""

from __future__ import annotations

from .. import dummy as _dummy  # noqa: F401  (registers built-in builders)
from .facade import DEFAULT_MODEL, DEFAULT_ROBOT, QDGrasp, load
from .protocols import GraspDataset, GraspModel
from .results import GraspResults

__all__ = (
    "DEFAULT_MODEL",
    "DEFAULT_ROBOT",
    "GraspDataset",
    "GraspModel",
    "GraspResults",
    "QDGrasp",
    "load",
)
