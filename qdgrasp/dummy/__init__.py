"""Dummy model and dataset that make the Phase 1 lifecycle runnable.

Importing this package registers ``dummy_grasp`` and ``dummy_points`` in the
configuration allowlist.  Both are development fixtures: they are replaced by
the real data layer (Phase 3) and model (Phase 4) and carry no research claim.
"""

from __future__ import annotations

from .data import DummyPointDataset, build_dummy_points
from .model import DummyGraspModel, build_dummy_grasp

__all__ = ("DummyGraspModel", "DummyPointDataset", "build_dummy_grasp", "build_dummy_points")
