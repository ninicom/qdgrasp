"""Shared fingertip contact-state primitive for the kinematic solvers.

Both DLS solvers used to carry their own private copies of `contact_position`,
`contact_direction` and `compute_single`.  The copies had drifted: the evaluation
path scored the configured fingertip contact axis while the autodiff path
differentiated a parent-to-tip vector, so the solver optimised a different
function from the one it graded (RC-01 in the Phase 3.2.1 plan).

This module holds the single definition of that state.  It is deliberately
behaviour-preserving: `DirectionMode` still exposes the divergent
`"parent_to_tip"` formula so the extraction itself cannot change any result.
Work package P3.2.1-03 is what points the autodiff path at `"configured"`.
"""

from __future__ import annotations

from typing import Any, Literal, Mapping, Tuple

import torch

from qdgrasp.robot.spec import RobotSpec

# "configured": rotate the fingertip's configured approach axis into world frame,
#   falling back to the parent-to-tip-frame vector when no axis is configured.
# "parent_to_tip": always the vector from the parent link origin to the contact
#   point.  Retained only so the P3.2.1-02 extraction is a no-op; RC-01.
DirectionMode = Literal["configured", "parent_to_tip"]

# The direction formula the autodiff residual currently differentiates.  Flipping
# this to "configured" is the RC-01 intervention and is expected to move the
# frozen failure corpus.
AUTODIFF_DIRECTION_MODE: DirectionMode = "parent_to_tip"


def _parent_origin(
    spec: RobotSpec,
    transforms: Mapping[str, torch.Tensor],
    tip: str,
    fallback_origin: torch.Tensor,
) -> torch.Tensor:
    """World-frame origin of the fingertip's parent link, or the fallback."""
    link = getattr(spec, "links", {}).get(tip)
    parent = getattr(link, "parent_link", None)
    if parent in transforms:
        return transforms[parent][:, :3, 3]
    return fallback_origin


def contact_position(
    spec: RobotSpec, transforms: Mapping[str, torch.Tensor], tip: str
) -> torch.Tensor:
    """[B, 3] world-frame contact point: tip frame plus its configured offset."""
    transform = transforms[tip]
    offset: Any = getattr(spec, "fingertip_contact_offsets", {}).get(tip)
    if offset is None:
        return transform[:, :3, 3]
    offset_t = torch.as_tensor(offset, dtype=transform.dtype, device=transform.device)
    return transform[:, :3, 3] + torch.matmul(
        transform[:, :3, :3], offset_t.view(3, 1)
    ).squeeze(-1)


def contact_direction(
    spec: RobotSpec,
    transforms: Mapping[str, torch.Tensor],
    tip: str,
    *,
    fallback_origin: torch.Tensor,
    mode: DirectionMode = "configured",
) -> torch.Tensor:
    """[B, 3] unit contact direction under the requested convention."""
    transform = transforms[tip]
    if mode == "configured":
        configured_axis: Any = getattr(spec, "fingertip_contact_axes", {}).get(tip)
        if configured_axis is not None:
            axis_t = torch.as_tensor(
                configured_axis, dtype=transform.dtype, device=transform.device
            )
            return torch.nn.functional.normalize(
                torch.matmul(transform[:, :3, :3], axis_t.view(3, 1)).squeeze(-1),
                dim=-1,
                eps=1e-8,
            )
        # No configured axis: the tip frame origin relative to its parent.  Note
        # this fallback intentionally ignores the contact offset, matching the
        # evaluation path it was extracted from.
        origin = _parent_origin(spec, transforms, tip, fallback_origin)
        return torch.nn.functional.normalize(
            transform[:, :3, 3] - origin, dim=-1, eps=1e-8
        )

    if mode == "parent_to_tip":
        origin = _parent_origin(spec, transforms, tip, fallback_origin)
        return torch.nn.functional.normalize(
            contact_position(spec, transforms, tip) - origin, dim=-1, eps=1e-8
        )

    raise ValueError(f"unknown direction mode: {mode!r}")


def contact_state(
    spec: RobotSpec,
    palm_pos: torch.Tensor,
    palm_rot: torch.Tensor,
    q: torch.Tensor,
    *,
    mode: DirectionMode = "configured",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Contact points [B, K, 3] and contact directions [B, K, 3] at ``q``."""
    transforms = spec.forward_kinematics(palm_pos, palm_rot, q)
    return contact_state_from_transforms(
        spec, transforms, fallback_origin=palm_pos, mode=mode
    )


def contact_state_from_transforms(
    spec: RobotSpec,
    transforms: Mapping[str, torch.Tensor],
    *,
    fallback_origin: torch.Tensor,
    mode: DirectionMode = "configured",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Same as `contact_state` for an FK result that has already been computed."""
    positions = torch.stack(
        [contact_position(spec, transforms, tip) for tip in spec.fingertip_links], dim=1
    )
    directions = torch.stack(
        [
            contact_direction(
                spec, transforms, tip, fallback_origin=fallback_origin, mode=mode
            )
            for tip in spec.fingertip_links
        ],
        dim=1,
    )
    return positions, directions


def contact_residual_features(
    spec: RobotSpec,
    q_single: torch.Tensor,
    palm_pos_single: torch.Tensor,
    palm_rot_single: torch.Tensor,
    *,
    normal_weight: float,
    mode: DirectionMode = AUTODIFF_DIRECTION_MODE,
) -> torch.Tensor:
    """Flat [6K] task vector for one sample: positions then weighted directions.

    This is the function the solver's Jacobian differentiates, and it must stay
    the same function the solver's residual scores — that identity is what RC-01
    broke and what the derivative oracle in the plan's section 5.1 checks.
    """
    positions, directions = contact_state(
        spec,
        palm_pos_single.unsqueeze(0),
        palm_rot_single.unsqueeze(0),
        q_single.unsqueeze(0),
        mode=mode,
    )
    return torch.cat(
        (positions[0].reshape(-1), (directions[0] * normal_weight).reshape(-1)), dim=0
    )
