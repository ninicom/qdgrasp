"""Explicit binding between the hand a model was trained on and the hand it runs on.

``PLAN.md`` §9.8: the two roles are different facts and a single ``robot_hash``
cannot hold both.  An exact-match gate on that one field is wrong in both
directions -- it forbids the cross-embodiment inference the protocol exists to
measure, and it lets an artifact produced for one hand be reported under another
as long as the field is overwritten somewhere.

So a transfer is a declared object rather than a relaxed check.  It names both
profiles, records the joint mapping it will use, and cites the locked protocol
that permits it.  Without such a citation there is no transfer: a hand that no
protocol pairs with the training hand is refused, however similar its kinematics
happen to be.
"""

from __future__ import annotations

import dataclasses
from typing import Any

from ..config.schema import ConfigError

EMBODIMENT_BINDING_SCHEMA = "qdgrasp/embodiment-binding/v1"


class CompatibilityError(ConfigError):
    """The runtime hand may not be driven by weights trained on the other one."""


@dataclasses.dataclass(frozen=True)
class EmbodimentBinding:
    """One permitted (training hand -> runtime hand) pairing, with its evidence."""

    training_robot: str
    training_robot_hash: str
    runtime_robot: str
    runtime_robot_hash: str
    frame: str
    joint_mapping: tuple[tuple[str, str], ...]
    permitted_by: str
    protocol_hash: str | None = None

    @property
    def is_transfer(self) -> bool:
        """Is this a cross-embodiment binding rather than the identity one?"""

        return self.training_robot_hash != self.runtime_robot_hash

    def to_document(self) -> dict[str, Any]:
        return {
            "schema": EMBODIMENT_BINDING_SCHEMA,
            "training_robot": self.training_robot,
            "training_robot_hash": self.training_robot_hash,
            "runtime_robot": self.runtime_robot,
            "runtime_robot_hash": self.runtime_robot_hash,
            "frame": self.frame,
            "joint_mapping": [list(pair) for pair in self.joint_mapping],
            "permitted_by": self.permitted_by,
            "protocol_hash": self.protocol_hash,
            "is_transfer": self.is_transfer,
        }


def _joints(config: Any) -> tuple[str, ...]:
    return tuple(getattr(config, "joints", ()))


def _fingertips(config: Any) -> tuple[str, ...]:
    return tuple(getattr(config, "fingertip_links", ()))


def bind_embodiment(training: Any, runtime: Any, *, protocol: Any | None = None) -> EmbodimentBinding:
    """Bind a runtime profile to a training profile, or refuse to.

    Args:
        training: Robot profile the weights were produced for.
        runtime: Robot profile inference will be run against.
        protocol: Locked protocol whose held-out embodiment permits the transfer.
            Required whenever the two profiles differ.

    Raises:
        CompatibilityError: The pairing is not permitted, or the profiles cannot
            express the same grasp.
    """

    training_hash = training.content_hash()
    runtime_hash = runtime.content_hash()
    training_name = str(getattr(training, "name", ""))
    runtime_name = str(getattr(runtime, "name", ""))

    if training_hash == runtime_hash:
        return EmbodimentBinding(
            training_robot=training_name,
            training_robot_hash=training_hash,
            runtime_robot=runtime_name,
            runtime_robot_hash=runtime_hash,
            frame=str(getattr(training, "frame", "")),
            joint_mapping=tuple((name, name) for name in _joints(training)),
            permitted_by="identity",
        )

    if protocol is None:
        raise CompatibilityError(
            f"weights trained on {training_name!r} may not be run against {runtime_name!r} without a locked "
            "protocol that declares the transfer; similar kinematics are not a permission"
        )

    embodiment = getattr(protocol, "heldout_embodiment", None)
    permitted = (
        embodiment is not None
        and getattr(embodiment, "train_hand", None) == training_name
        and getattr(embodiment, "test_hand", None) == runtime_name
    )
    if not permitted:
        declared = (
            f"{getattr(embodiment, 'train_hand', None)!r} -> {getattr(embodiment, 'test_hand', None)!r}"
            if embodiment is not None
            else "nothing"
        )
        raise CompatibilityError(
            f"protocol {getattr(protocol, 'name', '?')!r} declares {declared}, not "
            f"{training_name!r} -> {runtime_name!r}"
        )

    training_frame = str(getattr(training, "frame", ""))
    runtime_frame = str(getattr(runtime, "frame", ""))
    if training_frame != runtime_frame:
        raise CompatibilityError(
            f"{training_name} poses are expressed in {training_frame!r} and {runtime_name} in "
            f"{runtime_frame!r}; the same numbers would mean two different placements"
        )

    training_joints = _joints(training)
    runtime_joints = _joints(runtime)
    if len(training_joints) != len(runtime_joints):
        raise CompatibilityError(
            f"{training_name} has {len(training_joints)} actuated joints and {runtime_name} has "
            f"{len(runtime_joints)}; a positional mapping between them would be an invention"
        )
    if len(_fingertips(training)) != len(_fingertips(runtime)):
        raise CompatibilityError(
            f"{training_name} and {runtime_name} declare different fingertip counts, so the FK targets the "
            "model was trained against do not exist on the runtime hand"
        )

    return EmbodimentBinding(
        training_robot=training_name,
        training_robot_hash=training_hash,
        runtime_robot=runtime_name,
        runtime_robot_hash=runtime_hash,
        frame=training_frame,
        joint_mapping=tuple(zip(training_joints, runtime_joints, strict=True)),
        permitted_by=str(getattr(protocol, "name", "protocol")),
        protocol_hash=getattr(protocol, "protocol_hash", None),
    )


__all__ = (
    "EMBODIMENT_BINDING_SCHEMA",
    "CompatibilityError",
    "EmbodimentBinding",
    "bind_embodiment",
)
