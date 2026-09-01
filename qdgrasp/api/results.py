"""``GraspResults``: the public prediction container.

The container is a plain dataclass over tensors -- no pickled modules, no
framework objects -- so it can cross device, process and file boundaries with a
stable field contract.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..geometry import is_rotation_matrix


@dataclass(frozen=True)
class GraspResults:
    """``K`` ranked grasps for one observation.

    Attributes:
        translation: Palm translation ``[K, 3]`` in :attr:`frame`.
        rotation: Palm rotation matrices ``[K, 3, 3]``.
        joint_names: Ordered actuated joint names, length ``J``.
        joint_values: Joint state ``[K, J]`` aligned with :attr:`joint_names`.
        score: Ranking score ``[K]``, sorted descending.
        seed_points: Seed point per grasp ``[K, 3]``.
        frame: Name of the frame the poses are expressed in.
        model_hash: Content hash of the model configuration that produced this.
        training_robot_hash: Content hash of the profile the weights were
            produced for.
        runtime_robot_hash: Content hash of the profile these poses were
            generated against.  Equal to ``training_robot_hash`` unless an
            explicit cross-embodiment binding was used.
    """

    translation: torch.Tensor
    rotation: torch.Tensor
    joint_names: tuple[str, ...]
    joint_values: torch.Tensor
    score: torch.Tensor
    seed_points: torch.Tensor
    frame: str
    model_hash: str
    training_robot_hash: str
    runtime_robot_hash: str

    def __post_init__(self) -> None:
        count = self.translation.shape[0]
        joints = len(self.joint_names)
        expected = {
            "translation": (count, 3),
            "rotation": (count, 3, 3),
            "joint_values": (count, joints),
            "score": (count,),
            "seed_points": (count, 3),
        }
        for name, shape in expected.items():
            actual = tuple(getattr(self, name).shape)
            if actual != shape:
                raise ValueError(f"GraspResults.{name} must have shape {shape}, got {actual}")
        if not torch.isfinite(self.joint_values).all():
            raise ValueError("GraspResults.joint_values contains non-finite entries")
        if not is_rotation_matrix(self.rotation):
            raise ValueError("GraspResults.rotation is not a batch of valid SO(3) matrices")

    def __len__(self) -> int:
        return int(self.translation.shape[0])

    @property
    def device(self) -> torch.device:
        """Device the tensors currently live on."""

        return self.translation.device

    def to(self, device: str | torch.device) -> "GraspResults":
        """Return a copy with every tensor moved to ``device``."""

        target = torch.device(device)
        return replace(
            self,
            translation=self.translation.to(target),
            rotation=self.rotation.to(target),
            joint_values=self.joint_values.to(target),
            score=self.score.to(target),
            seed_points=self.seed_points.to(target),
        )

    def cpu(self) -> "GraspResults":
        """Return a CPU copy."""

        return self.to("cpu")

    def numpy(self) -> dict[str, Any]:
        """Detached NumPy view of every field, safe for serialisation."""

        detached = self.cpu()
        return {
            "translation": detached.translation.detach().numpy(),
            "rotation": detached.rotation.detach().numpy(),
            "joint_names": np.array(self.joint_names, dtype=object),
            "joint_values": detached.joint_values.detach().numpy(),
            "score": detached.score.detach().numpy(),
            "seed_points": detached.seed_points.detach().numpy(),
            "frame": self.frame,
            "model_hash": self.model_hash,
            "training_robot_hash": self.training_robot_hash,
            "runtime_robot_hash": self.runtime_robot_hash,
        }

    def metadata(self) -> dict[str, Any]:
        """Non-tensor provenance fields."""

        return {
            "count": len(self),
            "joint_names": list(self.joint_names),
            "frame": self.frame,
            "model_hash": self.model_hash,
            "training_robot_hash": self.training_robot_hash,
            "runtime_robot_hash": self.runtime_robot_hash,
        }

    def save(self, path: str | Path) -> Path:
        """Write an ``.npz`` archive with the tensors and a JSON metadata entry."""

        target = Path(path)
        if target.suffix != ".npz":
            target = target.with_suffix(".npz")
        target.parent.mkdir(parents=True, exist_ok=True)
        arrays = self.numpy()
        np.savez(
            target,
            translation=arrays["translation"],
            rotation=arrays["rotation"],
            joint_values=arrays["joint_values"],
            score=arrays["score"],
            seed_points=arrays["seed_points"],
            metadata=np.array(json.dumps(self.metadata(), sort_keys=True)),
        )
        return target

    @classmethod
    def load(cls, path: str | Path) -> "GraspResults":
        """Read back an archive written by :meth:`save`."""

        with np.load(Path(path), allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata"]))
            return cls(
                translation=torch.from_numpy(archive["translation"]),
                rotation=torch.from_numpy(archive["rotation"]),
                joint_names=tuple(metadata["joint_names"]),
                joint_values=torch.from_numpy(archive["joint_values"]),
                score=torch.from_numpy(archive["score"]),
                seed_points=torch.from_numpy(archive["seed_points"]),
                frame=metadata["frame"],
                model_hash=metadata["model_hash"],
                training_robot_hash=metadata["training_robot_hash"],
                runtime_robot_hash=metadata["runtime_robot_hash"],
            )

    def summary(self) -> str:
        """Human-readable one-block summary, no plotting dependency required."""

        score = self.score.detach().cpu()
        best = float(score.max()) if len(self) else float("nan")
        worst = float(score.min()) if len(self) else float("nan")
        return (
            f"GraspResults(count={len(self)}, joints={len(self.joint_names)}, frame='{self.frame}', "
            f"device='{self.device}', score=[{worst:.4f}, {best:.4f}], "
            f"model={self.model_hash[:12]}, trained_on={self.training_robot_hash[:12]}, "
            f"run_on={self.runtime_robot_hash[:12]})"
        )

    def plot(self, *_args: Any, **_kwargs: Any) -> None:
        """Visualisation hook; the renderer lands with the robot layer in Phase 2.

        Raises:
            NotImplementedError: always -- Phase 1 refuses to pull a plotting
                dependency into the base install just to satisfy the signature.
        """

        raise NotImplementedError(
            "GraspResults.plot() needs the Phase 2 robot/mesh layer; use summary(), numpy() or save() for now"
        )
