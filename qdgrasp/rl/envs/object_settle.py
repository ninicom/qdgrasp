"""``QDGrasp-ObjectSettle-v0`` -- the ingest/drop/settle debug environment.

This environment has no hand and no grasp.  Its whole job is to make the
asset → scene → drop → settle path steppable and inspectable, so a failure there
is diagnosed on its own rather than through a grasp task that would blame the
policy for it.

It is registered as an environment because that is the most convenient shape for
a debugging tool, not because settling is a control problem.  The action is a
no-op by construction: there is nothing to control, and an action that did
something here would be a way of writing the scene state by hand.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from typing import Any

import mujoco
import numpy as np

from qdgrasp.rl.contracts import (
    BoxSpace,
    ObservationField,
    ObservationSchema,
    RewardBreakdown,
    StepResult,
    TerminalReason,
)
from qdgrasp.rl.randomization import DomainRandomization, SeedStreams, apply_randomization, scene_signature
from qdgrasp.scenes.resolver import ResolvedScene, resolve_scene
from qdgrasp.scenes.settle import SettleOutcome, certify_settle
from qdgrasp.scenes.virtual_drop import DropObjectRequest, SettleThresholds, VirtualDropSceneSpec

#: Per-object observation blocks.  The schema is built for the configured object
#: count, because a variable-width observation is not a space.
_PER_OBJECT_FIELDS: tuple[tuple[str, int, str, str], ...] = (
    ("position", 3, "m", "world"),
    ("orientation_6d", 6, "unitless", "world"),
    ("linear_velocity", 3, "m/s", "world"),
    ("angular_velocity", 3, "rad/s", "world"),
)


def build_settle_observation_schema(object_count: int) -> ObservationSchema:
    fields: list[ObservationField] = []
    for index in range(object_count):
        for name, size, unit, frame in _PER_OBJECT_FIELDS:
            fields.append(
                ObservationField(
                    name=f"object_{index}_{name}",
                    size=size,
                    unit=unit,
                    frame=frame,
                    description=f"dynamic object {index}",
                )
            )
    fields.append(ObservationField("contact_count", 1, "count", "scene"))
    fields.append(ObservationField("max_penetration", 1, "m", "scene"))
    fields.append(ObservationField("time_remaining", 1, "unitless", "episode"))
    schema = ObservationSchema(fields=tuple(fields))
    schema.validate()
    return schema


@dataclasses.dataclass(frozen=True)
class ObjectSettleConfig:
    """Everything the settle environment needs, fixed before the first reset."""

    objects: tuple[DropObjectRequest, ...]
    virtual_scene: VirtualDropSceneSpec = dataclasses.field(default_factory=VirtualDropSceneSpec)
    randomization: DomainRandomization = dataclasses.field(default_factory=DomainRandomization)
    #: Control steps per episode; each advances the scene by ``substeps``.
    max_steps: int = 60
    substeps: int = 20

    def validate(self) -> None:
        if not self.objects:
            raise ValueError("the settle environment needs at least one object")
        if self.max_steps < 1 or self.substeps < 1:
            raise ValueError("max_steps and substeps must be positive")
        self.virtual_scene.validate()
        self.randomization.validate()


class ObjectSettleEnv:
    """Step a generated or loaded scene while it settles."""

    environment_id = "QDGrasp-ObjectSettle-v0"

    def __init__(self, config: ObjectSettleConfig, *, scene_ref: str | None = None) -> None:
        config.validate()
        self.config = config
        self.scene_ref = scene_ref
        self.schema = build_settle_observation_schema(len(config.objects))
        self._resolved: ResolvedScene | None = None
        self._data: mujoco.MjData | None = None
        self._step_index = 0
        self._done = False
        self._signature: str | None = None

    # -- spaces -----------------------------------------------------------

    def observation_space(self) -> BoxSpace:
        return self.schema.space()

    def action_space(self) -> BoxSpace:
        # A single inert channel: the settle environment has nothing to command,
        # and a wider space would imply otherwise.
        return BoxSpace(name="action", shape=(1,), low=-1.0, high=1.0, dtype="float32")

    # -- lifecycle --------------------------------------------------------

    def reset(self, *, seed: int, options: Mapping[str, Any] | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        streams = SeedStreams(episode_seed=seed)
        scene_rng = streams.generator("scene")
        physics_rng = streams.generator("physics")

        resolved = resolve_scene(
            scene_ref=self.scene_ref,
            objects=self.config.objects,
            virtual_scene_config=self.config.virtual_scene,
            seed=int(scene_rng.integers(0, 2**31 - 1)),
            scene_id=f"settle-{seed}",
        )
        self._resolved = resolved
        self._signature = scene_signature(resolved.spec)

        sample = self.config.randomization.sample(physics_rng)
        object_ids = [item.object_id for item in resolved.spec.objects]
        geom_names = [
            mujoco.mj_id2name(resolved.model, mujoco.mjtObj.mjOBJ_GEOM, index) for index in range(resolved.model.ngeom)
        ]
        target_geoms = [name for name in geom_names if name and name.split("::")[0] in set(object_ids)]
        applied = apply_randomization(resolved.model, object_ids, target_geoms, sample)
        # Recompute the mass-derived solver constants after the mass edit; a
        # stale `body_invweight0` gives a randomised mass the contact compliance
        # of the compiled one.
        mujoco.mj_setConst(resolved.model, mujoco.MjData(resolved.model))

        self._data = mujoco.MjData(resolved.model)
        mujoco.mj_forward(resolved.model, self._data)
        self._step_index = 0
        self._done = False
        info = {
            "scene_source": resolved.source.value,
            "scene_signature": self._signature,
            "scene_id": resolved.spec.scene_id,
            "randomization": sample,
            "randomization_applied": applied,
            "randomization_hash": self.config.randomization.content_hash(),
            "observation_schema_hash": self.schema.content_hash(),
        }
        return self._observation(), info

    def step(self, action: Sequence[float]) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self._done:
            raise RuntimeError("step() called on a finished episode; call reset() first")
        assert self._resolved is not None and self._data is not None
        model, data = self._resolved.model, self._data
        np.asarray(action, dtype=np.float64)  # accepted and ignored: nothing here is controllable

        for _ in range(self.config.substeps):
            mujoco.mj_step(model, data)
        self._step_index += 1

        finite = bool(np.all(np.isfinite(data.qpos)) and np.all(np.isfinite(data.qvel)))
        quiet = self._is_quiet()
        terminated = not finite or quiet
        truncated = (not terminated) and self._step_index >= self.config.max_steps
        self._done = terminated or truncated

        if not finite:
            reason = TerminalReason.INVALID_STATE
        elif quiet:
            reason = TerminalReason.SUCCESS
        elif truncated:
            reason = TerminalReason.HORIZON
        else:
            reason = TerminalReason.NONE

        # No shaping: this environment reports whether the scene came to rest,
        # and inventing a reward for it would invite tuning the settle criterion.
        reward = RewardBreakdown(terms={})
        info: dict[str, Any] = {
            "terminal_reason": reason,
            "reward_terms": reward.to_document(),
            "settled": bool(quiet),
            "contact_count": int(data.ncon),
            "max_penetration_m": self._max_penetration(),
            "scene_signature": self._signature,
        }
        result = StepResult(self._observation(), reward.total, terminated, truncated, info)
        return result.as_tuple()

    def certify(self) -> Any:
        """Run the full settle certifier on the current scene from its placement."""

        assert self._resolved is not None
        data = mujoco.MjData(self._resolved.model)
        return certify_settle(
            self._resolved.spec,
            self._resolved.model,
            data,
            self.config.virtual_scene.settle_thresholds,
            spawn_region=self.config.virtual_scene.spawn_region,
        )

    def close(self) -> None:
        self._resolved = None
        self._data = None

    # -- measurement ------------------------------------------------------

    def _object_indices(self) -> list[tuple[int, int, int]]:
        assert self._resolved is not None
        model = self._resolved.model
        indices = []
        for item in self._resolved.spec.objects:
            body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, item.object_id)
            joint_id = int(model.body_jntadr[body_id])
            indices.append((body_id, int(model.jnt_qposadr[joint_id]), int(model.jnt_dofadr[joint_id])))
        return indices

    def _is_quiet(self) -> bool:
        assert self._data is not None
        thresholds: SettleThresholds = self.config.virtual_scene.settle_thresholds
        data = self._data
        for _, _, qvel_adr in self._object_indices():
            linear = float(np.linalg.norm(data.qvel[qvel_adr : qvel_adr + 3]))
            angular = float(np.linalg.norm(data.qvel[qvel_adr + 3 : qvel_adr + 6]))
            if linear > thresholds.linear_velocity_mps or angular > thresholds.angular_velocity_radps:
                return False
        return self._max_penetration() <= thresholds.max_penetration_m

    def _max_penetration(self) -> float:
        assert self._data is not None
        return max((float(-self._data.contact[index].dist) for index in range(int(self._data.ncon))), default=0.0)

    def _observation(self) -> np.ndarray:
        assert self._resolved is not None and self._data is not None
        data = self._data
        parts: dict[str, np.ndarray] = {}
        for index, (body_id, _qpos_adr, qvel_adr) in enumerate(self._object_indices()):
            rotation = np.array(data.xmat[body_id]).reshape(3, 3)
            parts[f"object_{index}_position"] = np.array(data.xpos[body_id], dtype=np.float64)
            parts[f"object_{index}_orientation_6d"] = rotation[:, :2].T.reshape(-1)
            parts[f"object_{index}_linear_velocity"] = np.array(data.qvel[qvel_adr : qvel_adr + 3])
            parts[f"object_{index}_angular_velocity"] = np.array(data.qvel[qvel_adr + 3 : qvel_adr + 6])
        parts["contact_count"] = np.array([float(data.ncon)])
        parts["max_penetration"] = np.array([self._max_penetration()])
        parts["time_remaining"] = np.array([1.0 - self._step_index / self.config.max_steps])
        observation = self.schema.assemble(parts)
        return np.nan_to_num(observation, nan=0.0, posinf=0.0, neginf=0.0)


def settle_outcome_of(env: ObjectSettleEnv) -> SettleOutcome:
    """Certify the environment's current scene and return the outcome class."""

    return env.certify().outcome
