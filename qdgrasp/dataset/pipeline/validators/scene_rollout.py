"""Measured integration between multi-object MuJoCo rollout and scene validation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import mujoco
import numpy as np

from qdgrasp.config.schema import ConfigError
from qdgrasp.dataset.pipeline.contracts import DynamicValidation
from qdgrasp.dataset.pipeline.validators.mujoco_rollout import (
    RolloutSceneObject,
    validate_grasp_rollout,
)
from qdgrasp.dataset.pipeline.validators.scene_dynamic import (
    SceneDynamicValidator,
    SceneState,
    hash_scene_state,
)
from qdgrasp.objects.schema import SubGeomSpec


class SceneRolloutEvidenceCollector:
    """Collect stage poses, hand contacts, and non-target contact impulses."""

    def __init__(self, target_object_id: str, non_target_object_ids: Sequence[str]):
        if not target_object_id:
            raise ConfigError("target object ID must be non-empty")
        if len(non_target_object_ids) != len(set(non_target_object_ids)):
            raise ConfigError("non-target object IDs must be unique")
        if target_object_id in non_target_object_ids:
            raise ConfigError("target object cannot also be a non-target object")
        self.target_object_id = target_object_id
        self.non_target_object_ids = tuple(non_target_object_ids)
        self.stage_states: dict[str, SceneState] = {}
        self.contact_object_ids: set[str] = set()
        self.non_target_impulses = {object_id: 0.0 for object_id in self.non_target_object_ids}
        self._model_identity: int | None = None
        self._body_ids: dict[str, int] = {}
        self._geom_owners: dict[int, str] = {}
        self._floor_geom_id = -1

    def _initialize(self, model: mujoco.MjModel) -> None:
        identity = id(model)
        if self._model_identity == identity:
            return
        if self._model_identity is not None:
            raise ConfigError("scene evidence collector cannot observe multiple MuJoCo models")
        body_names = {
            self.target_object_id: "target_object",
            **{object_id: object_id for object_id in self.non_target_object_ids},
        }
        for object_id, body_name in body_names.items():
            body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            if body_id < 0:
                raise ConfigError(f"scene rollout body missing for {object_id}: {body_name}")
            self._body_ids[object_id] = body_id
        for geom_id in range(model.ngeom):
            body_id = int(model.geom_bodyid[geom_id])
            for object_id, object_body_id in self._body_ids.items():
                ancestor = body_id
                while ancestor > 0 and ancestor != object_body_id:
                    ancestor = int(model.body_parentid[ancestor])
                if ancestor == object_body_id:
                    self._geom_owners[geom_id] = object_id
                    break
        self._floor_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        self._model_identity = identity

    def observe_stage(self, stage: str, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        self._initialize(model)
        if stage in self.stage_states:
            raise ConfigError(f"scene rollout stage observed more than once: {stage}")
        self.stage_states[stage] = {
            object_id: {
                "pos": np.asarray(data.xpos[body_id], dtype=np.float64).copy(),
                "quat": np.asarray(data.xquat[body_id], dtype=np.float64).copy(),
            }
            for object_id, body_id in self._body_ids.items()
        }

    def observe_step(self, stage: str, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        del stage
        self._initialize(model)
        timestep = float(model.opt.timestep)
        for contact_index in range(int(data.ncon)):
            contact = data.contact[contact_index]
            geom_1, geom_2 = int(contact.geom1), int(contact.geom2)
            owner_1 = self._geom_owners.get(geom_1)
            owner_2 = self._geom_owners.get(geom_2)
            if owner_1 is None and owner_2 is None:
                continue
            object_id = owner_1 if owner_1 is not None else owner_2
            other_geom = geom_2 if owner_1 is not None else geom_1
            other_owner = owner_2 if owner_1 is not None else owner_1
            if other_owner is None and other_geom != self._floor_geom_id:
                self.contact_object_ids.add(object_id)
            for non_target_id in self.non_target_object_ids:
                if non_target_id not in (owner_1, owner_2) or other_geom == self._floor_geom_id:
                    continue
                force = np.zeros(6, dtype=np.float64)
                mujoco.mj_contactForce(model, data, contact_index, force)
                self.non_target_impulses[non_target_id] += float(np.linalg.norm(force[:3]) * timestep)

    @property
    def state_hashes(self) -> dict[str, str]:
        return {stage: hash_scene_state(state) for stage, state in self.stage_states.items()}


def validate_scene_grasp_rollout(
    hand_xml_path: str,
    collision_geoms: Sequence[SubGeomSpec],
    fingertip_body_names: Sequence[str],
    *,
    target_object_id: str,
    non_target_objects: Sequence[RolloutSceneObject],
    protocol_hash: str,
    recipe_hash: str,
    source_hash: str,
    scene_validator: SceneDynamicValidator | None = None,
    rollout_kwargs: Mapping[str, Any] | None = None,
) -> DynamicValidation:
    """Run one physical multi-object rollout and validate only its measured evidence."""
    options = dict(rollout_kwargs or {})
    forbidden = sorted(
        {"non_target_objects", "initial_observer", "stage_observer", "step_observer"}.intersection(options)
    )
    if forbidden:
        raise ConfigError(f"scene rollout options cannot override evidence hooks: {forbidden}")
    collector = SceneRolloutEvidenceCollector(
        target_object_id,
        [item.object_id for item in non_target_objects],
    )
    base_validation = validate_grasp_rollout(
        hand_xml_path,
        collision_geoms,
        fingertip_body_names,
        non_target_objects=non_target_objects,
        initial_observer=collector.observe_stage,
        stage_observer=collector.observe_stage,
        step_observer=collector.observe_step,
        **options,
    )
    initial_state = collector.stage_states.get("initial", {})
    final_state = collector.stage_states.get("perturbation", initial_state)
    validator = scene_validator or SceneDynamicValidator()
    return validator.validate(
        target_object_id=target_object_id,
        initial_scene_state=initial_state,
        final_scene_state=final_state,
        target_lifted=base_validation.passed,
        base_validation=base_validation,
        stage_scene_states=collector.stage_states,
        state_hashes=collector.state_hashes,
        contact_object_ids=collector.contact_object_ids,
        non_target_impulses=collector.non_target_impulses,
        protocol_hash=protocol_hash,
        recipe_hash=recipe_hash,
        source_hash=source_hash,
    )
