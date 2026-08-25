import mujoco
import numpy as np

from qdgrasp.dataset.pipeline.contracts import DynamicValidation
from qdgrasp.dataset.pipeline.validators import scene_rollout as scene_rollout_module
from qdgrasp.dataset.pipeline.validators.mujoco_rollout import RolloutSceneObject
from qdgrasp.dataset.pipeline.validators.scene_rollout import (
    SceneRolloutEvidenceCollector,
    run_scene_grasp_rollout,
    validate_scene_grasp_rollout,
)


def _model():
    return mujoco.MjModel.from_xml_string(
        """
        <mujoco>
          <option timestep="0.001" gravity="0 0 0"/>
          <worldbody>
            <body name="target_object"><freejoint/><geom name="target_geom" type="sphere" size="0.03"/></body>
            <body name="obstacle" pos="0.3 0 0"><freejoint/><geom name="obstacle_geom" type="sphere" size="0.03"/></body>
            <body name="hand_stub" pos="0.04 0 0.05"><geom name="hand_geom" type="sphere" size="0.03"/></body>
            <body name="obstacle_hand_stub" pos="0.34 0 0"><geom name="obstacle_hand_geom" type="sphere" size="0.03"/></body>
            <geom name="floor" type="plane" size="1 1 0.1" pos="0 0 -1"/>
          </worldbody>
        </mujoco>
        """
    )


def _set_target_height(model, data, height):
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target_object")
    joint_id = int(model.body_jntadr[body_id])
    qpos_address = int(model.jnt_qposadr[joint_id])
    data.qpos[qpos_address + 2] = height
    mujoco.mj_forward(model, data)


def test_collector_snapshots_canonical_states_and_measures_hand_contact():
    model = _model()
    data = mujoco.MjData(model)
    collector = SceneRolloutEvidenceCollector("target", ["obstacle"])
    for stage, height in (
        ("initial", 0.0),
        ("squeeze", 0.005),
        ("lift", 0.05),
        ("perturbation", 0.05),
    ):
        _set_target_height(model, data, height)
        collector.observe_stage(stage, model, data)
        collector.observe_step(stage, model, data)
    assert set(collector.stage_states) == {"initial", "squeeze", "lift", "perturbation"}
    assert collector.contact_object_ids == {"target", "obstacle"}
    assert collector.non_target_impulses["obstacle"] > 0.0
    assert all(len(value) == 64 for value in collector.state_hashes.values())


def test_scene_rollout_wrapper_feeds_only_observed_evidence(monkeypatch):
    model = _model()
    data = mujoco.MjData(model)

    def fake_rollout(*args, initial_observer, stage_observer, step_observer, **kwargs):
        del args, kwargs
        for stage, height in (
            ("initial", 0.0),
            ("squeeze", 0.005),
            ("lift", 0.05),
            ("perturbation", 0.05),
        ):
            _set_target_height(model, data, height)
            (initial_observer if stage == "initial" else stage_observer)(stage, model, data)
            step_observer(stage, model, data)
        return DynamicValidation(
            trajectory_metrics={
                "lift_achieved": 0.045,
                "final_active_fingers": 2.0,
                "swept_clearance_passed": 1.0,
            },
            per_finger_loads=np.ones((2, 6)),
            failure_stage="none",
            passed=True,
        )

    monkeypatch.setattr(scene_rollout_module, "validate_grasp_rollout", fake_rollout)
    result = validate_scene_grasp_rollout(
        "unused.xml",
        [],
        ["tip_1", "tip_2"],
        target_object_id="target",
        non_target_objects=[],
        protocol_hash="a" * 64,
        recipe_hash="b" * 64,
        source_hash="c" * 64,
    )
    assert result.passed
    assert result.trajectory_metrics["validated_stages"] == [
        "initial",
        "squeeze",
        "lift",
        "perturbation",
    ]
    assert result.trajectory_metrics["scene_state_hashes"]

    disturbed = validate_scene_grasp_rollout(
        "unused.xml",
        [],
        ["tip_1", "tip_2"],
        target_object_id="target",
        non_target_objects=[RolloutSceneObject("obstacle", [], (0.3, 0.0, 0.0))],
        protocol_hash="a" * 64,
        recipe_hash="b" * 64,
        source_hash="c" * 64,
    )
    assert not disturbed.passed
    assert disturbed.failure_stage == "wrong_object_contact"
    assert disturbed.trajectory_metrics["wrong_contacts"] == ["obstacle"]


def test_scene_rollout_result_exposes_same_stage_observer_evidence(monkeypatch):
    model = _model()
    data = mujoco.MjData(model)

    def fake_rollout(*args, initial_observer, stage_observer, step_observer, **kwargs):
        del args, kwargs
        for stage, height in (
            ("initial", 0.0),
            ("squeeze", 0.005),
            ("lift", 0.05),
            ("perturbation", 0.05),
        ):
            _set_target_height(model, data, height)
            (initial_observer if stage == "initial" else stage_observer)(stage, model, data)
            step_observer(stage, model, data)
        return DynamicValidation(
            trajectory_metrics={
                "lift_achieved": 0.045,
                "final_active_fingers": 2.0,
                "swept_clearance_passed": 1.0,
            },
            per_finger_loads=np.ones((2, 6)),
            failure_stage="none",
            passed=True,
        )

    monkeypatch.setattr(scene_rollout_module, "validate_grasp_rollout", fake_rollout)
    observed = []
    result = run_scene_grasp_rollout(
        "unused.xml",
        [],
        ["tip_1", "tip_2"],
        target_object_id="target",
        non_target_objects=[],
        protocol_hash="a" * 64,
        recipe_hash="b" * 64,
        source_hash="c" * 64,
        evidence_stage_observer=lambda stage, model, data: observed.append((stage, float(data.time))),
    )
    assert result.validation.passed
    assert [stage for stage, _ in observed] == [
        "initial",
        "squeeze",
        "lift",
        "perturbation",
    ]
    assert result.state_hashes == result.validation.trajectory_metrics["scene_state_hashes"]
