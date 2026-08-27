"""S6 — what it takes to release a positive (G04, B-05).

The failure pinned here is that agreement was treated as admission. Two
backends that both say "insufficient lift" have confirmed each other's failure;
v1 folded that into ``certified`` and let a matching pair of failures reach the
release path.

The rest is provenance: a release decision now re-checks the original outcomes
rather than accepting two loose booleans, refuses a certificate that was issued
against a different capsule or a different compiled model, and keeps a ledger
whose stages each have their own denominator.
"""

from __future__ import annotations

import dataclasses
import hashlib
from pathlib import Path

import mujoco
import numpy as np
import pytest

from qdgrasp.dataset.dynamic_contracts import (
    ContactClass,
    ContactPairKind,
    CpuReplayCertificate,
    DynamicGraspTrajectory,
    DynamicSearchOutcome,
    TrajectoryStage,
    TrajectoryTimebase,
)
from qdgrasp.dynamic.capsule import capture_capsule
from qdgrasp.dynamic.certify import (
    ReleaseLedger,
    certify_replay,
    certify_terminal_grasp,
    release_decision,
)

MICRO_SCENE = (
    Path(__file__).resolve().parents[1] / "dynamic_grasp" / "micro_scene.xml"
).read_text(encoding="utf-8")
MODEL_SHA = hashlib.sha256(MICRO_SCENE.encode("utf-8")).hexdigest()
SAMPLE_PERIOD_S = 0.01


def certificate(**over) -> CpuReplayCertificate:
    defaults = {
        "backend_id": "mujoco_cpu",
        "capsule_sha256": "a" * 64,
        "command_sha256": "b" * 64,
        "model_sha256": "c" * 64,
        "timestep_s": 0.002,
        "terminal_certified": True,
        "safety_certified": True,
        "outcome_class": "pass",
    }
    defaults.update(over)
    return CpuReplayCertificate(**defaults)


def _event(time_index: int, kind: ContactPairKind, contact_class: ContactClass, **over):
    from qdgrasp.dataset.dynamic_contracts import ContactEvent

    defaults = {
        "time_index": time_index,
        "contact_class": contact_class,
        "geom_a": "tip",
        "geom_b": "target",
        "body_a": "distal_0",
        "body_b": "target",
        "point": np.zeros(3),
        "frame": np.eye(3),
        "normal_force_N": 1.0,
        "tangential_force_N": 0.0,
        "normal_impulse_Ns": 0.0,
        "tangential_impulse_Ns": 0.0,
        "penetration_m": 0.0,
        "relative_velocity_mps": 0.0,
        "slip_m": 0.0,
        "work_J": 0.0,
        "budget_margin": 0.5,
        "pair_kind": kind,
    }
    defaults.update(over)
    return ContactEvent(**defaults)


def trajectory(*, lift: float = 0.05, steps: int = 4, events=None) -> DynamicGraspTrajectory:
    palm = np.zeros((steps, 7))
    palm[:, 3] = 1.0
    pose = np.zeros((steps, 1, 7))
    pose[:, :, 3] = 1.0
    pose[-1, 0, 2] = pose[0, 0, 2] + lift
    if events is None:
        events = tuple(
            _event(0, ContactPairKind.TARGET_ROBOT, ContactClass.TARGET_INTENTIONAL,
                   body_a=f"distal_{i}")
            for i in range(2)
        )
    return DynamicGraspTrajectory(
        time=np.arange(steps, dtype=float) * SAMPLE_PERIOD_S,
        palm_pose=palm,
        joint_state=np.zeros((steps, 16)),
        actuator_command=np.zeros((steps, 16)),
        object_pose=pose,
        object_velocity=np.zeros((steps, 1, 6)),
        stage=(
            TrajectoryStage.ENCLOSE,
            TrajectoryStage.SUPPORT_RELEASE,
            TrajectoryStage.LIFT,
            TrajectoryStage.PERTURB,
        )[:steps],
        timebase=TrajectoryTimebase(simulator_dt=SAMPLE_PERIOD_S, sample_every=1),
        contact_graph=events,
    )


def passing(**over) -> DynamicSearchOutcome:
    defaults = {
        "trajectory_ref": "t:0",
        "passed": True,
        "failure_stage": "none",
        "failure_reason": "none",
        "cpu_replay_evidence": certificate(),
    }
    defaults.update(over)
    return DynamicSearchOutcome(**defaults)


def failing(reason: str = "insufficient_lift") -> DynamicSearchOutcome:
    return DynamicSearchOutcome(
        trajectory_ref="t:0", passed=False, failure_stage="lift", failure_reason=reason
    )


# -- matching failures ----------------------------------------------------


def test_two_matching_failures_are_parity_evidence_not_a_certificate() -> None:
    result = certify_replay(
        search_outcome=failing(),
        replay_outcome=failing(),
        search_trajectory=trajectory(),
        replay_trajectory=trajectory(),
    )
    assert result.parity_confirmed is True
    assert result.certified is False
    assert result.reason == "insufficient_lift"


def test_a_matching_failure_cannot_be_released() -> None:
    parity = certify_replay(
        search_outcome=failing("support_not_released"),
        replay_outcome=failing("support_not_released"),
        search_trajectory=trajectory(),
        replay_trajectory=trajectory(),
    )
    decision = release_decision(
        replay=parity,
        terminal=certify_terminal_grasp(trajectory()),
        certificate=certificate(),
    )
    assert not decision.passed
    assert decision.failure_reason == "support_not_released"


def test_a_matching_pass_certifies() -> None:
    result = certify_replay(
        search_outcome=passing(),
        replay_outcome=passing(),
        search_trajectory=trajectory(),
        replay_trajectory=trajectory(),
    )
    assert result.certified
    assert result.parity_confirmed
    assert result.outcome_class == "pass"


def test_divergence_is_still_divergence() -> None:
    result = certify_replay(
        search_outcome=passing(),
        replay_outcome=failing(),
        search_trajectory=trajectory(),
        replay_trajectory=trajectory(),
    )
    assert not result.certified
    assert not result.parity_confirmed
    assert result.reason == "backend_divergence"


# -- provenance -----------------------------------------------------------


def good_replay():
    return certify_replay(
        search_outcome=passing(),
        replay_outcome=passing(),
        search_trajectory=trajectory(),
        replay_trajectory=trajectory(),
    )


def test_release_rechecks_the_original_outcomes() -> None:
    decision = release_decision(
        replay=good_replay(),
        terminal=certify_terminal_grasp(trajectory()),
        certificate=certificate(),
        search_outcome=failing("damaging_contact"),
        replay_outcome=passing(),
    )
    assert not decision.passed
    assert decision.failure_reason == "damaging_contact"


def test_release_rechecks_the_replay_outcome_too() -> None:
    decision = release_decision(
        replay=good_replay(),
        terminal=certify_terminal_grasp(trajectory()),
        certificate=certificate(),
        search_outcome=passing(),
        replay_outcome=failing("perturbation_slip"),
    )
    assert not decision.passed
    assert decision.failure_reason == "perturbation_slip"


def test_release_accepts_matching_originals() -> None:
    decision = release_decision(
        replay=good_replay(),
        terminal=certify_terminal_grasp(trajectory()),
        certificate=certificate(),
        search_outcome=passing(),
        replay_outcome=passing(),
    )
    assert decision.passed
    assert decision.cpu_replay_evidence.is_positive


def test_a_certificate_bound_to_another_capsule_is_refused() -> None:
    model = mujoco.MjModel.from_xml_string(MICRO_SCENE)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    capsule = capture_capsule(
        model,
        data,
        capsule_id="capsule:0",
        robot_profile="micro_pusher",
        scene_signature="bucket:micro",
        model_sha256=MODEL_SHA,
        control_sequence=np.zeros((5, int(model.nu))),
        seed=1,
        strategy_id="primitive_sequence",
        strategy_parameters={},
        safety_budget_id="micro-conservative-v1",
        safety_budget_hash="d" * 64,
    )
    stale = certificate()  # placeholder hashes, not this capsule's
    decision = release_decision(
        replay=good_replay(),
        terminal=certify_terminal_grasp(trajectory()),
        certificate=stale,
        capsule=capsule,
    )
    assert not decision.passed
    assert decision.failure_reason == "evidence_hash_mismatch"

    bound = certificate(
        capsule_sha256=capsule.capsule_sha256,
        command_sha256=capsule.command_sha256,
        model_sha256=capsule.model.model_sha256,
    )
    assert release_decision(
        replay=good_replay(),
        terminal=certify_terminal_grasp(trajectory()),
        certificate=bound,
        capsule=capsule,
    ).passed


def test_a_certificate_that_did_not_certify_cannot_release() -> None:
    decision = release_decision(
        replay=good_replay(),
        terminal=certify_terminal_grasp(trajectory()),
        certificate=certificate(safety_certified=False),
    )
    assert not decision.passed
    assert decision.failure_reason == "evidence_hash_mismatch"


def test_a_cuda_backend_can_never_issue_the_cpu_certificate() -> None:
    with pytest.raises(ValueError, match="cannot name a CUDA backend"):
        certificate(backend_id="mjwarp_cuda")


def test_an_unsafe_replay_is_refused_even_when_the_classes_match() -> None:
    unsafe = trajectory(
        events=(
            _event(0, ContactPairKind.TARGET_ROBOT, ContactClass.TARGET_INTENTIONAL),
            _event(1, ContactPairKind.TARGET_ROBOT, ContactClass.DAMAGING, body_a="distal_1"),
        )
    )
    result = certify_replay(
        search_outcome=passing(),
        replay_outcome=passing(),
        search_trajectory=trajectory(),
        replay_trajectory=unsafe,
    )
    assert not result.certified
    assert result.reason == "replay_violates_safety_budget"


# -- ledger ---------------------------------------------------------------


def test_the_ledger_keeps_a_denominator_per_stage() -> None:
    ledger = ReleaseLedger()
    terminal = certify_terminal_grasp(trajectory())

    # One released.
    ledger.record_candidate(
        search_outcome=passing(),
        gpu_survived=True,
        replay=good_replay(),
        released=release_decision(
            replay=good_replay(), terminal=terminal, certificate=certificate()
        ),
    )
    # One replayed and refused on parity.
    parity = certify_replay(
        search_outcome=failing(),
        replay_outcome=failing(),
        search_trajectory=trajectory(),
        replay_trajectory=trajectory(),
    )
    ledger.record_candidate(search_outcome=failing(), gpu_survived=True, replay=parity)
    # One never replayed at all.
    ledger.record_candidate(search_outcome=failing("damaging_contact"))

    assert ledger.searched == 3
    assert ledger.gpu_survived == 2
    assert ledger.replayed == 2
    assert ledger.cpu_confirmed == 1
    assert ledger.released == 1
    assert ledger.parity_confirmed_failures == 1
    assert ledger.reconcile()


def test_a_sample_that_was_never_replayed_is_not_in_the_released_denominator() -> None:
    ledger = ReleaseLedger()
    for _ in range(9):
        ledger.record_candidate(search_outcome=failing("damaging_contact"))
    ledger.record_candidate(
        search_outcome=passing(),
        replay=good_replay(),
        released=release_decision(
            replay=good_replay(),
            terminal=certify_terminal_grasp(trajectory()),
            certificate=certificate(),
        ),
    )
    # One released out of one replayed, not one out of ten sampled.
    assert ledger.release_rate == pytest.approx(1.0)
    assert ledger.searched == 10
    assert ledger.reconcile()


def test_the_ledger_reports_dispositions_that_add_up() -> None:
    ledger = ReleaseLedger()
    ledger.record_candidate(search_outcome=failing("insufficient_lift"))
    ledger.record_candidate(search_outcome=failing("damaging_contact"))
    payload = ledger.to_dict()
    assert sum(payload["dispositions"].values()) == payload["searched"]
    assert payload["reconciled"] is True


def test_the_ledger_notices_an_impossible_count() -> None:
    ledger = ReleaseLedger()
    ledger.record_candidate(search_outcome=failing())
    ledger = dataclasses.replace(ledger, released=5)
    assert not ledger.reconcile()
