"""COR-08: resume matches tensors, not runs.

``ResumeState`` carries weights, optimiser, scheduler, EMA and RNG.  It does not
carry the model configuration, the robot it was trained on, the data or the
protocol, so the only thing standing between a LEAP run and an Allegro run is
that both hands have sixteen actuated joints.  They do, so nothing stands there.

``scaler`` is the other half: the field is named for the AMP scaler and holds
the AMP *flag* plus the batch stream.  A mixed-precision run that resumes
therefore starts with a fresh scale factor and a few wasted steps of overflow
recovery, which is a small error made invisible by a field that looks handled.
"""

from __future__ import annotations

import torch
from _corrective_support import characterization, refuses


def _train(robot: str, run_name: str):
    from qdgrasp.api import QDGrasp

    grasper = QDGrasp("qdgrasp-dummy-n.yaml", robot=robot, seed=0)
    result = grasper.train(
        "dummy-tiny.yaml",
        max_steps=4,
        run_name=run_name,
        project_dir="runs",
    )
    return grasper, result


@characterization("COR-08", note="a LEAP resume is accepted into an Allegro run")
def test_resume_refuses_state_trained_on_another_hand() -> None:
    from qdgrasp.api import QDGrasp

    _leap, result = _train("leap_hand.yaml", "corrective-resume-leap")
    allegro = QDGrasp("qdgrasp-dummy-n.yaml", robot="wonik_allegro.yaml", seed=0)

    refuses(
        lambda: allegro.train(
            "dummy-tiny.yaml",
            max_steps=8,
            resume=result.artifacts["resume"],
            run_name="corrective-resume-allegro",
            project_dir="runs",
        ),
        because=(
            "an Allegro run continued from LEAP state because both hands have sixteen joints; resume matched "
            "tensor shapes where it needed to match the run"
        ),
    )


@characterization("COR-08", note="the resume artifact records no run identity")
def test_a_resume_artifact_says_which_run_it_continues() -> None:
    _grasper, result = _train("leap_hand.yaml", "corrective-resume-identity")
    payload = torch.load(result.artifacts["resume"], map_location="cpu", weights_only=True)

    required = ("model_config_hash", "robot_config_hash", "data_manifest_hash", "effective_run_config")
    missing = [key for key in required if key not in payload]
    assert not missing, (
        f"the resume artifact carries no {missing}; without them 'continue this run' cannot be distinguished "
        "from 'start a different run from these tensors'"
    )


@characterization("COR-08", note="the AMP scaler state is never written")
def test_a_resume_artifact_records_the_amp_scaler_it_was_running() -> None:
    _grasper, result = _train("leap_hand.yaml", "corrective-resume-scaler")
    payload = torch.load(result.artifacts["resume"], map_location="cpu", weights_only=True)
    scaler = payload["scaler"]

    assert "grad_scaler" in scaler, (
        f"the 'scaler' field holds {sorted(scaler)}: the AMP flag and the batch stream, but not the scale "
        "factor the run was actually using, so a mixed-precision resume restarts its overflow search"
    )
