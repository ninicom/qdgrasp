import dataclasses

import numpy as np
import pytest

from qdgrasp.dataset.pipeline.contracts import (
    ContactProposal,
    DynamicValidation,
    PipelineOutcome,
    RegistryError,
    StaticCertificate,
    get_recipe,
)


def test_registry_allows_valid_recipes():
    """Test that the registry correctly returns valid recipes."""
    recipe = get_recipe("surface_fixed_v1")
    assert recipe["proposal"] == "surface_fixed"
    assert recipe["solver"] == "fixed_contact_dls"

def test_registry_rejects_invalid_recipes():
    """Test that the registry raises RegistryError for invalid recipes."""
    with pytest.raises(RegistryError):
        get_recipe("unapproved_recipe")

def test_contact_proposal_immutability():
    """Test that ContactProposal is frozen and immutable."""
    proposal = ContactProposal(
        target_points=np.zeros((4, 3)),
        face_ids=np.zeros(4, dtype=int),
        inward_normals=np.zeros((4, 3)),
        finger_ids=np.arange(4)
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        proposal.provenance = "hacked"

def test_pipeline_outcome_retains_rejection_reason():
    """Ensure PipelineOutcome forces explicit rejection reasons."""
    outcome = PipelineOutcome(
        proposal_valid=True,
        ik_valid=False,
        collision_valid=False,
        static_force_valid=False,
        dynamic_valid=False,
        failure_stage="ik",
        failure_reason="non-converged"
    )
    assert outcome.failure_stage == "ik"
    assert not outcome.ik_valid


def test_static_certifier_contract():
    """Verify StaticCertificate data contract."""
    cert = StaticCertificate(
        force_solution=np.ones((4, 3)),
        cone_residual=1e-5,
        object_wrench=np.zeros(6),
        quality_margin=0.12,
        passed=True
    )
    assert cert.passed
    assert cert.quality_margin == 0.12

def test_dynamic_validator_contract():
    """Verify DynamicValidation data contract."""
    val = DynamicValidation(
        trajectory_metrics={"lift_achieved": 0.05, "max_penetration": 0.001},
        per_finger_loads=np.ones((4, 6)),
        failure_stage="none",
        passed=True
    )
    assert val.passed
    assert val.failure_stage == "none"
    assert val.trajectory_metrics["lift_achieved"] == 0.05
