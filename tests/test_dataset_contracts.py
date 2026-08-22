import pytest
import dataclasses
import numpy as np
from qdgrasp.dataset.pipeline.contracts import (
    ContactProposal,
    KinematicSolution,
    StaticCertificate,
    DynamicValidation,
    PipelineOutcome,
    get_recipe,
    RegistryError
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

# Failing test by default to ensure modules implement this properly later
def test_static_certifier_fails_if_not_implemented():
    """Placeholder failing test to remind implementing certifier logic."""
    pytest.fail("Static certifier tests are not fully implemented yet.")

def test_dynamic_validator_fails_if_not_implemented():
    """Placeholder failing test for dynamic validator."""
    pytest.fail("Dynamic validator tests are not fully implemented yet.")
