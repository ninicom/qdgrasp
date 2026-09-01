"""COR-03: the guard against mixed hands is not on the path that collates.

The old side adapter guarded mixed robots, while the Runner used a different
collator.  The canonical Runner collator now owns the identity check and carries
the robot profile and joint order with every batch.
"""

from __future__ import annotations

from _corrective_support import characterization, refuses, sample


@characterization("COR-03", note="the Runner collates with default_collate")
def test_the_runner_collator_refuses_a_batch_that_mixes_hands() -> None:
    from qdgrasp.engine.sampling import collate_indices

    items = [sample(robot_name="leap_hand"), sample(robot_name="wonik_allegro")]

    refuses(
        lambda: collate_indices(items, [0, 1]),
        because=(
            "a LEAP sample and an Allegro sample collated into one batch; both hands have sixteen joints, so "
            "nothing downstream can notice that half the batch is being evaluated against the wrong graph"
        ),
    )


@characterization("COR-03", note="the collated batch carries no profile identity")
def test_a_collated_batch_carries_the_hand_it_was_built_from() -> None:
    """§9.6: robot name, profile hash and ordered joint names travel with the batch."""

    from qdgrasp.engine.sampling import collate_indices

    items = [sample(robot_name="leap_hand"), sample(robot_name="leap_hand")]
    batch = collate_indices(items, [0, 1])

    missing = [key for key in ("robot_name", "robot_profile_hash", "joint_names") if key not in batch]
    assert not missing, (
        f"the collated batch is missing {missing}; without profile identity a model cannot assert that the "
        "joints it is about to regress are the joints the batch was built from"
    )
