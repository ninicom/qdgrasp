import numpy as np
import mujoco

class ClearanceError(Exception):
    def __init__(self, reason: str, details: str):
        self.reason = reason
        self.details = details
        super().__init__(f"{reason}: {details}")

def check_approach_clearance(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    target_object_id: str,
    approach_path: np.ndarray,
    hand_geom_ids: list[int]
) -> bool:
    """
    Checks if the hand along the approach path collides with any non-target
    object or the environment.

    Args:
        model: MjModel of the scene.
        data: MjData of the scene (with the hand already spawned but not colliding at start).
        target_object_id: ID of the target object.
        approach_path: [N, 4, 4] trajectory of the hand root.
        hand_geom_ids: List of geom IDs belonging to the hand.

    Raises:
        ClearanceError: If a collision is detected.
    Returns:
        True if the path is clear.
    """
    # Note: A real implementation would interpolate the approach path and
    # step kinematics (mj_kinematics or mj_forward without integration)
    # then check data.contact.

    # Mock skeleton for the check:
    if len(approach_path) == 0:
        raise ClearanceError("approach_blocked", "Approach path is empty")

    # Simulate swept check by iterating through path (mock loop)
    for step_idx, T in enumerate(approach_path):
        # 1. Update hand root pose to T
        # 2. mujoco.mj_kinematics(model, data)
        # 3. mujoco.mj_collision(model, data)

        # 4. Check contacts
        for i in range(data.ncon):
            contact = data.contact[i]
            geom1 = contact.geom1
            geom2 = contact.geom2

            # Check if one of the geoms is the hand
            is_hand1 = geom1 in hand_geom_ids
            is_hand2 = geom2 in hand_geom_ids

            if is_hand1 or is_hand2:
                other_geom = geom2 if is_hand1 else geom1

                # Check if other_geom belongs to the target object
                # (In a real implementation, map geom ID to body name to object ID)
                # If not target, we have a collision!

                # Mock: we assume no collision for this skeleton unless explicitly mocked
                pass

    return True
