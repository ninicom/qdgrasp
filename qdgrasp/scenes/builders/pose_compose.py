import mujoco
from qdgrasp.scenes.contracts import SceneSpec
from qdgrasp.scenes.builders.base import build_base_mujoco_model

def compose_scene(spec: SceneSpec) -> tuple[mujoco.MjModel, mujoco.MjData]:
    """
    Composes a scene from a layout template, applying a short burst of physics
    steps to repair any minor collisions (interpenetrations) between objects.
    """
    model = build_base_mujoco_model(spec)
    data = mujoco.MjData(model)

    # Ideally add objects here

    # Repair steps
    for _ in range(50):
        mujoco.mj_step(model, data)

    return model, data
