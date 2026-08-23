import mujoco
from qdgrasp.scenes.contracts import SceneSpec
from qdgrasp.scenes.builders.base import build_base_mujoco_model

def build_replay_scene(spec: SceneSpec) -> tuple[mujoco.MjModel, mujoco.MjData]:
    """
    Reconstructs an imported scene purely from the static poses defined in SceneSpec.
    This doesn't run physics simulation, it just places objects where the external
    dataset specified them.
    """
    model = build_base_mujoco_model(spec)
    data = mujoco.MjData(model)

    # Ideally, we would dynamically load object meshes into the model here.
    # For now, this is a skeleton for the interface.
    mujoco.mj_forward(model, data)
    return model, data
