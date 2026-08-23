import mujoco
import numpy as np

def render_camera_view(model: mujoco.MjModel, data: mujoco.MjData, camera_name: str, width: int = 640, height: int = 480) -> dict[str, np.ndarray]:
    """
    Renders RGB, Depth, and Instance Mask from a specified MuJoCo camera.
    """
    # Create the renderer
    renderer = mujoco.Renderer(model, height, width)

    # Render RGB
    renderer.update_scene(data, camera=camera_name)
    rgb = renderer.render()

    # Enable segmentation/depth rendering flags
    renderer.enable_depth_rendering()
    renderer.update_scene(data, camera=camera_name)
    depth = renderer.render()

    renderer.disable_depth_rendering()
    renderer.enable_segmentation_rendering()
    renderer.update_scene(data, camera=camera_name)
    seg = renderer.render()

    # Calculate visibility (mocked simple ratio for now)
    # Extract instance IDs from the seg mask
    geom_ids = seg[:, :, 0]
    unique_ids, counts = np.unique(geom_ids, return_counts=True)
    total_pixels = height * width

    visibility = {}
    for uid, count in zip(unique_ids, counts):
        if uid >= 0:
            visibility[str(uid)] = count / total_pixels

    return {
        "rgb": rgb,
        "depth": depth,
        "segmentation": seg,
        "visibility": visibility
    }
