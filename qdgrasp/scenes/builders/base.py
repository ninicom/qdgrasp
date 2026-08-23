import mujoco
from qdgrasp.scenes.contracts import SceneSpec

def build_base_mujoco_model(spec: SceneSpec) -> mujoco.MjModel:
    """
    Builds the XML for a MuJoCo model from a SceneSpec, without any objects
    (just environment supports and global settings).
    """
    xml = [
        '<mujoco>',
        '  <option timestep="{}" gravity="{} {} {}"/>'.format(
            spec.timestep, *spec.gravity
        ),
        '  <asset>',
        '    <material name="support_mat" rgba="0.8 0.8 0.8 1"/>',
        '  </asset>',
        '  <worldbody>',
    ]

    # Add supports
    for support in spec.supports:
        if support.geom_type == "box":
            size = support.params.get("size", [1, 1, 1])
            half_size = [s/2.0 for s in size]
            pos = support.T_world_support[:3, 3]
            xml.append(
                f'    <body name="{support.support_id}" pos="{pos[0]} {pos[1]} {pos[2]}">'
                f'      <geom type="box" size="{half_size[0]} {half_size[1]} {half_size[2]}" '
                f'            material="support_mat" contype="1" conaffinity="1"/>'
                f'    </body>'
            )

    # Add cameras
    for cam in spec.cameras:
        pos = cam.T_world_camera[:3, 3]
        # In mujoco, cameras look down -z. Here we mock a generic camera.
        xml.append(
            f'    <camera name="{cam.camera_id}" pos="{pos[0]} {pos[1]} {pos[2]}" '
            f'            mode="fixed" fovy="45"/>'
        )

    xml.append('  </worldbody>')
    xml.append('</mujoco>')

    xml_str = "\n".join(xml)
    return mujoco.MjModel.from_xml_string(xml_str)
