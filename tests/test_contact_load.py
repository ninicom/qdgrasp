import mujoco
import numpy as np

from qdgrasp.dataset.pipeline.observers.contact_load import extract_contact_loads


def test_contact_force_sign_and_per_finger_sum_match_object_wrench():
    model = mujoco.MjModel.from_xml_string(
        """
        <mujoco>
          <option timestep="0.002" gravity="0 0 -9.81"/>
          <worldbody>
            <body name="tip_0" pos="0 0 0.01">
              <geom name="finger_geom" type="box" size="0.03 0.03 0.01"/>
            </body>
            <body name="target_object" pos="0 0 0.06">
              <freejoint/>
              <geom name="object_subgeom_0" type="sphere" size="0.02" mass="0.1"/>
            </body>
          </worldbody>
        </mujoco>
        """
    )
    data = mujoco.MjData(model)
    for _ in range(300):
        mujoco.mj_step(model, data)

    object_geom = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_GEOM, "object_subgeom_0"
    )
    loads = extract_contact_loads(
        model,
        data,
        {object_geom},
        ["tip_0"],
        palm_body_names=(),
    )

    assert loads["active_fingers_count"] == 1
    assert loads["net_wrench"][2] > 0.0
    np.testing.assert_allclose(
        loads["per_finger_forces"].sum(axis=0),
        loads["net_wrench"][:3],
        rtol=1e-6,
        atol=1e-8,
    )
    np.testing.assert_allclose(
        loads["net_fingertip_wrench"],
        loads["per_finger_loads"].sum(axis=0),
        rtol=1e-6,
        atol=1e-8,
    )
