from __future__ import annotations

import tempfile
from pathlib import Path
import pytest

from qdgrasp.robot.normalize import normalize_urdf, sha256_file
from qdgrasp.sim.mujoco import MujocoSim


def test_allegro_urdf_normalization_reproducibility() -> None:
    src = Path(
        ".references/robot-assets/wonik-allegro-ros2/src/allegro_hand_controllers/urdf/allegro_hand_description_right_A.urdf"
    )
    if not src.is_file():
        pytest.skip("Allegro URDF reference not present")

    mesh_dir = Path(".references/robot-assets/wonik-allegro-ros2/src/allegro_hand_controllers/meshes").resolve()

    with tempfile.TemporaryDirectory() as tmpdir:
        out1 = Path(tmpdir) / "run1" / "allegro.urdf"
        out2 = Path(tmpdir) / "run2" / "allegro.urdf"

        m1 = normalize_urdf(
            src,
            out1,
            package_replacements={"package://allegro_hand_controllers/meshes": str(mesh_dir)},
            sanitize_inertias=True,
        )
        m2 = normalize_urdf(
            src,
            out2,
            package_replacements={"package://allegro_hand_controllers/meshes": str(mesh_dir)},
            sanitize_inertias=True,
        )

        assert m1["output_sha256"] == m2["output_sha256"]
        assert sha256_file(out1) == sha256_file(out2)
        assert m1["modified"] is True
        assert "resolve_package_uris" in m1["transforms_applied"]
        assert "sanitize_inertias" in m1["transforms_applied"]

        # Verify it loads in MuJoCo
        sim = MujocoSim(out1)
        sim.forward()
        assert sim.nq == 16
