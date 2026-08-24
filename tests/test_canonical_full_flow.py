import hashlib

import numpy as np

from qdgrasp.dataset.pipeline.canonical_full_flow import (
    build_canonical_full_flow_matrix,
)


def _mesh_hash(mesh) -> str:
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(mesh.vertices, dtype=np.float64).tobytes())
    digest.update(np.ascontiguousarray(mesh.faces, dtype=np.int64).tobytes())
    return digest.hexdigest()


def test_canonical_matrix_has_four_pinned_hand_independent_families():
    first = build_canonical_full_flow_matrix()
    second = build_canonical_full_flow_matrix()

    assert [item.name.split("_", 1)[0] for item in first] == [
        "box",
        "cylinder",
        "superquadric",
        "compound",
    ]
    assert [_mesh_hash(item.mesh) for item in first] == [
        _mesh_hash(item.mesh) for item in second
    ]
    assert all(item.mass == 0.005 for item in first)
    assert all(item.object_pos[2] >= 0.0 for item in first)
