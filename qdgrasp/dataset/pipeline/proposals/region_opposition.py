import numpy as np
import trimesh
from qdgrasp.dataset.pipeline.contracts import ContactProposal


def _sample_face_points(
    mesh: trimesh.Trimesh,
    face_ids: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample one exact barycentric point for every face id."""
    r1 = rng.random(len(face_ids))
    r2 = rng.random(len(face_ids))
    sqrt_r1 = np.sqrt(r1)
    barycentric = np.stack(
        [1.0 - sqrt_r1, r2 * sqrt_r1, (1.0 - r2) * sqrt_r1], axis=1
    )
    vertices = mesh.vertices[mesh.faces[face_ids]]
    return np.einsum("ij,ijk->ik", barycentric, vertices)

def generate_region_opposition_proposal(
    mesh: trimesh.Trimesh,
    num_fingers: int,
    rng: np.random.Generator,
    finger_ids: np.ndarray,
    thumb_index: int = 0,
    max_retries: int = 10,
    region_size: int = 32,
) -> ContactProposal:
    """
    Generate region anchors for fingers, explicitly separating a thumb from opposing fingers.
    This creates scale-normalized regions ensuring opposition without semantic models.
    """
    if len(finger_ids) != num_fingers:
        raise ValueError(f"Expected {num_fingers} finger_ids, got {len(finger_ids)}")

    if num_fingers < 2:
        raise ValueError("region_opposition requires at least two fingers")

    # 1. Prepare area-based probability for faces
    areas = mesh.area_faces
    total_area = np.sum(areas)
    if total_area <= 0:
        raise ValueError("Mesh has zero or negative surface area.")

    probabilities = areas / total_area
    cumulative_probs = np.cumsum(probabilities)

    # Pre-calculate face centers and inward normals
    face_centers = mesh.triangles.mean(axis=1)
    inward_normals = -mesh.face_normals

    # Bounding box scale to normalize distance checks
    scale = np.linalg.norm(mesh.extents)
    max_dist = scale * 0.8  # Max distance for opposing fingers
    min_dist = scale * 0.1  # Min distance for opposing fingers

    for _ in range(max_retries):
        # 2. Pick a face for the thumb anchor
        r_thumb = rng.random()
        thumb_face_id = np.searchsorted(cumulative_probs, r_thumb)
        thumb_face_id = np.clip(thumb_face_id, 0, len(mesh.faces) - 1)

        thumb_center = face_centers[thumb_face_id]
        thumb_normal = inward_normals[thumb_face_id]

        # 3. Find valid opposing faces
        # Condition 1: Dot product of inward normals < -0.2 (pointing roughly at each other)
        # Condition 2: Distance is within [min_dist, max_dist]

        vec_to_faces = face_centers - thumb_center
        distances = np.linalg.norm(vec_to_faces, axis=1)

        # Check opposition
        dots = np.sum(inward_normals * thumb_normal, axis=1)

        valid_mask = (dots < -0.2) & (distances > min_dist) & (distances < max_dist)
        valid_indices = np.where(valid_mask)[0]

        if len(valid_indices) > 0:
            # We found opposing faces! Pick (num_fingers - 1) faces from them
            # We can re-weight probabilities among valid_indices, or just sample uniformly from valid for simplicity
            # We'll use uniform sampling among valid for simplicity of anchor generation
            opposing_face_ids = rng.choice(valid_indices, size=num_fingers - 1, replace=True)

            # Combine
            final_face_ids = np.zeros(num_fingers, dtype=int)

            # Assign thumb
            final_face_ids[thumb_index] = thumb_face_id

            # Assign opposing
            opposing_idx = 0
            for i in range(num_fingers):
                if i != thumb_index:
                    final_face_ids[i] = opposing_face_ids[opposing_idx]
                    opposing_idx += 1

            # 4. Build fixed-size regions from nearby, similarly oriented faces.
            # Every region sample remains an exact barycentric point on the mesh.
            region_face_ids = np.empty((num_fingers, region_size), dtype=np.int64)
            region_points = np.empty((num_fingers, region_size, 3), dtype=np.float64)
            region_normals = np.empty((num_fingers, region_size, 3), dtype=np.float64)
            region_radius = max(scale * 0.35, 1e-6)
            for finger_idx, anchor_face in enumerate(final_face_ids):
                center_delta = np.linalg.norm(face_centers - face_centers[anchor_face], axis=1)
                normal_dot = inward_normals @ inward_normals[anchor_face]
                nearby = np.where((center_delta <= region_radius) & (normal_dot >= 0.8))[0]
                if len(nearby) == 0:
                    nearby = np.array([anchor_face], dtype=np.int64)
                chosen = rng.choice(nearby, size=region_size, replace=len(nearby) < region_size)
                region_face_ids[finger_idx] = chosen
                region_points[finger_idx] = _sample_face_points(mesh, chosen, rng)
                region_normals[finger_idx] = inward_normals[chosen]

            target_points = region_points[:, 0]

            final_inward_normals = inward_normals[final_face_ids]

            # Normalize
            norms = np.linalg.norm(final_inward_normals, axis=1, keepdims=True)
            final_inward_normals = final_inward_normals / np.clip(norms, a_min=1e-12, a_max=None)

            return ContactProposal(
                target_points=target_points,
                face_ids=final_face_ids,
                inward_normals=final_inward_normals,
                finger_ids=finger_ids,
                region_points=region_points,
                region_face_ids=region_face_ids,
                region_normals=region_normals,
                provenance="region_opposition"
            )

    raise ValueError("could not find an opposing surface region")
