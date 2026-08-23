import numpy as np
import trimesh
from qdgrasp.dataset.pipeline.contracts import ContactProposal
from qdgrasp.dataset.pipeline.proposals.surface_fixed import generate_surface_fixed_proposal

def generate_region_opposition_proposal(
    mesh: trimesh.Trimesh,
    num_fingers: int,
    rng: np.random.Generator,
    finger_ids: np.ndarray,
    thumb_index: int = 0,
    max_retries: int = 10,
) -> ContactProposal:
    """
    Generate region anchors for fingers, explicitly separating a thumb from opposing fingers.
    This creates scale-normalized regions ensuring opposition without semantic models.
    """
    if len(finger_ids) != num_fingers:
        raise ValueError(f"Expected {num_fingers} finger_ids, got {len(finger_ids)}")

    if num_fingers < 2:
        # Fallback to surface fixed if we can't form an opposition
        return generate_surface_fixed_proposal(mesh, num_fingers, rng, finger_ids)

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

            # 4. Generate points for these faces
            r1 = rng.random(num_fingers)
            r2 = rng.random(num_fingers)

            sqrt_r1 = np.sqrt(r1)
            u = 1.0 - sqrt_r1
            v = r2 * sqrt_r1
            w = 1.0 - u - v

            barycentric = np.vstack([u, v, w]).T
            vertices = mesh.vertices[mesh.faces[final_face_ids]]
            target_points = np.einsum('ij,ijk->ik', barycentric, vertices)

            final_inward_normals = inward_normals[final_face_ids]

            # Normalize
            norms = np.linalg.norm(final_inward_normals, axis=1, keepdims=True)
            final_inward_normals = final_inward_normals / np.clip(norms, a_min=1e-12, a_max=None)

            return ContactProposal(
                target_points=target_points,
                face_ids=final_face_ids,
                inward_normals=final_inward_normals,
                finger_ids=finger_ids,
                provenance="region_opposition"
            )

    # Fallback if no opposition could be found after max_retries
    return generate_surface_fixed_proposal(mesh, num_fingers, rng, finger_ids)
