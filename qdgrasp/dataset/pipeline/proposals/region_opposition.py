import numpy as np
import trimesh
from qdgrasp.dataset.pipeline.contracts import ContactProposal
from qdgrasp.dataset.pipeline.proposals.identity import stable_candidate_id


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
    opposing_finger_index: int | None = None,
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
    floor_z = float(mesh.bounds[0, 2])
    face_centers = mesh.triangles.mean(axis=1)
    floor_support_faces = (
        (
            face_centers[:, 2]
            <= floor_z + max(float(np.linalg.norm(mesh.extents)) * 1e-6, 1e-9)
        )
        & (mesh.face_normals[:, 2] < 0.0)
    )
    probabilities = np.where(floor_support_faces, 0.0, probabilities)
    if float(np.sum(probabilities)) <= 0.0:
        raise ValueError("region_opposition has no floor-accessible faces")
    probabilities /= np.sum(probabilities)
    cumulative_probs = np.cumsum(probabilities)

    # Pre-calculate face centers and inward normals
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

        valid_mask = (
            (dots < -0.2)
            & (distances > min_dist)
            & (distances < max_dist)
            & ~floor_support_faces
        )
        valid_indices = np.where(valid_mask)[0]

        if len(valid_indices) > 0:
            non_thumb_fingers = np.array(
                [index for index in range(num_fingers) if index != thumb_index],
                dtype=np.int64,
            )
            # The proposal contract requires two opposing finger groups, not
            # an arbitrary three-fingertip task.  Use one thumb and one opposing
            # finger; force closure and sustained-contact gates independently
            # decide whether that pinch is physically sufficient.
            active_opposing_count = 1
            if opposing_finger_index is None:
                opposing_finger_index = int(non_thumb_fingers[0])
            if opposing_finger_index not in non_thumb_fingers:
                raise ValueError(
                    "opposing_finger_index must identify a non-thumb fingertip"
                )
            active_opposing_fingers = np.array(
                [opposing_finger_index], dtype=np.int64
            )
            active_fingers = np.zeros(num_fingers, dtype=bool)
            active_fingers[thumb_index] = True
            active_fingers[active_opposing_fingers] = True

            # We found opposing faces! Pick (num_fingers - 1) faces from them
            # Active fingers need distinct, spatially separated anchors.  Inactive
            # fingers still receive a surface target to keep the dense contract,
            # but those targets are not part of the task.
            min_active_spacing = max(scale * 0.05, 1e-6)
            ordered_faces = np.asarray(rng.permutation(valid_indices), dtype=np.int64)
            active_faces: list[int] = []
            for face_id in ordered_faces:
                if all(
                    np.linalg.norm(face_centers[face_id] - face_centers[chosen])
                    >= min_active_spacing
                    for chosen in active_faces
                ):
                    active_faces.append(int(face_id))
                    if len(active_faces) == active_opposing_count:
                        break
            if len(active_faces) < active_opposing_count:
                continue

            inactive_opposing_fingers = np.array(
                [
                    index
                    for index in non_thumb_fingers
                    if index not in set(active_opposing_fingers.tolist())
                ],
                dtype=np.int64,
            )

            # Combine
            final_face_ids = np.zeros(num_fingers, dtype=int)

            # Assign thumb
            final_face_ids[thumb_index] = thumb_face_id

            for finger_index, face_id in zip(active_opposing_fingers, active_faces):
                final_face_ids[finger_index] = face_id
            if len(inactive_opposing_fingers):
                inactive_faces = rng.choice(
                    valid_indices,
                    size=len(inactive_opposing_fingers),
                    replace=len(valid_indices) < len(inactive_opposing_fingers),
                )
                final_face_ids[inactive_opposing_fingers] = inactive_faces

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

            opposition_pairs = np.stack(
                [
                    np.full(active_opposing_count, thumb_index, dtype=np.int64),
                    active_opposing_fingers,
                ],
                axis=1,
            )
            candidate_id = stable_candidate_id(
                "region_opposition",
                target_points=target_points,
                inward_normals=final_inward_normals,
                face_ids=final_face_ids,
                finger_ids=finger_ids,
                active_fingers=active_fingers,
                opposition_pairs=opposition_pairs,
            )

            return ContactProposal(
                target_points=target_points,
                face_ids=final_face_ids,
                inward_normals=final_inward_normals,
                finger_ids=finger_ids,
                active_fingers=active_fingers,
                opposition_pairs=opposition_pairs,
                candidate_id=candidate_id,
                region_points=region_points,
                region_face_ids=region_face_ids,
                region_normals=region_normals,
                provenance="region_opposition"
            )

    raise ValueError("could not find an opposing surface region")
