import numpy as np
import trimesh
from qdgrasp.dataset.pipeline.contracts import ContactProposal

def generate_surface_fixed_proposal(
    mesh: trimesh.Trimesh,
    num_fingers: int,
    rng: np.random.Generator,
    finger_ids: np.ndarray,
) -> ContactProposal:
    """
    Deterministically sample 'num_fingers' points on the surface of 'mesh'
    using area-weighted probabilities, without calling trimesh.sample.sample_surface
    which uses a global/implicit RNG.
    """
    if len(finger_ids) != num_fingers:
        raise ValueError(f"Expected {num_fingers} finger_ids, got {len(finger_ids)}")

    # 1. Calculate cumulative area for each face
    areas = mesh.area_faces
    total_area = np.sum(areas)
    if total_area <= 0:
        raise ValueError("Mesh has zero or negative surface area.")
        
    probabilities = areas / total_area
    cumulative_probs = np.cumsum(probabilities)
    
    # 2. Pick faces
    r_faces = rng.random(num_fingers)
    face_ids = np.searchsorted(cumulative_probs, r_faces)
    # Handle numerical roundoff edge cases where searchsorted might go out of bounds
    face_ids = np.clip(face_ids, 0, len(mesh.faces) - 1)
    
    # 3. Generate barycentric coordinates for the triangles
    r1 = rng.random(num_fingers)
    r2 = rng.random(num_fingers)
    
    # Square root transformation to ensure uniform sampling within the triangle
    sqrt_r1 = np.sqrt(r1)
    u = 1.0 - sqrt_r1
    v = r2 * sqrt_r1
    w = 1.0 - u - v
    
    # [K, 3]
    barycentric = np.vstack([u, v, w]).T
    
    # 4. Compute target points from vertices
    vertices = mesh.vertices[mesh.faces[face_ids]]  # [K, 3, 3]
    target_points = np.einsum('ij,ijk->ik', barycentric, vertices) # [K, 3]
    
    # 5. Get face inward normals
    # Trimesh normals point outward, so we negate them to get inward normals
    face_normals = mesh.face_normals[face_ids]
    inward_normals = -face_normals
    
    # Normalize just in case
    norms = np.linalg.norm(inward_normals, axis=1, keepdims=True)
    inward_normals = inward_normals / np.clip(norms, a_min=1e-12, a_max=None)
    
    return ContactProposal(
        target_points=target_points,
        face_ids=face_ids,
        inward_normals=inward_normals,
        finger_ids=finger_ids,
        provenance="surface_fixed"
    )
