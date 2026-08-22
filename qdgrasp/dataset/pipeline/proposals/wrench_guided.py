import numpy as np
import trimesh
from qdgrasp.dataset.pipeline.contracts import ContactProposal
from qdgrasp.dataset.pipeline.proposals.region_opposition import generate_region_opposition_proposal

def compute_preliminary_wrench_score(target_points: np.ndarray, inward_normals: np.ndarray, centroid: np.ndarray) -> float:
    """
    Compute a preliminary wrench score based on the Minimum Singular Value 
    of the Grasp Matrix (G). 
    A higher minimum singular value implies a more robust grasp configuration.
    """
    K = target_points.shape[0]
    G = np.zeros((6, K))
    
    for i in range(K):
        # Force direction
        n = inward_normals[i]
        # Moment arm (cross product of radius vector and normal)
        r = target_points[i] - centroid
        tau = np.cross(r, n)
        
        G[0:3, i] = n
        G[3:6, i] = tau
        
    # SVD of G
    # If K < 6, the rank is at most K, so the min singular value of full 6D space is technically 0.
    # However, since we often have 4 fingers, we might just look at the singular values we do have,
    # or evaluate the volumetric/sum of singular values as a heuristic.
    # We will use the product of singular values or the minimum non-zero singular value as a proxy score.
    U, S, Vh = np.linalg.svd(G, full_matrices=False)
    
    # We use the product of singular values (volume of the wrench ellipsoid in its subspace)
    # Plus a small penalty if singular values drop too quickly
    score = np.prod(S)
    return float(score)

def generate_wrench_guided_proposal(
    mesh: trimesh.Trimesh,
    num_fingers: int,
    rng: np.random.Generator,
    finger_ids: np.ndarray,
    thumb_index: int = 0,
    num_candidates: int = 20,
) -> ContactProposal:
    """
    Generates multiple region opposition candidates and ranks them 
    using a preliminary static wrench feasibility score. 
    Returns the ContactProposal with the highest score.
    """
    best_proposal = None
    best_score = -np.inf
    
    centroid = mesh.centroid
    
    for _ in range(num_candidates):
        # We reuse region_opposition logic for generating a candidate
        proposal = generate_region_opposition_proposal(
            mesh=mesh,
            num_fingers=num_fingers,
            rng=rng,
            finger_ids=finger_ids,
            thumb_index=thumb_index,
            max_retries=5
        )
        
        score = compute_preliminary_wrench_score(
            proposal.target_points, 
            proposal.inward_normals, 
            centroid
        )
        
        if score > best_score:
            best_score = score
            best_proposal = proposal
            
    # Modify provenance to reflect the wrapper module
    if best_proposal is not None:
        # We must create a new dataclass instance to change provenance since it's frozen
        best_proposal = ContactProposal(
            target_points=best_proposal.target_points,
            face_ids=best_proposal.face_ids,
            inward_normals=best_proposal.inward_normals,
            finger_ids=best_proposal.finger_ids,
            force_hints=best_proposal.force_hints,
            provenance="wrench_guided"
        )
        return best_proposal
        
    # Fallback (should theoretically never reach here as generate_region_opposition_proposal always returns something)
    from qdgrasp.dataset.pipeline.proposals.surface_fixed import generate_surface_fixed_proposal
    return generate_surface_fixed_proposal(mesh, num_fingers, rng, finger_ids)
