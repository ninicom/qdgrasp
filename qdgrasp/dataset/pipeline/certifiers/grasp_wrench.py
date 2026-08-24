import numpy as np
from scipy.spatial import ConvexHull, QhullError
from qdgrasp.dataset.pipeline.contracts import StaticCertificate

def compute_grasp_wrench_space_quality(
    target_points: np.ndarray,
    inward_normals: np.ndarray,
    centroid: np.ndarray,
    mu: float = 0.5,
    num_edges: int = 8,
    torque_scale: float = 1.0, # scale factor for torque vs force units
    torsional_friction: float = 0.005,
) -> StaticCertificate:
    """
    Computes Ferrari-Canny epsilon-metric on the Grasp Wrench Space (GWS).
    Constructs primitive friction-cone edge wrenches and computes the radius
    of the largest 6D ball centered at the origin that fits within the convex hull.
    """
    K = target_points.shape[0]
    wrenches = []

    for i in range(K):
        n = inward_normals[i]
        r = (target_points[i] - centroid) * torque_scale

        # Orthonormal basis in tangent plane
        if np.abs(n[0]) > 0.9:
            v_temp = np.array([0.0, 1.0, 0.0])
        else:
            v_temp = np.array([1.0, 0.0, 0.0])

        t1 = np.cross(n, v_temp)
        t1 = t1 / np.linalg.norm(t1)
        t2 = np.cross(n, t1)

        for j in range(num_edges):
            theta = 2.0 * np.pi * j / num_edges
            # Unit-length or friction-cone edge vector
            v_edge = n + mu * (np.cos(theta) * t1 + np.sin(theta) * t2)
            # Normalize edge vector or keep unit normal component
            # Here we keep normal component = 1.0
            torque = np.cross(r, v_edge)
            torsional_radius = torsional_friction * torque_scale
            for torsion_sign in (-1.0, 1.0):
                wrench = np.concatenate(
                    [v_edge, torque + torsion_sign * torsional_radius * n]
                )
                wrenches.append(wrench)

    wrenches = np.array(wrenches) # [K * num_edges, 6]

    # Must have at least 7 points to form a 6D simplex, usually K*num_edges = 4*8 = 32 >= 7
    if wrenches.shape[0] < 7:
        return StaticCertificate(
            force_solution=np.zeros((K, 3)),
            cone_residual=float('inf'),
            object_wrench=np.zeros(6),
            quality_margin=0.0,
            passed=False
        )

    try:
        # Compute 6D convex hull
        hull = ConvexHull(wrenches)
        # Equation: A . x + d <= 0 for points inside.
        # For origin (0, 0, 0, 0, 0, 0), evaluate A . 0 + d = d.
        # If d < 0 for all facets, origin is inside.
        # Distance to facet is -d / ||A||. In scipy ConvexHull, ||A|| = 1.
        dists = -hull.equations[:, -1]

        if np.all(dists > 0):
            epsilon = float(np.min(dists))
            return StaticCertificate(
                force_solution=np.zeros((K, 3)),
                cone_residual=0.0,
                object_wrench=np.zeros(6),
                quality_margin=epsilon,
                passed=bool(epsilon > 1e-4)
            )
        else:
            return StaticCertificate(
                force_solution=np.zeros((K, 3)),
                cone_residual=float('inf'),
                object_wrench=np.zeros(6),
                quality_margin=0.0,
                passed=False
            )
    except QhullError:
        # Degenerate points (e.g. all points lie in a lower-dimensional subspace)
        return StaticCertificate(
            force_solution=np.zeros((K, 3)),
            cone_residual=float('inf'),
            object_wrench=np.zeros(6),
            quality_margin=0.0,
            passed=False
        )
