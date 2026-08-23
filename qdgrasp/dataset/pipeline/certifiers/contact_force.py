import numpy as np
from qdgrasp.dataset.pipeline.contracts import StaticCertificate
from qdgrasp.dataset.pipeline.certifiers.grasp_wrench import compute_grasp_wrench_space_quality

def certify_force_closure(
    target_points: np.ndarray,
    inward_normals: np.ndarray,
    centroid: np.ndarray,
    mass: float = 1.0,
    mu: float = 0.5,
    gravity: np.ndarray | None = None,
) -> StaticCertificate:
    """
    Certify both six-dimensional force closure and gravity equilibrium.

    Balancing gravity alone is not force closure: a single supporting contact
    can do that.  We therefore require a positive Ferrari-Canny-style GWS margin
    before solving the gravity load distribution.
    """
    target_points = np.asarray(target_points, dtype=np.float64)
    inward_normals = np.asarray(inward_normals, dtype=np.float64)
    centroid = np.asarray(centroid, dtype=np.float64)
    gravity_vec = (
        np.array([0.0, 0.0, -9.81], dtype=np.float64)
        if gravity is None
        else np.asarray(gravity, dtype=np.float64)
    )
    K = target_points.shape[0] if target_points.ndim == 2 else 0

    if (
        K < 2
        or target_points.shape != inward_normals.shape
        or target_points.shape[1:] != (3,)
        or centroid.shape != (3,)
        or gravity_vec.shape != (3,)
        or mass <= 0.0
        or mu <= 0.0
        or not np.all(np.isfinite(target_points))
        or not np.all(np.isfinite(centroid))
        or not np.all(np.isfinite(gravity_vec))
    ):
        return StaticCertificate(
            force_solution=np.zeros((K, 3)),
            cone_residual=float("inf"),
            object_wrench=np.zeros(6),
            quality_margin=0.0,
            passed=False,
        )

    normal_lengths = np.linalg.norm(inward_normals, axis=1)
    if not np.all(np.isfinite(normal_lengths)) or np.any(normal_lengths < 1e-8):
        return StaticCertificate(
            force_solution=np.zeros((K, 3)),
            cone_residual=float("inf"),
            object_wrench=np.zeros(6),
            quality_margin=0.0,
            passed=False,
        )
    normals = inward_normals / normal_lengths[:, None]
    object_scale = max(float(np.max(np.linalg.norm(target_points - centroid, axis=1))), 1e-6)
    gws = compute_grasp_wrench_space_quality(
        target_points,
        normals,
        centroid,
        mu=mu,
        torque_scale=1.0 / object_scale,
    )
    if not gws.passed:
        return gws

    from scipy.optimize import linprog

    # Linearize friction cone using 8-sided pyramid
    num_edges = 8

    # We will express the force at each contact as a positive combination of the 8 edges of the friction cone
    # f_i = sum(lambda_{i,j} * v_{i,j}) where lambda_{i,j} >= 0
    # v_{i,j} = n_i + mu * (cos(theta) * t1 + sin(theta) * t2)

    V_cols = []

    for i in range(K):
        n = normals[i]
        r = target_points[i] - centroid

        # Find tangent basis
        # Find a vector not parallel to n
        if np.abs(n[0]) > 0.9:
            v_temp = np.array([0.0, 1.0, 0.0])
        else:
            v_temp = np.array([1.0, 0.0, 0.0])

        t1 = np.cross(n, v_temp)
        t1 = t1 / np.linalg.norm(t1)
        t2 = np.cross(n, t1)

        for j in range(num_edges):
            theta = 2.0 * np.pi * j / num_edges
            # Direction of the pyramid edge
            v_edge = n + mu * (np.cos(theta) * t1 + np.sin(theta) * t2)

            # Wrench produced by this edge
            torque = np.cross(r, v_edge)
            wrench = np.concatenate([v_edge, torque])
            V_cols.append(wrench)

    # V is [6, K * 8]
    V = np.column_stack(V_cols)

    # External wrench
    w_ext = np.zeros(6)
    w_ext[0:3] = mass * gravity_vec

    # We want V @ lam = -w_ext
    # Objective: minimize sum(lam)
    c = np.ones(K * num_edges)

    res = linprog(
        c,
        A_eq=V,
        b_eq=-w_ext,
        bounds=(0, None),
        method='highs'
    )

    if res.success:
        lam = res.x
        f_opt = np.zeros((K, 3))
        for i in range(K):
            f_i = np.zeros(3)
            for j in range(num_edges):
                idx = i * num_edges + j
                n = normals[i]
                if np.abs(n[0]) > 0.9:
                    v_temp = np.array([0.0, 1.0, 0.0])
                else:
                    v_temp = np.array([1.0, 0.0, 0.0])
                t1 = np.cross(n, v_temp)
                t1 = t1 / np.linalg.norm(t1)
                t2 = np.cross(n, t1)

                theta = 2.0 * np.pi * j / num_edges
                v_edge = n + mu * (np.cos(theta) * t1 + np.sin(theta) * t2)
                f_i += lam[idx] * v_edge
            f_opt[i] = f_i

        return StaticCertificate(
            force_solution=f_opt,
            cone_residual=0.0,
            object_wrench=V @ lam,
            quality_margin=gws.quality_margin,
            passed=True
        )
    else:
        return StaticCertificate(
            force_solution=np.zeros((K, 3)),
            cone_residual=float('inf'),
            object_wrench=np.zeros(6),
            quality_margin=-1.0,
            passed=False
        )
