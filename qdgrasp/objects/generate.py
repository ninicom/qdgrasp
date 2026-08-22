"""Procedural generation of 3D objects and exact collision representations."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import trimesh

from .schema import SubGeomSpec


def _signed_pow(val: np.ndarray | float, p: float) -> np.ndarray | float:
    """Compute sign-preserving power: sign(val) * |val|^p."""
    return np.sign(val) * (np.abs(val) ** p)


def generate_box(
    rng: np.random.Generator,
    size_range: tuple[float, float] = (0.02, 0.08),
    density: float = 1000.0,
) -> tuple[trimesh.Trimesh, list[SubGeomSpec], dict[str, Any], float, tuple[float, float, float]]:
    """Generate a random 3D box."""
    dx = float(rng.uniform(size_range[0], size_range[1]))
    dy = float(rng.uniform(size_range[0], size_range[1]))
    dz = float(rng.uniform(size_range[0], size_range[1]))

    mesh = trimesh.creation.box(extents=(dx, dy, dz))
    geoms = [
        SubGeomSpec(
            type="box",
            size=(dx * 0.5, dy * 0.5, dz * 0.5),
            pos=(0.0, 0.0, 0.0),
            density=density,
        )
    ]
    volume = dx * dy * dz
    mass = density * volume
    ixx = (mass / 12.0) * (dy * dy + dz * dz)
    iyy = (mass / 12.0) * (dx * dx + dz * dz)
    izz = (mass / 12.0) * (dx * dx + dy * dy)
    params = {"dx": dx, "dy": dy, "dz": dz, "density": density}
    return mesh, geoms, params, mass, (ixx, iyy, izz)


def generate_sphere(
    rng: np.random.Generator,
    radius_range: tuple[float, float] = (0.015, 0.045),
    density: float = 1000.0,
) -> tuple[trimesh.Trimesh, list[SubGeomSpec], dict[str, Any], float, tuple[float, float, float]]:
    """Generate a random 3D sphere."""
    r = float(rng.uniform(radius_range[0], radius_range[1]))
    mesh = trimesh.creation.icosphere(subdivisions=3, radius=r)
    geoms = [
        SubGeomSpec(
            type="sphere",
            size=(r,),
            pos=(0.0, 0.0, 0.0),
            density=density,
        )
    ]
    volume = (4.0 / 3.0) * math.pi * (r**3)
    mass = density * volume
    i_val = 0.4 * mass * (r**2)
    params = {"radius": r, "density": density}
    return mesh, geoms, params, mass, (i_val, i_val, i_val)


def generate_cylinder(
    rng: np.random.Generator,
    radius_range: tuple[float, float] = (0.01, 0.035),
    height_range: tuple[float, float] = (0.03, 0.10),
    density: float = 1000.0,
) -> tuple[trimesh.Trimesh, list[SubGeomSpec], dict[str, Any], float, tuple[float, float, float]]:
    """Generate a random 3D cylinder."""
    r = float(rng.uniform(radius_range[0], radius_range[1]))
    h = float(rng.uniform(height_range[0], height_range[1]))
    mesh = trimesh.creation.cylinder(radius=r, height=h, sections=32)
    geoms = [
        SubGeomSpec(
            type="cylinder",
            size=(r, h * 0.5),
            pos=(0.0, 0.0, 0.0),
            density=density,
        )
    ]
    volume = math.pi * (r**2) * h
    mass = density * volume
    ixx = (mass / 12.0) * (3.0 * (r**2) + (h**2))
    iyy = ixx
    izz = 0.5 * mass * (r**2)
    params = {"radius": r, "height": h, "density": density}
    return mesh, geoms, params, mass, (ixx, iyy, izz)


def generate_capsule(
    rng: np.random.Generator,
    radius_range: tuple[float, float] = (0.01, 0.03),
    height_range: tuple[float, float] = (0.03, 0.08),
    density: float = 1000.0,
) -> tuple[trimesh.Trimesh, list[SubGeomSpec], dict[str, Any], float, tuple[float, float, float]]:
    """Generate a random 3D capsule."""
    r = float(rng.uniform(radius_range[0], radius_range[1]))
    h = float(rng.uniform(height_range[0], height_range[1]))
    mesh = trimesh.creation.capsule(radius=r, height=h, count=[16, 16])
    geoms = [
        SubGeomSpec(
            type="capsule",
            size=(r, h * 0.5),
            pos=(0.0, 0.0, 0.0),
            density=density,
        )
    ]
    # Volume: cylinder + 2 hemispheres (1 sphere)
    v_cyl = math.pi * (r**2) * h
    v_sph = (4.0 / 3.0) * math.pi * (r**3)
    volume = v_cyl + v_sph
    mass = density * volume
    m_cyl = density * v_cyl
    m_sph = density * v_sph
    # Inertia estimation
    izz = 0.5 * m_cyl * (r**2) + 0.4 * m_sph * (r**2)
    ixx = (m_cyl / 12.0) * (3 * r**2 + h**2) + m_sph * (0.4 * r**2 + 0.75 * r * h + 0.5 * h**2)
    iyy = ixx
    params = {"radius": r, "height": h, "density": density}
    return mesh, geoms, params, mass, (ixx, iyy, izz)


def generate_superquadric(
    rng: np.random.Generator,
    scale_range: tuple[float, float] = (0.02, 0.05),
    shape_range: tuple[float, float] = (0.2, 1.5),
    density: float = 1000.0,
    n_eta: int = 32,
    n_omega: int = 32,
) -> tuple[trimesh.Trimesh, list[SubGeomSpec], dict[str, Any], float, tuple[float, float, float]]:
    """Generate a parametric superquadric mesh and its convex collision envelope."""
    a = float(rng.uniform(scale_range[0], scale_range[1]))
    b = float(rng.uniform(scale_range[0], scale_range[1]))
    c = float(rng.uniform(scale_range[0], scale_range[1]))
    e1 = float(rng.uniform(shape_range[0], shape_range[1]))
    e2 = float(rng.uniform(shape_range[0], shape_range[1]))

    # Sample parametric grid
    eta = np.linspace(-math.pi / 2.0, math.pi / 2.0, n_eta)
    omega = np.linspace(-math.pi, math.pi, n_omega, endpoint=False)
    eta_grid, omega_grid = np.meshgrid(eta, omega, indexing="ij")

    cos_eta = np.cos(eta_grid)
    sin_eta = np.sin(eta_grid)
    cos_omega = np.cos(omega_grid)
    sin_omega = np.sin(omega_grid)

    x = a * _signed_pow(cos_eta, e1) * _signed_pow(cos_omega, e2)
    y = b * _signed_pow(cos_eta, e1) * _signed_pow(sin_omega, e2)
    z = c * _signed_pow(sin_eta, e1)

    vertices = np.stack([x.flatten(), y.flatten(), z.flatten()], axis=-1)

    # Build faces
    faces = []
    for i in range(n_eta - 1):
        for j in range(n_omega):
            j_next = (j + 1) % n_omega
            v0 = i * n_omega + j
            v1 = (i + 1) * n_omega + j
            v2 = (i + 1) * n_omega + j_next
            v3 = i * n_omega + j_next
            faces.append([v0, v1, v2])
            faces.append([v0, v2, v3])

    mesh = trimesh.Trimesh(vertices=vertices, faces=np.array(faces, dtype=np.int32), process=True)

    # Approximate volume & inertia from bounding box / beta function
    dx, dy, dz = 2.0 * a, 2.0 * b, 2.0 * c
    volume = (
        2.0
        * a
        * b
        * c
        * e1
        * e2
        * math.gamma(e1 / 2.0 + 1.0)
        * math.gamma(e2 / 2.0)
        * math.gamma(e2 / 2.0)
        / (math.gamma(e1 + 1.0) * math.gamma(e2))
        if e1 > 0 and e2 > 0 and e1 < 3.0 and e2 < 3.0
        else dx * dy * dz * 0.7
    )
    volume = max(1e-6, float(volume))
    mass = density * volume
    ixx = (mass / 5.0) * (b * b + c * c)
    iyy = (mass / 5.0) * (a * a + c * c)
    izz = (mass / 5.0) * (a * a + b * b)

    geoms = [
        SubGeomSpec(
            type="box",
            size=(a, b, c),
            pos=(0.0, 0.0, 0.0),
            density=density,
        )
    ]
    params = {"a": a, "b": b, "c": c, "e1": e1, "e2": e2, "density": density}
    return mesh, geoms, params, mass, (ixx, iyy, izz)


def generate_compound_convex(
    rng: np.random.Generator,
    shape_family: str = "t_shape",  # "t_shape", "l_shape", "dumbbell"
    scale_range: tuple[float, float] = (0.02, 0.04),
    density: float = 1000.0,
) -> tuple[trimesh.Trimesh, list[SubGeomSpec], dict[str, Any], float, tuple[float, float, float]]:
    """Generate compound objects from exact unions of convex sub-geometries."""
    if shape_family == "t_shape":
        # Vertical stem (box) + Horizontal crossbar (box)
        w_stem = float(rng.uniform(scale_range[0] * 0.4, scale_range[1] * 0.6))
        h_stem = float(rng.uniform(scale_range[0] * 1.5, scale_range[1] * 2.0))
        w_bar = float(rng.uniform(scale_range[0] * 1.5, scale_range[1] * 2.5))
        h_bar = float(rng.uniform(scale_range[0] * 0.4, scale_range[1] * 0.6))
        depth = float(rng.uniform(scale_range[0] * 0.4, scale_range[1] * 0.6))

        m_stem = trimesh.creation.box(extents=(w_stem, depth, h_stem))
        m_bar = trimesh.creation.box(extents=(w_bar, depth, h_bar))
        m_bar.apply_translation([0.0, 0.0, (h_stem + h_bar) * 0.5 - h_bar * 0.5])
        mesh = trimesh.util.concatenate([m_stem, m_bar])

        geoms = [
            SubGeomSpec(
                type="box",
                size=(w_stem * 0.5, depth * 0.5, h_stem * 0.5),
                pos=(0.0, 0.0, 0.0),
                density=density,
            ),
            SubGeomSpec(
                type="box",
                size=(w_bar * 0.5, depth * 0.5, h_bar * 0.5),
                pos=(0.0, 0.0, (h_stem + h_bar) * 0.5 - h_bar * 0.5),
                density=density,
            ),
        ]
        vol = (w_stem * depth * h_stem) + (w_bar * depth * h_bar)
        mass = density * vol
        params = {"shape_family": "t_shape", "w_stem": w_stem, "h_stem": h_stem, "w_bar": w_bar, "h_bar": h_bar}

    elif shape_family == "dumbbell":
        # Handle cylinder + 2 sphere ends
        r_handle = float(rng.uniform(scale_range[0] * 0.25, scale_range[1] * 0.35))
        len_handle = float(rng.uniform(scale_range[0] * 1.8, scale_range[1] * 2.5))
        r_head = float(rng.uniform(scale_range[0] * 0.6, scale_range[1] * 0.9))

        m_handle = trimesh.creation.cylinder(radius=r_handle, height=len_handle, sections=24)
        m_head1 = trimesh.creation.icosphere(subdivisions=2, radius=r_head)
        m_head1.apply_translation([0.0, 0.0, len_handle * 0.5])
        m_head2 = trimesh.creation.icosphere(subdivisions=2, radius=r_head)
        m_head2.apply_translation([0.0, 0.0, -len_handle * 0.5])
        mesh = trimesh.util.concatenate([m_handle, m_head1, m_head2])

        geoms = [
            SubGeomSpec(
                type="cylinder",
                size=(r_handle, len_handle * 0.5),
                pos=(0.0, 0.0, 0.0),
                density=density,
            ),
            SubGeomSpec(
                type="sphere",
                size=(r_head,),
                pos=(0.0, 0.0, len_handle * 0.5),
                density=density,
            ),
            SubGeomSpec(
                type="sphere",
                size=(r_head,),
                pos=(0.0, 0.0, -len_handle * 0.5),
                density=density,
            ),
        ]
        vol = (math.pi * r_handle**2 * len_handle) + (2.0 * (4.0 / 3.0) * math.pi * r_head**3)
        mass = density * vol
        params = {"shape_family": "dumbbell", "r_handle": r_handle, "len_handle": len_handle, "r_head": r_head}

    else:  # l_shape default
        w = float(rng.uniform(scale_range[0] * 0.4, scale_range[1] * 0.6))
        h = float(rng.uniform(scale_range[0] * 1.5, scale_range[1] * 2.0))
        d_foot = float(rng.uniform(scale_range[0] * 1.0, scale_range[1] * 1.5))
        thickness = float(rng.uniform(scale_range[0] * 0.4, scale_range[1] * 0.6))

        m_vert = trimesh.creation.box(extents=(w, thickness, h))
        m_foot = trimesh.creation.box(extents=(w, d_foot, thickness))
        m_foot.apply_translation([0.0, (d_foot - thickness) * 0.5, -(h - thickness) * 0.5])
        mesh = trimesh.util.concatenate([m_vert, m_foot])

        geoms = [
            SubGeomSpec(
                type="box",
                size=(w * 0.5, thickness * 0.5, h * 0.5),
                pos=(0.0, 0.0, 0.0),
                density=density,
            ),
            SubGeomSpec(
                type="box",
                size=(w * 0.5, d_foot * 0.5, thickness * 0.5),
                pos=(0.0, (d_foot - thickness) * 0.5, -(h - thickness) * 0.5),
                density=density,
            ),
        ]
        vol = (w * thickness * h) + (w * d_foot * thickness)
        mass = density * vol
        params = {"shape_family": "l_shape", "w": w, "h": h, "d_foot": d_foot, "thickness": thickness}

    # Bounding inertia approximation
    bounds = mesh.bounds
    extents = bounds[1] - bounds[0]
    ixx = (mass / 12.0) * (extents[1] ** 2 + extents[2] ** 2)
    iyy = (mass / 12.0) * (extents[0] ** 2 + extents[2] ** 2)
    izz = (mass / 12.0) * (extents[0] ** 2 + extents[1] ** 2)

    return mesh, geoms, params, mass, (ixx, iyy, izz)
