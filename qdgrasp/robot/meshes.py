"""Mesh resolution and geometry processing using trimesh."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh

from ..config.schema import ConfigError


def resolve_mesh_path(
    reference: str,
    *,
    base_dir: Path | str | None = None,
    mesh_root: Path | str | None = None,
    package_roots: dict[str, str] | None = None,
) -> Path:
    """Resolve a URDF/MJCF mesh reference to an existing file path.

    Supports:
      - ``package://package_name/path/to/mesh.stl``
      - ``model://model_name/path/to/mesh.stl``
      - Relative paths relative to ``base_dir``, ``mesh_root``, or project root.
    """
    pkg_roots = package_roots or {}

    if reference.startswith("package://"):
        stripped = reference[len("package://"):]
        parts = stripped.split("/", 1)
        pkg_name = parts[0]
        subpath = parts[1] if len(parts) > 1 else ""
        if pkg_name in pkg_roots:
            candidate = Path(pkg_roots[pkg_name]) / subpath
            if candidate.is_file():
                return candidate.resolve()
        # Search common fallback package locations if not explicitly mapped
        candidates = [
            Path(".references/robot-assets/wonik-allegro-ros2/src") / pkg_name / subpath,
            Path(".references/robot-assets/leap-hand-sim") / subpath,
            Path(".references/robot-assets/leap-hand-sim/assets") / subpath,
            Path(".references/robot-assets/dex-urdf") / subpath,
        ]
        for cand in candidates:
            if cand.is_file():
                return cand.resolve()
        raise ConfigError(f"unable to resolve package URI '{reference}'; package '{pkg_name}' not in {pkg_roots}")

    if reference.startswith("model://"):
        stripped = reference[len("model://"):]
        if mesh_root is not None:
            cand = Path(mesh_root) / stripped
            if cand.is_file():
                return cand.resolve()
        raise ConfigError(f"unable to resolve model URI '{reference}'")

    path_candidate = Path(reference)
    if path_candidate.is_file():
        return path_candidate.resolve()

    search_dirs: list[Path] = []
    if base_dir is not None:
        b_dir = Path(base_dir)
        search_dirs.extend([b_dir, b_dir / "assets", b_dir / "meshes", b_dir.parent / "meshes"])
    if mesh_root is not None:
        m_dir = Path(mesh_root)
        search_dirs.extend([m_dir, m_dir / "assets", m_dir / "meshes"])

    for s_dir in search_dirs:
        cand = s_dir / reference
        if cand.is_file():
            return cand.resolve()

    raise ConfigError(f"mesh file '{reference}' not found (searched in {[str(d) for d in search_dirs]})")


def load_mesh(path: str | Path) -> trimesh.Trimesh:
    """Load a mesh file (STL/OBJ/PLY/DAE) into a single trimesh.Trimesh object."""
    p = Path(path)
    if not p.is_file():
        raise ConfigError(f"mesh path does not exist: {p}")
    loaded = trimesh.load(str(p), force="mesh")
    if isinstance(loaded, trimesh.Scene):
        if len(loaded.geometry) == 0:
            raise ConfigError(f"empty mesh scene in {p}")
        # Combine all geometries in scene
        geoms = [g for g in loaded.geometry.values() if isinstance(g, trimesh.Trimesh)]
        if not geoms:
            raise ConfigError(f"no valid meshes in scene {p}")
        return trimesh.util.concatenate(geoms)
    if not isinstance(loaded, trimesh.Trimesh):
        raise ConfigError(f"loaded object is {type(loaded).__name__}, expected Trimesh")
    return loaded


def sample_mesh_surface(
    mesh: trimesh.Trimesh,
    count: int = 64,
    *,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample ``count`` points and surface normals uniformly from mesh surface."""
    rng = np.random.RandomState(seed)
    if len(mesh.vertices) == 0:
        return np.zeros((count, 3), dtype=np.float32), np.zeros((count, 3), dtype=np.float32)
    # trimesh.sample.sample_surface
    points, face_indices = trimesh.sample.sample_surface(mesh, count, seed=rng)
    normals = mesh.face_normals[face_indices]
    return points.astype(np.float32), normals.astype(np.float32)
