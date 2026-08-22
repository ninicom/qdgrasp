"""Pydantic schemas and dataclasses for procedural objects."""

from __future__ import annotations

from typing import Any, Literal, Sequence
from pydantic import BaseModel, ConfigDict, Field


SubGeomType = Literal["box", "sphere", "cylinder", "capsule", "ellipsoid"]


class SubGeomSpec(BaseModel):
    """Specification of an elementary convex collision geometry in MuJoCo."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    type: SubGeomType
    size: tuple[float, ...]  # box: (dx/2, dy/2, dz/2); sphere: (r,); cylinder/capsule: (r, h/2)
    pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)  # (w, x, y, z)
    density: float = 1000.0  # kg / m^3


class ObjectManifestSpec(BaseModel):
    """Complete serializable metadata for a procedurally generated object."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = Field(default="qdgrasp/object-manifest/v1", alias="schema")
    object_id: str
    family: str  # "primitive", "superquadric", "compound"
    shape_type: str  # "box", "sphere", "cylinder", "capsule", "superquadric", "t_shape", etc.
    params: dict[str, Any]
    mesh_filename: str
    mesh_sha256: str
    mass: float
    inertia: tuple[float, float, float]  # diagonal elements (Ixx, Iyy, Izz)
    bounding_box: tuple[float, float, float, float, float, float]  # (xmin, ymin, zmin, xmax, ymax, zmax)
    collision_geoms: tuple[SubGeomSpec, ...]
    license: str = "CC0-1.0"
