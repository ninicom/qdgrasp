"""Reproducible normalization transforms for URDF/robot assets."""

from __future__ import annotations

import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import numpy as np

from ..config.schema import ConfigError


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sanitize_inertia_matrix(
    ixx: float, ixy: float, ixz: float, iyy: float, iyz: float, izz: float, mass: float
) -> tuple[float, float, float, float, float, float]:
    """Sanitize inertia tensor so that eigenvalues are strictly positive and satisfy physics."""
    mat = np.array([[ixx, ixy, ixz], [ixy, iyy, iyz], [ixz, iyz, izz]], dtype=np.float64)
    # Check eigenvalues
    eigvals = np.linalg.eigvalsh(mat)
    min_eig = float(np.min(eigvals))

    min_diag = max(mass * 1e-4, 1e-6)
    # The triangle inequality is checked on the pass-through path too: an inertia
    # tensor can be positive definite and still be rejected by MuJoCo for
    # violating A + B >= C, which is one of the errors this transform exists to
    # repair.
    violates_triangle = (ixx + iyy < izz) or (ixx + izz < iyy) or (iyy + izz < ixx)
    if (
        min_eig <= 0
        or violates_triangle
        or (abs(ixy) > max(ixx, iyy))
        or (abs(ixz) > max(ixx, izz))
        or (abs(iyz) > max(iyy, izz))
    ):
        # Off-diagonal corruption: regularize to diagonal positive-definite inertia
        d_xx = max(abs(ixx), min_diag)
        d_yy = max(abs(iyy), min_diag)
        d_zz = max(abs(izz), min_diag)
        # Ensure triangle inequalities: A+B >= C
        if d_xx + d_yy < d_zz:
            d_zz = (d_xx + d_yy) * 0.99
        if d_xx + d_zz < d_yy:
            d_yy = (d_xx + d_zz) * 0.99
        if d_yy + d_zz < d_xx:
            d_xx = (d_yy + d_zz) * 0.99
        return (float(d_xx), 0.0, 0.0, float(d_yy), 0.0, float(d_zz))

    return (ixx, ixy, ixz, iyy, iyz, izz)


def normalize_urdf(
    source_path: str | Path,
    output_path: str | Path,
    *,
    package_replacements: dict[str, str] | None = None,
    sanitize_inertias: bool = True,
    transform_id: str = "urdf_normalize_v1",
) -> dict[str, Any]:
    """Apply deterministic normalization transforms to a URDF file and write manifest."""
    src = Path(source_path)
    if not src.is_file():
        raise ConfigError(f"source URDF not found: {src}")

    src_bytes = src.read_bytes()
    src_hash = sha256_bytes(src_bytes)

    tree = ET.parse(src)
    root = tree.getroot()
    transforms_applied: list[str] = []

    pkg_map = package_replacements or {}
    if pkg_map:
        transforms_applied.append("resolve_package_uris")
        for elem in root.iter():
            for key in ("filename",):
                val = elem.get(key)
                if val:
                    for pkg_prefix, target_prefix in sorted(pkg_map.items()):
                        if val.startswith(pkg_prefix):
                            elem.set(key, val.replace(pkg_prefix, target_prefix, 1))

    if sanitize_inertias:
        transforms_applied.append("sanitize_inertias")
        for link in root.findall("link"):
            inertial = link.find("inertial")
            if inertial is not None:
                mass_elem = inertial.find("mass")
                mass = float(mass_elem.get("value", "0.01")) if mass_elem is not None else 0.01
                iner_elem = inertial.find("inertia")
                if iner_elem is not None:
                    ixx = float(iner_elem.get("ixx", "0.0"))
                    ixy = float(iner_elem.get("ixy", "0.0"))
                    ixz = float(iner_elem.get("ixz", "0.0"))
                    iyy = float(iner_elem.get("iyy", "0.0"))
                    iyz = float(iner_elem.get("iyz", "0.0"))
                    izz = float(iner_elem.get("izz", "0.0"))

                    s_ixx, s_ixy, s_ixz, s_iyy, s_iyz, s_izz = sanitize_inertia_matrix(
                        ixx, ixy, ixz, iyy, iyz, izz, mass
                    )
                    iner_elem.set("ixx", f"{s_ixx:.8e}")
                    iner_elem.set("ixy", f"{s_ixy:.8e}")
                    iner_elem.set("ixz", f"{s_ixz:.8e}")
                    iner_elem.set("iyy", f"{s_iyy:.8e}")
                    iner_elem.set("iyz", f"{s_iyz:.8e}")
                    iner_elem.set("izz", f"{s_izz:.8e}")

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    # Deterministic XML serialisation: 2-space indentation, and ElementTree keeps
    # attributes in document order, so the same input always yields the same bytes.
    ET.indent(tree, space="  ")
    out_xml_bytes = ET.tostring(root, encoding="utf-8", xml_declaration=True)
    out.write_bytes(out_xml_bytes)

    out_hash = sha256_bytes(out_xml_bytes)

    manifest: dict[str, Any] = {
        "transform_id": transform_id,
        "source_path": str(src),
        "source_sha256": src_hash,
        "output_path": str(out),
        "output_sha256": out_hash,
        "modified": True,
        "transforms_applied": transforms_applied,
    }

    # Named after the artifact, not after the directory: a fixed
    # ``normalization_manifest.json`` would be overwritten by the next asset
    # normalised into the same output directory, destroying its provenance.
    manifest_path = out.with_suffix(out.suffix + ".manifest.json")
    manifest["manifest_path"] = str(manifest_path)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest
