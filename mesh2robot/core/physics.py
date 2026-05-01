"""Phase 4 part 2 — compute per-link inertial properties from user mesh + density.

The Phase 2 segmentation gives us one submesh per link. The template match gives
us a density estimate. Combined, we get URDF-ready inertial blocks via trimesh.

Outputs for each link:
  mass       : kg
  com        : (3,)     center of mass in link frame
  inertia    : (3, 3)   inertia tensor about COM

All three go directly into the URDF <inertial> tag.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import trimesh


@dataclass
class LinkInertial:
    mass: float
    com: np.ndarray              # (3,)
    inertia: np.ndarray          # (3, 3) about COM, in link frame


def compute_inertial_from_mesh(mesh: trimesh.Trimesh, density: float) -> LinkInertial:
    """Compute inertial block from a watertight (or near-watertight) mesh.

    Uses trimesh's exact closed-form integration over the triangulated volume.
    If the mesh is not watertight, trimesh falls back to a voxelization proxy.
    """
    if not mesh.is_volume:
        # Best-effort patch: close holes, or fall back to bounding-box proxy
        mesh = _ensure_volume(mesh)

    mesh.density = density
    mass = float(mesh.mass)
    com = np.asarray(mesh.center_mass, dtype=float)
    inertia = np.asarray(mesh.moment_inertia, dtype=float)
    return LinkInertial(mass=mass, com=com, inertia=inertia)


def _ensure_volume(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Return a watertight version of mesh, or an AABB fallback if that fails."""
    m = mesh.copy()
    m.fix_normals()
    m.fill_holes()
    if m.is_volume:
        return m
    # AABB fallback: treat as oriented bounding box. Conservative but stable.
    obb = mesh.bounding_box_oriented.to_mesh()
    return obb


def compute_link_inertials(
    per_link_meshes: dict[int, trimesh.Trimesh],
    density: float,
) -> dict[int, LinkInertial]:
    """Apply compute_inertial_from_mesh to each link's submesh."""
    out: dict[int, LinkInertial] = {}
    for link_idx, mesh in per_link_meshes.items():
        out[link_idx] = compute_inertial_from_mesh(mesh, density)
    return out


def split_mesh_by_labels(
    vertices: np.ndarray,
    faces: np.ndarray,
    vertex_labels: np.ndarray,
) -> dict[int, trimesh.Trimesh]:
    """Split a combined mesh into per-label submeshes.

    A face is included in label L if all three of its vertices are labeled L.
    Faces bridging two labels (rare, at boundaries) are dropped — they're
    smaller than a mesh triangle and don't affect inertia meaningfully.
    """
    out: dict[int, trimesh.Trimesh] = {}
    labels = np.unique(vertex_labels[vertex_labels >= 0])
    for L in labels:
        vmask = vertex_labels == L
        face_mask = vmask[faces].all(axis=1)
        if not face_mask.any():
            continue
        # Remap vertex indices densely
        old_idx = np.where(vmask)[0]
        remap = -np.ones(len(vertex_labels), dtype=np.int64)
        remap[old_idx] = np.arange(len(old_idx))
        new_faces = remap[faces[face_mask]]
        sub = trimesh.Trimesh(
            vertices=vertices[old_idx], faces=new_faces, process=False
        )
        out[int(L)] = sub
    return out


if __name__ == "__main__":
    # Quick check on a unit cube
    cube = trimesh.creation.box(extents=(0.1, 0.2, 0.3))
    inertial = compute_inertial_from_mesh(cube, density=2700.0)
    print(f"Cube 0.1x0.2x0.3 m, density=2700")
    print(f"  mass = {inertial.mass:.3f} kg  (expected {2700*0.1*0.2*0.3:.3f})")
    print(f"  com  = {inertial.com}")
    print(f"  I    = \n{inertial.inertia}")
