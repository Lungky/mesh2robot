"""Phase 5 helper — convex decomposition for collision geometry.

Uses CoACD (Collision-Aware Convex Decomposition, SIGGRAPH 2022) to convert a
non-convex link mesh into a small set of convex hulls that simulators like
Isaac Sim / PyBullet prefer. Falls back to the mesh's single convex hull if
CoACD is unavailable or fails.
"""

from __future__ import annotations

import numpy as np
import trimesh

try:
    import coacd
    _HAS_COACD = True
except ImportError:
    _HAS_COACD = False


def convex_decompose(
    mesh: trimesh.Trimesh,
    threshold: float = 0.05,
    max_hulls: int = 16,
    preprocess_resolution: int = 50,
) -> list[trimesh.Trimesh]:
    """Return a list of convex-hull trimeshes approximating `mesh`.

    Parameters mirror CoACD defaults. Lower `threshold` = tighter fit but more
    hulls. If CoACD is missing or fails, returns [mesh.convex_hull] as fallback.
    """
    if not _HAS_COACD or len(mesh.faces) < 4:
        return [mesh.convex_hull]

    try:
        m = coacd.Mesh(np.asarray(mesh.vertices), np.asarray(mesh.faces))
        parts = coacd.run_coacd(
            m,
            threshold=threshold,
            max_convex_hull=max_hulls,
            preprocess_resolution=preprocess_resolution,
        )
    except Exception:
        return [mesh.convex_hull]

    hulls = []
    for v, f in parts:
        v = np.asarray(v)
        f = np.asarray(f)
        if len(v) >= 4 and len(f) >= 1:
            h = trimesh.Trimesh(vertices=v, faces=f, process=False)
            hulls.append(h)
    if not hulls:
        return [mesh.convex_hull]
    return hulls


def combine_hulls(hulls: list[trimesh.Trimesh]) -> trimesh.Trimesh:
    """Concatenate convex hulls into one non-convex trimesh for simple export.

    URDF spec allows multiple <collision> tags per link, but for MVP we export a
    single combined mesh and rely on the importer's convex-decomposition flag.
    """
    if len(hulls) == 1:
        return hulls[0]
    verts = []
    faces = []
    offset = 0
    for h in hulls:
        verts.append(np.asarray(h.vertices))
        faces.append(np.asarray(h.faces) + offset)
        offset += len(h.vertices)
    return trimesh.Trimesh(
        vertices=np.vstack(verts), faces=np.vstack(faces), process=False
    )


if __name__ == "__main__":
    # Smoke test on an L-shaped concave mesh
    box1 = trimesh.creation.box(extents=(0.2, 0.1, 0.1))
    box2 = trimesh.creation.box(extents=(0.1, 0.1, 0.3))
    box2.apply_translation([0.05, 0, 0.15])
    L = trimesh.util.concatenate([box1, box2])
    print(f"Concave L-shape: {len(L.vertices)} verts, convex={L.is_convex}")
    hulls = convex_decompose(L, threshold=0.05)
    print(f"CoACD produced {len(hulls)} hulls")
    for i, h in enumerate(hulls):
        print(f"  hull {i}: {len(h.vertices)} verts, volume={h.volume:.6f}")
