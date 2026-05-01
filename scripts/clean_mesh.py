"""Merge duplicate vertices and strip small connected components ("floaters").

Common problem on OBJ meshes exported through Blender: every face gets its own
vertex triple, making the mesh look like 200k+ disconnected triangles under a
naive connected-components check. This script:

  1. Merges vertices within `merge-distance` of each other.
  2. Splits into connected components and drops everything smaller than
     `min-faces`.
  3. Writes the cleaned mesh back.

Usage:
    python scripts/clean_mesh.py \\
        --input  input/test_2/milo/xarm6.obj \\
        --output input/test_2/milo/xarm6_clean.obj \\
        --min-faces 500
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import trimesh
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components


def run(
    input_path: Path,
    output_path: Path,
    merge_distance: float = 1e-5,
    min_faces: int = 500,
) -> None:
    mesh = trimesh.load(str(input_path), force="mesh")
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)
    print(f"Loaded {input_path}")
    print(f"  before: {len(mesh.vertices)} verts, {len(mesh.faces)} faces")

    mesh.merge_vertices(merge_tex=False, merge_norm=True)
    print(f"  after merge: {len(mesh.vertices)} verts, {len(mesh.faces)} faces")

    # Compute face-level connected components
    faces = np.asarray(mesh.faces)
    V = len(mesh.vertices)
    edges = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    G = csr_matrix(
        (np.ones(len(edges), dtype=np.int8), (edges[:, 0], edges[:, 1])),
        shape=(V, V),
    )
    G = G + G.T
    n_comp, labels = connected_components(G, directed=False)
    face_labels = labels[faces[:, 0]]

    # Count faces per component
    uniq, counts = np.unique(face_labels, return_counts=True)
    keep_comp = uniq[counts >= min_faces]
    keep_face_mask = np.isin(face_labels, keep_comp)

    print(f"  components: {n_comp}")
    print(f"  largest: {counts.max()} faces")
    print(f"  components >= {min_faces} faces: {len(keep_comp)}")
    print(f"  faces dropped: {len(faces) - int(keep_face_mask.sum())} "
          f"({(len(faces) - int(keep_face_mask.sum())) / len(faces) * 100:.2f}%)")

    kept_faces = faces[keep_face_mask]
    used_vidx = np.unique(kept_faces)
    remap = -np.ones(V, dtype=np.int64)
    remap[used_vidx] = np.arange(len(used_vidx))
    new_faces = remap[kept_faces]

    cleaned = trimesh.Trimesh(
        vertices=mesh.vertices[used_vidx],
        faces=new_faces,
        process=False,
    )
    print(f"  final: {len(cleaned.vertices)} verts, {len(cleaned.faces)} faces")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned.export(output_path)
    print(f"Wrote {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--merge-distance", type=float, default=1e-5,
                        help="Vertices within this (meters) are merged")
    parser.add_argument("--min-faces", type=int, default=500,
                        help="Drop connected components smaller than this")
    args = parser.parse_args()
    run(
        input_path=args.input,
        output_path=args.output,
        merge_distance=args.merge_distance,
        min_faces=args.min_faces,
    )


if __name__ == "__main__":
    main()
