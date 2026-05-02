"""Phase D.4 — Geometric joint extraction from clean per-link segmentation.

Given a mesh + per-face link labels (where every face has a label, e.g.
the output of the strict-mode merge in predict_urdf_interactive.py),
fit each joint by:

  1. Find boundary edges between adjacent links (face adjacency where the
     two faces have different labels).
  2. Collect the boundary vertices.
  3. Fit a 3D circle to those vertices (PCA → plane, algebraic 2D fit
     within the plane).
  4. Circle normal → joint axis (pointing roughly along the chain dir).
  5. Circle center → joint origin.
  6. Joint type from boundary geometry:
       - small in-plane radius spread + closed loop → revolute
       - large in-plane spread, no clear closure → planar/fixed
     (prismatic disambiguation needs motion data — D.3 — so we treat
      ambiguous cases as revolute, the most common joint type).

The output is a list of (axis, origin, type) triples in WORLD frame,
ready to drop into predict_urdf_interactive.py's URDF builder in place
of the ML's joint-head predictions.

This module has zero ML dependency — it only needs the mesh + labels.
That makes the joint output:
  - independent of the model's 38° axis_deg val error,
  - robust to the model's pose-sensitive embedding bias,
  - quality-bounded by segmentation cleanliness instead.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import trimesh


@dataclass
class GeometricJoint:
    """Result of fitting a joint between two adjacent links."""
    parent_label: int
    child_label: int
    axis: np.ndarray               # (3,) unit vector in world frame
    origin: np.ndarray             # (3,) world-frame point
    radius: float                  # of the fitted circle (m)
    plane_residual: float          # mean abs distance of boundary verts to plane (m)
    n_boundary_edges: int          # number of mesh edges between the two link regions
    type: str                      # "revolute" / "fixed"
    confidence: float              # in [0, 1] — how circular the boundary is


def _fit_3d_circle(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Algebraic 3D circle fit.

    Returns
    -------
    center : (3,) — circle center in world frame
    normal : (3,) — unit vector normal to the circle's plane
    radius : float — circle radius
    plane_residual : float — mean abs in-direction-of-normal residual
                             (signal of how well the points fit a plane)
    """
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) < 3:
        return (
            pts.mean(axis=0) if len(pts) else np.zeros(3),
            np.array([0.0, 0.0, 1.0]),
            0.0,
            float("inf"),
        )

    # 1. Best-fit plane via PCA
    centroid = pts.mean(axis=0)
    centered = pts - centroid
    _u, _s, vh = np.linalg.svd(centered, full_matrices=False)
    normal = vh[-1]  # smallest singular value direction
    normal = normal / (np.linalg.norm(normal) + 1e-12)

    # Plane residual = mean abs |centered · normal|
    plane_residual = float(np.mean(np.abs(centered @ normal)))

    # 2. Project onto plane; pick two orthogonal in-plane basis vectors
    e1 = vh[0]
    e2 = vh[1]
    proj = centered @ np.stack([e1, e2], axis=1)   # (N, 2)

    # 3. Algebraic 2D circle fit:
    #    (x - a)^2 + (y - b)^2 = r^2
    #    -> 2ax + 2by + (r^2 - a^2 - b^2) = x^2 + y^2
    #    Solve A · [a, b, c] = b_vec  (least squares)
    x = proj[:, 0]
    y = proj[:, 1]
    A = np.stack([2 * x, 2 * y, np.ones_like(x)], axis=1)
    b_vec = x * x + y * y
    sol, _res, _rank, _sv = np.linalg.lstsq(A, b_vec, rcond=None)
    a, b, c_param = float(sol[0]), float(sol[1]), float(sol[2])
    r_sq = c_param + a * a + b * b
    radius = float(np.sqrt(max(r_sq, 0.0)))

    # 4. Convert center back to 3D
    center_world = centroid + a * e1 + b * e2

    return center_world, normal, radius, plane_residual


def _circularity_score(
    points: np.ndarray, center: np.ndarray, normal: np.ndarray, radius: float,
) -> float:
    """How well the boundary fits a perfect circle. 1.0 = perfect, 0.0 = bad."""
    if len(points) < 3 or radius < 1e-6:
        return 0.0
    # Project onto plane
    centered = points - center
    in_plane = centered - np.outer(centered @ normal, normal)
    in_plane_dist = np.linalg.norm(in_plane, axis=1)
    # Circularity: how close in-plane distance is to radius for every point
    rel_err = np.abs(in_plane_dist - radius) / (radius + 1e-12)
    score = float(np.exp(-np.mean(rel_err) * 4.0))   # scale: ~25% rel_err → 0.37
    return min(max(score, 0.0), 1.0)


def find_boundary_vertices(
    mesh: trimesh.Trimesh,
    face_labels: np.ndarray,
    label_a: int,
    label_b: int,
) -> tuple[np.ndarray, int]:
    """Vertices on the boundary between regions A and B.

    Returns (boundary_vertex_positions: (M, 3), n_edges: int).
    """
    adj = mesh.face_adjacency  # (E, 2) face index pairs sharing an edge
    edge_idx = mesh.face_adjacency_edges  # (E, 2) vertex index pairs
    if len(adj) == 0:
        return np.empty((0, 3), dtype=np.float64), 0

    la = face_labels[adj[:, 0]]
    lb = face_labels[adj[:, 1]]
    boundary_mask = ((la == label_a) & (lb == label_b)) | \
                    ((la == label_b) & (lb == label_a))
    if not boundary_mask.any():
        return np.empty((0, 3), dtype=np.float64), 0

    boundary_edge_verts = edge_idx[boundary_mask]   # (M_e, 2)
    n_edges = int(boundary_mask.sum())
    unique_v = np.unique(boundary_edge_verts.ravel())
    return np.asarray(mesh.vertices)[unique_v], n_edges


def _fit_one_joint(
    mesh: trimesh.Trimesh,
    face_labels: np.ndarray,
    parent_lbl: int,
    child_lbl: int,
    min_boundary_edges: int,
    circularity_revolute_threshold: float,
) -> GeometricJoint:
    """Fit a single GeometricJoint between two given link labels.
    Shared by the chain-order and tree-topology extractors below."""
    verts, n_edges = find_boundary_vertices(
        mesh, face_labels, parent_lbl, child_lbl,
    )

    if n_edges < min_boundary_edges or len(verts) < 3:
        # No clear interface — fixed joint at midpoint of link centroids
        parent_centroid = (np.asarray(
            mesh.vertices[np.unique(mesh.faces[face_labels == parent_lbl])
                          ].mean(axis=0))
            if (face_labels == parent_lbl).any() else np.zeros(3))
        child_centroid = (np.asarray(
            mesh.vertices[np.unique(mesh.faces[face_labels == child_lbl])
                          ].mean(axis=0))
            if (face_labels == child_lbl).any() else np.zeros(3))
        return GeometricJoint(
            parent_label=parent_lbl, child_label=child_lbl,
            axis=np.array([0.0, 0.0, 1.0]),
            origin=0.5 * (parent_centroid + child_centroid),
            radius=0.0,
            plane_residual=float("inf"),
            n_boundary_edges=n_edges,
            type="fixed",
            confidence=0.0,
        )

    center, normal, radius, plane_residual = _fit_3d_circle(verts)
    circ = _circularity_score(verts, center, normal, radius)
    jtype = "revolute" if circ >= circularity_revolute_threshold else "fixed"

    # Orient axis from parent toward child for URDF convention
    parent_centroid = np.asarray(mesh.vertices[
        np.unique(mesh.faces[face_labels == parent_lbl])
    ].mean(axis=0))
    child_centroid = np.asarray(mesh.vertices[
        np.unique(mesh.faces[face_labels == child_lbl])
    ].mean(axis=0))
    chain_dir = child_centroid - parent_centroid
    if (chain_dir @ normal) < 0:
        normal = -normal

    return GeometricJoint(
        parent_label=parent_lbl, child_label=child_lbl,
        axis=normal,
        origin=center,
        radius=radius,
        plane_residual=plane_residual,
        n_boundary_edges=n_edges,
        type=jtype,
        confidence=circ,
    )


def extract_joints_for_tree(
    mesh: trimesh.Trimesh,
    face_labels: np.ndarray,
    parent_child_pairs: Sequence[tuple[int, int]],
    min_boundary_edges: int = 6,
    circularity_revolute_threshold: float = 0.55,
) -> list[GeometricJoint]:
    """Fit a joint for each (parent_label, child_label) pair.

    Tree-topology generalisation of `extract_joints_from_segmentation`.
    Each pair becomes one `GeometricJoint`; the URDF assembler can
    consume them as multi-child branches off any parent.

    Parameters
    ----------
    mesh
        Full input mesh (world frame).
    face_labels
        (F,) per-face link label.
    parent_child_pairs
        List of (parent_id, child_id) tuples, in any order.
    min_boundary_edges, circularity_revolute_threshold
        Same as `extract_joints_from_segmentation`.
    """
    return [
        _fit_one_joint(
            mesh, face_labels, int(p), int(c),
            min_boundary_edges, circularity_revolute_threshold,
        )
        for p, c in parent_child_pairs
    ]


def extract_joints_from_segmentation(
    mesh: trimesh.Trimesh,
    face_labels: np.ndarray,
    chain_order: Sequence[int],
    min_boundary_edges: int = 6,
    circularity_revolute_threshold: float = 0.55,
) -> list[GeometricJoint]:
    """Fit a joint between each pair of adjacent links in `chain_order`.

    Parameters
    ----------
    mesh
        The full input mesh (world frame).
    face_labels
        (F,) per-face link label, one entry per face.  Should be the
        FINAL merged labels (every face has a label).
    chain_order
        Link IDs in chain order (low Z to high Z, base → tip). The
        URDF builder uses the same ordering.
    min_boundary_edges
        Below this, treat as a fixed joint (no clear interface — the
        two links may be fused or the segmentation is too noisy).
    circularity_revolute_threshold
        Boundary circularity score below this → fixed joint.

    Returns
    -------
    A list of `GeometricJoint`, one per adjacent pair (length =
    len(chain_order) - 1).
    """
    pairs = [(int(chain_order[i]), int(chain_order[i + 1]))
             for i in range(len(chain_order) - 1)]
    return extract_joints_for_tree(
        mesh, face_labels, pairs,
        min_boundary_edges=min_boundary_edges,
        circularity_revolute_threshold=circularity_revolute_threshold,
    )
