"""Rigid transform estimation and screw-axis recovery.

All transforms here are 4x4 SE(3) matrices in homogeneous form.
"""

from __future__ import annotations

import numpy as np


def horn(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Least-squares SE(3) aligning src -> dst (Horn / Kabsch, no reflection).

    src, dst: (N, 3) matched point clouds.
    Returns: 4x4 SE(3) matrix T such that T @ src ~ dst.
    """
    assert src.shape == dst.shape and src.shape[1] == 3
    cs = src.mean(0)
    cd = dst.mean(0)
    S = src - cs
    D = dst - cd
    H = S.T @ D
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    M = np.diag([1.0, 1.0, d])
    R = Vt.T @ M @ U.T
    t = cd - R @ cs
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def apply_transform(T: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Apply 4x4 SE(3) to (N, 3) points."""
    return (T[:3, :3] @ pts.T).T + T[:3, 3]


def transform_residuals(T: np.ndarray, src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Per-point residual norms after applying T to src."""
    return np.linalg.norm(apply_transform(T, src) - dst, axis=1)


def screw_from_transform(T: np.ndarray) -> dict:
    """Extract screw parameters from SE(3).

    Returns dict with:
      type   : 'revolute' | 'prismatic' | 'fixed'
      axis   : unit vector (3,)
      origin : point on axis (3,), closest to world origin
      angle  : rotation magnitude (rad), 0 for prismatic
      trans  : translation along axis (m), 0 for revolute
    """
    R = T[:3, :3]
    t = T[:3, 3]

    # Rotation angle from trace
    cos_a = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    angle = float(np.arccos(cos_a))

    if angle < 1e-6:
        # Pure translation (or identity)
        trans = float(np.linalg.norm(t))
        if trans < 1e-6:
            return {"type": "fixed", "axis": np.array([0.0, 0.0, 1.0]),
                    "origin": np.zeros(3), "angle": 0.0, "trans": 0.0}
        return {"type": "prismatic", "axis": t / trans,
                "origin": np.zeros(3), "angle": 0.0, "trans": trans}

    # Rotation axis from skew part of R
    axis = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    axis_norm = np.linalg.norm(axis)
    if axis_norm < 1e-8:
        # Angle ~ pi, need alternate extraction
        # R = 2 u u^T - I  when angle = pi
        B = (R + np.eye(3)) / 2.0
        axis = np.sqrt(np.clip(np.diag(B), 0.0, None))
        # Sign disambiguation
        for i in range(3):
            if axis[i] > 1e-6:
                for j in range(3):
                    if i != j:
                        axis[j] = np.sign(B[i, j]) * abs(axis[j])
                break
        axis /= np.linalg.norm(axis)
    else:
        axis = axis / axis_norm

    # Translation along axis
    trans_along = float(axis @ t)

    # Find a point on the screw axis.
    # For a screw motion: t = (I - R) @ origin + trans_along * axis
    # A = (I - R) has rank 2 with null space = axis; use least-squares.
    A = np.eye(3) - R
    t_perp = t - trans_along * axis
    origin = np.linalg.lstsq(A, t_perp, rcond=None)[0]

    return {"type": "revolute", "axis": axis, "origin": origin,
            "angle": angle, "trans": trans_along}
