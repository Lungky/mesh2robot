"""Motion-based segmentation of a multi-pose vertex cloud.

Given K pose-meshes with 1-to-1 vertex correspondence across poses, partition
the vertices into rigid bodies (each body = one link). A body is a maximal
subset of vertices that shares a single SE(3) transform between pose 0 and
each other pose.

Algorithm: multi-pose hierarchical RANSAC.
    1. While unassigned vertices remain:
         a. Randomly sample 3 unassigned vertices.
         b. For each pose t >= 1, fit T_t from those 3 correspondences.
         c. An unassigned vertex is an inlier if, for ALL poses t, the
            residual |T_t @ v(0) - v(t)| is below a threshold.
         d. Repeat n_trials times; keep the largest inlier set.
         e. Refit T_t from all inliers, mark them as the next body.
    2. Return label array.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mesh2robot.core.rigid_fit import apply_transform, horn, transform_residuals


@dataclass
class SegmentationResult:
    labels: np.ndarray                  # (N,) int; -1 = unassigned
    body_transforms: list[list[np.ndarray]]   # body_transforms[body_id][pose_t] = 4x4
    n_bodies: int


def _residual_across_poses(
    T_per_pose: list[np.ndarray],
    pose_pts: np.ndarray,   # (K, N, 3)
    indices: np.ndarray,
) -> np.ndarray:
    """Max residual across poses for the given vertex indices."""
    v0 = pose_pts[0, indices]
    max_err = np.zeros(len(indices))
    for t in range(1, pose_pts.shape[0]):
        err = np.linalg.norm(apply_transform(T_per_pose[t], v0) - pose_pts[t, indices], axis=1)
        max_err = np.maximum(max_err, err)
    return max_err


def _fit_T_per_pose(pose_pts: np.ndarray, indices: np.ndarray) -> list[np.ndarray] | None:
    """Fit SE(3) per pose from the given vertex indices. Returns None on failure."""
    K = pose_pts.shape[0]
    src = pose_pts[0, indices]
    Ts = [np.eye(4)]
    try:
        for t in range(1, K):
            Ts.append(horn(src, pose_pts[t, indices]))
    except np.linalg.LinAlgError:
        return None
    return Ts


def _refine_body(
    pose_pts: np.ndarray,
    initial_inliers: np.ndarray,
    threshold: float,
    unassigned: np.ndarray,
    max_iters: int = 5,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """LO-RANSAC refinement: alternate between refit-from-inliers and
    re-collect-inliers until the inlier set stabilizes.
    """
    current = initial_inliers
    Ts = _fit_T_per_pose(pose_pts, current)
    for _ in range(max_iters):
        if Ts is None:
            break
        err = _residual_across_poses(Ts, pose_pts, unassigned)
        new_in = unassigned[err < threshold]
        if len(new_in) == len(current) and np.array_equal(
            np.sort(new_in), np.sort(current)
        ):
            break
        if len(new_in) < 3:
            break
        current = new_in
        Ts = _fit_T_per_pose(pose_pts, current)
    return current, (Ts if Ts is not None else [np.eye(4)] * pose_pts.shape[0])


def _sample_spatially_diverse(
    rng: np.random.Generator,
    candidates: np.ndarray,
    pts: np.ndarray,
    min_spread: float,
    max_attempts: int = 50,
) -> np.ndarray | None:
    """Sample 3 indices whose pose-0 positions span at least min_spread pairwise."""
    for _ in range(max_attempts):
        if len(candidates) < 3:
            return None
        sample = rng.choice(candidates, size=3, replace=False)
        p = pts[sample]
        d01 = np.linalg.norm(p[1] - p[0])
        d02 = np.linalg.norm(p[2] - p[0])
        d12 = np.linalg.norm(p[2] - p[1])
        if min(d01, d02, d12) >= min_spread and not _degenerate_triple(p):
            return sample
    return None


def segment_multi_pose(
    pose_pts: np.ndarray,       # (K, N, 3)
    threshold: float = 1e-3,
    min_inliers: int = 100,
    max_bodies: int = 12,
    n_trials: int = 200,
    rng_seed: int = 0,
    min_spread: float = 0.02,   # 2 cm minimum triangle side
    lo_ransac: bool = True,
) -> SegmentationResult:
    """Partition N vertices into rigid bodies using K poses.

    Uses LO-RANSAC: each successful 3-point trial is refined by iteratively
    refitting T from its inliers until the set stabilizes. This makes the
    algorithm robust to per-vertex noise because the final body transform is
    estimated from hundreds-to-thousands of points, not from 3 potentially
    bad samples.

    Spatial diversity: RANSAC samples are drawn such that the three pose-0
    positions are at least `min_spread` apart pairwise. Widely-spaced
    samples give much lower rotation error per sample.
    """
    assert pose_pts.ndim == 3 and pose_pts.shape[2] == 3
    K, N, _ = pose_pts.shape
    rng = np.random.default_rng(rng_seed)

    labels = np.full(N, -1, dtype=np.int32)
    bodies_T: list[list[np.ndarray]] = []

    for body_id in range(max_bodies):
        unassigned = np.where(labels == -1)[0]
        if len(unassigned) < min_inliers:
            break

        best_inlier_count = 0
        best_inliers: np.ndarray | None = None
        best_T: list[np.ndarray] | None = None

        for _ in range(n_trials):
            sample = _sample_spatially_diverse(
                rng, unassigned, pose_pts[0], min_spread
            )
            if sample is None:
                break

            Ts = _fit_T_per_pose(pose_pts, sample)
            if Ts is None:
                continue

            err = _residual_across_poses(Ts, pose_pts, unassigned)
            inlier_mask = err < threshold
            n_in = int(inlier_mask.sum())
            if n_in < 3:
                continue

            if lo_ransac:
                refined_inliers, refined_T = _refine_body(
                    pose_pts, unassigned[inlier_mask], threshold, unassigned
                )
                n_in = len(refined_inliers)
                if n_in > best_inlier_count:
                    best_inlier_count = n_in
                    best_inliers = refined_inliers
                    best_T = refined_T
            else:
                if n_in > best_inlier_count:
                    best_inlier_count = n_in
                    best_inliers = unassigned[inlier_mask]
                    best_T = Ts

        if best_inliers is None or best_inlier_count < min_inliers:
            break

        # Final refit with all inliers (already done by LO-RANSAC, but idempotent).
        refined_T = _fit_T_per_pose(pose_pts, best_inliers) or best_T

        labels[best_inliers] = body_id
        bodies_T.append(refined_T)

    return SegmentationResult(labels=labels, body_transforms=bodies_T, n_bodies=len(bodies_T))


def _degenerate_triple(pts: np.ndarray, eps: float = 1e-6) -> bool:
    """Check if three points are collinear or coincident."""
    v1 = pts[1] - pts[0]
    v2 = pts[2] - pts[0]
    return np.linalg.norm(np.cross(v1, v2)) < eps


def _transform_distance(Ta: list[np.ndarray], Tb: list[np.ndarray]) -> float:
    """Max Frobenius distance between two per-pose transform lists.

    For each pose t, compute ||T_a(t) - T_b(t)||_F and return the max.
    Two bodies that represent the same rigid group will have near-identical
    transforms at every pose (modulo RANSAC noise).
    """
    K = len(Ta)
    return max(float(np.linalg.norm(Ta[t] - Tb[t])) for t in range(K))


def assign_orphans_to_nearest_body(
    seg: SegmentationResult,
    pose_pts: np.ndarray,
    max_residual: float | None = None,
) -> SegmentationResult:
    """Assign each unlabeled vertex to the body whose transform best explains
    its motion. Optionally cap by max_residual (leave truly bad vertices as -1).

    This recovers vertices that failed the RANSAC inlier threshold due to
    outlier noise but clearly belong to one of the found bodies.
    """
    unassigned = np.where(seg.labels == -1)[0]
    if len(unassigned) == 0 or seg.n_bodies == 0:
        return seg

    K = pose_pts.shape[0]
    # For each body, compute max residual across poses for each unassigned vertex.
    best_err = np.full(len(unassigned), np.inf)
    best_body = np.full(len(unassigned), -1, dtype=np.int32)
    for b, Ts in enumerate(seg.body_transforms):
        max_e = np.zeros(len(unassigned))
        v0 = pose_pts[0, unassigned]
        for t in range(1, K):
            e = np.linalg.norm(apply_transform(Ts[t], v0) - pose_pts[t, unassigned], axis=1)
            max_e = np.maximum(max_e, e)
        better = max_e < best_err
        best_err[better] = max_e[better]
        best_body[better] = b

    labels = seg.labels.copy()
    if max_residual is None:
        ok = np.ones(len(unassigned), dtype=bool)
    else:
        ok = best_err < max_residual
    labels[unassigned[ok]] = best_body[ok]

    return SegmentationResult(
        labels=labels, body_transforms=seg.body_transforms, n_bodies=seg.n_bodies
    )


def merge_duplicate_bodies(
    seg: SegmentationResult,
    pose_pts: np.ndarray,
    merge_tol: float = 0.01,
) -> SegmentationResult:
    """Merge bodies whose per-pose transforms are within merge_tol.

    After hierarchical RANSAC under noise, one real link can get split into
    several "bodies" with near-identical transforms. This collapses them.
    """
    n = seg.n_bodies
    if n <= 1:
        return seg

    # Union-find
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(n):
        for j in range(i + 1, n):
            if _transform_distance(seg.body_transforms[i], seg.body_transforms[j]) < merge_tol:
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[rj] = ri

    # Build new labels
    roots = sorted({find(i) for i in range(n)})
    root_to_new = {r: k for k, r in enumerate(roots)}

    new_labels = seg.labels.copy()
    for i in range(n):
        new_id = root_to_new[find(i)]
        new_labels[seg.labels == i] = new_id

    # Refit transforms per merged body
    new_T: list[list[np.ndarray]] = []
    for r in roots:
        idxs = np.where(new_labels == root_to_new[r])[0]
        Ts = _fit_T_per_pose(pose_pts, idxs) or seg.body_transforms[r]
        new_T.append(Ts)

    return SegmentationResult(
        labels=new_labels, body_transforms=new_T, n_bodies=len(roots)
    )
