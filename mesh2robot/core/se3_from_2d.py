"""Fit an SE(3) that maps known 3D points to observed 2D image observations.

Given:
  - X_0  : (N, 3) 3D points in world frame (from rendering the home-pose mesh)
  - p'   : (N, 2) observed image pixels where each X_0 ends up after the motion
  - T_world_to_cam_1 : camera pose of the state-1 image (ArUco-derived)
  - K    : camera intrinsics

We want the rigid SE(3) `T` such that

    project(K, T_world_to_cam_1, T @ X_0_i) ~ p'_i

RANSAC proposes random triples, solves for `T` in closed form per triple, then
counts inliers by reprojection error. The inlier set both identifies the
moving link (vertices on the rotating body) and refines `T`.

Used by the per-joint motion extractor. State-1 depth is NOT required — the
image observation + the known camera pose + the rigid-body assumption provide
enough constraint.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mesh2robot.core.mesh_projection import project_world_to_pixels
from mesh2robot.core.rigid_fit import horn


@dataclass
class RigidFit:
    T: np.ndarray              # 4x4 SE(3) world->world transform
    inliers: np.ndarray        # (M,) indices into the input arrays
    reprojection_err_px: float # mean pixel error over inliers


@dataclass
class BodyFit:
    """One rigid body extracted by multi-body RANSAC."""
    T: np.ndarray              # 4x4 SE(3) applied to this body's world points
    inliers: np.ndarray        # (M,) indices into the original correspondence arrays
    reprojection_err_px: float
    is_static: bool            # True if this body was identified as the static cluster


def _reprojection_error(
    X0: np.ndarray,
    p_obs: np.ndarray,
    T: np.ndarray,
    T_world_to_cam: np.ndarray,
    K: np.ndarray,
) -> np.ndarray:
    X1 = (T @ np.concatenate([X0, np.ones((len(X0), 1))], axis=1).T).T[:, :3]
    uv, z = project_world_to_pixels(X1, T_world_to_cam, K)
    err = np.linalg.norm(uv - p_obs, axis=1)
    # Invalidate points that went behind the camera
    err[z <= 0] = np.inf
    return err


def _fit_T_from_known_3d3d(X0: np.ndarray, X1: np.ndarray) -> np.ndarray:
    """Closed-form SE(3) from N>=3 3D-3D correspondences (Horn)."""
    return horn(X0, X1)


def _triangulate_with_depth_prior(
    X0: np.ndarray,
    p_obs: np.ndarray,
    T_world_to_cam: np.ndarray,
    K: np.ndarray,
    max_depth_delta: float = 0.5,
) -> np.ndarray:
    """Given X0 and p_obs, approximate X1 by assuming the depth stayed similar
    to X0's depth in the state-1 camera frame. Useful only as a RANSAC seed.

    Returns (N, 3) approximate X1 in world frame.
    """
    # Depth of X0 in the state-1 camera
    X0_h = np.concatenate([X0, np.ones((len(X0), 1))], axis=1)
    cam0 = (T_world_to_cam @ X0_h.T).T[:, :3]
    z_guess = np.clip(cam0[:, 2], 1e-3, None)  # preserve sign/positivity

    K_inv = np.linalg.inv(K)
    pix_h = np.concatenate([p_obs, np.ones((len(p_obs), 1))], axis=1)
    cam_dirs = (K_inv @ pix_h.T).T  # (N, 3) in camera frame
    # Scale by depth to get X1 in camera frame
    cam_X1 = cam_dirs * z_guess[:, None] / np.maximum(cam_dirs[:, 2:3], 1e-6)
    # Back to world
    T_cam_to_world = np.linalg.inv(T_world_to_cam)
    cam_X1_h = np.concatenate([cam_X1, np.ones((len(cam_X1), 1))], axis=1)
    X1 = (T_cam_to_world @ cam_X1_h.T).T[:, :3]
    return X1


def refine_T_via_pnp(
    X0: np.ndarray,                   # (N, 3) 3D points in world frame (state 0)
    p_obs: np.ndarray,                # (N, 2) observed state-1 pixels
    T_world_to_cam_1: np.ndarray,     # 4x4 state-1 camera pose
    K: np.ndarray,                    # 3x3 intrinsics
    dist_coeffs: np.ndarray | None = None,
) -> np.ndarray | None:
    """Refine the rigid motion T via PnP, bypassing depth-prior triangulation.

    The rigid motion T moves a world point X0 to a new world point T @ X0.
    That new point is then seen through the (known) state-1 camera at pixel
    p_obs. The combined transform `M = T_world_to_cam_1 @ T` maps X0 directly
    to the camera frame producing p_obs.

    We can solve for M via cv2.solvePnP (standard 3D-2D pose estimation), then
    recover T = inv(T_world_to_cam_1) @ M. No depth assumption needed.
    """
    import cv2
    if dist_coeffs is None:
        dist_coeffs = np.zeros(5)

    if len(X0) < 4:
        return None

    obj = X0.reshape(-1, 1, 3).astype(np.float64)
    img = p_obs.reshape(-1, 1, 2).astype(np.float64)

    try:
        ok, rvec, tvec = cv2.solvePnP(
            obj, img, K, dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
    except cv2.error:
        return None
    if not ok:
        return None

    # M = [R|t] maps world points (X0) -> state-1 camera frame
    R, _ = cv2.Rodrigues(rvec)
    M = np.eye(4)
    M[:3, :3] = R
    M[:3, 3] = tvec.reshape(3)

    # T = inv(T_world_to_cam_1) @ M  gives T s.t. project(cam1, T @ X0) = p_obs
    T = np.linalg.inv(T_world_to_cam_1) @ M
    return T


def ransac_rigid_from_2d_obs(
    X0: np.ndarray,                   # (N, 3)
    p_obs: np.ndarray,                # (N, 2)
    T_world_to_cam: np.ndarray,       # 4x4
    K: np.ndarray,                    # 3x3
    reproj_threshold_px: float = 3.0,
    n_trials: int = 500,
    min_inliers: int = 30,
    rng_seed: int = 0,
) -> RigidFit | None:
    """RANSAC for the SE(3) that best maps X0 -> observed state-1 pixels p_obs.

    The candidate model is generated from a 3-point subset by:
      1. Approximating X1 for each of the 3 samples via depth-prior triangulation.
      2. Running Horn on the 3 (X0, X1) triples → candidate T.
      3. Scoring by reprojection error across all N correspondences.

    Returns None if no model reaches `min_inliers` inliers.
    """
    N = len(X0)
    if N < 3:
        return None
    rng = np.random.default_rng(rng_seed)

    best_inliers: np.ndarray | None = None
    best_err = np.inf

    for _ in range(n_trials):
        idx = rng.choice(N, size=3, replace=False)
        X0_s = X0[idx]
        p_s = p_obs[idx]
        X1_seed = _triangulate_with_depth_prior(X0_s, p_s, T_world_to_cam, K)
        try:
            T = _fit_T_from_known_3d3d(X0_s, X1_seed)
        except np.linalg.LinAlgError:
            continue

        errs = _reprojection_error(X0, p_obs, T, T_world_to_cam, K)
        inlier_mask = errs < reproj_threshold_px
        n_in = int(inlier_mask.sum())
        if n_in < 3:
            continue

        # Refit T from the larger inlier set via the depth-prior triangulation
        inlier_idx = np.where(inlier_mask)[0]
        X0_in = X0[inlier_idx]
        p_in = p_obs[inlier_idx]
        X1_in = _triangulate_with_depth_prior(X0_in, p_in, T_world_to_cam, K)
        try:
            T_refit = _fit_T_from_known_3d3d(X0_in, X1_in)
        except np.linalg.LinAlgError:
            T_refit = T

        errs_refit = _reprojection_error(X0, p_obs, T_refit, T_world_to_cam, K)
        inlier_mask_refit = errs_refit < reproj_threshold_px
        n_in_refit = int(inlier_mask_refit.sum())
        mean_err = float(errs_refit[inlier_mask_refit].mean()) if n_in_refit > 0 else np.inf

        score = (n_in_refit, -mean_err)
        best_score = (len(best_inliers) if best_inliers is not None else -1,
                      -best_err if best_inliers is not None else -np.inf)
        if score > best_score:
            best_inliers = np.where(inlier_mask_refit)[0]
            best_err = mean_err

    if best_inliers is None or len(best_inliers) < min_inliers:
        return None

    # Final refit on best inlier set
    X0_in = X0[best_inliers]
    p_in = p_obs[best_inliers]
    X1_in = _triangulate_with_depth_prior(X0_in, p_in, T_world_to_cam, K)
    T_final = _fit_T_from_known_3d3d(X0_in, X1_in)
    errs_final = _reprojection_error(X0, p_obs, T_final, T_world_to_cam, K)
    return RigidFit(
        T=T_final,
        inliers=np.where(errs_final < reproj_threshold_px)[0],
        reprojection_err_px=float(errs_final[errs_final < reproj_threshold_px].mean()),
    )


def multi_body_ransac_2d(
    X0: np.ndarray,                   # (N, 3)
    p_obs: np.ndarray,                # (N, 2)
    T_world_to_cam: np.ndarray,       # 4x4 of state-1 camera
    K: np.ndarray,                    # 3x3
    reproj_threshold_px: float = 3.0,
    identity_tolerance_px: float = 2.0,
    n_trials: int = 500,
    min_inliers: int = 30,
    max_bodies: int = 8,
    rng_seed: int = 0,
) -> list[BodyFit]:
    """Hierarchical RANSAC: peel off one rigid body at a time.

    Algorithm:
      1. Test the identity hypothesis (static cluster). Any correspondence
         whose reprojection under T=I is below `identity_tolerance_px` is an
         inlier to the static body. Remove those first.
      2. On remaining correspondences, run `ransac_rigid_from_2d_obs` to find
         the next-largest rigid cluster.
      3. Remove its inliers; repeat until no cluster exceeds `min_inliers`.

    Each returned BodyFit carries the absolute inlier indices (relative to the
    original X0/p_obs arrays), NOT the ones after prior peels.
    """
    N = len(X0)
    if N < 3:
        return []

    # Stage 1: identity (static body)
    err_identity = _reprojection_error(X0, p_obs, np.eye(4), T_world_to_cam, K)
    static_inliers_mask = err_identity < identity_tolerance_px
    bodies: list[BodyFit] = []
    if int(static_inliers_mask.sum()) >= min_inliers:
        static_idx = np.where(static_inliers_mask)[0]
        bodies.append(BodyFit(
            T=np.eye(4),
            inliers=static_idx,
            reprojection_err_px=float(err_identity[static_idx].mean()),
            is_static=True,
        ))
        remaining_mask = ~static_inliers_mask
    else:
        remaining_mask = np.ones(N, dtype=bool)

    remaining = np.where(remaining_mask)[0]

    # Stage 2: iterative RANSAC on remaining
    for _ in range(max_bodies):
        if len(remaining) < min_inliers:
            break
        fit = ransac_rigid_from_2d_obs(
            X0=X0[remaining],
            p_obs=p_obs[remaining],
            T_world_to_cam=T_world_to_cam,
            K=K,
            reproj_threshold_px=reproj_threshold_px,
            n_trials=n_trials,
            min_inliers=min_inliers,
            rng_seed=rng_seed + len(bodies),
        )
        if fit is None:
            break
        abs_inliers = remaining[fit.inliers]

        # Refine T directly from 2D reprojection on all inliers (PnP).
        # Skips the depth-prior triangulation used by the initial Horn fit,
        # which biases the axis direction toward the triangulation error.
        T_refined = refine_T_via_pnp(
            X0=X0[abs_inliers],
            p_obs=p_obs[abs_inliers],
            T_world_to_cam_1=T_world_to_cam,
            K=K,
        )
        if T_refined is not None:
            # Sanity check: new reprojection error shouldn't balloon
            err_new = _reprojection_error(X0, p_obs, T_refined, T_world_to_cam, K)
            inl_new = err_new < reproj_threshold_px
            if inl_new.sum() >= 3:
                mean_err = float(err_new[inl_new].mean())
                bodies.append(BodyFit(
                    T=T_refined,
                    inliers=np.where(inl_new)[0],
                    reprojection_err_px=mean_err,
                    is_static=False,
                ))
                remaining_mask[np.where(inl_new)[0]] = False
                remaining = np.where(remaining_mask)[0]
                continue

        # Fall back to the Horn fit if PnP refinement failed.
        bodies.append(BodyFit(
            T=fit.T,
            inliers=abs_inliers,
            reprojection_err_px=fit.reprojection_err_px,
            is_static=False,
        ))
        remaining_mask[abs_inliers] = False
        remaining = np.where(remaining_mask)[0]

    return bodies
