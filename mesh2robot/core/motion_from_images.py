"""Per-pair motion extraction via ORB features + multi-body RANSAC.

Replaces dense optical flow with discrete keypoint matching. Benefits:
  - Handles LARGE inter-state motion (flow breaks beyond its search window)
  - Handles rotation in place (textured wheel spinning on its axle)
  - Handles self-occlusion (features have scale / orientation invariance)
  - Natively produces multi-body segmentation (each rigid cluster = one body)

Per image pair we emit a list of BodyMotion — one per rigid body detected:
one "static" body (stationary parts of the robot) plus one body per moving
sub-chain. For a serial arm with only joint_k moved between state0 and state1,
we get 2 bodies: static (links 0..k-1) and moving (links k..N).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import trimesh

from mesh2robot.core.feature_matching import MatchResult, detect_and_match
from mesh2robot.core.mesh_projection import lift_keypoints_to_mesh, render_mesh_depth
from mesh2robot.core.rigid_fit import screw_from_transform
from mesh2robot.core.se3_from_2d import BodyFit, multi_body_ransac_2d
from mesh2robot.fiducial.pose import BoardDetection, detect_board


# ---------------------------------------------------------------------------
# Per-pair result
# ---------------------------------------------------------------------------

@dataclass
class BodyMotion:
    """One rigid body identified within a state pair."""
    T: np.ndarray                     # 4x4 SE(3) applied to this body's world points
    is_static: bool
    inlier_feature_indices: np.ndarray  # into the match arrays
    face_indices: np.ndarray            # mesh face indices the inlier features sit on
    vertex_mask: np.ndarray             # (V,) bool — vertices belonging to this body
    reprojection_err_px: float


@dataclass
class PairMotion:
    """Full result of one adjacent-state pair."""
    state0_detection: BoardDetection
    state1_detection: BoardDetection
    matches: MatchResult
    n_features_hit_mesh: int
    bodies: list[BodyMotion]


def _face_indices_to_vertex_mask(
    mesh: trimesh.Trimesh, face_idx: np.ndarray,
) -> np.ndarray:
    """Union of vertices incident on the given faces."""
    V = len(mesh.vertices)
    mask = np.zeros(V, dtype=bool)
    if len(face_idx) == 0:
        return mask
    faces = np.asarray(mesh.faces)
    mask[faces[face_idx].flatten()] = True
    return mask


def _compute_robot_silhouette_mask(
    mesh: trimesh.Trimesh,
    T_world_to_cam: np.ndarray,
    K: np.ndarray,
    image_shape: tuple[int, int],
    dilate_px: int = 7,
    step_px: int = 2,
) -> np.ndarray:
    """Render the mesh from the given camera, return a uint8 mask (255 = robot)."""
    H, W = image_shape
    proj = render_mesh_depth(mesh, T_world_to_cam, K, (W, H), step_px=step_px)
    hit = proj.hit_mask
    if hit.shape != (H, W):
        hit = cv2.resize(hit.astype(np.uint8), (W, H),
                         interpolation=cv2.INTER_NEAREST).astype(bool)
    mask8 = (hit.astype(np.uint8)) * 255
    if dilate_px > 0:
        kernel = np.ones((dilate_px, dilate_px), np.uint8)
        mask8 = cv2.dilate(mask8, kernel)
    return mask8


def extract_pair_bodies(
    mesh: trimesh.Trimesh,
    state0_path: str | Path,
    state1_path: str | Path,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    n_features: int = 8000,
    ratio: float = 0.8,
    use_sift: bool = False,
    reproj_threshold_px: float = 3.0,
    identity_tolerance_px: float = 2.0,
    min_inliers: int = 30,
    ransac_trials: int = 500,
    max_bodies: int = 6,
    mask_features_to_robot: bool = True,
    silhouette_dilate_px: int = 7,
) -> PairMotion | None:
    """Detect bodies moving between state 0 and state 1 of one joint's capture."""
    img0 = cv2.imread(str(state0_path))
    img1 = cv2.imread(str(state1_path))
    if img0 is None:
        raise FileNotFoundError(state0_path)
    if img1 is None:
        raise FileNotFoundError(state1_path)
    # Crop both to common size (handheld size drift)
    if img0.shape != img1.shape:
        h = min(img0.shape[0], img1.shape[0])
        w = min(img0.shape[1], img1.shape[1])
        img0 = img0[:h, :w]
        img1 = img1[:h, :w]

    det0 = detect_board(img0, camera_matrix, dist_coeffs)
    det1 = detect_board(img1, camera_matrix, dist_coeffs)
    if det0 is None or det1 is None:
        return None

    # Mask ORB detection to the robot silhouette in state-0. Without this, 95%
    # of features land on the textured ArUco board / truss / pallet, and the
    # mesh-hit filter throws them away. Masking inverts that — nearly every
    # detected keypoint sits on the robot and will ray-cast onto the mesh.
    mask0 = None
    if mask_features_to_robot:
        mask0 = _compute_robot_silhouette_mask(
            mesh=mesh,
            T_world_to_cam=det0.T_world_to_cam,
            K=camera_matrix,
            image_shape=(img0.shape[0], img0.shape[1]),
            dilate_px=silhouette_dilate_px,
        )

    match = detect_and_match(
        img0, img1,
        n_features=n_features, ratio=ratio, use_sift=use_sift,
        mask0=mask0,
    )
    if len(match.p0) < 8:
        return PairMotion(
            state0_detection=det0, state1_detection=det1, matches=match,
            n_features_hit_mesh=0, bodies=[],
        )

    # Lift state-0 keypoints onto the mesh (world frame).
    X0_world, face_idx, hit_mask = lift_keypoints_to_mesh(
        mesh=mesh,
        keypoints_2d=match.p0,
        T_world_to_cam=det0.T_world_to_cam,
        K=camera_matrix,
    )
    if int(hit_mask.sum()) < min_inliers:
        return PairMotion(
            state0_detection=det0, state1_detection=det1, matches=match,
            n_features_hit_mesh=int(hit_mask.sum()), bodies=[],
        )

    X0_on_mesh = X0_world[hit_mask]
    p1_on_mesh = match.p1[hit_mask]
    face_on_mesh = face_idx[hit_mask]
    # Keep a mapping from "on-mesh index" back to original match index
    orig_indices = np.where(hit_mask)[0]

    body_fits: list[BodyFit] = multi_body_ransac_2d(
        X0=X0_on_mesh,
        p_obs=p1_on_mesh,
        T_world_to_cam=det1.T_world_to_cam,
        K=camera_matrix,
        reproj_threshold_px=reproj_threshold_px,
        identity_tolerance_px=identity_tolerance_px,
        n_trials=ransac_trials,
        min_inliers=min_inliers,
        max_bodies=max_bodies,
    )

    bodies: list[BodyMotion] = []
    for fit in body_fits:
        inlier_on_mesh = fit.inliers
        orig_inlier = orig_indices[inlier_on_mesh]
        inlier_faces = face_on_mesh[inlier_on_mesh]
        inlier_faces = inlier_faces[inlier_faces >= 0]
        vmask = _face_indices_to_vertex_mask(mesh, inlier_faces)
        bodies.append(BodyMotion(
            T=fit.T,
            is_static=fit.is_static,
            inlier_feature_indices=orig_inlier,
            face_indices=inlier_faces,
            vertex_mask=vmask,
            reprojection_err_px=fit.reprojection_err_px,
        ))

    return PairMotion(
        state0_detection=det0, state1_detection=det1, matches=match,
        n_features_hit_mesh=int(hit_mask.sum()), bodies=bodies,
    )


# ---------------------------------------------------------------------------
# Multi-state aggregation: pick the "moving" body per pair, then combine
# ---------------------------------------------------------------------------

@dataclass
class JointMultiStateResult:
    """Aggregated motion for one joint across N+1 states (→ N pairs)."""
    joint_name: str
    T_total: np.ndarray                  # composed SE(3) state_0 → state_N
    axis_world: np.ndarray               # averaged screw axis
    origin_world: np.ndarray             # averaged screw origin
    total_angle_rad: float               # sum of signed per-pair angles
    moved_vertex_mask: np.ndarray        # union across pairs
    per_pair: list[PairMotion]           # raw per-pair data
    per_pair_moving_body_idx: list[int]  # which body in each pair was judged "moving"
    axis_spread_deg: float
    n_pairs_ok: int


def _pick_moving_body(bodies: list[BodyMotion]) -> int | None:
    """Pick the body representing the actuated joint's motion.

    Heuristic: non-static body with largest inlier count. For single-joint
    captures this is the "moving sub-chain". For multi-joint scenarios we'd
    need to match bodies across pairs — for now we assume one joint per folder.
    """
    candidates = [(i, b) for i, b in enumerate(bodies) if not b.is_static]
    if not candidates:
        return None
    candidates.sort(key=lambda ib: -len(ib[1].inlier_feature_indices))
    return candidates[0][0]


def _align_axis_sign(reference: np.ndarray, axis: np.ndarray) -> tuple[np.ndarray, int]:
    d = float(reference @ axis)
    if d < 0:
        return -axis, -1
    return axis, +1


def extract_joint_motion_multi(
    mesh: trimesh.Trimesh,
    state_paths: list[str | Path],
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    joint_name: str = "joint",
    max_axis_spread_deg: float = 15.0,
    **pair_kwargs,
) -> JointMultiStateResult | None:
    """Run per-pair body extraction, pick the moving body per pair, aggregate."""
    if len(state_paths) < 2:
        return None

    per_pair: list[PairMotion] = []
    moving_idx: list[int | None] = []
    for i in range(len(state_paths) - 1):
        pm = extract_pair_bodies(
            mesh=mesh,
            state0_path=state_paths[i],
            state1_path=state_paths[i + 1],
            camera_matrix=camera_matrix,
            dist_coeffs=dist_coeffs,
            **pair_kwargs,
        )
        per_pair.append(pm)
        if pm is None or not pm.bodies:
            moving_idx.append(None)
        else:
            moving_idx.append(_pick_moving_body(pm.bodies))

    ok_pairs = [(p, mi) for p, mi in zip(per_pair, moving_idx)
                if p is not None and mi is not None]
    if not ok_pairs:
        # No pair produced a moving body. Two sub-cases:
        #  A) Every pair found only a STATIC body. This is the correct answer
        #     for a genuinely static robot (calibration-sanity check).
        #  B) Every pair failed ArUco or feature lift entirely. That's a real
        #     failure.
        all_static = all(
            p is not None and len(p.bodies) >= 1 and all(b.is_static for b in p.bodies)
            for p in per_pair
        )
        if all_static:
            V = len(mesh.vertices)
            return JointMultiStateResult(
                joint_name=joint_name,
                T_total=np.eye(4),
                axis_world=np.array([0.0, 0.0, 1.0]),   # placeholder
                origin_world=np.zeros(3),
                total_angle_rad=0.0,
                moved_vertex_mask=np.zeros(V, dtype=bool),
                per_pair=per_pair,
                per_pair_moving_body_idx=[-1] * len(per_pair),
                axis_spread_deg=0.0,
                n_pairs_ok=0,
            )
        return None

    # Aggregate the moving body's SE(3) across pairs.
    screws = [screw_from_transform(p.bodies[mi].T) for p, mi in ok_pairs]
    ref_axis = screws[0]["axis"].copy()
    aligned = [ref_axis.copy()]
    signed_angles = [float(screws[0]["angle"])]
    origins = [screws[0]["origin"].copy()]
    for sc in screws[1:]:
        a, sign = _align_axis_sign(ref_axis, sc["axis"])
        aligned.append(a)
        signed_angles.append(sign * float(sc["angle"]))
        origins.append(sc["origin"].copy())

    axis_spreads = [float(np.rad2deg(np.arccos(np.clip(a @ ref_axis, -1, 1))))
                    for a in aligned]
    axis_spread_deg = float(max(axis_spreads))

    axis = np.asarray(aligned).mean(axis=0)
    axis /= (np.linalg.norm(axis) + 1e-12)
    origin = np.asarray(origins).mean(axis=0)
    total_angle = float(sum(signed_angles))

    # Compose: state_0 -> state_N
    T_total = np.eye(4)
    for (p, mi) in ok_pairs:
        T_total = p.bodies[mi].T @ T_total

    # Union moved vertex masks across pairs (the moving body in each pair).
    V = len(mesh.vertices)
    moved_mask = np.zeros(V, dtype=bool)
    for (p, mi) in ok_pairs:
        moved_mask |= p.bodies[mi].vertex_mask

    return JointMultiStateResult(
        joint_name=joint_name,
        T_total=T_total,
        axis_world=axis,
        origin_world=origin,
        total_angle_rad=total_angle,
        moved_vertex_mask=moved_mask,
        per_pair=per_pair,
        per_pair_moving_body_idx=[mi if mi is not None else -1 for mi in moving_idx],
        axis_spread_deg=axis_spread_deg,
        n_pairs_ok=len(ok_pairs),
    )
