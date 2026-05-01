"""Detect the ArUco grid board in an image and recover the camera pose.

Returns a transform from world (robot base, as defined in `board.py`) to the
camera's optical frame, plus the raw board-frame transform in case callers
need it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from mesh2robot.fiducial.board import (
    ARUCO_DICT_ID,
    BOARD_TO_WORLD,
    MOUNT_CENTER_BOARD,
    create_grid_board,
)


@dataclass
class BoardDetection:
    n_markers: int                       # number of markers detected
    ids: np.ndarray                      # (N,) marker ids
    corners: list[np.ndarray]            # list of (1, 4, 2) image-space marker corners
    rvec_board_to_cam: np.ndarray        # (3,) Rodrigues
    tvec_board_to_cam: np.ndarray        # (3,) meters
    reprojection_error_px: float         # mean reproj error across matched corners
    T_world_to_cam: np.ndarray           # 4x4; transforms WORLD points to camera frame
    T_cam_to_world: np.ndarray           # 4x4; transforms CAMERA points to world frame


def _rvec_tvec_to_T(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    """Convert (rvec, tvec) from solvePnP to a 4x4 SE(3) matrix that maps
    object-frame points to camera-frame points."""
    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = tvec.reshape(3)
    return T


def _compose(T_a: np.ndarray, T_b: np.ndarray) -> np.ndarray:
    return T_a @ T_b


def _T_board_to_world() -> np.ndarray:
    """SE(3) mapping BOARD points to WORLD points.

    The world origin sits at the midpoint of the two mount markers, and axes
    rotate per `BOARD_TO_WORLD` from board axes to world axes.
    """
    T = np.eye(4)
    T[:3, :3] = BOARD_TO_WORLD
    # world = BOARD_TO_WORLD @ (board - MOUNT_CENTER_BOARD)
    # so board-to-world as SE(3) is a rotation followed by translation; its
    # action on a board point is: world = R @ board - R @ MOUNT_CENTER_BOARD.
    T[:3, 3] = -BOARD_TO_WORLD @ MOUNT_CENTER_BOARD
    return T


def detect_board(
    image_bgr: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    min_markers: int = 4,
) -> BoardDetection | None:
    """Detect the grid board in a single image and solve for the camera pose.

    image_bgr: H x W x 3, BGR (cv2 default).
    camera_matrix: 3x3 intrinsics.
    dist_coeffs: (k,) lens distortion, e.g. from cv2.calibrateCamera.
    min_markers: abort if fewer than this many markers are detected.

    Returns None if detection fails (too few markers, PnP diverges, etc.).
    """
    board = create_grid_board()
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT_ID)
    params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(dictionary, params)

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY) if image_bgr.ndim == 3 else image_bgr
    corners, ids, _ = detector.detectMarkers(gray)
    if ids is None or len(ids) < min_markers:
        return None

    # Match detected 2D corners to board's known 3D corners
    obj_pts, img_pts = board.matchImagePoints(corners, ids)
    if obj_pts is None or len(obj_pts) < 4:
        return None

    ok, rvec, tvec = cv2.solvePnP(
        obj_pts, img_pts, camera_matrix, dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        return None

    # Reprojection error
    reproj, _ = cv2.projectPoints(obj_pts, rvec, tvec, camera_matrix, dist_coeffs)
    err = float(np.mean(np.linalg.norm(reproj.reshape(-1, 2) - img_pts.reshape(-1, 2), axis=1)))

    T_board_to_cam = _rvec_tvec_to_T(rvec, tvec)
    T_board_to_world = _T_board_to_world()
    T_world_to_board = np.linalg.inv(T_board_to_world)
    T_world_to_cam = _compose(T_board_to_cam, T_world_to_board)
    T_cam_to_world = np.linalg.inv(T_world_to_cam)

    return BoardDetection(
        n_markers=len(ids),
        ids=ids.flatten(),
        corners=list(corners),
        rvec_board_to_cam=rvec.reshape(3),
        tvec_board_to_cam=tvec.reshape(3),
        reprojection_error_px=err,
        T_world_to_cam=T_world_to_cam,
        T_cam_to_world=T_cam_to_world,
    )


def detect_camera_pose_world(
    image_path: str | Path,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    min_markers: int = 4,
) -> BoardDetection | None:
    """Convenience wrapper: load image from path, then detect."""
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    return detect_board(img, camera_matrix, dist_coeffs, min_markers=min_markers)
