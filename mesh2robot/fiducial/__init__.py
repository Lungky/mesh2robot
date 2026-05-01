"""ArUco-based world-frame registration for motion reference images."""

from mesh2robot.fiducial.board import (
    BOARD_COLS,
    BOARD_ROWS,
    ARUCO_DICT_ID,
    MARKER_LENGTH,
    MARKER_SEPARATION,
    MOUNT_ID_A,
    MOUNT_ID_B,
    board_to_world,
    create_grid_board,
    marker_center_board,
    world_to_board,
)
from mesh2robot.fiducial.pose import (
    BoardDetection,
    detect_board,
    detect_camera_pose_world,
)

__all__ = [
    "BOARD_COLS",
    "BOARD_ROWS",
    "ARUCO_DICT_ID",
    "MARKER_LENGTH",
    "MARKER_SEPARATION",
    "MOUNT_ID_A",
    "MOUNT_ID_B",
    "board_to_world",
    "create_grid_board",
    "marker_center_board",
    "world_to_board",
    "BoardDetection",
    "detect_board",
    "detect_camera_pose_world",
]
