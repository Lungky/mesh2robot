"""Board constants and coordinate transforms for ArUco grid board calibration.

Supplied by the user; this module is the source of truth for the physical
board's geometry and world-frame origin definition.
"""

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Board geometry
# ---------------------------------------------------------------------------
BOARD_COLS = 12
BOARD_ROWS = 12
ARUCO_DICT_ID = cv2.aruco.DICT_6X6_250
MARKER_LENGTH = 0.08        # meters
MARKER_SEPARATION = 0.01    # meters
CELL_PITCH = MARKER_LENGTH + MARKER_SEPARATION  # 0.09m
TEXTURE_EXTENT = BOARD_COLS * MARKER_LENGTH + (BOARD_COLS - 1) * MARKER_SEPARATION  # 1.07m
TEX_SIZE = 107 * 23          # 2461 pixels -- all marker corners on exact integer pixels

# Physical rows actually present on the printed poster
PHYS_ROW_START = 1
PHYS_ROW_COUNT = 6

# Mount markers (used to define world-frame origin)
MOUNT_ID_A = 17  # row 1, col 5
MOUNT_ID_B = 18  # row 1, col 6

# ---------------------------------------------------------------------------
# Board-to-world rotation matrix
# ---------------------------------------------------------------------------
#
# OpenCV Board Coords (what solvePnP uses):
#   +X = column direction (marker 0 -> 1 -> 2 -> ...)
#   +Y = row direction (marker 0 -> 12 -> 24 -> ...)
#   +Z = INTO the board (down when face-up on table)
#
# World Coords (= xArm base frame, right-handed Z-up):
#   +X = arm forward (board +Y direction, rows increasing)
#   +Y = operator left (board +X direction, cols increasing, 17->18)
#   +Z = up (= board -Z)
#
# det = +1.  X x Y = [0,1,0] x [1,0,0] = [0,0,-1] = Z.
BOARD_TO_WORLD = np.array([
    [0, 1, 0],
    [1, 0, 0],
    [0, 0, -1],
], dtype=np.float64)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def marker_center_board(marker_id: int) -> np.ndarray:
    """Return the center of *marker_id* in board coordinates ``[x, y, 0]``."""
    col = marker_id % BOARD_COLS
    row = marker_id // BOARD_COLS
    x = col * CELL_PITCH + MARKER_LENGTH / 2
    y = row * CELL_PITCH + MARKER_LENGTH / 2
    return np.array([x, y, 0.0])


MOUNT_CENTER_BOARD = (marker_center_board(MOUNT_ID_A) + marker_center_board(MOUNT_ID_B)) / 2


def board_to_world(board_point) -> np.ndarray:
    """Convert board coordinates to world coordinates (origin at mount center)."""
    return BOARD_TO_WORLD @ (np.asarray(board_point, dtype=np.float64) - MOUNT_CENTER_BOARD)


def world_to_board(world_point) -> np.ndarray:
    """Convert world coordinates to board coordinates."""
    return BOARD_TO_WORLD.T @ np.asarray(world_point, dtype=np.float64) + MOUNT_CENTER_BOARD


def create_grid_board() -> cv2.aruco.GridBoard:
    """Create the ArUco GridBoard object matching the physical poster."""
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT_ID)
    return cv2.aruco.GridBoard(
        (BOARD_COLS, BOARD_ROWS),
        MARKER_LENGTH,
        MARKER_SEPARATION,
        dictionary,
    )
