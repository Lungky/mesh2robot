"""Proper camera calibration using the existing ArUco grid board as target.

Takes a folder of photos of the board (varied angles / distances / positions
in the frame) and produces a calibration.json with fx, fy, cx, cy, and lens
distortion coefficients.

No extra checkerboard required — reuses the same board defined in
mesh2robot.fiducial.board.

Usage:
    python scripts/calibrate_camera_aruco.py \\
        --images-dir path/to/calibration/shots \\
        --output     input/test_1/calibration.json

Capture tips are in the docstring below.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from mesh2robot.fiducial.board import ARUCO_DICT_ID, create_grid_board


CAPTURE_TIPS = """
Capture protocol for best calibration
-------------------------------------
Take 15-25 photos of the ArUco grid board with the SAME PHONE / SAME SETTINGS
you'll use for robot-state captures. Before shooting:
  - Lock focus (tap once, then long-press to lock on iPhone / Android).
  - Disable zoom; use 1x native.
  - Consistent orientation (all landscape or all portrait — match your
    robot-state captures).

For each photo, vary:
  - Distance: half should be close (~40 cm), half farther (~1.5 m).
  - Board position in frame: include some where the board is at the top-left,
    top-right, bottom-left, bottom-right, and center.
  - Tilt: straight-on plus ~30° tilts left / right / up / down. DO NOT include
    ultra-oblique shots (>60°); they destabilize distortion fit.
  - Rotation in image plane: include ~15°, ~30° in-plane rotations.

All shots must show at least 10 markers clearly. Blurry shots → discard.
"""


def main() -> None:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
        epilog=CAPTURE_TIPS,
    )
    parser.add_argument("--images-dir", type=Path, required=True,
                        help="Folder with photos of the ArUco board")
    parser.add_argument("--output", type=Path, required=True,
                        help="Path to write calibration.json")
    parser.add_argument("--min-markers", type=int, default=10,
                        help="Skip photos with fewer detected markers")
    parser.add_argument("--save-diagnostics", type=Path, default=None,
                        help="Optional folder to write detection overlays")
    args = parser.parse_args()

    board = create_grid_board()
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT_ID)
    params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(dictionary, params)

    image_paths = []
    for ext in (".jpg", ".jpeg", ".png", ".JPG", ".PNG"):
        image_paths.extend(args.images_dir.glob(f"*{ext}"))
    image_paths = sorted(image_paths)
    if not image_paths:
        raise SystemExit(f"No images found in {args.images_dir}")
    print(f"Found {len(image_paths)} images in {args.images_dir}")

    all_corners: list[np.ndarray] = []
    all_ids: list[np.ndarray] = []
    counter: list[int] = []
    image_size: tuple[int, int] | None = None
    accepted = 0
    rejected = 0

    if args.save_diagnostics is not None:
        args.save_diagnostics.mkdir(parents=True, exist_ok=True)

    for p in image_paths:
        img = cv2.imread(str(p))
        if img is None:
            print(f"  {p.name}: unreadable")
            rejected += 1
            continue

        if image_size is None:
            image_size = (img.shape[1], img.shape[0])
        elif (img.shape[1], img.shape[0]) != image_size:
            print(f"  {p.name}: size {img.shape[1]}x{img.shape[0]} differs from "
                  f"first image {image_size} — skipped.")
            rejected += 1
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = detector.detectMarkers(gray)
        if ids is None or len(ids) < args.min_markers:
            n = 0 if ids is None else len(ids)
            print(f"  {p.name}: {n} markers (< {args.min_markers}) — skipped.")
            rejected += 1
            continue

        all_corners.extend(corners)
        all_ids.extend(ids.flatten().tolist())
        counter.append(len(ids))
        accepted += 1

        if args.save_diagnostics is not None:
            vis = img.copy()
            cv2.aruco.drawDetectedMarkers(vis, corners, ids)
            cv2.imwrite(str(args.save_diagnostics / p.name), vis)

        print(f"  {p.name}: {len(ids)} markers")

    print(f"\nAccepted {accepted} / {len(image_paths)} images ({rejected} rejected)")
    if accepted < 5:
        raise SystemExit("Need at least 5 accepted images for a stable calibration.")

    all_ids_arr = np.array(all_ids).reshape(-1, 1)
    counter_arr = np.array(counter, dtype=np.int32)

    print("\nRunning cv2.aruco.calibrateCameraAruco ...")
    ret, K, dist, _rvecs, _tvecs = cv2.aruco.calibrateCameraAruco(
        all_corners, all_ids_arr, counter_arr, board,
        image_size, None, None,
    )

    print(f"\nCalibration converged.")
    print(f"  RMS reprojection error: {ret:.3f} px")
    print(f"  K =\n{K}")
    print(f"  dist = {dist.flatten()}")

    data = {
        "fx": float(K[0, 0]),
        "fy": float(K[1, 1]),
        "cx": float(K[0, 2]),
        "cy": float(K[1, 2]),
        "width": int(image_size[0]),
        "height": int(image_size[1]),
        "dist_coeffs": [float(x) for x in dist.flatten()],
        "_source": "checkerboard_aruco",
        "_n_images": accepted,
        "_reprojection_error_px": float(ret),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, indent=2))
    print(f"\nWrote {args.output}")
    if ret > 1.0:
        print(f"\n[warn] RMS error {ret:.2f} px is high. Common fixes:")
        print("  - Add more images at underrepresented angles.")
        print("  - Remove blurry or low-marker-count shots.")
        print("  - Ensure focus is locked during all captures.")


if __name__ == "__main__":
    main()
