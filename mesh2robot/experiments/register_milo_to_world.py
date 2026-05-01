"""Register the MILO world frame against the ArUco board world frame.

Reads:
  - cameras.json from MILO  (per-image poses in MILO's world frame)
  - a directory of MILO input images (the ones the 3DGS was trained on)
  - optional intrinsics override; otherwise uses fx/fy from cameras.json with
    principal point at image center.

For each image where ArUco detection succeeds, computes an estimate:
    T_milo_to_world = T_cam_to_world_aruco @ T_milo_to_cam_milo

Aggregates via geodesic-mean rotation and median translation for robustness.
Writes the final 4x4 transform to `--output` (a .npy file) and prints a
summary including per-image reprojection errors.

The output file can be consumed by urdf_from_images.py via --mesh-to-world.

Usage:
    python -m mesh2robot.experiments.register_milo_to_world \\
        --cameras       path/to/milo/cameras.json \\
        --images-dir    path/to/milo/input/images \\
        --output        path/to/T_milo_to_world.npy \\
        [--max-images 50]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R

from mesh2robot.fiducial.pose import detect_board
from mesh2robot.io.milo_output import load_milo_cameras


def _compose_T_milo_to_world(
    T_milo_to_cam: np.ndarray,
    T_world_to_cam_aruco: np.ndarray,
) -> np.ndarray:
    """T_milo_to_world such that world_point = T_milo_to_world @ milo_point.

    Derivation:
      T_world_to_cam @ p_world = p_cam_aruco
      T_milo_to_cam @ p_milo   = p_cam_milo
    MILO and ArUco share the physical scene viewed by the same camera, so
      p_cam_aruco == p_cam_milo
    →  p_world = inv(T_world_to_cam) @ T_milo_to_cam @ p_milo
    →  T_milo_to_world = inv(T_world_to_cam) @ T_milo_to_cam
    """
    T_cam_to_world = np.linalg.inv(T_world_to_cam_aruco)
    return T_cam_to_world @ T_milo_to_cam


def _average_SE3(transforms: list[np.ndarray]) -> np.ndarray:
    """Geodesic-mean rotation + median translation across a list of 4x4 SE(3)."""
    if not transforms:
        raise ValueError("no transforms to average")
    rots = R.from_matrix(np.stack([T[:3, :3] for T in transforms]))
    # scipy's Rotation.mean() uses quaternion average — good enough for small
    # spread and much faster than iterative Karcher.
    R_mean = rots.mean().as_matrix()
    t_med = np.median(np.stack([T[:3, 3] for T in transforms]), axis=0)
    T_avg = np.eye(4)
    T_avg[:3, :3] = R_mean
    T_avg[:3, 3] = t_med
    return T_avg


def _rotational_deviation_deg(transforms: list[np.ndarray], mean_T: np.ndarray) -> list[float]:
    mean_R = mean_T[:3, :3]
    devs = []
    for T in transforms:
        dR = T[:3, :3] @ mean_R.T
        cos = np.clip((np.trace(dR) - 1) / 2, -1, 1)
        devs.append(float(np.rad2deg(np.arccos(cos))))
    return devs


def run(
    cameras_json: Path,
    images_dir: Path,
    output_path: Path,
    max_images: int | None = None,
    min_markers: int = 8,
) -> None:
    print(f"Loading MILO cameras from {cameras_json} ...")
    milo_cameras = load_milo_cameras(cameras_json, schema="3dgs")
    print(f"  {len(milo_cameras)} cameras")

    images_dir = Path(images_dir)
    candidates = sorted(images_dir.glob("*.jpg")) + sorted(images_dir.glob("*.png"))
    print(f"  {len(candidates)} candidate images in {images_dir}")
    if max_images is not None:
        # Evenly sample across the list so we cover diverse viewing angles
        step = max(1, len(candidates) // max_images)
        candidates = candidates[::step][:max_images]
        print(f"  sampling {len(candidates)} of them evenly")

    estimates: list[tuple[str, np.ndarray, float, int]] = []   # (name, T, reproj_px, n_markers)
    failures: list[tuple[str, str]] = []

    for path in candidates:
        img_name = path.stem
        if img_name not in milo_cameras:
            failures.append((path.name, "not in cameras.json"))
            continue
        mc = milo_cameras[img_name]

        img = cv2.imread(str(path))
        if img is None:
            failures.append((path.name, "unreadable"))
            continue

        det = detect_board(img, mc.K, mc.dist_coeffs, min_markers=min_markers)
        if det is None:
            failures.append((path.name, "aruco not detected"))
            continue

        T_milo_to_world = _compose_T_milo_to_world(mc.T_world_to_cam, det.T_world_to_cam)
        estimates.append((img_name, T_milo_to_world, det.reprojection_error_px, det.n_markers))

    if not estimates:
        print("\nNo successful ArUco detections. Failures:")
        for name, why in failures[:10]:
            print(f"  {name}: {why}")
        raise SystemExit(1)

    Ts = [e[1] for e in estimates]
    T_avg = _average_SE3(Ts)
    devs_deg = _rotational_deviation_deg(Ts, T_avg)
    t_dists = [np.linalg.norm(T[:3, 3] - T_avg[:3, 3]) * 1000 for T in Ts]

    print(f"\nSuccessful detections: {len(estimates)} / {len(candidates)}")
    print(f"Rotational spread (deg):  median={np.median(devs_deg):.3f}  "
          f"90%={np.percentile(devs_deg, 90):.3f}  max={max(devs_deg):.3f}")
    print(f"Translational spread (mm): median={np.median(t_dists):.3f}  "
          f"90%={np.percentile(t_dists, 90):.3f}  max={max(t_dists):.3f}")
    reprs = [e[2] for e in estimates]
    print(f"Detection reprojection error (px): median={np.median(reprs):.3f}  "
          f"max={max(reprs):.3f}")

    print("\nFinal averaged T_milo_to_world:")
    print(np.round(T_avg, 4))

    # Sanity decomposition: what does the rotation look like?
    R_avg = T_avg[:3, :3]
    t_avg = T_avg[:3, 3]
    axis_angle = R.from_matrix(R_avg).as_rotvec()
    angle_deg = float(np.rad2deg(np.linalg.norm(axis_angle)))
    axis = axis_angle / (np.linalg.norm(axis_angle) + 1e-12)
    print(f"\nRotation axis: {np.round(axis, 3)}   angle: {angle_deg:.2f} deg")
    print(f"Translation (m): {np.round(t_avg, 4)}")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, T_avg)
    print(f"\nSaved T_milo_to_world to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cameras", type=Path, required=True)
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-images", type=int, default=50,
                        help="Sample this many images uniformly from the set")
    parser.add_argument("--min-markers", type=int, default=8)
    args = parser.parse_args()
    run(
        cameras_json=args.cameras,
        images_dir=args.images_dir,
        output_path=args.output,
        max_images=args.max_images,
        min_markers=args.min_markers,
    )


if __name__ == "__main__":
    main()
