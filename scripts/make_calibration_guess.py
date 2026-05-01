"""Generate a rough-guess calibration.json from one of your motion images.

For a proper calibration you should use a checkerboard + OpenCV
`calibrateCamera`. But as a quick starting point — good enough for the
feasibility test — this script assumes:
  fx = fy ≈ image_width        (~53° horizontal FOV, typical phone)
  (cx, cy) = (W/2, H/2)        (principal point at image center)
  zero lens distortion

Usage:
    python scripts/make_calibration_guess.py \\
        --image mesh2robot/input/motion/joint_1/state0.png \\
        --output mesh2robot/input/calibration.json

If the first test fails with high reprojection error, do a proper calibration
and replace this file.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fov-deg", type=float, default=60.0,
                        help="Assumed horizontal field-of-view in degrees")
    args = parser.parse_args()

    img = cv2.imread(str(args.image))
    if img is None:
        raise FileNotFoundError(args.image)
    H, W = img.shape[:2]

    import math
    fx = W / (2.0 * math.tan(math.radians(args.fov_deg / 2.0)))
    fy = fx                   # assume square pixels

    data = {
        "fx": float(fx),
        "fy": float(fy),
        "cx": float(W / 2.0),
        "cy": float(H / 2.0),
        "width": int(W),
        "height": int(H),
        "dist_coeffs": [0.0, 0.0, 0.0, 0.0, 0.0],
        "_note": (
            f"Rough guess from {args.image.name} at "
            f"{args.fov_deg:.0f}° horizontal FOV. Replace with checkerboard "
            "calibration for production."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, indent=2))
    print(f"Wrote {args.output}")
    print(f"  image size: {W} x {H}")
    print(f"  fx = fy = {fx:.1f} px  (at {args.fov_deg}° FOV)")


if __name__ == "__main__":
    main()
