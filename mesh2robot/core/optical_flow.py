"""Thin wrapper around OpenCV's Farnebäck dense optical flow.

Produces a per-pixel (dx, dy) flow field mapping pixels from frame 0 to frame 1.
"""

from __future__ import annotations

import cv2
import numpy as np


def compute_flow(
    image0_bgr: np.ndarray,
    image1_bgr: np.ndarray,
    pyr_scale: float = 0.5,
    levels: int = 5,
    winsize: int = 25,
    iterations: int = 5,
    poly_n: int = 7,
    poly_sigma: float = 1.5,
) -> np.ndarray:
    """Return a (H, W, 2) float32 flow field.

    flow[y, x] = (dx, dy) such that the pixel at (x, y) in image0 moved to
    approximately (x + dx, y + dy) in image1.

    Defaults tuned for moderately-textured industrial arms captured handheld.
    """
    if image0_bgr.ndim == 3:
        g0 = cv2.cvtColor(image0_bgr, cv2.COLOR_BGR2GRAY)
        g1 = cv2.cvtColor(image1_bgr, cv2.COLOR_BGR2GRAY)
    else:
        g0, g1 = image0_bgr, image1_bgr
    return cv2.calcOpticalFlowFarneback(
        g0, g1, None,
        pyr_scale, levels, winsize, iterations, poly_n, poly_sigma, 0,
    )


def flow_magnitude(flow: np.ndarray) -> np.ndarray:
    """(H, W) per-pixel flow vector length."""
    return np.linalg.norm(flow, axis=-1)


def flow_to_color(flow: np.ndarray, max_magnitude: float | None = None) -> np.ndarray:
    """HSV visualization of the flow field (BGR uint8, H x W x 3)."""
    h, w = flow.shape[:2]
    hsv = np.zeros((h, w, 3), dtype=np.uint8)
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1], angleInDegrees=True)
    if max_magnitude is None:
        max_magnitude = float(mag.max()) + 1e-6
    hsv[..., 0] = (ang / 2).astype(np.uint8)      # hue: 0..179
    hsv[..., 1] = 255
    hsv[..., 2] = np.clip(mag / max_magnitude * 255.0, 0, 255).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
