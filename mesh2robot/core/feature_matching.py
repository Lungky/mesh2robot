"""ORB feature detection + matching across image pairs.

Replaces dense optical flow for motion estimation. Features survive large
motions, in-place rotations, and most self-occlusion — the failure modes
that break Farnebäck.

Output: two arrays of matched pixel coordinates (p0, p1), such that the
world point seen at pixel p0[i] in state 0 appears at pixel p1[i] in state 1.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class MatchResult:
    p0: np.ndarray       # (N, 2) float32 pixel coords in state 0
    p1: np.ndarray       # (N, 2) float32 pixel coords in state 1
    distances: np.ndarray  # (N,) float32 descriptor distances (smaller=better)


def detect_and_match(
    image0_bgr: np.ndarray,
    image1_bgr: np.ndarray,
    n_features: int = 5000,
    ratio: float = 0.8,
    use_sift: bool = False,
    mask0: np.ndarray | None = None,
    mask1: np.ndarray | None = None,
) -> MatchResult:
    """Detect ORB (or SIFT) keypoints and match via BF + Lowe's ratio test.

    Parameters
    ----------
    n_features : maximum features per image
    ratio : Lowe's ratio threshold (lower = stricter, 0.7 typical for SIFT, 0.8 for ORB)
    use_sift : if True, SIFT instead of ORB. Better for textureless / plain surfaces
               but slower.
    mask0, mask1 : optional uint8 masks (same size as images) restricting where
                   keypoints may be detected. Use mask0 = robot silhouette in
                   state-0 so features concentrate on the robot instead of the
                   textured background (ArUco board, truss, etc.).
    """
    g0 = cv2.cvtColor(image0_bgr, cv2.COLOR_BGR2GRAY) if image0_bgr.ndim == 3 else image0_bgr
    g1 = cv2.cvtColor(image1_bgr, cv2.COLOR_BGR2GRAY) if image1_bgr.ndim == 3 else image1_bgr

    if use_sift:
        detector = cv2.SIFT_create(nfeatures=n_features)
        norm = cv2.NORM_L2
    else:
        detector = cv2.ORB_create(nfeatures=n_features)
        norm = cv2.NORM_HAMMING

    kp0, des0 = detector.detectAndCompute(g0, mask0)
    kp1, des1 = detector.detectAndCompute(g1, mask1)
    if des0 is None or des1 is None or len(kp0) < 2 or len(kp1) < 2:
        return MatchResult(
            p0=np.empty((0, 2), np.float32),
            p1=np.empty((0, 2), np.float32),
            distances=np.empty(0, np.float32),
        )

    bf = cv2.BFMatcher(norm, crossCheck=False)
    raw = bf.knnMatch(des0, des1, k=2)

    good = []
    for pair in raw:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < ratio * n.distance:
            good.append(m)

    if not good:
        return MatchResult(
            p0=np.empty((0, 2), np.float32),
            p1=np.empty((0, 2), np.float32),
            distances=np.empty(0, np.float32),
        )

    p0 = np.float32([kp0[m.queryIdx].pt for m in good])
    p1 = np.float32([kp1[m.trainIdx].pt for m in good])
    dist = np.float32([m.distance for m in good])
    return MatchResult(p0=p0, p1=p1, distances=dist)


def draw_matches(
    image0_bgr: np.ndarray,
    image1_bgr: np.ndarray,
    match: MatchResult,
    max_draw: int = 200,
) -> np.ndarray:
    """Side-by-side visualization of matched keypoints (yellow lines)."""
    h0, w0 = image0_bgr.shape[:2]
    h1, w1 = image1_bgr.shape[:2]
    h = max(h0, h1)
    canvas = np.zeros((h, w0 + w1, 3), dtype=np.uint8)
    canvas[:h0, :w0] = image0_bgr
    canvas[:h1, w0:w0 + w1] = image1_bgr

    n = min(len(match.p0), max_draw)
    if n == 0:
        return canvas
    # Sort by descriptor distance, show the best matches
    order = np.argsort(match.distances)[:n]
    for i in order:
        x0, y0 = match.p0[i]
        x1, y1 = match.p1[i]
        p0 = (int(x0), int(y0))
        p1 = (int(x1) + w0, int(y1))
        cv2.circle(canvas, p0, 3, (0, 255, 255), -1)
        cv2.circle(canvas, p1, 3, (0, 255, 255), -1)
        cv2.line(canvas, p0, p1, (0, 255, 0), 1, cv2.LINE_AA)
    return canvas
