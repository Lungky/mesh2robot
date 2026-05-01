"""Data augmentations for training-example generation.

Goal: bridge the sim-to-real gap between pristine URDF meshes and
realistic MILO scans. Each augmentation simulates a known MILO failure
mode:
  - Vertex Gaussian noise:  surface measurement noise (~0.5–5 mm).
  - Cluster hole-punching:  occluded scan regions (camera blind spots).
  - Random rigid transform: arbitrary mounting orientation in the world.
  - Random scale (±20%):    different physical robot sizes within a family.
  - Point dropout:           sparser MILO captures.

All augmentations operate on (points, labels, joint_axes_world,
joint_origins_world) tuples and return new tuples. None mutate inputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass
class AugmentConfig:
    """Per-call augmentation hyperparameters. Used by `apply()` to roll
    a random pipeline of augmentations weighted toward MILO-realistic noise."""
    vertex_noise_sigma_m: float = 0.002         # 2mm default
    hole_count: int = 4                          # number of cluster holes per example
    hole_radius_min: float = 0.02                # 2 cm
    hole_radius_max: float = 0.08                # 8 cm
    point_dropout_frac: float = 0.10             # drop 10% of points uniformly
    rigid_translation_m: float = 0.5             # random translation per axis
    rigid_rotation_deg: float = 30.0             # random rotation around random axis
    scale_min: float = 0.8                       # random scale range
    scale_max: float = 1.2


def vertex_noise(
    points: np.ndarray, sigma_m: float, rng: np.random.Generator,
) -> np.ndarray:
    """Add iid Gaussian noise to each point. Labels unchanged."""
    if sigma_m <= 0:
        return points
    return points + rng.normal(0.0, sigma_m, size=points.shape)


def hole_punch(
    points: np.ndarray,
    labels: np.ndarray,
    n_holes: int,
    radius_range: tuple[float, float],
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Drop points within `n_holes` random spherical regions, simulating
    MILO scan occlusion (camera couldn't see those surfaces).

    Hole centers are sampled from existing points. Total drop is bounded
    to ≤ 30% of the cloud to avoid degenerate empties.
    """
    if n_holes <= 0 or len(points) == 0:
        return points, labels
    keep = np.ones(len(points), dtype=bool)
    for _ in range(n_holes):
        center_idx = rng.integers(0, len(points))
        center = points[center_idx]
        r = rng.uniform(radius_range[0], radius_range[1])
        d2 = np.sum((points - center) ** 2, axis=1)
        in_hole = d2 < r * r
        if (~keep).sum() + in_hole.sum() > 0.3 * len(points):
            break
        keep &= ~in_hole
    return points[keep], labels[keep]


def random_rigid(
    points: np.ndarray,
    joint_axes: np.ndarray,
    joint_origins: np.ndarray,
    translation_m: float,
    rotation_deg: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply a random SE(3) transform to points + joint axes/origins.

    Translation is uniform in a cube of edge `2 * translation_m`. Rotation
    is around a uniformly-random axis with a uniformly-random angle in
    [0, rotation_deg]. Joint axes get the rotation only (they're directions);
    joint origins get the full SE(3).
    """
    # Random rotation axis (uniform on unit sphere)
    z = rng.uniform(-1.0, 1.0)
    phi = rng.uniform(0, 2 * np.pi)
    s = np.sqrt(max(0.0, 1.0 - z * z))
    axis = np.array([s * np.cos(phi), s * np.sin(phi), z])
    angle = np.deg2rad(rng.uniform(0.0, rotation_deg))
    R = _rodrigues(axis, angle)
    t = rng.uniform(-translation_m, translation_m, size=3)

    new_points = (points @ R.T) + t
    new_axes = joint_axes @ R.T
    new_origins = (joint_origins @ R.T) + t
    return new_points, new_axes, new_origins


def random_scale(
    points: np.ndarray,
    joint_origins: np.ndarray,
    scale_min: float,
    scale_max: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Apply uniform scale around the centroid. Joint axes are unitless
    so they don't change; joint origins do."""
    s = rng.uniform(scale_min, scale_max)
    centroid = points.mean(axis=0) if len(points) else np.zeros(3)
    new_points = (points - centroid) * s + centroid
    new_origins = (joint_origins - centroid) * s + centroid
    return new_points, new_origins, s


def point_dropout(
    points: np.ndarray, labels: np.ndarray,
    frac: float, rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Uniformly drop `frac` of the points (+ labels)."""
    if frac <= 0 or len(points) == 0:
        return points, labels
    keep_n = int(round((1.0 - frac) * len(points)))
    if keep_n >= len(points):
        return points, labels
    idx = rng.choice(len(points), size=keep_n, replace=False)
    return points[idx], labels[idx]


def apply(
    points: np.ndarray,
    labels: np.ndarray,
    joint_axes: np.ndarray,
    joint_origins: np.ndarray,
    cfg: AugmentConfig,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    """Apply a random subset of augmentations and return the perturbed example.

    Returns (points, labels, joint_axes, joint_origins, metadata).
    """
    meta: dict = {}
    # Order: rigid → scale → noise → holes → dropout. Geometry-altering
    # transforms first; topology-altering (drop) last.
    if cfg.rigid_translation_m > 0 or cfg.rigid_rotation_deg > 0:
        points, joint_axes, joint_origins = random_rigid(
            points, joint_axes, joint_origins,
            cfg.rigid_translation_m, cfg.rigid_rotation_deg, rng,
        )
        meta["rigid"] = True
    if cfg.scale_min < cfg.scale_max:
        points, joint_origins, s = random_scale(
            points, joint_origins, cfg.scale_min, cfg.scale_max, rng,
        )
        meta["scale"] = float(s)
    if cfg.vertex_noise_sigma_m > 0:
        points = vertex_noise(points, cfg.vertex_noise_sigma_m, rng)
        meta["noise_sigma_m"] = cfg.vertex_noise_sigma_m
    if cfg.hole_count > 0:
        points, labels = hole_punch(
            points, labels, cfg.hole_count,
            (cfg.hole_radius_min, cfg.hole_radius_max), rng,
        )
        meta["hole_count"] = cfg.hole_count
    if cfg.point_dropout_frac > 0:
        points, labels = point_dropout(
            points, labels, cfg.point_dropout_frac, rng,
        )
        meta["dropout_frac"] = cfg.point_dropout_frac
    return points, labels, joint_axes, joint_origins, meta


def _rodrigues(axis: np.ndarray, theta: float) -> np.ndarray:
    """Rotation matrix from axis-angle (Rodrigues' formula)."""
    a = axis / (np.linalg.norm(axis) + 1e-12)
    K = np.array([
        [    0.0, -a[2],  a[1]],
        [  a[2],   0.0, -a[0]],
        [ -a[1],  a[0],   0.0],
    ])
    s = np.sin(theta)
    c = np.cos(theta)
    return np.eye(3) + s * K + (1.0 - c) * (K @ K)
