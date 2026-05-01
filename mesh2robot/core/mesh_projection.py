"""Project a mesh through a camera: depth, visibility, per-pixel 3D lookup.

Given a mesh (in world frame) and a camera pose (world→cam), produce:
  - per-pixel 3D surface point (world frame), via ray-casting
  - per-pixel triangle index (for vertex / face attribution)

Uses trimesh.ray.ray_pyembree if available (fast), otherwise falls back to
the pure-Python trimesh ray intersector (correct but slow).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import trimesh


def _get_ray_engine(mesh: trimesh.Trimesh):
    try:
        from trimesh.ray.ray_pyembree import RayMeshIntersector
        return RayMeshIntersector(mesh)
    except (ImportError, ModuleNotFoundError):
        return mesh.ray


@dataclass
class Projection:
    hit_mask: np.ndarray       # (H, W) bool
    world_xyz: np.ndarray      # (H, W, 3) world-frame 3D point; invalid where hit_mask=False
    depth: np.ndarray          # (H, W) camera-Z at the hit; 0 where hit_mask=False
    face_idx: np.ndarray       # (H, W) int; -1 where no hit


def project_world_to_pixels(
    world_pts: np.ndarray,
    T_world_to_cam: np.ndarray,
    K: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Project (N, 3) world points through a pinhole camera.

    Returns:
      uv : (N, 2) float pixel coordinates
      z  : (N,) camera-frame depth (positive in front of camera)
    """
    pts_h = np.concatenate([world_pts, np.ones((len(world_pts), 1))], axis=1)
    cam_pts = (T_world_to_cam @ pts_h.T).T[:, :3]
    z = cam_pts[:, 2]
    uv_h = (K @ cam_pts.T).T
    uv = uv_h[:, :2] / np.where(np.abs(uv_h[:, 2:3]) > 1e-12, uv_h[:, 2:3], 1e-12)
    return uv, z


def unproject_pixel_to_ray(
    u: float, v: float,
    T_cam_to_world: np.ndarray,
    K: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (origin_world, direction_world) for the ray through pixel (u, v)."""
    K_inv = np.linalg.inv(K)
    cam_dir = K_inv @ np.array([u, v, 1.0])
    cam_dir /= np.linalg.norm(cam_dir)
    R = T_cam_to_world[:3, :3]
    t = T_cam_to_world[:3, 3]
    world_dir = R @ cam_dir
    return t.copy(), world_dir


def lift_keypoints_to_mesh(
    mesh: trimesh.Trimesh,
    keypoints_2d: np.ndarray,      # (N, 2) pixels in some image
    T_world_to_cam: np.ndarray,    # 4x4
    K: np.ndarray,                 # 3x3
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Ray-cast pixel locations onto the mesh. One ray per keypoint.

    Returns
    -------
    X_world : (N, 3)  mesh hit point in world frame (zeros where miss)
    face_idx : (N,)   triangle index (-1 where miss)
    hit_mask : (N,)   bool, True if the ray hit the mesh
    """
    N = len(keypoints_2d)
    if N == 0:
        return (np.zeros((0, 3)), np.zeros(0, np.int64), np.zeros(0, bool))

    T_cam_to_world = np.linalg.inv(T_world_to_cam)
    R = T_cam_to_world[:3, :3]
    origin = T_cam_to_world[:3, 3]
    K_inv = np.linalg.inv(K)

    pix_h = np.concatenate([keypoints_2d, np.ones((N, 1))], axis=1)
    cam_dirs = (K_inv @ pix_h.T).T
    cam_dirs /= np.linalg.norm(cam_dirs, axis=1, keepdims=True)
    world_dirs = (R @ cam_dirs.T).T
    origins = np.broadcast_to(origin, world_dirs.shape).copy()

    engine = _get_ray_engine(mesh)
    locations, index_ray, index_tri = engine.intersects_location(
        origins, world_dirs, multiple_hits=False,
    )

    X = np.zeros((N, 3))
    face = np.full(N, -1, dtype=np.int64)
    hit = np.zeros(N, dtype=bool)
    if len(index_ray) > 0:
        X[index_ray] = locations
        face[index_ray] = index_tri
        hit[index_ray] = True
    return X, face, hit


def render_mesh_depth(
    mesh: trimesh.Trimesh,
    T_world_to_cam: np.ndarray,
    K: np.ndarray,
    image_size: tuple[int, int],
    step_px: int = 1,
) -> Projection:
    """Ray-cast one ray per pixel; record first-hit surface point.

    image_size = (width, height).
    step_px: subsample pixel grid for speed. step_px=1 = every pixel.
    """
    W, H = image_size
    xs = np.arange(0, W, step_px)
    ys = np.arange(0, H, step_px)
    uu, vv = np.meshgrid(xs, ys)
    uv = np.stack([uu.flatten(), vv.flatten()], axis=1).astype(np.float64)

    # Build ray origin + direction in world frame
    T_cam_to_world = np.linalg.inv(T_world_to_cam)
    R = T_cam_to_world[:3, :3]
    origin = T_cam_to_world[:3, 3]

    # Pixel -> camera-frame direction -> world-frame direction
    K_inv = np.linalg.inv(K)
    pix_h = np.concatenate([uv, np.ones((len(uv), 1))], axis=1)      # (N, 3)
    cam_dirs = (K_inv @ pix_h.T).T                                    # (N, 3)
    cam_dirs /= np.linalg.norm(cam_dirs, axis=1, keepdims=True)
    world_dirs = (R @ cam_dirs.T).T

    origins = np.broadcast_to(origin, world_dirs.shape).copy()

    engine = _get_ray_engine(mesh)
    locations, index_ray, index_tri = engine.intersects_location(
        origins, world_dirs, multiple_hits=False,
    )

    # Full-resolution output (subsampled grid expanded back)
    sub_h = len(ys)
    sub_w = len(xs)
    hit_mask_sub = np.zeros((sub_h, sub_w), dtype=bool)
    xyz_sub = np.zeros((sub_h, sub_w, 3), dtype=np.float64)
    face_sub = -np.ones((sub_h, sub_w), dtype=np.int64)
    depth_sub = np.zeros((sub_h, sub_w), dtype=np.float64)

    if len(index_ray) > 0:
        # Un-flatten ray index → (row, col) on subsampled grid
        rr = index_ray // sub_w
        cc = index_ray % sub_w
        hit_mask_sub[rr, cc] = True
        xyz_sub[rr, cc] = locations
        face_sub[rr, cc] = index_tri
        # Camera-frame depth (z along camera axis)
        world_hits_h = np.concatenate([locations, np.ones((len(locations), 1))], axis=1)
        cam_hits = (T_world_to_cam @ world_hits_h.T).T[:, :3]
        depth_sub[rr, cc] = cam_hits[:, 2]

    if step_px == 1:
        return Projection(
            hit_mask=hit_mask_sub,
            world_xyz=xyz_sub,
            depth=depth_sub,
            face_idx=face_sub,
        )

    # Upsample to full resolution via nearest (for callers that want full-size)
    hit_mask = np.repeat(np.repeat(hit_mask_sub, step_px, axis=0), step_px, axis=1)[:H, :W]
    world_xyz = np.repeat(np.repeat(xyz_sub, step_px, axis=0), step_px, axis=1)[:H, :W, :]
    depth = np.repeat(np.repeat(depth_sub, step_px, axis=0), step_px, axis=1)[:H, :W]
    face_idx = np.repeat(np.repeat(face_sub, step_px, axis=0), step_px, axis=1)[:H, :W]
    return Projection(
        hit_mask=hit_mask, world_xyz=world_xyz, depth=depth, face_idx=face_idx,
    )
