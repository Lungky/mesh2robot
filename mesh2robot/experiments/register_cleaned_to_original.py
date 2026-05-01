"""ICP-register a hand-cleaned mesh back to MILO's original output frame.

Blender's PLY↔OBJ roundtrip commonly applies axis-swaps (Y-up ↔ Z-up) and can
offset/scale the mesh. The cleaned mesh is geometrically a SUBSET of the
original output — so ICP from cleaned → original yields the transform that
puts the cleaned mesh back into MILO's camera frame.

Strategy:
  1. Sample dense points from both meshes.
  2. Crop the original to a ROI so ICP isn't overwhelmed by the studio scene.
     The ROI is chosen as a sphere around the origin large enough to contain
     the robot — adjust via --roi-radius.
  3. Try a set of canonical initial rotations (axis permutations + flips) so
     ICP can escape poor initializations if Blender flipped axes.
  4. For each init, run point-to-point ICP; pick the one with lowest error.
  5. Save the final T_cleaned_to_original as .npy.
"""

from __future__ import annotations

import argparse
from itertools import permutations
from pathlib import Path

import numpy as np
import trimesh


def _sample_points(mesh: trimesh.Trimesh, n: int = 20000) -> np.ndarray:
    pts, _ = trimesh.sample.sample_surface(mesh, min(n, len(mesh.faces)))
    return np.asarray(pts)


def _crop_sphere(pts: np.ndarray, center: np.ndarray, radius: float) -> np.ndarray:
    mask = np.linalg.norm(pts - center, axis=1) < radius
    return pts[mask]


def _crop_above_z(pts: np.ndarray, z_min: float) -> np.ndarray:
    """Keep only points above a given z cutoff.

    Useful for excluding the ArUco board / floor plane so ICP can't collapse
    the robot mesh onto it.
    """
    return pts[pts[:, 2] > z_min]


def _canonical_rotations() -> list[np.ndarray]:
    """Enumerate 24 axis-aligned rotations (all orientations of a cube).

    Good enough to find the Blender axis-swap, if any, as a starting point.
    """
    rots = []
    for perm in permutations([0, 1, 2]):
        for signs in range(8):
            s = np.array([(-1) ** ((signs >> i) & 1) for i in range(3)])
            R = np.zeros((3, 3))
            for i, p in enumerate(perm):
                R[i, p] = s[i]
            if np.linalg.det(R) > 0:      # keep only right-handed
                rots.append(R)
    # De-dup (some perms+signs yield the same R)
    seen = []
    unique = []
    for R in rots:
        key = tuple(R.flatten().round(6))
        if key in seen:
            continue
        seen.append(key)
        unique.append(R)
    return unique


def _icp(
    src: np.ndarray,
    tgt: np.ndarray,
    init_T: np.ndarray,
    max_iter: int = 50,
    tol: float = 1e-6,
) -> tuple[np.ndarray, float]:
    """Simple point-to-point ICP. Returns (T, rms_error)."""
    from scipy.spatial import cKDTree
    tree = cKDTree(tgt)
    T = init_T.copy()
    prev = np.inf
    for _ in range(max_iter):
        src_t = (T[:3, :3] @ src.T).T + T[:3, 3]
        d, idx = tree.query(src_t, k=1)
        matched = tgt[idx]

        # Filter outliers (10× median distance)
        keep = d < 10.0 * np.median(d)
        if keep.sum() < 10:
            break
        A = src_t[keep]
        B = matched[keep]

        # Horn alignment for the incremental update
        ca = A.mean(0)
        cb = B.mean(0)
        H = (A - ca).T @ (B - cb)
        U, _, Vt = np.linalg.svd(H)
        dsign = np.sign(np.linalg.det(Vt.T @ U.T))
        M = np.diag([1.0, 1.0, dsign])
        R_inc = Vt.T @ M @ U.T
        t_inc = cb - R_inc @ ca

        T_inc = np.eye(4)
        T_inc[:3, :3] = R_inc
        T_inc[:3, 3] = t_inc
        T = T_inc @ T

        rms = float(np.sqrt((d[keep] ** 2).mean()))
        if abs(prev - rms) < tol:
            break
        prev = rms
    return T, prev


def run(
    cleaned_path: Path,
    original_path: Path,
    output_path: Path,
    roi_center: tuple[float, float, float] = (0.0, 0.0, 0.0),
    roi_radius: float = 1.5,
    roi_z_min: float | None = None,
    n_src: int = 20000,
    n_tgt: int = 100000,
) -> None:
    print(f"Loading cleaned mesh: {cleaned_path}")
    cleaned = trimesh.load(str(cleaned_path), force="mesh")
    if isinstance(cleaned, trimesh.Scene):
        cleaned = cleaned.dump(concatenate=True)
    print(f"  {len(cleaned.vertices)} verts   AABB={np.round(cleaned.bounds, 3).tolist()}")

    print(f"Loading original mesh: {original_path}")
    original = trimesh.load(str(original_path), force="mesh")
    if isinstance(original, trimesh.Scene):
        original = original.dump(concatenate=True)
    print(f"  {len(original.vertices)} verts   AABB={np.round(original.bounds, 3).tolist()}")

    src = _sample_points(cleaned, n_src)
    tgt_full = _sample_points(original, n_tgt)
    print(f"Sampled {len(src)} src, {len(tgt_full)} tgt points")

    tgt = _crop_sphere(tgt_full, np.asarray(roi_center), roi_radius)
    print(f"Cropped target to {len(tgt)} points within {roi_radius} m of {roi_center}")
    if roi_z_min is not None:
        before = len(tgt)
        tgt = _crop_above_z(tgt, roi_z_min)
        print(f"Further cropped target to {len(tgt)} points with z > {roi_z_min} "
              f"(from {before})")

    # Initial centroid alignment (apply to each candidate rotation)
    src_centroid = src.mean(0)
    tgt_centroid = tgt.mean(0)

    rotations = _canonical_rotations()
    print(f"Trying {len(rotations)} canonical initial rotations ...")

    best_T = np.eye(4)
    best_rms = np.inf
    for i, R0 in enumerate(rotations):
        T0 = np.eye(4)
        T0[:3, :3] = R0
        T0[:3, 3] = tgt_centroid - R0 @ src_centroid
        T, rms = _icp(src, tgt, T0, max_iter=40)
        if rms < best_rms:
            best_rms = rms
            best_T = T
            print(f"  init {i:2d}: rms={rms:.4f} m   (new best)")

    print(f"\nBest RMS after ICP: {best_rms*1000:.2f} mm")
    print(f"T_cleaned_to_original:\n{np.round(best_T, 4)}")

    # Diagnostic: how far did the centroid move?
    src_aligned = (best_T[:3, :3] @ src.T).T + best_T[:3, 3]
    print(f"Aligned centroid: {np.round(src_aligned.mean(0), 3)}")
    print(f"Target centroid: {np.round(tgt.mean(0), 3)}")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, best_T)
    print(f"\nSaved T_cleaned_to_original to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cleaned", type=Path, required=True)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--roi-center", type=float, nargs=3, default=[0.0, 0.0, 0.0])
    parser.add_argument("--roi-radius", type=float, default=1.5,
                        help="Keep only target points within this radius from center")
    parser.add_argument("--roi-z-min", type=float, default=None,
                        help="Optional: additionally keep only target points with z > z_min "
                             "(use to exclude the board/floor plane)")
    parser.add_argument("--n-src", type=int, default=20000)
    parser.add_argument("--n-tgt", type=int, default=100000)
    args = parser.parse_args()
    run(
        cleaned_path=args.cleaned,
        original_path=args.original,
        output_path=args.output,
        roi_center=tuple(args.roi_center),
        roi_radius=args.roi_radius,
        roi_z_min=args.roi_z_min,
        n_src=args.n_src,
        n_tgt=args.n_tgt,
    )


if __name__ == "__main__":
    main()
