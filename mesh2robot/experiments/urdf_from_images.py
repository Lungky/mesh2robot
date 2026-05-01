"""End-to-end Path B driver: MILO mesh + per-joint state images → URDF.

Works for any N ≥ 1 joint. The user provides one joint_*/ folder per DOF;
each folder holds K_i ≥ 2 state images showing the robot before/after that
one joint rotates.

Layout:
    input/.../motion/joint_1/state*.png
    input/.../motion/joint_2/state*.png
    ...

Per joint folder:
  1. Extract motion (axis, origin, moved vertex mask) from ORB + multi-body
     RANSAC + PnP on adjacent state-image pairs.

Globally:
  2. Compute each joint's cut_point = origin + biggest_gap_along_motion * axis.
  3. Order joints by cut_point projected onto the base axis (low → high).
  4. For each joint in order:
       link_dir = RANSAC_pivot[i] − RANSAC_pivot[i−1]
       classify ROLL vs PITCH from |motion · link_dir|
       ROLL  : axis = cut_normal = snap(motion) to canonical ‖/⊥-base
       PITCH : cut_normal = snap(link_dir), axis = snap(motion, ⊥cut_normal)
       moving_side_sign = sign(link_dir · cut_normal)
       partition remaining mesh by (cut_point, cut_normal)
  5. Build URDF: link_base → joint_1 → link_1 → ... → joint_N → link_tip.

Result: a properly chained (N+1)-link, N-joint URDF with canonical joint
axes and principled cut planes.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, replace
from pathlib import Path

import cv2
import numpy as np
import trimesh
from yourdfpy import URDF

from mesh2robot.core.joint_extraction import JointEstimate
from mesh2robot.core.motion_from_images import (
    JointMultiStateResult,
    extract_joint_motion_multi,
)
from mesh2robot.core.physics import compute_link_inertials
from mesh2robot.core.robot_retrieval import (
    MatchResult,
    match_robot,
    summarize as summarize_match,
)
from mesh2robot.core.template_match import match as match_template
from mesh2robot.core.urdf_assembly import AssemblyInput, assemble


@dataclass
class JointInfo:
    name: str
    result: JointMultiStateResult
    axis: np.ndarray         # rotation axis (used in URDF)
    origin: np.ndarray       # a point on the rotation axis
    cut_offset: float        # offset along axis from origin (legacy, retained)
    cut_normal: np.ndarray = None  # cut plane normal (may differ from axis for pitch joints)
    cut_point: np.ndarray = None   # a 3D point on the cut plane


def _find_joint_dirs(motion_dir: Path) -> list[Path]:
    """All subfolders matching `joint_*`, sorted by name."""
    dirs = [p for p in sorted(motion_dir.iterdir())
            if p.is_dir() and p.name.startswith("joint_")]
    return dirs


def _find_state_images(states_dir: Path) -> list[Path]:
    """state*.png/.jpg/.jpeg sorted by numeric suffix."""
    paths = []
    for ext in (".png", ".jpg", ".jpeg"):
        paths.extend(states_dir.glob(f"state*{ext}"))
    def _key(p: Path) -> int:
        digits = "".join(c for c in p.stem if c.isdigit())
        return int(digits) if digits else 0
    paths.sort(key=_key)
    return paths


def _load_calibration(path: Path) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    """Read fx/fy/cx/cy + dist_coeffs from a calibration.json."""
    data = json.loads(Path(path).read_text())
    K = np.array([
        [float(data["fx"]), 0.0, float(data["cx"])],
        [0.0, float(data["fy"]), float(data["cy"])],
        [0.0, 0.0, 1.0],
    ])
    dist = np.asarray(data.get("dist_coeffs", [0.0] * 5), dtype=np.float64)
    size = (int(data["width"]), int(data["height"]))
    return K, dist, size


def _biggest_gap_cut(seed_proj: np.ndarray) -> float:
    sorted_sp = np.sort(seed_proj)
    if len(sorted_sp) < 2:
        return float(sorted_sp[0]) if len(sorted_sp) else 0.0
    gaps = np.diff(sorted_sp)
    k = int(np.argmax(gaps))
    return (sorted_sp[k] + sorted_sp[k + 1]) / 2.0


def _snap_to_canonical(
    axis: np.ndarray,
    reference: np.ndarray,
    perpendicular_only: bool = False,
) -> tuple[np.ndarray, float, int]:
    """Snap `axis` to the nearest canonical direction relative to `reference`.

    Canonicals = {parallel to reference} ∪ {6 perpendicular directions}
    sampled at 30° increments around `reference`.

    Returns (snapped_axis, snap_angle_deg, canonical_index).
    """
    axis_u = axis / (np.linalg.norm(axis) + 1e-12)
    canonicals = _canonical_axes_relative_to_base(reference)
    if perpendicular_only:
        canonicals = canonicals[1:]
        start_idx = 1
    else:
        start_idx = 0
    best_c = None
    best_ang = float("inf")
    best_i = -1
    for i, c in enumerate(canonicals):
        cos_a = abs(float(c @ axis_u))
        ang = np.rad2deg(np.arccos(np.clip(cos_a, -1.0, 1.0)))
        if ang < best_ang:
            best_ang = ang
            best_c = c
            best_i = start_idx + i
    if best_c is None:
        return axis_u, 0.0, 0
    if float(best_c @ axis_u) < 0:
        best_c = -best_c
    return best_c, best_ang, best_i


def _canonical_label(i: int) -> str:
    if i == 0:
        return "‖"
    return f"⊥{(i - 1) * 30}°"


def _fit_local_primitive(
    verts: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Fit an oriented bounding box (via PCA) to a vertex cloud.

    Returns (center, primary_direction, extents). primary_direction is the
    first PCA eigenvector (long axis of the cloud); extents[0] ≥ extents[1]
    ≥ extents[2] are half-widths along each PCA axis. For an elongated link
    (like a horizontal forearm), `primary_direction` points along its long
    axis — which is the natural cut-normal direction for separating that
    link from its parent.
    """
    if len(verts) < 10:
        return None
    center = verts.mean(axis=0)
    centered = verts - center
    cov = centered.T @ centered / max(len(centered) - 1, 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(-eigvals)
    eigvecs = eigvecs[:, order]
    primary = eigvecs[:, 0]
    # Project onto the PCA axes to get true half-widths (range/2 rather than
    # sigma, which gives a tight OBB).
    proj = centered @ eigvecs
    extents = (proj.max(axis=0) - proj.min(axis=0)) / 2.0
    return center, primary, extents


def _find_first_housing_peak_z(
    verts: np.ndarray,
    base_axis: np.ndarray,
    remaining_vert: np.ndarray,
    search_range_above: float = 0.45,
    z_step: float = 0.015,
    slab_thick: float = 0.04,
    min_slab_verts: int = 30,
    min_prominence_cm: float = 0.5,
) -> float | None:
    """Find the first lateral-extent peak in the remaining region along
    base_axis. On industrial arms, joints live inside **wider housings**
    (motor/bearing mounts), with narrower link tubes between them — so
    joint Z positions show up as *lateral-extent maxima*, not minima.

    Starts at the bottom of the remaining region and walks upward until the
    first local-maximum of mean-lateral-distance that exceeds the preceding
    valley by `min_prominence_cm`.

    Returns the Z of that first prominent peak, or None if the scan is too
    flat to find a clean one.
    """
    base_u = base_axis / (np.linalg.norm(base_axis) + 1e-12)
    remaining_set = np.zeros(len(verts), dtype=bool)
    remaining_set[remaining_vert] = True
    z_proj_all = verts @ base_u

    remaining_z = z_proj_all[remaining_vert]
    if len(remaining_z) == 0:
        return None
    z_start = float(remaining_z.min())
    z_end = min(float(remaining_z.max()), z_start + search_range_above)

    z_values = np.arange(z_start, z_end + z_step, z_step)
    laterals = []
    for z in z_values:
        slab_mask = (np.abs(z_proj_all - z) < slab_thick / 2) & remaining_set
        if slab_mask.sum() < min_slab_verts:
            laterals.append(None)
            continue
        slab = verts[slab_mask]
        center = slab.mean(axis=0)
        rel = slab - center
        rel_perp = rel - (rel @ base_u)[:, None] * base_u[None, :]
        laterals.append(float(np.mean(np.linalg.norm(rel_perp, axis=1))))

    profile = " ".join(
        f"{v*100:.1f}" if v is not None else "---" for v in laterals
    )
    print(f"    [housing-peak scan] Z range [{z_start:.3f}, {z_end:.3f}]  "
          f"lateral_mean(cm): {profile}")

    # Walk from the start: find the first local max whose prominence (drop
    # from peak to the NEXT valley) exceeds the threshold.
    n = len(z_values)
    min_prom = min_prominence_cm / 100.0
    i = 1
    while i < n - 1:
        v_prev, v_here, v_next = laterals[i - 1], laterals[i], laterals[i + 1]
        if v_prev is None or v_here is None or v_next is None:
            i += 1
            continue
        if v_here > v_prev and v_here > v_next:
            # Check prominence by looking further right for the next valley
            next_valley = v_next
            for k in range(i + 2, min(n, i + 10)):
                vk = laterals[k]
                if vk is None:
                    continue
                next_valley = min(next_valley, vk)
            if v_here - next_valley >= min_prom:
                z_peak = float(z_values[i])
                print(f"    [housing-peak scan] first peak at Z={z_peak:.3f} "
                      f"(lateral={v_here*100:.1f} cm, prom={(v_here-next_valley)*100:.1f} cm)")
                return z_peak
        i += 1
    return None


def _pitch_cut_from_slab(
    verts: np.ndarray,
    pivot: np.ndarray,
    base_axis: np.ndarray,
    remaining_vert: np.ndarray,
    seed_verts: np.ndarray,
    slab_hi: float = 0.06,
    parent_lo: float = -0.06,
    margin: float = 0.01,
) -> tuple[np.ndarray, np.ndarray, float, int] | None:
    """Determine cut_normal and cut_point for a pitch joint by analyzing the
    mesh local geometry around the joint pivot.

    Approach:
      1. Take a thin Z-slab ABOVE the pivot (child region: the first few cm
         of the child link beyond the joint).
      2. PCA of slab verts → primary direction = child link's long axis.
      3. If this axis is substantially horizontal (not parallel to base),
         use it as cut_normal. Snap to nearest perpendicular-to-base canonical.
      4. Orient cut_normal so +cut_normal points toward the seed centroid
         (moving side).
      5. Find parent's maximum extent along cut_normal (verts BELOW the pivot
         in remaining region). Cut at that max + small margin — i.e., just
         past where the parent link ends in the child's direction.

    Returns (cut_normal, cut_point, horizontality, canonical_index) or None
    if the slab is too vertical or has too few verts.
    """
    base_u = base_axis / (np.linalg.norm(base_axis) + 1e-12)
    proj_base = (verts - pivot) @ base_u
    remaining_set = np.zeros(len(verts), dtype=bool)
    remaining_set[remaining_vert] = True

    # Slab above pivot (child region)
    slab_above_mask = (proj_base > 0) & (proj_base < slab_hi) & remaining_set
    if slab_above_mask.sum() < 30:
        return None
    fit = _fit_local_primitive(verts[slab_above_mask])
    if fit is None:
        return None
    _, slab_dir, _ = fit

    # Horizontality: 1.0 = fully perpendicular to base, 0.0 = parallel
    horiz = float(np.sqrt(1.0 - (slab_dir @ base_u) ** 2))
    if horiz < 0.7:
        return None

    # Orient slab_dir to point from parent toward child (using seeds as hint)
    if len(seed_verts) > 0:
        seed_centroid = seed_verts.mean(axis=0)
        if float(slab_dir @ (seed_centroid - pivot)) < 0:
            slab_dir = -slab_dir

    # Snap to nearest perpendicular-to-base canonical, preserving orientation
    snapped, _, cn_i = _snap_to_canonical(slab_dir, base_u, perpendicular_only=True)
    if float(snapped @ slab_dir) < 0:
        snapped = -snapped
    cut_normal = snapped

    # Parent's extent along cut_normal — verts BELOW the pivot (along base).
    # Additionally restrict to verts NEAR the parent link's centerline in the
    # direction perpendicular to both cut_normal and base_axis. Without this,
    # a wider link_2 housing whose mesh extends downward into link_1's Z
    # range inflates parent_max and pushes the cut outside the arm.
    side_dir = np.cross(base_u, cut_normal)
    side_n = float(np.linalg.norm(side_dir))
    if side_n > 1e-9:
        side_dir = side_dir / side_n
        # Parent link's centerline along side_dir: estimated from the slab
        # directly above the previous cut (bottom of remaining region, away
        # from any link_2 overlap).
        remaining_z = (verts @ base_u)[remaining_vert]
        bottom_z = float(remaining_z.min())
        bottom_slab_mask = ((verts @ base_u) > bottom_z + 0.01) & \
                           ((verts @ base_u) < bottom_z + 0.05) & \
                           remaining_set
        if bottom_slab_mask.sum() >= 30:
            side_center = float(np.median(verts[bottom_slab_mask] @ side_dir))
        else:
            side_center = float(np.median(verts[remaining_vert] @ side_dir))
        side_proj = verts @ side_dir
        side_mask = np.abs(side_proj - side_center) < 0.05   # 5 cm from centerline
    else:
        side_mask = np.ones(len(verts), dtype=bool)

    parent_mask = (proj_base < 0) & (proj_base > parent_lo) & remaining_set & side_mask
    if parent_mask.sum() < 10:
        return cut_normal, pivot.copy(), horiz, cn_i

    parent_proj = (verts[parent_mask] - pivot) @ cut_normal
    parent_max = float(np.max(parent_proj))
    cut_point = pivot + (parent_max + margin) * cut_normal
    return cut_normal, cut_point, horiz, cn_i


def _count_crossing_faces(
    verts: np.ndarray,
    faces: np.ndarray,
    face_idx: np.ndarray,
    cut_point: np.ndarray,
    cut_normal: np.ndarray,
) -> int:
    """Count faces (restricted to face_idx) whose vertices straddle the cut plane.

    A face "crosses" if it has at least one vertex on each side of the plane.
    Faces that intersect a narrow mechanical neck produce a low count; faces
    through thick geometry produce a high count. Minimum crossing count across
    candidate cut_normals ≈ the natural mechanical break between links.
    """
    cn_u = cut_normal / (np.linalg.norm(cut_normal) + 1e-12)
    proj = (verts - cut_point) @ cn_u
    face_verts_proj = proj[faces[face_idx]]        # shape (nf, 3)
    has_pos = (face_verts_proj > 0).any(axis=1)
    has_neg = (face_verts_proj < 0).any(axis=1)
    return int((has_pos & has_neg).sum())


def _pick_cut_normal_by_neck(
    verts: np.ndarray,
    faces: np.ndarray,
    remaining_face: np.ndarray,
    cut_point: np.ndarray,
    motion_axis_u: np.ndarray,
    base_axis: np.ndarray,
    verbose: bool = False,
) -> tuple[np.ndarray, int, str]:
    """Try each canonical axis as a candidate cut_normal and pick the one that
    (a) is motion-compatible (either ≈parallel to motion = roll-joint case, or
    ≈perpendicular to motion = pitch-joint case; a 45°-tilted axis would be
    neither) and (b) minimizes mesh-crossing face count (i.e. passes through
    the narrowest mechanical neck).

    Returns (cut_normal, canonical_index, compat_kind). compat_kind is "roll"
    or "pitch". If no compatible candidate exists, falls back to the
    closest-to-motion canonical and classifies as roll.
    """
    canonicals = _canonical_axes_relative_to_base(base_axis)
    cos_roll = np.cos(np.deg2rad(20.0))     # |m·n| > this → parallel → roll
    cos_pitch = np.cos(np.deg2rad(70.0))    # |m·n| < this → perpendicular → pitch

    candidates: list[tuple[int, int, np.ndarray, str]] = []
    for i, c in enumerate(canonicals):
        cos_m = abs(float(c @ motion_axis_u))
        if cos_m > cos_roll:
            compat = "roll"
        elif cos_m < cos_pitch:
            compat = "pitch"
        else:
            continue                         # ambiguous 20°–70° band, skip
        crossing = _count_crossing_faces(
            verts, faces, remaining_face, cut_point, c,
        )
        candidates.append((crossing, i, c, compat))

    if not candidates:
        # Fallback: snap motion to nearest canonical and treat as roll
        fallback, _, fb_i = _snap_to_canonical(motion_axis_u, base_axis)
        return fallback, fb_i, "roll"

    candidates.sort(key=lambda r: r[0])
    if verbose:
        for cr, i, c, compat in candidates[:5]:
            print(f"    neck candidate {_canonical_label(i):>5}  "
                  f"crossings={cr:>6}  {compat}")
    best_crossing, best_i, best_c, best_compat = candidates[0]
    if float(best_c @ motion_axis_u) < 0 and best_compat == "roll":
        best_c = -best_c
    return best_c, best_i, best_compat


def _canonical_axes_relative_to_base(base_axis: np.ndarray) -> list[np.ndarray]:
    """Candidate joint axes under the 'parallel or perpendicular to base' prior.

    Most industrial manipulators have every joint axis either parallel to the
    robot's mounting axis (roll/twist joints) or perpendicular to it (pitch/yaw
    joints). We return one parallel candidate plus 6 perpendicular candidates
    sampled around the perpendicular great-circle (30° increments, 0..180°
    since axis direction is sign-ambiguous).
    """
    base = base_axis / (np.linalg.norm(base_axis) + 1e-12)
    tmp = np.array([1.0, 0.0, 0.0]) if abs(base[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = np.cross(base, tmp); e1 /= (np.linalg.norm(e1) + 1e-12)
    e2 = np.cross(base, e1)
    cands = [base.copy()]
    for phi in np.linspace(0.0, np.pi, 6, endpoint=False):
        cands.append(np.cos(phi) * e1 + np.sin(phi) * e2)
    return cands


def _split_remaining_by_plane(
    verts: np.ndarray,
    faces: np.ndarray,
    vert_idx: np.ndarray,   # indices into the FULL mesh that make up the "remaining" region
    face_idx: np.ndarray,   # corresponding face indices
    axis: np.ndarray,
    origin: np.ndarray,
    moving_side_sign: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split (vert_idx, face_idx) by a plane (axis, origin).

    Returns (below_vert, below_face, above_vert, above_face) as indices into
    the full mesh.

    "Below" = the side OPPOSITE `moving_side_sign`.
    "Above" = the side of `moving_side_sign` (the moving chain).
    """
    axis_u = axis / (np.linalg.norm(axis) + 1e-12)
    proj = (verts - origin) @ axis_u          # signed proj for full mesh
    if moving_side_sign > 0:
        above_mask = proj > 0
    else:
        above_mask = proj < 0

    # Restrict to "remaining" region
    region_mask = np.zeros(len(verts), dtype=bool)
    region_mask[vert_idx] = True

    above_verts_mask = above_mask & region_mask
    below_verts_mask = (~above_mask) & region_mask

    # Face goes above/below based on majority of its vertices
    face_in_region = np.isin(faces[:, 0], vert_idx) & np.isin(faces[:, 1], vert_idx) & \
                     np.isin(faces[:, 2], vert_idx)
    faces_region = np.where(face_in_region)[0]
    region_faces = faces[face_in_region]
    above_count = above_verts_mask[region_faces].sum(axis=1)
    above_face_local = above_count >= 2
    above_faces = faces_region[above_face_local]
    below_faces = faces_region[~above_face_local]

    above_vert = np.unique(faces[above_faces]) if len(above_faces) else np.empty(0, np.int64)
    below_vert = np.unique(faces[below_faces]) if len(below_faces) else np.empty(0, np.int64)

    return below_vert, below_faces, above_vert, above_faces




def _slice_mesh(mesh: trimesh.Trimesh, face_idx: np.ndarray) -> trimesh.Trimesh:
    """Build a sub-mesh from the given face indices, dropping unused vertices."""
    verts = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.faces)
    sub_faces = faces[face_idx]
    used = np.unique(sub_faces)
    remap = -np.ones(len(verts), dtype=np.int64)
    remap[used] = np.arange(len(used))
    return trimesh.Trimesh(
        vertices=verts[used], faces=remap[sub_faces], process=False,
    )


def run(
    mesh_path: Path,
    motion_dir: Path,
    calibration_path: Path,
    output_dir: Path,
    mesh_to_world_npy: Path | None = None,
    min_inliers: int = 5,
    n_features: int = 20000,
    use_sift: bool = False,
    ratio: float = 0.95,
    reproj_threshold_px: float = 4.0,
    identity_tolerance_px: float = 4.0,
    base_axis: tuple[float, float, float] = (0.0, 0.0, 1.0),
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    K, dist, _ = _load_calibration(calibration_path)
    mesh = trimesh.load(str(mesh_path), force="mesh")
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)
    if mesh_to_world_npy is not None:
        T = np.load(mesh_to_world_npy)
        mesh = mesh.copy()
        mesh.apply_transform(T)
    verts = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.faces)
    V = len(verts)
    F = len(faces)
    print(f"Loaded mesh: {V} verts, {F} faces")

    joint_dirs = _find_joint_dirs(motion_dir)
    if not joint_dirs:
        raise SystemExit(f"No joint_* folders under {motion_dir}")
    print(f"Found {len(joint_dirs)} joint folders: {[d.name for d in joint_dirs]}")

    # --- Extract motion per joint ---
    joints_raw: list[JointInfo] = []
    for jdir in joint_dirs:
        states = _find_state_images(jdir)
        if len(states) < 2:
            print(f"  {jdir.name}: <2 states, skipping")
            continue
        print(f"\n== {jdir.name} ({len(states)} states) ==")
        result = extract_joint_motion_multi(
            mesh=mesh, state_paths=states,
            camera_matrix=K, dist_coeffs=dist,
            joint_name=jdir.name,
            n_features=n_features, use_sift=use_sift, ratio=ratio,
            min_inliers=min_inliers,
            reproj_threshold_px=reproj_threshold_px,
            identity_tolerance_px=identity_tolerance_px,
        )
        if result is None:
            print(f"  no motion extracted")
            continue
        print(f"  angle: {np.rad2deg(result.total_angle_rad):+.2f}°  "
              f"axis: {np.round(result.axis_world, 3)}  "
              f"moved verts: {int(result.moved_vertex_mask.sum())}")
        joints_raw.append(JointInfo(
            name=jdir.name,
            result=result,
            axis=result.axis_world.copy(),
            origin=result.origin_world.copy(),
            cut_offset=0.0,   # set in next phase
        ))

    if not joints_raw:
        raise SystemExit("No joints extracted motion — cannot assemble.")

    # --- Determine per-joint cut point from motion (biggest-gap heuristic) ---
    # We keep TWO points per joint:
    #   - j.origin: the RANSAC pivot (motion-axis-independent; any point on the
    #     rotation axis — good for measuring link direction from joint to joint).
    #   - j.cut_point: origin + biggest-gap-along-motion-axis (a point on the
    #     mechanical cut plane — good for locating the partition plane).
    base_axis_np = np.asarray(base_axis, dtype=np.float64)
    base_axis_np = base_axis_np / (np.linalg.norm(base_axis_np) + 1e-12)
    print(f"\nGeometric prior: base axis = {np.round(base_axis_np, 3).tolist()}")
    for j in joints_raw:
        seed_idx = np.where(j.result.moved_vertex_mask)[0]
        motion_axis_u = j.axis / (np.linalg.norm(j.axis) + 1e-12)
        seed_proj = (verts[seed_idx] - j.origin) @ motion_axis_u
        bias = _biggest_gap_cut(seed_proj)
        j.cut_point = j.origin + bias * motion_axis_u

    # --- Order joints by height along base axis (using cut_point for cut Z) ---
    joints_raw.sort(key=lambda j: float(j.cut_point @ base_axis_np))
    print("\nJoints ordered by cut height:")
    for j in joints_raw:
        print(f"  {j.name}: cut along base={float(j.cut_point @ base_axis_np):+.3f}  "
              f"cut_point={np.round(j.cut_point, 3).tolist()}")

    # --- Robot retrieval: match captured geometry against URDF database ---
    # The match gives us per-joint axis priors at home pose. When matched
    # confidently to a known robot family, we can override the canonical
    # snap with the matched reference's expected axis (e.g. xArm6 says
    # joint_2 axis = Y, even when the noisy motion estimate is 30° off
    # and would otherwise snap to ⊥30°).
    mesh_z_min = float(verts[:, 2].min())
    mesh_z_max = float(verts[:, 2].max())
    mesh_height = max(mesh_z_max - mesh_z_min, 1e-6)
    snapped_axes_for_match = []
    captured_z_fracs = []
    for j in joints_raw:
        ma = j.axis / (np.linalg.norm(j.axis) + 1e-12)
        snapped, _, _ = _snap_to_canonical(ma, base_axis_np)
        snapped_axes_for_match.append(snapped)
        captured_z_fracs.append(
            (float(j.cut_point @ base_axis_np) - mesh_z_min) / mesh_height
        )
    match = match_robot(snapped_axes_for_match, captured_z_fracs)
    print()
    print(summarize_match(match))

    # --- Classify roll/pitch via mesh-neck detection, and partition ---
    # For each joint in cut-height order we pick a cut plane by searching over
    # the 7 canonical directions (parallel to base + 6 perpendiculars) and
    # selecting the one that minimizes the count of mesh faces crossing the
    # plane — i.e., the cut passes through the narrowest mechanical neck.
    # Motion axis compatibility filters the candidates to roll-like or
    # pitch-like orientations; 45°-tilted canonicals are rejected as ambiguous.
    # This decouples "where to slice" (mesh geometry question) from "which way
    # does the joint rotate" (motion question) without relying on noisy
    # RANSAC-pivot link directions, which degrade when camera coverage is poor.
    print("\nSuccessive mesh partitioning:")
    remaining_vert = np.arange(V)
    remaining_face = np.arange(F)
    ordered_joints: list[JointInfo] = []
    link_face_sets: list[np.ndarray] = []
    # If retrieval matched a known robot, expose its per-joint axis priors.
    # We trust the match for axis/origin transfer whenever the score is
    # above a baseline (0.6), regardless of margin: a low margin usually
    # means a tie between very similar robots (e.g. xArm6 vs xArm7 vs
    # UR5e — all share j1=Z, j2=Y), so the per-joint axis prior is the
    # same anyway.
    use_match_prior = (
        match.record is not None
        and (match.confidence in ("high", "medium", "hinted")
             or match.score >= 0.6)
    )
    if use_match_prior:
        ref_axes_world = [np.asarray(a) for a in match.record["joint_axes_world"]]
        ref_z_fractions = list(match.record["joint_z_fractions"])
        print(f"    [retrieval] using {match.record['name']} "
              f"as axis/origin prior ({match.confidence} confidence, "
              f"score {match.score:.2f})")

    prev_ransac_pivot = None
    for joint_idx, j in enumerate(joints_raw):
        ransac_pivot_this = j.origin.copy()   # capture BEFORE we overwrite j.origin
        motion_axis_u = j.axis / (np.linalg.norm(j.axis) + 1e-12)

        # Link direction from RANSAC pivots (motion-axis-independent).
        if prev_ransac_pivot is None:
            link_dir = base_axis_np.copy()
        else:
            link_vec = ransac_pivot_this - prev_ransac_pivot
            link_len = float(np.linalg.norm(link_vec))
            link_dir = link_vec / link_len if link_len > 1e-6 else base_axis_np.copy()

        # Roll vs pitch: motion axis aligned with link direction?
        motion_link_cos = abs(float(motion_axis_u @ link_dir))
        is_roll = motion_link_cos > np.cos(np.deg2rad(30.0))

        if is_roll:
            snapped_axis, axis_snap_ang, axis_i = _snap_to_canonical(
                motion_axis_u, base_axis_np,
            )
            cut_normal = snapped_axis.copy()
            cn_i = axis_i
            kind = "ROLL"
        else:
            # PITCH — correct pivot Z. Prefer the matched reference's
            # expected joint Z-fraction if available; otherwise fall back to
            # the mesh's first lateral-extent peak above the previous cut.
            guess_z = float(j.cut_point @ base_axis_np)
            corrected_z = None
            if use_match_prior and joint_idx < len(ref_z_fractions):
                ref_zf = ref_z_fractions[joint_idx]
                predicted_z = mesh_z_min + ref_zf * mesh_height
                if abs(predicted_z - guess_z) > 0.02:
                    print(f"    [pivot-Z prior from {match.record['name']}] "
                          f"motion Z={guess_z:.3f} → ref Z-fraction "
                          f"{ref_zf:.2f} × {mesh_height:.3f}m = {predicted_z:.3f}")
                corrected_z = predicted_z
            else:
                peak_z = _find_first_housing_peak_z(
                    verts=verts, base_axis=base_axis_np,
                    remaining_vert=remaining_vert,
                )
                if peak_z is not None and abs(peak_z - guess_z) > 0.02:
                    print(f"    [pivot-Z correction] motion Z={guess_z:.3f} "
                          f"→ mesh housing peak Z={peak_z:.3f}")
                    corrected_z = peak_z
            if corrected_z is not None:
                j.cut_point = j.cut_point + (corrected_z - guess_z) * base_axis_np

            # Cut normal — use base_axis (horizontal cut) by default.
            cut_normal, _, cn_i = _snap_to_canonical(link_dir, base_axis_np)

            # Joint rotation axis — prefer matched reference's axis when
            # available and within 60° of the motion observation.
            if use_match_prior and joint_idx < len(ref_axes_world):
                ref_a = ref_axes_world[joint_idx]
                ref_a = ref_a / (np.linalg.norm(ref_a) + 1e-12)
                cos_to_motion = abs(float(ref_a @ motion_axis_u))
                ang_to_motion = np.rad2deg(np.arccos(np.clip(cos_to_motion, -1, 1)))
                if ang_to_motion <= 60.0:
                    if float(ref_a @ motion_axis_u) < 0:
                        ref_a = -ref_a
                    snapped_axis = ref_a
                    axis_snap_ang = float(ang_to_motion)
                    # Determine canonical index for label/diagnostics
                    _, _, axis_i = _snap_to_canonical(
                        ref_a, cut_normal, perpendicular_only=True,
                    )
                    print(f"    [axis prior] joint_{joint_idx+1} "
                          f"snap to ref axis {np.round(ref_a, 3).tolist()} "
                          f"(motion {ang_to_motion:.1f}° away)")
                else:
                    snapped_axis, axis_snap_ang, axis_i = _snap_to_canonical(
                        motion_axis_u, cut_normal, perpendicular_only=True,
                    )
            else:
                snapped_axis, axis_snap_ang, axis_i = _snap_to_canonical(
                    motion_axis_u, cut_normal, perpendicular_only=True,
                )
            kind = "PITCH"

        # Moving side = the distal side (sign of link_dir projected onto cut_normal).
        moving_side_sign = +1 if float(link_dir @ cut_normal) > 0 else -1

        # Partition
        below_v, below_f, above_v, above_f = _split_remaining_by_plane(
            verts=verts, faces=faces,
            vert_idx=remaining_vert, face_idx=remaining_face,
            axis=cut_normal, origin=j.cut_point,
            moving_side_sign=moving_side_sign,
        )
        ax_label = _canonical_label(axis_i)
        print(f"  {j.name} [{kind}]  link_dir={np.round(link_dir, 2).tolist()} "
              f"(motion·link={motion_link_cos:.2f})")
        print(f"    axis={np.round(snapped_axis, 2).tolist()} "
              f"({ax_label}, snap {axis_snap_ang:.1f}°)  "
              f"cut_normal={np.round(cut_normal, 2).tolist()}")
        print(f"    → below: {len(below_v)} verts / {len(below_f)} faces")
        print(f"    → above: {len(above_v)} verts / {len(above_f)} faces")

        if len(below_f) < 100:
            print(f"    [warn] below-plane region very small; joint may be mis-located")

        # Refine cut_point to sit at the interface-ring centroid — the
        # geometric center of where the two links meet. Prevents the joint
        # axis line from being offset in-plane by motion-axis bias
        # (e.g., joint_1 motion axis 12° off Z puts cut_point 3-5 cm off
        # the physical base-cylinder center, even after axis snaps to Z).
        # We move cut_point WITHIN its own cut plane, so the partition above
        # is unaffected.
        cn_u = cut_normal / (np.linalg.norm(cut_normal) + 1e-12)
        proj_to_cut = (verts - j.cut_point) @ cn_u
        band_mask = np.abs(proj_to_cut) < 0.015
        region_verts_idx = np.concatenate([below_v, above_v]) if len(above_v) \
                           else below_v
        if len(region_verts_idx) > 0:
            region_mask = np.zeros(len(verts), dtype=bool)
            region_mask[region_verts_idx] = True
            interface_mask = band_mask & region_mask
            if interface_mask.sum() >= 20:
                interface_centroid = verts[interface_mask].mean(axis=0)
                # Project the centroid back onto the original cut plane
                # (preserve the cut's along-normal position, only shift
                # within the plane).
                delta = interface_centroid - j.cut_point
                in_plane_shift = delta - (delta @ cn_u) * cn_u
                j.cut_point = j.cut_point + in_plane_shift
                print(f"    interface refine: shifted cut_point by "
                      f"{np.round(in_plane_shift, 3).tolist()} "
                      f"(n_ring={int(interface_mask.sum())})")

        # Update joint record for URDF emission
        j.axis = snapped_axis
        j.origin = j.cut_point.copy()
        j.cut_normal = cut_normal
        j.cut_offset = 0.0

        link_face_sets.append(below_f)
        ordered_joints.append(j)
        remaining_vert, remaining_face = above_v, above_f
        prev_ransac_pivot = ransac_pivot_this

    link_face_sets.append(remaining_face)
    print(f"  final link: {len(remaining_vert)} verts / {len(remaining_face)} faces")

    # --- Build sub-meshes for each link ---
    per_link_meshes: dict[int, trimesh.Trimesh] = {}
    per_link_collisions: dict[int, trimesh.Trimesh] = {}
    for i, face_set in enumerate(link_face_sets):
        if len(face_set) < 3:
            continue
        sub = _slice_mesh(mesh, face_set)
        per_link_meshes[i] = sub
        per_link_collisions[i] = sub

    # Name links
    link_name_map = {0: "link_base"}
    for i in range(1, len(link_face_sets)):
        link_name_map[i] = f"link_{i}" if i < len(link_face_sets) - 1 else f"link_tip"

    # --- Build JointEstimate list ---
    je_list = []
    for i, j in enumerate(ordered_joints):
        je_list.append(JointEstimate(
            parent_body=i,
            child_body=i + 1,
            type="revolute",
            axis=j.axis.copy(),
            origin=j.origin.copy(),
            angles=[0.0, float(j.result.total_angle_rad)],
            lower=min(0.0, float(j.result.total_angle_rad)),
            upper=max(0.0, float(j.result.total_angle_rad)),
        ))

    # --- Phase 4 physics ---
    dof = len(je_list)
    tpl = match_template(dof, ["revolute"] * dof)
    print(f"\nTemplate: {tpl.name}  density={tpl.density:.0f} kg/m^3")
    inertials = compute_link_inertials(per_link_meshes, density=tpl.density)
    for i, ine in sorted(inertials.items()):
        print(f"  {link_name_map[i]}: mass={ine.mass:.3f} kg")

    # --- Phase 5 URDF assembly ---
    inp = AssemblyInput(
        robot_name="xarm_multijoint",
        per_link_meshes=per_link_meshes,
        per_link_collisions=per_link_collisions,
        joints=je_list,
        inertials=inertials,
        template=tpl,
        body_transforms_pose0=[np.eye(4)] * (dof + 1),
        link_name_map=link_name_map,
    )
    urdf_path = assemble(inp, output_dir)
    print(f"\nWrote {urdf_path}")

    # Verify
    try:
        r = URDF.load(str(urdf_path))
        print(f"  URDF loads OK: {len(r.link_map)} links, "
              f"{len(r.actuated_joint_names)} actuated joints")
    except Exception as e:
        print(f"  URDF load failed: {e}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--motion-dir", type=Path, required=True,
                        help="Parent folder containing joint_1/, joint_2/, ...")
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mesh-to-world", type=Path, default=None)
    parser.add_argument("--min-inliers", type=int, default=5)
    parser.add_argument("--n-features", type=int, default=20000)
    parser.add_argument("--use-sift", action="store_true")
    parser.add_argument("--ratio", type=float, default=0.95)
    parser.add_argument("--reproj-px", type=float, default=4.0)
    parser.add_argument("--identity-px", type=float, default=4.0)
    parser.add_argument(
        "--base-axis", type=float, nargs=3, default=[0.0, 0.0, 1.0],
        metavar=("X", "Y", "Z"),
        help="Robot base (mounting) axis in world frame. Joint axes are "
             "assumed to be parallel or perpendicular to this direction.",
    )
    args = parser.parse_args()
    run(
        mesh_path=args.mesh,
        motion_dir=args.motion_dir,
        calibration_path=args.calibration,
        output_dir=args.output,
        mesh_to_world_npy=args.mesh_to_world,
        min_inliers=args.min_inliers,
        n_features=args.n_features,
        use_sift=args.use_sift,
        ratio=args.ratio,
        reproj_threshold_px=args.reproj_px,
        identity_tolerance_px=args.identity_px,
        base_axis=tuple(args.base_axis),
    )


if __name__ == "__main__":
    main()
