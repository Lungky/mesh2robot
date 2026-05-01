"""Feasibility experiment: recover xArm6 kinematics from synthetic pose meshes.

Pipeline:
  1. Load 13 synthetic pose meshes (produced by mesh2robot.io.synthetic_poses).
  2. Phase 2 — motion-based segmentation.
  3. Phase 3 — joint axis extraction.
  4. Compare recovered axes / origins against the official xArm6 URDF ground
     truth. Report axis error (degrees) and origin error (mm) per joint.

Pass criteria (from ROADMAP.md):
    axis direction error  < 2 deg
    origin error          < 5 mm
    vertex assignment     > 95 %
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from robot_descriptions.loaders.yourdfpy import load_robot_description

from mesh2robot.core.joint_extraction import extract_joints
from mesh2robot.core.motion_segmentation import segment_multi_pose


DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "synthetic" / "xarm6"


def load_pose_meshes() -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray]:
    """Load K synthetic poses and stack into (K, N, 3)."""
    meta = json.loads((DATA_DIR / "metadata.json").read_text())
    K = len(meta["configurations"])
    per_pose = [np.load(DATA_DIR / f"pose_{i:02d}.npz") for i in range(K)]
    pose_pts = np.stack([p["vertices"] for p in per_pose])          # (K, N, 3)
    vertex_link_gt = per_pose[0]["vertex_link"]                      # (N,)
    link_names = list(per_pose[0]["link_names"])
    # Ground-truth per-link transforms for each pose: (K, L, 4, 4)
    gt_link_T = np.stack([p["link_transforms"] for p in per_pose])
    return pose_pts, vertex_link_gt, link_names, gt_link_T


def match_bodies_to_links(
    pred_labels: np.ndarray, gt_labels: np.ndarray, n_links: int
) -> dict[int, int]:
    """For each predicted body, find the GT link with maximal vertex overlap."""
    n_bodies = int(pred_labels.max()) + 1 if pred_labels.max() >= 0 else 0
    mapping = {}
    for b in range(n_bodies):
        mask = pred_labels == b
        if not mask.any():
            continue
        counts = np.bincount(gt_labels[mask], minlength=n_links)
        mapping[b] = int(np.argmax(counts))
    return mapping


def evaluate_segmentation(pred: np.ndarray, gt: np.ndarray, n_links: int) -> dict:
    mapping = match_bodies_to_links(pred, gt, n_links)
    remapped = np.full_like(pred, -1)
    for b, l in mapping.items():
        remapped[pred == b] = l
    correct = (remapped == gt).sum()
    total = len(gt)
    return {
        "body_to_link": mapping,
        "accuracy": correct / total,
        "n_bodies": int(pred.max()) + 1 if pred.max() >= 0 else 0,
        "n_unassigned": int((pred == -1).sum()),
    }


def ground_truth_joints(description_name: str = "xarm6_description") -> list[dict]:
    """Pull joint axis + origin for each revolute joint from the URDF at home pose."""
    urdf = load_robot_description(description_name)
    urdf.update_cfg({j: 0.0 for j in urdf.actuated_joint_names})
    gts = []
    for jname in urdf.actuated_joint_names:
        j = urdf.joint_map[jname]
        # Axis and origin at home: transform parent frame to world, apply joint origin,
        # then axis is in the child-frame z (by URDF convention here).
        T_parent = urdf.get_transform(j.parent, "world")
        T_joint_parent = np.eye(4)
        if j.origin is not None:
            T_joint_parent = j.origin
        T_joint_world = T_parent @ T_joint_parent
        axis_local = np.asarray(j.axis, dtype=float)
        axis_local /= np.linalg.norm(axis_local)
        axis_world = T_joint_world[:3, :3] @ axis_local
        origin_world = T_joint_world[:3, 3]
        gts.append({
            "name": jname, "parent": j.parent, "child": j.child,
            "axis": axis_world, "origin": origin_world,
        })
    return gts


def line_distance(p_a: np.ndarray, a_a: np.ndarray,
                  p_b: np.ndarray, a_b: np.ndarray) -> float:
    """Shortest distance between two infinite lines defined by point+direction."""
    a_a = a_a / np.linalg.norm(a_a)
    a_b = a_b / np.linalg.norm(a_b)
    n = np.cross(a_a, a_b)
    nn = np.linalg.norm(n)
    if nn < 1e-8:
        # Parallel: distance from one point to the other line
        return float(np.linalg.norm(np.cross(p_b - p_a, a_a)))
    return float(abs((p_b - p_a) @ n) / nn)


def axis_angle_error(a: np.ndarray, b: np.ndarray) -> float:
    """Smallest angle between axes (undirected), in degrees."""
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    dot = abs(float(a @ b))
    return float(np.rad2deg(np.arccos(min(1.0, dot))))


def main():
    print("Loading synthetic pose meshes ...")
    pose_pts, gt_labels, link_names, gt_link_T = load_pose_meshes()
    K, N, _ = pose_pts.shape
    print(f"  poses={K}  vertices={N}  links={len(link_names)}")

    print("Phase 2: motion segmentation ...")
    seg = segment_multi_pose(
        pose_pts,
        threshold=5e-4,     # 0.5 mm tolerance (synthetic, noise-free)
        min_inliers=200,
        max_bodies=10,
        n_trials=300,
        rng_seed=0,
    )
    print(f"  bodies found: {seg.n_bodies}  (GT links: {len(link_names)})")

    ev = evaluate_segmentation(seg.labels, gt_labels, len(link_names))
    print(f"  vertex accuracy: {ev['accuracy']*100:.2f}%")
    print(f"  unassigned: {ev['n_unassigned']}")
    print(f"  body -> link mapping: {ev['body_to_link']}")

    print("Phase 3: joint extraction ...")
    joints = extract_joints(seg.body_transforms)
    print(f"  joints inferred: {len(joints)}")

    # Map body indices back to link names for reporting
    body_to_link = ev["body_to_link"]
    print(f"\n{'joint':>20s}  {'axis_err_deg':>14s}  {'origin_err_mm':>14s}  range (deg)")

    gts = ground_truth_joints()
    # Build GT axis/origin indexed by child link name
    gt_by_child = {g["child"]: g for g in gts}

    reports = []
    for je in joints:
        child_link_idx = body_to_link.get(je.child_body, -1)
        if child_link_idx < 0:
            continue
        child_link_name = link_names[child_link_idx]
        g = gt_by_child.get(child_link_name)
        if g is None:
            continue  # Fixed joint or not a matched revolute

        ae = axis_angle_error(je.axis, g["axis"])
        # Origin error: distance between axes (lines), not just point-to-point
        oe = line_distance(je.origin, je.axis, g["origin"], g["axis"]) * 1000.0
        rng_deg = (np.rad2deg(je.lower), np.rad2deg(je.upper))
        print(f"{child_link_name:>20s}  {ae:>14.3f}  {oe:>14.3f}  "
              f"[{rng_deg[0]:+.1f}, {rng_deg[1]:+.1f}]")
        reports.append({
            "joint_child": child_link_name, "axis_err_deg": ae,
            "origin_err_mm": oe, "range_deg": rng_deg,
        })

    # Summary
    if reports:
        mean_ae = np.mean([r["axis_err_deg"] for r in reports])
        mean_oe = np.mean([r["origin_err_mm"] for r in reports])
        print(f"\nSUMMARY  mean axis err = {mean_ae:.3f} deg,  "
              f"mean origin err = {mean_oe:.3f} mm")
        pass_seg = ev["accuracy"] >= 0.95
        pass_axis = mean_ae < 2.0
        pass_origin = mean_oe < 5.0
        print(f"  segmentation > 95%: {pass_seg}")
        print(f"  axis err     < 2°:  {pass_axis}")
        print(f"  origin err   < 5mm: {pass_origin}")
        if pass_seg and pass_axis and pass_origin:
            print("\n  >>> FEASIBILITY PASSED <<<")
        else:
            print("\n  >>> FEASIBILITY INCOMPLETE — see values above <<<")


if __name__ == "__main__":
    main()
