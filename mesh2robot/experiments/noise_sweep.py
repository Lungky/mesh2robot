"""Noise robustness sweep for Phases 2 + 3.

Injects iid Gaussian noise into synthetic vertex positions and runs the
full segmentation + joint extraction pipeline at several noise levels.
Reports degradation curves for: segmentation accuracy, mean axis error,
mean origin error, and fraction of joints recovered.

Real MILO meshes typically have sub-mm to few-mm vertex error depending on
capture quality, so σ = 0.1 .. 5 mm spans the realistic operating range.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from mesh2robot.core.joint_extraction import extract_joints
from mesh2robot.core.motion_segmentation import (
    assign_orphans_to_nearest_body,
    merge_duplicate_bodies,
    segment_multi_pose,
)
from mesh2robot.experiments.feasibility_xarm6 import (
    axis_angle_error,
    evaluate_segmentation,
    ground_truth_joints,
    line_distance,
    load_pose_meshes,
)


def inject_noise(pose_pts: np.ndarray, sigma_m: float, rng: np.random.Generator) -> np.ndarray:
    """Add iid N(0, sigma) Gaussian noise to each vertex in each pose."""
    if sigma_m <= 0.0:
        return pose_pts.copy()
    return pose_pts + rng.normal(0.0, sigma_m, size=pose_pts.shape)


def _evaluate_joints(joints, body_to_link, link_names) -> tuple[float, float, int]:
    """Match predicted joints against GT and compute mean errors."""
    gts = ground_truth_joints()
    gt_by_child = {g["child"]: g for g in gts}

    axis_errs = []
    origin_errs = []
    matched = 0
    for je in joints:
        child_idx = body_to_link.get(je.child_body, -1)
        if child_idx < 0:
            continue
        child_name = link_names[child_idx]
        g = gt_by_child.get(child_name)
        if g is None:
            continue
        matched += 1
        axis_errs.append(axis_angle_error(je.axis, g["axis"]))
        origin_errs.append(line_distance(je.origin, je.axis, g["origin"], g["axis"]) * 1000.0)

    if not axis_errs:
        return float("nan"), float("nan"), 0
    return float(np.mean(axis_errs)), float(np.mean(origin_errs)), matched


def run_sweep(
    sigmas_mm: list[float],
    seeds: list[int],
    threshold_scale: float = 4.0,
    min_threshold_m: float = 5e-4,
    merge_bodies: bool = True,
) -> list[dict]:
    """Run the full pipeline at each (sigma, seed) pair and collect metrics."""
    pose_pts_clean, gt_labels, link_names, _ = load_pose_meshes()

    rows = []
    for sigma_mm in sigmas_mm:
        sigma_m = sigma_mm * 1e-3
        # Scale inlier threshold with noise level: 4 sigma, floored at 0.5 mm.
        thr = max(min_threshold_m, threshold_scale * sigma_m)
        # Merge tolerance: bodies whose transforms differ by less than 10*threshold
        # are considered the same physical link that got fragmented.
        merge_tol = 10.0 * thr

        for seed in seeds:
            rng = np.random.default_rng(seed)
            pose_pts = inject_noise(pose_pts_clean, sigma_m, rng)

            seg = segment_multi_pose(
                pose_pts,
                threshold=thr,
                min_inliers=200,
                max_bodies=10,
                n_trials=300,
                rng_seed=seed,
            )
            if merge_bodies:
                seg = merge_duplicate_bodies(seg, pose_pts, merge_tol=merge_tol)
            # Reassign RANSAC-rejected vertices to their closest-matching body
            seg = assign_orphans_to_nearest_body(seg, pose_pts)
            ev = evaluate_segmentation(seg.labels, gt_labels, len(link_names))
            joints = extract_joints(seg.body_transforms) if seg.n_bodies > 0 else []
            mean_ae, mean_oe, n_matched = _evaluate_joints(
                joints, ev["body_to_link"], link_names
            )

            rows.append({
                "sigma_mm": sigma_mm,
                "seed": seed,
                "threshold_mm": thr * 1000.0,
                "n_bodies": seg.n_bodies,
                "seg_accuracy": ev["accuracy"],
                "n_joints_matched": n_matched,
                "mean_axis_err_deg": mean_ae,
                "mean_origin_err_mm": mean_oe,
            })
            print(f"  sigma={sigma_mm:5.2f}mm seed={seed}  "
                  f"bodies={seg.n_bodies}  acc={ev['accuracy']*100:5.2f}%  "
                  f"joints={n_matched}/6  axis={mean_ae:6.3f}deg  "
                  f"origin={mean_oe:6.3f}mm")

    return rows


def summarize(rows: list[dict]) -> list[dict]:
    """Aggregate by sigma across seeds: mean + std of each metric."""
    from collections import defaultdict
    by_sigma = defaultdict(list)
    for r in rows:
        by_sigma[r["sigma_mm"]].append(r)

    summary = []
    for sigma, group in sorted(by_sigma.items()):
        acc = np.array([g["seg_accuracy"] for g in group])
        ae = np.array([g["mean_axis_err_deg"] for g in group])
        oe = np.array([g["mean_origin_err_mm"] for g in group])
        nj = np.array([g["n_joints_matched"] for g in group])
        summary.append({
            "sigma_mm": sigma,
            "seg_accuracy_mean": float(np.nanmean(acc)),
            "seg_accuracy_std": float(np.nanstd(acc)),
            "axis_err_mean_deg": float(np.nanmean(ae)),
            "axis_err_std_deg": float(np.nanstd(ae)),
            "origin_err_mean_mm": float(np.nanmean(oe)),
            "origin_err_std_mm": float(np.nanstd(oe)),
            "joints_matched_mean": float(np.nanmean(nj)),
        })
    return summary


def main():
    sigmas_mm = [0.0, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    seeds = [0, 1, 2]
    print(f"Noise sweep: sigmas_mm={sigmas_mm}  seeds={seeds}")
    print("-" * 90)
    rows = run_sweep(sigmas_mm, seeds)

    # Save raw rows
    out_dir = Path(__file__).resolve().parents[2] / "data" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "noise_sweep.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nRaw results -> {csv_path}")

    print("\nSUMMARY (mean across seeds)")
    print(f"{'sigma_mm':>10s} {'seg_acc':>10s} {'axis_err':>12s} {'origin_err':>14s} {'joints':>8s}")
    for s in summarize(rows):
        print(f"{s['sigma_mm']:>10.2f} "
              f"{s['seg_accuracy_mean']*100:>9.2f}% "
              f"{s['axis_err_mean_deg']:>10.3f}deg "
              f"{s['origin_err_mean_mm']:>12.3f}mm "
              f"{s['joints_matched_mean']:>7.1f}/6")


if __name__ == "__main__":
    main()
