"""Quick sanity check: does body-merge post-processing fix over-segmentation?

Runs sigma in {0.1, 0.5, 1.0, 5.0} with seed=0, compares with/without merge.
"""
from __future__ import annotations

import numpy as np

from mesh2robot.core.joint_extraction import extract_joints
from mesh2robot.core.motion_segmentation import (
    assign_orphans_to_nearest_body,
    merge_duplicate_bodies,
    segment_multi_pose,
)
from mesh2robot.experiments.feasibility_xarm6 import (
    evaluate_segmentation,
    load_pose_meshes,
)
from mesh2robot.experiments.noise_sweep import _evaluate_joints, inject_noise


def main():
    pose_pts_clean, gt_labels, link_names, _ = load_pose_meshes()

    print(f"{'sigma':>7s} {'mode':>13s} {'bodies':>7s} {'acc':>7s} "
          f"{'axis':>9s} {'origin':>10s} {'joints':>8s}")
    for sigma_mm in [0.1, 0.5, 1.0, 5.0]:
        sigma_m = sigma_mm * 1e-3
        thr = max(5e-4, 4.0 * sigma_m)
        rng = np.random.default_rng(0)
        pose_pts = inject_noise(pose_pts_clean, sigma_m, rng)

        for mode in ["raw", "merge", "merge+orphan"]:
            seg = segment_multi_pose(
                pose_pts, threshold=thr, min_inliers=200,
                max_bodies=10, n_trials=300, rng_seed=0,
            )
            if mode in ("merge", "merge+orphan"):
                seg = merge_duplicate_bodies(seg, pose_pts, merge_tol=10.0 * thr)
            if mode == "merge+orphan":
                seg = assign_orphans_to_nearest_body(seg, pose_pts)
            ev = evaluate_segmentation(seg.labels, gt_labels, len(link_names))
            joints = extract_joints(seg.body_transforms) if seg.n_bodies > 0 else []
            ae, oe, nj = _evaluate_joints(joints, ev["body_to_link"], link_names)
            print(f"{sigma_mm:>7.2f} {mode:>13s} {seg.n_bodies:>7d} "
                  f"{ev['accuracy']*100:>6.2f}% {ae:>7.2f}deg "
                  f"{oe:>8.2f}mm {nj:>5d}/6")


if __name__ == "__main__":
    main()
