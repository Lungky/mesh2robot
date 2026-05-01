"""End-to-end × noise sweep.

At each noise level, run the full pipeline (Phase 2 + 3 + 4 + 5) and verify
the resulting URDF reloads into yourdfpy without errors. This answers: do
URDFs built from noisy inputs remain structurally valid?

Unlike `noise_sweep.py`, this one skips CoACD (too slow per-run) and reuses
the visual mesh for collision to keep the sweep tractable.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from yourdfpy import URDF

from mesh2robot.core.joint_extraction import extract_joints
from mesh2robot.core.motion_segmentation import (
    assign_orphans_to_nearest_body,
    merge_duplicate_bodies,
    segment_multi_pose,
)
from mesh2robot.core.physics import compute_link_inertials, split_mesh_by_labels
from mesh2robot.core.physics_defaults import make_default_template
from mesh2robot.core.urdf_assembly import AssemblyInput, assemble
from mesh2robot.experiments.feasibility_xarm6 import (
    axis_angle_error,
    evaluate_segmentation,
    ground_truth_joints,
    line_distance,
    load_pose_meshes,
)
from mesh2robot.experiments.noise_sweep import inject_noise


OUT_ROOT = Path(__file__).resolve().parents[2] / "data" / "generated"


def _run_once(sigma_mm: float, seed: int, faces: np.ndarray,
              pose_pts_clean, gt_labels, link_names) -> dict:
    sigma_m = sigma_mm * 1e-3
    thr = max(5e-4, 4.0 * sigma_m)
    merge_tol = 10.0 * thr

    rng = np.random.default_rng(seed)
    pose_pts = inject_noise(pose_pts_clean, sigma_m, rng)

    # Phases 2 + 3
    seg = segment_multi_pose(
        pose_pts, threshold=thr, min_inliers=200,
        max_bodies=10, n_trials=300, rng_seed=seed,
    )
    seg = merge_duplicate_bodies(seg, pose_pts, merge_tol=merge_tol)
    seg = assign_orphans_to_nearest_body(seg, pose_pts)
    ev = evaluate_segmentation(seg.labels, gt_labels, len(link_names))
    joints = extract_joints(seg.body_transforms)

    # Phase 4
    query_dof = sum(1 for j in joints if j.type == "revolute")
    tpl = make_default_template(query_dof)
    per_link_meshes = split_mesh_by_labels(pose_pts[0], faces, seg.labels)
    inertials = compute_link_inertials(per_link_meshes, density=tpl.density)

    # Phase 5 (collision = visual to keep sweep fast)
    body_T0 = [Ts[0] for Ts in seg.body_transforms]
    body_to_link = ev["body_to_link"]
    name_map = {b: link_names[i] for b, i in body_to_link.items()}

    out_dir = OUT_ROOT / f"sweep_sigma{sigma_mm:.2f}_seed{seed}"
    inp = AssemblyInput(
        robot_name="xarm6_scanned",
        per_link_meshes=per_link_meshes,
        per_link_collisions=per_link_meshes,
        joints=joints,
        inertials=inertials,
        template=tpl,
        body_transforms_pose0=body_T0,
        link_name_map=name_map,
    )
    urdf_path = assemble(inp, out_dir)

    # Verify
    urdf_ok = False
    fk_ok = False
    n_links = n_joints = 0
    urdf_err = ""
    try:
        r = URDF.load(str(urdf_path))
        urdf_ok = True
        n_links = len(r.link_map)
        n_joints = len(r.joint_map)
        r.update_cfg({n: 0.0 for n in r.actuated_joint_names})
        # FK from each joint's child back to its parent — URDF-relative, no
        # assumption about a global "world" frame name.
        for jn in r.actuated_joint_names:
            j = r.joint_map[jn]
            _ = r.get_transform(j.child, j.parent)
        fk_ok = True
    except Exception as e:
        urdf_err = str(e)[:120]

    # Joint error summary for context
    gts = ground_truth_joints()
    gt_by_child = {g["child"]: g for g in gts}
    axis_errs = []
    origin_errs = []
    for je in joints:
        child_idx = body_to_link.get(je.child_body, -1)
        if child_idx < 0:
            continue
        g = gt_by_child.get(link_names[child_idx])
        if g is None:
            continue
        axis_errs.append(axis_angle_error(je.axis, g["axis"]))
        origin_errs.append(line_distance(je.origin, je.axis,
                                          g["origin"], g["axis"]) * 1000.0)

    return {
        "sigma_mm": sigma_mm, "seed": seed,
        "seg_accuracy": ev["accuracy"],
        "n_bodies": seg.n_bodies,
        "n_joints": len(joints),
        "urdf_path": str(urdf_path),
        "urdf_loads": urdf_ok,
        "urdf_fk_ok": fk_ok,
        "n_links": n_links,
        "n_urdf_joints": n_joints,
        "urdf_err": urdf_err,
        "mean_axis_err_deg": float(np.mean(axis_errs)) if axis_errs else float("nan"),
        "mean_origin_err_mm": float(np.mean(origin_errs)) if origin_errs else float("nan"),
    }


def main():
    pose_pts_clean, gt_labels, link_names, _ = load_pose_meshes()
    faces = np.load(
        Path(__file__).resolve().parents[2] / "data" / "synthetic" / "xarm6" / "pose_00.npz"
    )["faces"]

    sigmas = [0.0, 0.1, 0.5, 1.0, 2.0, 5.0]   # skip 10 — known to break Phases 2+3
    seeds = [0, 1, 2]

    print(f"{'sigma_mm':>8s} {'seed':>4s} {'bodies':>7s} {'seg_acc':>8s} "
          f"{'axis':>8s} {'origin':>10s} {'urdf':>5s} {'fk':>4s}")
    rows = []
    for s in sigmas:
        for sd in seeds:
            r = _run_once(s, sd, faces, pose_pts_clean, gt_labels, link_names)
            rows.append(r)
            print(f"{r['sigma_mm']:>8.2f} {r['seed']:>4d} "
                  f"{r['n_bodies']:>7d} {r['seg_accuracy']*100:>7.2f}% "
                  f"{r['mean_axis_err_deg']:>6.3f}d "
                  f"{r['mean_origin_err_mm']:>8.3f}mm "
                  f"{'OK' if r['urdf_loads'] else 'ERR':>5s} "
                  f"{'OK' if r['urdf_fk_ok'] else 'ERR':>4s}")

    # Summary
    n_ok = sum(1 for r in rows if r["urdf_loads"] and r["urdf_fk_ok"])
    print(f"\nURDF validity: {n_ok}/{len(rows)} runs parsed + FK'd successfully")
    failures = [r for r in rows if not r["urdf_loads"]]
    if failures:
        print("Failures:")
        for r in failures:
            print(f"  sigma={r['sigma_mm']} seed={r['seed']}: {r['urdf_err']}")


if __name__ == "__main__":
    main()
