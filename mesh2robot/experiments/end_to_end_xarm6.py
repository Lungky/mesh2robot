"""End-to-end test: synthetic xArm6 meshes -> Phases 2/3/4/5 -> working URDF.

Takes the 13 synthetic pose meshes generated earlier, runs the full pipeline,
writes a URDF + meshes, then reloads the URDF with yourdfpy to verify it's
structurally valid.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh
from yourdfpy import URDF

from dataclasses import replace

from mesh2robot.core.collision import combine_hulls, convex_decompose
from mesh2robot.core.joint_extraction import extract_joints, refine_joint_origins
from mesh2robot.core.joint_limits import (
    load_yaml_overrides,
    resolve_limits,
    summarize as summarize_limits,
    sweep_self_collision_limits,
)
from mesh2robot.core.motion_segmentation import (
    assign_orphans_to_nearest_body,
    merge_duplicate_bodies,
    segment_multi_pose,
)
from mesh2robot.core.physics import compute_link_inertials, split_mesh_by_labels
from mesh2robot.core.physics_defaults import make_default_template
from mesh2robot.core.urdf_assembly import AssemblyInput, assemble
from mesh2robot.experiments.feasibility_xarm6 import (
    evaluate_segmentation,
    load_pose_meshes,
)


OUT_DIR = Path(__file__).resolve().parents[2] / "data" / "generated" / "xarm6"


def build_collisions(per_link_meshes, use_coacd: bool = True):
    """Run CoACD on each link mesh and produce a combined-hull collision mesh."""
    import time
    collisions = {}
    for b, m in sorted(per_link_meshes.items()):
        if use_coacd:
            t0 = time.time()
            hulls = convex_decompose(m, threshold=0.1, max_hulls=8,
                                      preprocess_resolution=30)
            dt = time.time() - t0
            print(f"  body {b}: {len(hulls)} hulls ({dt:.1f}s)")
            collisions[b] = combine_hulls(hulls)
        else:
            collisions[b] = m
    return collisions


def main():
    print("=" * 70)
    print("mesh2robot end-to-end test: synthetic xArm6 -> URDF")
    print("=" * 70)

    # --- Phase 2 input: synthetic pose meshes ---
    pose_pts, gt_labels, link_names, gt_link_T = load_pose_meshes()
    K, N, _ = pose_pts.shape
    # Reconstruct the combined mesh faces from pose 0
    import json
    meta_path = Path(__file__).resolve().parents[2] / "data" / "synthetic" / "xarm6" / "pose_00.npz"
    raw = np.load(meta_path)
    faces = raw["faces"]
    print(f"Input:  {K} poses x {N} vertices  (faces={len(faces)})")

    # --- Phase 2: segment ---
    print("\nPhase 2: motion segmentation ...")
    seg = segment_multi_pose(
        pose_pts, threshold=5e-4, min_inliers=200,
        max_bodies=10, n_trials=300, rng_seed=0,
    )
    seg = merge_duplicate_bodies(seg, pose_pts, merge_tol=5e-3)
    seg = assign_orphans_to_nearest_body(seg, pose_pts)
    ev = evaluate_segmentation(seg.labels, gt_labels, len(link_names))
    print(f"  bodies={seg.n_bodies}  accuracy={ev['accuracy']*100:.2f}%")

    # --- Phase 3: joints ---
    print("\nPhase 3: joint extraction ...")
    joints = extract_joints(seg.body_transforms)
    print(f"  joints={len(joints)}  ({sum(1 for j in joints if j.type=='revolute')} revolute)")

    # --- Phase 4: template match + per-link physics ---
    print("\nPhase 4: physics defaults + inertials ...")
    query_dof = sum(1 for j in joints if j.type == "revolute")
    tpl = make_default_template(query_dof)
    print(f"  density={tpl.density:.0f} kg/m^3 (fixed)")

    per_link_meshes = split_mesh_by_labels(pose_pts[0], faces, seg.labels)

    # Refine joint origins: pick physically-meaningful points on each axis.
    joints = refine_joint_origins(joints, per_link_meshes)
    inertials = compute_link_inertials(per_link_meshes, density=tpl.density)
    print(f"  inertials computed for {len(inertials)} bodies")
    for b, ine in sorted(inertials.items()):
        print(f"    body {b}: mass={ine.mass:.3f} kg")

    # --- Phase 5a: collision decomposition ---
    import os
    use_coacd = os.environ.get("USE_COACD", "1") == "1"
    print(f"\nPhase 5a: collision decomposition (CoACD={use_coacd}) ...")
    collisions = build_collisions(per_link_meshes, use_coacd=use_coacd)

    # --- Phase 5: assemble URDF ---
    print("\nPhase 5: URDF assembly ...")
    # Map body -> GT link name (using the body_to_link from evaluation) so the
    # output reads nicely.
    body_to_link = ev["body_to_link"]
    name_map = {b: link_names[link_idx] for b, link_idx in body_to_link.items()}

    # Per-body transforms at pose 0 (from segmentation output)
    body_T0 = [Ts[0] for Ts in seg.body_transforms]

    inp = AssemblyInput(
        robot_name="xarm6_scanned",
        per_link_meshes=per_link_meshes,
        per_link_collisions=collisions,
        joints=joints,
        inertials=inertials,
        template=tpl,
        body_transforms_pose0=body_T0,
        link_name_map=name_map,
    )
    urdf_path = assemble(inp, OUT_DIR)
    print(f"  wrote {urdf_path} (pass 1, template limits)")

    # --- Phase 5b: collision-aware + user-override limit refinement ---
    print("\nPhase 5b: joint-limit resolution ...")
    # Load URDF and re-read the joint names in chain order — the assembler
    # renames them canonically.
    from yourdfpy import URDF
    r = URDF.load(str(urdf_path))
    joint_names = list(r.actuated_joint_names)

    print("  sweeping self-collision envelope ...")
    collision_lims = sweep_self_collision_limits(urdf_path, step_deg=2.0)
    print(f"    found envelopes for {len(collision_lims)} joints")

    overrides_path = OUT_DIR / "overrides.yaml"
    overrides = load_yaml_overrides(overrides_path)
    if overrides:
        print(f"  loaded {len(overrides)} overrides from {overrides_path}")
    else:
        print(f"  no user overrides at {overrides_path}")

    observed = [(je.lower, je.upper) for je in joints]
    final = resolve_limits(
        joint_names=joint_names,
        template=tpl.limits_per_joint,
        observed=observed,
        collision=collision_lims,
        override=overrides,
    )
    print(summarize_limits(
        joint_names, final, tpl.limits_per_joint, collision_lims, overrides
    ))

    # --- Phase 5c: re-emit URDF with resolved limits ---
    print("\nPhase 5c: URDF re-emission with resolved limits ...")
    inp_final = replace(inp, final_limits_per_joint=final)
    urdf_path = assemble(inp_final, OUT_DIR)
    print(f"  wrote {urdf_path}")

    # --- Verify: reload the URDF ---
    print("\nVerification: reload URDF with yourdfpy ...")
    try:
        r = URDF.load(str(urdf_path))
        print(f"  OK  links={len(r.link_map)}  joints={len(r.joint_map)}  "
              f"actuated={len(r.actuated_joint_names)}")
        # Try FK at home pose
        r.update_cfg({n: 0.0 for n in r.actuated_joint_names})
        for jn in r.actuated_joint_names:
            T = r.get_transform(r.joint_map[jn].child, r.joint_map[jn].parent)
            origin = T[:3, 3]
            print(f"    {jn}  child_at_home = {origin}")
        print("\n  >>> END-TO-END PIPELINE: PASSED <<<")
    except Exception as e:
        print(f"  FAILED to reload URDF: {e}")
        print("\n  >>> END-TO-END PIPELINE: INCOMPLETE <<<")


if __name__ == "__main__":
    main()
