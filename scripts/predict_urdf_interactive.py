"""Phase D.3 — semi-automatic URDF prediction with multi-stage refinement.

Pipeline stages (each is optional; ML always runs once):

    Stage 1: User annotation (default ON)
        Open viewer with plain-grey mesh; user draws rough rectangle
        selections and labels them 0-9. Strict-mode merge means user's
        rough hints + ML compensates for human errors (see
        merge_user_and_ml_labels for the semantics).
        Skip with `--show-ml-first` (post-select instead) or `--no-gui`.

    Stage 2: ML inference (always)
        PT-V3 predicts per-vertex segmentation, per-joint axes / origins
        / types / validity. Runs exactly once.

    Stage 3: Motion-image joint refinement (opt-in via --motion-dir)
        For each joint_N/ subfolder of the motion directory, run the
        legacy Path-B image-pair RANSAC + PnP pipeline against the
        provided state*.png images and ArUco-calibrated camera. Recovers
        per-joint screw axis / origin to ~0.1° precision (vs ML's ~38°).
        These values OVERRIDE the corresponding ML joint predictions
        for the joints where motion succeeds; ML predictions are kept
        for joints with no motion data or failed extraction.

    Stage 4: URDF assembly (always)
        Build per-link sub-meshes, sort chain by Z, template-match for
        physics, write URDF + meshes.

Usage:
    # Pre-select annotation only (no motion):
    python scripts/predict_urdf_interactive.py \\
        --checkpoint data/checkpoints/model_v2_ptv3_25ep/checkpoint_epoch_025.pt \\
        --encoder ptv3 \\
        --mesh input/test_2/milo/xarm6_clean.obj \\
        --mesh-to-world input/test_2/T_cleaned_to_original.npy \\
        --output output/test_2_pred

    # Pre-select + motion images (best precision when capture rig
    # available; both the mesh-cleanup and per-joint photo pairs from
    # input/test_2/motion/joint_*):
    python scripts/predict_urdf_interactive.py \\
        --checkpoint data/checkpoints/model_v2_ptv3_25ep/checkpoint_epoch_025.pt \\
        --encoder ptv3 \\
        --mesh input/test_2/milo/xarm6_clean.obj \\
        --mesh-to-world input/test_2/T_cleaned_to_original.npy \\
        --motion-dir input/test_2/motion \\
        --camera-intrinsics input/test_2/calibration.json \\
        --output output/test_2_pred_with_motion
"""

from __future__ import annotations

import argparse
import sys
import pathlib
from pathlib import Path

import numpy as np
import torch
import trimesh

# Cross-platform checkpoint loading: checkpoints saved on Linux contain
# PosixPath objects in args; pickle can't instantiate PosixPath on Windows.
# Aliasing it to WindowsPath at unpickle time resolves it transparently.
if sys.platform == "win32":
    pathlib.PosixPath = pathlib.WindowsPath

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mesh2robot.core.joint_extraction import JointEstimate
from mesh2robot.core.physics import compute_link_inertials
from mesh2robot.core.physics_defaults import make_default_template
from mesh2robot.core.urdf_assembly import AssemblyInput, assemble
from mesh2robot.data_gen.urdf_loader import INT_TO_JOINT_TYPE
from mesh2robot.model.model import Mesh2RobotModel

# Reuse helpers from predict_urdf.py
from predict_urdf import (
    project_labels_to_faces,
    sample_mesh_to_points,
    split_mesh_by_face_labels,
)


# ---------------------------------------------------------------------------
# Refinement: re-permute ML labels using a single user anchor
# ---------------------------------------------------------------------------

def refine_labels_by_anchor_region(
    face_labels: np.ndarray,
    face_centers: np.ndarray,
    anchor_face_indices: np.ndarray,
    user_label: int,
) -> tuple[np.ndarray, dict[str, object]]:
    """Refine face labels using a user-drawn region (set of face indices).

    Two-step refinement:

      Step 1 — HARD OVERRIDE inside the user's selection:
        Every face the user selected gets `user_label`, regardless of
        what ML thought it was. This honours the user's freedom to draw
        any area without being constrained by ML's cluster boundaries.

      Step 2 — RENUMBER everything outside the selection by Z-distance:
        Compute the anchor centroid (mean position of selected faces).
        For each ML label that survives outside the selection (i.e. has
        ≥ 1 unselected face), compute its centroid and sort by distance
        from the anchor. Assign sequential new labels starting at
        user_label + 1.

    Why this design: the user has freedom to override ML's grouping
    (Step 1 handles cases where ML drew wrong boundaries — over-
    segmenting or under-segmenting). The remaining ML clusters keep
    their groupings (which are usually right in spirit, just mis-
    numbered) but get re-rooted from the user's anchor (Step 2).

    Returns
    -------
    refined : (F,) int array
    info    : dict with keys
                - 'remap' : {ml_label_outside_selection: new_label}
                - 'overridden_face_count' : int
                - 'merged_ml_labels'      : list of ML labels that the
                                            user's selection covered
                                            entirely or partially
    """
    refined = face_labels.copy()
    anchor_face_indices = np.asarray(anchor_face_indices, dtype=np.int64)
    if anchor_face_indices.size == 0:
        return refined, {"remap": {}, "overridden_face_count": 0,
                         "merged_ml_labels": []}

    # Which ML labels did the user touch?
    merged_ml = sorted(set(int(l) for l in face_labels[anchor_face_indices]
                           if l >= 0))

    # Step 1 — hard override
    refined[anchor_face_indices] = int(user_label)

    # Step 2 — renumber remaining ML clusters by Z-distance to anchor
    anchor_centroid = face_centers[anchor_face_indices].mean(axis=0)

    # Find ML labels that still have faces OUTSIDE the user's selection
    selection_mask = np.zeros(len(face_labels), dtype=bool)
    selection_mask[anchor_face_indices] = True

    surviving_labels: dict[int, np.ndarray] = {}
    for lbl in np.unique(face_labels):
        if lbl < 0:
            continue
        outside_mask = (face_labels == lbl) & (~selection_mask)
        if outside_mask.any():
            surviving_labels[int(lbl)] = face_centers[outside_mask].mean(axis=0)

    # Sort by distance from anchor centroid (ascending = closest first)
    other_labels = sorted(
        surviving_labels.keys(),
        key=lambda l: float(np.linalg.norm(
            surviving_labels[l] - anchor_centroid
        )),
    )

    # Skip the user's chosen label so we don't collide
    remap: dict[int, int] = {}
    next_label = int(user_label) + 1
    for l in other_labels:
        # If this ML label was also touched by the user (some faces moved
        # into the selection), the unselected faces still need a number,
        # but they're now a "remainder" — give them a fresh label.
        remap[l] = next_label
        next_label += 1

    # Apply remap to faces NOT in the user's selection
    for old, new in remap.items():
        mask = (face_labels == old) & (~selection_mask)
        refined[mask] = new

    return refined, {
        "remap": remap,
        "overridden_face_count": int(selection_mask.sum()),
        "merged_ml_labels": merged_ml,
    }


def merge_user_and_ml_labels(
    user_face_labels: dict[int, int],
    ml_face_labels: np.ndarray,
    face_centers: np.ndarray,
    propagation_threshold: float = 0.30,
) -> tuple[np.ndarray, dict[str, object]]:
    """Strict-mode merge: user-annotated faces are hard overrides; ML
    fills in the rest, only propagating to a whole cluster when the
    user covered enough of it to make propagation safe.

    Designed around: "user gives rough hints, ML compensates for human
    errors (overlapping selections, missed spots)."

    Behaviour
    ---------
    For each ML cluster:
      • If the user tagged > `propagation_threshold` (default 30%) of
        the cluster's faces with one dominant label, the WHOLE cluster
        gets that label. (Honours rough-but-substantial hints. e.g.
        user covered 90% of a link → remaining 10% inherits the label.)
      • Otherwise, only the exact user-tagged faces get the user label;
        the rest of the cluster keeps its ML grouping but is renumbered
        to a fresh label after `max(user_labels)` so it doesn't collide.
        (Handles accidental bleed-over: if user's rectangle catches
        5 stray faces from an adjacent ML cluster, those 5 faces flip
        but the other 995 stay together as the next link.)

    ML clusters with NO user faces at all keep their groupings and get
    renumbered into the chain by Z-distance from the user-annotated
    centroid (closest = next after user's max label).

    Returns
    -------
    refined : (F,) int array
    info    : {
        'n_user_overrides':    int
        'propagated_clusters': dict[ml_lbl, user_lbl] — clusters where
                               user coverage exceeded threshold
        'split_clusters':      dict[ml_lbl, list[user_lbl]] — clusters
                               where user only tagged a small portion;
                               the tagged faces got their user labels,
                               the un-tagged remainder got a fresh
                               renumbered label (see untouched_remap)
        'untouched_remap':     dict[ml_lbl, new_lbl] — clusters with
                               either zero user touch OR sub-threshold
                               touch; renumbered into the chain
    }
    """
    refined = ml_face_labels.copy()

    if not user_face_labels:
        return refined, {
            "n_user_overrides": 0,
            "propagated_clusters": {},
            "split_clusters": {},
            "untouched_remap": {},
        }

    user_indices = np.array(list(user_face_labels.keys()), dtype=np.int64)
    user_label_arr = np.array([user_face_labels[i] for i in user_indices])
    max_user_label = int(user_label_arr.max())
    user_centroid = face_centers[user_indices].mean(axis=0)

    # Boolean mask of which faces were user-tagged (vectorised)
    user_mask = np.zeros(len(ml_face_labels), dtype=bool)
    user_mask[user_indices] = True

    # Classify each ML cluster: propagate, split, or untouched
    propagated_clusters: dict[int, int] = {}     # ml_lbl -> user_lbl
    split_clusters: dict[int, list[int]] = {}    # ml_lbl -> user labels seen
    needs_renumber_centroid: dict[int, np.ndarray] = {}  # ml_lbl -> centroid

    from collections import Counter
    for ml_lbl in np.unique(ml_face_labels):
        if ml_lbl < 0:
            continue
        ml_lbl = int(ml_lbl)
        cluster_mask = ml_face_labels == ml_lbl
        n_cluster = int(cluster_mask.sum())
        if n_cluster == 0:
            continue

        cluster_user_mask = cluster_mask & user_mask
        n_user_in_cluster = int(cluster_user_mask.sum())

        if n_user_in_cluster == 0:
            # ML cluster the user never touched — needs renumbering.
            needs_renumber_centroid[ml_lbl] = face_centers[cluster_mask].mean(axis=0)
            continue

        # Some user touch — see if it's strong enough to propagate.
        cluster_user_indices = np.where(cluster_user_mask)[0]
        votes = [user_face_labels[int(i)] for i in cluster_user_indices]
        top_label, top_count = Counter(votes).most_common(1)[0]
        coverage = top_count / n_cluster

        if coverage > propagation_threshold:
            propagated_clusters[ml_lbl] = int(top_label)
        else:
            split_clusters[ml_lbl] = sorted(set(votes))
            # The un-tagged faces of this cluster need renumbering too.
            unsel_mask = cluster_mask & ~user_mask
            if unsel_mask.any():
                needs_renumber_centroid[ml_lbl] = face_centers[unsel_mask].mean(axis=0)

    # Renumber the "needs renumber" clusters by Z-distance from user centroid
    sorted_untouched = sorted(
        needs_renumber_centroid.keys(),
        key=lambda l: float(np.linalg.norm(
            needs_renumber_centroid[l] - user_centroid
        )),
    )
    untouched_remap: dict[int, int] = {}
    next_label = max_user_label + 1
    for ml_lbl in sorted_untouched:
        untouched_remap[ml_lbl] = next_label
        next_label += 1

    # Build the refined array:
    # 1. Start by carrying over ML labels.
    refined = ml_face_labels.copy()

    # 2. Apply propagation: full cluster → user_lbl (touched + untouched faces).
    for ml_lbl, user_lbl in propagated_clusters.items():
        cluster_mask = ml_face_labels == ml_lbl
        refined[cluster_mask] = user_lbl

    # 3. Apply renumbering: un-tagged faces of split / untouched clusters
    #    get the fresh chain-ordered label.
    for ml_lbl, new_lbl in untouched_remap.items():
        cluster_mask = ml_face_labels == ml_lbl
        # In split clusters, only un-user-tagged faces get the renumber.
        # In fully-untouched clusters, all faces get the renumber.
        target_mask = cluster_mask & ~user_mask
        refined[target_mask] = new_lbl

    # 4. Hard override on user-tagged faces, ALWAYS — wins over any prior step.
    for face_idx, user_lbl in user_face_labels.items():
        refined[int(face_idx)] = int(user_lbl)

    return refined, {
        "n_user_overrides": len(user_face_labels),
        "propagated_clusters": propagated_clusters,
        "split_clusters": split_clusters,
        "untouched_remap": untouched_remap,
        "propagation_threshold": propagation_threshold,
    }


# Backwards-compatible single-click wrapper (used internally for the
# 'click → use ML cluster' fallback when rect drag isn't available).
def refine_labels_by_anchor(
    face_labels: np.ndarray,
    face_centers: np.ndarray,
    anchor_face_idx: int,
    user_label: int,
) -> tuple[np.ndarray, dict[int, int]]:
    """Single-click path: pick the ML cluster of the clicked face,
    then call refine_labels_by_anchor_region with all faces in that cluster.
    Kept for backwards compat with the unit test."""
    ml_anchor_label = int(face_labels[anchor_face_idx])
    selection = np.where(face_labels == ml_anchor_label)[0]
    refined, info = refine_labels_by_anchor_region(
        face_labels, face_centers, selection, user_label,
    )
    # Reconstruct legacy remap that includes the anchor's own ML label
    remap: dict[int, int] = {ml_anchor_label: int(user_label)}
    remap.update(info["remap"])
    return refined, remap


# ---------------------------------------------------------------------------
# URDF builder (extracted from predict_urdf.py main, parameterised on
# face_labels so we can call it twice — once with original, once refined)
# ---------------------------------------------------------------------------

def build_urdf_from_predictions(
    mesh: trimesh.Trimesh,
    face_labels: np.ndarray,
    pred_axes: np.ndarray,
    pred_origins_world: np.ndarray,
    pred_valid: np.ndarray,
    pred_types: np.ndarray,
    output_dir: Path,
    robot_name: str = "ai_predicted",
    motion_overrides: dict[int, tuple[np.ndarray, np.ndarray]] | None = None,
    use_geometric_joints: bool = False,
    pred_limits: np.ndarray | None = None,
    collision_sweep: bool = False,
    sweep_steps: int = 64,
    topology_mode: str = "serial",
    expected_link_count: int | None = None,
    cleanup_clusters: bool = True,
) -> Path | None:
    """Build a URDF from face labels + joint predictions. Mirrors
    `predict_urdf.py`'s steps 4–8.

    `motion_overrides`, if provided, maps `chain_i` to (axis, origin)
    tuples derived from motion-image RANSAC.

    `use_geometric_joints`: if True (Phase D.4), every joint's
    (axis, origin, type) comes from the boundary geometry between
    adjacent link meshes (3D circle fit) instead of the ML head's
    output. This sidesteps the model's ~38° axis_deg val error and
    ML's pose-sensitive joint predictions.

    `pred_limits`: optional (J_MAX, 2) array of model-predicted
    (lower, upper) per joint slot. When the checkpoint includes a
    trained LimitsHead, these are used directly instead of the ±π
    fallback. The same chain-order projection used for axes/origins
    applies (slot j_slot → chain_i via valid_idx_by_z).

    `collision_sweep`: if True, after writing the URDF, run the
    PyBullet self-collision sweep (`mesh2robot.core.collision_sweep`)
    against the just-written URDF to refine each joint's limits to
    the largest collision-free interval around the home pose, then
    rewrite the URDF with those refined limits.

    Precedence: geometric > motion > ML for axes/origins/types.
    Limits source order: model > ±π fallback. Sweep refines whatever
    limits ended up emitted."""
    # ── Connected-component cleanup ───────────────────────────────────
    # Each label can span disconnected fragments scattered across the
    # mesh ("one link is a bunch of random shards" failure mode).
    # Split each label into its connected components: largest keeps the
    # label, sizable detached components get a fresh label, tiny stray
    # fragments are absorbed into their dominant neighbour.
    if cleanup_clusters:
        from mesh2robot.core.topology import (
            clean_disconnected_clusters, cleanup_summary,
        )
        cleaned = clean_disconnected_clusters(
            mesh, face_labels, min_component_faces=30,
        )
        if not np.array_equal(cleaned, face_labels):
            print(cleanup_summary(face_labels, cleaned))
            face_labels = cleaned

    # ── VLM-prior cluster pruning ─────────────────────────────────────
    # If the VLM said "this robot has N links", drop the smallest
    # clusters until we're at most expected_link_count + 2. The +2
    # buffer lets the strict-mode merge absorb any noise. We do this
    # by bumping `min_faces` adaptively, which is what
    # split_mesh_by_face_labels uses to filter.
    pruning_floor = 30
    if expected_link_count is not None and expected_link_count > 0:
        from collections import Counter
        # Only count clusters that already pass the basic floor — anything
        # smaller is out anyway.
        counts = sorted(
            (n for lid, n in Counter(face_labels.tolist()).items()
             if int(lid) >= 0 and n >= pruning_floor),
            reverse=True,
        )
        target = expected_link_count + 2
        if len(counts) > target:
            # Keep top `target` clusters: bump min_faces above the
            # (target+1)-th largest cluster's size.
            pruning_floor = max(pruning_floor, counts[target] + 1)
            print(f"  VLM-prune: {len(counts)} ML clusters → keeping top "
                  f"{target} (expected_link_count={expected_link_count} + 2 "
                  f"buffer); min_faces bumped to {pruning_floor}.")

    per_link_meshes = split_mesh_by_face_labels(mesh, face_labels,
                                                 min_faces=pruning_floor)
    if len(per_link_meshes) < 2:
        print(f"  Need >= 2 links to assemble; got {len(per_link_meshes)}. Skipping.")
        return None

    # ──────────────────────────────────────────────────────────────────
    # Chain / topology decision (Phase E.2)
    # ──────────────────────────────────────────────────────────────────
    # Three modes:
    #   serial — Z-sort links, emit a single chain (legacy behaviour;
    #            correct for industrial arms).
    #   tree   — infer parent-child via face-adjacency BFS from a
    #            largest-mesh root (correct for humanoids, quadrupeds).
    #   auto   — infer tree, but if the result is a path, fall back to
    #            serial Z-sort for stable chain ordering.
    # Resulting structures (used downstream):
    #   link_ids_in_chain_or_bfs : ordered list of link IDs (drives body_id)
    #   pair_list                : list of (parent_lbl, child_lbl) tuples
    if topology_mode in ("tree", "auto"):
        from mesh2robot.core.topology import infer_topology_auto
        # In `auto`, prefer lowest-Z root: for serial robots this matches the
        # legacy serial-mode chain (base → tip), so a chain stays a chain.
        # In `tree`, prefer largest-mesh root: for humanoids/quadrupeds the
        # torso is the natural branching point.
        prefer_z = (topology_mode == "auto")
        topo = infer_topology_auto(
            mesh, face_labels, per_link_meshes,
            prefer_lowest_z_root=prefer_z,
        )
        chain_order = topo.chain_order()
        if topology_mode == "auto" and chain_order is not None:
            print(f"  Topology=auto → graph is serial; using Z-sorted chain")
            link_ids_in_chain_or_bfs = sorted(
                per_link_meshes.keys(),
                key=lambda lid: float(per_link_meshes[lid].centroid[2]),
            )
            pair_list = [
                (link_ids_in_chain_or_bfs[i], link_ids_in_chain_or_bfs[i + 1])
                for i in range(len(link_ids_in_chain_or_bfs) - 1)
            ]
        else:
            print(f"  Topology=tree, root=link_{topo.root}, "
                  f"{topo.n_joints} joints inferred from face adjacency")
            print(topo)
            # BFS-order the links so parent body_id < child body_id (URDF
            # convention). Root first.
            link_ids_in_chain_or_bfs = [topo.root]
            queue = [topo.root]
            while queue:
                cur = queue.pop(0)
                for ch in topo.children_of.get(cur, []):
                    link_ids_in_chain_or_bfs.append(ch)
                    queue.append(ch)
            pair_list = [
                (parent, child) for child, parent in topo.parent_of.items()
            ]
    else:
        # serial (legacy)
        link_ids_in_chain_or_bfs = sorted(
            per_link_meshes.keys(),
            key=lambda lid: float(per_link_meshes[lid].centroid[2]),
        )
        print(f"  Links ordered by Z (low → high): {link_ids_in_chain_or_bfs}")
        pair_list = [
            (link_ids_in_chain_or_bfs[i], link_ids_in_chain_or_bfs[i + 1])
            for i in range(len(link_ids_in_chain_or_bfs) - 1)
        ]

    # Backwards-compat alias used by old code paths below.
    link_ids_in_order = link_ids_in_chain_or_bfs

    # Phase D.4 — geometric joints (now keyed by (parent, child) pair)
    geometric_joints_by_pair: dict[tuple[int, int], object] | None = None
    if use_geometric_joints:
        try:
            from mesh2robot.core.geometric_joints import (
                extract_joints_for_tree,
            )
            joints_list = extract_joints_for_tree(
                mesh, face_labels, pair_list,
            )
            geometric_joints_by_pair = {
                (j.parent_label, j.child_label): j for j in joints_list
            }
            n_rev = sum(1 for j in joints_list if j.type == "revolute")
            n_fixed = sum(1 for j in joints_list if j.type == "fixed")
            print(f"  GEOMETRIC JOINTS: {len(joints_list)} fitted "
                  f"({n_rev} revolute, {n_fixed} fixed)")
        except Exception as e:
            print(f"  (geometric joint extraction failed: {e}; falling back to ML)")
            geometric_joints_by_pair = None

    valid_idx = np.where(pred_valid)[0]
    valid_idx_by_z = sorted(
        valid_idx, key=lambda j: float(pred_origins_world[j, 2]),
    )
    n_joints_needed = len(pair_list)
    valid_idx_by_z = valid_idx_by_z[:n_joints_needed]

    body_id_of_link = {lid: i for i, lid in enumerate(link_ids_in_order)}
    je_list: list[JointEstimate] = []
    # Per-joint model-predicted limits (or None to fall through to ±π).
    chain_limits: list[tuple[float, float] | None] = []
    for chain_i, (parent_lbl, child_lbl) in enumerate(pair_list):
        # Source priority: geometric > motion > ML
        gj = (geometric_joints_by_pair or {}).get((parent_lbl, child_lbl))
        if gj is not None:
            axis_use = np.asarray(gj.axis, dtype=np.float64)
            origin_use = np.asarray(gj.origin, dtype=np.float64)
            jt = gj.type
            print(f"  joint_{chain_i+1} (link_{parent_lbl}→link_{child_lbl}): "
                  f"GEOMETRIC  type={jt}  "
                  f"axis={np.round(axis_use, 3).tolist()}  "
                  f"origin={np.round(origin_use, 3).tolist()}  "
                  f"(conf={gj.confidence:.2f}, r={gj.radius:.3f}m, "
                  f"resid={gj.plane_residual:.4f}m)")
        elif motion_overrides is not None and chain_i in motion_overrides:
            axis_use, origin_use = motion_overrides[chain_i]
            axis_use = np.asarray(axis_use, dtype=np.float64)
            origin_use = np.asarray(origin_use, dtype=np.float64)
            jt = "revolute"  # motion stage doesn't classify type
            print(f"  joint_{chain_i+1}: MOTION OVERRIDE  "
                  f"axis={np.round(axis_use, 3).tolist()}  "
                  f"origin={np.round(origin_use, 3).tolist()}")
        else:
            # ML fallback
            if chain_i >= len(valid_idx_by_z):
                # Out of valid ML joints — emit a fixed joint at link interface
                print(f"  joint_{chain_i+1}: no ML valid joint slot; emitting fixed")
                axis_use = np.array([0.0, 0.0, 1.0])
                origin_use = np.zeros(3)
                jt = "fixed"
            else:
                j_slot = valid_idx_by_z[chain_i]
                jt_int = int(pred_types[j_slot])
                jt = INT_TO_JOINT_TYPE.get(jt_int, "revolute")
                axis_use = pred_axes[j_slot].copy()
                origin_use = pred_origins_world[j_slot].copy()
        # Limits: prefer model-predicted (when LimitsHead is trained);
        # fall back to ±π. The model's ML slot is `j_slot` if we landed
        # in the ML branch, else we use the chain_i slot directly (the
        # geometric/motion branches don't observe ML's slot indexing).
        slot_for_limits = None
        if pred_limits is not None and pred_limits.shape[0] > 0:
            if (gj is None and motion_overrides is None
                    and chain_i < len(valid_idx_by_z)):
                slot_for_limits = int(valid_idx_by_z[chain_i])
            elif chain_i < pred_limits.shape[0]:
                slot_for_limits = chain_i
        lower_use = -float(np.pi)
        upper_use = float(np.pi)
        if slot_for_limits is not None:
            lo, hi = float(pred_limits[slot_for_limits, 0]), \
                     float(pred_limits[slot_for_limits, 1])
            # Sanity: ensure ordered + within physical caps for the type
            if hi > lo:
                lower_use = lo
                upper_use = hi
        chain_limits.append((lower_use, upper_use))
        je_list.append(JointEstimate(
            parent_body=body_id_of_link[parent_lbl],
            child_body=body_id_of_link[child_lbl],
            type=jt,
            axis=axis_use,
            origin=origin_use,
            angles=[0.0, 0.0],
            lower=lower_use,
            upper=upper_use,
        ))

    per_link_meshes_by_body = {
        body_id_of_link[lid]: per_link_meshes[lid]
        for lid in link_ids_in_order
    }
    per_link_collisions_by_body = dict(per_link_meshes_by_body)

    dof = len(je_list)
    tpl = make_default_template(dof)
    inertials = compute_link_inertials(per_link_meshes_by_body, density=tpl.density)

    n_bodies = len(link_ids_in_order)
    link_name_map = {0: "link_base"}
    for i in range(1, n_bodies - 1):
        link_name_map[i] = f"link_{i}"
    if n_bodies >= 2:
        link_name_map[n_bodies - 1] = "link_tip"

    inp = AssemblyInput(
        robot_name=robot_name,
        per_link_meshes=per_link_meshes_by_body,
        per_link_collisions=per_link_collisions_by_body,
        joints=je_list,
        inertials=inertials,
        template=tpl,
        body_transforms_pose0=[np.eye(4)] * n_bodies,
        link_name_map=link_name_map,
        final_limits_per_joint=chain_limits,
    )
    urdf_path = assemble(inp, output_dir)

    # Persist body_id → original-cluster-id mapping so downstream tools
    # (e.g. VLM critic auto-fix) can translate URDF-side link indices
    # back to the face-label cluster IDs they were derived from.
    if urdf_path is not None:
        import json as _json
        mapping_path = output_dir / "body_to_cluster.json"
        mapping_path.write_text(_json.dumps({
            "link_ids_in_order": [int(x) for x in link_ids_in_order],
            "link_name_map": {str(k): v for k, v in link_name_map.items()},
        }, indent=2))

    # Tier-3 sweep refinement, opt-in. Loads the just-written URDF in
    # PyBullet, sweeps each joint, returns refined (lo, hi). Then we
    # re-assemble with the refined limits replacing the model prior.
    if collision_sweep and urdf_path is not None and len(je_list) > 0:
        try:
            from mesh2robot.core.collision_sweep import sweep_collision_free
            print("  Running PyBullet collision sweep on initial URDF ...")
            refined = sweep_collision_free(
                urdf_path,
                priors=chain_limits,
                joint_types=[j.type for j in je_list],
                n_steps=sweep_steps,
                verbose=True,
            )
            inp_refined = AssemblyInput(
                robot_name=robot_name,
                per_link_meshes=per_link_meshes_by_body,
                per_link_collisions=per_link_collisions_by_body,
                joints=je_list,
                inertials=inertials,
                template=tpl,
                body_transforms_pose0=[np.eye(4)] * n_bodies,
                link_name_map=link_name_map,
                final_limits_per_joint=refined,
            )
            urdf_path = assemble(inp_refined, output_dir)
            print(f"  Re-wrote URDF with sweep-refined limits.")
        except Exception as e:
            print(f"  (collision sweep failed: {e}; keeping model-prior URDF)")

    return urdf_path


# ---------------------------------------------------------------------------
# Stage 3 — Motion-image refinement (legacy Path B, opt-in)
# ---------------------------------------------------------------------------

def _load_camera_intrinsics(json_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read fx/fy/cx/cy + dist_coeffs from a calibration JSON."""
    import json
    d = json.loads(json_path.read_text())
    K = np.array([
        [d["fx"], 0.0, d["cx"]],
        [0.0, d["fy"], d["cy"]],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)
    dist = np.array(d.get("dist_coeffs", [0.0] * 5), dtype=np.float64).flatten()
    if dist.size < 5:
        dist = np.concatenate([dist, np.zeros(5 - dist.size)])
    return K, dist


def extract_motion_overrides(
    mesh: trimesh.Trimesh,
    motion_dir: Path,
    camera_intrinsics_path: Path,
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Walk `motion/joint_<N>/state*.png` and run the legacy image-pair
    RANSAC pipeline to recover per-joint screw axis + origin.

    Returns a dict `{chain_i (0-indexed): (axis, origin)}`. Joint folders
    are 1-indexed by convention, so `joint_1/` maps to chain_i=0 (the
    base joint), `joint_2/` to chain_i=1, etc.

    Joints whose motion extraction fails (no ArUco detection, too few
    feature matches, etc.) are silently omitted — the caller will fall
    back to the ML prediction for those.
    """
    import re
    from mesh2robot.core.motion_from_images import extract_joint_motion_multi

    K, dist = _load_camera_intrinsics(camera_intrinsics_path)
    print(f"  Camera intrinsics: fx={K[0,0]:.1f} fy={K[1,1]:.1f} "
          f"cx={K[0,2]:.1f} cy={K[1,2]:.1f}")

    overrides: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    joint_dirs = sorted(p for p in motion_dir.iterdir() if p.is_dir())
    if not joint_dirs:
        print(f"  (no joint subdirectories under {motion_dir})")
        return overrides

    for jd in joint_dirs:
        m = re.match(r"^joint_(\d+)$", jd.name)
        if not m:
            continue
        chain_i = int(m.group(1)) - 1   # joint_1 → chain index 0
        states = sorted(jd.glob("state*.png")) + sorted(jd.glob("state*.jpg"))
        if len(states) < 2:
            print(f"  {jd.name}: only {len(states)} state images; need ≥2 — skipped")
            continue

        print(f"  {jd.name}: running image-pair RANSAC over "
              f"{len(states)} states...")
        try:
            result = extract_joint_motion_multi(
                mesh=mesh,
                state_paths=[str(s) for s in states],
                camera_matrix=K,
                dist_coeffs=dist,
                joint_name=jd.name,
            )
        except Exception as e:
            print(f"    failed: {type(e).__name__}: {e}")
            continue

        if result is None:
            print(f"    no moving body recovered (ArUco / feature lift failed)")
            continue
        if result.n_pairs_ok == 0:
            print(f"    all pairs returned static (calibration sanity);"
                  f" no override emitted")
            continue

        axis = np.asarray(result.axis_world, dtype=np.float64)
        axis /= (np.linalg.norm(axis) + 1e-12)
        origin = np.asarray(result.origin_world, dtype=np.float64)
        overrides[chain_i] = (axis, origin)
        print(f"    OK: chain_i={chain_i}  axis={np.round(axis, 3).tolist()}  "
              f"origin={np.round(origin, 3).tolist()}  "
              f"{result.n_pairs_ok} pairs OK, "
              f"axis_spread={result.axis_spread_deg:.1f}°")

    return overrides


# ---------------------------------------------------------------------------
# PyVista interactive viewer
# ---------------------------------------------------------------------------

def _make_palette(n: int = 64, seed: int = 42) -> np.ndarray:
    """Distinct RGB colour per link (matches predict_on_mesh.py palette)."""
    rng = np.random.default_rng(seed)
    hues = (np.arange(n) * 0.61803398875) % 1.0
    sats = 0.7 + rng.uniform(-0.1, 0.1, n)
    vals = 0.85 + rng.uniform(-0.1, 0.05, n)

    def hsv_to_rgb(h, s, v):
        i = int(h * 6.0)
        f = h * 6.0 - i
        p, q, t = v * (1 - s), v * (1 - s * f), v * (1 - s * (1 - f))
        i %= 6
        return [(v, t, p), (q, v, p), (p, v, t),
                (p, q, v), (t, p, v), (v, p, q)][i]

    out = np.zeros((n, 3))
    for i in range(n):
        out[i] = hsv_to_rgb(hues[i], sats[i], vals[i])
    return out


def run_pre_selection(
    mesh: trimesh.Trimesh,
    annotations_path: Path | None = None,
) -> dict[int, int]:
    """Open the mesh viewer BEFORE ML runs so the user can annotate
    cold (no ML colour cues, just the mesh shaded by surface normals).

    Returns a dict {face_idx: user_label} of all annotations the user
    committed via 'S'. Empty dict if the user pressed 'Q' or didn't
    annotate.
    """
    try:
        import pyvista as pv
    except ImportError:
        raise SystemExit(
            "pyvista is required for interactive refinement. "
            "Install with `pip install pyvista`."
        )

    palette = _make_palette()

    verts = np.asarray(mesh.vertices, dtype=np.float32)
    faces_tri = np.asarray(mesh.faces, dtype=np.int64)
    n_faces = len(faces_tri)
    flat_faces = np.column_stack(
        [np.full(n_faces, 3, dtype=np.int64), faces_tri]
    ).reshape(-1)
    pv_mesh = pv.PolyData(verts, flat_faces)

    # Initial colour: light gray with shaded normals — gives visual depth
    # without biasing the user toward an ML-suggested segmentation.
    base_rgb = np.full((n_faces, 3), [0.78, 0.80, 0.85], dtype=np.float32)
    pv_mesh.cell_data["rgb"] = base_rgb

    plotter = pv.Plotter(window_size=(1100, 850))
    plotter.set_background("white")
    plotter.add_mesh(
        pv_mesh, scalars="rgb", rgb=True,
        show_edges=False, smooth_shading=True,
    )

    state = {
        "user_annotations": {},     # face_idx -> user_label
        "pending_face_indices": None,
    }

    instr_actor = [None]
    status_actor = [None]

    def show_instructions(msg: str) -> None:
        if instr_actor[0] is not None:
            try:
                plotter.remove_actor(instr_actor[0])
            except Exception:
                pass
        instr_actor[0] = plotter.add_text(
            msg, position="upper_left", font_size=10, color="black",
        )

    def show_status(msg: str | None) -> None:
        """Big status banner — re-asserted on every event-loop tick for
        reliability against the actor-clearing VTK quirk."""
        if status_actor[0] is not None:
            try:
                plotter.remove_actor(status_actor[0])
            except Exception:
                pass
            status_actor[0] = None
        state["status_msg"] = msg
        if msg:
            status_actor[0] = plotter.add_text(
                msg, position="upper_right", font_size=18, color="red",
            )

    show_instructions(
        "PRE-SELECT MODE [Rect] — Tab cycles Rect → Lasso → Camera → Rect.\n"
        "  • RECT (now): LEFT drag = rectangle select (red box).\n"
        "  • LASSO: LEFT drag = freeform polygon trace.\n"
        "  • CAMERA: LEFT drag rotates, scroll zooms.\n"
        "  • Digit 0-9 = label highlighted area.\n"
        "  • SPACE / Enter = commit + run ML.\n"
        "  • ESC = skip annotation."
    )

    def colour_for_user(annotations: dict[int, int]) -> np.ndarray:
        rgb = np.full((n_faces, 3), [0.78, 0.80, 0.85], dtype=np.float32)
        for face_idx, lbl in annotations.items():
            rgb[face_idx] = palette[int(lbl) % len(palette)]
        return rgb

    def on_rect_pick(picked):
        if picked is None or picked.n_cells == 0:
            print("  (empty selection — try a larger rectangle)")
            return
        # PyVista renamed the field; try both names.
        ids = picked.cell_data.get("vtkOriginalCellIds")
        if ids is None:
            ids = picked.cell_data.get("orig_extract_id")
        if ids is None:
            ids = picked.cell_data.get("original_cell_ids")
        if ids is None or len(ids) == 0:
            picked_centers = picked.cell_centers().points
            mesh_centers = pv_mesh.cell_centers().points
            from scipy.spatial import cKDTree
            tree = cKDTree(mesh_centers)
            _, ids = tree.query(picked_centers, k=1)
        ids = np.unique(np.asarray(ids, dtype=np.int64))
        state["pending_face_indices"] = ids
        show_status(f"✓ SELECTED {len(ids)} FACES\nPRESS DIGIT 0-9 TO LABEL")
        show_instructions(
            f"Selected {len(ids)} faces — pending label.\n"
            "Press a digit 0-9 to label this region."
        )
        print(f"  Selected {len(ids)} faces (pending label).")

    plotter.enable_cell_picking(
        callback=on_rect_pick,
        through=True,
        show=True,
        start=True,
        style="wireframe",
        color="red",
        line_width=4,
        show_message=False,
        font_size=14,
    )

    # Auto-frame the camera on the mesh from a sensible angle.
    plotter.camera_position = "iso"
    plotter.reset_camera()

    state["mode"] = "rect"  # rect | lasso | camera

    # Pre-compute face centers in WORLD coords (used for lasso projection).
    face_centers_world = np.asarray(mesh.vertices)[mesh.faces].mean(axis=1)

    def _enter_rect_mode():
        plotter.disable_picking()
        plotter.enable_cell_picking(
            callback=on_rect_pick, through=True, show=True, start=True,
            style="wireframe", color="red", line_width=4,
            show_message=False, font_size=14,
        )
        state["mode"] = "rect"
        show_instructions(
            "RECT-PICK — LEFT drag = rectangle select (red highlight).\n"
            "Digit 0-9 = label. Tab cycles: Rect → Lasso → Camera → Rect.")
        print("  [mode = RECT]")

    def _enter_camera_mode():
        plotter.disable_picking()
        plotter.enable_trackball_style()
        state["mode"] = "camera"
        show_instructions(
            "CAMERA MODE — drag rotates, scroll zooms.\n"
            "Tab cycles: Rect → Lasso → Camera → Rect.")
        print("  [mode = CAMERA]")

    def _project_face_centers_to_screen():
        """Project all 3D face centers to 2D pixel coordinates of the
        current viewport. Returns (n_faces, 2) array."""
        import vtk
        coord = vtk.vtkCoordinate()
        coord.SetCoordinateSystemToWorld()
        renderer = plotter.renderer
        screen = np.empty((len(face_centers_world), 2), dtype=np.float64)
        for i, c in enumerate(face_centers_world):
            coord.SetValue(float(c[0]), float(c[1]), float(c[2]))
            sx, sy = coord.GetComputedDisplayValue(renderer)
            screen[i] = (sx, sy)
        return screen

    def _show_lasso_highlight(face_ids: np.ndarray) -> None:
        """Add a separate VTK actor (bypassing PyVista's add_mesh) showing
        the selected faces as a red wireframe. PyVista's add_mesh seems
        to remove our actor on subsequent renders; raw VTK insertion
        keeps it stuck in the renderer's actor list."""
        try:
            import vtk
            # Remove any previous lasso highlight actor (raw VTK)
            old = state.get("lasso_vtk_actor")
            if old is not None:
                try:
                    plotter.renderer.RemoveActor(old)
                except Exception:
                    pass
                state.pop("lasso_vtk_actor", None)

            sub_faces = faces_tri[face_ids]
            used = np.unique(sub_faces)
            if len(used) == 0:
                return
            remap = -np.ones(len(verts), dtype=np.int64)
            remap[used] = np.arange(len(used))
            sub_verts = verts[used]
            sub_faces_remap = remap[sub_faces]

            # Build vtkPolyData manually
            vtk_pts = vtk.vtkPoints()
            vtk_pts.SetNumberOfPoints(len(sub_verts))
            for i, v in enumerate(sub_verts):
                vtk_pts.SetPoint(i, float(v[0]), float(v[1]), float(v[2]))
            vtk_polys = vtk.vtkCellArray()
            for tri in sub_faces_remap:
                vtk_polys.InsertNextCell(3)
                for v in tri:
                    vtk_polys.InsertCellPoint(int(v))
            poly = vtk.vtkPolyData()
            poly.SetPoints(vtk_pts)
            poly.SetPolys(vtk_polys)

            mapper = vtk.vtkDataSetMapper()
            mapper.SetInputData(poly)
            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            prop = actor.GetProperty()
            prop.SetColor(1.0, 0.0, 0.0)
            prop.SetRepresentationToWireframe()
            prop.SetLineWidth(3)
            prop.SetLighting(False)
            actor.SetPickable(False)
            # Render order: ensure on top so it's visible
            actor.SetForceOpaque(True)

            plotter.renderer.AddActor(actor)
            state["lasso_vtk_actor"] = actor
            plotter.render()
        except Exception as e:
            print(f"  (lasso highlight render failed: {e})")

    def _restore_pending_highlight():
        """Undo the red recolor (if any), restoring whatever was there before
        — either the gray base color or the previously assigned user color."""
        if "pending_highlight_indices" not in state:
            return
        try:
            ids = state["pending_highlight_indices"]
            orig = state.get("pending_highlight_orig_rgb")
            rgb = pv_mesh.cell_data["rgb"]
            if orig is not None and len(ids) == len(orig):
                rgb[ids] = orig
                pv_mesh.cell_data["rgb"] = rgb
                pv_mesh.Modified()
        except Exception:
            pass
        finally:
            state.pop("pending_highlight_indices", None)
            state.pop("pending_highlight_orig_rgb", None)

    def _process_lasso_polygon(poly_2d: np.ndarray) -> None:
        """Given a polygon in pixel coords, select faces whose centers
        project inside it."""
        from matplotlib.path import Path as MplPath
        try:
            screen_pts = _project_face_centers_to_screen()
        except Exception as e:
            print(f"  (lasso projection failed: {e})")
            return
        poly = MplPath(poly_2d)
        inside = poly.contains_points(screen_pts)
        ids = np.where(inside)[0]
        if len(ids) == 0:
            print("  (lasso enclosed 0 face centers — try a bigger trace)")
            return
        state["pending_face_indices"] = ids
        print(f"  Lasso-selected {len(ids)} faces (pending label).")
        _show_lasso_highlight(ids)
        show_status(f"✓ SELECTED {len(ids)} FACES\nPRESS DIGIT 0-9 TO LABEL")
        show_instructions(
            f"Lasso selected {len(ids)} faces — pending label.\n"
            "Press a digit 0-9 to label this region.\n"
            "Tab cycles: Rect → Lasso → Camera → Rect.")

    def _enter_lasso_mode():
        """Freehand lasso: vtkInteractorStyleDrawPolygon draws the visible
        trace; our parallel observers capture the mouse positions to
        compute the actual selection (the style's GetPolygonPoints isn't
        wrapped to Python in this VTK version, so we have to track ourselves)."""
        import vtk
        plotter.disable_picking()
        # DrawPolygon style → renders the visible line trace as user drags.
        style = vtk.vtkInteractorStyleDrawPolygon()
        # Force the polygon trace to draw as raster pixels (otherwise some
        # VTK builds don't render it visibly).
        try:
            style.DrawPolygonPixelsOn()
        except Exception:
            pass
        # Resolve underlying VTK interactor robustly across PyVista versions.
        iren_wrapper = plotter.iren
        vtk_iren = getattr(iren_wrapper, "interactor", None)
        if vtk_iren is None:
            vtk_iren = getattr(iren_wrapper, "_interactor", None)
        if vtk_iren is None:
            vtk_iren = iren_wrapper
        vtk_iren.SetInteractorStyle(style)
        state["lasso_style"] = style
        state["lasso_iren"] = vtk_iren

        lasso_track = {"active": False, "points": []}

        def on_left_press(caller, ev):
            x, y = vtk_iren.GetEventPosition()
            lasso_track["active"] = True
            lasso_track["points"] = [(x, y)]
            print(f"  [lasso] press @ ({x},{y})")

        def on_mouse_move(caller, ev):
            if not lasso_track["active"]:
                return
            x, y = vtk_iren.GetEventPosition()
            last_x, last_y = lasso_track["points"][-1]
            if (x - last_x) ** 2 + (y - last_y) ** 2 >= 9:
                lasso_track["points"].append((x, y))

        def on_left_release(caller, ev):
            if not lasso_track["active"]:
                return
            lasso_track["active"] = False
            pts = lasso_track["points"]
            print(f"  [lasso] release with {len(pts)} polygon points")
            if len(pts) < 3:
                print("  (lasso too short — need at least 3 points)")
                return
            poly_2d = np.array(pts + [pts[0]], dtype=np.float64)
            _process_lasso_polygon(poly_2d)

        # Generic key-press logger so we know if digits/space/tab fire in this mode.
        def on_any_key(caller, ev):
            ks = vtk_iren.GetKeySym()
            print(f"  [key in LASSO] keysym={ks}")

        ids_obs = []
        # Priority 1.0 fires BEFORE the style (priority 0.0) — we capture
        # the mouse position cleanly before the style does its visual work.
        ids_obs.append(vtk_iren.AddObserver("LeftButtonPressEvent",
                                             on_left_press, 1.0))
        ids_obs.append(vtk_iren.AddObserver("MouseMoveEvent",
                                             on_mouse_move, 1.0))
        ids_obs.append(vtk_iren.AddObserver("LeftButtonReleaseEvent",
                                             on_left_release, 1.0))
        ids_obs.append(vtk_iren.AddObserver("KeyPressEvent",
                                             on_any_key, 1.0))
        state["lasso_observer_ids"] = ids_obs

        state["mode"] = "lasso"
        show_instructions(
            "LASSO MODE — LEFT drag traces a polygon (visible line).\n"
            "Release to finalize.\n"
            "Press a digit 0-9 IMMEDIATELY after release to label.\n"
            "Tab cycles: Rect → Lasso → Camera → Rect.")
        print("  [mode = LASSO]  (style: DrawPolygon — visual trace enabled)")

    def _leave_lasso_mode():
        """Cleanup: remove our manual mouse observers."""
        if "lasso_observer_ids" in state and "lasso_iren" in state:
            vtk_iren = state["lasso_iren"]
            for oid in state["lasso_observer_ids"]:
                try:
                    vtk_iren.RemoveObserver(oid)
                except Exception:
                    pass
            state.pop("lasso_observer_ids", None)
            state.pop("lasso_iren", None)
            state.pop("lasso_style", None)

    def cycle_mode():
        if state["mode"] == "rect":
            _enter_lasso_mode()
        elif state["mode"] == "lasso":
            _leave_lasso_mode()
            _enter_camera_mode()
        else:  # camera
            _enter_rect_mode()

    plotter.add_key_event("Tab", cycle_mode)

    def _clear_pick_highlight():
        """Remove every selection / lasso highlight overlay actor."""
        # PyVista-managed actors (rect picker, etc.)
        candidates = [
            "_cell_picking_selection",
            "_picked_cells",
            "cell_pick_highlight",
            "_lasso_highlight",
        ]
        for name in candidates:
            try:
                plotter.remove_actor(name, render=False)
            except Exception:
                pass
        try:
            for name in list(plotter.actors.keys()):
                lname = str(name).lower()
                if any(t in lname for t in ("pick", "selection", "highlight", "lasso")):
                    try:
                        plotter.remove_actor(name, render=False)
                    except Exception:
                        pass
        except Exception:
            pass
        # Raw-VTK lasso highlight actor (bypasses PyVista's actor registry)
        old = state.get("lasso_vtk_actor")
        if old is not None:
            try:
                plotter.renderer.RemoveActor(old)
            except Exception:
                pass
            state.pop("lasso_vtk_actor", None)

    def make_key_handler(digit: int):
        def _h():
            print(f"  [digit {digit} keypress fired]")  # debug: always print
            if state["pending_face_indices"] is None:
                print(f"  (digit {digit} pressed but no area selected — "
                      "drag a rectangle or lasso with LEFT mouse first)")
                return
            ids = state["pending_face_indices"]
            for face_idx in ids:
                state["user_annotations"][int(face_idx)] = digit
            print(f"  → Marked {len(ids)} faces as user link {digit}.")
            # Persist to JSON immediately so the data is safe even if
            # the window misbehaves on close.
            if annotations_path is not None:
                import json
                annotations_path.parent.mkdir(parents=True, exist_ok=True)
                annotations_path.write_text(
                    json.dumps({str(k): v for k, v in state["user_annotations"].items()})
                )
            # Drop the "pending highlight" bookkeeping — the user's color
            # will replace the red on those cells via the recolor below.
            state.pop("pending_highlight_indices", None)
            state.pop("pending_highlight_orig_rgb", None)
            pv_mesh.cell_data["rgb"] = colour_for_user(state["user_annotations"])
            pv_mesh.Modified()
            _clear_pick_highlight()
            show_status(None)  # clear the "SELECTED X FACES" banner
            plotter.render()
            unique_assigned = sorted(set(state["user_annotations"].values()))
            show_instructions(
                f"Annotated: {len(state['user_annotations'])} faces "
                f"across {len(unique_assigned)} regions {unique_assigned}.\n"
                "Drag another rectangle, press a digit, or:\n"
                "  SPACE = commit and exit\n"
                "  ESC = abandon annotation"
            )
            state["pending_face_indices"] = None
        return _h

    for d in range(10):
        plotter.add_key_event(str(d), make_key_handler(d))

    # Reliable exit: use PyVista's interactive_update mode so show()
    # returns immediately, then drive the event loop ourselves and break
    # on a flag set from the keypress handler. This sidesteps VTK builds
    # where plotter.close() / TerminateApp don't unblock a blocking show().
    state["done"] = False
    state["skip"] = False

    def commit_and_exit():
        print("  [commit] flagging exit (annotations are already on disk)")
        state["done"] = True

    def skip_and_exit():
        print("  [skip] discarding annotations")
        if annotations_path is not None and annotations_path.exists():
            try:
                annotations_path.write_text("{}")
            except Exception:
                pass
        state["user_annotations"] = {}
        state["skip"] = True
        state["done"] = True

    plotter.add_key_event("space", commit_and_exit)
    plotter.add_key_event("Return", commit_and_exit)
    plotter.add_key_event("KP_Enter", commit_and_exit)
    plotter.add_key_event("Escape", skip_and_exit)

    # Non-blocking show + aggressive re-assertion. On every tick we
    # re-add the lasso highlight actor and the status banner, since
    # something on this VTK build keeps removing them after one render.
    # Cheap on a small sub-mesh; ugly but reliable.
    import time
    plotter.show(interactive_update=True, auto_close=False)
    iren_wrapper = plotter.iren
    vtk_iren_loop = getattr(iren_wrapper, "interactor", None) or \
               getattr(iren_wrapper, "_interactor", None) or iren_wrapper

    def _is_actor_in_renderer(actor):
        try:
            actors = plotter.renderer.GetActors()
            actors.InitTraversal()
            for _ in range(actors.GetNumberOfItems()):
                a = actors.GetNextActor()
                if a is actor:
                    return True
        except Exception:
            pass
        return False

    while not state["done"]:
        try:
            vtk_iren_loop.ProcessEvents()
        except Exception:
            try:
                plotter.update()
            except Exception:
                break

        # Re-add lasso highlight actor if it got removed
        try:
            actor = state.get("lasso_vtk_actor")
            if actor is not None and not _is_actor_in_renderer(actor):
                plotter.renderer.AddActor(actor)
                plotter.render()
        except Exception:
            pass

        # Re-add status banner if it disappeared
        try:
            sa = status_actor[0]
            if sa is not None and state.get("status_msg"):
                actors2d = plotter.renderer.GetActors2D()
                actors2d.InitTraversal()
                present = False
                for _ in range(actors2d.GetNumberOfItems()):
                    a = actors2d.GetNextActor2D()
                    if a is sa:
                        present = True
                        break
                if not present:
                    plotter.renderer.AddActor2D(sa)
                    plotter.render()
        except Exception:
            pass

        time.sleep(0.02)
    try:
        plotter.close()
    except Exception:
        pass
    if state["skip"]:
        return {}
    return state["user_annotations"]


def run_interactive(
    mesh: trimesh.Trimesh,
    face_labels: np.ndarray,
    face_centers: np.ndarray,
    on_region_anchor: callable,  # (face_indices_array, user_label) -> updated face_labels
) -> np.ndarray:
    """Open a PyVista plotter for free-form area selection + label.

    Interaction:
      * Press 'R' to enter rectangle-drag selection mode.
      * Drag a rectangle on screen — all visible faces inside become
        the user's anchor region (independent of ML's clustering).
      * Press a digit 0–9 to assign that whole region to a link ID.
      * Repeat as needed (each anchor refines further).
      * Press 'S' to save and exit, 'Q' to abandon refinement.

    The user is NOT constrained to ML's cluster boundaries. They can
    select any subset of faces — including faces ML grouped under
    different labels (which get merged), or only part of an ML cluster
    (with the rest staying as-is).
    """
    try:
        import pyvista as pv
    except ImportError:
        raise SystemExit(
            "pyvista is required for interactive refinement. "
            "Install with `pip install pyvista`."
        )

    palette = _make_palette()

    verts = np.asarray(mesh.vertices, dtype=np.float32)
    faces_tri = np.asarray(mesh.faces, dtype=np.int64)
    n_faces = len(faces_tri)
    flat_faces = np.column_stack(
        [np.full(n_faces, 3, dtype=np.int64), faces_tri]
    ).reshape(-1)
    pv_mesh = pv.PolyData(verts, flat_faces)
    pv_mesh.cell_data["link_id"] = face_labels.astype(np.int32)

    plotter = pv.Plotter(window_size=(1100, 850))
    plotter.set_background("white")

    state = {
        "face_labels": face_labels.copy(),
        "pending_face_indices": None,  # array of face idx after rect drag
    }

    def colour_for_labels(labels: np.ndarray) -> np.ndarray:
        unique = np.unique(labels)
        rgb = np.zeros((len(labels), 3), dtype=np.float32)
        for u in unique:
            if u < 0:
                continue
            rgb[labels == u] = palette[int(u) % len(palette)]
        return rgb

    pv_mesh.cell_data["rgb"] = colour_for_labels(state["face_labels"])
    plotter.add_mesh(
        pv_mesh, scalars="rgb", rgb=True,
        show_edges=False, smooth_shading=True,
    )

    instr_actor = [None]

    def show_instructions(msg: str) -> None:
        if instr_actor[0] is not None:
            try:
                plotter.remove_actor(instr_actor[0])
            except Exception:
                pass
        instr_actor[0] = plotter.add_text(
            msg, position="upper_left", font_size=10, color="black",
        )

    show_instructions(
        "POST-SELECT REFINEMENT [Pick] — Tab swaps modes.\n"
        "  • PICK: LEFT drag = rectangle-select (red highlight).\n"
        "  • CAMERA: drag rotates, scroll zooms.\n"
        "  • Tab → swap modes anytime.\n"
        "  • Digit 0-9 = label highlighted area.\n"
        "  • SPACE / Enter = save URDF and exit.\n"
        "  • ESC = quit without saving."
    )

    def on_rect_pick(picked):
        """Rectangle-drag callback — `picked` is a sub-mesh containing
        all cells inside the user's screen-space rectangle."""
        if picked is None or picked.n_cells == 0:
            print("  (empty selection — try drawing a bigger rectangle)")
            return
        ids = picked.cell_data.get("vtkOriginalCellIds")
        if ids is None:
            ids = picked.cell_data.get("orig_extract_id")
        if ids is None:
            ids = picked.cell_data.get("original_cell_ids")
        if ids is None or len(ids) == 0:
            # Fallback: nearest-face match by centroid
            picked_centers = picked.cell_centers().points
            mesh_centers = pv_mesh.cell_centers().points
            from scipy.spatial import cKDTree
            tree = cKDTree(mesh_centers)
            _, ids = tree.query(picked_centers, k=1)
        ids = np.asarray(ids, dtype=np.int64)
        ids = np.unique(ids)
        ml_labels_in_selection = sorted(set(
            int(l) for l in state["face_labels"][ids] if l >= 0
        ))
        state["pending_face_indices"] = ids
        show_instructions(
            f"Selected {len(ids)} faces "
            f"(spans ML labels: {ml_labels_in_selection}).\n"
            "Press a digit 0–9 to assign this region to a link ID.\n"
            "Or press 'P' again and redraw the rectangle."
        )
        print(f"  Rectangle-selected {len(ids)} faces; "
              f"spans ML labels {ml_labels_in_selection}.")

    plotter.enable_cell_picking(
        callback=on_rect_pick,
        through=True,
        show=True,
        start=True,
        style="wireframe",
        color="red",
        line_width=4,
        show_message=False,
        font_size=14,
    )

    state["mode"] = "pick"

    def toggle_mode():
        if state["mode"] == "pick":
            plotter.disable_picking()
            plotter.enable_trackball_style()
            state["mode"] = "camera"
            show_instructions(
                "CAMERA MODE — drag to rotate, scroll to zoom.\n"
                "Tab → back to Pick mode."
            )
            print("  [mode = CAMERA]")
        else:
            plotter.enable_cell_picking(
                callback=on_rect_pick,
                through=True,
                show=True,
                start=True,
                style="wireframe",
                color="red",
                line_width=4,
                show_message=False,
                font_size=14,
            )
            state["mode"] = "pick"
            show_instructions(
                "PICK MODE — LEFT drag selects, digit 0-9 labels.\n"
                "Tab → swap to Camera. SPACE = save. ESC = quit."
            )
            print("  [mode = PICK]")

    plotter.add_key_event("Tab", toggle_mode)

    def _clear_pick_highlight():
        for name in ["_cell_picking_selection", "_picked_cells", "cell_pick_highlight"]:
            try:
                plotter.remove_actor(name, render=False)
            except Exception:
                pass
        try:
            for name in list(plotter.actors.keys()):
                lname = str(name).lower()
                if "pick" in lname or "selection" in lname:
                    try:
                        plotter.remove_actor(name, render=False)
                    except Exception:
                        pass
        except Exception:
            pass

    def make_key_handler(digit: int):
        def _h():
            if state["pending_face_indices"] is None:
                print(f"  (digit {digit} pressed but no area selected — "
                      "drag a rectangle with LEFT mouse first)")
                return
            ids = state["pending_face_indices"]
            print(f"  → Assigning {len(ids)} faces as user link {digit}.")
            new_labels = on_region_anchor(ids, digit)
            state["face_labels"] = new_labels
            pv_mesh.cell_data["rgb"] = colour_for_labels(new_labels)
            pv_mesh.Modified()
            _clear_pick_highlight()
            plotter.render()
            unique_after = sorted(int(u) for u in np.unique(new_labels) if u >= 0)
            show_instructions(
                f"Refined: {len(unique_after)} unique link IDs.\n"
                "Press 'P' + drag for another area, then a digit, or:\n"
                "  SPACE = save URDF and exit"
            )
            state["pending_face_indices"] = None
        return _h

    for d in range(10):
        plotter.add_key_event(str(d), make_key_handler(d))

    plotter.add_key_event("space", plotter.close)
    plotter.add_key_event("Return", plotter.close)
    plotter.add_key_event("KP_Enter", plotter.close)
    plotter.add_key_event("Escape", plotter.close)

    plotter.show()
    return state["face_labels"]


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mesh-to-world", type=Path, default=None)
    parser.add_argument("--encoder", choices=["pointnet", "ptv3"],
                        default="ptv3")
    parser.add_argument("--n-points", type=int, default=16384)
    parser.add_argument("--device",
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-gui", action="store_true",
                        help="Skip the interactive viewer (build URDF from "
                             "raw ML predictions only — equivalent to "
                             "predict_urdf.py).")
    parser.add_argument("--show-ml-first", action="store_true",
                        help="Run ML before opening the viewer and show its "
                             "predicted segmentation as the starting colours. "
                             "User refines after seeing the ML guess. "
                             "Default is the opposite: pre-select first, "
                             "then ML runs once.")
    parser.add_argument("--propagation-threshold", type=float, default=0.30,
                        help="Strict-mode propagation threshold (0..1). For "
                             "each ML cluster, the user's selection has to "
                             "cover MORE than this fraction of the cluster's "
                             "faces with one label before that label "
                             "propagates to the whole cluster. Lower = "
                             "rougher hints honoured (good for sloppy "
                             "selections); higher = more conservative "
                             "(safer against accidental bleed-over). "
                             "Default 0.30 = a 30%% rough-coverage hint "
                             "propagates; a sub-30%% accidental touch does not.")
    parser.add_argument("--user-annotations", type=Path, default=None,
                        help="Load pre-saved user annotations from JSON "
                             "(produced by an earlier interactive session) "
                             "and skip the GUI. Useful when the GUI hung on "
                             "exit but annotations were persisted to disk.")
    parser.add_argument("--motion-dir", type=Path, default=None,
                        help="Stage 3: directory containing joint_<N>/ "
                             "subfolders with state*.png|jpg images. When "
                             "supplied, the legacy image-pair RANSAC + PnP "
                             "pipeline runs after ML and OVERRIDES the "
                             "ML-predicted joint axis / origin for every "
                             "joint where extraction succeeds. Provides "
                             "~0.1° axis precision when the capture rig "
                             "is available.")
    parser.add_argument("--geometric-joints", action="store_true",
                        help="Phase D.4: fit each joint's axis/origin from "
                             "the boundary between adjacent link meshes "
                             "(3D circle fit). Replaces the ML joint head's "
                             "predictions with geometry-derived ones. "
                             "Recommended whenever the user has annotated "
                             "the segmentation — sidesteps the model's "
                             "~38° axis_deg val error.")
    parser.add_argument("--collision-sweep", action="store_true",
                        help="Tier-3: after writing the URDF, sweep each "
                             "joint in PyBullet from the model-predicted "
                             "lower to upper bound, narrow each side to the "
                             "first self-collision, then rewrite the URDF "
                             "with refined limits. Catches cases where the "
                             "model overshoots actual self-collision-free "
                             "range. Adjacent-link contacts are filtered "
                             "(expected for revolute joints).")
    parser.add_argument("--sweep-steps", type=int, default=64,
                        help="Samples per side per joint for --collision-sweep. "
                             "Higher = tighter bound, slower (default 64).")
    parser.add_argument("--topology",
                        choices=["serial", "tree", "auto", "vlm"],
                        default="serial",
                        help="Kinematic chain layout. 'serial' (default) "
                             "Z-sorts links into a single chain — correct for "
                             "industrial arms. 'tree' infers parent-child via "
                             "face-adjacency BFS — correct for humanoids, "
                             "quadrupeds, multi-arm rigs. 'auto' tries tree "
                             "and falls back to serial when the graph is a "
                             "path. 'vlm' takes the VLM prior's topology answer "
                             "(requires --vlm-prior); falls back to 'auto' if "
                             "the VLM is unavailable or low-confidence.")
    # ── VLM prior ──────────────────────────────────────────────────────
    parser.add_argument("--vlm-prior", action="store_true",
                        help="Phase E.4a — before any other stage, render 4 "
                             "canonical-angle views of the mesh and ask a VLM "
                             "(see --vlm-backend) what kind of robot it is. "
                             "The structured prior is saved to <output>/vlm_prior.json "
                             "and used to: (a) hard-block when expected_dof=0 "
                             "(non-articulated mesh), (b) auto-pick --topology "
                             "when set to 'vlm', (c) prune the ML's "
                             "over-segmented clusters down to ~expected_link_count "
                             "before URDF assembly.")
    parser.add_argument("--vlm-backend", choices=["gemini"], default="gemini",
                        help="VLM backend for --vlm-prior. Currently only "
                             "Gemini (free tier). Anthropic / OpenAI will be "
                             "added later via the same Protocol.")
    parser.add_argument("--refresh-vlm", action="store_true",
                        help="Force a fresh VLM call even if "
                             "<output>/vlm_prior.json already exists. By "
                             "default the cached prior is reused to save "
                             "API calls and time.")
    # ── VLM critic ─────────────────────────────────────────────────────
    parser.add_argument("--vlm-critic", action="store_true",
                        help="Phase E.4b — after the URDF + sweep finishes, "
                             "render the assembled URDF from the same 4 "
                             "canonical angles, send to a VLM along with the "
                             "input mesh views, and ask: 'what's wrong with "
                             "this segmentation?'. Writes a structured "
                             "CritiqueResult to <output>/vlm_critic.json. By "
                             "default just reports issues; pass --vlm-auto-fix "
                             "to apply safe auto-merges.")
    parser.add_argument("--vlm-auto-fix", action="store_true",
                        help="When the critic flags 'duplicate_link' issues "
                             "with high severity AND a merge_into target, "
                             "relabel face_labels to merge those links and "
                             "re-assemble the URDF in place. Off by default; "
                             "the critic always reports first.")
    parser.add_argument("--no-cleanup-clusters", action="store_true",
                        help="Disable the connected-component cleanup that "
                             "splits each label into its connected pieces "
                             "(largest keeps the label; small strays get "
                             "absorbed into their dominant neighbour). "
                             "Default: on. Disable to inspect raw ML segmentation.")
    parser.add_argument("--camera-intrinsics", type=Path, default=None,
                        help="Path to calibration.json (fx/fy/cx/cy/dist). "
                             "If omitted but --motion-dir is set, defaults "
                             "to <motion-dir>/../calibration.json.")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    rng = np.random.default_rng(args.seed)

    # --- Load mesh ---
    print(f"Loading mesh: {args.mesh}")
    mesh = trimesh.load(str(args.mesh), force="mesh")
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)
    if args.mesh_to_world is not None:
        T = np.load(args.mesh_to_world)
        mesh = mesh.copy()
        mesh.apply_transform(T)
        print(f"  Applied mesh-to-world transform from {args.mesh_to_world}")
    print(f"  {len(mesh.vertices)} verts, {len(mesh.faces)} faces, "
          f"AABB extent={np.round(mesh.extents, 3).tolist()}")

    face_centers = np.asarray(mesh.vertices)[mesh.faces].mean(axis=1)

    # ── Phase E.4a — VLM prior ────────────────────────────────────────
    # Runs BEFORE annotation/ML so we can hard-block on rigid meshes,
    # auto-pick topology, and prune over-segmentation later.
    vlm_prior_obj = None
    if args.vlm_prior:
        import json
        from dataclasses import asdict
        from mesh2robot.core.vlm_prior import (
            get_robot_prior, _dict_to_prior, _prior_to_dict,
        )
        prior_path = args.output / "vlm_prior.json"
        if prior_path.exists() and not args.refresh_vlm:
            print(f"\n--- VLM prior (cached) ---")
            cached = json.loads(prior_path.read_text())
            vlm_prior_obj = _dict_to_prior(cached)
            print(f"  loaded from {prior_path} (use --refresh-vlm to re-call)")
        else:
            print(f"\n--- VLM prior ({args.vlm_backend}) ---")
            try:
                vlm_prior_obj = get_robot_prior(
                    mesh,
                    backend=args.vlm_backend,
                    save_dir=args.output / "vlm_views",
                )
                # Persist for caching + audit
                with open(prior_path, "w") as f:
                    json.dump(_prior_to_dict(vlm_prior_obj), f, indent=2)
                print(f"  saved prior to {prior_path}")
            except Exception as e:
                print(f"  [WARN] VLM prior failed: {type(e).__name__}: {e}")
                print(f"         continuing without prior (no auto-topology, "
                      f"no cluster pruning, no rigid-mesh block).")
                vlm_prior_obj = None

        if vlm_prior_obj is not None:
            print(vlm_prior_obj)

            # Hard-block on non-articulated meshes (only when no user
            # annotations — if the user is hand-labeling, they overrule
            # the VLM).
            if (vlm_prior_obj.expected_dof == 0
                    and args.user_annotations is None
                    and args.no_gui):
                print(f"\n[ABORT] VLM says expected_dof=0 — this mesh has no "
                      f"visible articulation (likely a rigid model or static "
                      f"object). No URDF will be generated.\n"
                      f"        If you want to override, run with annotations "
                      f"or skip --vlm-prior.")
                return

            # Auto-pick topology when --topology vlm
            if args.topology == "vlm":
                if (vlm_prior_obj.is_high_confidence()
                        and vlm_prior_obj.expected_chain_topology != "unknown"):
                    mapped = {
                        "serial": "serial",
                        "tree": "tree",
                        "parallel": "auto",  # try tree, may degrade
                    }.get(vlm_prior_obj.expected_chain_topology, "auto")
                    print(f"  --topology vlm → using {mapped!r} "
                          f"(VLM topology={vlm_prior_obj.expected_chain_topology}, "
                          f"confidence={vlm_prior_obj.confidence:.2f})")
                    args.topology = mapped
                else:
                    print(f"  --topology vlm → falling back to 'auto' "
                          f"(VLM low-confidence or topology=unknown)")
                    args.topology = "auto"

    # --- PRE-SELECT MODE (default): user annotates BEFORE the model runs.
    # The viewer shows the mesh shaded by surface normals; the user marks
    # any number of regions. ML runs once after the user commits.
    user_face_labels: dict[int, int] = {}
    if args.user_annotations is not None:
        # Load pre-saved annotations, skip GUI entirely.
        import json
        raw = json.loads(args.user_annotations.read_text())
        user_face_labels = {int(k): int(v) for k, v in raw.items()}
        unique = sorted(set(user_face_labels.values()))
        print(f"\nLoaded {len(user_face_labels)} pre-saved annotations from "
              f"{args.user_annotations}: link IDs {unique}")
    elif not args.no_gui and not args.show_ml_first:
        print("\n--- Pre-selection viewer (annotate before ML) ---")
        print("    Tab cycles modes:  Rect → Lasso → Camera → Rect")
        print()
        print("    RECT  : LEFT drag = rectangle select (red box)")
        print("    LASSO : LEFT drag = freeform polygon trace")
        print("    CAMERA: LEFT drag rotates, scroll zooms")
        print()
        print("    After a selection, press a digit 0-9 to label it as a link ID.")
        print("    Annotations persist to user_annotations.json on every digit.")
        print()
        print("    SPACE or Enter = commit and run ML")
        print("    ESC or close window = skip annotation")
        annot_path = args.output / "user_annotations.json"
        # GUI uses a non-blocking show + flag-based exit; returns normally.
        user_face_labels = run_pre_selection(
            mesh, annotations_path=annot_path,
        )
        if user_face_labels:
            unique = sorted(set(user_face_labels.values()))
            print(f"  Captured {len(user_face_labels)} annotations from GUI: "
                  f"link IDs {unique}")
        if user_face_labels:
            unique = sorted(set(user_face_labels.values()))
            print(f"  User annotated {len(user_face_labels)} faces "
                  f"into {len(unique)} regions: link IDs {unique}")
        else:
            print("  No user annotations — falling back to pure ML.")

    # --- Sample + normalize ---
    points = sample_mesh_to_points(mesh, args.n_points, rng)
    centroid = points.mean(axis=0)
    pts_n = points - centroid
    radii = np.linalg.norm(pts_n, axis=1)
    scale = float(np.percentile(radii, 99)) + 1e-8
    pts_n = pts_n / scale

    # --- ML runs ONCE here ---
    print(f"\nLoading checkpoint: {args.checkpoint}")
    # Peek at checkpoint to recover the encoder_size used at training; fall
    # back to "small" for older checkpoints that didn't save it.
    ckpt_peek = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    encoder_size = ckpt_peek.get("args", {}).get("encoder_size", "small")
    print(f"  Encoder size from checkpoint: {encoder_size}")
    del ckpt_peek
    model = Mesh2RobotModel(
        feat_dim=256, encoder=args.encoder, encoder_size=encoder_size,
    ).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    # strict=False so checkpoints predating the LimitsHead still load. Any
    # missing-key warnings are surfaced once at the bottom.
    load_info = model.load_state_dict(ckpt["model_state_dict"], strict=False)
    if load_info.missing_keys:
        print(f"  [info] checkpoint missing {len(load_info.missing_keys)} keys "
              f"(e.g. {load_info.missing_keys[0]}); these heads run from "
              f"random init — retrain on v3 shards to use them.")
    model.eval()

    pts_t = torch.from_numpy(pts_n.astype(np.float32)).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = model(pts_t)
    pred_labels = pred["seg_logits"][0].argmax(dim=-1).cpu().numpy()
    pred_axes = pred["axis"][0].cpu().numpy()
    pred_axes /= (np.linalg.norm(pred_axes, axis=-1, keepdims=True) + 1e-9)
    pred_origins = pred["origin"][0].cpu().numpy()
    pred_valid = (torch.sigmoid(pred["valid_logit"][0]).cpu().numpy() > 0.5)
    pred_types = pred["type_logits"][0].argmax(dim=-1).cpu().numpy()
    pred_origins_world = pred_origins * scale + centroid
    # LimitsHead output (only meaningful for v3-trained checkpoints; for
    # older checkpoints loaded with strict=False the head is randomly
    # initialized and we should skip the model-limits branch).
    pred_limits = None
    if "limits" in pred and "joint_head.limits_head.weight" in ckpt.get(
        "model_state_dict", {}
    ):
        pred_limits = pred["limits"][0].cpu().numpy()
        print(f"  Limits head present (trained checkpoint).")
    else:
        print(f"  Limits head missing or randomly init'd; using ±π fallback.")

    print(f"\nML predictions:")
    print(f"  Unique link IDs: {sorted(set(int(l) for l in pred_labels))}")
    print(f"  Valid joints:    {int(pred_valid.sum())}")

    # --- Project to faces ---
    face_labels = project_labels_to_faces(
        sampled_points=points, sampled_labels=pred_labels,
        mesh_verts=np.asarray(mesh.vertices),
        mesh_faces=np.asarray(mesh.faces),
        k=5,
    )

    # --- Stage 3 (optional): motion-image refinement of joint geometry ---
    motion_overrides: dict[int, tuple[np.ndarray, np.ndarray]] | None = None
    if args.motion_dir is not None:
        print(f"\n--- Stage 3: Motion-image joint refinement ({args.motion_dir}) ---")
        if not args.motion_dir.exists():
            print(f"  WARN: --motion-dir not found, skipping. ({args.motion_dir})")
        else:
            cam_path = args.camera_intrinsics
            if cam_path is None:
                cam_path = args.motion_dir.parent / "calibration.json"
            if not cam_path.exists():
                print(f"  WARN: camera intrinsics not found at {cam_path}; "
                      "specify --camera-intrinsics. Skipping motion stage.")
            else:
                motion_overrides = extract_motion_overrides(
                    mesh=mesh,
                    motion_dir=args.motion_dir,
                    camera_intrinsics_path=cam_path,
                )
                print(f"  Motion overrides recovered: {len(motion_overrides)} joints")

    # --- Build the pure-ML URDF for comparison (always emitted) ---
    print("\n--- Building pure-ML URDF (no user input, no motion) ---")
    original_dir = args.output / "original"
    original_dir.mkdir(parents=True, exist_ok=True)
    orig_urdf = build_urdf_from_predictions(
        mesh, face_labels.copy(), pred_axes, pred_origins_world,
        pred_valid, pred_types, original_dir,
        robot_name="ai_predicted_original",
        pred_limits=pred_limits,
        collision_sweep=args.collision_sweep,
        sweep_steps=args.sweep_steps,
        topology_mode=args.topology,
        expected_link_count=(vlm_prior_obj.expected_link_count
                              if vlm_prior_obj is not None else None),
        cleanup_clusters=not args.no_cleanup_clusters,
    )
    if orig_urdf is not None:
        print(f"  Wrote {orig_urdf}")

    # --- Apply pre-select annotations OR enter post-select mode ---
    refined_face_labels: np.ndarray | None = None
    history: list[dict] = []

    if user_face_labels:
        # Pre-select path: merge user labels with ML labels and we're done.
        refined_face_labels, info = merge_user_and_ml_labels(
            user_face_labels, face_labels, face_centers,
            propagation_threshold=args.propagation_threshold,
        )
        history.append({
            "mode": "pre-select",
            **info,
        })
        print(f"\nStrict-mode merge (threshold={info['propagation_threshold']:.2f}):")
        print(f"  user-overridden faces: {info['n_user_overrides']}")
        print(f"  propagated clusters (≥threshold coverage): "
              f"{info['propagated_clusters']}")
        print(f"  split clusters (sub-threshold; user faces overridden, "
              f"rest renumbered): {info['split_clusters']}")
        print(f"  untouched/split-remainder remap → fresh labels: "
              f"{info['untouched_remap']}")

    elif args.show_ml_first and not args.no_gui:
        # Post-select path: user sees ML output first, refines after.
        print("\n--- Post-select viewer (ML predictions shown; refine on top) ---")
        run_interactive_state = {"face_labels": face_labels.copy()}

        def on_region_anchor(face_indices: np.ndarray, user_label: int) -> np.ndarray:
            current = run_interactive_state["face_labels"]
            new_labels, info = refine_labels_by_anchor_region(
                current, face_centers, face_indices, user_label,
            )
            history.append({
                "mode": "post-select",
                "n_faces_selected": int(len(face_indices)),
                "user_label": int(user_label),
                "ml_labels_merged": info["merged_ml_labels"],
                "remap_outside": info["remap"],
            })
            print(f"     Hard override on {info['overridden_face_count']} faces; "
                  f"merged ML labels {info['merged_ml_labels']}; "
                  f"renumber-outside remap = {info['remap']}")
            run_interactive_state["face_labels"] = new_labels
            return new_labels

        refined_face_labels = run_interactive(
            mesh, face_labels, face_centers,
            on_region_anchor=on_region_anchor,
        )

    # --- Build refined URDF if user annotated OR motion overrides exist ---
    has_user_input = refined_face_labels is not None and \
        not np.array_equal(refined_face_labels, face_labels)
    has_motion = motion_overrides is not None and len(motion_overrides) > 0

    if not has_user_input and not has_motion:
        print("\nNo user input or motion data — refined URDF == original URDF.")
        # Wire `refined_urdf` to the original so downstream GLB export +
        # VLM critic (Phase E.4b) still run. final_labels mirrors the
        # raw ML labels for the auto-fix path.
        refined_urdf = orig_urdf
        refined_face_labels = face_labels.copy()
        final_labels = face_labels.copy()
        # Skip the explicit refined-URDF-build block below by jumping
        # past it via this guard.
        skip_refined_build = True
    else:
        # Use refined labels if user provided them, else the raw ML labels.
        final_labels = refined_face_labels if has_user_input else face_labels.copy()
        skip_refined_build = False

    if not skip_refined_build:
        suffix_parts = []
        if has_user_input:
            suffix_parts.append("user")
        if has_motion:
            suffix_parts.append(f"motion({len(motion_overrides)}j)")
        if args.geometric_joints:
            suffix_parts.append("geom-joints")
        print(f"\n--- Building refined URDF [{' + '.join(suffix_parts)}] ---")
        refined_dir = args.output / "refined"
        refined_dir.mkdir(parents=True, exist_ok=True)
        refined_urdf = build_urdf_from_predictions(
            mesh, final_labels, pred_axes, pred_origins_world,
            pred_valid, pred_types, refined_dir,
            robot_name="ai_predicted_refined",
            motion_overrides=motion_overrides,
            use_geometric_joints=args.geometric_joints,
            pred_limits=pred_limits,
            collision_sweep=args.collision_sweep,
            sweep_steps=args.sweep_steps,
            topology_mode=args.topology,
            expected_link_count=(vlm_prior_obj.expected_link_count
                                  if vlm_prior_obj is not None else None),
            cleanup_clusters=not args.no_cleanup_clusters,
        )
        if refined_urdf is not None:
            print(f"  Wrote {refined_urdf}")
            try:
                from yourdfpy import URDF
                r = URDF.load(str(refined_urdf))
                print(f"  URDF loads OK: {len(r.link_map)} links, "
                      f"{len(r.actuated_joint_names)} actuated joints")
            except Exception as e:
                print(f"  (yourdfpy reload failed: {e})")

    # --- Side-by-side GLB exports for rough-vs-refined comparison ---
    # 1. user_annotation.glb : the user's directly-clicked faces colored,
    #    untouched faces grey. Shows the ROUGH input.
    # 2. refined_assembled.glb : the refined URDF with FK applied so
    #    every link sits at its assembled-pose world position. Shows
    #    the FINAL output.
    # Both land in args.output so a side-by-side viewer comparison is
    # one click away.
    try:
        from visualize_refined import (
            render_face_labels_on_mesh,
            render_from_urdf,
        )
    except ImportError:
        # scripts/ may not be on sys.path when invoked oddly; import by file.
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "visualize_refined",
            str(Path(__file__).resolve().parent / "visualize_refined.py"),
        )
        viz_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(viz_mod)
        render_face_labels_on_mesh = viz_mod.render_face_labels_on_mesh
        render_from_urdf = viz_mod.render_from_urdf

    if user_face_labels:
        print("\n--- Exporting comparison GLBs ---")
        n_faces = len(mesh.faces)
        rough_labels = -np.ones(n_faces, dtype=np.int64)
        for fi, lbl in user_face_labels.items():
            if 0 <= fi < n_faces:
                rough_labels[fi] = int(lbl)
        rough_scene = render_face_labels_on_mesh(mesh, rough_labels)
        rough_glb = args.output / "user_annotation.glb"
        rough_scene.export(str(rough_glb))
        n_unlabeled = int((rough_labels < 0).sum())
        print(f"  Rough user annotation : {rough_glb} "
              f"({len(user_face_labels)} tagged, {n_unlabeled} grey)")

    if refined_urdf is not None:
        try:
            assembled_scene = render_from_urdf(refined_urdf)
            assembled_glb = args.output / "refined_assembled.glb"
            assembled_scene.export(str(assembled_glb))
            print(f"  Refined assembled URDF: {assembled_glb}")
        except Exception as e:
            print(f"  (refined_assembled.glb export failed: {e})")

    # ── Phase E.4b — VLM critic ───────────────────────────────────────
    # Compares the assembled URDF against the input mesh and reports
    # abnormalities. Optional auto-fix applies safe link-merges.
    if args.vlm_critic and refined_urdf is not None:
        print("\n--- VLM critic ---")
        try:
            import json
            from mesh2robot.core.vlm_critic import (
                render_urdf_canonical_views,
                critique_urdf,
                _critique_to_dict,
            )
            from mesh2robot.core.vlm_prior import render_canonical_views
            urdf_views = render_urdf_canonical_views(refined_urdf)
            input_views = render_canonical_views(mesh)
            # Persist URDF canonical views for audit
            critic_views_dir = args.output / "vlm_critic_views"
            critic_views_dir.mkdir(parents=True, exist_ok=True)
            for name, img in zip(
                ["front", "right", "back", "three_quarter"], urdf_views,
            ):
                (critic_views_dir / f"urdf_view_{name}.png").write_bytes(img)
            # Topology summary for the prompt
            from yourdfpy import URDF as _URDF
            _u = _URDF.load(str(refined_urdf))
            topo_summary = (f"{len(_u.link_map)} links, "
                             f"{len(_u.actuated_joint_names)} actuated joints, "
                             f"topology mode '{args.topology}'")
            critic = critique_urdf(
                input_views=input_views,
                urdf_views=urdf_views,
                prior=vlm_prior_obj,
                n_links=len(_u.link_map),
                n_actuated=len(_u.actuated_joint_names),
                topology_summary=topo_summary,
            )
            critic_path = args.output / "vlm_critic.json"
            with open(critic_path, "w") as f:
                json.dump(_critique_to_dict(critic), f, indent=2)
            print(critic)
            print(f"  saved critique to {critic_path}")

            # Auto-fix: apply safe link merges if enabled
            merge_actions = critic.merge_actions()
            if args.vlm_auto_fix and merge_actions:
                print(f"\n--- Auto-fix: applying {len(merge_actions)} link "
                      f"merge(s) ---")
                # Translate URDF body_id → original cluster id via the
                # mapping persisted by build_urdf_from_predictions.
                mapping_path = refined_urdf.parent / "body_to_cluster.json"
                body_to_cluster: list[int] | None = None
                if mapping_path.exists():
                    try:
                        body_to_cluster = json.loads(
                            mapping_path.read_text()
                        )["link_ids_in_order"]
                    except Exception:
                        body_to_cluster = None
                if body_to_cluster is None:
                    print(f"  [WARN] body_to_cluster.json missing; "
                          f"interpreting critic IDs as raw cluster IDs.")

                final_labels_fix = (
                    refined_face_labels.copy()
                    if refined_face_labels is not None else face_labels.copy()
                )
                for action in merge_actions:
                    dst_body = int(action.target)
                    dst_cluster = (body_to_cluster[dst_body]
                                    if body_to_cluster is not None and
                                       0 <= dst_body < len(body_to_cluster)
                                    else dst_body)
                    for src_body in action.sources:
                        src_cluster = (body_to_cluster[int(src_body)]
                                        if body_to_cluster is not None and
                                           0 <= int(src_body) < len(body_to_cluster)
                                        else int(src_body))
                        n_changed = int(
                            (final_labels_fix == src_cluster).sum()
                        )
                        final_labels_fix[
                            final_labels_fix == src_cluster
                        ] = dst_cluster
                        print(f"  merged URDF link_{src_body} (cluster_{src_cluster}) "
                              f"→ link_{dst_body} (cluster_{dst_cluster})  "
                              f"[{n_changed} faces] — {action.rationale}")

                fixed_dir = args.output / "refined_fixed"
                fixed_dir.mkdir(parents=True, exist_ok=True)
                fixed_urdf = build_urdf_from_predictions(
                    mesh, final_labels_fix, pred_axes, pred_origins_world,
                    pred_valid, pred_types, fixed_dir,
                    robot_name="ai_predicted_refined_fixed",
                    motion_overrides=motion_overrides,
                    use_geometric_joints=args.geometric_joints,
                    pred_limits=pred_limits,
                    collision_sweep=args.collision_sweep,
                    sweep_steps=args.sweep_steps,
                    topology_mode=args.topology,
                    expected_link_count=(vlm_prior_obj.expected_link_count
                                          if vlm_prior_obj is not None else None),
                    cleanup_clusters=not args.no_cleanup_clusters,
                )
                if fixed_urdf is not None:
                    print(f"  Wrote auto-fixed URDF: {fixed_urdf}")
                    try:
                        fixed_scene = render_from_urdf(fixed_urdf)
                        fixed_glb = args.output / "refined_fixed_assembled.glb"
                        fixed_scene.export(str(fixed_glb))
                        print(f"  Fixed-URDF GLB: {fixed_glb}")
                    except Exception as e:
                        print(f"  (fixed GLB export failed: {e})")
            elif merge_actions:
                print(f"\n  {len(merge_actions)} merge action(s) suggested; "
                      f"re-run with --vlm-auto-fix to apply.")
        except Exception as e:
            print(f"  [WARN] VLM critic failed: {type(e).__name__}: {e}")

    print(f"\nHistory entries: {len(history)}")
    for i, h in enumerate(history):
        print(f"  #{i+1}: {h}")


if __name__ == "__main__":
    main()
