"""Quick test of `geometric_joints.extract_joints_from_segmentation`
on the test_2 user-annotated mesh. Compares result vs the ML's joint
head.

Re-runs the strict-mode merge to get final per-face labels, then
extracts joints geometrically and prints them next to ML's predictions.

Usage:
    python scripts/test_geometric_joints.py \
        --checkpoint data/checkpoints/model_v2_ptv3_25ep/checkpoint_epoch_025.pt \
        --encoder ptv3 \
        --mesh input/test_2/milo/xarm6_clean.obj \
        --mesh-to-world input/test_2/T_cleaned_to_original.npy \
        --user-annotations output/test_2_full_annotation/user_annotations.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from mesh2robot.core.geometric_joints import (
    extract_joints_from_segmentation,
)
from mesh2robot.model.model import Mesh2RobotModel

# Re-use the merge + projection helpers from the interactive script
from predict_urdf import project_labels_to_faces, sample_mesh_to_points
from predict_urdf_interactive import merge_user_and_ml_labels


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--encoder", choices=["pointnet", "ptv3"], default="ptv3")
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--mesh-to-world", type=Path, default=None)
    parser.add_argument("--user-annotations", type=Path, required=True)
    parser.add_argument("--n-points", type=int, default=16384)
    parser.add_argument("--device",
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.30)
    args = parser.parse_args()

    print(f"Loading mesh: {args.mesh}")
    mesh = trimesh.load(str(args.mesh), force="mesh")
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)
    if args.mesh_to_world is not None:
        T = np.load(args.mesh_to_world)
        mesh = mesh.copy()
        mesh.apply_transform(T)
    print(f"  {len(mesh.vertices)} verts, {len(mesh.faces)} faces")

    # Sample + ML inference (we need ML's per-face labels to feed the merge)
    rng = np.random.default_rng(args.seed)
    points = sample_mesh_to_points(mesh, args.n_points, rng)
    centroid = points.mean(axis=0)
    pts_n = points - centroid
    radii = np.linalg.norm(pts_n, axis=1)
    scale = float(np.percentile(radii, 99)) + 1e-8
    pts_n = (pts_n / scale).astype(np.float32)

    device = torch.device(args.device)
    print(f"Loading checkpoint: {args.checkpoint}")
    model = Mesh2RobotModel(feat_dim=256, encoder=args.encoder).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    pts_t = torch.from_numpy(pts_n).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = model(pts_t)
    pred_seg = pred["seg_logits"][0].argmax(dim=-1).cpu().numpy()
    pred_axes = pred["axis"][0].cpu().numpy()
    pred_axes /= (np.linalg.norm(pred_axes, axis=-1, keepdims=True) + 1e-9)
    pred_origins = pred["origin"][0].cpu().numpy()
    pred_valid = (torch.sigmoid(pred["valid_logit"][0]).cpu().numpy() > 0.5)
    pred_origins_world = pred_origins * scale + centroid

    # Project per-point labels to per-face
    ml_face_labels = project_labels_to_faces(
        sampled_points=points, sampled_labels=pred_seg,
        mesh_verts=np.asarray(mesh.vertices),
        mesh_faces=np.asarray(mesh.faces), k=5,
    )

    # Load user annotations and run the strict-mode merge
    print(f"Loading user annotations: {args.user_annotations}")
    raw = json.loads(args.user_annotations.read_text())
    user_face_labels = {int(k): int(v) for k, v in raw.items()}
    print(f"  {len(user_face_labels)} user-tagged faces")

    face_centers = np.asarray(mesh.vertices)[mesh.faces].mean(axis=1)
    refined_labels, info = merge_user_and_ml_labels(
        user_face_labels, ml_face_labels, face_centers,
        propagation_threshold=args.threshold,
    )
    print(f"  merged labels: {sorted(set(int(l) for l in np.unique(refined_labels)))}")

    # Build chain order by Z of each link's centroid (matches predict_urdf logic)
    unique_labels = sorted(int(l) for l in np.unique(refined_labels) if l >= 0)
    z_per = {}
    for lbl in unique_labels:
        mask = refined_labels == lbl
        z_per[lbl] = float(np.asarray(mesh.vertices[
            np.unique(mesh.faces[mask])
        ]).mean(axis=0)[2])
    chain = sorted(unique_labels, key=lambda l: z_per[l])
    print(f"\nChain order (low → high Z): {chain}")

    # Extract joints geometrically
    print("\n--- Geometric joint extraction ---")
    joints = extract_joints_from_segmentation(mesh, refined_labels, chain)
    for i, j in enumerate(joints):
        ax = np.round(j.axis, 3).tolist()
        org = np.round(j.origin, 3).tolist()
        print(f"  joint_{i+1} (link {j.parent_label} → {j.child_label}): "
              f"type={j.type:8s}  conf={j.confidence:.2f}  "
              f"radius={j.radius:.3f}m  plane_resid={j.plane_residual:.4f}m  "
              f"edges={j.n_boundary_edges:4d}\n"
              f"      axis={ax}  origin={org}")

    # Compare with ML predictions: pick the closest ML joint by origin distance
    print("\n--- ML joint head predictions (for comparison) ---")
    valid_idx = np.where(pred_valid)[0]
    valid_idx_by_z = sorted(
        valid_idx, key=lambda j: float(pred_origins_world[j, 2]),
    )[: len(joints)]
    for chain_i, j_slot in enumerate(valid_idx_by_z):
        ax = np.round(pred_axes[j_slot], 3).tolist()
        org = np.round(pred_origins_world[j_slot], 3).tolist()
        print(f"  joint_{chain_i+1}:  axis={ax}  origin={org}")

    # Pairwise angular comparison ML vs geometric
    print("\n--- Axis angular difference: geometric vs ML ---")
    for i, j_slot in enumerate(valid_idx_by_z):
        if i >= len(joints):
            break
        a_ml = pred_axes[j_slot]
        a_geo = joints[i].axis
        cos = abs(float(a_ml @ a_geo))   # sign-invariant
        deg = float(np.degrees(np.arccos(np.clip(cos, -1, 1))))
        print(f"  joint_{i+1}: |angle(ML, geometric)| = {deg:5.1f}°")


if __name__ == "__main__":
    main()
