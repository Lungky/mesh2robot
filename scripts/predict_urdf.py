"""Phase D — model predictions → URDF.

End-to-end inference path: takes a checkpoint + a real mesh, runs the trained
model, and assembles a complete URDF (link meshes + joints + physics) using
the existing Phase 5 assembler.

Pipeline:
  1. Load mesh, sample point cloud, normalize like training Dataset.
  2. Forward through Mesh2RobotModel → per-point seg + per-joint axes/origins/types/valid.
  3. KNN-project per-point labels onto mesh faces; split mesh into per-link sub-meshes.
  4. Build a SERIAL kinematic chain by ordering predicted-valid joints by world-Z
     of their origins. Joint i connects link[i] (parent) to link[i+1] (child),
     where links are ordered identically by their centroid Z. This works for
     6-DOF industrial arms; for branching robots we'd need explicit topology
     prediction (model doesn't have that yet).
  5. Compute per-link inertials with `compute_link_inertials`.
  6. Template-match by (DOF, joint_types) for physics defaults.
  7. Call existing `urdf_assembly.assemble()`.

Caveats / known limitations:
  - Per-link mesh contiguity is not enforced. KNN voting may produce
    "speckled" faces from one link landing inside another. We could
    add a connected-components cleanup but skip for v1.
  - Topology assumed to be a serial chain (no branching, no parallel links).
  - Joint axes/origins predicted in normalized space → undone before assembly.

Usage:
    python scripts/predict_urdf.py
        --checkpoint data/checkpoints/model_v1_pointnet/checkpoint_epoch_050.pt
        --mesh input/test_2/milo/xarm6_clean.obj
        --output output/test_2_predicted_urdf
        [--mesh-to-world input/test_2/T_cleaned_to_original.npy]
        [--encoder pointnet|ptv3]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mesh2robot.core.joint_extraction import JointEstimate
from mesh2robot.core.physics import compute_link_inertials
from mesh2robot.core.physics_defaults import make_default_template
from mesh2robot.core.urdf_assembly import AssemblyInput, assemble
from mesh2robot.data_gen.urdf_loader import INT_TO_JOINT_TYPE
from mesh2robot.model.dataset import K_LINKS_MAX
from mesh2robot.model.model import Mesh2RobotModel


def sample_mesh_to_points(
    mesh: trimesh.Trimesh, n_points: int, rng: np.random.Generator,
) -> np.ndarray:
    face_areas = mesh.area_faces
    if face_areas.sum() <= 0:
        raise SystemExit("Mesh has zero surface area")
    face_idx = rng.choice(len(mesh.faces), size=n_points,
                          p=face_areas / face_areas.sum())
    r1 = rng.random(n_points)
    r2 = rng.random(n_points)
    swap = r1 + r2 > 1.0
    r1[swap] = 1.0 - r1[swap]
    r2[swap] = 1.0 - r2[swap]
    bary = np.stack([1.0 - r1 - r2, r1, r2], axis=1)
    tris = mesh.vertices[mesh.faces[face_idx]]
    return np.einsum("ni,nij->nj", bary, tris)


def project_labels_to_faces(
    sampled_points: np.ndarray, sampled_labels: np.ndarray,
    mesh_verts: np.ndarray, mesh_faces: np.ndarray, k: int = 5,
) -> np.ndarray:
    """KNN majority-vote per face."""
    from scipy.spatial import cKDTree
    face_centers = mesh_verts[mesh_faces].mean(axis=1)
    tree = cKDTree(sampled_points)
    _, idx = tree.query(face_centers, k=min(k, len(sampled_points)))
    if idx.ndim == 1:
        idx = idx[:, None]
    K = int(sampled_labels.max()) + 2  # +2 to be safe (label could equal max)
    counts = np.zeros((len(mesh_faces), K), dtype=np.int32)
    for kk in range(idx.shape[1]):
        labs = sampled_labels[idx[:, kk]]
        valid = labs >= 0
        rows = np.where(valid)[0]
        counts[rows, labs[valid]] += 1
    return counts.argmax(axis=1).astype(np.int32)


def split_mesh_by_face_labels(
    mesh: trimesh.Trimesh, face_labels: np.ndarray,
    min_faces: int = 30,
) -> dict[int, trimesh.Trimesh]:
    """Build a per-link sub-mesh dict. Drops links with < min_faces."""
    out: dict[int, trimesh.Trimesh] = {}
    verts = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.faces)
    for lbl in np.unique(face_labels):
        if lbl < 0:
            continue
        mask = face_labels == lbl
        if mask.sum() < min_faces:
            continue
        sub_faces = faces[mask]
        used = np.unique(sub_faces)
        remap = -np.ones(len(verts), dtype=np.int64)
        remap[used] = np.arange(len(used))
        sub = trimesh.Trimesh(
            vertices=verts[used],
            faces=remap[sub_faces],
            process=False,
        )
        out[int(lbl)] = sub
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mesh-to-world", type=Path, default=None)
    parser.add_argument("--encoder", choices=["pointnet", "ptv3"],
                        default="pointnet")
    parser.add_argument("--n-points", type=int, default=16384)
    parser.add_argument("--device",
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    rng = np.random.default_rng(args.seed)

    # --- 1. Load mesh ---
    print(f"Loading mesh: {args.mesh}")
    mesh = trimesh.load(str(args.mesh), force="mesh")
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)
    if args.mesh_to_world is not None:
        T = np.load(args.mesh_to_world)
        mesh = mesh.copy()
        mesh.apply_transform(T)
    print(f"  {len(mesh.vertices)} verts, {len(mesh.faces)} faces, "
          f"AABB extent={np.round(mesh.extents, 3).tolist()}")

    # --- 2. Sample + normalize ---
    points = sample_mesh_to_points(mesh, args.n_points, rng)
    centroid = points.mean(axis=0)
    pts_n = points - centroid
    radii = np.linalg.norm(pts_n, axis=1)
    scale = float(np.percentile(radii, 99)) + 1e-8
    pts_n = pts_n / scale
    print(f"  Normalized: centroid={np.round(centroid, 3).tolist()}  "
          f"scale={scale:.3f}m")

    # --- 3. Inference ---
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt_peek = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    encoder_size = ckpt_peek.get("args", {}).get("encoder_size", "small")
    print(f"  Encoder size from checkpoint: {encoder_size}")
    del ckpt_peek
    model = Mesh2RobotModel(
        feat_dim=256, encoder=args.encoder, encoder_size=encoder_size,
    ).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    load_info = model.load_state_dict(ckpt["model_state_dict"], strict=False)
    if load_info.missing_keys:
        print(f"  [info] checkpoint missing {len(load_info.missing_keys)} keys "
              f"(e.g. {load_info.missing_keys[0]}); retrain on v3 shards "
              f"to populate them.")
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

    # Denormalize joint origins back to world coords
    pred_origins_world = pred_origins * scale + centroid

    print(f"\nPredicted segmentation:  {len(np.unique(pred_labels))} unique link IDs")
    print(f"Predicted valid joints:  {int(pred_valid.sum())}")

    # --- 4. Project to faces, split mesh ---
    print("Projecting per-point predictions onto mesh faces...")
    face_labels = project_labels_to_faces(
        sampled_points=points, sampled_labels=pred_labels,
        mesh_verts=np.asarray(mesh.vertices),
        mesh_faces=np.asarray(mesh.faces),
        k=5,
    )
    per_link_meshes = split_mesh_by_face_labels(mesh, face_labels, min_faces=30)
    print(f"  Built {len(per_link_meshes)} per-link sub-meshes "
          f"(dropped tiny links with < 30 faces)")
    for lbl, m in sorted(per_link_meshes.items()):
        print(f"    link {lbl}: {len(m.vertices)} verts / {len(m.faces)} faces")

    if len(per_link_meshes) < 2:
        raise SystemExit("Need ≥ 2 links to assemble a URDF")

    # --- 5. Build kinematic chain ---
    # Order links by world-Z of their centroids (low to high).
    link_ids_in_order = sorted(
        per_link_meshes.keys(),
        key=lambda lid: float(per_link_meshes[lid].centroid[2]),
    )
    print(f"\nLinks ordered by Z (low → high): {link_ids_in_order}")

    # Order valid joints by world-Z of their origins (low to high).
    valid_idx = np.where(pred_valid)[0]
    valid_idx_by_z = sorted(
        valid_idx, key=lambda j: float(pred_origins_world[j, 2]),
    )
    # Cap to N_links - 1 joints (a serial chain has #joints = #links - 1).
    n_joints_needed = len(link_ids_in_order) - 1
    if len(valid_idx_by_z) > n_joints_needed:
        print(f"  Predicted {len(valid_idx_by_z)} valid joints; capping to "
              f"{n_joints_needed} (one less than link count)")
        valid_idx_by_z = valid_idx_by_z[:n_joints_needed]
    elif len(valid_idx_by_z) < n_joints_needed:
        print(f"  Only {len(valid_idx_by_z)} predicted valid joints; "
              f"need {n_joints_needed}. Some links will be unconnected.")

    # Build JointEstimate list. parent/child indices are positional in
    # link_ids_in_order, so the assembler's body IDs correspond to link
    # indices in chain order.
    body_id_of_link: dict[int, int] = {
        lid: i for i, lid in enumerate(link_ids_in_order)
    }
    je_list: list[JointEstimate] = []
    for chain_i, j_slot in enumerate(valid_idx_by_z):
        parent_body = chain_i             # link[chain_i]
        child_body = chain_i + 1          # link[chain_i + 1]
        jt_int = int(pred_types[j_slot])
        jt = INT_TO_JOINT_TYPE.get(jt_int, "revolute")
        # Skip non-actuated types — assembler treats fixed/floating differently
        # but we keep them so URDF is structurally complete.
        je_list.append(JointEstimate(
            parent_body=parent_body,
            child_body=child_body,
            type=jt,
            axis=pred_axes[j_slot].copy(),
            origin=pred_origins_world[j_slot].copy(),
            angles=[0.0, 0.0],            # we don't have measured angle
            lower=-np.pi,
            upper=np.pi,
        ))
        print(f"  joint_{chain_i+1}: link[{parent_body}] → link[{child_body}]  "
              f"axis={np.round(pred_axes[j_slot], 3).tolist()}  "
              f"origin={np.round(pred_origins_world[j_slot], 3).tolist()}  "
              f"type={jt}")

    # Renumber per_link_meshes by chain order (body 0 = lowest-Z link, etc.)
    per_link_meshes_by_body = {
        body_id_of_link[lid]: per_link_meshes[lid]
        for lid in link_ids_in_order
    }
    per_link_collisions_by_body = dict(per_link_meshes_by_body)

    # --- 6. Inertials ---
    # Physics defaults (density / friction / damping / effort / velocity)
    # are fixed conservative values; nothing here is a DB lookup.
    dof = len(je_list)
    tpl = make_default_template(dof)
    print(f"\nPhysics defaults: density={tpl.density:.0f} kg/m^3 (fixed)")
    inertials = compute_link_inertials(per_link_meshes_by_body, density=tpl.density)
    for body_id, ine in sorted(inertials.items()):
        print(f"  body {body_id} mass={ine.mass:.3f} kg")

    # --- 7. Link names ---
    n_bodies = len(link_ids_in_order)
    link_name_map = {0: "link_base"}
    for i in range(1, n_bodies - 1):
        link_name_map[i] = f"link_{i}"
    if n_bodies >= 2:
        link_name_map[n_bodies - 1] = "link_tip"

    # --- 8. Assemble URDF ---
    inp = AssemblyInput(
        robot_name="ai_predicted",
        per_link_meshes=per_link_meshes_by_body,
        per_link_collisions=per_link_collisions_by_body,
        joints=je_list,
        inertials=inertials,
        template=tpl,
        body_transforms_pose0=[np.eye(4)] * n_bodies,
        link_name_map=link_name_map,
    )
    urdf_path = assemble(inp, args.output)
    print(f"\nWrote URDF: {urdf_path}")

    # Reload to verify it parses
    try:
        from yourdfpy import URDF
        r = URDF.load(str(urdf_path))
        print(f"  URDF loads OK: {len(r.link_map)} links, "
              f"{len(r.actuated_joint_names)} actuated joints")
    except Exception as e:
        print(f"  URDF load failed: {e}")


if __name__ == "__main__":
    main()
