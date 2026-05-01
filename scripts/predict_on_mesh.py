"""Run a trained checkpoint on a real mesh file (.obj/.ply/.stl) and save a
GLB visualization of the predicted per-vertex segmentation.

Pipeline:
  1. Load mesh with trimesh, sample N points uniformly over its surface area
  2. Apply the same normalization the training Dataset uses (centroid +
     99th-percentile radius scale)
  3. Forward through the model
  4. Save a colored point cloud as GLB (each predicted link = own material)

Use this to test whether a model trained on synthetic URDF / MJCF data
transfers to real MILO scans.

Usage:
    python scripts/predict_on_mesh.py
        --checkpoint data/checkpoints/model_v1_pointnet/checkpoint_epoch_050.pt
        --mesh input/test_2/milo/xarm6_clean.obj
        --out data/visualizations/test_2_baseline.glb
        [--n-points 16384]
        [--encoder pointnet|ptv3]
        [--mesh-to-world input/test_2/T_cleaned_to_original.npy]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mesh2robot.model.dataset import K_LINKS_MAX
from mesh2robot.model.model import Mesh2RobotModel


def _hsv_to_rgb(h: float, s: float, v: float) -> tuple[float, float, float]:
    i = int(h * 6.0)
    f = h * 6.0 - i
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))
    i %= 6
    return [(v, t, p), (q, v, p), (p, v, t),
            (p, q, v), (t, p, v), (v, p, q)][i]


def _make_palette(n: int = K_LINKS_MAX, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    hues = (np.arange(n) * 0.61803398875) % 1.0
    sats = 0.7 + rng.uniform(-0.1, 0.1, n)
    vals = 0.85 + rng.uniform(-0.1, 0.05, n)
    out = np.zeros((n, 3), dtype=np.float32)
    for i in range(n):
        out[i] = _hsv_to_rgb(hues[i], sats[i], vals[i])
    return out


def sample_mesh_to_points(
    mesh: trimesh.Trimesh, n_points: int, rng: np.random.Generator,
) -> np.ndarray:
    """Sample N points uniformly over the mesh surface area."""
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


def project_labels_to_mesh_faces(
    sampled_points: np.ndarray,        # (N_sampled, 3) — points fed to the model
    sampled_labels: np.ndarray,        # (N_sampled,)   — predicted label per point
    mesh_verts: np.ndarray,            # (V, 3)
    mesh_faces: np.ndarray,            # (F, 3) of vertex indices
    k: int = 5,
) -> np.ndarray:
    """For each mesh face, vote a label by KNN over the sampled points.

    Returns (F,) array of per-face predicted labels.
    """
    from scipy.spatial import cKDTree
    face_centers = mesh_verts[mesh_faces].mean(axis=1)   # (F, 3)
    tree = cKDTree(sampled_points)
    _, idx = tree.query(face_centers, k=min(k, len(sampled_points)))
    if idx.ndim == 1:
        return sampled_labels[idx]
    # Majority vote per face
    K = sampled_labels.max() + 1
    if K < 1:
        K = 1
    counts = np.zeros((len(mesh_faces), K + 1), dtype=np.int32)
    for kk in range(idx.shape[1]):
        labs = sampled_labels[idx[:, kk]]
        valid = labs >= 0
        counts[np.where(valid)[0], labs[valid]] += 1
    return counts.argmax(axis=1)


def build_solid_segmented_scene(
    mesh_verts: np.ndarray,
    mesh_faces: np.ndarray,
    face_labels: np.ndarray,
    palette: np.ndarray,
) -> trimesh.Scene:
    """Build a Scene with one solid sub-mesh per predicted label, each
    given its own PBR material. The full robot surface is rendered as a
    proper solid mesh — much easier to read than per-point spheres."""
    scene = trimesh.Scene()
    unique = np.unique(face_labels)
    for lbl in unique:
        if lbl < 0:
            continue
        face_mask = face_labels == lbl
        if not face_mask.any():
            continue
        sub_faces = mesh_faces[face_mask]
        used = np.unique(sub_faces)
        remap = -np.ones(len(mesh_verts), dtype=np.int64)
        remap[used] = np.arange(len(used))
        m = trimesh.Trimesh(
            vertices=mesh_verts[used],
            faces=remap[sub_faces],
            process=False,
        )
        rgb = palette[int(lbl) % len(palette)]
        m.visual = trimesh.visual.TextureVisuals(
            material=trimesh.visual.material.PBRMaterial(
                baseColorFactor=[float(rgb[0]), float(rgb[1]), float(rgb[2]), 1.0],
                name=f"pred_link_{int(lbl)}",
            ),
        )
        scene.add_geometry(m, node_name=f"pred_link_{int(lbl)}")
    return scene


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--n-points", type=int, default=16384)
    parser.add_argument("--encoder", choices=["pointnet", "ptv3"],
                        default="pointnet")
    parser.add_argument("--mesh-to-world", type=Path, default=None,
                        help="Optional 4x4 .npy transform applied to mesh")
    parser.add_argument("--device",
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    rng = np.random.default_rng(args.seed)

    # Load mesh
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

    # Sample points
    points = sample_mesh_to_points(mesh, args.n_points, rng)
    print(f"  Sampled {len(points)} surface points")

    # Normalize identically to ShardDataset (centroid-shift + 99th-pct scale)
    centroid = points.mean(axis=0)
    pts_n = points - centroid
    radii = np.linalg.norm(pts_n, axis=1)
    scale = float(np.percentile(radii, 99)) + 1e-8
    pts_n = pts_n / scale
    print(f"  Normalized: centroid={np.round(centroid, 3).tolist()}  "
          f"scale={scale:.3f}m")

    # Build model + load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    model = Mesh2RobotModel(feat_dim=256, encoder=args.encoder).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"  Loaded epoch {ckpt.get('epoch', '?')} ({args.encoder} encoder)")

    # Inference
    pts_t = torch.from_numpy(pts_n.astype(np.float32)).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = model(pts_t)
    pred_labels = pred["seg_logits"][0].argmax(dim=-1).cpu().numpy()
    pred_axes = pred["axis"][0].cpu().numpy()
    pred_axes /= (np.linalg.norm(pred_axes, axis=-1, keepdims=True) + 1e-9)
    pred_origins = pred["origin"][0].cpu().numpy()
    pred_valid = (torch.sigmoid(pred["valid_logit"][0]).cpu().numpy() > 0.5)
    pred_types = pred["type_logits"][0].argmax(dim=-1).cpu().numpy()

    # Report
    unique = sorted(set(int(l) for l in pred_labels if l >= 0))
    print(f"\nPredicted segmentation:")
    print(f"  {len(unique)} unique link IDs: {unique}")
    print(f"  predicted valid joints: {int(pred_valid.sum())}")
    print(f"  joint-type counts:")
    type_names = {0: "revolute", 1: "continuous", 2: "prismatic",
                   3: "fixed", 4: "floating", 5: "planar"}
    for jt, name in type_names.items():
        n = int(((pred_types == jt) & pred_valid).sum())
        if n > 0:
            print(f"    {name:12s}  {n}")

    # Project per-point predictions onto the original mesh's faces by
    # KNN majority vote. Sampled points were in WORLD coords (not the
    # normalized version we fed to the model), so KNN works directly
    # against the mesh's world-frame faces.
    palette = _make_palette()
    print("Projecting per-point predictions onto mesh faces (KNN vote)...")
    face_labels = project_labels_to_mesh_faces(
        sampled_points=points,            # world-frame
        sampled_labels=pred_labels,
        mesh_verts=np.asarray(mesh.vertices),
        mesh_faces=np.asarray(mesh.faces),
        k=5,
    )
    scene = build_solid_segmented_scene(
        mesh_verts=np.asarray(mesh.vertices),
        mesh_faces=np.asarray(mesh.faces),
        face_labels=face_labels,
        palette=palette,
    )

    # Joint axes/origins were predicted in NORMALIZED space (since the
    # model was given normalized points). Reverse the normalization so
    # the arrows align with the world-coord solid mesh.
    pred_origins_world = pred_origins * scale + centroid
    arrow_len = 0.10 * scale     # reasonable visible length

    for j in range(len(pred_axes)):
        if not pred_valid[j]:
            continue
        ax = pred_axes[j]
        origin = pred_origins_world[j]
        n = float(np.linalg.norm(ax))
        if n < 1e-6:
            continue
        ax = ax / n
        try:
            arrow = trimesh.creation.cylinder(
                radius=0.005 * scale,
                segment=[origin, origin + arrow_len * ax],
            )
            arrow.visual = trimesh.visual.TextureVisuals(
                material=trimesh.visual.material.PBRMaterial(
                    baseColorFactor=[0.0, 0.0, 0.0, 1.0],
                    name=f"joint_{j}",
                ),
            )
            scene.add_geometry(arrow, node_name=f"joint_{j}")
        except Exception:
            pass

    scene.export(args.out)
    n_colored = len(np.unique(face_labels))
    print(f"\nSaved {args.out}")
    print(f"  Solid mesh ({len(mesh.faces)} faces) colored as "
          f"{n_colored} links + {int(pred_valid.sum())} joint arrows")


if __name__ == "__main__":
    main()
