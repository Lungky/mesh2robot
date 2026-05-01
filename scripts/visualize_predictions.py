"""Visualize model predictions on training-shard examples.

Loads a checkpoint, runs inference on N examples from given shards, and
saves color-coded point clouds (as GLB) showing:
  - ground-truth per-vertex segmentation (one color per link index)
  - predicted per-vertex segmentation (same palette so they're comparable)

Also overlays predicted joint axes/origins as small arrow primitives.

Use this to sanity-check what the model is actually learning beyond
aggregate metrics — e.g. is it picking up gross robot topology? Is the
segmentation tracking real link boundaries or just clustering by spatial
position?

Usage:
    python scripts/visualize_predictions.py
        --checkpoint data/checkpoints/model_v1_smoke/checkpoint_epoch_001.pt
        --shard-dir data/training_shards_v1
        --out-dir data/visualizations
        --n 8
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import trimesh
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mesh2robot.model.dataset import (
    K_LINKS_MAX,
    J_MAX,
    ShardDataset,
    collate_examples,
)
from mesh2robot.model.model import Mesh2RobotModel


def _make_palette(n: int = K_LINKS_MAX, seed: int = 42) -> np.ndarray:
    """Distinct, deterministic colors per link index. RGB in [0,1]."""
    rng = np.random.default_rng(seed)
    # Use HSV for well-spaced hues, then convert to RGB.
    hues = (np.arange(n) * 0.61803398875) % 1.0   # golden-ratio hopping
    sats = 0.7 + rng.uniform(-0.1, 0.1, n)
    vals = 0.85 + rng.uniform(-0.1, 0.05, n)
    out = np.zeros((n, 3), dtype=np.float32)
    for i in range(n):
        out[i] = _hsv_to_rgb(hues[i], sats[i], vals[i])
    return out


def _hsv_to_rgb(h: float, s: float, v: float) -> tuple[float, float, float]:
    i = int(h * 6.0)
    f = h * 6.0 - i
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))
    i %= 6
    return [(v, t, p), (q, v, p), (p, v, t),
            (p, q, v), (t, p, v), (v, p, q)][i]


def _points_to_scene(
    points: np.ndarray,
    labels: np.ndarray,
    palette: np.ndarray,
    arrow_axes: np.ndarray | None = None,
    arrow_origins: np.ndarray | None = None,
    arrow_valid: np.ndarray | None = None,
    sphere_radius: float = 0.005,
) -> trimesh.Scene:
    """Build a Scene where each unique label is a separate mesh node.

    Points are rendered as small spheres (deterministically sampled across
    the cloud) so you can SEE them — raw point clouds in GLB usually
    render invisible. We keep at most ~3000 points per cloud to keep file
    sizes reasonable.
    """
    n_keep = min(3000, len(points))
    if len(points) > n_keep:
        idx = np.random.default_rng(0).choice(len(points), n_keep, replace=False)
        points = points[idx]
        labels = labels[idx]

    scene = trimesh.Scene()
    # Group points by label, build one icosphere mesh per label using
    # instancing-by-translation-plus-color.
    unique = np.unique(labels)
    for lbl in unique:
        if lbl < 0:
            continue
        mask = labels == lbl
        pts_lbl = points[mask]
        if len(pts_lbl) == 0:
            continue
        # One sphere prototype, then duplicate by translation
        proto = trimesh.creation.icosphere(subdivisions=1, radius=sphere_radius)
        v_all = np.concatenate(
            [proto.vertices + p for p in pts_lbl], axis=0,
        )
        f_all = np.concatenate(
            [proto.faces + i * len(proto.vertices) for i in range(len(pts_lbl))],
            axis=0,
        )
        m = trimesh.Trimesh(vertices=v_all, faces=f_all, process=False)
        rgb = palette[lbl % len(palette)]
        m.visual = trimesh.visual.TextureVisuals(
            material=trimesh.visual.material.PBRMaterial(
                baseColorFactor=[rgb[0], rgb[1], rgb[2], 1.0],
                name=f"link_{int(lbl)}",
            ),
        )
        scene.add_geometry(m, node_name=f"link_{int(lbl)}")

    # Arrows for joints
    if arrow_axes is not None and arrow_origins is not None:
        for i in range(len(arrow_axes)):
            if arrow_valid is not None and not arrow_valid[i]:
                continue
            ax = arrow_axes[i]
            n = float(np.linalg.norm(ax))
            if n < 1e-6:
                continue
            ax = ax / n
            origin = arrow_origins[i]
            arrow = trimesh.creation.cylinder(
                radius=sphere_radius * 0.4,
                segment=[origin, origin + 0.15 * ax],
            )
            arrow.visual = trimesh.visual.TextureVisuals(
                material=trimesh.visual.material.PBRMaterial(
                    baseColorFactor=[0.0, 0.0, 0.0, 1.0],
                    name=f"joint_{i}",
                ),
            )
            scene.add_geometry(arrow, node_name=f"joint_{i}")

    return scene


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--shard-dir", action="append", required=True,
                        help="Repeatable")
    parser.add_argument("--out-dir", type=Path,
                        default=Path("data/visualizations"))
    parser.add_argument("--n", type=int, default=8,
                        help="Number of examples to visualize")
    parser.add_argument("--n-points", type=int, default=4096,
                        help="Sample size for visualization (smaller is OK)")
    parser.add_argument("--device",
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    print(f"Device: {device}")

    # Collect shards
    shard_paths: list[Path] = []
    for d in args.shard_dir:
        shard_paths.extend(sorted(Path(d).glob("*.npz")))
    if not shard_paths:
        raise SystemExit("No shards found")
    print(f"Loaded {len(shard_paths)} shards")

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = Mesh2RobotModel(feat_dim=256).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded checkpoint from epoch {ckpt.get('epoch', '?')}")

    # Use validation-mode dataset (no rotation aug, no random subsample) so
    # results are deterministic.
    ds = ShardDataset(
        shard_paths, n_points=args.n_points,
        augment_subsample=False,
        normalize=True,    # train-time setting
        rotate_aug=False,
    )
    print(f"Dataset: {len(ds)} examples")

    palette = _make_palette()

    # Pull a deterministic spread of examples
    n = min(args.n, len(ds))
    indices = np.linspace(0, len(ds) - 1, n).astype(int)

    for k, idx in enumerate(indices):
        sample = ds[int(idx)]
        # Add batch dim
        batch = {k: v.unsqueeze(0).to(device) for k, v in sample.items()}
        with torch.no_grad():
            pred = model(batch["points"])

        pts = sample["points"].cpu().numpy()                  # (N, 3)
        gt_labels = sample["point_labels"].cpu().numpy()      # (N,)
        pred_labels = pred["seg_logits"][0].argmax(dim=-1).cpu().numpy()  # (N,)

        gt_axes = sample["joint_axes"].cpu().numpy()          # (J, 3)
        gt_origins = sample["joint_origins"].cpu().numpy()    # (J, 3)
        gt_valid = sample["joint_valid"].cpu().numpy()        # (J,)

        pred_axes = pred["axis"][0].cpu().numpy()
        pred_axes = pred_axes / (np.linalg.norm(pred_axes, axis=-1, keepdims=True) + 1e-9)
        pred_origins = pred["origin"][0].cpu().numpy()
        pred_valid = (torch.sigmoid(pred["valid_logit"][0]).cpu().numpy() > 0.5)

        # Save GT scene
        gt_scene = _points_to_scene(pts, gt_labels, palette,
                                     gt_axes, gt_origins, gt_valid)
        gt_scene.export(args.out_dir / f"sample_{k:03d}_idx{idx:06d}_gt.glb")

        # Save predicted scene (same color palette so links can be compared)
        pred_scene = _points_to_scene(pts, pred_labels, palette,
                                        pred_axes, pred_origins, pred_valid)
        pred_scene.export(args.out_dir / f"sample_{k:03d}_idx{idx:06d}_pred.glb")

        # Quick summary
        gt_unique = sorted(set(int(l) for l in gt_labels if l >= 0))
        pred_unique = sorted(set(int(l) for l in pred_labels if l >= 0))
        n_correct = int(((pred_labels == gt_labels) & (gt_labels >= 0)).sum())
        n_valid = int((gt_labels >= 0).sum())
        print(f"  [{k+1}/{n}] idx={idx:6d}  "
              f"gt_links={gt_unique}  pred_links={pred_unique}  "
              f"seg_acc={n_correct/max(1,n_valid):.2%}  "
              f"gt_valid_joints={int(gt_valid.sum())}  "
              f"pred_valid_joints={int(pred_valid.sum())}")

    print(f"\nSaved {2 * n} GLBs to {args.out_dir}")


if __name__ == "__main__":
    main()
