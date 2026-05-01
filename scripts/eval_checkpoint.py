"""Evaluate a trained checkpoint against a held-out validation set.

Useful for:
  - Re-running val on a different split (e.g. after improving the
    canonical-filter dedup) to compare against the original val numbers.
  - Quick sanity check on any checkpoint without launching a fresh
    training run.

Usage:
    python scripts/eval_checkpoint.py \
        --checkpoint data/checkpoints/model_v1_pointnet/checkpoint_epoch_050.pt \
        --shard-dir data/training_shards_v1 \
        --shard-dir data/training_shards_v1_mjcf \
        --canonical-manifest data/robot_manifest_research.json \
        --encoder pointnet \
        --val-frac 0.1 --seed 0 --batch-size 8 --n-points 16384
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mesh2robot.model.dataset import (
    ShardDataset,
    collate_examples,
    load_canonical_robot_names,
    split_robots,
    stratified_split_canonical,
)
from mesh2robot.model.losses import LossWeights, compute_losses
from mesh2robot.model.model import Mesh2RobotModel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--shard-dir", type=Path, action="append", required=True)
    parser.add_argument("--canonical-manifest", type=Path, default=None)
    parser.add_argument("--stratified-split", action="store_true",
                        help="Use stratified by-robot split (requires "
                             "--canonical-manifest).")
    parser.add_argument("--encoder", choices=["pointnet", "ptv3"],
                        default="pointnet")
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--n-points", type=int, default=16384)
    parser.add_argument("--device",
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--in-memory", action="store_true")
    parser.add_argument("--max-batches", type=int, default=0,
                        help="Cap number of val batches (0 = no cap).")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    print(f"Device: {device}")

    # Aggregate shards
    all_shards: list[Path] = []
    for d in args.shard_dir:
        all_shards.extend(sorted(d.glob("*.npz")))
    if not all_shards:
        raise SystemExit(f"No shards found in any --shard-dir: {args.shard_dir}")
    print(f"Total shards: {len(all_shards)}")

    # Same by-robot split as training
    if args.stratified_split:
        if args.canonical_manifest is None:
            raise SystemExit("--stratified-split requires --canonical-manifest")
        train_robots, val_robots = stratified_split_canonical(
            args.canonical_manifest, all_shards,
            val_frac=args.val_frac, seed=args.seed,
        )
        print(f"Stratified canonical split: "
              f"{len(train_robots)} train / {len(val_robots)} val")
    else:
        train_robots, val_robots = split_robots(all_shards, args.val_frac, args.seed)
        print(f"Robots (raw): {len(train_robots)} train / {len(val_robots)} val")
        if args.canonical_manifest is not None:
            canonical = load_canonical_robot_names(args.canonical_manifest)
            before_train, before_val = len(train_robots), len(val_robots)
            train_robots = train_robots & canonical
            val_robots = val_robots & canonical
            print(f"Canonical filter: train {before_train}->{len(train_robots)}, "
                  f"val {before_val}->{len(val_robots)}, "
                  f"canonical pool {len(canonical)}")

    val_ds = ShardDataset(
        all_shards, n_points=args.n_points,
        augment_subsample=False, normalize=True, rotate_aug=False,
        robot_filter=val_robots,
        in_memory=args.in_memory, verbose=True,
    )
    print(f"Val examples: {len(val_ds)}")

    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=0, collate_fn=collate_examples,
        pin_memory=(device.type == "cuda"),
    )

    # Load model + checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    model = Mesh2RobotModel(feat_dim=256, encoder=args.encoder).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Checkpoint epoch: {ckpt.get('epoch', '?')}  "
          f"step: {ckpt.get('global_step', '?')}")

    # Run val pass
    loss_w = LossWeights()
    metrics_acc: dict[str, list[float]] = {}
    t0 = time.time()
    n_batches = 0
    with torch.no_grad():
        for step, batch in enumerate(val_loader):
            if args.max_batches and step >= args.max_batches:
                break
            batch = {k: v.to(device) for k, v in batch.items()}
            pred = model(batch["points"])
            _, metrics = compute_losses(pred, batch, loss_w)
            for k, v in metrics.items():
                metrics_acc.setdefault(k, []).append(v.item())
            n_batches += 1
            if step % 25 == 0 and step > 0:
                print(f"  batch {step}/{len(val_loader)}")
    elapsed = time.time() - t0
    print(f"\nVal pass: {n_batches} batches in {elapsed:.1f}s")

    print("\n=== Validation metrics ===")
    for k in sorted(metrics_acc.keys()):
        v = np.array(metrics_acc[k])
        print(f"  {k:20s}  mean={v.mean():8.4f}  std={v.std():.4f}  n={len(v)}")


if __name__ == "__main__":
    main()
