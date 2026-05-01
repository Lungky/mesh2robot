"""Phase C smoke trainer — load shards, instantiate model, train a few steps,
print metrics, save a checkpoint.

This is a baseline trainer: PointNet encoder + multi-task heads, AdamW,
cosine LR. Once the pipeline runs end-to-end we'll swap the backbone for
Point Transformer V3 and scale up.

Usage:
    python scripts/train_model.py
        --shard-dir data/training_shards_v0
        --epochs 5
        --batch-size 8
        --n-points 16384
        --device cpu          # or cuda

Outputs:
    data/checkpoints/model_v0/checkpoint_epoch_<N>.pt
    data/checkpoints/model_v0/train_log.csv
"""

from __future__ import annotations

import argparse
import csv
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
    enumerate_robots,
    load_canonical_robot_names,
    split_robots,
    stratified_split_canonical,
)
from mesh2robot.model.losses import LossWeights, compute_losses
from mesh2robot.model.model import Mesh2RobotModel


def _to_device(batch: dict, device: torch.device) -> dict:
    return {k: v.to(device) for k, v in batch.items()}


def _format_metrics(metrics: dict[str, torch.Tensor]) -> str:
    return "  ".join(
        f"{k}={v.item():.4f}" for k, v in metrics.items()
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-dir", type=Path, action="append", required=True,
                        help="Directory of training shards. Repeat to combine sources.")
    parser.add_argument("--out-dir", type=Path,
                        default=Path("data/checkpoints/model_v0"))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--n-points", type=int, default=16384)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=0,
                        help="DataLoader workers; 0 = main process (Windows-safe)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps-per-epoch", type=int, default=0,
                        help="Cap iterations per epoch for fast smoke runs (0=no cap)")
    parser.add_argument("--encoder", choices=["pointnet", "ptv3"],
                        default="pointnet",
                        help="Backbone encoder. 'pointnet' is the 0.5M baseline; "
                             "'ptv3' is the Point Transformer V3 upgrade.")
    parser.add_argument("--encoder-size", choices=["small", "base"],
                        default="small",
                        help="PT-V3 size: 'small' (~30M, fits 24 GB GPUs) or "
                             "'base' (~120M, designed for H200-class 80+ GB GPUs). "
                             "Saved in checkpoint args; predict scripts read it back.")
    parser.add_argument("--in-memory", action="store_true",
                        help="Preload all shards into RAM at startup (5 GB). "
                             "Eliminates per-batch disk I/O — significant speedup "
                             "for any training run, at the cost of ~5 GB RAM.")
    parser.add_argument("--canonical-manifest", type=Path, default=None,
                        help="Path to research manifest (e.g. "
                             "data/robot_manifest_research.json). When given, "
                             "training restricts to robots in canonical_train_set "
                             "— a leak-free set after cross-source dedup.")
    parser.add_argument("--stratified-split", action="store_true",
                        help="Use stratified by-robot train/val split (over "
                             "fidelity x DOF buckets) instead of random. "
                             "Requires --canonical-manifest. Guarantees val "
                             "sees every (fidelity, DOF-bucket) cell.")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    print(f"Device: {device}")

    # Aggregate shards from all --shard-dir arguments
    all_shards: list[Path] = []
    for d in args.shard_dir:
        all_shards.extend(sorted(d.glob("*.npz")))
    if not all_shards:
        raise SystemExit(f"No shards found in any --shard-dir: {args.shard_dir}")
    print(f"Total shards: {len(all_shards)} across {len(args.shard_dir)} directories")

    # BY-ROBOT split: every example of a given robot goes either entirely to
    # train or entirely to val, so val measures generalization to unseen
    # robots (not just unseen *poses* of training robots).
    if args.stratified_split:
        if args.canonical_manifest is None:
            raise SystemExit("--stratified-split requires --canonical-manifest")
        train_robots, val_robots = stratified_split_canonical(
            args.canonical_manifest, all_shards,
            val_frac=args.val_frac, seed=args.seed,
        )
        print(f"Stratified split (canonical): "
              f"{len(train_robots)} train / {len(val_robots)} val")
    else:
        train_robots, val_robots = split_robots(all_shards, args.val_frac, args.seed)
        print(f"Robots: {len(train_robots)} train / {len(val_robots)} val")

        # Canonical filter applied AFTER random split so we keep the
        # original split's robot identity but drop non-canonical dups.
        if args.canonical_manifest is not None:
            canonical = load_canonical_robot_names(args.canonical_manifest)
            before_train, before_val = len(train_robots), len(val_robots)
            train_robots = train_robots & canonical
            val_robots = val_robots & canonical
            print(f"Canonical filter ({args.canonical_manifest.name}): "
                  f"train {before_train}->{len(train_robots)}, "
                  f"val {before_val}->{len(val_robots)}, "
                  f"canonical pool {len(canonical)}")

    train_ds = ShardDataset(
        all_shards, n_points=args.n_points,
        augment_subsample=True, normalize=True, rotate_aug=True,
        robot_filter=train_robots,
        in_memory=args.in_memory, verbose=True,
    )
    val_ds = ShardDataset(
        all_shards, n_points=args.n_points,
        augment_subsample=False, normalize=True, rotate_aug=False,
        robot_filter=val_robots,
        in_memory=args.in_memory,
    )
    print(f"Examples: {len(train_ds)} train / {len(val_ds)} val")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=collate_examples,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_examples,
        pin_memory=(device.type == "cuda"),
    )

    model = Mesh2RobotModel(
        feat_dim=256,
        encoder=args.encoder,
        encoder_size=args.encoder_size,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params/1e6:.1f}M  (encoder={args.encoder}, "
          f"size={args.encoder_size})")

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr,
                              weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim, T_max=max(1, args.epochs * len(train_loader)),
    )
    loss_w = LossWeights()

    log_path = args.out_dir / "train_log.csv"
    log_keys: list[str] | None = None
    log_rows: list[dict] = []

    t_run_start = time.time()
    global_step = 0
    for epoch in range(1, args.epochs + 1):
        # --- train ---
        model.train()
        t_epoch = time.time()
        train_metrics_acc: dict[str, list[float]] = {}
        for step, batch in enumerate(train_loader):
            if args.max_steps_per_epoch and step >= args.max_steps_per_epoch:
                break
            batch = _to_device(batch, device)
            pred = model(batch["points"])
            total, metrics = compute_losses(pred, batch, loss_w)
            optim.zero_grad()
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            sched.step()
            global_step += 1
            for k, v in metrics.items():
                train_metrics_acc.setdefault(k, []).append(v.item())
            if step % 10 == 0:
                lr = optim.param_groups[0]["lr"]
                print(f"  ep{epoch} step{step}: lr={lr:.5f}  "
                      f"{_format_metrics(metrics)}")

        train_avg = {k: float(np.mean(v)) for k, v in train_metrics_acc.items()}

        # --- val ---
        model.eval()
        val_metrics_acc: dict[str, list[float]] = {}
        with torch.no_grad():
            for step, batch in enumerate(val_loader):
                if args.max_steps_per_epoch and step >= args.max_steps_per_epoch:
                    break
                batch = _to_device(batch, device)
                pred = model(batch["points"])
                _, metrics = compute_losses(pred, batch, loss_w)
                for k, v in metrics.items():
                    val_metrics_acc.setdefault(k, []).append(v.item())
        val_avg = {f"val/{k}": float(np.mean(v)) for k, v in val_metrics_acc.items()}

        elapsed = time.time() - t_epoch
        print(f"\n[epoch {epoch}/{args.epochs}] {elapsed:.1f}s")
        print(f"  train: {_format_metrics({k: torch.tensor(v) for k,v in train_avg.items()})}")
        if val_avg:
            print(f"  val:   {_format_metrics({k: torch.tensor(v) for k,v in val_avg.items()})}")

        # Save checkpoint
        ckpt = {
            "epoch": epoch,
            "global_step": global_step,
            "model_state_dict": model.state_dict(),
            "optim_state_dict": optim.state_dict(),
            "args": vars(args),
        }
        torch.save(ckpt, args.out_dir / f"checkpoint_epoch_{epoch:03d}.pt")

        row = {"epoch": epoch, "elapsed_s": elapsed, **train_avg, **val_avg}
        if log_keys is None:
            log_keys = list(row.keys())
        log_rows.append({k: row.get(k) for k in log_keys})
        with open(log_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=log_keys)
            writer.writeheader()
            writer.writerows(log_rows)

    total_s = time.time() - t_run_start
    print(f"\nDone. Total training time: {total_s:.1f}s "
          f"({total_s/60:.1f} min, {total_s/3600:.2f} hr) "
          f"over {args.epochs} epochs.")
    print(f"Logs at {log_path}")


if __name__ == "__main__":
    main()
