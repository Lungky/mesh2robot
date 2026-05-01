"""Compute summary statistics over a directory of training shards.

Useful for sanity-checking the synthetic data before kicking off training:
  - example count, shard count, total disk usage
  - per-robot example count
  - joint count distribution
  - link count distribution (per example, # unique link labels)
  - point-label class balance (any link massively over- or under-represented?)
  - mesh-points magnitude distribution (catches bad scales / outlier robots)
  - joint origin magnitude distribution

Usage:
    python scripts/inspect_training_shards.py
        --shard-dir data/training_shards_v1
        [--shard-dir data/training_shards_v1_mjcf]
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--shard-dir", action="append", required=True,
        help="Directory of .npz shards. Repeatable to combine multiple sources.",
    )
    parser.add_argument(
        "--out", type=Path, default=Path("data/shard_stats.json"),
        help="Where to write the JSON summary",
    )
    args = parser.parse_args()

    shard_paths: list[Path] = []
    for d in args.shard_dir:
        shard_paths.extend(sorted(Path(d).glob("*.npz")))

    if not shard_paths:
        raise SystemExit("No shards found in any directory.")

    print(f"Found {len(shard_paths)} shards across "
          f"{len(args.shard_dir)} directories.")

    n_examples = 0
    robots: set[str] = set()
    examples_per_robot: Counter = Counter()
    joint_count_dist: Counter = Counter()
    link_count_dist: Counter = Counter()
    label_counts: Counter = Counter()
    point_norms: list[float] = []
    origin_norms: list[float] = []
    joint_type_counts: Counter = Counter()
    total_bytes = 0

    for p in shard_paths:
        total_bytes += p.stat().st_size
        with np.load(p, allow_pickle=True) as z:
            B = int(z["points"].shape[0])
            n_examples += B
            names = list(z["names"])
            robot_idx = z["robot_idx"]
            joint_valid = z["joint_valid"]
            joint_types = z["joint_types"]
            point_labels = z["point_labels"]
            joint_origins = z["joint_origins_world"]
            points = z["points"]
            for b in range(B):
                rname = names[robot_idx[b]]
                robots.add(str(rname))
                examples_per_robot[str(rname)] += 1
                joint_count_dist[int(joint_valid[b].sum())] += 1
                link_count_dist[int(np.unique(point_labels[b]).size)] += 1
                # subsample 256 points/example to keep this fast
                idx_sub = np.random.choice(point_labels[b].size,
                                            min(256, point_labels[b].size),
                                            replace=False)
                for lbl in point_labels[b][idx_sub]:
                    label_counts[int(lbl)] += 1
                # Magnitude stats: max distance of points from centroid
                ctr = points[b].mean(axis=0)
                pn = float(np.linalg.norm(points[b] - ctr, axis=1).max())
                point_norms.append(pn)
                # Joint origin norms (only valid slots)
                for j in range(joint_valid.shape[1]):
                    if not joint_valid[b, j]:
                        continue
                    origin_norms.append(float(np.linalg.norm(joint_origins[b, j])))
                    joint_type_counts[int(joint_types[b, j])] += 1

    point_norms_arr = np.asarray(point_norms)
    origin_norms_arr = np.asarray(origin_norms)

    print(f"\n=== Summary ===")
    print(f"Total examples: {n_examples}")
    print(f"Total disk:     {total_bytes / 1e9:.2f} GB")
    print(f"Unique robots:  {len(robots)}")
    print(f"Avg examples/robot: {n_examples / len(robots):.1f}")
    print(f"\nJoint count distribution (top 10):")
    for jc, n in joint_count_dist.most_common(10):
        print(f"  J={jc:3d}  {n} examples ({100 * n / n_examples:.1f}%)")
    print(f"\nLink count distribution (top 10):")
    for lc, n in link_count_dist.most_common(10):
        print(f"  links={lc:3d}  {n} examples ({100 * n / n_examples:.1f}%)")
    print(f"\nPer-point label class counts (top 16):")
    total_label = sum(label_counts.values())
    for lbl, n in sorted(label_counts.items())[:16]:
        print(f"  link_idx={lbl:3d}  {n} pts ({100 * n / total_label:.1f}%)")
    print(f"\nJoint type distribution:")
    type_names = {-1: "padding", 0: "revolute", 1: "continuous",
                   2: "prismatic", 3: "fixed", 4: "floating", 5: "planar"}
    for jt, n in sorted(joint_type_counts.items()):
        print(f"  {type_names.get(jt, str(jt)):12s}  {n}")
    print(f"\nPoint-cloud max-radius stats (m):")
    print(f"  min={point_norms_arr.min():.3f}  "
          f"p50={np.median(point_norms_arr):.3f}  "
          f"p90={np.percentile(point_norms_arr, 90):.3f}  "
          f"max={point_norms_arr.max():.3f}")
    print(f"\nJoint origin norm stats (m):")
    print(f"  min={origin_norms_arr.min():.3f}  "
          f"p50={np.median(origin_norms_arr):.3f}  "
          f"p90={np.percentile(origin_norms_arr, 90):.3f}  "
          f"max={origin_norms_arr.max():.3f}")
    print(f"\nTop 10 robots by example count:")
    for rname, n in examples_per_robot.most_common(10):
        print(f"  {rname:50s}  {n}")
    print(f"\nBottom 10 robots by example count:")
    for rname, n in sorted(examples_per_robot.items(), key=lambda x: x[1])[:10]:
        print(f"  {rname:50s}  {n}")

    # Save to JSON for programmatic use
    args.out.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "total_examples": n_examples,
        "total_bytes": total_bytes,
        "shard_count": len(shard_paths),
        "unique_robots": len(robots),
        "joint_count_dist": dict(joint_count_dist),
        "link_count_dist": dict(link_count_dist),
        "joint_type_counts": dict(joint_type_counts),
        "point_radius_p50_m": float(np.median(point_norms_arr)),
        "point_radius_max_m": float(point_norms_arr.max()),
        "origin_norm_p50_m": float(np.median(origin_norms_arr)),
        "origin_norm_max_m": float(origin_norms_arr.max()),
        "examples_per_robot": dict(examples_per_robot.most_common()),
    }
    args.out.write_text(json.dumps(summary, indent=2))
    print(f"\nWrote summary JSON to {args.out}")


if __name__ == "__main__":
    main()
