"""Phase B driver — walk the trainable URDFs in `data/robot_manifest.json`,
generate N training examples per robot (random configs + augmentations),
and write them to disk as `.npz` shards under `--output-dir`.

Each shard packs many examples for fast loading. Within a shard the layout is:

    {
        "points":            (B, N, 3) float32        # surface points (world)
        "point_labels":      (B, N)    int32          # per-point link index
        "joint_axes_world":  (B, J_max, 3) float32    # padded
        "joint_origins_world":(B, J_max, 3) float32   # padded
        "joint_types":       (B, J_max) int32         # padded
        "joint_topology":    (B, J_max, 2) int32      # (parent_link, child_link)
        "joint_valid":       (B, J_max) bool          # mask: 1 if joint exists
        "robot_idx":         (B,)      int32          # index into a per-shard "names" list
        "names":             list[str]                # source URDF identifiers
        "vendors":           list[str]
    }

Variable-length items (B robots in a shard each with different J) are
padded to the shard's max J and a `joint_valid` mask is provided.

Usage:
    python scripts/generate_training_data.py
        --manifest data/robot_manifest.json
        --raw-dir data/raw_robots
        --output-dir data/training_shards
        --n-configs 50      # examples per robot
        --n-points 16384
        --shard-size 32
        --workers 0          # 0 = single process (Windows default; safest)

Resumes safely: existing shards are detected by filename and skipped.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mesh2robot.data_gen import (
    articulate_and_label,
    load_robot,
    sample_point_cloud,
    sample_random_config,
)
from mesh2robot.data_gen.augment import AugmentConfig, apply as apply_augment
from mesh2robot.data_gen.mjcf_loader import (
    articulate_and_label_mjcf,
    load_robot_mjcf,
    sample_random_config_mjcf,
)


def _is_trainable(entry: dict, format_filter: list[str] | None = None) -> bool:
    if entry.get("status") != "ok":
        return False
    if entry.get("dof", 0) < 1:
        return False
    if format_filter and entry.get("format") not in format_filter:
        return False
    res = entry.get("meshes_resolved", 0)
    unres = entry.get("meshes_unresolved", 0)
    if res + unres == 0:
        return False
    return res / (res + unres) >= 0.8


def _sample_one(
    robot,
    n_points: int,
    rng: np.random.Generator,
    augment_cfg: AugmentConfig | None,
    mesh_cache: dict,
    is_mjcf: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """Generate ONE training example. Returns None on failure."""
    try:
        if is_mjcf:
            cfg = sample_random_config_mjcf(robot, rng=rng)
            mesh, vlabels, axes_w, origins_w, jtypes, topo, jlimits = \
                articulate_and_label_mjcf(robot, cfg)
        else:
            cfg = sample_random_config(robot, rng=rng)
            mesh, vlabels, axes_w, origins_w, jtypes, topo, jlimits = \
                articulate_and_label(robot, cfg, mesh_cache=mesh_cache)
    except Exception:
        return None
    if len(mesh.vertices) == 0:
        return None

    points, plabels = sample_point_cloud(mesh, vlabels, n_points=n_points, rng=rng)
    if len(points) == 0:
        return None

    if augment_cfg is not None:
        points, plabels, axes_w, origins_w, _ = apply_augment(
            points, plabels, axes_w, origins_w, augment_cfg, rng,
        )
        if len(points) == 0:
            return None
        # Augmentations may drop points; pad/repeat back to n_points so the
        # tensor shape stays fixed.
        if len(points) < n_points:
            pad_idx = rng.choice(len(points), n_points - len(points), replace=True)
            points = np.concatenate([points, points[pad_idx]], axis=0)
            plabels = np.concatenate([plabels, plabels[pad_idx]], axis=0)
        elif len(points) > n_points:
            keep = rng.choice(len(points), n_points, replace=False)
            points = points[keep]
            plabels = plabels[keep]

    return points, plabels, axes_w, origins_w, jtypes, topo, jlimits


def _shard_path(out_dir: Path, shard_id: int) -> Path:
    return out_dir / f"shard_{shard_id:06d}.npz"


def _process_one_robot(
    raw_dir_str: str,
    entry: dict,
    n_configs: int,
    n_points: int,
    augment: bool,
    seed: int,
    robot_idx: int,
) -> tuple[list[dict], int, str]:
    """Worker function: load one robot, generate `n_configs` examples,
    return them. Designed to be called via ProcessPoolExecutor.

    Returns (examples, n_failed_configs, error_msg). examples may be empty
    if the URDF failed to load.
    """
    raw_dir = Path(raw_dir_str)
    rng = np.random.default_rng(seed + robot_idx * 100003)
    aug_cfg = AugmentConfig() if augment else None
    examples: list[dict] = []
    n_failed = 0

    full_path = raw_dir / entry["path"]
    is_mjcf = entry.get("format") == "mjcf"
    try:
        if is_mjcf:
            robot = load_robot_mjcf(full_path)
        else:
            robot = load_robot(full_path)
    except Exception as e:
        return [], n_configs, f"load_failed: {type(e).__name__}: {str(e)[:100]}"

    mesh_cache: dict = {}
    # Include parent directory in the name so robots from different
    # directories that share a filename (e.g. robosuite/baxter/robot.xml
    # vs robosuite/iiwa/robot.xml) don't collapse to the same identity.
    # Earlier shards used just the stem, conflating ~21 distinct robots.
    p = Path(entry['path'])
    name_safe = f"{entry['source']}/{p.parent.name}/{p.stem}"
    vendor = entry.get("vendor", "")

    for k in range(n_configs):
        res = _sample_one(robot, n_points, rng, aug_cfg, mesh_cache, is_mjcf=is_mjcf)
        if res is None:
            n_failed += 1
            continue
        points, plabels, axes_w, origins_w, jtypes, topo, jlimits = res
        examples.append({
            "name": name_safe,
            "vendor": vendor,
            "points": points.astype(np.float32),
            "plabels": plabels.astype(np.int32),
            "joint_axes": axes_w.astype(np.float32),
            "joint_origins": origins_w.astype(np.float32),
            "joint_types": jtypes.astype(np.int32),
            "topology": topo.astype(np.int32),
            "joint_limits": jlimits.astype(np.float32),
        })
    return examples, n_failed, ""


def _pack_shard(
    examples: list[dict], out_path: Path,
) -> None:
    """Pack a list of per-example dicts into one .npz file with padded
    joint dimension and a joint_valid mask."""
    if not examples:
        return
    n_pts = examples[0]["points"].shape[0]
    j_max = max(e["joint_axes"].shape[0] for e in examples)
    B = len(examples)

    points = np.zeros((B, n_pts, 3), dtype=np.float32)
    plabels = np.zeros((B, n_pts), dtype=np.int32)
    axes = np.zeros((B, j_max, 3), dtype=np.float32)
    origins = np.zeros((B, j_max, 3), dtype=np.float32)
    jtypes = np.full((B, j_max), -1, dtype=np.int32)
    topo = np.full((B, j_max, 2), -1, dtype=np.int32)
    jvalid = np.zeros((B, j_max), dtype=bool)
    jlimits = np.zeros((B, j_max, 2), dtype=np.float32)
    robot_idx = np.zeros(B, dtype=np.int32)

    name_to_idx: dict[str, int] = {}
    names: list[str] = []
    vendors: list[str] = []
    for i, e in enumerate(examples):
        points[i] = e["points"]
        plabels[i] = e["plabels"]
        J = e["joint_axes"].shape[0]
        axes[i, :J] = e["joint_axes"]
        origins[i, :J] = e["joint_origins"]
        jtypes[i, :J] = e["joint_types"]
        topo[i, :J] = e["topology"]
        jvalid[i, :J] = True
        # joint_limits may be missing on legacy callers — fall back to zeros.
        jlims_e = e.get("joint_limits")
        if jlims_e is not None:
            jlimits[i, :J] = jlims_e[:J]
        if e["name"] not in name_to_idx:
            name_to_idx[e["name"]] = len(names)
            names.append(e["name"])
            vendors.append(e["vendor"])
        robot_idx[i] = name_to_idx[e["name"]]

    np.savez_compressed(
        out_path,
        points=points, point_labels=plabels,
        joint_axes_world=axes, joint_origins_world=origins,
        joint_types=jtypes, joint_topology=topo, joint_valid=jvalid,
        joint_limits=jlimits,
        robot_idx=robot_idx,
        names=np.array(names),
        vendors=np.array(vendors),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path,
                        default=Path("data/robot_manifest.json"))
    parser.add_argument("--raw-dir", type=Path,
                        default=Path("data/raw_robots"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("data/training_shards"))
    parser.add_argument("--n-configs", type=int, default=50,
                        help="Examples per robot")
    parser.add_argument("--n-points", type=int, default=16384,
                        help="Points per example")
    parser.add_argument("--shard-size", type=int, default=32,
                        help="Examples packed per shard file")
    parser.add_argument("--limit-robots", type=int, default=0,
                        help="If > 0, only process this many robots (smoke run)")
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--format", choices=["urdf", "mjcf", "all"],
                        default="urdf",
                        help="Which formats to process; URDF only by default")
    parser.add_argument("--workers", type=int, default=0,
                        help="Parallel processes; 0 = single-process (default), "
                             "-1 = os.cpu_count()-1, >0 = explicit")
    args = parser.parse_args()
    if args.workers < 0:
        args.workers = max(1, (os.cpu_count() or 2) - 1)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(args.manifest.read_text())

    fmt_filter = None if args.format == "all" else [args.format]
    trainable = [e for e in manifest if _is_trainable(e, fmt_filter)]
    if args.limit_robots > 0:
        trainable = trainable[: args.limit_robots]
    print(f"Trainable robots after filter: {len(trainable)}")
    print(f"Target examples: {len(trainable)} × {args.n_configs} = "
          f"{len(trainable) * args.n_configs}")

    aug_cfg = None if args.no_augment else AugmentConfig()
    rng_global = np.random.default_rng(args.seed)

    examples_buffer: list[dict] = []
    shard_id = 0
    n_total = 0
    n_failed = 0
    t_start = time.time()

    def _flush(buffer: list[dict], sid: int) -> int:
        """Write `shard_size`-sized chunks from buffer to disk; return new sid."""
        while len(buffer) >= args.shard_size:
            chunk = buffer[: args.shard_size]
            del buffer[: args.shard_size]
            _pack_shard(chunk, _shard_path(args.output_dir, sid))
            sid += 1
        return sid

    if args.workers <= 1:
        # Single-process path (legacy, simple)
        for robot_i, entry in enumerate(trainable):
            full_path = args.raw_dir / entry["path"]
            is_mjcf = entry.get("format") == "mjcf"
            try:
                robot = load_robot_mjcf(full_path) if is_mjcf else load_robot(full_path)
            except Exception as e:
                print(f"  [{robot_i+1}/{len(trainable)}] LOAD FAILED  "
                      f"{entry['source']}/{Path(entry['path']).name}: "
                      f"{type(e).__name__}: {str(e)[:80]}")
                n_failed += 1
                continue
            rng = np.random.default_rng(args.seed + robot_i * 100003)
            mesh_cache: dict = {}
            n_ok_this_robot = 0
            for k in range(args.n_configs):
                res = _sample_one(robot, args.n_points, rng, aug_cfg, mesh_cache,
                                  is_mjcf=is_mjcf)
                if res is None:
                    continue
                points, plabels, axes_w, origins_w, jtypes, topo, jlimits = res
                _p = Path(entry['path'])
                examples_buffer.append({
                    "name": f"{entry['source']}/{_p.parent.name}/{_p.stem}",
                    "vendor": entry.get("vendor", ""),
                    "points": points.astype(np.float32),
                    "plabels": plabels.astype(np.int32),
                    "joint_axes": axes_w.astype(np.float32),
                    "joint_origins": origins_w.astype(np.float32),
                    "joint_types": jtypes.astype(np.int32),
                    "topology": topo.astype(np.int32),
                    "joint_limits": jlimits.astype(np.float32),
                })
                n_ok_this_robot += 1
                n_total += 1
                shard_id = _flush(examples_buffer, shard_id)
            elapsed = time.time() - t_start
            rate = n_total / max(1e-3, elapsed)
            print(f"  [{robot_i+1}/{len(trainable)}] "
                  f"{entry['source']}/{Path(entry['path']).stem}: "
                  f"{n_ok_this_robot}/{args.n_configs} OK  "
                  f"(total={n_total}, {rate:.1f} ex/s)")
    else:
        # Parallel path: each worker handles ONE robot, returns its examples.
        # Main process aggregates and writes shards as they arrive.
        print(f"Using {args.workers} parallel workers")
        with ProcessPoolExecutor(max_workers=args.workers) as exe:
            futures = {}
            for robot_i, entry in enumerate(trainable):
                fut = exe.submit(
                    _process_one_robot,
                    str(args.raw_dir),
                    entry,
                    args.n_configs,
                    args.n_points,
                    not args.no_augment,
                    args.seed,
                    robot_i,
                )
                futures[fut] = (robot_i, entry)
            done_i = 0
            for fut in as_completed(futures):
                robot_i, entry = futures[fut]
                done_i += 1
                try:
                    examples, n_robot_failed, err = fut.result()
                except Exception as e:
                    print(f"  worker exception on "
                          f"{entry['source']}/{Path(entry['path']).stem}: {e}")
                    n_failed += 1
                    continue
                if not examples:
                    n_failed += 1
                    print(f"  [{done_i}/{len(trainable)}] FAIL "
                          f"{entry['source']}/{Path(entry['path']).stem}: {err}")
                    continue
                examples_buffer.extend(examples)
                n_total += len(examples)
                shard_id = _flush(examples_buffer, shard_id)
                if done_i % 10 == 0 or done_i == len(trainable):
                    elapsed = time.time() - t_start
                    rate = n_total / max(1e-3, elapsed)
                    print(f"  [{done_i}/{len(trainable)}] "
                          f"{entry['source']}/{Path(entry['path']).stem}: "
                          f"{len(examples)}/{args.n_configs} OK  "
                          f"(total={n_total}, {rate:.1f} ex/s)")

    # Flush remaining
    if examples_buffer:
        _pack_shard(examples_buffer, _shard_path(args.output_dir, shard_id))
        shard_id += 1

    elapsed = time.time() - t_start
    print(f"\nDone in {elapsed:.0f}s")
    print(f"  shards written: {shard_id}")
    print(f"  examples total: {n_total}")
    print(f"  robots failed:  {n_failed}")


if __name__ == "__main__":
    main()
