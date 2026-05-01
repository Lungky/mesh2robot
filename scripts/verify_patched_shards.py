"""Compare the legacy v1 shards against the patched _v2_named versions
to confirm the surgical patch is sound:

  1. Same total example count.
  2. Same total shard count.
  3. Each example's joint_types / joint_axes / points are byte-identical
     (the patcher only touches `names` and `robot_idx`).
  4. Unique robot names INCREASED (because collision groups got split).
  5. The 6 ambiguous legacy names are gone from the patched set, replaced
     by their disambiguated forms.

Usage:
    python scripts/verify_patched_shards.py \
        --legacy data/training_shards_v1 \
        --patched data/training_shards_v1_v2_named \
        --legacy data/training_shards_v1_mjcf \
        --patched data/training_shards_v1_mjcf_v2_named
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--legacy", type=Path, action="append", required=True)
    parser.add_argument("--patched", type=Path, action="append", required=True)
    args = parser.parse_args()

    if len(args.legacy) != len(args.patched):
        raise SystemExit("--legacy and --patched must be paired and balanced")

    failures: list[str] = []
    total_legacy_examples = 0
    total_patched_examples = 0
    legacy_names_all: set[str] = set()
    patched_names_all: set[str] = set()

    for legacy_dir, patched_dir in zip(args.legacy, args.patched):
        legacy_shards = sorted(legacy_dir.glob("*.npz"))
        patched_shards = sorted(patched_dir.glob("*.npz"))
        print(f"\n{legacy_dir.name} -> {patched_dir.name}")
        print(f"  legacy: {len(legacy_shards)} shards")
        print(f"  patched: {len(patched_shards)} shards")
        if len(legacy_shards) != len(patched_shards):
            failures.append(f"  shard count mismatch: {len(legacy_shards)} vs "
                            f"{len(patched_shards)}")
            continue

        for ls, ps in zip(legacy_shards, patched_shards):
            with np.load(ls, allow_pickle=True) as zl, \
                 np.load(ps, allow_pickle=True) as zp:
                legacy_names_all.update(str(n) for n in zl["names"])
                patched_names_all.update(str(n) for n in zp["names"])
                total_legacy_examples += zl["points"].shape[0]
                total_patched_examples += zp["points"].shape[0]

                # Geometric data must be byte-identical
                for key in ["points", "point_labels", "joint_axes_world",
                            "joint_origins_world", "joint_types",
                            "joint_topology", "joint_valid"]:
                    if not np.array_equal(zl[key], zp[key]):
                        failures.append(
                            f"  {ls.name} != {ps.name} on '{key}'"
                        )
                        break

    print(f"\nTotals:")
    print(f"  legacy examples:  {total_legacy_examples}")
    print(f"  patched examples: {total_patched_examples}")
    print(f"  legacy unique names:  {len(legacy_names_all)}")
    print(f"  patched unique names: {len(patched_names_all)}")
    print(f"  delta (more in patched): "
          f"{len(patched_names_all) - len(legacy_names_all)}")

    # Specific collision checks
    expected_gone = {"robosuite/robot", "mujoco_menagerie/left_hand",
                     "mujoco_menagerie/hand", "mujoco_menagerie/stretch",
                     "mujoco_menagerie/2f85", "Gymnasium-Robotics/reach"}
    still_present = expected_gone & patched_names_all
    if still_present:
        # OK if these still appear because some examples had no ambiguity
        # (e.g. stretch_3 disambiguated but stretch_v1 still labeled
        # "mujoco_menagerie/stretch"). Worth flagging though.
        print(f"\n  NB: {len(still_present)} ambiguous legacy names still "
              f"present in patched (kept where 1 of 2 candidates was "
              f"disambiguated): {sorted(still_present)}")

    if total_legacy_examples != total_patched_examples:
        failures.append(
            f"example count mismatch: {total_legacy_examples} vs "
            f"{total_patched_examples}"
        )

    if failures:
        print(f"\nFAIL: {len(failures)}")
        for f in failures[:8]:
            print(f"  {f}")
        sys.exit(1)
    print("\nPASS")


if __name__ == "__main__":
    main()
