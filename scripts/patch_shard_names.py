"""Surgically rewrite robot names in v1 shards to use the new
`<source>/<parent_dir>/<stem>` convention.

The shard generator originally named robots by file stem only, so robots
with colliding stems (e.g. robosuite has 9 distinct robots all in
`<vendor>/robot.xml` files) got conflated into one name. This patcher:

  1. Walks each shard, identifies examples whose name is ambiguous
     (the same name maps to >1 manifest entry).
  2. Groups those examples by `(dof, joint_types_signature)`, which
     uniquely identifies the underlying robot among the collision set.
  3. Looks up the new name format from the manifest and rewrites the
     shard's `names` and `robot_idx` arrays to match.
  4. Saves the shard back atomically (write to .tmp, rename).

Non-affected shards are skipped untouched. Originals can be restored
from git or backup since the operation is mechanical.

Usage:
    python scripts/patch_shard_names.py \
        --shard-dir data/training_shards_v1 \
        --shard-dir data/training_shards_v1_mjcf \
        --manifest data/robot_manifest_research.json
        [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-dir", type=Path, action="append", required=True)
    parser.add_argument("--manifest", type=Path,
                        default=Path("data/robot_manifest_research.json"))
    parser.add_argument("--out-suffix", type=str, default="_v2_named",
                        help="Suffix to append to each input dir for the "
                             "output dir. Default: '_v2_named' so "
                             "training_shards_v1 -> training_shards_v1_v2_named. "
                             "Use empty string '' to overwrite in place.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    print(f"Loading manifest: {args.manifest}")
    entries = json.loads(args.manifest.read_text())

    # Build legacy-name -> [canonical_entry] map. Only canonical entries
    # matter — two non-canonical duplicates of the same robot don't need
    # disambiguating. We only care when 2+ distinct canonical robots
    # collide on the same legacy name (e.g. robosuite/robot covers 9
    # different robots).
    legacy_to_canon: dict[str, list[dict]] = defaultdict(list)
    # Also build legacy-name -> any-trainable-entry for the unambiguous
    # rename path.
    legacy_to_any: dict[str, list[dict]] = defaultdict(list)
    for e in entries:
        if e.get("status") != "ok":
            continue
        path = e.get("path", "")
        if not path:
            continue
        p = Path(path)
        legacy_name = f"{e['source']}/{p.stem}"
        legacy_to_any[legacy_name].append(e)
        if e.get("canonical_train_set"):
            legacy_to_canon[legacy_name].append(e)

    # Ambiguous = legacy name maps to >1 canonical entry (real collision)
    ambiguous = {n: es for n, es in legacy_to_canon.items() if len(es) > 1}
    print(f"Ambiguous legacy names: {len(ambiguous)}")
    for name, es in sorted(ambiguous.items())[:6]:
        print(f"  {name} -> {len(es)} entries (DOFs: "
              f"{sorted(set(e['dof'] for e in es))})")

    # Walk shards. Track each shard's source dir so we can write the
    # patched copy to a parallel output directory.
    shard_to_out: dict[Path, Path] = {}
    for d in args.shard_dir:
        out_dir = d if args.out_suffix == "" else d.parent / (d.name + args.out_suffix)
        if not args.dry_run:
            out_dir.mkdir(parents=True, exist_ok=True)
        for s in sorted(d.glob("*.npz")):
            shard_to_out[s] = out_dir / s.name
    all_shards = list(shard_to_out.keys())
    print(f"\nScanning {len(all_shards)} shards ...")
    if args.out_suffix:
        for d in args.shard_dir:
            print(f"  output dir: {d.parent / (d.name + args.out_suffix)}")

    n_patched = 0
    n_failed = 0
    n_skipped = 0
    n_examples_relabeled = 0

    for shard_path in all_shards:
        with np.load(shard_path, allow_pickle=True) as z:
            data = {k: z[k] for k in z.files}
        names = [str(n) for n in data["names"]]
        robot_idx = data["robot_idx"]
        joint_types = data["joint_types"]   # (B, J_max), -1 padding
        joint_valid = data["joint_valid"]   # (B, J_max) bool

        # Find ambiguous name slots (real canonical collisions)
        ambig_slots = [(i, names[i]) for i in range(len(names))
                       if names[i] in ambiguous]

        # Build new names list and remap robot_idx
        new_names: list[str] = []
        new_idx_map: dict[int, int] = {}  # old slot_idx -> new slot_idx
        examples_changed = 0
        slot_failed = False

        # First: for unambiguous slots, just rewrite the name to new format
        # by looking up the matching canonical entry.
        for i, name in enumerate(names):
            if name in ambiguous:
                continue  # handled below
            # Look up canonical for this legacy name
            canons = legacy_to_canon.get(name, [])
            if len(canons) == 1:
                e = canons[0]
                p = Path(e["path"])
                new_name = f"{e['source']}/{p.parent.name}/{p.stem}"
            elif len(canons) == 0:
                # Robot is non-canonical (will be filtered out of training).
                # Keep the legacy name so existing dual-form
                # load_canonical_robot_names() can still drop it.
                new_name = name
            else:
                # Shouldn't happen because we already split on `ambiguous`.
                new_name = name
            if new_name not in new_names:
                new_names.append(new_name)
            new_idx_map[i] = new_names.index(new_name)

        # Now handle ambiguous slots

        # JOINT_TYPE int -> string (must match urdf_loader.JOINT_TYPE_TO_INT)
        INT_TO_TYPE = {0: "revolute", 1: "continuous", 2: "prismatic",
                       3: "fixed", 4: "floating", 5: "planar"}

        # Per-example new robot_idx, computed below for ambig slots.
        new_robot_idx = np.empty(robot_idx.shape, dtype=np.int32)
        # First fill from unambiguous mapping.
        for i in range(len(names)):
            if i in new_idx_map:
                example_mask = (robot_idx == i)
                new_robot_idx[example_mask] = new_idx_map[i]

        for slot_idx, legacy_name in ambig_slots:
            example_mask = (robot_idx == slot_idx)
            example_ids = np.where(example_mask)[0]
            # Group examples sharing this slot by their actual joint
            # type signature — that's what tells us which underlying
            # canonical robot each example came from.
            groups: dict[tuple, list[int]] = defaultdict(list)
            for ei in example_ids:
                valid = joint_valid[ei]
                actual_jt = joint_types[ei][valid]
                actuated = tuple(
                    INT_TO_TYPE.get(int(t), "?")
                    for t in actual_jt
                    if INT_TO_TYPE.get(int(t)) in
                    ("revolute", "continuous", "prismatic")
                )
                key = (len(actuated), actuated)
                groups[key].append(int(ei))

            candidates = ambiguous[legacy_name]
            for key, ex_ids in groups.items():
                dof, jt_tuple = key
                matches = [
                    e for e in candidates
                    if e["dof"] == dof
                    and tuple(sorted(e.get("joint_types", []))) == tuple(sorted(jt_tuple))
                ]
                if len(matches) != 1:
                    SRC_RANK = {
                        "mujoco_menagerie": 0, "robot-assets": 1,
                        "drake_models": 2, "robosuite": 3,
                        "Gymnasium-Robotics": 4, "urdf_files_dataset": 5,
                        "bullet3": 6,
                    }
                    if matches:
                        matches.sort(key=lambda e: SRC_RANK.get(e["source"], 99))
                    if not matches:
                        # No manifest match at all — keep legacy name for
                        # this group of examples by mapping them to the
                        # original slot_idx (which we'll preserve below).
                        if legacy_name not in new_names:
                            new_names.append(legacy_name)
                        keep_idx = new_names.index(legacy_name)
                        for ei in ex_ids:
                            new_robot_idx[ei] = keep_idx
                        slot_failed = True
                        continue
                e = matches[0]
                p = Path(e["path"])
                new_name = f"{e['source']}/{p.parent.name}/{p.stem}"
                if new_name not in new_names:
                    new_names.append(new_name)
                new_idx = new_names.index(new_name)
                for ei in ex_ids:
                    new_robot_idx[ei] = new_idx
                examples_changed += len(ex_ids)

        if slot_failed:
            n_failed += 1
            # Don't skip the shard — partial fix is still better than none.

        # Did anything actually change? (Names rewritten OR ambig disambig'd)
        names_changed = (new_names != names)
        idx_changed = bool((new_robot_idx != robot_idx).any())
        if not names_changed and not idx_changed:
            # Still need to copy the file to the output dir so the dataset
            # is complete. Only when writing in-place do we skip.
            if not args.dry_run and args.out_suffix:
                import shutil
                shutil.copy2(shard_path, shard_to_out[shard_path])
            n_skipped += 1
            continue

        if not args.dry_run:
            data["names"] = np.array(new_names)
            data["robot_idx"] = new_robot_idx.astype(np.int32)
            out_path = shard_to_out[shard_path]
            if args.out_suffix:
                # Writing to parallel dir — no existing file to overwrite,
                # safe to write directly.
                np.savez_compressed(out_path.with_suffix(""), **data)
            else:
                # In-place — write to a sibling .tmp_<name> first then
                # rename atomically. NumPy auto-appends .npz, so we use
                # `.with_suffix("")` and let it add the extension.
                tmp_name = out_path.parent / f".tmp_{out_path.stem}"
                np.savez_compressed(tmp_name, **data)
                os.replace(tmp_name.with_suffix(".npz"), out_path)
        n_patched += 1
        n_examples_relabeled += examples_changed
        if n_patched <= 3:
            print(f"  patched {shard_path.name}: {examples_changed} disambiguated, "
                  f"{len(new_names) - len(names)} new slots")

    print(f"\nResult: patched {n_patched} shards, {n_examples_relabeled} examples relabeled")
    print(f"        skipped {n_skipped} (no ambiguous names)")
    print(f"        failed  {n_failed} (couldn't disambiguate at least one slot)")
    if args.dry_run:
        print("        DRY RUN — nothing written to disk")


if __name__ == "__main__":
    main()
