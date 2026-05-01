"""Validate the research manifest's internal consistency.

Run after every `build_research_manifest.py` invocation to catch
regressions before they propagate to training.

Checks:
  1. Every dedup group has exactly one canonical entry.
  2. Every `dup_of` value points to a real path in the manifest.
  3. Every canonical entry has non-empty license metadata.
  4. Every canonical entry has a valid fidelity_class (high/medium/low/unknown).
  5. Joint-range data is present for ≥80% of canonical entries.
  6. No stem-collisions in canonical robot names (which would silently
     drop robots from the canonical filter).
  7. `is_canonical` and `dup_of=None` agree (canonical iff dup_of is None).
  8. Source priority is honoured — for every multi-source group, the
     canonical is from the highest-ranked source present.

Exits with code 1 if any check fails. Prints a summary either way.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path


SOURCE_PRIORITY = [
    "mujoco_menagerie", "robot-assets", "drake_models",
    "robosuite", "Gymnasium-Robotics", "urdf_files_dataset", "bullet3",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path,
                        default=Path("data/robot_manifest_research.json"))
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text())
    print(f"Loaded {len(manifest)} entries from {args.manifest}")

    canonical = [e for e in manifest if e.get("canonical_train_set")]
    print(f"Canonical: {len(canonical)}")

    failures: list[str] = []
    warnings: list[str] = []
    paths_in_manifest = {e["path"] for e in manifest}

    # 1. Each dedup group has exactly one canonical
    by_group: dict[str, list[dict]] = defaultdict(list)
    for e in manifest:
        gid = e.get("dedup_group_id")
        if gid:
            by_group[gid].append(e)
    bad_groups = [(gid, members) for gid, members in by_group.items()
                  if sum(1 for m in members if m.get("is_canonical")) != 1]
    if bad_groups:
        failures.append(
            f"[1] {len(bad_groups)} dedup groups don't have exactly one canonical entry"
        )
        for gid, members in bad_groups[:3]:
            n_canon = sum(1 for m in members if m.get("is_canonical"))
            failures.append(f"    {gid}: {n_canon} canonical out of {len(members)}")

    # 2. Every dup_of points to a real path
    bad_dups = [e for e in manifest
                if e.get("dup_of") and e["dup_of"] not in paths_in_manifest]
    if bad_dups:
        failures.append(f"[2] {len(bad_dups)} entries point to a dup_of path not in manifest")
        for e in bad_dups[:3]:
            failures.append(f"    {e['path']} -> {e['dup_of']}")

    # 3. License metadata
    no_license = [e for e in canonical if not e.get("license")
                  or e.get("license") == "Unknown"]
    if no_license:
        warnings.append(
            f"[3] {len(no_license)} canonical entries have empty/Unknown license"
        )
        for e in no_license[:3]:
            warnings.append(f"    {e['source']} {e['path']}")

    # 4. fidelity_class
    valid_fidelity = {"high", "medium", "low", "unknown"}
    bad_fidelity = [e for e in canonical
                    if e.get("fidelity_class") not in valid_fidelity]
    if bad_fidelity:
        failures.append(f"[4] {len(bad_fidelity)} canonical entries have invalid fidelity_class")

    # 5. Joint-range coverage
    has_range = sum(1 for e in canonical
                    if e.get("joint_range_total_rad", 0) > 0
                    or e.get("joint_range_total_m", 0) > 0)
    pct = has_range / len(canonical) if canonical else 0
    if pct < 0.80:
        warnings.append(
            f"[5] joint-range coverage low: {pct:.0%} (expected >=80%) — "
            "did you forget --probe-mesh-bytes?"
        )

    # 6. Stem-collision in canonical names (would silently drop robots)
    name_counts: Counter = Counter()
    for e in canonical:
        stem = Path(e["path"]).stem
        name = f"{e['source']}/{stem}"
        name_counts[name] += 1
    collisions = {n: c for n, c in name_counts.items() if c > 1}
    if collisions:
        warnings.append(
            f"[6] {len(collisions)} canonical-name stem collisions "
            f"(losing {sum(c-1 for c in collisions.values())} robots from canonical filter):"
        )
        for n, c in list(collisions.items())[:5]:
            warnings.append(f"    {c}x  {n}")

    # 7. is_canonical iff dup_of is None
    inconsistent = [
        e for e in manifest
        if e.get("dedup_group_id")
        and bool(e.get("is_canonical")) != (e.get("dup_of") is None)
    ]
    if inconsistent:
        failures.append(f"[7] {len(inconsistent)} entries have inconsistent is_canonical/dup_of")

    # 8. Source priority within multi-source groups
    src_idx = {s: i for i, s in enumerate(SOURCE_PRIORITY)}
    bad_priority = []
    for gid, members in by_group.items():
        sources = {m["source"] for m in members}
        if len(sources) <= 1:
            continue
        canon = next((m for m in members if m.get("is_canonical")), None)
        if canon is None:
            continue
        canon_rank = src_idx.get(canon["source"], 99)
        best_rank = min(src_idx.get(s, 99) for s in sources)
        if canon_rank != best_rank:
            bad_priority.append((gid, canon["source"], sources))
    if bad_priority:
        failures.append(
            f"[8] {len(bad_priority)} multi-source groups picked a non-best-source canonical"
        )
        for gid, src, all_srcs in bad_priority[:3]:
            failures.append(f"    {gid}: chose {src} from {all_srcs}")

    # Report
    print()
    if failures:
        print(f"FAIL: {len(failures)} hard failures:")
        for f in failures:
            print(f"  {f}")
    if warnings:
        print(f"WARN: {len(warnings)} soft warnings:")
        for w in warnings:
            print(f"  {w}")
    if not failures and not warnings:
        print("PASS: all checks clean.")
    elif not failures:
        print("PASS (with warnings).")

    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
