"""Distill per-joint limit priors from the 371 canonical training URDFs.

We don't have a model head that predicts joint limits, so the model's
"learned knowledge" about reasonable limits is exactly the distribution
of limits across the training set. This script extracts that
distribution once and serializes it to `data/joint_limit_priors.json`,
which the URDF assembler consumes as a default when no Tier-1 user
override is provided.

Bucketing strategy:

  Two lookup keys, tried in order at inference time:

    (joint_type, scale_class, chain_position_norm)
        e.g. ("revolute", "tabletop", 0.5)  → "the elbow of a tabletop arm"
        Computed as: floor(chain_index * 5 / dof) → bucket 0..4
        (5 buckets: base, shoulder, elbow, wrist, tip-region)

    (joint_type, scale_class)
        Coarser fallback when the first key has no samples.

  For each bucket we keep:
    p25 / p50 / p75  of the joint's (upper - lower) span, in radians
                      for revolute/continuous, metres for prismatic
    p25_lower / p50_lower / p75_lower : signed lower-bound percentiles
    p25_upper / p50_upper / p75_upper : signed upper-bound percentiles
    n_samples         : how many joints contributed

The assembler picks p50_lower / p50_upper for a "typical" prior, then a
collision sweep refines down to whatever's geometrically valid.

Why bucket by scale_class? A 6-DOF gripper-finger joint has very
different limits from a 6-DOF arm joint at the same chain position.
scale_class separates those.

Usage:
    python scripts/build_joint_limit_priors.py
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


def extract_urdf_joints(urdf_path: Path) -> list[dict] | None:
    """Return [{type, lower, upper, axis}, ...] for actuated joints in a URDF."""
    try:
        from yourdfpy import URDF
        urdf = URDF.load(str(urdf_path), build_scene_graph=False, load_meshes=False)
    except Exception:
        return None
    out: list[dict] = []
    for jn in urdf.actuated_joint_names:
        j = urdf.joint_map[jn]
        jt = j.type
        lo: float | None = None
        hi: float | None = None
        if jt == "continuous":
            lo, hi = -math.pi, math.pi   # canonical "no limit" — use ±π as the prior support
        elif j.limit is not None and j.limit.lower is not None and j.limit.upper is not None:
            lo, hi = float(j.limit.lower), float(j.limit.upper)
        else:
            continue
        # Reject sentinel "infinite" limits (some URDFs encode "no limit" as ±1e6)
        span = hi - lo
        if jt in ("revolute", "continuous"):
            if span <= 0 or span > 4 * math.pi:    # >720° → assume sentinel
                continue
        elif jt == "prismatic":
            if span <= 0 or span > 10.0:
                continue
        else:
            continue
        axis = list(j.axis) if j.axis is not None else [0.0, 0.0, 1.0]
        out.append({"type": jt, "lower": lo, "upper": hi, "axis": axis})
    return out


def extract_mjcf_joints(mjcf_path: Path) -> list[dict] | None:
    try:
        import mujoco
    except Exception:
        return None
    try:
        model = mujoco.MjModel.from_xml_path(str(mjcf_path))
    except Exception:
        return None
    out: list[dict] = []
    # mj joint type codes: 0=free, 1=ball, 2=slide (prismatic), 3=hinge (revolute)
    type_map = {2: "prismatic", 3: "revolute"}
    for ji in range(model.njnt):
        jt_code = int(model.jnt_type[ji])
        if jt_code not in type_map:
            continue
        jt = type_map[jt_code]
        if not bool(model.jnt_limited[ji]):
            if jt == "revolute":
                # Treat unlimited revolute as continuous → ±π prior support
                out.append({
                    "type": "continuous",
                    "lower": -math.pi, "upper": math.pi,
                    "axis": list(model.jnt_axis[ji]),
                })
            continue
        lo = float(model.jnt_range[ji, 0])
        hi = float(model.jnt_range[ji, 1])
        span = hi - lo
        if jt == "revolute" and (span <= 0 or span > 4 * math.pi):
            continue
        if jt == "prismatic" and (span <= 0 or span > 10.0):
            continue
        out.append({
            "type": jt, "lower": lo, "upper": hi,
            "axis": list(model.jnt_axis[ji]),
        })
    return out


def harvest_canonical_set(manifest_path: Path, raw_dir: Path) -> list[dict]:
    """Return one record per canonical robot:
        {dof, scale_class, joints: [{type, lower, upper, chain_idx, chain_pos_norm}, ...]}
    """
    manifest = json.loads(manifest_path.read_text())
    canonical = [e for e in manifest if e.get("canonical_train_set")]
    print(f"Harvesting per-joint limits from {len(canonical)} canonical entries ...")

    records: list[dict] = []
    fail = 0
    for i, e in enumerate(canonical):
        if i % 50 == 0 and i > 0:
            print(f"  {i}/{len(canonical)}  (fail so far: {fail})")
        path = (raw_dir / e["path"]).resolve()
        if e["format"] == "urdf":
            joints = extract_urdf_joints(path)
        elif e["format"] == "mjcf":
            joints = extract_mjcf_joints(path)
        else:
            joints = None
        if joints is None or len(joints) == 0:
            fail += 1
            continue
        dof = len(joints)
        for chain_idx, j in enumerate(joints):
            j["chain_idx"] = chain_idx
            j["chain_pos_norm"] = chain_idx / max(dof - 1, 1)
            j["chain_bucket"] = min(int(j["chain_pos_norm"] * 5), 4)
        records.append({
            "path": e["path"],
            "family": e.get("family"),
            "dof": dof,
            "scale_class": e.get("scale_class", "unknown"),
            "joints": joints,
        })
    print(f"  {len(records)} robots harvested, {fail} failed")
    return records


def compute_priors(records: list[dict]) -> dict:
    """Compute percentile statistics per (joint_type, scale_class, chain_bucket).

    Also computes a (joint_type, scale_class)-only fallback bucket,
    and a (joint_type)-only universal fallback.
    """
    fine_bucket: dict[tuple[str, str, int], list[tuple[float, float]]] = defaultdict(list)
    coarse_bucket: dict[tuple[str, str], list[tuple[float, float]]] = defaultdict(list)
    global_bucket: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for rec in records:
        sc = rec["scale_class"]
        for j in rec["joints"]:
            jt = j["type"]
            cb = j["chain_bucket"]
            sample = (j["lower"], j["upper"])
            fine_bucket[(jt, sc, cb)].append(sample)
            coarse_bucket[(jt, sc)].append(sample)
            global_bucket[jt].append(sample)

    def percentile_pack(samples: list[tuple[float, float]]) -> dict:
        if not samples:
            return {"n": 0}
        lows = np.array([s[0] for s in samples], dtype=np.float64)
        highs = np.array([s[1] for s in samples], dtype=np.float64)
        spans = highs - lows
        return {
            "n": len(samples),
            "lower_p25": float(np.percentile(lows, 25)),
            "lower_p50": float(np.percentile(lows, 50)),
            "lower_p75": float(np.percentile(lows, 75)),
            "upper_p25": float(np.percentile(highs, 25)),
            "upper_p50": float(np.percentile(highs, 50)),
            "upper_p75": float(np.percentile(highs, 75)),
            "span_p25":  float(np.percentile(spans, 25)),
            "span_p50":  float(np.percentile(spans, 50)),
            "span_p75":  float(np.percentile(spans, 75)),
        }

    out = {
        "fine":   {f"{jt}|{sc}|{cb}": percentile_pack(samples)
                   for (jt, sc, cb), samples in fine_bucket.items()},
        "coarse": {f"{jt}|{sc}": percentile_pack(samples)
                   for (jt, sc), samples in coarse_bucket.items()},
        "global": {jt: percentile_pack(samples)
                   for jt, samples in global_bucket.items()},
        "schema": {
            "lookup_order": [
                "fine: (joint_type, scale_class, chain_bucket=floor(chain_idx*5/dof)) — most specific",
                "coarse: (joint_type, scale_class)",
                "global: (joint_type)",
            ],
            "field_meaning": {
                "lower_pXX": "XX-th percentile of joint's <limit lower> across the bucket",
                "upper_pXX": "XX-th percentile of joint's <limit upper>",
                "span_pXX":  "XX-th percentile of (upper - lower)",
            },
            "units": {
                "revolute":   "radians",
                "continuous": "radians (samples treated as ±π)",
                "prismatic":  "metres",
            },
        },
    }
    return out


def report(priors: dict) -> None:
    print()
    print("=== Joint-limit prior summary ===")
    print()
    print("Coarse bucket (joint_type | scale_class) — n, p50_lower, p50_upper, p50_span:")
    for k, v in sorted(priors["coarse"].items()):
        if v["n"] == 0:
            continue
        print(f"  {k:32s}  n={v['n']:4d}  "
              f"[{v['lower_p50']:7.3f}, {v['upper_p50']:7.3f}]  span={v['span_p50']:6.3f}")
    print()
    print("Global bucket (joint_type only) — universal fallback:")
    for jt, v in sorted(priors["global"].items()):
        if v["n"] == 0:
            continue
        print(f"  {jt:12s}  n={v['n']:5d}  "
              f"[{v['lower_p50']:7.3f}, {v['upper_p50']:7.3f}]  span={v['span_p50']:6.3f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path,
                        default=Path("data/robot_manifest_research.json"))
    parser.add_argument("--raw-dir", type=Path,
                        default=Path("data/raw_robots"))
    parser.add_argument("--output", type=Path,
                        default=Path("data/joint_limit_priors.json"))
    args = parser.parse_args()

    records = harvest_canonical_set(args.manifest, args.raw_dir)
    priors = compute_priors(records)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(priors, indent=2))
    print(f"\nWrote {args.output}")
    report(priors)


if __name__ == "__main__":
    main()
