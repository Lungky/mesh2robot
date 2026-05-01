"""Produce a human-readable summary of the canonical robot set.

Outputs two artifacts:
  - data/canonical_robots.md     paper-ready markdown table with one row
                                 per canonical robot, sortable by family
  - data/canonical_robots.csv    flat CSV (same data) for spreadsheets

Used to:
  1. Sanity-check the manifest by eyeballing the table
  2. Embed in a paper appendix or supplementary material
  3. Pick specific robots for benchmarks ("our 6-DOF revolute set is...")

Usage:
    python scripts/summarize_manifest.py
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path,
                        default=Path("data/robot_manifest_research.json"))
    parser.add_argument("--md-out", type=Path,
                        default=Path("data/canonical_robots.md"))
    parser.add_argument("--csv-out", type=Path,
                        default=Path("data/canonical_robots.csv"))
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text())
    canonical = [e for e in manifest if e.get("canonical_train_set")]
    canonical.sort(key=lambda e: (e.get("vendor") or "_unknown",
                                  e.get("family") or "",
                                  e.get("dof", 0),
                                  e.get("path") or ""))
    print(f"Canonical robots: {len(canonical)}")

    # ---- CSV ----
    fields = [
        "vendor", "family", "source", "dof", "link_count",
        "fidelity_class", "scale_class", "mesh_bytes_total",
        "aabb_x_m", "aabb_y_m", "aabb_z_m",
        "joint_range_total_rad", "joint_range_total_m",
        "license", "dedup_group_size", "path",
    ]
    args.csv_out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.csv_out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for e in canonical:
            row = {k: e.get(k, "") for k in fields}
            aabb = e.get("aabb_extent_m") or [0.0, 0.0, 0.0]
            row["aabb_x_m"] = round(float(aabb[0]), 4)
            row["aabb_y_m"] = round(float(aabb[1]), 4)
            row["aabb_z_m"] = round(float(aabb[2]), 4)
            w.writerow(row)
    print(f"Wrote {args.csv_out}")

    # ---- Markdown ----
    by_vendor = Counter(e.get("vendor") or "<unmatched>" for e in canonical)
    by_fidelity = Counter(e.get("fidelity_class") or "unknown" for e in canonical)
    by_scale = Counter(e.get("scale_class") or "unknown" for e in canonical)
    by_source = Counter(e["source"] for e in canonical)
    dof_dist = Counter(e["dof"] for e in canonical)

    lines: list[str] = []
    lines.append("# Canonical Robot Set\n")
    lines.append("Auto-generated from `data/robot_manifest_research.json` by "
                 "`scripts/summarize_manifest.py`. One row per canonical "
                 "(deduped) robot in the research-grade training pool.\n")

    lines.append("## Headline numbers\n")
    lines.append(f"- **Canonical robots: {len(canonical)}**")
    lines.append(f"- Vendors covered: {sum(1 for v in by_vendor if v != '<unmatched>' and v != '<research>')}")
    lines.append(f"- DOF range: {min(dof_dist)}–{max(dof_dist)}")
    lines.append(f"- Fidelity: {by_fidelity.get('high', 0)} high / "
                 f"{by_fidelity.get('medium', 0)} medium / "
                 f"{by_fidelity.get('low', 0)} low")
    lines.append(f"- Scale: {by_scale.get('compact', 0)} compact / "
                 f"{by_scale.get('tabletop', 0)} tabletop / "
                 f"{by_scale.get('fullsize', 0)} fullsize / "
                 f"{by_scale.get('huge', 0)} huge / "
                 f"{by_scale.get('unit_bug', 0)} unit_bug / "
                 f"{by_scale.get('unknown', 0)} unknown\n")

    lines.append("## By vendor (top 15)\n")
    lines.append("| Vendor | N | Sample family |")
    lines.append("|---|---:|---|")
    for vendor, n in by_vendor.most_common(15):
        sample = next((e["family"] for e in canonical
                       if (e.get("vendor") or "<unmatched>") == vendor), "—")
        lines.append(f"| {vendor} | {n} | {sample} |")
    lines.append("")

    lines.append("## By source\n")
    lines.append("| Source | N |")
    lines.append("|---|---:|")
    for source, n in by_source.most_common():
        lines.append(f"| {source} | {n} |")
    lines.append("")

    lines.append("## DOF distribution\n")
    lines.append("| DOF | Count |")
    lines.append("|---:|---:|")
    for dof, n in sorted(dof_dist.items()):
        lines.append(f"| {dof} | {n} |")
    lines.append("")

    lines.append("## Scale class distribution\n")
    lines.append(
        "Scale class buckets are derived from the link-origin AABB at "
        "zero pose (a lower bound on the robot's full extent — meshes "
        "could push it further). `unit_bug` flags suspected mm-encoded "
        "URDFs (>50 m chain), `unknown` is one-link or all-coincident "
        "fixtures where FK can't separate origins.\n"
    )
    lines.append("| Scale | Max-axis range | Count |")
    lines.append("|---|---|---:|")
    scale_ranges = {
        "compact":  "<0.3 m",
        "tabletop": "0.3–1.0 m",
        "fullsize": "1.0–2.5 m",
        "huge":     ">2.5 m",
        "unit_bug": ">50 m (mm-bug)",
        "unknown":  "—",
    }
    for cls in ("compact", "tabletop", "fullsize", "huge", "unit_bug", "unknown"):
        lines.append(f"| {cls} | {scale_ranges[cls]} | {by_scale.get(cls, 0)} |")
    lines.append("")

    lines.append("## Full canonical robot list\n")
    lines.append("Sorted by vendor, family, DOF.\n")
    lines.append("| Vendor | Family | DOF | Links | Fidelity | Scale | "
                 "Mesh MB | AABB (m) | Range (rad) | Range (m) | License | Path |")
    lines.append("|---|---|---:|---:|---|---|---:|---|---:|---:|---|---|")
    for e in canonical:
        mb = e.get("mesh_bytes_total", 0) / 1e6
        rad = e.get("joint_range_total_rad", 0.0)
        m = e.get("joint_range_total_m", 0.0)
        aabb = e.get("aabb_extent_m") or [0.0, 0.0, 0.0]
        aabb_str = f"{aabb[0]:.2f}×{aabb[1]:.2f}×{aabb[2]:.2f}"
        license = (e.get("license") or "").replace("|", "\\|")[:60]
        path = e.get("path") or ""
        lines.append(
            f"| {e.get('vendor') or ''} "
            f"| {e.get('family') or ''} "
            f"| {e.get('dof', 0)} "
            f"| {e.get('link_count', 0)} "
            f"| {e.get('fidelity_class') or '?'} "
            f"| {e.get('scale_class') or '?'} "
            f"| {mb:.2f} "
            f"| {aabb_str} "
            f"| {rad:.2f} "
            f"| {m:.2f} "
            f"| {license} "
            f"| `{path}` |"
        )

    args.md_out.parent.mkdir(parents=True, exist_ok=True)
    args.md_out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {args.md_out} ({len(lines)} lines)")


if __name__ == "__main__":
    main()
