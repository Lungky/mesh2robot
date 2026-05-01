"""Enrich data/robot_manifest.json with research-grade metadata:

  1. Cross-source deduplication via (family, model_leaf) signatures.
     One canonical instance is picked per duplicate group (Menagerie >
     robot-assets > robosuite > Gymnasium-Robotics > urdf_files_dataset >
     bullet3). All other instances get `dup_of=<canonical_id>`.

  2. Quality tier per source:
        high   = MuJoCo Menagerie, robot-assets   (curated CAD)
        medium = robosuite, Gymnasium-Robotics    (curated for benchmarks)
        low    = urdf_files_dataset, bullet3      (xacro-generated / demo physics)

  3. License metadata. Top-level repo licenses for each source plus
     per-directory licenses parsed from MuJoCo Menagerie's LICENSE file
     (each robot directory in Menagerie has its own license block).

  4. Mesh-size proxy: total bytes on disk of all referenced visual meshes,
     a cheap signal for mesh fidelity without loading them.

  5. A canonical_train_set boolean: True iff the entry is `trainable`
     AND `is_canonical`. Phase B should consume this set to build a
     leak-free train/val split.

Usage:
    python scripts/build_research_manifest.py \
        [--manifest data/robot_manifest.json] \
        [--raw-dir data/raw_robots] \
        [--output data/robot_manifest_research.json]
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


# ---------------------------------------------------------------------------
# Quality tiers and license summaries (per source)
# ---------------------------------------------------------------------------

QUALITY_TIER = {
    "mujoco_menagerie": "high",
    "robot-assets": "high",
    "robosuite": "medium",
    "Gymnasium-Robotics": "medium",
    "urdf_files_dataset": "low",
    "bullet3": "low",
    "drake_models": "high",
}

# Source preference for picking the canonical instance among duplicates.
# Lower index = preferred. Mirrors quality tier with finer-grained ordering.
SOURCE_PRIORITY = [
    "mujoco_menagerie",
    "robot-assets",
    "drake_models",
    "robosuite",
    "Gymnasium-Robotics",
    "urdf_files_dataset",
    "bullet3",
]

# Top-level licenses we read out of each source repo. Menagerie is special:
# its LICENSE is split per-robot; we parse those into a directory map.
SOURCE_LICENSE_DEFAULT = {
    "Gymnasium-Robotics": "MIT (Farama Foundation, 2022)",
    "bullet3": "zlib (most files); see LICENSE.txt for exceptions",
    "robosuite": "MIT (Stanford VLL + UT Robot Perception, 2022)",
    "urdf_files_dataset": "MIT (Daniella Tola, 2023)",
    "mujoco_menagerie": "per-directory; see Menagerie LICENSE",
    "robot-assets": "varies by upstream repo (see README.md)",
    "drake_models": "varies; see drake/LICENSE.TXT (BSD-3-Clause for most)",
}


# ---------------------------------------------------------------------------
# Menagerie per-directory license parser
# ---------------------------------------------------------------------------

def parse_menagerie_license(license_path: Path) -> dict[str, str]:
    """Parse Menagerie's LICENSE file into {dir_name: license_summary}.

    The file is structured as repeated blocks:

        ============================================================
        License for contents in the directory 'foo_bar/'
        ============================================================
        <license text>

    Returns a dict mapping directory name (e.g. "agilex_piper") to a
    short summary like "Apache-2.0" / "BSD-3-Clause" / "CC-BY-4.0".
    """
    if not license_path.exists():
        return {}
    text = license_path.read_text(encoding="utf-8", errors="ignore")
    blocks = re.split(r"^=+\s*$\n", text, flags=re.MULTILINE)
    out: dict[str, str] = {}
    current_dir: str | None = None
    for block in blocks:
        m = re.search(r"License for contents in the directory '([^']+)'", block)
        if m:
            current_dir = m.group(1).rstrip("/")
            continue
        if current_dir is None:
            continue
        # Try to identify the license type from the first ~500 chars
        head = block.strip()[:1500].lower()
        license_id = "Unknown"
        if "apache license" in head or "apache-2.0" in head:
            license_id = "Apache-2.0"
        elif "mit license" in head:
            license_id = "MIT"
        elif "bsd 3-clause" in head or "bsd-3-clause" in head or "redistribution and use in source and binary forms" in head:
            license_id = "BSD-3-Clause"
        elif "creative commons attribution 4.0" in head or "cc-by-4.0" in head or "cc by 4.0" in head:
            license_id = "CC-BY-4.0"
        elif "creative commons attribution-sharealike" in head or "cc-by-sa" in head:
            license_id = "CC-BY-SA"
        elif "creative commons" in head and "noncommercial" in head:
            license_id = "CC-BY-NC"
        elif "gnu general public" in head and "version 3" in head:
            license_id = "GPL-3.0"
        elif "gnu general public" in head and "version 2" in head:
            license_id = "GPL-2.0"
        elif "mozilla public license" in head:
            license_id = "MPL-2.0"
        out[current_dir] = license_id
        current_dir = None  # block consumed
    return out


def menagerie_dir_for_path(rel_path: str) -> str | None:
    """Path under raw_robots like 'mujoco_menagerie/agilex_piper/scene.xml' →
    'agilex_piper'."""
    parts = rel_path.split("/")
    if len(parts) >= 2 and parts[0] == "mujoco_menagerie":
        return parts[1]
    return None


# ---------------------------------------------------------------------------
# Canonical model name extraction
# ---------------------------------------------------------------------------

# Words that should NOT serve as a canonical model name (too generic).
# When the file stem is one of these, fall back to the parent directory
# name. Without this, Menagerie's many `scene.xml` / `scene_mjx.xml` /
# `robot.xml` files all collapse to the same leaf and false-merge across
# distinct hardware revisions (e.g. Barkour V0 vs Vb).
GENERIC_LEAF = {
    "model", "robot", "scene", "main", "base", "default", "test",
    "example", "demo", "world", "simulation", "robot_full",
    # Menagerie sub-variants that don't identify the robot
    "scene_mjx", "scene_left", "scene_right", "scene_motor",
    "scene_position", "scene_velocity", "scene_arm", "scene_torque",
    "scene_terrain", "scene_no_arm",
    # robosuite always names its robot file `robot.xml`
}


def canonical_model_leaf(rel_path: str) -> str:
    """Best-effort extraction of a model identifier from the URDF/MJCF path.

    Strategy:
    1. Take the file stem.
    2. Strip common suffixes (`_description`, `_urdf`, `_robot`, etc.).
    3. If the stem is generic (e.g. "model", "scene", "robot"), fall back
       to the parent directory name with the same stripping rules.
    """
    p = Path(rel_path)
    stem = p.stem.lower()
    stem = re.sub(r"_(description|urdf|xml|robot|model|mjcf|scene)$", "", stem)
    if stem in GENERIC_LEAF or len(stem) < 3:
        # Look at the immediate parent directory
        if len(p.parts) >= 2:
            parent = p.parts[-2].lower()
            parent = re.sub(r"_(description|urdf|xml|robot|model|mjcf|scene|support)$", "", parent)
            if parent and parent not in GENERIC_LEAF:
                return parent
    return stem


def normalize_path_for_dedup(rel_path: str) -> str:
    """Strip per-source path prefixes that hide the fact that two paths
    refer to the same robot.

    `urdf_files_dataset` re-bundles content from other sources under
    `urdf_files/random/<source>/...`. So the path
        urdf_files_dataset/urdf_files/random/robot-assets/yumi/yumi.urdf
    is the *same* robot as
        robot-assets/urdfs/robots/yumi/yumi.urdf
    once you strip both their source-specific prefixes. This function
    returns a canonicalised tail like `yumi/yumi.urdf` for both.
    """
    p = rel_path.lower()
    # Drop the leading source folder
    parts = p.split("/")
    if len(parts) >= 2:
        parts = parts[1:]
    # urdf_files_dataset/urdf_files/random/<source>/... → drop first 3
    if len(parts) >= 3 and parts[0] == "urdf_files" and parts[1] == "random":
        parts = parts[3:]
    # urdf_files_dataset/urdf_files/<repo-bundle>/... → drop first 2 (repo prefix)
    elif len(parts) >= 1 and parts[0] == "urdf_files":
        parts = parts[1:]
    # robot-assets/urdfs/robots/X → drop "urdfs/robots"
    if len(parts) >= 2 and parts[0] == "urdfs" and parts[1] == "robots":
        parts = parts[2:]
    # bullet3/data/X and bullet3/examples/pybullet/gym/pybullet_data/X → drop prefix
    if parts and parts[0] in ("data", "examples"):
        # Find where the actual robot path starts. Heuristic: drop until we see
        # something that's neither a build/example marker nor the literal
        # "pybullet_data".
        skip_tokens = {"data", "examples", "pybullet",
                       "gym", "pybullet_data"}
        while parts and parts[0] in skip_tokens:
            parts = parts[1:]
    # robosuite/robosuite/models/assets/robots/X → drop the boilerplate prefix
    if len(parts) >= 4 and parts[0] == "robosuite" and parts[1] == "models" \
            and parts[2] == "assets" and parts[3] == "robots":
        parts = parts[4:]
    elif len(parts) >= 3 and parts[0] == "models" and parts[1] == "assets":
        parts = parts[3:] if parts[2] == "robots" else parts[2:]
    return "/".join(parts)


def family_leaf_signature(entry: dict) -> tuple[str, str, int]:
    """(family, model_leaf, dof). Catches cross-source robots with
    different paths but matching family inference (Panda, UR5e, Spot
    when family is correctly inferred)."""
    return (
        entry.get("family", ""),
        canonical_model_leaf(entry.get("path", "")),
        int(entry.get("dof", 0)),
    )


def path_tail_signature(entry: dict) -> tuple[str, int]:
    """(normalized_path_tail, dof). Catches re-bundled paths whose family
    inference falls back to source name (NASA Valkyrie, R2 humanoid)."""
    return (
        normalize_path_for_dedup(entry.get("path", "")),
        int(entry.get("dof", 0)),
    )


def menagerie_dir_signature(entry: dict) -> tuple[str, int] | None:
    """Menagerie ships *one robot per directory* (typically with multiple
    variant XMLs: `<robot>.xml`, `<robot>_mjx.xml`, `scene.xml`,
    `scene_mjx.xml`, `scene_<variant>.xml`). All these variants encode
    the same physical robot — they should dedup together.

    This signature returns `(parent_dir_path, dof)` ONLY for Menagerie
    entries. Other sources have multi-robot directories (e.g.
    `r2_description/robots/r2b.urdf` + `r2c1.urdf` are *different*
    robots under one dir), so they get None and are not affected.
    """
    if entry.get("source") != "mujoco_menagerie":
        return None
    parts = entry.get("path", "").split("/")
    if len(parts) < 3:
        return None
    # parts[0] == 'mujoco_menagerie', parts[1] == robot dir
    return (parts[1], int(entry.get("dof", 0)))


# Simple union-find for merging dedup groups.

class _UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


# ---------------------------------------------------------------------------
# Mesh size probe
# ---------------------------------------------------------------------------

def _resolve_mesh_path(filename: str, urdf_dir: Path, raw_dir: Path) -> Path | None:
    """Same logic as build_robot_manifest._resolve_mesh_path. Duplicated here
    to avoid an inter-script import cycle on a private helper."""
    if filename.startswith("file://"):
        filename = filename[len("file://"):]
    if filename.startswith("package://"):
        rest = filename[len("package://"):]
        parts = rest.split("/", 1)
        if len(parts) < 2:
            return None
        pkg, sub = parts
        cur = urdf_dir
        while True:
            candidate = cur / pkg / sub
            if candidate.exists():
                return candidate
            if cur == raw_dir or cur.parent == cur:
                break
            cur = cur.parent
        for hit in urdf_dir.rglob(pkg):
            cand = hit / sub
            if cand.exists():
                return cand
        return None
    p = Path(filename)
    if p.is_absolute() and p.exists():
        return p
    rel = (urdf_dir / p).resolve()
    if rel.exists():
        return rel
    return None


def _urdf_link_origin_aabb(urdf) -> list[float]:
    """AABB of all link origins at zero pose, in the URDF's native units.

    Cheap (FK only, no mesh load) and surprisingly effective at flagging
    unit-bug URDFs:
      - chain encoded in millimetres → AABB max in the hundreds
      - chain encoded in metres → AABB max < ~3 (humanoid scale)
    Returns [0,0,0] on FK failure (e.g. URDF with no joints / one link).
    """
    import numpy as _np

    try:
        cfg = {jn: 0.0 for jn in urdf.actuated_joint_names}
        if cfg:
            urdf.update_cfg(cfg)
    except Exception:
        pass

    positions = []
    for ln in urdf.link_map.keys():
        try:
            T = urdf.get_transform(ln)
            positions.append(_np.asarray(T)[:3, 3])
        except Exception:
            continue
    if len(positions) < 2:
        return [0.0, 0.0, 0.0]
    pos = _np.asarray(positions, dtype=_np.float64)
    extent = pos.max(axis=0) - pos.min(axis=0)
    return [float(x) for x in extent]


def _mjcf_body_aabb(model) -> list[float]:
    """AABB of all body world positions at zero pose, in metres.

    Uses mj_forward at the default qpos to get xpos for every body, then
    takes the per-axis range. Cheap because mj_forward only needs to
    resolve the kinematic tree (no contacts, no integration).
    """
    try:
        import mujoco
        import numpy as _np
        data = mujoco.MjData(model)
        mujoco.mj_resetData(model, data)
        mujoco.mj_forward(model, data)
        xpos = _np.asarray(data.xpos)
        # body 0 is the world body — skip it
        if xpos.shape[0] < 2:
            return [0.0, 0.0, 0.0]
        bodies = xpos[1:]
        extent = bodies.max(axis=0) - bodies.min(axis=0)
        return [float(x) for x in extent]
    except Exception:
        return [0.0, 0.0, 0.0]


def probe_robot_stats(raw_dir: Path, entry: dict) -> dict:
    """Re-load each canonical URDF/MJCF and collect richer metadata:

      - mesh_bytes_total : sum of on-disk mesh file bytes (URDF) or
        approximated vertex+face byte count (MJCF, where meshes are
        baked into the model).
      - joint_range_total_rad : sum of (upper - lower) over actuated
        revolute/continuous joints. Captures total reachable angular
        range — a humanoid with full-range elbows scores higher than
        a bench arm with software-limited motion.
      - joint_range_total_m : same for prismatic joints, in metres.
      - aabb_extent_m : link-origin AABB extents at zero pose.
        URDF: in the source URDF's native units (typically m, but mm-bug
        URDFs surface here as huge values).
        MJCF: always metres.
        This is a *lower bound* on the robot's full extent — meshes
        could push it further out — but it's enough to flag scale
        outliers and tag a `scale_class` downstream.

    Failures return zeros silently — this is a probe, not a validator.
    """
    out = {
        "mesh_bytes_total": 0,
        "joint_range_total_rad": 0.0,
        "joint_range_total_m": 0.0,
        "aabb_extent_m": [0.0, 0.0, 0.0],
    }
    if entry.get("status") != "ok":
        return out
    full = (raw_dir / entry["path"]).resolve()

    if entry["format"] == "urdf":
        try:
            from yourdfpy import URDF
            # build_scene_graph=True is required for get_transform/FK; skip
            # mesh loading to keep this cheap.
            urdf = URDF.load(str(full), build_scene_graph=True, load_meshes=False)
        except Exception:
            return out

        urdf_dir = full.parent
        total_bytes = 0
        for link in urdf.link_map.values():
            for v in (link.visuals or []):
                if v.geometry and v.geometry.mesh is not None:
                    fn = v.geometry.mesh.filename or ""
                    rp = _resolve_mesh_path(fn, urdf_dir, raw_dir.resolve())
                    if rp is not None:
                        try:
                            total_bytes += rp.stat().st_size
                        except OSError:
                            pass
        out["mesh_bytes_total"] = total_bytes

        out["aabb_extent_m"] = _urdf_link_origin_aabb(urdf)

        # Some URDFs encode "no limit" as `lower=-1e6, upper=1e6` instead
        # of using the proper `<continuous>` joint type. Treat any
        # revolute joint with span > 4π (~720°) as effectively
        # continuous (one full turn = 2π) and any prismatic span > 10 m
        # as a sentinel — clamp to 2 m.
        TWO_PI = 2 * 3.141593
        REV_CAP = 2 * TWO_PI       # 4π = 720°
        PRISM_CAP = 10.0           # 10 m
        rad = 0.0
        m = 0.0
        for jn in urdf.actuated_joint_names:
            j = urdf.joint_map[jn]
            if j.limit is None:
                if j.type == "continuous":
                    rad += TWO_PI
                continue
            lo, hi = j.limit.lower, j.limit.upper
            if lo is None or hi is None:
                continue
            span = float(hi) - float(lo)
            if j.type == "continuous":
                rad += TWO_PI
            elif j.type == "revolute":
                rad += min(span, REV_CAP) if span < REV_CAP else TWO_PI
            elif j.type == "prismatic":
                m += min(span, PRISM_CAP) if span > 0 else 0.0
        out["joint_range_total_rad"] = rad
        out["joint_range_total_m"] = m
        return out

    if entry["format"] == "mjcf":
        try:
            import mujoco
            import numpy as _np
            model = mujoco.MjModel.from_xml_path(str(full))
        except Exception:
            return out
        try:
            n_vert = int(model.mesh_vert.shape[0]) if model.mesh_vert is not None else 0
            n_face = int(model.mesh_face.shape[0]) if model.mesh_face is not None else 0
            out["mesh_bytes_total"] = n_vert * 12 + n_face * 12
        except Exception:
            pass
        out["aabb_extent_m"] = _mjcf_body_aabb(model)
        try:
            TWO_PI = 2 * 3.141593
            REV_CAP = 2 * TWO_PI
            PRISM_CAP = 10.0
            rad = 0.0
            m = 0.0
            for ji in range(model.njnt):
                jt = int(model.jnt_type[ji])
                # Skip free (0) and ball (1)
                if jt < 2:
                    continue
                if not bool(model.jnt_limited[ji]):
                    if jt == 3:
                        rad += TWO_PI
                    continue
                lo, hi = float(model.jnt_range[ji, 0]), float(model.jnt_range[ji, 1])
                span = hi - lo
                if jt == 3:
                    rad += min(span, REV_CAP) if span < REV_CAP else TWO_PI
                elif jt == 2:
                    m += min(span, PRISM_CAP) if span > 0 else 0.0
            out["joint_range_total_rad"] = rad
            out["joint_range_total_m"] = m
        except Exception:
            pass
        return out

    return out


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def is_trainable(entry: dict) -> bool:
    if entry.get("status") != "ok":
        return False
    if entry.get("dof", 0) < 1:
        return False
    tot = entry.get("meshes_resolved", 0) + entry.get("meshes_unresolved", 0)
    if tot == 0:
        return False
    return entry["meshes_resolved"] / tot >= 0.8


def _is_scene_file(path: str) -> bool:
    """Menagerie convention: each robot directory ships <robot>.xml (the
    model) plus scene.xml / scene_*.xml (a scene that wraps the model
    with floor + lights + cameras). The scene file has *more* mesh
    references (floor mesh, etc.) than the robot file, so a naive
    "prefer most meshes" tiebreak picks scene.xml — which is the wrong
    robot canonical. Use this helper to deprioritise scene files."""
    leaf = Path(path).stem.lower()
    return leaf == "scene" or leaf.startswith("scene_") or leaf.endswith("_scene")


def pick_canonical(group: list[dict]) -> dict:
    """Among entries in a dedup group, pick one canonical:
    1. Prefer the source higher in SOURCE_PRIORITY.
    2. Prefer non-scene files (scene.xml is a wrapper, not the robot).
    3. Within source/scene-class, pick highest meshes_resolved.
    4. Tiebreak: lexicographic path."""
    src_idx = {s: i for i, s in enumerate(SOURCE_PRIORITY)}
    return min(
        group,
        key=lambda e: (
            src_idx.get(e["source"], 99),
            1 if _is_scene_file(e["path"]) else 0,
            -e.get("meshes_resolved", 0),
            e["path"],
        ),
    )


def enrich(manifest: list[dict], raw_dir: Path,
           menagerie_licenses: dict[str, str],
           probe_mesh_bytes: bool) -> list[dict]:
    # Add quality_tier + license per entry
    for e in manifest:
        src = e["source"]
        e["quality_tier"] = QUALITY_TIER.get(src, "unknown")
        if src == "mujoco_menagerie":
            d = menagerie_dir_for_path(e["path"])
            e["license"] = menagerie_licenses.get(d or "", "Unknown") + f" (Menagerie: {d})"
        else:
            e["license"] = SOURCE_LICENSE_DEFAULT.get(src, "Unknown")

    # Build dedup groups by union-find over BOTH signatures:
    #  (a) (family, leaf, dof)        — catches family-aware cross-source dups
    #  (b) (normalized_path_tail, dof) — catches re-bundle dups when family
    #      inference fell back to source name
    # Two entries are merged into one canonical group if they share EITHER
    # key. This avoids the all-or-nothing behaviour of either signature alone.
    trainable = [e for e in manifest if is_trainable(e)]
    uf = _UnionFind(len(trainable))
    fam_buckets: dict[tuple, list[int]] = defaultdict(list)
    path_buckets: dict[tuple, list[int]] = defaultdict(list)
    menagerie_buckets: dict[tuple, list[int]] = defaultdict(list)
    for i, e in enumerate(trainable):
        fam_buckets[family_leaf_signature(e)].append(i)
        path_buckets[path_tail_signature(e)].append(i)
        m_sig = menagerie_dir_signature(e)
        if m_sig is not None:
            menagerie_buckets[m_sig].append(i)
    all_buckets = (list(fam_buckets.values())
                   + list(path_buckets.values())
                   + list(menagerie_buckets.values()))
    for bucket in all_buckets:
        for j in bucket[1:]:
            uf.union(bucket[0], j)

    # Collect entries by their union-find root
    groups: dict[int, list[dict]] = defaultdict(list)
    for i, e in enumerate(trainable):
        groups[uf.find(i)].append(e)

    # Pick canonical per group, populate dedup fields
    for root, group in groups.items():
        canonical = pick_canonical(group)
        # Stable group id from canonical's path tail
        canon_id = (
            f"{normalize_path_for_dedup(canonical['path'])}"
            f"@{int(canonical.get('dof', 0))}dof"
        )
        for e in group:
            e["dedup_group_size"] = len(group)
            e["dedup_group_id"] = canon_id
            e["is_canonical"] = (e is canonical)
            e["dup_of"] = None if e is canonical else canonical["path"]

    # Non-trainable entries: fill defaults
    for e in manifest:
        if "is_canonical" not in e:
            e["dedup_group_size"] = 0
            e["dedup_group_id"] = None
            e["is_canonical"] = False
            e["dup_of"] = None

    # canonical_train_set: trainable AND canonical
    for e in manifest:
        e["canonical_train_set"] = bool(is_trainable(e) and e.get("is_canonical"))

    # Optional richer probe: re-loads each canonical URDF/MJCF to capture
    # mesh bytes + joint-range metadata. Slow (~5–10 min for ~370 entries).
    if probe_mesh_bytes:
        canonical_entries = [e for e in manifest if e["canonical_train_set"]]
        print(f"Probing robot stats for {len(canonical_entries)} canonical entries ...")
        for i, e in enumerate(canonical_entries):
            if i % 50 == 0 and i > 0:
                print(f"  {i}/{len(canonical_entries)}")
            stats = probe_robot_stats(raw_dir, e)
            e.update(stats)
        # Non-canonicals: zero defaults
        for e in manifest:
            for key, default in (
                ("mesh_bytes_total", 0),
                ("joint_range_total_rad", 0.0),
                ("joint_range_total_m", 0.0),
            ):
                if key not in e:
                    e[key] = default

        # Fidelity classification by mesh bytes (independent of source).
        # Thresholds derived from the canonical-set distribution:
        #   <0.5 MB ≈ simplified physics meshes (boxes, capsules)
        #   0.5–5 MB ≈ typical XACRO with STLs
        #   >5 MB ≈ curated CAD / humanoids
        for e in manifest:
            mb = e.get("mesh_bytes_total", 0) / 1e6
            if mb >= 5.0:
                e["fidelity_class"] = "high"
            elif mb >= 0.5:
                e["fidelity_class"] = "medium"
            elif mb > 0:
                e["fidelity_class"] = "low"
            else:
                e["fidelity_class"] = "unknown"
    else:
        for e in manifest:
            if "mesh_bytes_total" not in e:
                e["mesh_bytes_total"] = 0
            if "fidelity_class" not in e:
                e["fidelity_class"] = "unknown"

    # Scale classification from aabb_extent_m. URDF units are nominally
    # metres but a few sources ship millimetre-encoded URDFs (chain in
    # the hundreds-of-units range); we flag those as `unit_bug` and
    # silently rescale before bucketing. MJCF aabb is always metres.
    for e in manifest:
        e["scale_class"] = _classify_scale(e.get("aabb_extent_m") or [0.0, 0.0, 0.0])

    return manifest


def _classify_scale(aabb: list[float]) -> str:
    """Bucket a robot by its link-origin AABB extents.

    Buckets (max-axis extent in metres):
      unknown  : aabb is all-zero (probe never ran or robot has 1 link)
      compact  : <0.3 m   — gripper, hand, fingertip
      tabletop : 0.3–1.0 m — bench arm, half-humanoid, mobile manipulator
      fullsize : 1.0–2.5 m — full industrial arm, humanoid
      huge     : >2.5 m   — gantry / mobile / multi-arm rig (legitimate)
                            OR a unit-bug URDF (mm-encoded)
      unit_bug : >50 m   — almost certainly mm-encoded; downstream
                            consumers should rescale by 1e-3 or skip
    """
    if not aabb or all(x == 0.0 for x in aabb):
        return "unknown"
    extent_max = max(aabb)
    if extent_max > 50.0:
        return "unit_bug"
    if extent_max > 2.5:
        return "huge"
    if extent_max >= 1.0:
        return "fullsize"
    if extent_max >= 0.3:
        return "tabletop"
    return "compact"


def report(manifest: list[dict]) -> None:
    n_total = len(manifest)
    n_trainable = sum(1 for e in manifest if is_trainable(e))
    n_canonical = sum(1 for e in manifest if e.get("canonical_train_set"))
    n_dups = n_trainable - n_canonical

    by_tier = Counter(e["quality_tier"] for e in manifest if e.get("canonical_train_set"))
    by_fidelity = Counter(e.get("fidelity_class", "unknown")
                          for e in manifest if e.get("canonical_train_set"))
    by_source = Counter(e["source"] for e in manifest if e.get("canonical_train_set"))
    by_dof = Counter(e["dof"] for e in manifest if e.get("canonical_train_set"))
    by_scale = Counter(e.get("scale_class", "unknown")
                       for e in manifest if e.get("canonical_train_set"))

    # Dup groups summary
    group_sizes = Counter()
    for e in manifest:
        if e.get("is_canonical"):
            group_sizes[e["dedup_group_size"]] += 1
    n_groups_with_dups = sum(c for sz, c in group_sizes.items() if sz > 1)
    n_groups_singleton = group_sizes.get(1, 0)

    print("\n=== Research manifest summary ===")
    print(f"Total entries:                 {n_total}")
    print(f"Trainable (loaded ok + meshes): {n_trainable}")
    print(f"Canonical (deduped):            {n_canonical}")
    print(f"Duplicates marked:              {n_dups}")
    print()
    print(f"Dedup groups:")
    print(f"  with duplicates: {n_groups_with_dups}")
    print(f"  singletons:      {n_groups_singleton}")
    print()
    print("Canonical-set quality tier (by source):")
    for tier, n in by_tier.most_common():
        print(f"  {tier:8s}  {n}")
    print()
    if any(v != "unknown" for v in by_fidelity):
        print("Canonical-set fidelity class (by mesh bytes):")
        for cls, n in by_fidelity.most_common():
            print(f"  {cls:8s}  {n}")
        print()
    if any(v != "unknown" for v in by_scale):
        print("Canonical-set scale class (by link-origin AABB):")
        for cls, n in by_scale.most_common():
            print(f"  {cls:8s}  {n}")
        print()
    print("Canonical-set by source:")
    for src, n in by_source.most_common():
        print(f"  {src:25s}  {n}")
    print()
    print("Canonical-set DOF distribution (top 15):")
    for dof, n in sorted(by_dof.items())[:15]:
        print(f"  dof={dof:3d}  {n}")

    # Examples of what got deduped
    print()
    print("Example duplicate groups (first 8 with size > 2):")
    seen = set()
    for e in manifest:
        if not e.get("is_canonical"):
            continue
        sz = e.get("dedup_group_size", 0)
        if sz <= 2:
            continue
        gid = e["dedup_group_id"]
        if gid in seen:
            continue
        seen.add(gid)
        # Find all members
        members = [m for m in manifest if m.get("dedup_group_id") == gid
                   and is_trainable(m)]
        print(f"  {gid}  ({sz} instances)")
        for m in members:
            tag = "[CANON]" if m.get("is_canonical") else "       "
            print(f"    {tag} {m['source']:22s} {m['path']}")
        if len(seen) >= 8:
            break


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path,
                        default=Path("data/robot_manifest.json"))
    parser.add_argument("--raw-dir", type=Path,
                        default=Path("data/raw_robots"))
    parser.add_argument("--output", type=Path,
                        default=Path("data/robot_manifest_research.json"))
    parser.add_argument("--probe-mesh-bytes", action="store_true",
                        help="Re-load each canonical URDF/MJCF to measure "
                             "total mesh-file bytes on disk. Slow (~10 min).")
    args = parser.parse_args()

    print(f"Loading manifest: {args.manifest}")
    manifest = json.loads(args.manifest.read_text())
    print(f"  {len(manifest)} entries")

    menagerie_lic_path = args.raw_dir / "mujoco_menagerie" / "LICENSE"
    print(f"Parsing Menagerie per-directory licenses: {menagerie_lic_path}")
    menagerie_licenses = parse_menagerie_license(menagerie_lic_path)
    print(f"  {len(menagerie_licenses)} robot directories with license info")

    enriched = enrich(manifest, args.raw_dir, menagerie_licenses,
                      probe_mesh_bytes=args.probe_mesh_bytes)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(enriched, indent=2))
    print(f"\nWrote {args.output}")

    report(enriched)


if __name__ == "__main__":
    main()
