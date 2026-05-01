"""Walk data/raw_robots/, find every URDF/MJCF/SDF, try to load each, and
emit a manifest at data/robot_manifest.json.

Each manifest entry records:
  - source dataset (folder name under raw_robots/)
  - relative path to the URDF/MJCF/SDF
  - format (urdf | mjcf | sdf)
  - load status: "ok" | "load_failed: <reason>" | "skipped: <reason>"
  - basic stats when load succeeded:
        dof, joint_types, link_count, total_mass, link_mesh_count,
        approx_z_extent (if FK + AABB available)
  - vendor / family inferred from path

This is a discovery pass — we just want to know what robots we have. Phase B
(synthetic data generator) will do the heavy lifting of articulation + mesh
extraction.

Usage:
    python scripts/build_robot_manifest.py
        [--raw-dir data/raw_robots]
        [--output data/robot_manifest.json]
        [--load-meshes]   # try loading visual meshes too (slow)
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class ManifestEntry:
    source: str           # which dataset (e.g. "mujoco_menagerie")
    path: str             # relative path under raw_robots
    format: str           # urdf | mjcf | sdf
    family: str = ""      # inferred from path (e.g. "xarm", "ur", "panda")
    vendor: str = ""      # inferred (e.g. "ufactory", "universal_robots")
    status: str = ""      # "ok" or "load_failed: ..." or "skipped: ..."
    dof: int = 0
    joint_types: list[str] = field(default_factory=list)
    link_count: int = 0
    actuated_link_count: int = 0
    total_mass: float = 0.0
    link_mesh_count: int = 0          # mesh references found in URDF/MJCF
    meshes_resolved: int = 0          # how many mesh paths actually exist on disk
    meshes_unresolved: int = 0
    notes: str = ""

    @property
    def trainable(self) -> bool:
        """An entry is suitable for training if it loaded, has at least one
        actuated joint, and at least 80% of its referenced meshes resolve."""
        if self.status != "ok":
            return False
        if self.dof < 1:
            return False
        total_mesh = self.meshes_resolved + self.meshes_unresolved
        if total_mesh == 0:
            return False
        return self.meshes_resolved / total_mesh >= 0.8


# ---------------------------------------------------------------------------
# Vendor / family inference from path
# ---------------------------------------------------------------------------

# Light heuristic: match common substrings against vendor names. Can be
# refined incrementally as we see new datasets.
VENDOR_PATTERNS = [
    ("xarm", "ufactory"),
    ("panda", "franka"),
    ("franka", "franka"),
    ("ur3", "universal_robots"),
    ("ur5", "universal_robots"),
    ("ur10", "universal_robots"),
    ("ur16", "universal_robots"),
    ("iiwa", "kuka"),
    ("kuka", "kuka"),
    ("kinova", "kinova"),
    ("gen3", "kinova"),
    ("jaco", "kinova"),
    ("sawyer", "rethink"),
    ("baxter", "rethink"),
    ("fetch", "fetch_robotics"),
    ("pr2", "willow_garage"),
    ("yumi", "abb"),
    ("abb", "abb"),
    ("anymal", "anybotics"),
    ("spot", "boston_dynamics"),
    ("atlas", "boston_dynamics"),
    ("h1", "unitree"),
    ("g1", "unitree"),
    ("a1", "unitree"),
    ("go1", "unitree"),
    ("go2", "unitree"),
    ("aliengo", "unitree"),
    ("z1", "unitree"),
    ("h12", "unitree"),
    ("digit", "agility"),
    ("cassie", "agility"),
    ("apollo", "apptronik"),
    ("optimus", "tesla"),
    ("aloha", "google"),
    ("allegro", "wonik"),
    ("shadow", "shadow_robot"),
    ("robotiq", "robotiq"),
    ("barret", "barrett"),
    ("barrett", "barrett"),
    ("bd1", "boston_dynamics"),
    ("nao", "softbank"),
    ("pepper", "softbank"),
    ("jackal", "clearpath"),
    ("husky", "clearpath"),
    ("turtlebot", "open_robotics"),
    ("ridgeback", "clearpath"),
    ("quadrotor", "<generic>"),
    ("piper", "agilex"),
    ("cobotta", "denso"),
    ("scara", "<generic>"),
    ("doris", "<generic>"),
    ("nimbro", "<research>"),
    ("kuavo", "<research>"),
    ("apollo", "apptronik"),
    ("digit", "agility"),
    ("tiago", "pal_robotics"),
    ("talos", "pal_robotics"),
    ("hsr", "toyota"),
    ("tinymal", "<research>"),
    ("solo", "<research>"),
    ("op3", "robotis"),
    ("dynaarm", "<research>"),
    ("cyberdog", "xiaomi"),
    ("kawasaki", "kawasaki"),
    ("rs010", "kawasaki"),
    ("yaskawa", "yaskawa"),
    ("motoman", "yaskawa"),
    ("doosan", "doosan"),
    ("doris", "<generic>"),
    ("flexiv", "flexiv"),
    ("dobot", "dobot"),
    ("crx", "fanuc"),
    ("fanuc", "fanuc"),
    ("comau", "comau"),
    ("staubli", "staubli"),
    ("tx2", "staubli"),
    ("rb", "rainbow_robotics"),
    ("indy", "neuromeka"),
    ("h1_2", "unitree"),
    # NASA — Robonaut 2 and Valkyrie. r2_description / val_description
    # appear as standalone folders under robot-assets and as re-bundles
    # under urdf_files_dataset/urdf_files/random/. Use the longer substr
    # to avoid false positives on "r2" appearing inside other names.
    ("r2_description", "nasa"),
    ("val_description", "nasa"),
    ("valkyrie", "nasa"),
    # Quadrupeds + legged research platforms
    ("laikago", "unitree"),
    ("barkour", "google"),
    ("mini_cheetah", "mit"),
    ("solo", "open_dynamic_robot"),
    ("tinymal", "<research>"),
    # Manufacturing arm vendors
    ("schunk", "schunk"),
    ("lite6", "ufactory"),
    ("uf850", "ufactory"),
    ("xarm5", "ufactory"),
    ("xarm7", "ufactory"),
    # Mobile manipulators / household
    ("tidybot", "stanford"),
    ("stretch", "hello_robot"),
    # Trossen / wonik / berkeley etc.
    ("vx300", "trossen"),
    ("wx250", "trossen"),
    ("wxai", "trossen"),
    ("widowx", "trossen"),
    ("trs_so_arm", "the_robot_studio"),
    ("so101", "the_robot_studio"),
    ("berkeley_humanoid", "uc_berkeley"),
    # Hand / dexterous
    ("aero_hand", "tetheria"),
    ("softfoot", "iit"),
    ("dexee", "shadow_robot"),
    ("leap", "<research>"),
    # New humanoids
    ("adam_lite", "pndbotics"),
    ("booster_t1", "booster_robotics"),
    ("fourier_n1", "fourier_intelligence"),
    ("toddlerbot", "<research>"),
    # Misc / research / niche
    ("onrobot", "onrobot"),
    ("flybody", "<research>"),
    ("yam", "i2rt"),
    ("arx", "agile_x"),
    ("racecar", "<generic>"),
    ("z1_arm", "unitree"),
    ("k1", "<research>"),
    ("apptronik", "apptronik"),
    # Final batch — quadrupeds, hands, gym demos, niche research
    ("minitaur", "ghost_robotics"),
    ("vision60", "ghost_robotics"),
    ("microtaur", "<research>"),
    ("adroit", "<research>"),
    ("dynamixel", "robotis"),
    ("google_robot", "google"),
    ("robot_soccer_kit", "<research>"),
    ("low_cost_robot_arm", "<research>"),
]


def _infer_vendor_family(rel_path: str) -> tuple[str, str]:
    p = rel_path.lower()
    family = ""
    vendor = ""
    for pat, ven in VENDOR_PATTERNS:
        if pat in p:
            family = pat
            vendor = ven
            break
    if not family:
        # fall back to first folder-name token
        parts = re.split(r"[/\\]", p)
        for tok in parts:
            if tok and tok not in ("urdfs", "robots", "mjcf",
                                   "model", "models", "asset",
                                   "assets", "robot_assets"):
                family = tok
                break
    return vendor, family


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------

# Single case-insensitive glob per format. On Windows the FS is case-
# insensitive, so listing both "*.urdf" and "*.URDF" double-counts every
# URDF. Glob is itself case-sensitive in pattern matching though, so we
# lowercase + dedup the discovered paths instead.
URDF_PATTERNS = ("**/*.urdf",)
MJCF_PATTERNS = ("**/*.xml",)
SDF_PATTERNS = ("**/*.sdf",)


def _is_likely_robot_mjcf(path: Path) -> bool:
    """An XML in a robot dataset is usually MuJoCo. We do a cheap text check
    to filter out random XMLs like config/manifest files."""
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            head = f.read(4096).lower()
    except OSError:
        return False
    return ("<mujoco" in head) or ("<worldbody" in head) or ("<body" in head)


def discover_files(raw_dir: Path) -> list[ManifestEntry]:
    entries: list[ManifestEntry] = []
    seen_rel: set[str] = set()
    for source_dir in sorted(raw_dir.iterdir()):
        if not source_dir.is_dir():
            continue
        source = source_dir.name

        def _emit(p: Path, fmt: str) -> None:
            rel = p.relative_to(raw_dir).as_posix()
            # Case-fold to dedup on case-insensitive filesystems
            key = rel.lower()
            if key in seen_rel:
                return
            seen_rel.add(key)
            vendor, family = _infer_vendor_family(rel)
            entries.append(ManifestEntry(
                source=source, path=rel, format=fmt,
                family=family, vendor=vendor,
            ))

        for pattern in URDF_PATTERNS:
            for p in source_dir.glob(pattern):
                _emit(p, "urdf")
        for pattern in SDF_PATTERNS:
            for p in source_dir.glob(pattern):
                _emit(p, "sdf")
        for pattern in MJCF_PATTERNS:
            for p in source_dir.glob(pattern):
                if not _is_likely_robot_mjcf(p):
                    continue
                _emit(p, "mjcf")
    return entries


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def _resolve_mesh_path(filename: str, urdf_dir: Path, raw_dir: Path) -> Path | None:
    """Resolve a mesh filename string (which may be plain relative, file://,
    or package://) against on-disk paths. Returns the resolved Path if found.

    package:// URIs are matched by walking up to raw_dir and trying each
    parent's child directory matching the package name.
    """
    if filename.startswith("file://"):
        filename = filename[len("file://"):]
    if filename.startswith("package://"):
        rest = filename[len("package://"):]
        parts = rest.split("/", 1)
        if len(parts) < 2:
            return None
        pkg, sub = parts
        # Walk up from urdf_dir to raw_dir looking for a folder named `pkg`
        cur = urdf_dir
        while True:
            candidate = cur / pkg / sub
            if candidate.exists():
                return candidate
            if cur == raw_dir or cur.parent == cur:
                break
            cur = cur.parent
        # Fallback: search any folder named `pkg` under urdf_dir
        for hit in urdf_dir.rglob(pkg):
            cand = hit / sub
            if cand.exists():
                return cand
        return None
    # Plain relative or absolute
    p = Path(filename)
    if p.is_absolute() and p.exists():
        return p
    rel = (urdf_dir / p).resolve()
    if rel.exists():
        return rel
    return None


def _load_urdf(raw_dir: Path, entry: ManifestEntry) -> None:
    """Try loading with yourdfpy. Populate stats on success."""
    try:
        from yourdfpy import URDF
    except ImportError:
        entry.status = "skipped: yourdfpy unavailable"
        return
    full = (raw_dir / entry.path).resolve()
    try:
        urdf = URDF.load(str(full), build_scene_graph=False, load_meshes=False)
    except Exception as e:
        entry.status = f"load_failed: {type(e).__name__}: {e}"
        return
    try:
        entry.dof = len(urdf.actuated_joint_names)
        entry.joint_types = [urdf.joint_map[n].type
                             for n in urdf.actuated_joint_names]
        entry.link_count = len(urdf.link_map)
        actuated_links = {urdf.joint_map[n].child
                          for n in urdf.actuated_joint_names}
        entry.actuated_link_count = len(actuated_links)
        total_mass = 0.0
        mesh_count = 0
        resolved = 0
        unresolved = 0
        urdf_dir = full.parent
        for lname, link in urdf.link_map.items():
            if link.inertial is not None and link.inertial.mass is not None:
                total_mass += float(link.inertial.mass)
            for v in (link.visuals or []):
                if v.geometry and v.geometry.mesh is not None:
                    mesh_count += 1
                    fn = v.geometry.mesh.filename or ""
                    if fn and _resolve_mesh_path(fn, urdf_dir, raw_dir.resolve()) is not None:
                        resolved += 1
                    else:
                        unresolved += 1
        entry.total_mass = total_mass
        entry.link_mesh_count = mesh_count
        entry.meshes_resolved = resolved
        entry.meshes_unresolved = unresolved
        entry.status = "ok"
    except Exception as e:
        entry.status = f"parse_failed: {type(e).__name__}: {e}"


def _load_mjcf(raw_dir: Path, entry: ManifestEntry) -> None:
    """Try loading with mujoco. Populate stats on success."""
    try:
        import mujoco
    except ImportError:
        entry.status = "skipped: mujoco unavailable"
        return
    full = raw_dir / entry.path
    try:
        model = mujoco.MjModel.from_xml_path(str(full))
    except Exception as e:
        entry.status = f"load_failed: {type(e).__name__}: {str(e)[:200]}"
        return
    try:
        # Count actuated joints (hinge or slide; not free or ball usually).
        # For now, approximate dof = number of hinge + slide joints attached
        # to a body other than worldbody.
        n_jnt = model.njnt
        n_actuated = 0
        joint_types: list[str] = []
        for ji in range(n_jnt):
            jt = model.jnt_type[ji]
            if jt == 0:  # FREE
                continue
            if jt == 1:  # BALL — non-trivial DOF, skip for arm-style
                continue
            if jt == 2:  # SLIDE
                joint_types.append("prismatic")
            elif jt == 3:  # HINGE
                joint_types.append("revolute")
            n_actuated += 1
        entry.dof = n_actuated
        entry.joint_types = joint_types
        entry.link_count = model.nbody  # includes worldbody
        entry.actuated_link_count = sum(1 for ji in range(n_jnt)
                                        if model.jnt_type[ji] in (2, 3))
        # Mass from body inertia
        try:
            entry.total_mass = float(model.body_mass.sum())
        except Exception:
            entry.total_mass = 0.0
        # Geom count as a proxy for visual meshes. Since MJCF loaded
        # successfully, mujoco resolved all referenced meshes — count them
        # all as resolved.
        n_mesh_geoms = int((model.geom_type == 7).sum())  # type 7 = mesh
        entry.link_mesh_count = n_mesh_geoms
        entry.meshes_resolved = n_mesh_geoms
        entry.meshes_unresolved = 0
        entry.status = "ok"
    except Exception as e:
        entry.status = f"parse_failed: {type(e).__name__}: {e}"


def load_entry(raw_dir: Path, entry: ManifestEntry) -> None:
    if entry.format == "urdf":
        _load_urdf(raw_dir, entry)
    elif entry.format == "mjcf":
        _load_mjcf(raw_dir, entry)
    elif entry.format == "sdf":
        # SDF support is harder (yourdfpy doesn't load it). Mark as skipped
        # for now; we'd need pysdf or libsdformat to parse properly.
        entry.status = "skipped: sdf parsing not implemented"
    else:
        entry.status = f"skipped: unknown format {entry.format}"


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def build_manifest(raw_dir: Path, output: Path) -> list[dict]:
    print(f"Discovering files under {raw_dir} ...")
    entries = discover_files(raw_dir)
    print(f"  found {len(entries)} candidate files")

    print("Attempting to load each ...")
    for i, entry in enumerate(entries):
        if i % 50 == 0 and i > 0:
            print(f"  {i}/{len(entries)} ...")
        load_entry(raw_dir, entry)

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps([asdict(e) for e in entries], indent=2))

    # Summary
    by_source = Counter(e.source for e in entries)
    by_status = Counter(_status_short(e.status) for e in entries)
    by_format = Counter(e.format for e in entries)
    by_vendor = Counter(e.vendor for e in entries if e.vendor)
    by_dof = Counter(e.dof for e in entries if e.status == "ok")
    n_trainable = sum(1 for e in entries if e.trainable)
    n_articulated = sum(1 for e in entries if e.status == "ok" and e.dof >= 1)
    by_trainable_source = Counter(e.source for e in entries if e.trainable)
    by_trainable_dof = Counter(e.dof for e in entries if e.trainable)

    print("\n=== Manifest summary ===")
    print(f"Total entries: {len(entries)}")
    print(f"  loaded ok        : {by_status.get('ok', 0)}")
    print(f"  articulated (≥1 DOF): {n_articulated}")
    print(f"  trainable (≥1 DOF + ≥80% meshes resolved): {n_trainable}")
    print(f"\nBy format:")
    for fmt, n in by_format.most_common():
        print(f"  {fmt:6s}  {n}")
    print(f"\nBy source (all):")
    for src, n in by_source.most_common():
        tr = by_trainable_source.get(src, 0)
        print(f"  {src:30s}  {n:5d} total / {tr:5d} trainable")
    print(f"\nBy vendor (top 25):")
    for ven, n in by_vendor.most_common(25):
        print(f"  {ven:30s}  {n}")
    print(f"\nBy DOF (trainable only):")
    for dof, n in sorted(by_trainable_dof.items()):
        print(f"  dof={dof:3d}  {n}")

    print(f"\nWrote {output}")
    return [asdict(e) for e in entries]


def _status_short(status: str) -> str:
    if status.startswith("load_failed:"):
        return "load_failed"
    if status.startswith("parse_failed:"):
        return "parse_failed"
    if status.startswith("skipped:"):
        return "skipped"
    return status


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path,
                        default=Path("data/raw_robots"))
    parser.add_argument("--output", type=Path,
                        default=Path("data/robot_manifest.json"))
    args = parser.parse_args()
    build_manifest(args.raw_dir, args.output)


if __name__ == "__main__":
    main()
