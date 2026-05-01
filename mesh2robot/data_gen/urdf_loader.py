"""URDF → labeled-mesh pipeline for synthetic data generation.

Per URDF, this module supports:
  1. `load_robot()`         load + parse, expose joints/links/limits in a
                            uniform structure usable by the rest of the pipeline.
  2. `sample_random_config()`   random joint angles within limits.
  3. `articulate_and_label()`   FK every link to world, concat all link
                                visual meshes into ONE mesh whose vertices
                                carry per-vertex link indices.
  4. `sample_point_cloud()`     uniform-area sampling of N points from the
                                combined mesh, preserving link labels.

The output of (3) and (4) is what the foundation model trains on.

We intentionally keep this focused on URDF (yourdfpy). MJCF support will
be added in a sibling module if needed, since the loader API differs.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
import trimesh


# ---------------------------------------------------------------------------
# Joint-type encoding shared with the model
# ---------------------------------------------------------------------------

JOINT_TYPE_TO_INT: dict[str, int] = {
    "revolute":   0,
    "continuous": 1,
    "prismatic":  2,
    "fixed":      3,
    "floating":   4,
    "planar":     5,
}

INT_TO_JOINT_TYPE: dict[int, str] = {v: k for k, v in JOINT_TYPE_TO_INT.items()}


# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------

@dataclass
class LoadedRobot:
    """Parsed URDF with everything Phase B needs to articulate + label."""
    name: str
    urdf_path: Path
    urdf: object   # yourdfpy.URDF — kept opaque to avoid type-import cost

    # Link info — chain order from a BFS off the root link
    link_names: list[str]
    link_idx: dict[str, int]                # name → index used in labels
    root_link: str

    # Joint info — actuated only, in URDF actuated_joint_names order
    actuated_joint_names: list[str]
    joint_types: list[str]
    joint_lower: list[float]
    joint_upper: list[float]
    joint_parent: list[int]                 # parent link index
    joint_child: list[int]                  # child link index


@dataclass
class TrainingExample:
    """One training example for the foundation model."""
    name: str                                # source URDF name
    config: np.ndarray                       # (J,) sampled joint angles
    points: np.ndarray                       # (N, 3) sampled surface points (world)
    point_labels: np.ndarray                 # (N,) per-point link index
    joint_axes_world: np.ndarray             # (J, 3) unit axes
    joint_origins_world: np.ndarray          # (J, 3) pivots
    joint_types: np.ndarray                  # (J,) integer encoding
    joint_parents: np.ndarray                # (J,) parent link index
    joint_children: np.ndarray               # (J,) child link index
    meta: dict = field(default_factory=dict) # source, family, vendor, etc.


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_robot(urdf_path: Path | str) -> LoadedRobot:
    """Parse a URDF with yourdfpy, expose what the data generator needs.

    Builds a stable link ordering via BFS from the root link, so downstream
    label indices are deterministic for the same URDF.
    """
    from yourdfpy import URDF

    urdf_path = Path(urdf_path).resolve()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        urdf = URDF.load(
            str(urdf_path),
            build_scene_graph=True,       # required for get_transform / FK
            load_meshes=False,            # we'll load meshes ourselves
            load_collision_meshes=False,
        )

    # Find the root: a link not appearing as any joint's child.
    children = {urdf.joint_map[j].child for j in urdf.joint_map}
    roots = [n for n in urdf.link_map if n not in children]
    if not roots:
        raise ValueError(f"No root link in {urdf_path} (joint graph cycles?)")
    root = roots[0]

    # BFS from root to get a stable link ordering.
    link_names: list[str] = []
    seen: set[str] = set()
    queue: list[str] = [root]
    out_edges: dict[str, list[str]] = {}
    for jname, j in urdf.joint_map.items():
        out_edges.setdefault(j.parent, []).append(j.child)
    while queue:
        n = queue.pop(0)
        if n in seen:
            continue
        seen.add(n)
        link_names.append(n)
        for c in out_edges.get(n, []):
            if c not in seen:
                queue.append(c)
    # Append any orphans at the end so we don't silently drop them.
    for n in urdf.link_map:
        if n not in seen:
            link_names.append(n)

    link_idx = {n: i for i, n in enumerate(link_names)}

    actuated = list(urdf.actuated_joint_names)
    joint_types: list[str] = []
    jlo: list[float] = []
    jhi: list[float] = []
    jpar: list[int] = []
    jch: list[int] = []
    # Sane physical caps: revolute/continuous in [-π, π]; prismatic in [-2m, 2m].
    # Some URDFs (e.g. Fetch wheel base) have millions-meter limits which
    # produce nonsense if sampled uniformly.
    REV_CAP_RAD = np.pi
    PRISM_CAP_M = 2.0
    for jname in actuated:
        j = urdf.joint_map[jname]
        joint_types.append(j.type)
        lim = j.limit
        if j.type == "continuous":
            lo, hi = -REV_CAP_RAD, REV_CAP_RAD
        elif lim is not None and lim.lower is not None and lim.upper is not None:
            lo, hi = float(lim.lower), float(lim.upper)
        else:
            # No limits given: assume revolute-like
            lo, hi = -REV_CAP_RAD, REV_CAP_RAD
        # Clamp to physical sanity caps
        if j.type in ("revolute", "continuous"):
            lo = max(lo, -REV_CAP_RAD)
            hi = min(hi,  REV_CAP_RAD)
        elif j.type == "prismatic":
            lo = max(lo, -PRISM_CAP_M)
            hi = min(hi,  PRISM_CAP_M)
        if hi < lo:
            lo, hi = -REV_CAP_RAD, REV_CAP_RAD   # fall back if clamp inverted
        jlo.append(lo)
        jhi.append(hi)
        jpar.append(link_idx.get(j.parent, -1))
        jch.append(link_idx.get(j.child, -1))

    return LoadedRobot(
        name=urdf_path.stem,
        urdf_path=urdf_path,
        urdf=urdf,
        link_names=link_names,
        link_idx=link_idx,
        root_link=root,
        actuated_joint_names=actuated,
        joint_types=joint_types,
        joint_lower=jlo,
        joint_upper=jhi,
        joint_parent=jpar,
        joint_child=jch,
    )


# ---------------------------------------------------------------------------
# Joint-config sampling
# ---------------------------------------------------------------------------

def sample_random_config(
    robot: LoadedRobot,
    rng: np.random.Generator | None = None,
    centered_fraction: float = 0.5,
) -> np.ndarray:
    """Sample a random joint configuration within limits.

    `centered_fraction` (0–1): with this probability we sample a value
    closer to the joint center (±25% of half-range), which produces fewer
    extreme self-collision cases. Otherwise uniform in [lower, upper].
    """
    if rng is None:
        rng = np.random.default_rng()
    cfg = np.zeros(len(robot.actuated_joint_names))
    for i, (lo, hi, jt) in enumerate(zip(
        robot.joint_lower, robot.joint_upper, robot.joint_types,
    )):
        if jt == "fixed":
            cfg[i] = 0.0
            continue
        if rng.random() < centered_fraction:
            mid = 0.5 * (lo + hi)
            half = 0.25 * (hi - lo)
            cfg[i] = rng.uniform(mid - half, mid + half)
        else:
            cfg[i] = rng.uniform(lo, hi)
    return cfg


# ---------------------------------------------------------------------------
# Articulation + mesh assembly
# ---------------------------------------------------------------------------

def _resolve_mesh_filename(filename: str, urdf_dir: Path) -> Path | None:
    """Resolve a URDF mesh filename (plain relative, file://, or package://)
    to an on-disk path. Walks parent directories for `package://` URIs."""
    if filename.startswith("file://"):
        filename = filename[len("file://"):]
    if filename.startswith("package://"):
        rest = filename[len("package://"):]
        parts = rest.split("/", 1)
        if len(parts) < 2:
            return None
        pkg, sub = parts
        cur = urdf_dir.resolve()
        # Walk up to the filesystem root
        while True:
            candidate = cur / pkg / sub
            if candidate.exists():
                return candidate
            if cur.parent == cur:
                break
            cur = cur.parent
        # Fallback: search any folder named `pkg` under urdf_dir
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


def _load_link_visual_mesh(
    urdf, urdf_path: Path, link_name: str,
    mesh_cache: dict[str, trimesh.Trimesh] | None = None,
) -> trimesh.Trimesh | None:
    """Load and concat all visual meshes belonging to a link, applying any
    per-visual `<origin>` offsets and per-mesh scale. Returns one trimesh in
    the LINK-LOCAL frame, or None if the link has no usable visuals."""
    link = urdf.link_map.get(link_name)
    if link is None or not link.visuals:
        return None
    parts: list[trimesh.Trimesh] = []
    for v in link.visuals:
        geom = v.geometry
        if geom is None or geom.mesh is None:
            continue
        fn = geom.mesh.filename or ""
        full = _resolve_mesh_filename(fn, urdf_path.parent)
        if full is None or not full.exists():
            continue
        # Cache by absolute path because some URDFs reuse a mesh across
        # multiple links.
        key = str(full)
        if mesh_cache is not None and key in mesh_cache:
            mesh = mesh_cache[key].copy()
        else:
            try:
                mesh = trimesh.load(str(full), force="mesh", process=False)
            except Exception:
                continue
            if isinstance(mesh, trimesh.Scene):
                mesh = mesh.dump(concatenate=True)
            if mesh_cache is not None:
                mesh_cache[key] = mesh.copy()
        if geom.mesh.scale is not None:
            mesh.apply_scale(geom.mesh.scale)
        if v.origin is not None:
            mesh.apply_transform(v.origin)
        parts.append(mesh)
    if not parts:
        return None
    if len(parts) == 1:
        return parts[0]
    return trimesh.util.concatenate(parts)


def articulate_and_label(
    robot: LoadedRobot,
    config: np.ndarray,
    mesh_cache: dict[str, trimesh.Trimesh] | None = None,
    return_per_link: bool = False,
) -> tuple:
    """Articulate the robot to `config` and produce a labeled mesh.

    Returns:
        combined_mesh    : trimesh in world frame, all link visuals merged
        vertex_labels    : (V,) per-vertex link index (matches `link_idx`)
        joint_axes_world : (J, 3) unit axes at the articulated pose
        joint_origins_world : (J, 3) pivots
        joint_types_int  : (J,) integer encoding
        joint_topology   : (J, 2) array of (parent_link_idx, child_link_idx)
        joint_limits     : (J, 2) — (lower, upper) per joint, in chain
                           order matching the rest. Radians for revolute
                           and continuous, metres for prismatic, (0,0)
                           for fixed/floating/planar. Used as the
                           regression target for the model's LimitsHead.

    If `return_per_link=True`, also returns:
        per_link_meshes  : dict[link_name -> trimesh in world frame]
                           — useful for building a properly-segmented Scene
                           visualization (one entry per link with its own
                           material) instead of relying on vertex colors of
                           a concatenated mesh, which many GLB viewers ignore.
    """
    urdf = robot.urdf
    # Apply config and ask yourdfpy for FK. yourdfpy expects an ordered
    # vector matching `actuated_joint_names`.
    urdf.update_cfg(np.asarray(config, dtype=float))

    parts: list[trimesh.Trimesh] = []
    label_chunks: list[np.ndarray] = []
    per_link_meshes: dict[str, trimesh.Trimesh] = {}
    for link_name in robot.link_names:
        link_mesh = _load_link_visual_mesh(urdf, robot.urdf_path, link_name, mesh_cache)
        if link_mesh is None:
            continue
        try:
            T_world = urdf.get_transform(frame_to=link_name)
        except Exception:
            # Some URDFs have orphan links; place at origin and warn.
            T_world = np.eye(4)
        link_mesh = link_mesh.copy()
        link_mesh.apply_transform(T_world)
        parts.append(link_mesh)
        label_chunks.append(np.full(len(link_mesh.vertices),
                                     robot.link_idx[link_name], dtype=np.int32))
        per_link_meshes[link_name] = link_mesh

    if not parts:
        # Robot had no usable visuals — return empty.
        empty = trimesh.Trimesh(vertices=np.zeros((0, 3)), faces=np.zeros((0, 3), dtype=int))
        result = (empty, np.zeros(0, dtype=np.int32),
                  np.zeros((0, 3)), np.zeros((0, 3)),
                  np.zeros(0, dtype=np.int32), np.zeros((0, 2), dtype=np.int32),
                  np.zeros((0, 2), dtype=np.float32))
        if return_per_link:
            return (*result, {})
        return result

    combined = trimesh.util.concatenate(parts)
    vertex_labels = np.concatenate(label_chunks).astype(np.int32)

    # Per-joint world-frame pose at this articulated configuration.
    J = len(robot.actuated_joint_names)
    axes_w = np.zeros((J, 3))
    origins_w = np.zeros((J, 3))
    jtypes_int = np.zeros(J, dtype=np.int32)
    topology = np.zeros((J, 2), dtype=np.int32)
    joint_limits = np.zeros((J, 2), dtype=np.float32)
    for i, jname in enumerate(robot.actuated_joint_names):
        j = urdf.joint_map[jname]
        T_parent_world = urdf.get_transform(frame_to=j.parent)
        T_joint_in_parent = j.origin if j.origin is not None else np.eye(4)
        T_world_joint = T_parent_world @ T_joint_in_parent
        # The child's frame at this articulation rotates by the joint angle
        # around its local axis. We bake that rotation into T_world_joint
        # so axis_world reflects the actual articulated state.
        jt = robot.joint_types[i]
        if jt in ("revolute", "continuous"):
            ax_local = np.asarray(j.axis, dtype=float)
            ax_local = ax_local / (np.linalg.norm(ax_local) + 1e-12)
            theta = float(config[i])
            R = _rodrigues(ax_local, theta)
            T_post = np.eye(4)
            T_post[:3, :3] = R
            T_world_child = T_world_joint @ T_post
        elif jt == "prismatic":
            ax_local = np.asarray(j.axis, dtype=float)
            ax_local = ax_local / (np.linalg.norm(ax_local) + 1e-12)
            T_post = np.eye(4)
            T_post[:3, 3] = ax_local * float(config[i])
            T_world_child = T_world_joint @ T_post
        else:  # fixed / floating / planar — treat as no extra transform
            T_world_child = T_world_joint

        axes_w[i] = T_world_child[:3, :3] @ np.asarray(j.axis, dtype=float)
        axes_w[i] /= (np.linalg.norm(axes_w[i]) + 1e-12)
        origins_w[i] = T_world_joint[:3, 3]
        jtypes_int[i] = JOINT_TYPE_TO_INT.get(jt, JOINT_TYPE_TO_INT["fixed"])
        topology[i, 0] = robot.joint_parent[i]
        topology[i, 1] = robot.joint_child[i]
        joint_limits[i, 0] = float(robot.joint_lower[i])
        joint_limits[i, 1] = float(robot.joint_upper[i])

    if return_per_link:
        return (combined, vertex_labels, axes_w, origins_w, jtypes_int,
                topology, joint_limits, per_link_meshes)
    return (combined, vertex_labels, axes_w, origins_w, jtypes_int,
            topology, joint_limits)


def _rodrigues(axis: np.ndarray, theta: float) -> np.ndarray:
    """Rotation matrix from axis-angle (Rodrigues' formula)."""
    a = axis / (np.linalg.norm(axis) + 1e-12)
    K = np.array([
        [    0.0, -a[2],  a[1]],
        [  a[2],   0.0, -a[0]],
        [ -a[1],  a[0],   0.0],
    ])
    s = np.sin(theta)
    c = np.cos(theta)
    return np.eye(3) + s * K + (1.0 - c) * (K @ K)


# ---------------------------------------------------------------------------
# Point-cloud sampling
# ---------------------------------------------------------------------------

def sample_point_cloud(
    mesh: trimesh.Trimesh,
    vertex_labels: np.ndarray,
    n_points: int = 16384,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample N points uniformly over the mesh surface (weighted by area),
    propagating per-vertex labels via the face's majority label.

    Returns (points, point_labels).
    """
    if rng is None:
        rng = np.random.default_rng()
    if len(mesh.faces) == 0:
        return np.zeros((0, 3)), np.zeros(0, dtype=np.int32)

    # Choose face indices weighted by area.
    face_areas = mesh.area_faces
    if face_areas.sum() <= 0:
        return np.zeros((0, 3)), np.zeros(0, dtype=np.int32)
    face_probs = face_areas / face_areas.sum()
    face_idx = rng.choice(len(mesh.faces), size=n_points, p=face_probs)

    # Random barycentric coordinates per chosen face.
    r1 = rng.random(n_points)
    r2 = rng.random(n_points)
    swap = r1 + r2 > 1.0
    r1[swap] = 1.0 - r1[swap]
    r2[swap] = 1.0 - r2[swap]
    bary = np.stack([1.0 - r1 - r2, r1, r2], axis=1)

    tris = mesh.vertices[mesh.faces[face_idx]]    # (n_points, 3, 3)
    points = np.einsum("ni,nij->nj", bary, tris)

    # Per-face label: majority of the 3 vertex labels (use label of vertex 0
    # as fallback when all three differ).
    face_vert_labels = vertex_labels[mesh.faces[face_idx]]   # (n_points, 3)
    # Cheap majority via mode along axis=1
    face_labels = _mode_axis1(face_vert_labels)
    return points, face_labels.astype(np.int32)


def _mode_axis1(arr: np.ndarray) -> np.ndarray:
    """Per-row mode for an integer array of shape (N, 3)."""
    a, b, c = arr[:, 0], arr[:, 1], arr[:, 2]
    out = a.copy()
    out[(b == c) & (b != a)] = b[(b == c) & (b != a)]
    return out
