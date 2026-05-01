"""Phase 0 — unified robot URDF database.

Enumerates serial-manipulator URDFs from `robot_descriptions`, extracts their
kinematic + physical signatures, and writes a JSON database used downstream
for retrieval, axis transfer, and physics defaults.

Each record contains:
  - Coarse signature: (dof, joint_types) for fast filtering.
  - Physics: link masses, density estimate, friction/damping defaults.
  - Per-joint kinematics: limits, effort, velocity, local-frame axes.
  - Home-pose geometry (added 2026-04-25 for the AI/retrieval approach):
      - joint_origins_world / joint_axes_world: per-joint world-frame pose
        when all joints are at angle 0. Used for axis transfer and matching.
      - link_aabb_min / link_aabb_max: per-link world-frame bounding boxes
        (None for links with no visual mesh).
      - total_height / total_width / joint_z_fractions: a compact mesh shape
        descriptor for retrieval.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import trimesh
from robot_descriptions.loaders.yourdfpy import load_robot_description


# Curated list of serial arms to ingest. Only actively supported ones — skip
# xacro-dependent or flaky loaders at this stage.
CANDIDATE_ARMS = [
    "xarm6_description",
    "xarm7_description",
    "panda_description",
    "ur3_description",
    "ur5_description",
    "ur10_description",
    "ur3e_description",
    "ur5e_description",
    "ur10e_description",
    "iiwa7_description",
    "iiwa14_description",
    "gen3_description",
    "gen3_lite_description",
]

DEFAULT_FRICTION = 0.5
DEFAULT_DAMPING = 0.1


@dataclass
class RobotRecord:
    name: str
    family: str
    dof: int
    joint_types: list[str]
    joint_axes: list[list[float]]
    joint_limits: list[list[float]]
    joint_efforts: list[float]
    joint_velocities: list[float]
    link_masses: list[float]
    total_mass: float
    density_guess: float
    friction_default: float = DEFAULT_FRICTION
    damping_default: float = DEFAULT_DAMPING
    # Home-pose geometry for retrieval + transfer
    joint_origins_world: list[list[float]] = field(default_factory=list)
    joint_axes_world: list[list[float]] = field(default_factory=list)
    link_names: list[str] = field(default_factory=list)
    link_aabb_min: list[list[float] | None] = field(default_factory=list)
    link_aabb_max: list[list[float] | None] = field(default_factory=list)
    total_height: float = 0.0
    total_width: float = 0.0
    arm_z_min: float = 0.0
    arm_z_max: float = 0.0
    joint_z_fractions: list[float] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def _extract_family(name: str) -> str:
    # "xarm6_description" -> "xarm"
    stem = name.replace("_description", "").replace("_mj", "")
    # Strip trailing digits + letter variants (e.g. "ur5e" -> "ur")
    core = stem.rstrip("0123456789")
    if core.endswith("e"):
        core = core[:-1]
    return core or stem


def _link_world_aabb(
    urdf, link_name: str, T_world_link: np.ndarray,
) -> tuple[list[float], list[float]] | None:
    """Compute the link's visual-mesh AABB in world coordinates at a given
    link pose. Returns (aabb_min, aabb_max) or None if no mesh available."""
    link = urdf.link_map.get(link_name)
    if link is None or not link.visuals:
        return None
    visual = link.visuals[0]
    geom = visual.geometry
    if geom.mesh is None:
        return None
    fn = geom.mesh.filename
    if fn.startswith("file://"):
        fn = fn[len("file://"):]
    try:
        mesh = trimesh.load(fn, force="mesh")
        if isinstance(mesh, trimesh.Scene):
            mesh = mesh.dump(concatenate=True)
        if geom.mesh.scale is not None:
            mesh.apply_scale(geom.mesh.scale)
        if visual.origin is not None:
            mesh.apply_transform(visual.origin)
        mesh.apply_transform(T_world_link)
        b = mesh.bounds  # shape (2, 3)
        return b[0].tolist(), b[1].tolist()
    except Exception:
        return None


def _link_volume_m3(urdf, link_name: str) -> float | None:
    """Return the visual-mesh volume (m^3), or None if no mesh is available."""
    link = urdf.link_map.get(link_name)
    if link is None or not link.visuals:
        return None
    geom = link.visuals[0].geometry
    if geom.mesh is None:
        return None
    fn = geom.mesh.filename
    if fn.startswith("file://"):
        fn = fn[len("file://"):]
    try:
        mesh = trimesh.load(fn, force="mesh")
        if geom.mesh.scale is not None:
            mesh.apply_scale(geom.mesh.scale)
        # trimesh.volume can be negative for inside-out normals — use signed absolute
        return float(abs(mesh.volume))
    except Exception:
        return None


def ingest(name: str) -> RobotRecord:
    errors: list[str] = []
    try:
        urdf = load_robot_description(name)
    except Exception as e:
        return RobotRecord(
            name=name, family=_extract_family(name), dof=0,
            joint_types=[], joint_axes=[], joint_limits=[],
            joint_efforts=[], joint_velocities=[], link_masses=[],
            total_mass=0.0, density_guess=0.0,
            errors=[f"load_failed: {e}"],
        )

    jtypes, jaxes, jlims, jeff, jvel = [], [], [], [], []
    for jname in urdf.actuated_joint_names:
        j = urdf.joint_map[jname]
        jtypes.append(j.type)
        ax = np.asarray(j.axis, dtype=float)
        ax = ax / (np.linalg.norm(ax) + 1e-12)
        jaxes.append(ax.tolist())
        lim = j.limit
        lo = float(lim.lower) if lim is not None and lim.lower is not None else 0.0
        hi = float(lim.upper) if lim is not None and lim.upper is not None else 0.0
        ef = float(lim.effort) if lim is not None and lim.effort is not None else 0.0
        vl = float(lim.velocity) if lim is not None and lim.velocity is not None else 0.0
        jlims.append([lo, hi])
        jeff.append(ef)
        jvel.append(vl)

    # Link masses in chain order (actuated-joint child order)
    link_names_chain = []
    for jname in urdf.actuated_joint_names:
        link_names_chain.append(urdf.joint_map[jname].child)

    link_masses: list[float] = []
    total_mass = 0.0
    total_volume = 0.0
    for lname in link_names_chain:
        link = urdf.link_map.get(lname)
        mass = 0.0
        if link is not None and link.inertial is not None:
            mass = float(link.inertial.mass or 0.0)
        link_masses.append(mass)
        total_mass += mass
        v = _link_volume_m3(urdf, lname)
        if v is not None:
            total_volume += v

    density = total_mass / total_volume if total_volume > 1e-9 else 0.0

    # ── Home-pose geometry ───────────────────────────────────────────────
    # Set all actuated joints to 0 and compute world-frame poses.
    joint_origins_world: list[list[float]] = []
    joint_axes_world: list[list[float]] = []
    try:
        urdf.update_cfg(np.zeros(len(urdf.actuated_joint_names)))
        for jname in urdf.actuated_joint_names:
            j = urdf.joint_map[jname]
            T_world_parent = urdf.get_transform(frame_to=j.parent)
            T_joint_in_parent = j.origin if j.origin is not None else np.eye(4)
            T_world_joint = T_world_parent @ T_joint_in_parent
            origin_w = T_world_joint[:3, 3]
            axis_local = np.asarray(j.axis, dtype=float)
            axis_local /= (np.linalg.norm(axis_local) + 1e-12)
            axis_w = T_world_joint[:3, :3] @ axis_local
            axis_w /= (np.linalg.norm(axis_w) + 1e-12)
            joint_origins_world.append(origin_w.tolist())
            joint_axes_world.append(axis_w.tolist())
    except Exception as e:
        errors.append(f"home_pose_fk_failed: {e}")

    # Per-link AABBs at home pose. We collect the root link plus the child
    # of every actuated joint (chain order = link_base, link1, link2, ...).
    link_names_for_aabb: list[str] = []
    children = {urdf.joint_map[j].child for j in urdf.joint_map}
    roots = [n for n in urdf.link_map if n not in children]
    if roots:
        link_names_for_aabb.append(roots[0])
    link_names_for_aabb.extend(link_names_chain)

    link_aabb_min: list[list[float] | None] = []
    link_aabb_max: list[list[float] | None] = []
    overall_min = np.array([np.inf, np.inf, np.inf])
    overall_max = np.array([-np.inf, -np.inf, -np.inf])
    for lname in link_names_for_aabb:
        try:
            T_world_link = urdf.get_transform(frame_to=lname)
        except Exception:
            link_aabb_min.append(None)
            link_aabb_max.append(None)
            continue
        ab = _link_world_aabb(urdf, lname, T_world_link)
        if ab is None:
            link_aabb_min.append(None)
            link_aabb_max.append(None)
            continue
        amin, amax = ab
        link_aabb_min.append(amin)
        link_aabb_max.append(amax)
        overall_min = np.minimum(overall_min, np.asarray(amin))
        overall_max = np.maximum(overall_max, np.asarray(amax))

    if np.isfinite(overall_min).all():
        arm_z_min = float(overall_min[2])
        arm_z_max = float(overall_max[2])
        total_height = float(arm_z_max - arm_z_min)
        total_width = float(max(
            overall_max[0] - overall_min[0],
            overall_max[1] - overall_min[1],
        ))
    else:
        arm_z_min = 0.0
        arm_z_max = 0.0
        total_height = 0.0
        total_width = 0.0

    # Joint Z-fractions: joint origin Z relative to overall Z range
    joint_z_fractions: list[float] = []
    if total_height > 1e-9:
        for ow in joint_origins_world:
            joint_z_fractions.append((ow[2] - arm_z_min) / total_height)
    else:
        joint_z_fractions = [0.0] * len(joint_origins_world)

    return RobotRecord(
        name=name,
        family=_extract_family(name),
        dof=len(urdf.actuated_joint_names),
        joint_types=jtypes,
        joint_axes=jaxes,
        joint_limits=jlims,
        joint_efforts=jeff,
        joint_velocities=jvel,
        link_masses=link_masses,
        total_mass=total_mass,
        density_guess=density,
        joint_origins_world=joint_origins_world,
        joint_axes_world=joint_axes_world,
        link_names=link_names_for_aabb,
        link_aabb_min=link_aabb_min,
        link_aabb_max=link_aabb_max,
        total_height=total_height,
        total_width=total_width,
        arm_z_min=arm_z_min,
        arm_z_max=arm_z_max,
        joint_z_fractions=joint_z_fractions,
        errors=errors,
    )


def build_database(
    candidates: list[str] | None = None,
    output_path: Path | None = None,
) -> list[dict[str, Any]]:
    candidates = candidates or CANDIDATE_ARMS
    records = []
    for name in candidates:
        print(f"Ingesting {name} ...")
        try:
            rec = ingest(name)
        except Exception as e:
            rec = RobotRecord(
                name=name, family=_extract_family(name), dof=0,
                joint_types=[], joint_axes=[], joint_limits=[],
                joint_efforts=[], joint_velocities=[], link_masses=[],
                total_mass=0.0, density_guess=0.0,
                errors=[f"ingest_failed: {type(e).__name__}: {e}"],
            )
        records.append(asdict(rec))
        if rec.errors:
            print(f"  ! {rec.errors}")
        else:
            print(f"  dof={rec.dof}  mass={rec.total_mass:.2f} kg  "
                  f"rho~{rec.density_guess:.0f} kg/m^3  "
                  f"H={rec.total_height:.3f}m  W={rec.total_width:.3f}m  "
                  f"j_z_frac={[f'{f:.2f}' for f in rec.joint_z_fractions]}")

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(records, indent=2))
        print(f"\nWrote {len(records)} records -> {output_path}")
    return records


if __name__ == "__main__":
    out = Path(__file__).resolve().parents[2] / "data" / "urdf_db.json"
    build_database(output_path=out)
