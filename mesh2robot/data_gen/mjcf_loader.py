"""MJCF (MuJoCo XML) → labeled-mesh pipeline.

Mirrors the URDF loader API for synthetic-data generation, but uses the
mujoco library to parse + FK + mesh-extract MJCF files. Needed for
MuJoCo Menagerie's 182 trainable robots (Apollo, Spot, Cassie, Atlas,
Trossen arms, Allegro/Shadow hands, etc.) which ship as MJCF, not URDF.

Public API parallels `urdf_loader`:

    load_robot_mjcf(mjcf_path) -> LoadedRobot
    sample_random_config_mjcf(robot, rng=...) -> np.ndarray
    articulate_and_label_mjcf(robot, config, ...) -> same tuple as URDF version

Internal differences:
  - "Links" in URDF == "bodies" in MuJoCo
  - "Joints" in URDF: actuated joints. MuJoCo joints can be hinge/slide/ball/free;
    we only sample hinge + slide here.
  - FK is `mj_kinematics(model, data)` after setting `data.qpos`.
  - Visual meshes live in `model.mesh_*` arrays, indexed via
    `geom_dataid` from each geom whose `geom_type == mjGEOM_MESH`.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import trimesh

from mesh2robot.data_gen.urdf_loader import JOINT_TYPE_TO_INT, _rodrigues


# MuJoCo geom type 7 = mesh
MJ_GEOM_MESH = 7
# MuJoCo joint types
MJ_JOINT_FREE = 0
MJ_JOINT_BALL = 1
MJ_JOINT_SLIDE = 2     # prismatic
MJ_JOINT_HINGE = 3     # revolute


@dataclass
class LoadedRobotMJCF:
    """MJCF analog of LoadedRobot."""
    name: str
    mjcf_path: Path
    model: object              # mujoco.MjModel
    # Body info (= "links")
    body_names: list[str]
    body_idx: dict[str, int]
    # Joint info — only HINGE + SLIDE are surfaced as "actuated"
    actuated_joint_ids: list[int]    # indices into model.jnt_*
    joint_qpos_addrs: list[int]      # entry points into qpos for each
    joint_types: list[str]           # "revolute" / "prismatic"
    joint_lower: list[float]
    joint_upper: list[float]
    joint_parent: list[int]          # parent body index (=link)
    joint_child: list[int]           # child body index (=link)
    qpos_size: int                   # full qpos dim (incl. free / ball joints)
    qpos_init: np.ndarray            # default home qpos


def load_robot_mjcf(mjcf_path: Path | str) -> LoadedRobotMJCF:
    """Parse an MJCF file with the mujoco library."""
    import mujoco

    mjcf_path = Path(mjcf_path).resolve()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = mujoco.MjModel.from_xml_path(str(mjcf_path))

    # Body names
    body_names: list[str] = []
    for bi in range(model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bi) or f"body_{bi}"
        body_names.append(name)
    body_idx = {n: i for i, n in enumerate(body_names)}

    # Actuated joints: surface hinge + slide only.
    actuated_ids: list[int] = []
    qpos_addrs: list[int] = []
    jtypes: list[str] = []
    jlo: list[float] = []
    jhi: list[float] = []
    jpar: list[int] = []
    jch: list[int] = []
    for ji in range(model.njnt):
        jt = int(model.jnt_type[ji])
        if jt not in (MJ_JOINT_HINGE, MJ_JOINT_SLIDE):
            continue
        actuated_ids.append(ji)
        qpos_addrs.append(int(model.jnt_qposadr[ji]))
        jtypes.append("revolute" if jt == MJ_JOINT_HINGE else "prismatic")

        if model.jnt_limited[ji]:
            lo = float(model.jnt_range[ji, 0])
            hi = float(model.jnt_range[ji, 1])
        else:
            lo, hi = (-np.pi, np.pi) if jt == MJ_JOINT_HINGE else (-2.0, 2.0)
        # Sanity caps (avoid Fetch-style millions of meters)
        if jt == MJ_JOINT_HINGE:
            lo, hi = max(lo, -np.pi), min(hi, np.pi)
        else:
            lo, hi = max(lo, -2.0), min(hi, 2.0)
        if hi < lo:
            lo, hi = -np.pi, np.pi
        jlo.append(lo)
        jhi.append(hi)

        child_body = int(model.jnt_bodyid[ji])
        # Parent body in MuJoCo is body_parentid
        parent_body = int(model.body_parentid[child_body])
        jpar.append(parent_body)
        jch.append(child_body)

    return LoadedRobotMJCF(
        name=mjcf_path.stem,
        mjcf_path=mjcf_path,
        model=model,
        body_names=body_names,
        body_idx=body_idx,
        actuated_joint_ids=actuated_ids,
        joint_qpos_addrs=qpos_addrs,
        joint_types=jtypes,
        joint_lower=jlo,
        joint_upper=jhi,
        joint_parent=jpar,
        joint_child=jch,
        qpos_size=int(model.nq),
        qpos_init=np.array(model.qpos0, copy=True),
    )


def sample_random_config_mjcf(
    robot: LoadedRobotMJCF,
    rng: np.random.Generator | None = None,
    centered_fraction: float = 0.5,
) -> np.ndarray:
    """Sample joint values for the actuated subset only.

    Returns a full (nq,) qpos vector with non-actuated entries left at
    `qpos0` (so free / ball joints stay at their default rest pose).
    """
    if rng is None:
        rng = np.random.default_rng()
    qpos = robot.qpos_init.copy()
    for jt, addr, lo, hi in zip(
        robot.joint_types, robot.joint_qpos_addrs,
        robot.joint_lower, robot.joint_upper,
    ):
        if rng.random() < centered_fraction:
            mid = 0.5 * (lo + hi)
            half = 0.25 * (hi - lo)
            v = rng.uniform(mid - half, mid + half)
        else:
            v = rng.uniform(lo, hi)
        qpos[addr] = v
    return qpos


def articulate_and_label_mjcf(
    robot: LoadedRobotMJCF,
    qpos: np.ndarray,
    return_per_link: bool = False,
) -> tuple:
    """MJCF analog of articulate_and_label.

    1. Set data.qpos = qpos; run mj_kinematics to update body world poses.
    2. For each body, pull its visual mesh geoms (geom_type == 7), assemble
       transformed verts, tag with body index.
    3. Concatenate into one trimesh + label vector.
    4. Per-actuated-joint world axis + origin computed from data.xpos +
       data.xmat of the joint's child body.
    """
    import mujoco

    model = robot.model
    data = mujoco.MjData(model)
    data.qpos[:] = qpos
    mujoco.mj_kinematics(model, data)

    parts: list[trimesh.Trimesh] = []
    label_chunks: list[np.ndarray] = []
    per_link_meshes: dict[str, trimesh.Trimesh] = {}

    # Geoms with mesh data
    for gi in range(model.ngeom):
        if int(model.geom_type[gi]) != MJ_GEOM_MESH:
            continue
        mesh_id = int(model.geom_dataid[gi])
        if mesh_id < 0:
            continue
        bi = int(model.geom_bodyid[gi])

        # Pull verts/faces for this mesh
        v_start = int(model.mesh_vertadr[mesh_id])
        v_n = int(model.mesh_vertnum[mesh_id])
        f_start = int(model.mesh_faceadr[mesh_id])
        f_n = int(model.mesh_facenum[mesh_id])
        verts = np.array(model.mesh_vert[v_start:v_start + v_n]).reshape(-1, 3)
        faces = np.array(model.mesh_face[f_start:f_start + f_n]).reshape(-1, 3)
        if v_n == 0 or f_n == 0:
            continue

        # World transform of the geom = body world pose @ geom local pose.
        # data.xpos[bi], data.xmat[bi]: body's world frame
        # geom_pos[gi], geom_quat[gi]: geom relative to its body (local)
        body_R = np.asarray(data.xmat[bi]).reshape(3, 3)
        body_t = np.asarray(data.xpos[bi])
        geom_local_pos = np.asarray(model.geom_pos[gi])
        geom_local_quat = np.asarray(model.geom_quat[gi])  # (w, x, y, z)
        geom_local_R = _quat_wxyz_to_mat(geom_local_quat)

        R_world = body_R @ geom_local_R
        t_world = body_R @ geom_local_pos + body_t

        verts_world = (R_world @ verts.T).T + t_world

        m = trimesh.Trimesh(vertices=verts_world, faces=faces, process=False)
        parts.append(m)
        label_chunks.append(np.full(len(verts_world), bi, dtype=np.int32))
        if robot.body_names[bi] in per_link_meshes:
            per_link_meshes[robot.body_names[bi]] = trimesh.util.concatenate(
                [per_link_meshes[robot.body_names[bi]], m],
            )
        else:
            per_link_meshes[robot.body_names[bi]] = m

    if not parts:
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

    # Per-actuated-joint world pose
    J = len(robot.actuated_joint_ids)
    axes_w = np.zeros((J, 3))
    origins_w = np.zeros((J, 3))
    jtypes_int = np.zeros(J, dtype=np.int32)
    topology = np.zeros((J, 2), dtype=np.int32)
    joint_limits = np.zeros((J, 2), dtype=np.float32)
    for slot, ji in enumerate(robot.actuated_joint_ids):
        # Joint axis in WORLD = child body's xmat @ jnt_axis (local)
        child_bi = int(model.jnt_bodyid[ji])
        body_R = np.asarray(data.xmat[child_bi]).reshape(3, 3)
        body_t = np.asarray(data.xpos[child_bi])
        ax_local = np.asarray(model.jnt_axis[ji])
        ax_world = body_R @ ax_local
        ax_world /= (np.linalg.norm(ax_world) + 1e-12)
        # Joint origin in WORLD = body pose @ jnt_pos (local)
        origin_local = np.asarray(model.jnt_pos[ji])
        origin_world = body_R @ origin_local + body_t

        axes_w[slot] = ax_world
        origins_w[slot] = origin_world
        jtypes_int[slot] = JOINT_TYPE_TO_INT.get(robot.joint_types[slot],
                                                  JOINT_TYPE_TO_INT["fixed"])
        topology[slot, 0] = robot.joint_parent[slot]
        topology[slot, 1] = robot.joint_child[slot]
        joint_limits[slot, 0] = float(robot.joint_lower[slot])
        joint_limits[slot, 1] = float(robot.joint_upper[slot])

    if return_per_link:
        return (combined, vertex_labels, axes_w, origins_w, jtypes_int,
                topology, joint_limits, per_link_meshes)
    return (combined, vertex_labels, axes_w, origins_w, jtypes_int,
            topology, joint_limits)


def _quat_wxyz_to_mat(q: np.ndarray) -> np.ndarray:
    """MuJoCo quaternion convention is (w, x, y, z). Returns 3×3 rotation."""
    w, x, y, z = q
    n = np.sqrt(w*w + x*x + y*y + z*z) + 1e-12
    w, x, y, z = w/n, x/n, y/n, z/n
    return np.array([
        [1 - 2*(y*y + z*z),  2*(x*y - w*z),    2*(x*z + w*y)],
        [2*(x*y + w*z),     1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y),     2*(y*z + w*x),    1 - 2*(x*x + y*y)],
    ])
