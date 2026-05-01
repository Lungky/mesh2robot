"""Generate synthetic K-pose mesh data from a URDF for feasibility testing.

Takes a URDF, a list of joint-configuration vectors, and produces one combined
mesh per pose (simulating a MILO scan of the robot in that pose) along with
per-vertex ground-truth link labels and ground-truth per-link transforms.

This is the noise-free input used to validate Phase 2 + Phase 3 before touching
real MILO output.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh
from robot_descriptions.loaders.yourdfpy import load_robot_description


@dataclass
class PoseMesh:
    """One synthetic pose: combined mesh + per-vertex link labels + link transforms."""

    pose_index: int
    joint_cfg: dict[str, float]
    vertices: np.ndarray              # (N, 3)
    faces: np.ndarray                 # (M, 3)
    vertex_link: np.ndarray           # (N,) int, index into link_names
    link_names: list[str]
    link_transforms: dict[str, np.ndarray]  # link_name -> 4x4


def _load_link_meshes(urdf) -> dict[str, trimesh.Trimesh]:
    """Load the visual mesh for each link that has one."""
    meshes = {}
    for lname, link in urdf.link_map.items():
        if not link.visuals:
            continue
        geom = link.visuals[0].geometry
        if geom.mesh is None:
            continue
        filename = geom.mesh.filename
        if filename.startswith("file://"):
            filename = filename[len("file://"):]
        mesh = trimesh.load(filename, force="mesh")
        origin = link.visuals[0].origin
        if origin is not None:
            mesh.apply_transform(origin)
        if geom.mesh.scale is not None:
            mesh.apply_scale(geom.mesh.scale)
        meshes[lname] = mesh
    return meshes


def generate_pose_meshes(
    description_name: str,
    configurations: list[dict[str, float]],
    output_dir: Path,
) -> list[PoseMesh]:
    """Produce one combined-mesh PoseMesh per joint configuration.

    configurations: list of {actuated_joint_name: angle_rad}.
    """
    urdf = load_robot_description(description_name)
    link_meshes = _load_link_meshes(urdf)
    link_names = list(link_meshes.keys())

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results: list[PoseMesh] = []
    for i, cfg in enumerate(configurations):
        urdf.update_cfg(cfg)

        all_v, all_f, vtx_link = [], [], []
        link_transforms = {}
        face_offset = 0
        for link_idx, lname in enumerate(link_names):
            T = urdf.get_transform(lname, "world")
            link_transforms[lname] = T.copy()
            mesh = link_meshes[lname].copy()
            mesh.apply_transform(T)
            n = len(mesh.vertices)
            all_v.append(np.asarray(mesh.vertices))
            all_f.append(np.asarray(mesh.faces) + face_offset)
            vtx_link.append(np.full(n, link_idx, dtype=np.int32))
            face_offset += n

        V = np.vstack(all_v)
        F = np.vstack(all_f)
        L = np.concatenate(vtx_link)

        pm = PoseMesh(
            pose_index=i,
            joint_cfg=cfg,
            vertices=V,
            faces=F,
            vertex_link=L,
            link_names=link_names,
            link_transforms=link_transforms,
        )
        results.append(pm)

        # Persist to disk for reuse
        npz_path = output_dir / f"pose_{i:02d}.npz"
        np.savez_compressed(
            npz_path,
            vertices=V,
            faces=F,
            vertex_link=L,
            link_names=np.array(link_names),
            link_transforms=np.stack([link_transforms[n] for n in link_names]),
        )
        # Dump combined mesh as OBJ for easy viewing
        trimesh.Trimesh(vertices=V, faces=F, process=False).export(
            output_dir / f"pose_{i:02d}.obj"
        )

    # Save pose metadata
    meta = {
        "description_name": description_name,
        "link_names": link_names,
        "actuated_joint_names": list(urdf.actuated_joint_names),
        "configurations": configurations,
    }
    (output_dir / "metadata.json").write_text(json.dumps(meta, indent=2))

    return results


def recommended_protocol(
    actuated_joints: list[str],
    angle_rad: float = np.deg2rad(45.0),
) -> list[dict[str, float]]:
    """Build the '2N + 1' capture protocol: home + each joint at +/- angle.

    Each pose isolates one joint's motion, matching the recommended capture
    protocol from the roadmap.
    """
    home = {j: 0.0 for j in actuated_joints}
    configs = [home]
    for j in actuated_joints:
        for sign in (-1.0, +1.0):
            cfg = dict(home)
            cfg[j] = sign * angle_rad
            configs.append(cfg)
    return configs


if __name__ == "__main__":
    urdf = load_robot_description("xarm6_description")
    cfgs = recommended_protocol(urdf.actuated_joint_names)
    out = Path(__file__).resolve().parents[2] / "data" / "synthetic" / "xarm6"
    results = generate_pose_meshes("xarm6_description", cfgs, out)
    print(f"Generated {len(results)} pose meshes -> {out}")
    print(f"Per-pose vertex count: {len(results[0].vertices)}")
    print(f"Links: {results[0].link_names}")
