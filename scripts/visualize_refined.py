"""GLB visualizer for the refined segmentation.

Two modes — pick whichever matches your intent:

  --from-urdf:  loads the URDF + its per-link meshes; renders each link's
                actual triangle set as it appears in the assembled URDF.
                This reflects the FINAL labels after the strict-mode
                merge has propagated user annotations through ML
                clusters. Faces the user didn't directly touch but that
                ended up assigned to a link via cluster majority vote
                appear in that link's color (no phantom "unlabeled"
                region).

  --from-annotations:  loads user_annotations.json and shows ONLY the
                       directly-clicked faces. Useful for spotting
                       holes in your annotation. Faces left unlabeled
                       render grey.

If both are given, --from-urdf wins.

Usage (from URDF, what you usually want):
    python scripts/visualize_refined.py \
        --urdf output/test_2_full_annotation/refined/robot.urdf \
        --out output/test_2_full_annotation/refined.glb

Usage (from raw user annotations, to see your lasso coverage):
    python scripts/visualize_refined.py \
        --mesh input/test_2/milo/xarm6_clean.obj \
        --mesh-to-world input/test_2/T_cleaned_to_original.npy \
        --annotations output/test_2_full_annotation/user_annotations.json \
        --out output/test_2_full_annotation/raw_annotations.glb
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import trimesh


def _hsv_to_rgb(h: float, s: float, v: float) -> tuple[float, float, float]:
    i = int(h * 6.0)
    f = h * 6.0 - i
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))
    i %= 6
    return [(v, t, p), (q, v, p), (p, v, t),
            (p, q, v), (t, p, v), (v, p, q)][i]


def make_palette(n: int = 64, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    hues = (np.arange(n) * 0.61803398875) % 1.0
    sats = 0.7 + rng.uniform(-0.1, 0.1, n)
    vals = 0.85 + rng.uniform(-0.1, 0.05, n)
    out = np.zeros((n, 3), dtype=np.float32)
    for i in range(n):
        out[i] = _hsv_to_rgb(hues[i], sats[i], vals[i])
    return out


def render_from_urdf(urdf_path: Path) -> trimesh.Scene:
    """Load the URDF, apply forward kinematics so each link's mesh sits
    at its assembled-pose world position, color each link distinctly.

    The per-link mesh files on disk are in LINK-LOCAL coords (each link
    centered around its own origin). Just loading them as-is would clump
    everything at (0,0,0). We use yourdfpy's scene-graph FK to compute
    each link's world transform and apply it to its mesh."""
    from yourdfpy import URDF
    urdf = URDF.load(str(urdf_path), build_scene_graph=True, load_meshes=True)
    print(f"  URDF has {len(urdf.link_map)} links, "
          f"{len(urdf.actuated_joint_names)} actuated joints")

    palette = make_palette()
    scene = trimesh.Scene()

    # Compute per-link world transforms via the joint chain at zero
    # configuration. yourdfpy stores them on each scene-graph node.
    # We walk the kinematic chain manually so we don't rely on any
    # specific yourdfpy version's helper.
    link_idx_for_name: dict[str, int] = {
        name: i for i, name in enumerate(urdf.link_map.keys())
    }

    # Build parent-joint map: for each link, find the joint where it's the child.
    parent_joint_of_link: dict[str, object] = {}
    for jname, j in urdf.joint_map.items():
        parent_joint_of_link[j.child] = j

    def link_world_transform(link_name: str) -> np.ndarray:
        """Recursively compose joint origins from base to this link."""
        if link_name not in parent_joint_of_link:
            return np.eye(4)
        j = parent_joint_of_link[link_name]
        parent_T = link_world_transform(j.parent)
        local_T = np.asarray(j.origin) if j.origin is not None else np.eye(4)
        return parent_T @ local_T

    all_extents: list[np.ndarray] = []
    urdf_dir = urdf_path.parent
    for link_name, link in urdf.link_map.items():
        link_idx = link_idx_for_name[link_name]
        T_world = link_world_transform(link_name)
        for v in (link.visuals or []):
            if v.geometry is None or v.geometry.mesh is None:
                continue
            fn = v.geometry.mesh.filename
            if not fn:
                continue
            mp = (urdf_dir / fn).resolve() if not Path(fn).is_absolute() else Path(fn)
            if not mp.exists():
                print(f"  WARN: missing mesh {mp}")
                continue
            sub = trimesh.load(str(mp), force="mesh")
            if isinstance(sub, trimesh.Scene):
                sub = sub.dump(concatenate=True)
            # Visual origin offset (some URDFs offset the visual within the link)
            T_local = np.asarray(v.origin) if v.origin is not None else np.eye(4)
            T = T_world @ T_local
            sub.apply_transform(T)
            color = palette[link_idx % len(palette)]
            sub.visual = trimesh.visual.TextureVisuals(
                material=trimesh.visual.material.PBRMaterial(
                    baseColorFactor=[float(color[0]), float(color[1]),
                                      float(color[2]), 1.0],
                    name=f"link_{link_idx}_{link_name}",
                ),
            )
            scene.add_geometry(sub, node_name=f"link_{link_idx}_{link_name}")
            all_extents.append(np.asarray(sub.extents))

    # Joint arrows in the assembled-pose world frame
    if all_extents:
        bbox_max = float(np.max([e.max() for e in all_extents]))
    else:
        bbox_max = 1.0
    scale = bbox_max * 0.08
    for jname in urdf.actuated_joint_names:
        j = urdf.joint_map[jname]
        # Joint origin in WORLD frame = parent-link world × joint origin
        T_parent = link_world_transform(j.parent)
        T_joint_local = np.asarray(j.origin) if j.origin is not None else np.eye(4)
        T_joint_world = T_parent @ T_joint_local
        origin = T_joint_world[:3, 3]
        # Joint axis is in the joint's local frame; rotate by the joint's
        # rotation in world for proper direction.
        axis_local = np.asarray(j.axis, dtype=np.float64) if j.axis is not None \
            else np.array([0.0, 0.0, 1.0])
        axis_local = axis_local / (np.linalg.norm(axis_local) + 1e-12)
        axis_world = T_joint_world[:3, :3] @ axis_local
        axis_world = axis_world / (np.linalg.norm(axis_world) + 1e-12)
        tip = origin + scale * axis_world
        try:
            arrow = trimesh.creation.cylinder(
                radius=0.005 * bbox_max,
                segment=[origin, tip],
            )
            arrow.visual = trimesh.visual.TextureVisuals(
                material=trimesh.visual.material.PBRMaterial(
                    baseColorFactor=[0.0, 0.0, 0.0, 1.0],
                    name=jname,
                ),
            )
            scene.add_geometry(arrow, node_name=jname)
        except Exception:
            pass
    return scene


def render_face_labels_on_mesh(
    mesh: trimesh.Trimesh,
    face_labels: np.ndarray,
) -> trimesh.Scene:
    """Render an already-loaded mesh colored by per-face labels.

    Faces with label < 0 render grey (useful for "rough annotation"
    views where the user only tagged some of the surface).
    """
    n_faces = len(mesh.faces)
    if len(face_labels) != n_faces:
        raise ValueError(
            f"face_labels length {len(face_labels)} != n_faces {n_faces}"
        )
    palette = make_palette()
    grey = np.array([0.65, 0.65, 0.65], dtype=np.float32)
    scene = trimesh.Scene()
    verts = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.faces)
    unique = sorted(int(l) for l in np.unique(face_labels) if l >= 0)

    for lbl in [-1] + unique:
        face_mask = face_labels == lbl
        if not face_mask.any():
            continue
        sub_faces = faces[face_mask]
        used = np.unique(sub_faces)
        remap = -np.ones(len(verts), dtype=np.int64)
        remap[used] = np.arange(len(used))
        sub_verts = verts[used]
        sub_faces_remap = remap[sub_faces]
        sub = trimesh.Trimesh(
            vertices=sub_verts, faces=sub_faces_remap, process=False,
        )
        color = grey if lbl < 0 else palette[lbl % len(palette)]
        name = "link_unlabeled" if lbl < 0 else f"link_{lbl}"
        sub.visual = trimesh.visual.TextureVisuals(
            material=trimesh.visual.material.PBRMaterial(
                baseColorFactor=[float(color[0]), float(color[1]),
                                  float(color[2]), 1.0],
                name=name,
            ),
        )
        scene.add_geometry(sub, node_name=name)
    return scene


def render_from_annotations(
    mesh_path: Path,
    mesh_to_world_path: Path | None,
    annotations_path: Path,
    urdf_path: Path | None,
) -> trimesh.Scene:
    """Render only the user-tagged faces colored, leaving untouched faces grey.
    Useful for spotting holes in the annotation."""
    print(f"Loading mesh: {mesh_path}")
    mesh = trimesh.load(str(mesh_path), force="mesh")
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)
    if mesh_to_world_path is not None:
        T = np.load(mesh_to_world_path)
        mesh = mesh.copy()
        mesh.apply_transform(T)
    print(f"  {len(mesh.vertices)} verts, {len(mesh.faces)} faces")

    print(f"Loading annotations: {annotations_path}")
    raw = json.loads(annotations_path.read_text())
    user_labels = {int(k): int(v) for k, v in raw.items()}
    n_faces = len(mesh.faces)
    face_labels = -np.ones(n_faces, dtype=np.int64)
    for face_idx, lbl in user_labels.items():
        if 0 <= face_idx < n_faces:
            face_labels[face_idx] = lbl
    n_unlabeled = int((face_labels < 0).sum())
    unique = sorted(int(l) for l in np.unique(face_labels) if l >= 0)
    print(f"  {len(user_labels)} user-labeled faces, {n_unlabeled} unlabeled "
          f"(rendered grey); link IDs {unique}")

    return render_face_labels_on_mesh(mesh, face_labels)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--urdf", type=Path, default=None,
                        help="Render the URDF's per-link meshes (post-merge "
                             "labels). Use this for the FINAL refined view.")
    parser.add_argument("--annotations", type=Path, default=None,
                        help="Render only user-tagged faces (raw lasso "
                             "coverage). Use this to spot un-annotated holes.")
    parser.add_argument("--mesh", type=Path, default=None,
                        help="Source mesh (required when --annotations is used).")
    parser.add_argument("--mesh-to-world", type=Path, default=None)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    if args.urdf is None and args.annotations is None:
        raise SystemExit("Pass --urdf (final view) or --annotations (raw coverage).")

    args.out.parent.mkdir(parents=True, exist_ok=True)

    if args.urdf is not None:
        print(f"Rendering from URDF: {args.urdf}")
        scene = render_from_urdf(args.urdf)
    else:
        if args.mesh is None:
            raise SystemExit("--mesh is required when --annotations is used.")
        print(f"Rendering from raw annotations: {args.annotations}")
        scene = render_from_annotations(
            args.mesh, args.mesh_to_world, args.annotations, args.urdf,
        )

    print(f"Exporting {args.out}")
    scene.export(args.out)
    print(f"  scene has {len(scene.geometry)} sub-meshes")


if __name__ == "__main__":
    main()
