"""Quick smoke test for the synthetic data generator.

Picks a few diverse trainable robots from the manifest, generates one
training example per robot, saves them to data/synthetic_train_smoke/,
and prints sanity stats (points/labels shape, label range, joint counts).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mesh2robot.data_gen import (
    articulate_and_label,
    load_robot,
    sample_point_cloud,
    sample_random_config,
)


def smoke_test_one(raw_dir: Path, entry: dict, out_dir: Path) -> bool:
    name = f"{entry['source']}/{Path(entry['path']).stem}"
    print(f"\n=== {name} (DOF={entry['dof']}, "
          f"vendor={entry.get('vendor', '?')}) ===")
    full_path = raw_dir / entry["path"]
    try:
        robot = load_robot(full_path)
    except Exception as e:
        print(f"  load_robot failed: {type(e).__name__}: {e}")
        return False
    print(f"  links={len(robot.link_names)}  joints={len(robot.actuated_joint_names)}")

    rng = np.random.default_rng(0)
    cfg = sample_random_config(robot, rng=rng)
    print(f"  sampled config: {np.round(cfg, 2).tolist()[:8]}{'...' if len(cfg) > 8 else ''}")

    try:
        result = articulate_and_label(
            robot, cfg, mesh_cache={}, return_per_link=True,
        )
        mesh, vlabels, axes_w, origins_w, jt_int, topo, jlims, per_link = result
    except Exception as e:
        print(f"  articulate_and_label failed: {type(e).__name__}: {e}")
        return False
    print(f"  combined mesh: {len(mesh.vertices)} verts, {len(mesh.faces)} faces")
    if len(mesh.vertices) == 0:
        print("  [warn] empty mesh — skipping")
        return False
    print(f"  vertex_labels: shape={vlabels.shape} "
          f"unique={np.unique(vlabels).tolist()[:10]}"
          f"{'...' if len(np.unique(vlabels)) > 10 else ''}")

    points, plabels = sample_point_cloud(mesh, vlabels, n_points=4096, rng=rng)
    print(f"  points: {points.shape}  labels: {plabels.shape}  "
          f"unique_labels={len(np.unique(plabels))}")
    print(f"  joint axes (first 3): {np.round(axes_w[:3], 3).tolist()}")
    print(f"  joint origins (first 3): {np.round(origins_w[:3], 3).tolist()}")

    # Save tensors
    out_dir.mkdir(parents=True, exist_ok=True)
    safe = name.replace("/", "__").replace("\\", "__")
    np.savez(
        out_dir / f"{safe}.npz",
        points=points.astype(np.float32),
        point_labels=plabels.astype(np.int32),
        joint_axes_world=axes_w.astype(np.float32),
        joint_origins_world=origins_w.astype(np.float32),
        joint_types=jt_int.astype(np.int32),
        joint_topology=topo.astype(np.int32),
        config=cfg.astype(np.float32),
    )

    # Visualization: a Scene with one node per link, each given a distinct
    # material color. This makes the segmentation OBVIOUSLY visible in any
    # GLB viewer (Blender, glb-viewer.com, three.js editor, etc.) instead
    # of relying on per-vertex colors which most viewers don't show.
    palette_rng = np.random.default_rng(42)
    scene = trimesh.Scene()
    for link_name, link_mesh in per_link.items():
        if len(link_mesh.vertices) == 0:
            continue
        # Bright, well-spaced HSV colors per link
        # (deterministic per-link via hash of name)
        hue = (hash(link_name) & 0xFFFFFF) / 0xFFFFFF
        rgb = _hsv_to_rgb(hue, 0.85, 0.95)
        link_mesh = link_mesh.copy()
        # Use a PBR material with a base color so it survives GLB export
        link_mesh.visual = trimesh.visual.TextureVisuals(
            material=trimesh.visual.material.PBRMaterial(
                baseColorFactor=[rgb[0], rgb[1], rgb[2], 1.0],
                name=f"link_{link_name}",
            ),
        )
        scene.add_geometry(link_mesh, node_name=link_name)
    scene.export(out_dir / f"{safe}.glb")
    print(f"  saved {out_dir / f'{safe}.npz'} + .glb (Scene with "
          f"{len(per_link)} link-meshes, each with its own material)")
    return True


def _hsv_to_rgb(h: float, s: float, v: float) -> tuple[float, float, float]:
    """HSV → RGB in [0,1]³, deterministic. Used for per-link palette."""
    i = int(h * 6.0)
    f = h * 6.0 - i
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))
    i %= 6
    return [(v, t, p), (q, v, p), (p, v, t),
            (p, q, v), (t, p, v), (v, p, q)][i]


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    raw_dir = repo / "data" / "raw_robots"
    manifest_path = repo / "data" / "robot_manifest.json"
    out_dir = repo / "data" / "synthetic_train_smoke"

    manifest = json.loads(manifest_path.read_text())
    trainable = [
        e for e in manifest
        if e["status"] == "ok" and e["dof"] >= 1
        and e["format"] == "urdf"
        and e["meshes_resolved"] / max(1, e["meshes_resolved"] + e["meshes_unresolved"]) >= 0.8
    ]
    print(f"Trainable URDFs in manifest: {len(trainable)}")

    # Diverse picks: an arm, a gripper, a humanoid, a quadruped, a wheeled.
    targets = []
    seen_vendors: set[str] = set()
    for vendor_target in ["ufactory", "franka", "kinova", "robotiq",
                          "anybotics", "boston_dynamics", "fetch_robotics",
                          "kuka", "universal_robots"]:
        for e in trainable:
            if e.get("vendor") == vendor_target and vendor_target not in seen_vendors:
                targets.append(e)
                seen_vendors.add(vendor_target)
                break

    if not targets:
        print("No suitable trainable URDFs found.")
        return

    print(f"Picked {len(targets)} robots for smoke test")
    n_ok = 0
    for entry in targets:
        ok = smoke_test_one(raw_dir, entry, out_dir)
        n_ok += 1 if ok else 0
    print(f"\n{'-'*60}\nSmoke test passed for {n_ok}/{len(targets)} robots")


if __name__ == "__main__":
    main()
