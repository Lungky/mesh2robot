"""Standalone CLI test for the VLM prior — Phase E.1.

Loads a mesh, renders 4 canonical views, asks the VLM what kind of
robot it is, prints the structured prior. Optionally saves the
rendered views to disk for human inspection.

Usage:
    python scripts/test_vlm_prior.py \\
        --mesh input/test_2/milo/xarm6_clean.obj \\
        --mesh-to-world input/test_2/T_cleaned_to_original.npy \\
        --save-views output/test_2_vlm_prior

    python scripts/test_vlm_prior.py \\
        --mesh input/test_3/milo/robot_custom.obj \\
        --save-views output/test_3_vlm_prior
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mesh2robot.core.vlm_prior import BACKENDS, get_robot_prior


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--mesh-to-world", type=Path, default=None,
                        help="Optional 4×4 transform .npy applied before VLM.")
    parser.add_argument("--backend", choices=list(BACKENDS), default="gemini")
    parser.add_argument("--save-views", type=Path, default=None,
                        help="If set, the 4 canonical-angle PNGs are saved "
                             "here so you can check what the VLM saw.")
    parser.add_argument("--resolution", type=int, default=768)
    args = parser.parse_args()

    print(f"Loading mesh: {args.mesh}")
    mesh = trimesh.load(str(args.mesh), force="mesh")
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)
    if args.mesh_to_world is not None:
        T = np.load(args.mesh_to_world)
        mesh = mesh.copy()
        mesh.apply_transform(T)
    print(f"  {len(mesh.vertices)} verts, {len(mesh.faces)} faces, "
          f"AABB extent={[round(x, 3) for x in mesh.extents]}")

    print(f"\nCalling VLM backend={args.backend} ...")
    t0 = time.time()
    prior = get_robot_prior(
        mesh,
        backend=args.backend,
        resolution=(args.resolution, args.resolution),
        save_dir=args.save_views,
    )
    elapsed = time.time() - t0
    print(f"  done in {elapsed:.1f} s")
    print()
    print(prior)


if __name__ == "__main__":
    main()
