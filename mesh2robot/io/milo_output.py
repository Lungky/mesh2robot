"""Load MILO outputs — generated mesh + camera intrinsics/extrinsics.

MILO writes its camera poses as a JSON file next to the mesh. The exact schema
hasn't been finalized for this project, so this module ships with a configurable
loader. Populate `schema` (or replace `load_milo_cameras` entirely) to match
whatever MILO actually produces.

Common 3DGS / MILO-adjacent conventions (so that the reader can cover both):
  - NeRF-studio transforms.json
      {"fl_x":..., "fl_y":..., "cx":..., "cy":..., "w":..., "h":...,
       "frames":[{"file_path":..., "transform_matrix":[[...],[...],[...],[...]]}, ...]}
      transform_matrix is typically camera-to-world in the OpenGL convention
      (X right, Y up, Z back). We convert to OpenCV (X right, Y down, Z forward).
  - COLMAP images.txt / cameras.txt (handled separately if needed).

Outputs a uniform `MiloScene` containing the mesh in world frame + a dict of
per-image camera poses in world->cam OpenCV convention.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import trimesh


# ---------------------------------------------------------------------------
# Conventions
# ---------------------------------------------------------------------------

# Convert OpenGL camera-to-world (X right, Y up, Z back) to OpenCV
# camera-to-world (X right, Y down, Z forward) by flipping Y and Z axes of
# the *camera* frame. Equivalent to multiplying the cam-to-world transform
# by diag([1,-1,-1,1]) on the right.
OPENGL_TO_OPENCV_CAMERA = np.diag([1.0, -1.0, -1.0, 1.0])


@dataclass
class MiloCamera:
    name: str                             # the filename or identifier
    T_world_to_cam: np.ndarray            # 4x4 OpenCV convention
    K: np.ndarray                         # 3x3 intrinsics
    image_size: tuple[int, int]           # (width, height) in pixels
    dist_coeffs: np.ndarray = field(default_factory=lambda: np.zeros(5))


@dataclass
class MiloScene:
    mesh: trimesh.Trimesh
    cameras: dict[str, MiloCamera]        # keyed by name
    schema: str                           # which loader produced this


def load_mesh(mesh_path: str | Path) -> trimesh.Trimesh:
    mesh = trimesh.load(str(mesh_path), force="mesh")
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)
    return mesh


def _nerfstudio_transforms_to_cameras(
    data: dict, base_dir: Path, convert_opengl: bool = True,
) -> dict[str, MiloCamera]:
    """Convert a NeRF-studio style transforms.json dict to MiloCameras.

    Shared intrinsics if top-level fl_x/cx/cy/w/h present; per-frame if not.
    """
    cameras: dict[str, MiloCamera] = {}
    top_k = _k_from_dict(data)
    top_size = _size_from_dict(data)

    for frame in data.get("frames", []):
        name = frame.get("file_path") or frame.get("name") or f"frame_{len(cameras):04d}"
        K = _k_from_dict(frame) or top_k
        size = _size_from_dict(frame) or top_size
        if K is None or size is None:
            continue

        M = np.asarray(frame["transform_matrix"], dtype=np.float64)   # cam-to-world
        if convert_opengl:
            M = M @ OPENGL_TO_OPENCV_CAMERA
        T_world_to_cam = np.linalg.inv(M)
        cameras[str(name)] = MiloCamera(
            name=str(name),
            T_world_to_cam=T_world_to_cam,
            K=K,
            image_size=size,
        )
    return cameras


def _k_from_dict(d: dict) -> np.ndarray | None:
    if "fl_x" in d:
        fx = float(d["fl_x"])
        fy = float(d.get("fl_y", fx))
        cx = float(d.get("cx", 0.0))
        cy = float(d.get("cy", 0.0))
        return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    if "camera_matrix" in d:
        return np.asarray(d["camera_matrix"], dtype=np.float64)
    return None


def _size_from_dict(d: dict) -> tuple[int, int] | None:
    if "w" in d and "h" in d:
        return int(d["w"]), int(d["h"])
    if "width" in d and "height" in d:
        return int(d["width"]), int(d["height"])
    if "image_size" in d:
        sz = d["image_size"]
        return int(sz[0]), int(sz[1])
    return None


def _3dgs_list_to_cameras(data: list) -> dict[str, MiloCamera]:
    """Parse the INRIA 3DGS / MILO camera list.

    Each entry:
      {
        "id": int, "img_name": "0001", "width": ..., "height": ...,
        "position": [x, y, z],             # camera center in world
        "rotation": [[...],[...],[...]],   # 3x3 world-to-camera rotation (OpenCV)
        "fx": ..., "fy": ...
      }
    """
    cameras: dict[str, MiloCamera] = {}
    for entry in data:
        name = str(entry.get("img_name") or entry.get("id") or len(cameras))
        fx = float(entry["fx"])
        fy = float(entry.get("fy", fx))
        w = int(entry["width"])
        h = int(entry["height"])
        # 3DGS output commonly omits principal point; assume image center.
        cx = float(entry.get("cx", w / 2.0))
        cy = float(entry.get("cy", h / 2.0))
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)

        # INRIA 3DGS convention (which MILO inherits):
        #   position = camera center in world
        #   rotation = camera-to-world 3x3 rotation
        # So T_world_to_cam needs the transpose and a recomputed translation.
        R_c2w = np.asarray(entry["rotation"], dtype=np.float64)
        C = np.asarray(entry["position"], dtype=np.float64)
        R_w2c = R_c2w.T
        T_world_to_cam = np.eye(4)
        T_world_to_cam[:3, :3] = R_w2c
        T_world_to_cam[:3, 3] = -R_w2c @ C

        cameras[name] = MiloCamera(
            name=name,
            T_world_to_cam=T_world_to_cam,
            K=K,
            image_size=(w, h),
        )
    return cameras


def load_milo_cameras(
    cameras_json: str | Path,
    schema: str = "3dgs",
) -> dict[str, MiloCamera]:
    """Dispatch to the appropriate parser.

    Supported values of `schema`:
      - "3dgs"        — INRIA 3DGS / MILO format: list of {position, rotation, fx, fy, ...}
      - "nerfstudio"  — transforms.json with 'frames' list (transform_matrix per frame)
    """
    p = Path(cameras_json)
    data = json.loads(p.read_text())

    if schema == "3dgs":
        if not isinstance(data, list):
            raise ValueError(
                "3dgs schema expects a top-level JSON list of camera entries; "
                f"got {type(data).__name__}"
            )
        return _3dgs_list_to_cameras(data)

    if schema == "nerfstudio":
        return _nerfstudio_transforms_to_cameras(data, base_dir=p.parent)

    raise ValueError(
        f"Unknown MILO camera schema: {schema!r}. Inspect the JSON and add a "
        f"parser in mesh2robot.io.milo_output."
    )


def load_milo_scene(
    mesh_path: str | Path,
    cameras_json: str | Path,
    schema: str = "3dgs",
) -> MiloScene:
    return MiloScene(
        mesh=load_mesh(mesh_path),
        cameras=load_milo_cameras(cameras_json, schema=schema),
        schema=schema,
    )
