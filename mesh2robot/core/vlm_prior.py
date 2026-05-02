"""VLM-based robot identification from a 3D mesh — Phase E.1.

Renders the input mesh from 4 canonical angles, sends to a multimodal
LLM, parses a structured robot description into a `RobotPrior`
dataclass. The prior is consumed downstream by
`predict_urdf_interactive.py` (behind the `--vlm-prior` flag) to:

  - prune the model's over-segmented clusters to `expected_link_count`
  - bound the chain length emitted in the URDF
  - sanity-check the segmentation post-hoc

Triggered by test_3 failing with 18 ML clusters on what's likely a
~6-link robot — the model has no semantic prior for unfamiliar
geometry. A VLM saying "this is a 6-DOF tabletop arm" cuts the
over-segmentation problem at the root.

Backend swappable via the `VLMClient` Protocol; default `GeminiVLM`
uses Google's free-tier `gemini-2.5-flash`.
"""

from __future__ import annotations

import io
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np
import trimesh


# ---------------------------------------------------------------------------
# Public dataclass + canonical class enum
# ---------------------------------------------------------------------------

ROBOT_CLASSES = [
    "industrial_arm",   # 6/7-DOF floor-mounted manipulator
    "tabletop_arm",     # smaller bench-top arm (xArm-class)
    "humanoid",         # bipedal with arms
    "biped",            # legs only, no upper body
    "quadruped",        # 4-legged
    "wheeled",          # mobile base
    "gripper",          # 2- to 5-finger end-effector (no DOF chain)
    "dexterous_hand",   # human-like multi-DOF hand
    "finger",           # single articulated finger
    "drone",            # aerial
    "other",
    "unknown",          # VLM couldn't tell
]

AXIS_HINTS = ["vertical", "horizontal", "diagonal", "unknown"]
TOPOLOGIES = ["serial", "tree", "parallel", "unknown"]


@dataclass
class JointHint:
    description: str   # "shoulder yaw, base of the arm"
    axis_hint: str     # one of AXIS_HINTS
    location: str      # rough location, e.g. "lower-base"


@dataclass
class RobotPrior:
    robot_class: str
    expected_dof: int
    expected_link_count: int
    expected_chain_topology: str
    visible_joints: list[JointHint] = field(default_factory=list)
    confidence: float = 0.0
    visual_summary: str = ""

    def is_high_confidence(self) -> bool:
        return self.confidence >= 0.7

    def __str__(self) -> str:
        lines = [
            "VLM prior:",
            f"  robot_class:    {self.robot_class}",
            f"  expected_dof:   {self.expected_dof}",
            f"  expected_links: {self.expected_link_count}",
            f"  topology:       {self.expected_chain_topology}",
            f"  confidence:     {self.confidence:.2f}",
            f"  summary:        {self.visual_summary}",
        ]
        if self.visible_joints:
            lines.append(f"  visible_joints ({len(self.visible_joints)}):")
            for j in self.visible_joints:
                lines.append(f"    - {j.description}  "
                             f"(axis={j.axis_hint}, at {j.location})")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Canonical-view renderer (PyVista offscreen)
# ---------------------------------------------------------------------------

def render_canonical_views(
    mesh: trimesh.Trimesh,
    resolution: tuple[int, int] = (768, 768),
    background: str = "white",
    mesh_color: str = "lightgray",
) -> list[bytes]:
    """Render 4 canonical-angle PNG bytes of the mesh.

    Order: front (+X), right (+Y), back (-X), 3/4 (45° az, 30° el).
    Camera always looks at the mesh's centroid; distance auto-fits
    by bbox diagonal × 1.5.

    Returns list of 4 PNG bytestrings, ready to send to a VLM.
    """
    import pyvista as pv
    from PIL import Image

    # Convert trimesh -> PyVista PolyData
    faces = np.asarray(mesh.faces)
    pv_faces = np.column_stack([np.full(len(faces), 3), faces]).flatten()
    pv_mesh = pv.PolyData(np.asarray(mesh.vertices), pv_faces)

    centroid = mesh.centroid
    bbox_diag = float(np.linalg.norm(mesh.extents))
    cam_dist = bbox_diag * 1.5

    # (eye, focal, up) for each canonical view
    views = [
        # Front — camera at +X
        ([centroid[0] + cam_dist, centroid[1], centroid[2]],
         centroid.tolist(), [0, 0, 1]),
        # Right side — camera at +Y
        ([centroid[0], centroid[1] + cam_dist, centroid[2]],
         centroid.tolist(), [0, 0, 1]),
        # Back — camera at -X
        ([centroid[0] - cam_dist, centroid[1], centroid[2]],
         centroid.tolist(), [0, 0, 1]),
        # 3/4 — 45° azimuth + slight elevation
        ([centroid[0] + cam_dist * 0.707,
          centroid[1] + cam_dist * 0.707,
          centroid[2] + cam_dist * 0.5],
         centroid.tolist(), [0, 0, 1]),
    ]

    images: list[bytes] = []
    for cam_pos in views:
        plotter = pv.Plotter(off_screen=True, window_size=resolution)
        plotter.background_color = background
        plotter.add_mesh(
            pv_mesh,
            color=mesh_color,
            show_edges=False,
            smooth_shading=True,
            ambient=0.3,
            diffuse=0.7,
        )
        plotter.camera_position = cam_pos
        plotter.enable_anti_aliasing()
        img_arr = plotter.screenshot(return_img=True, transparent_background=False)
        plotter.close()

        buf = io.BytesIO()
        Image.fromarray(img_arr).save(buf, format="PNG")
        images.append(buf.getvalue())

    return images


# ---------------------------------------------------------------------------
# VLM clients
# ---------------------------------------------------------------------------

PROMPT_TEMPLATE = """You are looking at 4 rendered views of a 3D robot mesh:
  1. Front view (camera on +X axis)
  2. Right side view (camera on +Y axis)
  3. Back view (camera on -X axis)
  4. 3/4 view (45° azimuth, slight elevation)

Identify what kind of robot this is and describe its kinematic structure
in detail. Be specific. This prior will be used to constrain a downstream
segmentation model that otherwise over-segments unfamiliar robots.

Guidelines:
- robot_class: pick the closest match from the schema's enum. If you
  genuinely cannot tell, use "unknown".
- expected_dof: the number of ACTUATED joints. Articulated finger
  segments count; passive rolling wheels do not.
- expected_link_count: typically expected_dof + 1 (base + N moving links).
  For tree topologies (humanoids, multi-arm rigs) it's higher.
- expected_chain_topology: "serial" for a single kinematic chain (most
  arms), "tree" for branching (humanoid: torso → arms + legs),
  "parallel" for parallel mechanisms (delta robots, Stewart platforms).
- visible_joints: describe each joint you identify with axis direction
  and rough location. List as many as you can.
- confidence: 0.0-1.0 honest self-assessment. Use < 0.5 if the views
  are ambiguous, the robot is unusual, or you're uncertain about
  expected_dof.
- visual_summary: 1-2 sentences in plain English describing what you see.
"""


class VLMClient(Protocol):
    """Backend-agnostic VLM interface."""
    def classify_robot(self, images: list[bytes]) -> RobotPrior: ...


# Schema dict in JSON Schema form; google-genai accepts this directly.
_RESPONSE_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "robot_class": {"type": "string", "enum": ROBOT_CLASSES},
        "expected_dof": {"type": "integer", "minimum": 0, "maximum": 100},
        "expected_link_count": {"type": "integer", "minimum": 1, "maximum": 100},
        "expected_chain_topology": {"type": "string", "enum": TOPOLOGIES},
        "visible_joints": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "description": {"type": "string"},
                    "axis_hint": {"type": "string", "enum": AXIS_HINTS},
                    "location": {"type": "string"},
                },
                "required": ["description", "axis_hint", "location"],
            },
        },
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "visual_summary": {"type": "string"},
    },
    "required": [
        "robot_class", "expected_dof", "expected_link_count",
        "expected_chain_topology", "visible_joints",
        "confidence", "visual_summary",
    ],
}


class GeminiVLM:
    """Google Gemini backend via google-genai SDK.

    Uses the free-tier `gemini-2.5-flash` model by default. Call costs
    ~$0.01 with 4 images at 768x768; well under the 1500 RPD free quota.
    """

    def __init__(
        self,
        model: str = "gemini-2.5-flash",
        api_key: str | None = None,
    ) -> None:
        from google import genai
        api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY env var not set. Export it first:\n"
                "  echo 'export GEMINI_API_KEY=\"AIza...\"' >> ~/.bashrc"
            )
        self.client = genai.Client(api_key=api_key)
        self.model = model

    def classify_robot(self, images: list[bytes]) -> RobotPrior:
        from google.genai import types
        import json

        contents: list = [PROMPT_TEMPLATE]
        for img_bytes in images:
            contents.append(
                types.Part.from_bytes(data=img_bytes, mime_type="image/png")
            )

        response = self.client.models.generate_content(
            model=self.model,
            contents=contents,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=_RESPONSE_SCHEMA,
                temperature=0.2,    # low — we want consistent classification
            ),
        )
        data = json.loads(response.text)
        return _dict_to_prior(data)


# Backend registry — add AnthropicVLM, OpenAIVLM, etc. here later.
BACKENDS: dict[str, type[VLMClient]] = {
    "gemini": GeminiVLM,
}


def _dict_to_prior(d: dict) -> RobotPrior:
    return RobotPrior(
        robot_class=d.get("robot_class", "unknown"),
        expected_dof=int(d.get("expected_dof", 0)),
        expected_link_count=int(d.get("expected_link_count", 0)),
        expected_chain_topology=d.get("expected_chain_topology", "unknown"),
        visible_joints=[
            JointHint(
                description=j.get("description", ""),
                axis_hint=j.get("axis_hint", "unknown"),
                location=j.get("location", ""),
            )
            for j in d.get("visible_joints", [])
        ],
        confidence=float(d.get("confidence", 0.0)),
        visual_summary=d.get("visual_summary", ""),
    )


# ---------------------------------------------------------------------------
# High-level convenience
# ---------------------------------------------------------------------------

def get_robot_prior(
    mesh: trimesh.Trimesh,
    backend: str = "gemini",
    resolution: tuple[int, int] = (768, 768),
    save_dir: Path | None = None,
) -> RobotPrior:
    """Render the mesh + ask the VLM what kind of robot it is.

    If `save_dir` is provided, the 4 canonical-angle PNGs are also
    written there (useful for debugging the VLM's view of the input).
    """
    images = render_canonical_views(mesh, resolution=resolution)

    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        names = ["front", "right", "back", "three_quarter"]
        for name, img_bytes in zip(names, images):
            (save_dir / f"vlm_view_{name}.png").write_bytes(img_bytes)
        print(f"  Saved 4 canonical views to {save_dir}")

    if backend not in BACKENDS:
        raise ValueError(
            f"Unknown VLM backend: {backend!r}. "
            f"Available: {list(BACKENDS)}"
        )
    vlm = BACKENDS[backend]()
    return vlm.classify_robot(images)
