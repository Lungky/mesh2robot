"""Phase E.4b — VLM critic for the assembled URDF.

After predict_urdf_interactive writes the refined URDF + runs the
collision sweep, this module:

  1. Renders the URDF (with FK applied) from the same 4 canonical
     angles used by `vlm_prior.render_canonical_views`.
  2. Sends BOTH sets of views (input mesh + assembled URDF) to a
     multimodal LLM along with the structured prior + a topology
     summary, and asks: "compare. What's wrong? Suggest fixes."
  3. Parses the structured response as `CritiqueResult` with a list
     of typed issues.
  4. Optionally returns merge actions the caller can apply (the
     v1 "actor" — only safe fix is merging two links the critic
     thinks are the same body part).

The critic does NOT directly modify the URDF. The caller decides
whether to apply the suggested merges and re-assemble. Riskier
actions (re-parent, delete) are deferred to v2.
"""

from __future__ import annotations

import io
import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import trimesh

from mesh2robot.core.vlm_prior import (
    BACKENDS,
    GeminiVLM,
    RobotPrior,
    _prior_to_dict,
)


# ---------------------------------------------------------------------------
# Structured response dataclasses
# ---------------------------------------------------------------------------

ISSUE_TYPES = [
    "extra_phantom_link",      # spurious cluster, no real body part
    "missing_link",            # expected body part not in URDF
    "wrong_topology",          # branching wrong (e.g. arm attached to leg)
    "wrong_joint_axis",        # axis direction looks off vs visible geometry
    "wrong_joint_location",    # joint origin in wrong place
    "duplicate_link",          # two clusters that should be merged
    "asymmetry_violation",     # left and right limbs mismatched
    "other",
]

SEVERITIES = ["low", "medium", "high", "critical"]


@dataclass
class Issue:
    type: str                        # one of ISSUE_TYPES
    severity: str                    # one of SEVERITIES
    description: str                 # plain English explanation
    affected_links: list[int] = field(default_factory=list)
    suggested_action: str = ""       # e.g. "merge link_5 into link_3"


@dataclass
class MergeAction:
    """A safe-to-auto-apply link merge: relabel `sources` → `target` in
    face_labels, then re-assemble the URDF."""
    target: int
    sources: list[int]
    rationale: str = ""


@dataclass
class CritiqueResult:
    matches_well: bool
    overall_score: float             # 0..1 — VLM's confidence the URDF is good
    issues: list[Issue] = field(default_factory=list)
    auto_fix_merges: list[MergeAction] = field(default_factory=list)
    summary: str = ""                # 1-2 sentences

    def has_high_severity(self) -> bool:
        return any(i.severity in ("high", "critical") for i in self.issues)

    def merge_actions(self) -> list[MergeAction]:
        """Return the merges the VLM marked as safe to auto-apply."""
        return [m for m in self.auto_fix_merges if m.sources]

    def __str__(self) -> str:
        lines = [
            f"VLM critique:",
            f"  matches_well: {self.matches_well}",
            f"  overall_score: {self.overall_score:.2f}",
            f"  summary: {self.summary}",
        ]
        if self.issues:
            lines.append(f"  issues ({len(self.issues)}):")
            for i in self.issues:
                tag = {"low": "·", "medium": "!", "high": "‼", "critical": "✗"}.get(
                    i.severity, "?"
                )
                lines.append(f"    [{tag}] {i.severity:8s} {i.type}: "
                             f"{i.description}")
                if i.affected_links:
                    lines.append(f"          affected_links: {i.affected_links}")
                if i.suggested_action:
                    lines.append(f"          suggested:      {i.suggested_action}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Render URDF canonical views (sister of vlm_prior.render_canonical_views)
# ---------------------------------------------------------------------------

def render_urdf_canonical_views(
    urdf_path: Path,
    resolution: tuple[int, int] = (768, 768),
    background: str = "white",
) -> list[bytes]:
    """Render 4 canonical-angle PNG bytes of the URDF (with FK applied).

    Same camera layout as `vlm_prior.render_canonical_views`. Each link
    is colour-coded so the critic can see segmentation directly.
    """
    import pyvista as pv
    from PIL import Image
    from yourdfpy import URDF

    urdf = URDF.load(str(urdf_path), build_scene_graph=True, load_meshes=True)
    urdf_dir = Path(urdf_path).parent

    # Build per-link world transforms (same recursive FK as visualize_refined)
    parent_joint_of_link: dict[str, object] = {}
    for jname, j in urdf.joint_map.items():
        parent_joint_of_link[j.child] = j

    def link_world_transform(link_name: str) -> np.ndarray:
        if link_name not in parent_joint_of_link:
            return np.eye(4)
        j = parent_joint_of_link[link_name]
        parent_T = link_world_transform(j.parent)
        local_T = np.asarray(j.origin) if j.origin is not None else np.eye(4)
        return parent_T @ local_T

    # Distinct colour per link
    rng = np.random.default_rng(42)
    n_links = len(urdf.link_map)
    palette = (rng.uniform(0.4, 0.95, (n_links, 3))).tolist()

    # Combine all link meshes (in world frame) for rendering
    all_polydatas: list[tuple[pv.PolyData, list[float]]] = []
    all_extents: list[np.ndarray] = []
    for link_idx, (link_name, link) in enumerate(urdf.link_map.items()):
        T_world = link_world_transform(link_name)
        for v in (link.visuals or []):
            if v.geometry is None or v.geometry.mesh is None:
                continue
            fn = v.geometry.mesh.filename
            if not fn:
                continue
            mp = ((urdf_dir / fn).resolve()
                  if not Path(fn).is_absolute() else Path(fn))
            if not mp.exists():
                continue
            sub = trimesh.load(str(mp), force="mesh")
            if isinstance(sub, trimesh.Scene):
                sub = sub.dump(concatenate=True)
            T_local = np.asarray(v.origin) if v.origin is not None else np.eye(4)
            T = T_world @ T_local
            sub.apply_transform(T)

            faces = np.asarray(sub.faces)
            pv_faces = np.column_stack(
                [np.full(len(faces), 3), faces]).flatten()
            pv_mesh = pv.PolyData(np.asarray(sub.vertices), pv_faces)
            all_polydatas.append((pv_mesh, palette[link_idx % len(palette)]))
            all_extents.append(np.asarray(sub.extents))

    if not all_polydatas:
        raise RuntimeError(f"URDF {urdf_path} has no renderable meshes")

    # Compute scene centroid + bbox for camera fitting
    all_centroids = np.stack(
        [pd.center for pd, _ in all_polydatas]
    )
    centroid = all_centroids.mean(axis=0)
    max_extent = float(max(e.max() for e in all_extents))
    bbox_diag = max_extent * len(all_polydatas) ** 0.5  # rough
    cam_dist = max(bbox_diag * 0.5, max_extent * 2.0)

    views = [
        ([centroid[0] + cam_dist, centroid[1], centroid[2]],
         centroid.tolist(), [0, 0, 1]),
        ([centroid[0], centroid[1] + cam_dist, centroid[2]],
         centroid.tolist(), [0, 0, 1]),
        ([centroid[0] - cam_dist, centroid[1], centroid[2]],
         centroid.tolist(), [0, 0, 1]),
        ([centroid[0] + cam_dist * 0.707,
          centroid[1] + cam_dist * 0.707,
          centroid[2] + cam_dist * 0.5],
         centroid.tolist(), [0, 0, 1]),
    ]

    images: list[bytes] = []
    for cam_pos in views:
        plotter = pv.Plotter(off_screen=True, window_size=resolution)
        plotter.background_color = background
        for pd, color in all_polydatas:
            plotter.add_mesh(
                pd, color=color,
                show_edges=False, smooth_shading=True,
                ambient=0.3, diffuse=0.7,
            )
        plotter.camera_position = cam_pos
        plotter.enable_anti_aliasing()
        img_arr = plotter.screenshot(return_img=True,
                                     transparent_background=False)
        plotter.close()

        buf = io.BytesIO()
        Image.fromarray(img_arr).save(buf, format="PNG")
        images.append(buf.getvalue())
    return images


# ---------------------------------------------------------------------------
# Critic prompt + Gemini call
# ---------------------------------------------------------------------------

CRITIC_PROMPT_TEMPLATE = """You are evaluating an automatically-generated URDF
robot model against the original 3D mesh.

INPUTS:
  - 4 canonical-angle views of the ORIGINAL input mesh (light grey, no
    segmentation).
  - 4 canonical-angle views of the GENERATED URDF, with each link
    colour-coded.

CONTEXT:
  - The robot was independently classified by a different VLM call as:
    {prior_summary}
  - The URDF has {n_links} links and {n_actuated} actuated joints.
  - Topology used: {topology_summary}
  - Link ID numbering: when you reference a link by index, use the integer
    that matches its position in the URDF (link_0 = root/torso, link_1 =
    first child, etc.). You'll have to identify links by their colours
    in the rendered views.

YOUR TASK has TWO parts:

PART 1 — Diagnostic issues (the `issues` array).
  List EVERY abnormality you can identify. For each:
    - type: pick the closest match from schema enum.
    - severity: 'low' / 'medium' / 'high' / 'critical'.
    - description: 1-2 sentences.
    - affected_links: list of integer link IDs.
    - suggested_action: a concrete instruction.
  Issues are for HUMAN REPORTING. They don't get auto-applied.

PART 2 — Auto-fix merges (the `auto_fix_merges` array).
  This is SEPARATE from issues. Populate it ONLY with link merges you are
  CONFIDENT are correct. The pipeline will relabel face_labels +
  re-assemble the URDF. Each merge:
    - target: the link ID that will REMAIN.
    - sources: list of link IDs to be DISSOLVED into target.
    - rationale: 1 sentence why.

  When to add a merge to auto_fix_merges:
    - You are SURE the source links are spurious sub-segments of the
      target. Bilateral-symmetry hands, over-segmented torsos, phantom
      tip links — these qualify.
  When NOT to add a merge:
    - The fix is "add a missing link" — there's no merge to do; report
      it in issues only.
    - You're uncertain which side is the right target.
    - The fix involves re-parenting (changing topology), not merging.

  If there are no safe merges, return auto_fix_merges as an empty array.

PART 3 — Overall verdict.
  - matches_well: true ONLY if every issue is severity 'low'.
  - overall_score: 0..1 confidence the URDF is good.
  - summary: 1-2 sentences plain-English verdict.
"""


_CRITIC_RESPONSE_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "matches_well": {"type": "boolean"},
        "overall_score": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "issues": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "type": {"type": "string", "enum": ISSUE_TYPES},
                    "severity": {"type": "string", "enum": SEVERITIES},
                    "description": {"type": "string"},
                    "affected_links": {
                        "type": "array",
                        "items": {"type": "integer"},
                    },
                    "suggested_action": {"type": "string"},
                },
                "required": ["type", "severity", "description"],
            },
        },
        "auto_fix_merges": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "target": {"type": "integer"},
                    "sources": {
                        "type": "array",
                        "items": {"type": "integer"},
                    },
                    "rationale": {"type": "string"},
                },
                "required": ["target", "sources", "rationale"],
            },
        },
        "summary": {"type": "string"},
    },
    "required": ["matches_well", "overall_score", "issues",
                 "auto_fix_merges", "summary"],
}


def critique_urdf(
    input_views: list[bytes],
    urdf_views: list[bytes],
    prior: RobotPrior | None,
    n_links: int,
    n_actuated: int,
    topology_summary: str,
    backend: str = "gemini",
    model: str = "gemini-2.5-flash",
) -> CritiqueResult:
    """One-shot critic call: input views || urdf views → CritiqueResult."""
    if backend != "gemini":
        raise ValueError(f"Critic only supports gemini backend; got {backend!r}")

    from google import genai
    from google.genai import types

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY env var not set")

    client = genai.Client(api_key=api_key)

    if prior is not None:
        prior_str = (f"{prior.robot_class} | dof≈{prior.expected_dof} | "
                     f"links≈{prior.expected_link_count} | "
                     f"topology={prior.expected_chain_topology} | "
                     f"summary: {prior.visual_summary}")
    else:
        prior_str = "no prior available"

    prompt = CRITIC_PROMPT_TEMPLATE.format(
        prior_summary=prior_str,
        n_links=n_links,
        n_actuated=n_actuated,
        topology_summary=topology_summary,
    )

    contents: list = [prompt]
    contents.append("\n--- INPUT MESH VIEWS ---\n")
    for img_bytes in input_views:
        contents.append(types.Part.from_bytes(
            data=img_bytes, mime_type="image/png"
        ))
    contents.append("\n--- GENERATED URDF VIEWS (links colour-coded) ---\n")
    for img_bytes in urdf_views:
        contents.append(types.Part.from_bytes(
            data=img_bytes, mime_type="image/png"
        ))

    response = client.models.generate_content(
        model=model,
        contents=contents,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=_CRITIC_RESPONSE_SCHEMA,
            temperature=0.2,
        ),
    )
    data = json.loads(response.text)
    return _dict_to_critique(data)


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------

def _dict_to_critique(d: dict) -> CritiqueResult:
    return CritiqueResult(
        matches_well=bool(d.get("matches_well", False)),
        overall_score=float(d.get("overall_score", 0.0)),
        issues=[
            Issue(
                type=i.get("type", "other"),
                severity=i.get("severity", "low"),
                description=i.get("description", ""),
                affected_links=[int(x) for x in i.get("affected_links", [])],
                suggested_action=i.get("suggested_action", ""),
            )
            for i in d.get("issues", [])
        ],
        auto_fix_merges=[
            MergeAction(
                target=int(m.get("target")),
                sources=[int(x) for x in m.get("sources", [])],
                rationale=m.get("rationale", ""),
            )
            for m in d.get("auto_fix_merges", [])
            if m.get("target") is not None and m.get("sources")
        ],
        summary=d.get("summary", ""),
    )


def _critique_to_dict(c: CritiqueResult) -> dict:
    return {
        "matches_well": c.matches_well,
        "overall_score": c.overall_score,
        "summary": c.summary,
        "issues": [
            {
                "type": i.type,
                "severity": i.severity,
                "description": i.description,
                "affected_links": list(i.affected_links),
                "suggested_action": i.suggested_action,
            }
            for i in c.issues
        ],
        "auto_fix_merges": [
            {
                "target": m.target,
                "sources": list(m.sources),
                "rationale": m.rationale,
            }
            for m in c.auto_fix_merges
        ],
    }
