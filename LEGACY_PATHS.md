# Legacy Heuristic Pipeline (pre-ML)

This document covers the two heuristic pipelines used in mesh2robot before the project pivoted to a learned 3D foundation model on 2026-04-25. Path A was the original roadmap (validated only on synthetic xArm6); **Path B** was the heuristic path that actually ran on a real MILO scan.

Both paths are retained as baselines for the paper and as fallbacks for cases where the ML model lacks confidence.

---

## 1. Pipeline diagrams

Two separate flowcharts (one per path) show inputs, processes, and inter-block artifacts. Both converge into the shared Phases 4–7 (last diagram).

### 1.1  Path A — K pose meshes (synthetic-only)

```
                              USER-PROVIDED INPUTS
                              ────────────────────
                ┌───────────────────────────────────────────┐
                │  K MILO meshes:                           │
                │    mesh_pose_0.ply  (reference)           │
                │    mesh_pose_1.ply  ...  mesh_pose_{K-1}  │
                │  Each in its own arbitrary world frame    │
                │  Gaussian anchors retained from MILO      │
                └───────────────────────────────────────────┘
                                    │
                                    ▼
       ┌──────────────────────────────────────────────────────────┐
       │  PHASE 1 — Co-register K meshes into common frame        │
       │                                                          │
       │   for k in 1..K-1:                                       │
       │       T_k = ICP(source=mesh_k.base, target=mesh_0.base)  │
       │       mesh_k := T_k @ mesh_k                             │
       │                                                          │
       │   out: K meshes in pose_0's frame                        │
       └──────────────────────────────────────────────────────────┘
                                    │
                                    ▼
       ┌──────────────────────────────────────────────────────────┐
       │  PHASE 2 — Motion-based segmentation                     │
       │                                                          │
       │   1. Track Gaussian anchors → per-vertex trajectory      │
       │        v_i(t)  for  t = 0 .. K-1                         │
       │   2. Hierarchical multi-pose LO-RANSAC                   │
       │        peel one rigid body per level                     │
       │   3. Body merge (collapse near-duplicate transforms)     │
       │   4. Orphan reassignment (unlabeled → nearest body)      │
       │                                                          │
       │   out: per-link meshes  +  T_body(t)  per body, per pose │
       └──────────────────────────────────────────────────────────┘
                                    │
                                    ▼
       ┌──────────────────────────────────────────────────────────┐
       │  PHASE 3 — Joint extraction                              │
       │                                                          │
       │   parent inference: motion-correlation between bodies    │
       │   for each (parent, child) pair:                         │
       │       rel(t)    = inv(T_parent(t)) @ T_child(t)          │
       │       motion(t) = inv(rel(0)) @ rel(t)                   │
       │       screw     = screw_from_transform(motion[t_best])   │
       │       axis      = R_parent_world @ screw.axis            │
       │       origin    = T_parent_world @ screw.origin          │
       │       limits    = [min, max] of per-pose angles          │
       │                                                          │
       │   out: list[JointEstimate]                               │
       └──────────────────────────────────────────────────────────┘
                                    │
                                    ▼
                          (shared Phases 4–7)
```

### 1.2  Path B — single mesh + per-joint photos (real-scan implementation)

```
                              USER-PROVIDED INPUTS  (2 categories)
                              ────────────────────────────────────
                ┌──────────────────────────────┐ ┌─────────────────────────┐
                │  MESH  (two files)           │ │  MOTION IMAGES          │
                │   ─ mesh.ply                 │ │   motion/joint_i/       │
                │     (MILO output: robot +    │ │     state_*.jpg         │
                │      whole environment,      │ │   (robot + ArUco board  │
                │      board-aligned frame)    │ │    visible in EVERY     │
                │   ─ cleaned.obj              │ │    frame — no separate  │
                │     (robot-only, derived in  │ │    calibration set)     │
                │      Blender, axis-flipped)  │ │                         │
                └──────────────────────────────┘ └─────────────────────────┘
                              │                              │
                              ▼                              ▼
       ┌──────────────────────────────────────────────────────────────┐
       │  PHASE 1 — Single-mesh prep (2 scripts; mesh + image lanes)  │
       │                                                              │
       │   MESH lane                          IMAGE lane              │
       │   ─────────                          ──────────              │
       │   ┌──────────────────────────┐  ┌──────────────────────────┐ │
       │   │ register_cleaned_to_     │  │ calibrate_camera_aruco.py│ │
       │   │   original.py            │  │                          │ │
       │   │                          │  │ in:  motion images       │ │
       │   │ in:  cleaned.obj  +      │  │      (the same shots used│ │
       │   │      mesh.ply            │  │      later for motion;   │ │
       │   │                          │  │      ArUco board in each)│ │
       │   │ ─ ICP(source=cleaned,    │  │ ─ detect board IDs       │ │
       │   │       target=mesh.ply)   │  │ ─ multi-view calibration │ │
       │   │   recovers Blender Y↔Z   │  │   (cv2.aruco)            │ │
       │   │   flip + drift           │  │                          │ │
       │   │                          │  │ out: camera_intrinsics   │ │
       │   │ out: T_cleaned_to_       │  │      .npz  (K + dist)    │ │
       │   │      original.npy        │  │                          │ │
       │   └──────────────┬───────────┘  └──────────────┬───────────┘ │
       │                  │                             │             │
       │   artifacts:  cleaned.obj, mesh.ply,                         │
       │               T_cleaned_to_original,  K + dist               │
       └──────────────────┬──────────────────────────────┬────────────┘
                          │                              │
                          └──────────────┬───────────────┘
                                         │
                                         │  + motion images (raw)
                                         ▼
       ┌──────────────────────────────────────────────────────────┐
       │  PHASE 2b — Motion from images                           │
       │                                                          │
       │   in: cleaned.obj, K, dist, T_cleaned_to_original,       │
       │       motion/joint_i/state_*.jpg                         │
       │                                                          │
       │   for each joint_i:                                      │
       │     for each adjacent state pair (j, j+1):               │
       │       1. ArUco detect → camera pose C_j, C_{j+1}         │
       │       2. Render mesh silhouette + depth from C_j         │
       │       3. ORB keypoints, restrict to silhouette pixels    │
       │       4. Lowe ratio matching (ratio = 0.95)              │
       │       5. Lift 2D keypoints onto mesh faces (via depth)   │
       │       6. Multi-body RANSAC over 2D-3D pairs              │
       │            → static body + moving body, each with SE(3)  │
       │       7. PnP refinement of moving-body SE(3)             │
       │       8. screw_from_transform → axis, origin, angle      │
       │                                                          │
       │   out: per-joint (axis, origin, angle, moved-vert mask)  │
       └──────────────────────────────────────────────────────────┘
                                    │
                                    ▼
       ┌──────────────────────────────────────────────────────────┐
       │  PHASE 3b — Geometric-prior snap + successive partition  │
       │                                                          │
       │   in: per-joint motion (from 2b)  +  cleaned.obj         │
       │                                                          │
       │   1. Order joints by cut height (base → tip)             │
       │   2. Classify each joint:                                │
       │        is_pitch = |motion · link_dir| < 0.5              │
       │   3. Snap axis to canonical:                             │
       │        roll  → axis := link_dir                          │
       │        pitch → axis := project_to_horizontal(motion)     │
       │   4. Decouple cut_normal from axis                       │
       │        (cut_normal follows link_dir, axis follows motion)│
       │   5. Topology-based moving side:                         │
       │        sign(link_dir · cut_normal) + vertex_mask vote    │
       │   6. Successive mesh partitioning of cleaned.obj         │
       │        loop:  parent_i, child_i = split(mesh_{i-1},      │
       │                                  cut_point, cut_normal)  │
       │                                                          │
       │   out: N+1 link meshes  +  list[JointEstimate]           │
       └──────────────────────────────────────────────────────────┘
                                    │
                                    ▼
                          (shared Phases 4–7)
```

### 1.3  Shared Phases 4–7 (both paths converge here)

```
       Path A or Path B output: list[JointEstimate] + per-link meshes
                                    │
                                    ▼
       ┌──────────────────────────────────────────────────────────┐
       │  PHASE 4 — Template match + physics                      │
       │                                                          │
       │   query Phase 0 KB by (DOF, joint-type sequence,         │
       │                       link-length ratios) → top-1 robot  │
       │   borrow: density, friction, damping, effort, velocity   │
       │   compute: per-link mass = volume × density              │
       │            per-link inertia = trimesh.moment_inertia     │
       └──────────────────────────────────────────────────────────┘
                                    │
                                    ▼
       ┌──────────────────────────────────────────────────────────┐
       │  PHASE 4b — Joint-limit resolver (4-tier)                │
       │                                                          │
       │   1. user_overrides.yaml      (authoritative if present) │
       │   2. template limits          (from Phase 4 match)       │
       │   3. PyBullet self-collision sweep  (safety cap)         │
       │   4. observed range × margin  (fallback)                 │
       │   final = intersect(tier 2, tier 3)                      │
       └──────────────────────────────────────────────────────────┘
                                    │
                                    ▼
       ┌──────────────────────────────────────────────────────────┐
       │  PHASE 5 — Collision meshes                              │
       │                                                          │
       │   for each link mesh:                                    │
       │     CoACD decomposition (~3–4 min/link, ~24 min/arm)     │
       │     fallback: convex hull (~5 min/arm) for fast iter     │
       └──────────────────────────────────────────────────────────┘
                                    │
                                    ▼
       ┌──────────────────────────────────────────────────────────┐
       │  PHASE 6 — URDF assembly + Isaac Sim                     │
       │                                                          │
       │   urdf_assembly.py (Jinja template):                     │
       │     emit chain-ordered <link> + <joint> blocks           │
       │     reference visuals + collisions on disk               │
       │   isaaclab convert_urdf.py  →  robot.usd                 │
       └──────────────────────────────────────────────────────────┘
                                    │
                                    ▼
       ┌──────────────────────────────────────────────────────────┐
       │  PHASE 7 — VLM actor-critic  (optional polish)           │
       │                                                          │
       │   PyBullet rollout vs recorded video                     │
       │   VLM diagnoses mismatches → URDF patch → retry          │
       └──────────────────────────────────────────────────────────┘
                                    │
                                    ▼
                              robot.urdf  +  robot.usd
```

---

## 2. Path A summary

Consumes K MILO meshes (one per robot pose, produced upstream by the MILO pipeline). Vertex correspondence across poses is established through Gaussian-anchor tracking from the MILO outputs. Multi-pose hierarchical RANSAC then partitions vertices into rigid bodies (one per link). Screw decomposition of each body's per-pose SE(3) trajectory yields joint axes, origins, types, and limits.

**Status:** validated end-to-end on synthetic xArm6 — 99.98% vertex assignment, 0.00° axis error, 0.00 mm origin error, robust to ~5 mm Gaussian vertex noise. Never run on real MILO output.

**Why deferred:** the K-pose-per-robot capture protocol is impractical. One MILO scan takes ~30 minutes; K = 13 poses (recommended for a 6-DOF arm) means a half-day per robot. Path B trades segmentation accuracy for a one-scan capture protocol.

---

## 3. Path B — the real-scan implementation, in detail

Path B's bet: **scan once, photograph many.** Replace expensive per-pose meshes with cheap per-joint photo pairs.

### 3.1 Inputs

MILO is run upstream and is **not** part of the mesh2robot pipeline. By the time Path B starts, the MILO mesh already exists. There are **two input categories**:

1. **Mesh** — two related mesh files describing the same robot in two frames.
2. **Motion images** — per-joint state photos. The ArUco board is in *every* frame, so the same images that drive motion recovery also calibrate the camera. No separate calibration set.

```
data/
├── mesh/
│   ├── mesh.ply               ← MILO output: robot + whole environment,
│   │                             aligned to the ArUco board frame
│   └── cleaned.obj            ← robot-only, prepared in Blender (axis-flipped)
└── motion/
    ├── joint_1/
    │   ├── state_0.jpg        ← joint at ~0°    (robot + ArUco board in shot)
    │   ├── state_1.jpg        ← joint at ~+30°  (robot + ArUco board in shot)
    │   └── state_2.jpg        ← joint at ~+60°  (robot + ArUco board in shot)
    ├── joint_2/
    │   └── ...
    └── ...
```

The ArUco board sits in the workspace next to the robot and appears in every motion photo. This single image set serves two purposes simultaneously: per-image camera pose recovery (extrinsics, via `solvePnP` on detected board corners) and intrinsic calibration (via `cv2.aruco.calibrateCameraAruco` over the multi-view set). No dedicated board-only calibration shoot is needed.

### 3.2 Phase 1 — single-mesh prep

| Script | Inputs | Job |
|---|---|---|
| [scripts/calibrate_camera_aruco.py](scripts/calibrate_camera_aruco.py) | motion images (board visible in each) | Detect ArUco board across the multi-view image set; solve for camera intrinsics `K` + distortion coefficients |
| [scripts/register_cleaned_to_original.py](scripts/register_cleaned_to_original.py) | `cleaned.obj` + `mesh.ply` | ICP between the two meshes → `T_cleaned_to_original.npy` (recovers Blender's Y↔Z flip plus any drift from the manual cleanup pass) |

Optional helper (only if cleaning is done programmatically rather than in Blender):
[scripts/clean_mesh.py](scripts/clean_mesh.py) — merge duplicate vertices, drop disconnected components below `--min-faces` to remove background floaters. In the typical workflow, the user prepares `cleaned.obj` manually in Blender (cropping the environment by hand) and `clean_mesh.py` is a fallback.

Output: `cleaned.obj`, `mesh.ply`, `T_cleaned_to_original.npy`, and `camera_intrinsics.npz` — four artifacts that together let Phase 2b lift 2D image features onto 3D mesh faces in a known board-anchored frame.

### 3.3 Phase 2b — motion from images (the core of Path B)

Implemented in [mesh2robot/core/motion_from_images.py](mesh2robot/core/motion_from_images.py) with helpers:
- [feature_matching.py](mesh2robot/core/feature_matching.py) — ORB + Lowe ratio
- [mesh_projection.py](mesh2robot/core/mesh_projection.py) — silhouette/depth render for masking
- [se3_from_2d.py](mesh2robot/core/se3_from_2d.py) — multi-body RANSAC + PnP
- [fiducial/pose.py](mesh2robot/fiducial/pose.py) — ArUco board pose detection

For each joint folder, for each adjacent state pair:

```
state_0.jpg  ────►  ArUco detect  ──►  camera pose C_0
state_1.jpg  ────►  ArUco detect  ──►  camera pose C_1
                            │
                            ▼
                Render mesh silhouette from C_0
                            │
                            ▼
                ORB on state_0 + state_1
                Restrict to silhouette pixels
                Lowe-ratio match (ratio = 0.95)
                            │
                            ▼
                Lift keypoints to mesh faces
                (via depth render at C_0)
                            │
                            ▼
                MULTI-BODY RANSAC over 2D-3D correspondences
                  body_0 = static (camera-side noise + still parts)
                  body_1 = moving (links above the actuated joint)
                            │
                            ▼
                PnP refinement of moving body's SE(3)
                            │
                            ▼
                Screw decomposition  →  axis, origin, angle
                            │
                            ▼
                BodyMotion + PairMotion
```

Output per joint: a `BodyMotion` with
- `T` — 4×4 SE(3) of the moving body
- `vertex_mask` — which mesh vertices are on the moving side
- `face_indices` — feature-bearing faces
- `reprojection_err_px` — sanity metric

### 3.4 Phase 3b — geometric-prior snap and successive partitioning

Implemented in [mesh2robot/core/joint_extraction.py](mesh2robot/core/joint_extraction.py) (real-scan branch).

Three mechanisms not present in Path A:

#### Mechanism 1 — decouple cut plane from rotation axis

Path A conflates the two: the joint's screw axis is used both as the URDF `<axis>` AND as the cut plane normal for splitting the mesh into parent/child. That's wrong for **pitch joints** on industrial arms:

| Joint type | Rotation axis | Cut plane normal |
|---|---|---|
| Roll (axis ∥ link direction) | along link | along link (matches axis) |
| Pitch (axis ⊥ link direction) | horizontal | along link (does *not* match axis) |

For a pitch joint, the rotation axis is horizontal, but the parent/child interface is a horizontal disc with normal pointing along the link. Using the rotation axis as the cut normal produces a diagonal slice through the link — visually wrong.

Path B fixes this by tracking `(cut_point, cut_normal)` separately from `(axis, origin)`.

#### Mechanism 2 — geometric-prior snap

For industrial arms, joint axes are nearly always parallel or perpendicular to the base axis. The recovered axis from Phase 2b might be off by a few degrees due to feature noise. Path B snaps:

```
classification:  is_pitch = |motion · link_dir| < 0.5    # axis ⊥ link?
snap target:
    if roll  : axis ∥ link_dir   → axis := link_dir
    if pitch : axis ⊥ base_axis  → axis := project_to_horizontal(motion)
```

This is an **industrial-arm assumption**. Robots with 45°-axis joints (some humanoids, certain hands) need an opt-out flag.

#### Mechanism 3 — topology-based moving-side determination

After successive cutting, decide which side of the cut belongs to the parent vs child:

```
moved_side = sign(link_direction · cut_normal)
```

Combined with the BodyMotion `vertex_mask` from Phase 2b, this resolves the orientation ambiguity that pure geometry can't.

#### Successive partitioning

The single mesh is cut once per joint, in order from base to tip:

```
mesh_0 = full_mesh
for each joint i in order_by_cut_height:
    cut_point  = origin_i
    cut_normal = pick_canonical(joint_i)
    parent_i, child_i = split_mesh(mesh_i-1, cut_point, cut_normal)
    mesh_i = child_i   # next cut applies to the child
```

Output: N+1 link meshes (base + N moving links).

---

## 4. Shared Phases 4–7 (post-convergence)

Both paths feed the same downstream pipeline. Each subsection below mirrors Section 3's structure: inputs → process → outputs.

### 4.1 Phase 4 — Template match (kinematic-signature retrieval)

Scores every robot in the Phase 0 knowledge base against the recovered kinematic structure and returns the closest match. Used to borrow physics defaults — never kinematics.

**Inputs**
- `joints: list[JointEstimate]` from Phase 3 / 3b (recovered axes, origins, types)
- Per-link mesh AABBs (from Phase 2 / 2b segmentation)
- Optional `hint_name` (override for evaluation)

**Process** ([robot_retrieval.py](mesh2robot/core/robot_retrieval.py), [template_match.py](mesh2robot/core/template_match.py)):

```
1. Build query signature from recovered joints:
     dof              = len(joints)
     axis_pattern     = [snap_to_canonical(j.axis) for j in joints]
     joint_z_fraction = [j.origin[2] / total_height for j in joints]
     workspace_aabb   = bbox over all link meshes

2. For each robot R in database:
     score = w_axis * axis_pattern_match(R, query)
           + w_z    * z_fraction_match(R, query)
           + w_dof  * dof_preference(R, query)        # prefer R.dof >= query.dof
     where:
       axis_pattern_match: 1 - mean(angle_error / 180°)
       z_fraction_match:   1 - mean(|R.zfrac - q.zfrac|)
       dof_preference:     0 if R.dof < q.dof, else 1 - 0.1*|R.dof - q.dof|

3. Sort robots by score, take top K (default K=5)

4. Confidence label from margin between top-1 and top-2:
     margin > 0.15  → "high"
     margin > 0.05  → "medium"
     margin <= 0.05 → "low"

5. If confidence in {high, medium}: return matched template
   If "low" or no candidate above threshold: fall back to category averages
                                              ("6-DOF industrial arm" defaults)
```

**Outputs**
- `Template { robot_id, density, friction, damping, effort_limits[], velocity_limits[] }`
- `confidence_label: str`
- `top_k_alternatives: list[Template]` (for diagnostics)

**Key design point:** kinematics (axes, origins, ranges) are **never** copied from the template. Only physics fields and per-joint actuator limits transfer. This separation keeps the user's actual robot geometry as the ground truth.

---

### 4.2 Phase 4b — Joint-limit resolver (4-tier)

Decides the final `(lower, upper)` for every joint by combining four candidate sources, ordered by trustworthiness.

**Inputs**
- Per-joint candidate ranges from each tier (whichever are available)
- Optional `user_overrides.yaml`
- Recovered URDF skeleton (for collision sweep)

**Process** ([joint_limits.py](mesh2robot/core/joint_limits.py)):

```
For each joint j:
  candidates = []

  ┌─ Tier 1 (highest priority) ─────────────────────────┐
  │ user_overrides.yaml :                                │
  │   joint1: { lower: -3.14, upper: +3.14 }             │
  │ → if present, use directly and SKIP all other tiers  │
  └──────────────────────────────────────────────────────┘
                          │
                          ▼ (if Tier 1 absent)
  ┌─ Tier 2 ─────────────────────────────────────────────┐
  │ Template limits from Phase 4 match                   │
  │ → e.g. UR5e match → joint1 in [-2π, +2π]             │
  └──────────────────────────────────────────────────────┘
                          │
                          ▼
  ┌─ Tier 3 ─────────────────────────────────────────────┐
  │ PyBullet self-collision sweep                        │
  │   1. Load recovered URDF                             │
  │   2. For each j, sweep θ in [Tier-2 lower, upper]    │
  │      in 5° steps                                     │
  │   3. At each θ, check self-collision                 │
  │   4. Trim range to last collision-free angles        │
  │ → safety cap; never exceeds Tier 2                   │
  └──────────────────────────────────────────────────────┘
                          │
                          ▼
  ┌─ Tier 4 (fallback if no template match) ─────────────┐
  │ Observed motion range × safety margin (e.g. 1.2×)    │
  │ → from j.angles in JointEstimate                     │
  └──────────────────────────────────────────────────────┘

Final = intersect(Tier 2, Tier 3) if both present,
        else first available tier.
```

**Outputs**
- `final_limits: dict[joint_name, (lower, upper)]`
- `tier_used: dict[joint_name, str]` (audit trail)

**Why the intersection of Tier 2 and Tier 3:** the template gives the manufacturer's mechanical range; the collision sweep gives the geometric reality of *this particular* visual mesh (which may be slightly different from the official one due to scan errors). The intersection is the safest bound that respects both.

---

### 4.3 Phase 5a — Inertial computation (per-link physics)

Turns mesh + density into URDF-ready inertial blocks.

**Inputs**
- Per-link visual meshes from Phase 2 / 3b
- Density `ρ` per link (from Phase 4 template, default 2700 kg/m³ for unknowns)

**Process** ([physics.py](mesh2robot/core/physics.py)):

```
For each link mesh M:
  if not M.is_watertight:
    M = repair_watertight(M)         # fill holes, reorient normals

  # trimesh's built-in mass properties (assumes uniform density)
  M.density = density

  mass    = M.mass                   # = volume × density
  com     = M.center_mass            # (3,)  in link frame
  inertia = M.moment_inertia         # (3,3) about COM, in link frame

  return LinkInertial(mass, com, inertia)
```

**Outputs**
- `inertials: dict[link_id, LinkInertial]`
- Each `LinkInertial`:
  - `mass: float` (kg)
  - `com: np.ndarray (3,)` (m, in link frame)
  - `inertia: np.ndarray (3,3)` (kg·m², about COM, in link frame)

**Failure modes & mitigations**
- *Non-watertight mesh* — `volume` becomes meaningless. Mitigation: `trimesh.repair.fill_holes` or fall back to convex-hull volume.
- *Wrong density* — error scales linearly with density. Industrial arms ≈ 2700 kg/m³ (Al alloy), 7850 (steel), 1100 (3D-printed PLA). Phase 4's template match picks the right one when it can.
- *Hollow real link, solid scan* — overestimates mass by ~2×. No good fix without per-link density measurement; flagged for VLM critic to detect via gravity-stability simulation.

---

### 4.4 Phase 5b — Collision decomposition (CoACD)

Converts each non-convex visual mesh into a small set of convex hulls suitable for PyBullet/Isaac Sim physics.

**Inputs**
- Per-link visual meshes from Phase 2 / 3b

**Process** ([collision.py](mesh2robot/core/collision.py)):

```
For each link mesh M:
  if coacd_available:
    parts = coacd.run_coacd(
      mesh                  = (M.vertices, M.faces),
      threshold             = 0.05,    # concavity tolerance; ↓ = more parts
      max_convex_hull       = 16,      # cap parts per link
      preprocess_resolution = 50,      # voxel grid for repair
    )
    return [trimesh.Trimesh(vertices=v, faces=f) for v, f in parts]
  else:
    # Fallback: single convex hull (loses concavities but always works)
    return [M.convex_hull]
```

Per-link runtime: ~3–4 minutes on xArm6-scale meshes (~24 min for full arm). Convex-hull fallback: ~1 second per link (~5 min full arm) — used during fast iteration cycles.

**Outputs**
- `collisions: dict[link_id, list[Trimesh]]` (1–16 hulls per link)

**Why decompose at all:** PyBullet/MuJoCo/Isaac Sim collision detection requires convex shapes. Using the raw mesh forces them to do per-triangle queries (slow) or fill concavities (wrong). CoACD's "collision-aware" cost preserves joint housings and gripper finger gaps — see ROADMAP justifications.

---

### 4.5 Phase 6 — URDF assembly + Isaac Sim conversion

Stitches every prior output into a single URDF file plus its supporting mesh assets, then converts to USD for Isaac Sim.

**Inputs**
- `per_link_meshes` (visual)
- `per_link_collisions` (CoACD output)
- `joints: list[JointEstimate]`
- `inertials: dict[link_id, LinkInertial]`
- `template: Template` (for actuator effort/velocity)
- `body_transforms` (Phase 2 pose-0 frames, for `<origin>` math)

**Process** ([urdf_assembly.py](mesh2robot/core/urdf_assembly.py)):

```
1. Reorder bodies in chain order (root → tip) using JointEstimate parent/child
2. Resolve link names (template-matched names, else "link_0", "link_1", ...)
3. Export each visual mesh:
     meshes/linkN.stl              (binary STL, smaller than OBJ)
4. Export each collision sub-mesh:
     meshes/linkN_collision_0.stl
     meshes/linkN_collision_1.stl
     ...
5. Render Jinja2 template:
     <robot name="{name}">
       {% for link in links %}
         <link name="{link.name}">
           <visual>    <mesh filename="meshes/{link.name}.stl"/> </visual>
           {% for ch in link.collision_hulls %}
             <collision> <mesh filename="meshes/{link.name}_collision_{ch}.stl"/>
                         <origin xyz="..."/>
             </collision>
           {% endfor %}
           <inertial>
             <mass value="{link.mass}"/>
             <origin xyz="{link.com}"/>
             <inertia ixx="..." ixy="..." .../>
           </inertial>
         </link>
       {% endfor %}
       {% for joint in joints %}
         <joint name="{joint.name}" type="{joint.type}">
           <parent link="{joint.parent}"/>
           <child  link="{joint.child}"/>
           <origin xyz="{joint.origin}" rpy="..."/>
           <axis   xyz="{joint.axis}"/>
           <limit  lower="..." upper="..." effort="..." velocity="..."/>
         </joint>
       {% endfor %}
     </robot>
6. Validate with yourdfpy.URDF.load(out_path) — fail loudly if XML invalid
7. Optional: invoke Isaac Lab convert_urdf.py → robot.usd
```

**Outputs**
```
out/
├── robot.urdf
├── meshes/
│   ├── link_0.stl                 (visual)
│   ├── link_0_collision_0.stl     (CoACD hulls)
│   ├── link_0_collision_1.stl
│   ├── link_1.stl
│   └── ...
└── robot.usd                      (if Isaac Lab conversion ran)
```

**Validation gate:** the assembler always re-loads its own URDF with yourdfpy before declaring success. This catches malformed `<axis>` (non-unit), unreachable links, or duplicate joint names that would otherwise only surface inside Isaac Sim.

---

### 4.6 Phase 7 — VLM actor-critic (optional polish)

Self-correction loop run when geometric pipeline output behaves wrong in simulation. **Not built** in the legacy pipeline; included here for completeness because it's part of the deferred design.

**Inputs**
- Generated `robot.urdf` from Phase 6
- Reference video of the real robot moving (the same captures used for Phase 2b motion images, but as continuous video)

**Process** (planned):

```
loop iter = 0..MAX_ITERS-1:
  1. Critic stage:
     a. Load URDF in PyBullet
     b. Drive joints through the recorded motion trajectory
     c. Render a video clip from a camera matching the user's
     d. Submit (real_video, sim_video, urdf_summary) to VLM
        (e.g. Gemini 2.5 Flash — see VLM choice discussion)
     e. Receive structured diagnosis:
        { "match": bool,
          "issue": str,
          "field_path": str,        # e.g. "joints[2].axis"
          "current_value": ...,
          "proposed_value": ...,
          "confidence": float }

  2. If diagnosis.match: return current URDF (success)

  3. Actor stage:
     - For deterministic patches (axis sign flip, limit widening):
         apply via regex / structured edit
     - For ambiguous patches (parent re-rooting):
         second VLM call to generate the patch
     - Re-validate URDF with yourdfpy

  4. Re-emit URDF, loop back

return current URDF (max iters reached)
```

**Outputs**
- Patched `robot.urdf` with diagnoses stored in a sidecar `vlm_log.jsonl`

**What it's good for:** axis sign flips (geometry can recover the *line* but not the *direction*), parent/child swaps when stillness heuristic mis-orders, joint-type misclassification, off-by-N joint limits.

**What it can't fix:** bad geometry (MILO holes), missing DOF (Phase 2 merged what should be separate), or order-of-magnitude wrong physics (it sees gross behavior differences but can't dial in friction = 0.43 vs 0.52).

**Cost:** ~$0.10–0.30 per robot using Gemini 2.5 Flash; ~$1 if Pro fallback triggers on hard cases. Self-hosted Qwen2.5-VL-72B is a reproducibility-friendly alternative.

---

### Summary table

| Phase | Component | Primary inputs | Primary outputs |
|---|---|---|---|
| 4 | [robot_retrieval.py](mesh2robot/core/robot_retrieval.py), [template_match.py](mesh2robot/core/template_match.py) | joints, mesh AABBs | `Template` + confidence |
| 4b | [joint_limits.py](mesh2robot/core/joint_limits.py) | template + URDF + YAML | per-joint final limits |
| 5a | [physics.py](mesh2robot/core/physics.py) | per-link meshes + density | per-link `LinkInertial` |
| 5b | [collision.py](mesh2robot/core/collision.py) | per-link meshes | per-link convex hulls |
| 6 | [urdf_assembly.py](mesh2robot/core/urdf_assembly.py) | all of the above | `robot.urdf` + `meshes/` + optional `robot.usd` |
| 7 | not built | URDF + reference video | patched URDF + diagnostic log |

---

## 5. Why Path B was retired (2026-04-25)

Path B works on industrial arms because of the geometric-prior snap. But it has fundamental limits:

1. **Requires per-joint photo capture** — still manual; not zero-effort like the ML approach.
2. **Industrial-arm bias** — fails on humanoids, hands, parallel mechanisms. The snap that makes it work also makes it narrow.
3. **Per-joint successive partitioning** — error accumulates joint-to-joint; a bad cut at joint 2 cascades to joints 3..N.
4. **No physics prediction** — only kinematics; mass/friction/damping all from template-match (Phase 4), which fails for novel robots.
5. **No generalization** — every new robot family needs prior tuning.

The ML pivot replaces all of Phase 2b + 3b with a single learned 3D foundation model that reads a point cloud and predicts segmentation, axes, origins, types, and inertias jointly. Path B remains in the repo as a baseline for the paper and a fallback for cases where the ML model lacks confidence.

---

## 6. One-paragraph summary of Path B

> Capture one MILO mesh of the robot in a single pose, then photograph each joint at two slightly different angles with an ArUco board in frame. ORB features + multi-body RANSAC recover each joint's SE(3) motion in board-anchored coordinates. Screw decomposition gives axis + origin + angle per joint. A geometric-prior snap classifies each joint as roll vs pitch and snaps the axis to the nearest canonical industrial-arm direction. The single mesh is then successively partitioned along per-joint cut planes (whose normals are decoupled from the rotation axes) into per-link sub-meshes. Output feeds the shared Phases 4–7 (template match, collision decomp, URDF assembly).

---

## 7. Decision matrix — when each path applies

| Situation | Recommended path |
|---|---|
| Can capture K multi-view photo sets per pose (K ≥ 3 poses) | A |
| Only one scan is feasible, robot can be manually actuated and photographed | **B** |
| Motion segmentation accuracy is paramount and Gaussian tracking is reliable | A |
| Robot is large / fragile / expensive to scan repeatedly | **B** |
| Want to leverage K full MILO meshes (richer geometry cues) | A |
| Image-based motion is good enough (industrial arm, clear silhouette) | **B** |
| Generalize to novel robots without retuning | Neither — use the ML path (active 2026-04-25+) |

---

## 8. Source files reference

```
mesh2robot/
├── core/
│   ├── motion_segmentation.py   ← Path A Phase 2 (multi-pose RANSAC)
│   ├── joint_extraction.py      ← Phase 3 / 3b (screw axis + geometric-prior snap)
│   ├── motion_from_images.py    ← Path B Phase 2b (image-pair motion)
│   ├── feature_matching.py      ← Path B helper (ORB + Lowe ratio)
│   ├── se3_from_2d.py           ← Path B helper (multi-body RANSAC + PnP)
│   ├── mesh_projection.py       ← Path B helper (silhouette/depth render)
│   ├── rigid_fit.py             ← shared (Horn SE(3) + screw decomposition)
│   ├── robot_retrieval.py       ← Phase 4 (template match)
│   ├── template_match.py        ← Phase 4
│   ├── joint_limits.py          ← Phase 4b
│   ├── physics.py               ← Phase 5
│   ├── collision.py             ← Phase 5
│   └── urdf_assembly.py         ← Phase 6
├── fiducial/
│   └── pose.py                  ← ArUco board detection
└── io/
    └── synthetic_poses.py       ← Path A test harness (synthetic data only)

scripts/
├── clean_mesh.py                ← Path B Phase 1
├── calibrate_camera_aruco.py    ← Path B Phase 1
└── register_cleaned_to_original.py  ← Path B Phase 1
```
