# mesh2robot — Research Roadmap

Real-to-sim robot asset generation pipeline. Takes a 3D scan (or multi-view photos) of a physical robot and outputs a working articulated URDF/USD with collision geometry and physics properties, ready to drop into Isaac Sim.

**As of 2026-04-25 the project pivoted from a heuristic single-plane-cut pipeline to a learned 3D foundation model trained on a multi-source URDF dataset (371 canonical robots after dedup, from 572 trainable / 2020 raw entries).** See [RESEARCH_LOG.md](RESEARCH_LOG.md) for the full timeline. The earlier heuristic phases below are retained for historical context but are no longer the active development path.

**Current state (2026-05-02):** **D.6 shipped.** PT-V3 base (106.5M params, ~3.3× the v2 small baseline) + new `LimitsHead` trained 50 epochs on remote H200 in 7.3 hr — val `seg_acc 53.45 %` (vs v2's 49.5 %), `axis_deg 33.3°` (vs 38.3°), `limits_mae 0.358` (new capability — ~21° per (lower, upper) bound for revolute joints). End-to-end on test_2 verified across three modes (full annotation → 7-link / 5-joint URDF, fresh annotation → 7-link / 4-joint URDF, pure ML zero-touch → 9-link / 7-joint URDF) — all produce realistic learned per-joint priors (e.g. shoulder ±1.71 rad, wrist [-1.63, +2.27]) instead of the old ±π fallback. **All matchmaking / DB-lookup code is purged** (`urdf_db.json`, `template_match.match()`, `robot_retrieval.py`, `urdf_database.py`, the rejected statistical-prior script — net −1836 lines). PyBullet self-collision sweep narrows model priors to the largest collision-free interval; in pure-ML mode it correctly clamped a phantom-link j3 from +2.31 rad to +0.46 rad, preventing a self-colliding URDF from shipping. **Next: optional dataset expansion (`--n-configs 200` → ~115 k examples) to close the 20.7-pt train/val seg gap, then Phase F real-scan benchmark.**

---

## 1. Motivation

Not all robots ship with an official URDF or USD. Custom builds, older industrial arms, and research prototypes often have no public asset. Existing approaches for articulated object reconstruction target household objects (cabinets, appliances) and either discard the user's high-fidelity geometry or cannot handle serial kinematic chains.

mesh2robot fills this gap by combining:

- **MILO** (Mesh-In-the-Loop Gaussian Splatting) for dense surface reconstruction
- **Motion-based part decomposition** — either from K pose meshes (Path A, validated on synthetic xArm6) or from per-joint photo pairs against a single mesh (Path B, implemented on real xArm6 scan)
- **Geometric joint fitting** — screw-axis decomposition + optional industrial-arm prior that snaps axes to canonical directions and decouples cut-plane normals from rotation axes
- **Robotics URDF database** as kinematic / physics prior
- **Isaac Sim URDF importer** for final deployment

---

## 2. Research gaps in prior work

| Gap | PARIS | Real2Code | URDFormer | Articulate-Anything | **mesh2robot** |
|---|---|---|---|---|---|
| Serial N-DOF kinematic chains | Only 2 parts | Tree, not chain | Furniture | PartNet bias | Path A: hierarchical RANSAC over K poses; Path B: per-joint image-pair motion + successive partitioning |
| Preserves user-captured geometry | Partial (NeRF) | No (OBB) | No (retrieval) | No (retrieval) | MILO mesh kept end-to-end |
| Leverages Gaussian-splat anchors | No | No | No | No | MILO Gaussians provide correspondence (Path A) |
| Generates physics properties | No | No | No | No | Mass / inertia from mesh, friction / damping from prior |
| Robotics-specific kinematic prior | No | No | No | No | MuJoCo Menagerie + robot_descriptions DB + geometric-prior snap for axes |
| Auto-segmentation without manual labels | Given as input | 2D SAM only | 2D boxes | Retrieval-masked | Motion clustering from pose set or from per-joint image pairs |
| Joint axis precision | Implicit field | Bounding-box heuristic | VLM-predicted | VLM-predicted | Geometric screw-axis fit, optionally snapped to base-relative canonicals |
| Cut plane independent of joint axis | — | — | — | — | Decoupled (Path B): cut normal follows link direction, axis follows motion |

### Headline contributions

1. **Motion-based N-link segmentation** from Gaussian-splat pose sets.
2. **Full physics URDF** generation (kinematics + inertials + friction/damping), not kinematics alone.
3. **Robot-arm-native priors** via unified robotics URDF database.

---

## 3. End-to-end pipeline

The **active** path as of 2026-04-25 is the learned-model path: a Point Transformer V3 trained on synthetic articulations of ~572 reference robots predicts segmentation, joint structure, and joint limits from any input mesh. PyBullet self-collision sweep narrows learned priors to physically valid ranges. The earlier heuristic paths (A and B in Section 3.7 below) are retained as comparison baselines but are no longer the development frontier.

### 3.1  Phase A — Multi-source URDF dataset (shipped 2026-04-26)

**What it produces.** A canonical, deduped, license-tagged manifest of robot URDFs/MJCFs covering the breadth of the robotics ecosystem.

**Pipeline.**
```
upstream repos (7) ──► raw_robots/   crawl + clone (offline, one-time)
       │                   │ 2,020 raw URDF/MJCF/SDF entries
       ▼                   ▼
build_robot_manifest.py ──► robot_manifest.json
       │                       │ 572 trainable (loaded OK + actuated joints + ≥80% meshes resolve)
       ▼                       ▼
build_research_manifest.py ──► robot_manifest_research.json
       │                              │ 371 canonical (union-find dedup)
       │                              │ + license per directory
       │                              │ + quality_tier (curated CAD vs xacro vs demo)
       │                              │ + scale_class (compact/tabletop/fullsize/huge/unit_bug)
       │                              │ + AABB / mesh_bytes / joint_range_total
       ▼                              ▼
summarize_manifest.py ──► canonical_robots.{md,csv}
                          (paper-appendix-ready table of every canonical robot)
```

**Sources crawled.** MuJoCo Menagerie, bullet3, urdf_files_dataset, robot-assets, robosuite, Gymnasium-Robotics, ROS-Industrial. Includes xArm6 (3 variants in bullet3), xArm7 (4 in Menagerie/robosuite), Panda, UR5/10/e, IIWA, Sawyer, Yumi, Spot, Atlas, Unitree H1/G1/Go2, dexterous hands (LEAP, Adroit, ShadowHand), grippers (Robotiq, UMI), quadrupeds (Anymal, Barkour), humanoids (Talos, Tiago).

**Dedup signatures (3-key union-find).** Same robot from different sources collapses to one canonical instance. (a) `(family, leaf-filename, dof)` matches identically-named Pandas across Menagerie + robot-assets + robosuite. (b) `(normalized-path-tail, dof)` matches re-bundled paths even when family inference fell back to source name. (c) `(menagerie-dir, dof)` collapses Menagerie's `<robot>.xml` + `scene.xml` + `<robot>_mjx.xml` variants — they're the same physical robot.

**Quality guardrails.**
- `unit_bug` flag for any URDF with link-origin AABB > 50 m (catches mm-encoded chains). **Zero unit-bug entries in the canonical set.**
- License-per-directory parsing of Menagerie's combined LICENSE file → per-robot Apache-2.0 / BSD-3 / CC-BY / MIT attribution.
- `scale_class` (compact / tabletop / fullsize / huge) prevents matching a 6-DOF gripper finger to a 6-DOF arm.

---

### 3.2  Phase B — Synthetic data generator (shipped 2026-04-26; v4 expanding)

**What it produces.** Training shards: (point_cloud, per-point link labels, per-joint targets including learned `(lower, upper)` limits) tuples, sampled from the canonical robots at random poses.

**Per-example pipeline (one robot, one config, one shard entry).**
```
LoadedRobot                              random joint config
(URDF or MJCF)                           uniform within (lower, upper)
       │                                          │
       └──────────► articulate_and_label ◄────────┘
                    (urdf_loader.py / mjcf_loader.py)
                              │
                              ▼
       ┌──── combined trimesh (world frame, all link visuals concatenated)
       ├──── vertex_labels[V]               per-vertex link index
       ├──── joint_axes_world[J, 3]          unit vector at this articulation
       ├──── joint_origins_world[J, 3]       pivot in world frame
       ├──── joint_types[J]                  int (revolute=0, continuous=1, prismatic=2, fixed=3)
       ├──── joint_topology[J, 2]            (parent_link_idx, child_link_idx)
       └──── joint_limits[J, 2]              (lower, upper) — from URDF source [v3+]
                              │
                              ▼
                     sample_point_cloud
                     uniform-area, 16,384 points
                              │
                              ▼
                       apply_augment
       Gaussian vertex noise σ ∈ [0.5, 5] mm     (sensor noise)
       Cluster hole-punching 5-15% drop          (MILO occlusion)
       Random rigid transform + scale ±20%       (camera-frame variation)
                              │
                              ▼
                  pack into shard (.npz)
```

**Why "synthetic" but the robots are real.** The robots' geometry, joint structure, and limits come from real manufacturer URDFs in the manifest. The *poses* are random, the *point clouds* are sampled (not LiDAR'd), the *noise* is simulated. The model learns to see real MILO scans because the synthetic noise distribution matches MILO's typical artefacts.

**v3 dataset (used for the shipped checkpoint):**
- 50 random configs × 572 robots = **28,400 examples** (URDF: 17,450 / MJCF: 10,950)
- 5.0 GB on disk (compressed npz shards, 32 examples per shard)
- Includes `joint_limits` field (added 2026-05-02 for the LimitsHead training)

**v4 dataset (in progress):** 200 configs × 572 robots ≈ **115k examples** (~20 GB). Goal: close the train/val seg gap (20.7 pts at v3 ep50 → target ≤12 pts).

**The "~1M target" in the original spec.** Original arithmetic was `925 robots × 200 configs × 5 augmentations ≈ 925k`. Two reductions vs that target: (a) post-dedup, 925 became 572 unique canonical robots; (b) augmentations are applied *inline* (one per example), not as multiplicative expansion. v4 brings us to 115k — about 12% of the original ambition, but enough for a 106M-param model.

---

### 3.3  Phase C — Foundation model training (v3 shipped 2026-05-02)

**Architecture.** Point Transformer V3 encoder + multi-task heads predicting per-vertex segmentation and per-joint properties. Built on the [Pointcept](https://github.com/Pointcept/Pointcept) PT-V3 backbone, vendored into `mesh2robot/model/ptv3/`.

**Two encoder sizes (`--encoder-size`):**

| Size | Params | enc_channels | enc_depths | Trained on | VRAM at training |
|---|---:|---|---|---|---|
| `small` | 31.8 M | (32, 64, 128, 256, 384) | (2,2,2,4,2) | local 3090 | ~10 GB at batch 32 |
| `base` (current) | 106.5 M | (64, 128, 256, 384, 512) | (2,2,4,6,2) | remote H200 | ~122 GB at batch 24 |

**Heads** (all in `mesh2robot/model/model.py`):

```
PT-V3 encoder ──► per_point[B, N, F]
                  global_feat[B, F]
                            │
            ┌───────────────┼─────────────────┐
            ▼               ▼                 ▼
       SegmentationHead  JointHead         (mass/inertia head — future work)
       per_point + g     g + slot_emb[J_MAX]
            │               │
            ▼               ├──► axis[B, J, 3]         unit vec, sign-invariant CE-cos loss
       seg_logits           ├──► origin[B, J, 3]       smooth-L1 in metres
       [B, N, K=64]         ├──► type_logits[B, J, 6]  CE over 6 joint types
                            ├──► valid_logit[B, J]     BCE — does this slot exist?
                            └──► limits[B, J, 2]       smooth-L1 on (lower, upper) [v3+]
```

**Loss weights** (`LossWeights` in `losses.py`): `seg=1.0, axis=1.0, origin=1.0, type=0.5, valid=0.5, limits=0.5`. Limits loss is masked by `joint_valid AND has_limits` so legacy v1/v2 shards (no limits) safely contribute nothing.

**Training run history.**

| Run | Encoder | Dataset | Hardware | Wall | Final val |
|---|---|---|---|---|---|
| v1 (PointNet baseline) | 0.5 M | v0 shards | local 3090 | ~5 min/50 ep | superseded |
| v2 (2026-04-28) | small (31.8 M) | v1 shards (no limits) | local 3090 | ~30 min / 25 ep | seg 49.5%, axis 38.3°, valid 96.9% |
| **v3 (2026-05-02)** | **base (106.5 M)** | **v3 shards (28,400 with limits)** | **remote H200** | **7.3 hr / 50 ep** | **seg 53.45%, axis 33.3°, valid 95.7%, limits_mae 0.358** |
| v4 (in progress) | base (106.5 M) | v4 shards (~115k with limits) | remote H200 | ~28 hr est. / 50 ep | TBD |

**Future heads** (not yet implemented): per-link mass / inertia regressor, friction / damping regressor. These are non-learnable from a static mesh signal alone (need system identification), so they currently come from fixed conservative defaults in `physics_defaults.py`.

---

### 3.4  Phase D — Inference pipeline (D.1, D.4, D.6 shipped; D.2 abandoned; D.3 wired-but-deferred; D.5 superseded)

**Master inference flow** (one invocation of `scripts/predict_urdf_interactive.py`):

```
INPUTS
  --mesh                     MILO scan (.obj / .ply)
  --mesh-to-world            optional (4×4) transform .npy
  --checkpoint               trained PT-V3 .pt
  --user-annotations         optional pre-saved annotations.json
  --motion-dir               optional per-joint photo pairs (D.3, off by default)
  --geometric-joints         flag — D.4 path on/off
  --collision-sweep          flag — D.6 sweep on/off

(0) MESH PREP
    load OBJ ──► trimesh
    apply T_cleaned_to_original (puts mesh in MILO's world frame)
    sample 16,384 points (uniform area)
    normalise to unit ball (centroid-shift, p99-radius scale)

(1) USER ANNOTATION (Stage 1, default ON)
    if --user-annotations <path>:
        load saved annotations.json (face_idx → link_id)
    elif not --no-gui:
        open PyVista viewer
        TAB cycles modes:  Rect → Lasso → Camera
        digit 0-9 labels current selection as link N
        every digit press persists to user_annotations.json
        SPACE commits → ML runs
    else:
        skip — pure ML mode

(2) ML INFERENCE (Stage 2, always)
    PT-V3 forward pass (single example, batch=1)
      ── pred_seg[N]              per-point cluster id
      ── pred_axes[J_MAX, 3]      world-frame unit vectors
      ── pred_origins[J_MAX, 3]   world-frame pivots
      ── pred_types[J_MAX]        joint type ints
      ── pred_valid[J_MAX]        sigmoid > 0.5 → joint exists
      ── pred_limits[J_MAX, 2]    (lower, upper) — only if checkpoint has LimitsHead

    project_labels_to_faces (k-NN from sampled points to mesh faces)
      ── face_labels[F]            per-face cluster id

(3) STRICT-MODE MERGE (when user annotated)
    For each ML cluster:
      if user covered ≥30% of its faces → entire cluster takes user label
                                          (handles user "rough hint")
      else                              → user faces become hard override,
                                          remaining faces get fresh label
    Output: refined_face_labels[F] — every face has a user-or-ML label

(4) MOTION-IMAGE REFINEMENT (Stage 3, opt-in via --motion-dir)
    Per-joint photo pairs run through legacy Path-B image RANSAC + PnP
    Outputs (axis, origin) overrides keyed by chain index.
    Currently bottlenecked on test_2 by rough calibration.json — deferred.

(5) URDF ASSEMBLY (Stage 4, always)
    split_mesh_by_face_labels                     → per-link visual meshes
    sort link_ids by Z-centroid                    → chain order
    for each adjacent (parent, child) in chain:
        if --geometric-joints AND user-annotated:
            ── walk face_adjacency, find boundary edges
            ── fit 3D circle to boundary verts (PCA plane + algebraic 2D fit)
            ── axis = circle normal, origin = circle center
            ── type = revolute if circularity > 0.55 else fixed
            ── (D.4 — geometric joints, source of truth)
        elif joint in motion_overrides:
            use motion-image (axis, origin)        (D.3, where it succeeded)
        else:
            use ML's predicted slot                (fallback)
        build JointEstimate(parent, child, axis, origin, type)
        attach (lower, upper) prior:
            if model has LimitsHead:               ── slot_for_limits = matched
                use pred_limits[slot_for_limits]
            else:
                use ±π fallback

    template = make_default_template(dof)         (fixed physics: density 2700 kg/m³,
                                                    friction 0.5, damping 0.1, ...)
    inertials = compute_link_inertials(per_link_meshes, density)

    ── write URDF v1 to refined/robot.urdf

(6) COLLISION SWEEP (Stage 5, opt-in via --collision-sweep) — D.6
    PyBullet load URDF v1 with USE_SELF_COLLISION
    Disable adjacent (parent, grandparent) collision pairs
        (those contacts are expected for revolute joints)
    For each actuated joint:
        sweep 0 → upper_prior in N steps; first self-collision step → upper bound
        sweep 0 → lower_prior in N steps; first self-collision step → lower bound
        result = (lower_safe, upper_safe) ⊆ (lower_prior, upper_prior)
    Re-assemble URDF v2 with refined limits → refined/robot.urdf  (overwrite)

(7) COMPARISON GLB EXPORT
    user_annotation.glb          rough user lasso coverage, untouched faces grey
    refined_assembled.glb        refined URDF with FK applied, link-coloured
    (both in world frame so they line up spatially in any GLB viewer)
```

**File outputs.** Every successful run produces:
```
output/<name>/
├── original/robot.urdf            pure-ML URDF (no user input applied)
├── refined/robot.urdf             user + geometric + sweep refined URDF
├── user_annotation.glb            rough annotation visualisation
└── refined_assembled.glb          final URDF with FK applied
```

**D.2 retrieval — abandoned.** Originally planned: PT-V3 global features + cosine sim against 371 canonical embeddings, return manufacturer's URDF on high similarity. Built and tested 2026-05-01: ranked KUKA IIWA above the actual xArm6 input on test_2 (sim 0.940 vs 0.78 for the correct match). PT-V3's pooled global features are pose-sensitive and biased toward over-represented training robots. Removed; `mesh2robot/core/robot_retrieval.py` deleted in the 2026-05-02 cleanup.

**D.5 dispatch wiring — superseded.** Original plan: retrieval first, geometric extraction as fallback. Since D.2 was abandoned, the dispatch became the in-line precedence chain inside `build_urdf_from_predictions`: **geometric > motion-image > ML**, with model-predicted limits feeding all three.

---

### 3.5  Phase E — VLM refinement (deferred)

Render mesh from 4 canonical angles, ask GPT-4V / Claude / Gemini whether the predicted segmentation + joint axes are visibly consistent. Apply suggested corrections to the URDF. Not yet implemented; the strict-mode merge + collision sweep already catch most of what a VLM would catch.

---

### 3.6  Phase F — Real-scan benchmark (deferred)

Plan: MILO scans of 3-5 different robots beyond test_2 (UR5e, Kinova Gen3, a custom DIY arm without URDF). Per-vertex segmentation IoU, per-joint axis/origin error vs official URDF where available. Comparison to Articulate-Anything baseline on the same inputs.

---

### 3.7  Legacy heuristic paths (retained for historical context)

Two earlier paths converge at Phase 4. **Path B (single-mesh + per-joint photos)** is the heuristic real-scan path implemented through 2026-04-24; **Path A (K pose meshes)** is the original roadmap and is validated end-to-end only on synthetic pose meshes. Both are now superseded by the learned-model path in Sections 3.1–3.4.

```
=================================================================
PATH A — K pose meshes                 PATH B — single mesh + photos
  (validated on synthetic xArm6)         (implemented on real xArm6 scan)
=================================================================

INPUTS                                 INPUTS
 - K multi-view photo sets             - 1 MILO scan of the robot in any pose
   (one per robot pose, K >= 3)        - Per-joint folders motion/joint_i/
 - Video for limit refinement            with K_i >= 2 state photos
 - Optional: robot type hint           - ArUco GridBoard calibration images
                                       - Optional: robot type hint

PHASE 1  MULTI-POSE GEOMETRY           PHASE 1  SINGLE-MESH PREP
 - Run MILO per pose                    - Run MILO once on the scan set
   -> meshes M_1..M_K + Gaussians       - clean_mesh.py: merge dupes,
 - Register into common world frame       drop components < N faces
                                        - calibrate_camera_aruco.py:
                                          K + dist from GridBoard
                                        - register_cleaned_to_original.py:
                                          ICP -> T_cleaned_to_original
                                        (mesh + board frame aligned)

PHASE 2  MOTION SEGMENTATION (novel)   PHASE 2b  MOTION FROM IMAGES
 - Cross-pose Gaussian tracking         - For each joint i, for each pair
   -> per-vertex trajectory               of state photos:
 - Hierarchical RANSAC on SE(3)           - ArUco -> per-image camera pose
   -> peel one link per level             - ORB + silhouette mask
 - Body-merge + orphan reassign           - Lowe's ratio matching
 - Output: per-link meshes +              - Multi-body RANSAC -> SE(3)
   transforms T_l(pose)                   - PnP refinement of final T
                                        - Screw decomposition -> axis +
                                          origin + angle per joint
                                        (no per-pose mesh needed; motion
                                         is measured purely from images)

PHASE 3  JOINT EXTRACTION              PHASE 3b  GEOMETRIC-PRIOR SNAP
 - Axis = screw axis of SE(3)           - Order joints by cut height
 - Parent-child from stillness          - Classify roll vs pitch from
   correlation                            |motion . link_dir|
 - Limits from pose range               - Snap axis to nearest canonical
                                          (parallel / perpendicular to
                                          base axis)
                                        - DECOUPLE cut_normal from axis
                                          (previously conflated)
                                        - Topology-based moving-side:
                                          sign(link_dir . cut_normal)
                                        - Successive mesh partitioning
                                          by (cut_point, cut_normal)
                                          -> per-link face sets

---------------------- converge -------------------------

PHASE 4  TEMPLATE MATCH + PHYSICS
 - Query (DOF, joint-type sequence) -> nearest URDF in DB
 - Borrow density, friction, damping, effort, velocity
 - trimesh volume * density -> mass / COM / inertia per link

PHASE 4b  JOINT-LIMIT RESOLUTION (tiered)
 - User YAML override (Tier 1, authoritative)
 - Template limits from DB match (Tier 2)
 - PyBullet self-collision sweep (Tier 3, safety cap)
 - Observed-range x margin fallback (Tier 4)
 - Intersect Tier 2 with Tier 3 for the final bound

PHASE 5  COLLISION MESHES
 - CoACD per link (offline; ~24 min for xArm6)
 - Convex-hull fallback (~5 min for xArm6) for fast iteration

PHASE 6  URDF ASSEMBLY + Isaac Sim
 - Jinja template -> robot.urdf (yourdfpy-valid)
 - Chain-ordered link/joint emission
 - Isaac Lab convert_urdf.py -> robot.usd

PHASE 7  VLM ACTOR-CRITIC (optional polish)
 - PyBullet rollout vs recorded video
 - VLM flags mismatched joints -> retarget axes / limits
```

**When to use which path:**

| Situation | Path |
|---|---|
| Can capture K multi-view photo sets per pose (K >= 3 poses) | A |
| Only one scan is feasible, robot can be manually actuated and photographed | **B** |
| Motion segmentation accuracy is paramount and Gaussian tracking is reliable | A |
| Robot is large / fragile / expensive to scan repeatedly | **B** |
| Want to leverage K full MILO meshes (richer geometry cues) | A |
| Image-based motion is good enough (industrial arm, clear silhouette) | **B** |

Both paths produce the same Phase 3 interface (per-joint axis, origin, angle, moved-vertex mask), so Phases 4–7 are shared.

---

## 4. Phased prototype plan (8-week build) — *original schedule, kept for historical context*

> The actual project trajectory diverged from this plan in two places. (1) After Phase 2 was validated in week 3 we pivoted to the **learned-model path** (Sections 3.1–3.4), which replaced Phases 0/4 with a much larger 371-canonical-robot dataset and replaced the per-pose RANSAC of Phase 2 with PT-V3 training. (2) Phases 3b/4b (geometric-prior snap + tiered limit resolver) were superseded by the trained `LimitsHead` + PyBullet collision sweep in D.6. The week-by-week deliverables below describe the original heuristic build and remain useful for understanding Path A / Path B in Section 3.7.

### Phase 0 — Foundation (Week 1)

**Deliverables**
- Unified robot URDF database with schema `{robot_id, dof, joint_types[], limits[], link_inertials, friction, damping}` across:
  - [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie)
  - [robot_descriptions.py](https://github.com/robot-descriptions/robot_descriptions.py)
  - [urdf_files_dataset](https://github.com/Daniella1/urdf_files_dataset)
- Capture protocol document covering both paths:
  - Path A: K multi-view photo sets (one per robot pose, K ≥ 3) with shared world anchors.
  - Path B: one multi-view scan set + per-joint state photos at K_i ≥ 2 joint angles + ArUco GridBoard reference.

**Success criterion:** 40+ robots ingested, queryable by DOF + joint-type signature.

---

### Phase 1 — Geometry capture & registration (Week 2)

**Deliverables (Path A — K pose meshes)**
- Wrapper script: K poses in → K meshes + Gaussians out, all in common coordinate frame.
- Fixed-base registration via ICP on the base link or an external fiducial marker.

**Deliverables (Path B — single mesh, implemented 2026-04-22 .. 2026-04-24)**
- `scripts/clean_mesh.py` — merge duplicate vertices, drop components below `--min-faces` to remove MILO floaters.
- `scripts/calibrate_camera_aruco.py` — camera intrinsics + distortion from an ArUco GridBoard.
- `scripts/register_cleaned_to_original.py` — ICP producing `T_cleaned_to_original.npy` that maps the cleaned-OBJ frame back into the original MILO/PLY frame (fixes Blender's Y↔Z roundtrip flip).

**Risks and mitigations**
- Path A: MILO may not expose Gaussians cleanly for downstream tracking. Fallback: mesh–mesh correspondence via feature matching (FPFH + ICP).
- Path B: image feature matching may miss small rotations (< 5°). Mitigation: capture K_i ≥ 3 states per joint; use Lowe's ratio `ratio=0.95` rather than default; fall back to biggest-gap cut location when motion is weak.

---

### Phase 2 — Motion recovery (Weeks 3–4) — novel research

**Path A — Motion-based mesh segmentation**
- `segment.py`: input K pose meshes → N link meshes + per-link SE(3) trajectories.
- Algorithm: hierarchical LO-RANSAC on vertex trajectories, body-merge, orphan reassignment.
- Evaluation: xArm6 with known ground truth (swap official meshes per pose to simulate MILO output).
- Success criterion: ≥ 95 % vertices correctly assigned on xArm6 reference geometry; ≥ 85 % on an actual MILO scan.

**Path B — Motion from image pairs**
- `mesh2robot/core/motion_from_images.py`: input = one MILO mesh + per-joint state photos → per-joint `(axis, origin, angle, moved_vertex_mask)` tuples.
- Algorithm: ORB + silhouette-mask-guided detection → Lowe's-ratio matching → multi-body RANSAC on 2D-3D correspondences → PnP refinement of the winning SE(3).
- Per-joint successive partitioning of the single mesh (Phase 3b) replaces the per-pose segmentation that Path A does.

Path A's motion segmentation is the **core research contribution**; Path B is an engineering alternative that trades segmentation accuracy for single-capture convenience. Both feed the same Phase 3 interface.

---

### Phase 3 — Joint extraction (Week 5)

**Deliverables**
- `extract_joints.py`: from per-link trajectories, compute screw axis, joint type, origin, limits.
- Uses exponential coordinates / Lie algebra (`log(SE(3))`) — standard formulation.

**Success criterion:** axis direction error < 2°, origin error < 5 mm vs official xArm6 URDF.

**Design refinement (2026-04-24):** for the real-scan path, the pipeline now **decouples two concepts that were previously conflated**:
- **Joint rotation axis** — the physical direction the joint rotates around (used in URDF `<axis>`).
- **Cut plane normal** — the direction perpendicular to which the mesh is partitioned into parent/child links (purely a segmentation question).

For roll joints (axis ∥ link direction) these coincide; for pitch joints on upright arms they do not (axis is horizontal but cut plane should be horizontal too — i.e., normal = base axis). A **geometric-prior snap** classifies each joint as roll vs pitch from `|motion · link_dir|`, then snaps both axes to canonical directions (parallel or perpendicular to the base). This eliminates the "diagonal cut" failure mode on industrial arms but is an industrial-arm assumption — opt-out needed for 45°-axis robots.

---

### Phase 4 — Template matching + physics (Week 6)

**Deliverables**
- `match_template.py`: query Phase 0 database with (DOF, joint-type sequence) → top-3 matches.
- `compute_physics.py`: trimesh inertials + template friction/damping.

**Success criterion:** retrieved template's physics defaults produce stable sim (no explosion under gravity, drive commands track).

---

### Phase 5 — URDF assembly + Isaac Sim (Week 7)

**Deliverables**
- `assemble_urdf.py`: Jinja template filled from Phases 2–4.
- Isaac Lab import script with correct self-collision and drive defaults.
- End-to-end test: raw photos → working robot in Isaac Sim.

**Success criterion:** imported robot holds pose under gravity, each joint drives correctly to commanded angles, TCP position within 2 cm of physical robot at matching joint angles.

---

### Phase 6 — Evaluation + paper draft (Week 8)

**Deliverables**
- Benchmark on at least 3 arms: xArm6 (reference), UR5e, and one unusual arm (Kinova Gen3 or a custom DIY arm without URDF).
- Metrics:
  - Joint axis error
  - Joint limit error
  - TCP tracking error
  - VLM success rate (if Phase 7 included)
- Comparison table vs Articulate-Anything's output on the same inputs (feed it the same video for fairness).

**Success criterion:** quantitative wins on all four gaps from the research-gaps table.

---

### Phase 7 — VLM critic (optional, Weeks 9–10)

Adds self-correction loop for edge cases (non-standard joints, parallel linkages). Skip if Phases 2 and 3 already hit target accuracy.

---

## 5. Critical path vs nice-to-have

| Component | Critical? | Rationale |
|---|---|---|
| MILO geometry (Phase 1, any path) | Yes | Source of visual + collision mesh; downstream is useless without it |
| Motion recovery (Phase 2, either path) | Yes | Supplies per-joint SE(3); no segmentation / joint extraction without it |
| Joint extraction (Phase 3 / 3b) | Yes | Cheap once Phase 2 is done; geometric-prior snap on Path B also runs here |
| Joint-limit resolver (Phase 4b) | Yes | Bespoke robots need some limit source; template + collision sweep + YAML covers it |
| URDF DB + template match (Phases 0, 4) | Yes for physics | MVP can hardcode friction = 0.5, density = 2700 kg/m³ |
| Robot type classifier (earlier design) | Dropped | Motion segmentation + template match replaces it |
| Video-based limits (earlier design) | Downgraded | Phase 2 pose range + collision sweep do the same job |
| VLM critic (Phase 7) | No | Nice-to-have for publication, not needed for working system |

Motion recovery **eliminates** the separate robot type classifier and the video limit extraction from the earlier plan — multi-pose capture (or per-joint photos on Path B) + template match + collision sweep cover all three jobs. Net simplification.

---

## 6. Dependency graph

```
novel research:       Phase 2 (motion seg)  -+
                      Phase 3 (screw axes)  -+--> publishable contribution
comparison baseline:  re-run A-A on inputs  -+

engineering glue:     Phases 0, 1, 4, 5, 6 - reliable but not novel
```

If time-limited, cut Phase 7 first, then Phase 4 (hardcoded physics), then Phase 0 (single reference URDF). The research contribution remains in Phases 2–3 regardless.

---

## 7. Critical feasibility experiment (completed)

Original plan (spend 2 days before committing 8 weeks):

1. Take the official xArm6 URDF.
2. Generate K = 5 pose meshes in Isaac Sim (synthetic, noise-free).
3. Run Phase 2 + Phase 3 code on these clean meshes.
4. Compare recovered joint axes to ground truth.

**Result (2026-04-22):** 99.98 % vertex assignment, 0.00° axis error, 0.00 mm origin error on synthetic xArm6. See RESEARCH_LOG.md. Extended noise sweep: < 1° axis error up to σ = 5 mm vertex noise.

**Follow-up on real scan (2026-04-24):** switched to Path B (single MILO scan + per-joint photos) for the actual robot. Produces a 5-link / 4-joint URDF that loads in yourdfpy; axes snap to canonical directions via the geometric-prior snap. No ground-truth URDF at the same pose, so per-joint axis error vs GT is not yet quantified.

---

## 8. Key references

### Prior work (articulated object reconstruction)
- PARIS — [project](https://3dlg-hcvc.github.io/paris/) · [arXiv](https://arxiv.org/abs/2308.07391)
- Real2Code — [project](https://real2code.github.io/) · [arXiv](https://arxiv.org/abs/2406.08474)
- URDFormer — [project](https://urdformer.github.io/) · [arXiv](https://arxiv.org/html/2405.11656v1)
- Articulate-Anything — [project](https://articulate-anything.github.io/) · [arXiv](https://arxiv.org/abs/2410.13882)

### Robot URDF datasets
- [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie)
- [robot_descriptions.py](https://github.com/robot-descriptions/robot_descriptions.py)
- [urdf_files_dataset (Daniella1)](https://github.com/Daniella1/urdf_files_dataset)
- [robot-assets (ankurhanda)](https://github.com/ankurhanda/robot-assets)
- [ROS-Industrial](https://github.com/ros-industrial)
- [Understanding URDF: A Dataset and Analysis (arXiv)](https://arxiv.org/abs/2308.00514)

### Supporting tools
- MILO — Mesh-In-the-Loop Gaussian Splatting
- [CoACD](https://github.com/SarahWeiii/CoACD) — Collision-Aware Convex Decomposition
- [trimesh](https://github.com/mikedh/trimesh) — mesh properties + inertial computation
- Isaac Sim URDF Importer
- Isaac Lab `convert_urdf.py`
