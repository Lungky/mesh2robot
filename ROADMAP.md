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

The **active** path as of 2026-04-25 is the learned-model path: a foundation model trained on ~925 reference robot URDFs predicts segmentation + joint structure from any input mesh. The earlier heuristic paths (A and B below) are retained as comparison baselines.

```
═══════════════════════════════════════════════════════════════
  ACTIVE PATH (2026-04-25+) — LEARNED 3D FOUNDATION MODEL
═══════════════════════════════════════════════════════════════

INPUTS
  - User MILO scan (any robot — arm, gripper, humanoid, quadruped, ...)
  - Optional: per-joint motion observations (for axis disambiguation)
  - Optional: VLM refinement queries

PHASE A  MULTI-SOURCE URDF DATASET (done)
  - Crawl Menagerie + bullet3 + urdf_files_dataset + robot-assets +
    robosuite + Gymnasium-Robotics + ROS-Industrial repos
  - 2020 raw URDF/MJCF/SDF entries
  - 572 trainable (loaded ok + actuators + meshes resolve)
  - 371 canonical after union-find dedup over (family, leaf, dof) +
    (path-tail, dof) + Menagerie-dir signatures
  - Per-canonical metadata: license, fidelity_class, scale_class
    (link-origin AABB-derived), joint_range_total, mesh_bytes_total,
    aabb_extent_m
  - Zero unit-bug URDFs (mm-encoded chain >50 m) in the canonical set
  - Manifest at data/robot_manifest_research.json
  - Includes xArm6 (3 variants in bullet3), xArm7 (4 variants in
    Menagerie/robosuite), Panda, UR5/10/e, IIWA, Sawyer, Yumi, Spot,
    Atlas, Unitree H1/G1/Go2, etc.

PHASE B  SYNTHETIC DATA GENERATOR (training-time)
  for each trainable URDF in manifest:
    for each random joint config (within limits, self-collision filtered):
      FK + concat per-link meshes -> combined mesh + per-vertex link labels
      Extract joint axes/origins/types in world frame
      Sample 16k surface point cloud
      Apply augmentations:
        Gaussian vertex noise sigma in [0.5, 5] mm
        Cluster hole-punching (5-15% drop, simulating MILO occlusion)
        Random rigid transform + scale +/- 20%
      Save (point_cloud, vertex_labels, joint_axes, joint_origins, joint_types)
  Target: ~1M training examples

PHASE C  FOUNDATION MODEL TRAINING
  Backbone: Point Transformer V3 (~500M-1B params)
  Pretraining: masked point modeling on full dataset (SSL)
  Fine-tuning heads (multi-task):
    1. robot type classifier
    2. articulation graph predictor (link/joint topology)
    3. per-vertex link segmentation
    4. per-joint axis + origin regression
    5. per-joint type classifier
    6. per-link mass / inertia regressor
  Hardware: NVIDIA RTX 3090 (24 GB VRAM, local workstation —
            i9-11900KF, 64 GB RAM)
            Sufficient for PointNet baseline (~minutes per 50 epochs)
            and PT-V3 medium ~100M params (~30 min per 50 epochs);
            larger models (~500M+) are tractable in ~1-2 hr per run.
  Output: trained checkpoint

PHASE D  PIPELINE INTEGRATION
  D.1 (done) End-to-end inference path:
    user mesh -> 16k point cloud -> PT-V3 -> 6 head outputs
    -> Phase 5 URDF assembly
    Implemented in scripts/predict_urdf.py and the interactive
    semi-auto variant scripts/predict_urdf_interactive.py
    (pre-select GUI: Rect/Lasso/Camera Tab cycle, persisted
    annotations, strict-mode merge with 30%-coverage propagation,
    in-process ML+URDF after SPACE; auto-emits side-by-side
    comparison GLBs at output root: user_annotation.glb (rough
    user coverage) + refined_assembled.glb (final URDF with FK))

  D.2 (next) RETRIEVAL FOR KNOWN ROBOTS:
    Pre-compute PT-V3 global features for each of the 371 canonical
    robots once -> data/canonical_embeddings.npy.
    At inference: compute input embedding, cosine-similarity vs
    canonicals; if max similarity > tau (default 0.85):
      pose-align canonical URDF to input via Procrustes / ICP
      emit canonical URDF (exact joints, exact masses)
    Else: fall through to D.3 / D.4 below.
    Effect: for known robots the joint axis error drops to ~0
    (uses manufacturer's URDF directly).

  D.3 (already wired) MOTION-IMAGE REFINEMENT (Stage 3, opt-in):
    --motion-dir flag points at motion/joint_<N>/state*.png|jpg
    Per-joint image-pair RANSAC + PnP via legacy Path-B code
    Sub-degree axis precision when calibration is good.
    Currently failing on test_2 because calibration.json is a
    rough guess; deferred until proper checkerboard calibration.

  D.4 (next) GEOMETRIC JOINT EXTRACTION (universal fallback):
    For each adjacent (link_i, link_{i+1}) pair:
      Find boundary loop via face-adjacency in the original mesh
      Fit a 3D circle (SVD plane fit + algebraic 2D fit)
      circle_normal -> joint axis
      circle_center -> joint origin
      Joint type from boundary shape:
        circular -> revolute
        planar straight -> prismatic
        diffuse / no clear loop -> fixed
    Independent of the ML joint head; quality scales with
    segmentation quality (so benefits from D.1's semi-auto user
    annotation directly).

  D.5 (next) DISPATCH WIRING:
    Add --use-retrieval flag to predict_urdf_interactive.py.
    Order: D.2 retrieval -> if low sim -> D.4 geometric extraction.
    The user-refined segmentation feeds D.4 directly.

  D.6 (shipped 2026-05-02) JOINT LIMITS AS A MODEL OUTPUT:
    The 13-row data/urdf_db.json template lookup is deleted (it was
    a matchmaking pattern that returned the closest-DOF stranger's
    limits for any custom robot). Replaced with a learned head +
    geometry-based safety check:
      - LimitsHead in model.py predicts (lower, upper) per slot
      - v3 training shards embed joint_limits per joint
        (889 shards / 28,400 examples / 572 robots / 5 GB)
      - smooth-L1 loss on (lower, upper), masked by joint_valid
        AND a has_limits flag so legacy v1/v2 shards don't pollute
      - PyBullet self-collision sweep at inference refines the model
        prior down to the largest collision-free interval around home,
        with adjacent-link contacts filtered (expected for revolute)
      - --collision-sweep flag in predict_urdf_interactive.py
    Final result: PT-V3 base trained 50 ep on H200, val limits_mae
    0.358; model produces realistic priors (shoulder ±1.71, wrist
    [-1.63, 2.27]) on test_2; sweep correctly narrows joints when
    geometry constrains tighter than the model prior. Density /
    friction / damping / effort / velocity remain fixed defaults
    (no static-mesh signal predicts them; future work: add those
    heads too).

PHASE E  VLM REFINEMENT (optional)
  When model uncertainty is high or sanity checks fail:
    Render mesh from 4 canonical angles
    Ask GPT-4V / Claude / Gemini:
      "this is a 3D mesh of a robot. Joint X is predicted at position Y
       with axis Z. Is this consistent with the visible geometry? Suggest
       corrections."
    Apply suggested corrections to the URDF

PHASE F  EVALUATION
  Real-scan benchmark (MILO scans of 3-5 different robots)
  Per-vertex segmentation IoU
  Per-joint axis angle error / origin distance error
  Comparison to: heuristic baseline (Path B), Articulate-Anything,
  PARIS (where applicable), URDFormer
```

### Legacy heuristic paths (retained for historical context)

Two earlier paths converge at Phase 4. **Path B (single-mesh + per-joint photos)** is the heuristic real-scan path implemented through 2026-04-24; **Path A (K pose meshes)** is the original roadmap and is validated end-to-end only on synthetic pose meshes.

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

## 4. Phased prototype plan (8-week build)

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
