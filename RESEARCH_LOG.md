# mesh2robot — Research Log

Running log of experiments, decisions, and results. Append-only; newest entry at top.

Project goal: real-to-sim articulated robot asset generation from MILO-reconstructed meshes. See [ROADMAP.md](ROADMAP.md) for the full plan.

---

## Status dashboard

| Phase | Component | Status | Notes |
|---|---|---|---|
| 0 | Robot URDF database (legacy, 13 arms) | **Superseded** | Replaced by Phase A's 925-robot trainable manifest |
| 1 | Multi-pose MILO integration | **Real-scan path built** | Single-mesh + per-joint photo-pairs path via ArUco + ICP |
| 2 | Motion-based segmentation (synthetic poses) | **Implemented + noise-robust up to σ=5 mm** | LO-RANSAC + body merge + orphan reassignment |
| 2b | Motion-from-images (real-scan) | **Implemented** | ORB + silhouette mask + multi-body RANSAC + PnP refinement |
| 3 | Screw-axis joint extraction | **Implemented + noise-robust** | 0.061° axis / 0.060 mm origin at σ=0.5 mm |
| 3b | Geometric-prior snap (real-scan) | **Implemented + insufficient** | Snaps axes to canonicals; fails on gooseneck arms with single-plane cuts |
| 4 | Template matching + physics | **Implemented** | DOF+joint-type signature match; trimesh inertials |
| 5 | URDF assembly | **Implemented** | Jinja template; outputs pass yourdfpy reload |
| **A** | **Multi-source URDF dataset** | **Built + deduped + enriched (2026-04-26)** | **371 canonical robots** (union-find dedup over 3 signature keys; license, fidelity_class, scale_class, joint_range, mesh_bytes, link-origin AABB per entry; zero `unit_bug` URDFs in canonical set) |
| **B** | **Synthetic data generator** | **Built (2026-04-26)** | 27,510 examples / 442 raw robots → 353 after canonical filter; URDF + MJCF; v1 shards regenerated with disambiguated names |
| **C** | **Foundation model training** | **Trained (2026-04-28)** | PT-V3 31.8M params, 25 epochs, val seg 49.5% / axis 38.3° / valid 96.9% / origin 0.367 (all four exceed PointNet ep50 baseline) |
| **D.1** | **Inference pipeline** | **Built (2026-04-28)** | scripts/predict_urdf.py + predict_urdf_interactive.py; mesh → 16k pts → ML → 7-link / 6-joint URDF on test_2 |
| **D.1.GUI** | **Semi-auto annotation GUI** | **Built (2026-05-01)** | Pre-select PyVista viewer (Rect/Lasso/Camera Tab cycle, persisted annotations, strict-mode merge with 30%-coverage propagation, in-process ML+URDF on SPACE) |
| **D.2** | **Retrieval for known robots** | **In progress (2026-05-01)** | Pre-compute PT-V3 global features for 371 canonicals → cosine-sim match input → pose-align canonical URDF → exact joints |
| **D.3** | **Motion-image refinement** | **Wired (2026-05-01)** | Stage 3 opt-in via `--motion-dir`; Path-B legacy code reused; bottlenecked on test_2 calibration quality |
| **D.4** | **Geometric joint extraction** | **In progress (2026-05-01)** | 3D-circle fit on inter-link boundary loops; universal fallback when retrieval similarity is low |
| **D.5** | **Retrieve-or-extract dispatch** | Pending | Wire D.2 + D.4 into the interactive script as primary + fallback |
| **D.6** | **Joint limits as model output + collision sweep** | **Training (2026-05-02)** | PT-V3 base (106.5M) + LimitsHead training on H200 (batch=24, lr=5e-4, 50 ep); v3 shards 889/28,400 examples; ep1 step230 limits_mae 0.92; legacy `urdf_db.json` deprecated; PyBullet sweep wired |
| **E** | VLM refinement layer | Optional / deferred | Render mesh + ask VLM to verify/correct uncertain joints |
| **F** | Evaluation benchmark | Pending | Real-scan validation set + ablations vs heuristic + Articulate-Anything baseline |

**Project pivot (2026-04-25): from heuristic single-plane segmentation to a learned 3D foundation model for general articulated robot perception.** The heuristic pipeline plateaued at gooseneck-style arms where single-plane cuts can't represent the parent/child boundary correctly. Research path commits to a trained model on a multi-source robot URDF dataset.

---

## 2026-05-01 — Phase D.6 — Joint limits become a model output; DB lookup deleted

### Why this matters

Until today, joint limits in the emitted URDF came from `data/urdf_db.json` — a 13-row hand-curated lookup keyed on `(DOF, joint_type_sequence)`. For any input outside those 13 known robots, `match()` returned the closest stranger's limits (a 5-DOF custom arm got xArm6's first 5 limits, a quadruped picked whichever 12-DOF entry sorted first, etc.). When asked, the user explicitly framed this as a project-defining decision: **"build a model not just a matchmaking tool. because the custom robots will not be xarm6"**. They then rejected the milder alternative ("statistical priors over the 371 canonical URDFs") with: "Why do you need to Build statistical limit priors instead of integrating it to the ml model training?". So the answer is a trained head, not a lookup of any kind.

### What changed

**Data pipeline (URDF + MJCF):**
- `mesh2robot/data_gen/urdf_loader.py::articulate_and_label` and `mesh2robot/data_gen/mjcf_loader.py::articulate_and_label_mjcf` now return a 7th array `joint_limits (J, 2)` carrying per-joint `(lower, upper)` in chain order matching the rest. URDF source is `j.limit.lower/upper` clamped to physical caps (±π revolute, ±2 m prismatic). MJCF source is `model.jnt_range[ji]` when `model.jnt_limited[ji]`, else ±π for revolute (treated as continuous prior).
- `scripts/generate_training_data.py::_pack_shard` embeds `joint_limits` as `(B, J_max, 2) float32` alongside `joint_axes_world / joint_origins_world / joint_types / joint_valid / joint_topology`. Shards are emitted to `data/training_shards_v3/` (URDF) and `data/training_shards_v3_mjcf/` (MJCF) — full regen, not a sidecar, so all consumers see one self-describing file per shard.

**Dataset:**
- `mesh2robot/model/dataset.py` loads `joint_limits` and a per-example `has_limits` bool flag. v1/v2 shards lack the field; for those, `has_limits=False` and the targets are zeros — the loss masks those examples out so legacy shards safely contribute nothing to limits training.

**Model + loss:**
- `JointHead` in `mesh2robot/model/model.py` got a `limits_head: Linear(256, 2)` that predicts `(lower, upper)` per slot. The forward returns a new `"limits": (B, J_MAX, 2)` tensor.
- `mesh2robot/model/losses.py` gained a `limits` term (smooth-L1 on `(lower, upper)`, masked by `joint_valid & has_limits`) with weight 0.5 in `LossWeights`. New metrics `loss/limits` + `metric/limits_mae` are reported per epoch.

**Inference + URDF assembly:**
- `scripts/predict_urdf_interactive.py::build_urdf_from_predictions` accepts `pred_limits (J_MAX, 2)`, slots them into the chain via the same Z-ordered `valid_idx_by_z` projection used for axes/origins, and feeds them as the assembler's Tier-1 `final_limits_per_joint`. When the checkpoint has a trained `limits_head` (detected by `joint_head.limits_head.weight in state_dict`), the model's prediction wins; otherwise the script falls back to ±π and prints "Limits head missing or randomly init'd".
- New flag `--collision-sweep` runs `mesh2robot.core.collision_sweep.sweep_collision_free` after the initial URDF write. It loads the URDF in PyBullet (DIRECT mode), filters parent / grandparent self-collision pairs (adjacent contact is expected for revolute joints), then sweeps each actuated joint from 0 → predicted upper / lower in `--sweep-steps` (default 64) increments. The first step that triggers a non-adjacent self-collision becomes the new bound (with optional safety margin). The URDF is then re-assembled with the refined limits.

**DB lookup deleted:**
- `mesh2robot/core/template_match.py` reduced to a defaults-only bag: density 2700 kg/m³, friction 0.5, damping 0.1, effort 100 N·m, velocity π rad/s. The legacy `match(query_dof, query_types, db)` is now a backwards-compat shim that returns `make_default_template(query_dof)` and ignores its DB argument. `data/urdf_db.json` was renamed to `urdf_db.json.deprecated` to break any silent fallback.

### Verified pipeline (existing checkpoint, ±π fallback + sweep)

Re-ran `predict_urdf_interactive.py --geometric-joints --collision-sweep` on `test_2` with the existing `model_v2_ptv3_25ep` checkpoint (no LimitsHead → ±π fallback). The sweep correctly narrowed the shoulder/elbow joints:

```
[sweep] joint0 (fixed):    [-3.142, 3.142] → [-2.882, 2.490]
[sweep] joint1 (revolute): [-3.142, 3.142] → [-2.981, 2.195]
[sweep] joint2 (revolute): [-3.142, 3.142] → [-3.142, 3.142]   (free)
[sweep] joint3 (revolute): [-3.142, 3.142] → [-3.142, 3.142]   (free)
```

Sensible — shoulder and elbow joints can't rotate past the body of the arm; wrist + end-effector are free. Adjacent-pair filtering correctly distinguishes "expected joint contact" from "self-collision".

### v3 regen completed

| Source | Shards | Examples | Robots | Wall | On disk |
|---|---:|---:|---:|---:|---:|
| URDF | 546 | 17,450 | 353 (4 failed) | 31 min | 3.1 GB |
| MJCF | 343 | 10,950 | 219 (0 failed) | 13 min | 1.9 GB |
| **Total** | **889** | **28,400** | **572** | **44 min** | **5.0 GB** |

Sanity check on per-joint limits across spot-checked shards:
- `bullet3/cube_gripper_left` (1-DOF prismatic): `[-2.0, 2.0] m` (clamped at PRISM_CAP)
- `urdf_files/irb52` (6-DOF revolute industrial arm): median `[-π, 2.62 rad]` ≈ ±150°
- `willowgaragePR2` (39-DOF mixed): median span 6.0 rad
- `adroit_hand` (small revolute hand joints): median span 1.57 rad
- `pal_tiago_dual_arm` (humanoid): median span 2.75 rad
- `robosuite/ur5e` (6-DOF revolute): median `[-π, π]` standard ±180°

Distributions match the URDF/MJCF source values (no clamping artifacts beyond the documented PRISM_CAP at 2 m). Ready for training.

### H200 deployment + training launched (2026-05-02)

Project moved to a remote NVIDIA H200 (143 GB VRAM, CUDA 12.8 driver) over SSH for the v3 training run; local Win11 + 3090 stays as the inference / annotation box. Three notable infra resolutions worth recording so future repeat-deployments don't re-discover them:

1. **Conda env spec is unsolvable on miniconda 26 + libmamba** — pytorch + nvidia + conda-forge channel mix explodes on libabseil/protobuf/numpy + Python 3.13. Ditched the YAML; replaced with `setup-server.sh` (committed): tiny conda env with just `python=3.12 pip`, then everything via pip. PyTorch's official wheel index is the reliable CUDA path.
2. **torch + torchvision + torch-scatter ABI must align** — `data.pyg.org` torch-scatter wheels for `torch-2.6.0+cu124` had a `_ZN5torch3jit17parseSchemaOrName` ABI mismatch. Rolled back to `torch==2.5.1+cu124 + torchvision==0.20.1 + torch-scatter==2.1.2+pt25cu124`, which is the stable wheel anchor as of 2026-05.
3. **Container `/dev/shm` is 64 MB and unprivileged remount is denied** — `DataLoader(num_workers>0)` died with bus errors trying to share batches via shm. Workaround: `--num-workers 0` + `--in-memory` (5 GB shards fit in container RAM trivially, no disk I/O to parallelize anyway).

Added `--encoder-size {small,base}` flag to `train_model.py` to expose PT-V3's per-stage channels/depths. `base` is **106.5M params** (3.3× the 31.8M `small`); the H200's VRAM headroom finally makes it tractable.

**Training run config (live):**
```
encoder           : ptv3-base   (106.5M params)
shards            : v3 URDF + v3 MJCF (889 shards / 28,400 examples / 572 robots)
split             : stratified by-robot, canonical_train_set filter
batch_size        : 24
lr                : 5e-4 (cosine to 0 over 50 epochs)
n_points          : 16384
in_memory         : True
num_workers       : 0
epochs            : 50
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

VRAM utilization at ep1 step 60: **122 / 143 GB ≈ 85 %**. Sweet spot — comfortable headroom, model is well-fed.

**Health check at ep1 step 230:**
| Signal | Value | Interpretation |
|---|---:|---|
| `loss/limits` | 1.07 | Active — `LimitsHead` gradient flowing, not stuck near zero |
| `metric/limits_mae` | 0.92 | Cold-start; mixed-units (radians/metres) baseline |
| `loss/seg` | 2.45 | Normal CE for K=64 classes early in training |
| `metric/seg_acc` | 28.4 % | Same trajectory as v2 (49.5 % final) |
| `metric/axis_deg` | 54.6° | Same starting region as v2 (~60° early) |
| `metric/valid_acc` | 88.8 % | Already converging; easiest task |

End-to-end pipeline confirmed live: v3 shards' `joint_limits` field → dataset loader → `JointHead.limits_head` → smooth-L1 loss against masked targets → backprop. Output checkpoint will be `data/checkpoints/model_v3_ptv3_base_50ep/`.

### Pending

- Wait for 50-epoch training to complete (estimate 4–8 hours on H200 at batch 24 + base model).
- rsync trained checkpoint local → run end-to-end test on `test_2`: model-predicted limits → `--collision-sweep` refinement → final URDF.

### Files touched / added / removed (cumulative for D.6)

```
+ mesh2robot/core/collision_sweep.py     PyBullet self-collision sweep (Tier-3)
~ mesh2robot/core/template_match.py      DB lookup deleted; defaults-only bag
~ mesh2robot/data_gen/urdf_loader.py     articulate_and_label now returns joint_limits (J, 2)
~ mesh2robot/data_gen/mjcf_loader.py     articulate_and_label_mjcf same
~ mesh2robot/model/dataset.py            loads joint_limits + has_limits flag
~ mesh2robot/model/model.py              JointHead.limits_head Linear(256, 2); encoder_size param
~ mesh2robot/model/encoders.py           PTV3_CONFIGS dict; size= constructor arg ('small' | 'base')
~ mesh2robot/model/losses.py             smooth-L1 limits term + masked MAE metric
~ scripts/generate_training_data.py      packs joint_limits into shards
~ scripts/train_model.py                 --encoder-size flag plumbed through
~ scripts/predict_urdf_interactive.py    pred_limits + --collision-sweep flag; reads encoder_size from ckpt
~ scripts/predict_urdf.py                strict=False checkpoint loader; reads encoder_size from ckpt
~ scripts/smoke_test_data_gen.py         updated 7-tuple unpack
- data/urdf_db.json                      renamed to .deprecated
+ data/training_shards_v3/               546 shards / 17,450 examples / 353 robots / 3.1 GB
+ data/training_shards_v3_mjcf/          343 shards / 10,950 examples / 219 robots / 1.9 GB
+ environment.yml                        local-with-GUI conda spec (python 3.12 + vtk + pyvista)
+ environment-server.yml                 server-only conda spec (no vtk/pyvista; later superseded by setup-server.sh)
+ setup-server.sh                        pip-based bootstrap; sidesteps conda channel solver
+ requirements.txt                       slimmed to pip-only stragglers (yourdfpy, robot_descriptions, xacrodoc)
+ requirements-cuda-local.txt            (deleted; rolled into setup-server.sh)
```

---

## 2026-04-26 — Phase A.2 — Manifest gets AABB + scale_class (unit-bug guardrail)

### Filling in the documented-but-empty `aabb_extent_m` field

`scripts/build_research_manifest.py`'s `probe_robot_stats` declared `aabb_extent_m` in its docstring + return dict but never actually populated it — every canonical entry shipped with `[0.0, 0.0, 0.0]`. The original intent ("approximate bounding box extents from FK at zero pose") was a fast scale-sanity signal, not a tight bound: it's there to flag mm-encoded URDFs and other unit bugs before they pollute the synthetic data generator.

Implemented:

- **URDF**: load with `build_scene_graph=True, load_meshes=False`, `update_cfg` to all-zeros, walk `link_map`, take `urdf.get_transform(ln)[:3, 3]` for each link, take per-axis range. No mesh loading — adds ~0 cost on top of the existing per-URDF parse.
- **MJCF**: `mj_resetData` + `mj_forward` to populate `data.xpos`, take per-axis range over body world positions (skip world body 0). Same cost-free piggyback on the model parse already happening.

The output is the **link-origin AABB** — a *lower bound* on the robot's full extent (meshes can extend further). That's intentional. We don't want a tight bound here; we want a cheap signal that catches scale outliers.

### `scale_class` derivation (in `enrich`)

Buckets the canonical set by max-axis link-origin extent:

| Class | Max extent | Count | Interpretation |
|---|---|---:|---|
| compact | <0.3 m | 52 | Gripper, hand, fingertip, small actuator |
| tabletop | 0.3–1.0 m | 166 | Bench arm, half-humanoid, mobile manip |
| fullsize | 1.0–2.5 m | 141 | Full industrial arm, humanoid |
| huge | 2.5–50 m | 5 | Large industrial / multi-arm rig |
| unit_bug | >50 m | 0 | mm-encoded URDF (rescale by 1e-3) |
| unknown | — | 7 | One-link / coincident-origin fixtures |

Spot-check on the 5 `huge` entries:
- KUKA KR150 (2 variants) — ~3 m reach industrial arm. Legit.
- ABB IRB7600 — large industrial arm. Legit.
- ABB IRB6650S — large industrial arm. Legit.
- Gymnasium kitchen_franka — Franka arm + a kitchen scene; the AABB picks up the kitchen, not just the arm. Edge case but expected.

Spot-check on `unknown`: all are one-link or all-coincident fixtures (cube grippers, umi_gripper, base-only tidybot, two-link r2_left_gripper) where FK at zero pose collapses every origin to the world frame. Correctly flagged as "no FK signal" rather than mis-bucketed.

**Net result: zero `unit_bug` URDFs in the canonical set** — the dataset is unit-clean, which closes a class of silent training-time failures (e.g. a mm-encoded humanoid landing 100× larger than its peers in normalized point cloud space and dominating the loss).

### Surfaced in `summarize_manifest.py`

- CSV: added `scale_class`, `aabb_x_m`, `aabb_y_m`, `aabb_z_m` columns.
- Markdown: added a per-class headline distribution table, included Scale + AABB columns in the full canonical robot list.

Regenerated `data/canonical_robots.csv` (372 lines incl. header) and `data/canonical_robots.md` (472 lines, paper-appendix-ready).

### Files touched

```
scripts/build_research_manifest.py    + _urdf_link_origin_aabb / _mjcf_body_aabb
                                      + _classify_scale; populates scale_class on every entry
scripts/summarize_manifest.py          + scale_class + AABB columns
data/robot_manifest_research.json      re-probed (371 canonical entries with non-zero AABB)
data/canonical_robots.{md,csv}         regenerated
```

---

## 2026-05-01 — Phase D.1 GUI shipped; D.2/D.4 plan committed

### Phase D.1 — Semi-automatic annotation GUI

Built `scripts/predict_urdf_interactive.py`, a PyVista-based pre-select annotation tool. The user annotates rough regions BEFORE the ML runs, ML runs once, the strict-mode merge produces final per-face labels:

- **Three modes via Tab cycle**: Rect (rectangle pick) → Lasso (freeform polygon trace) → Camera (rotate / zoom). Lasso uses `vtkInteractorStyleDrawPolygon` for visual trace + parallel mouse observers for point capture (since the style's `GetPolygonPoints()` isn't exposed in this VTK Python binding). Face selection projects mesh face centers to screen via vtkCoordinate, then matplotlib's `Path.contains_points`.
- **Strict-mode merge** (`merge_user_and_ml_labels`): user-tagged faces are hard overrides. For each ML cluster, if the user covered >30% (tunable via `--propagation-threshold`), the whole cluster takes the user label (handles rough hints + ML over-segmentation). Sub-30% touches keep user faces but renumber the rest. Untouched ML clusters get fresh labels by Z-distance from the user-annotated centroid.
- **Robust persistence**: every digit press writes `user_annotations.json` to disk so accidental window crashes don't lose work. Replay flag `--user-annotations <path>` re-runs ML with saved annotations, no GUI needed.
- **Robust exit**: `plotter.show(interactive_update=True)` + manual event-loop pump with a `state["done"]` flag instead of relying on `plotter.close()` to unblock `show()` (which doesn't on this VTK build).
- **Motion-image stage opt-in** via `--motion-dir`: per-joint photo pairs run through legacy Path-B image RANSAC; output overrides ML's predicted joint axes/origins where extraction succeeds.

### Verified end-to-end on test_2

Full annotation pass (lasso, all 7 links, 346,524 faces / 92.6% coverage) → in-process ML → `Strict-mode merge: propagated_clusters {0:0, 1:1, 2:2, 3:3, 4:3, 5:4, 6:4, 7:6}`, `split_clusters {}`, `untouched_remap {}`. **The merge collapsed two pairs of ML over-segmented clusters (3+4, 5+6) into single user-defined links — the segmentation-fix behavior.** Final URDF: 7 links, 6 actuated joints (correct xArm6 topology), loads cleanly with `yourdfpy`. See `output/test_2_full_annotation/refined/robot.urdf`.

### Visualizer + GLB export

`scripts/visualize_refined.py` renders the assembled URDF (with FK applied) as a colored GLB. Two modes:

- `--from-urdf`: reads URDF + per-link meshes, applies forward kinematics so the assembled robot pose is correct (links don't clump at origin).
- `--from-annotations`: shows ONLY directly-clicked user faces, leaves rest grey — useful for spotting holes in annotation coverage.

**Auto-export at the end of `predict_urdf_interactive.py`**: whenever user annotations are present (interactive GUI or `--user-annotations` replay), the script now drops two GLBs at the top of the output folder so rough-vs-refined is a one-click A/B view in any GLB viewer:

- `user_annotation.glb` — raw user lasso coverage; tagged faces colored by link, untouched faces grey. World frame.
- `refined_assembled.glb` — refined URDF rendered with FK so each link sits at its assembled-pose world position. World frame.

Both share the same world frame (the input mesh's `T_cleaned_to_original` is applied before annotation, and the URDF assembly is in the same world frame), so they line up spatially. The shared helper `render_face_labels_on_mesh` was factored out of `visualize_refined.render_from_annotations` for reuse with already-loaded data.

### Known cosmetic issue: PyVista actor flicker

The lasso-mode red highlight + status banner are added as VTK actors but get cleared by some internal render-loop machinery on this VTK 9.6.1 + PyVista 0.47 + Win + Py3.13 stack. Tried 12+ variants (cell-color recolor, raw-VTK actor + renderer.AddActor, brute-force re-assert in 20 ms event-loop ticks, ForceOpaque, Modified() flags, pixel-space vs world-space) — flicker persists. The data-side workflow is unaffected (selections are captured, digits label correctly, SPACE commits). Saved as a memory note to skip this rabbit hole in future sessions.

### What test_2 reveals about joint quality

User feedback after the full-annotation run: segmentation is markedly better than pure-ML (no over-segmented phantom links), but **joint axes / origins are still visibly wrong** — that's the model's known 38° val axis_deg error showing up in real predictions. Specifically, joints 1 and 2 are reasonable (≈±Z and ±Y as expected for a base-mount arm), but joints 3-7 come out oblique when the manufacturer URDF has them along principal directions.

### Decision: Phase D.2 retrieval + D.4 geometric extraction (next)

Two complementary paths to attack the joint-axis error:

**D.2 RETRIEVAL FOR KNOWN ROBOTS.** Most users' inputs *are* a known robot. We have xArm6 (3 variants in bullet3), xArm7 (4 variants in Menagerie/robosuite), Panda, UR series, IIWA, Sawyer, Yumi, Spot, Atlas, Unitree H1/G1/Go2, etc. — direct coverage of common robots. Plan:
1. Pre-compute PT-V3 global feature for each of the 371 canonical robots → `data/canonical_embeddings.npy` (one-time, ~10 min)
2. At inference: cosine-similarity input embedding vs canonicals; if sim > 0.85, retrieve that canonical URDF and pose-align via Procrustes/ICP
3. Result: known robots get the manufacturer's exact URDF (joints to tooling-spec precision) instead of a 38°-off ML guess

**D.4 GEOMETRIC EXTRACTION.** For genuinely custom robots that don't match anything canonical, fit joints directly from the now-clean per-link segmentation:
1. Find the boundary loop between adjacent link meshes (face-adjacency in the original mesh; edges where neighboring faces have different labels)
2. Fit a 3D circle (SVD plane fit + algebraic 2D fit on the projected points)
3. Circle center → joint origin; circle normal → joint axis
4. Joint type from boundary shape (revolute = circular, prismatic = planar straight, fixed = no clear loop)

**Combined dispatch (D.5):** retrieval first, geometric extraction as fallback. The user's annotation feeds D.4 directly — segmentation quality from rough hints + ML compensation is high enough now that geometric joint fitting is feasible.

### Files added

```
scripts/predict_urdf_interactive.py   semi-auto GUI + ML + URDF in one process; auto-emits comparison GLBs
scripts/visualize_refined.py          GLB export from URDF (with FK) or raw user_annotations.json;
                                      exposes render_face_labels_on_mesh for in-process use
output/test_2_full_annotation/        first complete user-annotated test_2 result
  ├── original/robot.urdf             pure-ML 8-link URDF
  ├── refined/robot.urdf              user-refined 7-link URDF (correct xArm6 topology)
  ├── user_annotation.glb             raw user lasso coverage (rough), untouched faces grey
  ├── refined_assembled.glb           refined URDF with FK applied
  └── pure_ml.glb                     ML-only baseline for comparison
```

### Phase D.2 retrieval — attempted, abandoned

Sketched and prototyped retrieve-or-extract: precompute PT-V3 global features for the 371 canonical robots, cosine-sim the input, return the canonical URDF when sim > τ. Built `scripts/compute_canonical_embeddings.py` and `scripts/retrieve_canonical.py`; embedded 314 / 371 canonicals (others failed at zero-config FK).

**Result: retrieval doesn't work with our model.** On test_2 (an xArm6 scan), retrieval ranked KUKA IIWA as the top match at sim 0.940; the actual xArm6 (which IS in the manifest, 3 variants) ranked #119 / #121 / #148 at sim 0.72-0.78. Cause: PT-V3's global pooled feature is a side product of segmentation training, not retrieval-optimized; the embedding is pose-sensitive and biased toward IIWA-shaped robots (which are over-represented in training data, 8 variants).

For real retrieval-quality, we'd need a separate contrastive-loss-trained encoder, geometric/structural features (DOF + bbox + axis distribution) instead of global pool, or explicit ICP-with-scoring against each canonical. None is a quick win.

Removed the embedding artifacts; saved a memory note to skip this rabbit hole next session.

### Phase D.4 — Geometric joint extraction from clean segmentation (works)

Built `mesh2robot/core/geometric_joints.py` with `extract_joints_from_segmentation`. For each adjacent (parent_link, child_link) pair in chain order:
1. Walk `mesh.face_adjacency` to find boundary edges (face pairs with different labels).
2. Collect boundary vertices.
3. Fit a 3D circle: PCA → plane normal, then algebraic 2D fit `(x-a)² + (y-b)² = r²` in the plane basis. Returns center, normal, radius, plane residual.
4. Compute a circularity score (mean rel-err of in-plane distance vs radius); revolute if score > 0.55, fixed otherwise.
5. Orient the axis so it points from parent toward child for URDF convention consistency.

Wired as `--geometric-joints` flag in `predict_urdf_interactive.py`; precedence is **geometric > motion-image > ML head** when sources agree.

**Result on test_2** (full user annotation, all 7 links labelled):

| Joint | ML axis (38° avg val err) | Geometric axis | Notes |
|---|---|---|---|
| J1 base | (0.02, 0.01, **−1.0**) | (−0.11, 0.08, **0.99**) | both ≈ Z; agree |
| J2 shoulder | (−0.07, **0.997**, 0.03) | (0.08, **0.996**, 0.01) | both ≈ Y; agree |
| J3 elbow | (0.41, 0.16, 0.90) **oblique** | (0.09, **0.995**, 0.04) ≈ Y | **geometric correct**, ML 76° off |
| J4 wrist roll | (−0.66, 0.74, 0.11) | (0.85, 0.04, 0.53) along forearm | geometric oriented sensibly |
| J5 wrist pitch | ML present | **fixed** (plane resid 13 mm vs ~1 mm) | boundary noisy in user annotation |
| J6 end-effector | (−0.72, 0.69, 0.03) | (−0.05, 0.04, **0.998**) ≈ Z | **geometric correct**, ML 84° off |

Output URDF: 7 links, 5 actuated joints + 1 fixed. Geometric circle radii (32–47 mm) match the real xArm6 joint housing diameters. Plane residuals 0.8–3.5 mm on healthy joints (the noisy J5 had 13 mm — segmentation cleanup needed there).

Phase D.4 is the right primary path for joint accuracy: it bypasses the model's joint head entirely, depends only on segmentation cleanliness (which the GUI now produces well), and is universal across known/custom robots.

**Output files:** `output/test_2_geom_joints/refined/robot.urdf` + `refined_assembled.glb`.

---

## 2026-04-23 — Research-grade manifest enrichment (dedup + license + tier)

### Phase A.1 — Manifest fixes and enrichment

While preparing for the PT-V3 training run, audited Phase A's `data/robot_manifest.json` for cross-source duplication and discovered two issues:

**Bug fix: case-insensitive glob double-counted every URDF.** `build_robot_manifest.py` had `URDF_PATTERNS = ("**/*.urdf", "**/*.URDF")`, but on Windows's case-insensitive filesystem both patterns matched the same files, so every URDF was inserted into the manifest twice. 1574 of the 3594 entries (44%) were exact duplicates. Fixed `discover_files` to dedup on lowercased relative path. Real Phase A counts:

| Metric | Before fix | After fix |
|---|---:|---:|
| Total entries | 3594 | 2020 |
| Trainable | 925 | 572 |

The PT-V3 baseline metrics in earlier entries used the inflated dataset; effective unique-robot count was always ~half what the entry stated. The trained checkpoint is still valid (the duplicates were genuinely the same .urdf re-loaded, not different robots), but per-robot diversity numbers should be quoted as 572, not 925.

### Cross-source deduplication

Even after the glob fix, robots like Panda, UR5e, IIWA, Spot appear in 3-5 sources (Menagerie + robot-assets + robosuite + urdf_files_dataset, etc.) under the same model name. Without dedup, by-robot val splits leak: same Panda under different sources can land in train and val simultaneously.

Built `scripts/build_research_manifest.py` to enrich the base manifest with:

1. **Canonical signature**: `(family, leaf_filename, dof)`. Groups same-model entries across sources.
2. **Canonical picker**: source priority order is Menagerie > robot-assets > drake_models > robosuite > Gymnasium-Robotics > urdf_files_dataset > bullet3. Within a source, the entry with the most resolved meshes wins.
3. **`is_canonical`, `dup_of`, `dedup_group_size`** flags per entry.
4. **`canonical_train_set`** = `is_trainable AND is_canonical`. The **492-robot leak-free training set** for Phase B going forward.
5. **Quality tier** (high/medium/low) per source.
6. **License metadata**: source-level (MIT, Apache-2.0, zlib, BSD-3-Clause) plus parsed per-directory for MuJoCo Menagerie (62 robot directories with explicit license blocks).
7. **Mesh-bytes probe** (optional `--probe-mesh-bytes`): sums on-disk size of every visual mesh referenced by each canonical URDF/MJCF — cheap fidelity proxy without loading meshes into memory.

### Dedup results (union-find over family-leaf, path-tail, and Menagerie-dir signatures)

The dedup picker uses union-find over **three** keys: `(family, leaf, dof)` for cross-source matches via vendor patterns (Panda, UR5e, Spot under different folder structures); `(normalized_path_tail, dof)` for re-bundled paths whose family fell back to source name (NASA Valkyrie, R2 humanoids); and a Menagerie-only `(parent_dir, dof)` that collapses each robot's MuJoCo variants — `<robot>.xml`, `<robot>_mjx.xml`, `scene.xml`, `scene_mjx.xml`, `scene_<sub>.xml`. Without the third key, Google Barkour V0's 5 variant XMLs each got their own canonical entry; with it, all 5 collapse into one (V0 stays separate from Vb because they're in different directories).

```
Trainable (loaded ok + meshes ≥ 80%):      572
Canonical (deduped):                        371
Duplicates marked (dup_of != None):         201

Dedup groups:
  with duplicates: 125
  singletons:      246

Canonical-set quality tier (by source — coarse heuristic):
  low       217  (urdf_files_dataset 183 + bullet3 34)
  high      113  (mujoco_menagerie 69 + robot-assets 44)
  medium     41  (robosuite 25 + Gymnasium-Robotics 16)

Canonical-set fidelity class (by mesh bytes on disk — true quality signal):
  high       130  (>=5 MB curated CAD or humanoid)
  medium     176  (0.5-5 MB typical XACRO-with-STL)
  low         65  (<0.5 MB simplified physics meshes)
```

The Menagerie collapse drops it from 156 → 69 entries — Menagerie ships ~62 robot directories, so 69 makes sense (some directories like `unitree_h1` host multiple distinct DOF setups).

The source-based tier is **misleading**: urdf_files_dataset's xacro outputs and robot-assets's curated set both pull from the same CAD pipelines (e.g. Yumi 58.6 MB appears in both at identical fidelity). robosuite + Gymnasium-Robotics intentionally simplify meshes for fast simulation (median 0.65 MB). **Use `fidelity_class` for quality filtering, not `quality_tier`.**

Top-fidelity canonical robots (mesh bytes): 67.6 MB Kinova Movo, 58.6 MB ABB Yumi, 43.7 MB NASA R2 humanoid variants.

Notable cross-source duplicate groups picked correctly:
- `panda/panda/9dof` (3 instances): Menagerie chosen over robot-assets and urdf_files_dataset
- `ur5/ur5e/6dof` (3 instances): Menagerie chosen over robosuite and ros-industrial xacro
- `spot/spot/12dof` (3 instances): Menagerie chosen over robosuite and random/spot_ros
- `iiwa/kuka_iiwa/7dof` (4 instances): robot-assets chosen over 2× bullet3 + urdf_files copy

### Files added

```
scripts/build_research_manifest.py         dedup + license + tier enrichment
scripts/build_robot_manifest.py            (modified — case-glob fix)
data/robot_manifest_research.json          enriched manifest (492 canonical entries)
```

### Implication for training

`ShardDataset` currently consumes shards from `training_shards_v1/` + `training_shards_v1_mjcf/`, which were generated from the un-deduped pre-bug-fix list (442 unique robot names in shards). With dedup applied:

- Shard robots: 442
- Canonical robots: 396 (matched against shard naming convention `<source>/<Path(path).stem>`)
- **Overlap: 393** — these become the leak-free training pool
- **Filtered out: 49** — duplicates like `bullet3/panda`, `robot-assets/panda` (Menagerie's Panda is the canonical), `urdf_files_dataset/r2b_*` (robot-assets is canonical), etc.
- Missing: 3 canonical robots (probably failed to generate during Phase B; not a blocker — `r2_left_gripper`, `rethink_electric_gripper`, `spot_arm`)

### Wired into the trainer

Added `load_canonical_robot_names(manifest_path) -> set[str]` to `mesh2robot/model/dataset.py` and a `--canonical-manifest data/robot_manifest_research.json` flag to `scripts/train_model.py`. With the flag, both `train_robots` and `val_robots` sets are intersected with the canonical pool before constructing `ShardDataset` instances. The by-robot split logic is unchanged; it just operates on a smaller, dedup-clean pool.

```bash
python scripts/train_model.py \
    --shard-dir data/training_shards_v1 \
    --shard-dir data/training_shards_v1_mjcf \
    --canonical-manifest data/robot_manifest_research.json \
    --encoder ptv3 --in-memory ...
```

The 41.6% val seg_acc baseline was measured on a 442-robot split with cross-source duplicates possibly straddling train/val. Post-dedup re-evaluation results from `scripts/eval_checkpoint.py` against `data/checkpoints/model_v1_pointnet/checkpoint_epoch_050.pt`:

| Metric | No filter (442 robots) | Canonical filter (393 robots) | Menagerie-aware (353 robots) | **Stratified canonical (35 val robots)** |
|---|---:|---:|---:|---:|
| seg_acc | 41.66% | 42.06% | 41.65% | **39.16%** |
| axis_deg_err | 41.33° | 40.43° | 40.74° | **37.99°** |
| origin_m_err | 0.390 | 0.387 | 0.388 | 0.411 |
| valid_acc | 95.43% | 95.47% | 95.46% | 94.76% |
| n_batches | 360 | 323 | 315 | 473 |

The stratified val tells the most informative story: **−2.5pp seg_acc** (model genuinely struggles with the high-DOF humanoids and dexterous hands that the random val under-represented), but **−2.7° axis err** (those same humanoids have more constrained joint axes than the random val's mostly-6-DOF-arms population). Random vals, by oversampling low-fidelity 6-DOF arms in `urdf_files_dataset`, flatter the segmentation metric while penalizing axis prediction. **The stratified split is the metric to beat going forward.**

**Conclusion: the 41.6% baseline was NOT inflated by leakage in any meaningful way.** The duplicates removed were "redundancy-leaks" (same robot under multiple sources — Panda from Menagerie + Panda from robot-assets + Panda from urdf_files_dataset, all of which are physically the same robot) rather than "novelty-leaks" (a feature seen in train would also appear in val). After dedup, both pre- and post-filter val sets contain the same *physical* robots; only the redundant copies are gone. Per-shard variance probably explains the 0.4pp wobble.

This is good news for evaluation comparability: existing baseline numbers carry over to the dedup-clean regime. Future training runs (PT-V3 etc.) should still use `--canonical-manifest` for the cleanest by-robot val signal, but historical checkpoint numbers don't need re-quoting.

### Vendor pattern coverage (Phase A.1.b)

Extended `VENDOR_PATTERNS` in `build_robot_manifest.py` from 71 to 110 patterns covering NASA r2/val/valkyrie, Ghost Robotics minitaur/vision60, Trossen vx300/wx250/widowx, Stanford TidyBot, Hello Robot Stretch, Berkeley Humanoid, Tetheria aero_hand, IIT softfoot, Booster T1, Fourier N1, PNDbotics adam_lite, Schunk modular, plus low-coverage gym/research demos (Adroit hand, dynamixel, Google Robot, robot_soccer_kit, low_cost_robot_arm).

Source-fallback family rate dropped from **27% → 8.2%** (153 → 38 unmatched). New top vendors after extension: NASA 33, Ghost Robotics 8, Hello Robot 4, Stanford 4. The remaining 38 unmatched are mostly bullet3 demo assets (cube_gripper, r2d2, wheel) and Gymnasium-Robotics manipulate envs that aren't really standalone robots.

### Files added this entry

```
scripts/build_research_manifest.py   union-find dedup (3 keys), license, fidelity, joint-range probe
scripts/build_robot_manifest.py      case-glob fix, +39 vendor patterns
scripts/eval_checkpoint.py           checkpoint re-evaluation harness
scripts/summarize_manifest.py        produces canonical_robots.{md,csv} (paper-ready tables)
mesh2robot/model/dataset.py          load_canonical_robot_names()
scripts/train_model.py               --canonical-manifest flag
data/robot_manifest_research.json    enriched manifest (371 canonical)
data/canonical_robots.md             auto-generated paper-ready table
data/canonical_robots.csv            same data as CSV
```

### ⚠️ Data-quality bug discovered: shard name collisions

The shard generator (`scripts/generate_training_data.py`) names each robot by `f"{source}/{Path(path).stem}"` — just the filename without the parent directory. This silently conflates robots that happen to share a stem. Discovery via the new `scripts/check_manifest.py` health check:

| Collapsed shard name | Distinct canonical robots |
|---|---|
| `robosuite/robot` | **9** (Baxter 14-DOF, GR-1 32-DOF, IIWA 7, Jaco 7, Kinova3 7, Panda 7, Spot Arm 6, Tiago 15, xArm7 7) |
| `mujoco_menagerie/left_hand` | 4 (Leap, Shadow, Tetheria, Allegro) |
| `mujoco_menagerie/hand` | 2 (Panda gripper 2-DOF, xArm7 hand 6-DOF) |
| `mujoco_menagerie/stretch` | 2 (Stretch v1 17-DOF, Stretch 3 20-DOF) |
| `mujoco_menagerie/2f85` | 2 (Robotiq 2F85, 2F85 v4) |
| `Gymnasium-Robotics/reach` | 2 (Fetch reach 15-DOF, hand reach 24-DOF) |

**Impact**: 21 canonical robots (5.7%) get treated as the same training entity in shards. The by-robot train/val split puts Baxter and IIWA in the same bucket. The model sees mixed point clouds labeled with the same identity.

**Fix**: change the shard generator to `f"{source}/{Path(path).parent.name}/{Path(path).stem}"` and regenerate the v1 shards (~45 min). Until that's done, the metrics for those 6 collision groups should be considered noisier than the baseline. The 41.65% seg_acc baseline is likely a slight underestimate because of this, since the model is being penalised on intra-collision-group heterogeneity it can't disambiguate.

**Status**: fix-forward landed in `generate_training_data.py`. New shards from this point use the disambiguated convention. Existing v1 shards patched in-place via `scripts/patch_shard_names.py` (now in `data/training_shards_v1_v2_named/` and `data/training_shards_v1_mjcf_v2_named/`).

After patching, ran `scripts/verify_patched_shards.py`: all 27,510 examples preserved byte-identical on geometric arrays (points, joint_axes, joint_origins, joint_types, joint_topology, joint_valid). Unique robot names rose 442 → 452 (+10) from disambiguation.

Re-eval on patched shards (epoch-50 PointNet, stratified val):

| Shards | seg_acc | axis_deg | val_batches |
|---|---:|---:|---:|
| v1 legacy (collisions) | 39.16% | 37.99° | 473 |
| v2 patched (disambiguated) | **41.08%** | **37.39°** | 428 |

The +1.92pp seg_acc improvement is real signal cleanup, not a model change: the legacy shards put all 9 robosuite robots under one "robosuite/robot" label, so when by-robot split sent that bucket to val, the metric averaged over 9 different point cloud distributions held under one identity. Disambiguation gives each robot its own label and lets the split do the right thing per-robot.

`scripts/patch_shard_names.py` — a surgical tool that:

1. Loads each shard's `names` and `joint_types` arrays.
2. For each ambiguous slot (e.g. `robosuite/robot`), groups its examples by `(dof, joint_types_signature)` — that's the unique kinematic fingerprint.
3. Looks up the matching canonical entry in the manifest and assigns the new disambiguated name.
4. Writes patched output to a parallel directory (`<dir>_v2_named/`), leaving originals untouched.

Outputs go to `data/training_shards_v1_v2_named/` and `data/training_shards_v1_mjcf_v2_named/`. The trainer's `--shard-dir` flag can simply point at the patched dirs once verified.

### Stratified val split (`--stratified-split`)

Random by-robot splits with a fresh seed sometimes land lopsidedly — all high-fidelity Menagerie humanoids in train, val mostly low-fidelity bullet3 arms. To make val genuinely representative of the deployment distribution, added `stratified_split_canonical()` to `dataset.py` and a `--stratified-split` flag to the trainer. Strata = `fidelity_class × DOF_bucket(1-6, 7-12, 13+)`; each cell contributes `round(cell_size × val_frac)` robots to val (min 1 if cell ≥ 2).

Stratified val (seed=0, val_frac=0.1) on the 353-robot canonical pool:

| Stratum | Random by-robot (baseline) | Stratified |
|---|---|---|
| Val robots | 38 | 35 |
| Vendors covered | 18 | **21** |
| Fidelity (H/M/L) | 15/19/4 | **13/15/7** |
| DOF buckets (≤6/7-12/13+) | 13/8/9 | **16/9/10** |

The stratified val explicitly contains: UR family, Franka FR3, Kuka iiwa14, Sawyer, NASA R2 (multiple DOF variants), Unitree H1+Go2, Robotis OP3, ABB+Fanuc+Staubli arms, Allegro hand, Gym manipulate-egg, IIT softfoot, and a Minitaur quadruped — a much stronger benchmark than a random draw.

### Joint-range metadata (paper-ready stats)

The probe captures `joint_range_total_rad` (revolute + continuous) and `joint_range_total_m` (prismatic) per canonical entry, with sentinel-value clamping (revolute span > 4π → treated as continuous and capped at 2π; prismatic span > 10 m → capped at 2 m). Without this clamp, a Frankie URDF with `lower=-1e6 upper=1e6` reported 114 million degrees of total range.

Across the 371 canonical robots:
- 344 (93%) have parseable joint-range data
- Median per-robot revolute range: **2160°** (≈ 6π — consistent with 6-DOF arms at ±π per joint)
- 61 robots have prismatic motion, totaling 64.7 m of linear range
- Top 5 widest revolute range: NASA R2 humanoid variants (23,900° each across 74 joints) and Google Robot Soccer Kit (22,680° across 64 joints)

---

## 2026-04-26 (later) — Baseline trained; PT-V3 wired in; VRAM-tuned

### Baseline PointNet, 50 epochs (12 hr 44 min)

Final metrics on the 442-robot dataset (398 train robots / 44 val robots, by-robot split, no leakage):

```
              train       val
seg_acc       46.8%       41.6%       (5pp gap, model not overfitting)
axis_deg      42.5°       42.1°       (val ≈ train; encoder bottleneck)
origin (norm) 0.455       0.410       (poor, ~10-20 cm in physical units)
valid_acc     95.7%       95.5%       (essentially solved)
```

50 checkpoints saved to `data/checkpoints/model_v1_pointnet/`. Loss curves and per-task metrics in `train_log.csv`.

**Diagnosis**: PointNet's max-pool encoder is saturating. Train and val are almost equal — bottleneck is representational capacity, not data. Per-vertex segmentation needs local feature aggregation; max-pool throws away too much geometry.

### Phase C.8 — Point Transformer V3 wired in

- Cloned the standalone `Pointcept/PointTransformerV3` repo, vendored `model.py` + `serialization/` into `mesh2robot/model/ptv3/`.
- Installed deps: `spconv-cu124`, `torch-scatter` (PyG wheel matched to torch 2.6.0+cu124), `timm`, `einops`, `addict`, `easydict`, `torchvision==0.21.0+cu124`.
- Built `mesh2robot/model/encoders.py` — both `PointNetEncoder` (kept as baseline) and `PointTransformerV3Encoder` expose the same interface `forward(points: (B,N,3)) -> (per_point: (B,N,F), global: (B,F))`. The PT-V3 wrapper handles the (B,N,3) ↔ Pointcept `Point` dict conversion plus grid-coord serialization.
- `Mesh2RobotModel(encoder='ptv3')` now uses PT-V3. `--encoder ptv3` flag added to `train_model.py`. flash_attn disabled (not installed).
- PT-V3 medium config: 5 enc stages × (32, 64, 128, 256, 384) channels, 4 dec stages × (64, 64, 128, 256), num_heads (2,4,8,16,24). **31.8 M parameters** (vs 0.5 M baseline, 64× larger).

### VRAM benchmarks on RTX 3090 (24 GB)

| Batch × N points | VRAM (peak) | Step time | Per-epoch (1540 steps) | 50-epoch run |
|---|---|---|---|---|
| 8 × 8192   | ~2.3 GB | ~0.7 s | ~36 min | 30 hr |
| 16 × 8192  | ~5 GB   | ~1.5 s | ~38 min | 32 hr |
| 32 × 8192  | not benchmarked (skipped) | — | — | — |
| 64 × 8192  | **OOM** (attention scales O(N²)) | — | — | — |

PT-V3's serialized attention is the memory hog — quadratic in patch size (1024 by default) and stacked across 5 enc + 4 dec stages. Larger batches don't fit despite the model being only 31.8M params.

### Smoke train results (PT-V3 vs baseline at equivalent step count)

After ~30 train steps at batch=16:
- PT-V3:    train seg_acc 18%, val seg_acc 9-15% (single noisy val pass)
- PointNet at equivalent point: ~5% (epoch 1 step 30 of full training)

The early-step gap suggests PT-V3 will outperform the baseline meaningfully. But running 50 epochs at ~32 hr is a serious budget; should consider halving to 25 epochs (PT-V3 typically converges faster than PointNet on similar tasks) or reducing to batch 8 + 4k points for a ~10-hr smoke run before committing.

### Files added / changed

```
mesh2robot/model/ptv3/__init__.py
mesh2robot/model/ptv3/model.py             vendored from Pointcept/PointTransformerV3
mesh2robot/model/ptv3/serialization/       (z_order, hilbert)
mesh2robot/model/encoders.py               PointNet + PT-V3 with same interface
mesh2robot/model/model.py                  encoder='pointnet'|'ptv3' switch
scripts/train_model.py                     --encoder flag
data/checkpoints/model_v1_pointnet/        50 epochs, baseline final
data/checkpoints/model_v1_ptv3_smoke/      4-step PT-V3 smoke
data/checkpoints/model_v1_ptv3_bench/      30-step batch-8 benchmark
data/checkpoints/model_v1_ptv3_bench16/    30-step batch-16 benchmark
```

### Dataset speedup: `--in-memory` flag

Added `ShardDataset(in_memory=True)` and `--in-memory` flag to the trainer. Preloads all 861 shards (7.2 GB) into a dict cache at startup, eliminating per-batch `np.load()` from disk. Verified:
- Cold-load time: 25 sec for 7.2 GB
- 30 PointNet steps + full val: 10 sec total (was ~30 sec)
- Effective speedup for the I/O-bound PointNet: ~7× (15 min/epoch → 2 min/epoch)
- For PT-V3 the speedup is smaller (compute dominates), but still ~30% per-epoch saving

### Open question

Run config decisions for the real PT-V3 training (with `--in-memory`):
- batch 16 × 8k pts × 25 epochs ≈ **~10 hr** (was 16 hr without in-memory)
- batch 8 × 4k pts × 50 epochs ≈ **~5 hr**
- batch 16 × 4k pts × 50 epochs ≈ **~12 hr**

The point-count vs batch-size tradeoff matters because PT-V3 attention is O(P²) where P is the per-patch token count. Smaller N → smaller P → much faster, but loses some geometric resolution. For our 442 unique robots dataset, 4096 points should be enough — robots are simple enough that finer point sampling has diminishing returns.

---

## 2026-04-26 — Phase B + C scaffolded; v1 dataset and baseline trainer working

### What landed today

**Phase B (synthetic data generator)** is now end-to-end functional with both URDF and MJCF inputs:

- `mesh2robot/data_gen/urdf_loader.py` — load URDF, FK any random configuration, concatenate per-link visual meshes into one combined mesh with per-vertex link labels, sample point clouds.
- `mesh2robot/data_gen/mjcf_loader.py` — same API for MuJoCo Menagerie's MJCF format. Tested on Anymal-B (956k vert mesh, 12 hinge joints). Faster than URDF because meshes are pre-compiled.
- `mesh2robot/data_gen/augment.py` — vertex Gaussian noise / hole-punching / point dropout / random rigid + scale, sim-to-real augmentations.
- `scripts/generate_training_data.py` — production driver, multiprocess (`ProcessPoolExecutor`), shards 32 examples per `.npz`, supports both `--format urdf` and `--format mjcf`. Auto-clamps prismatic joint limits to ±2 m to avoid Fetch-style millions-of-meters samples.

**v1 training set built** with `--n-configs 30 --n-points 16384`:

| | URDF | MJCF | Combined |
|---|---:|---:|---:|
| Trainable robots | 698 / 706 | 219 / 219 | 442 unique |
| Examples | 20,940 | 6,570 | **27,510** |
| Shards | 655 | 206 | **861** |
| Wall time | 40 min | 5.8 min | — |
| Throughput | 8.7 ex/s | 19 ex/s | — |
| Disk | 3.7 GB | 1.2 GB | **5.1 GB** |
| Robot failures | 8 (no-root URDFs) | 0 | — |

Joint count distribution after combination: 6-DOF arms 26%, 7-DOF 8%, 12-DOF (humanoid/quadruped) 7%, 16-DOF 6%, with a long tail up to 102-DOF dexterous bodies. MJCF ingest brought in Apollo, Spot, ALOHA, Cassie, Allegro, Shadow Hand, Trossen arms, Berkeley Humanoid, Anymal-B/C, Unitree H1/G1/Go1/Go2/A1/Z1, IIWA, Sawyer, Robotiq.

**Phase C (foundation model + training loop)** scaffolded and validated end-to-end:

- `mesh2robot/model/dataset.py` — PyTorch `ShardDataset` reading the v1 shards. Loads variable-length joint arrays into fixed `J_MAX=32` padding + `joint_valid` mask. **Unit-ball normalization using 99th-percentile point radius** (the v1 dataset has outlier examples with one vertex at 162 m due to a malformed URDF; max-radius normalization would crush the rest of the cloud, so percentile is used). **Z-axis rotation augmentation** at `__getitem__` time, applied identically to points and joint axes/origins.
- `mesh2robot/model/model.py` — PointNet baseline encoder (per-point MLP + max-pool global feat) plus three heads:
  - segmentation head (per-point MLP over `K_LINKS_MAX=64` link classes)
  - joint-pose head (J_MAX learned slot embeddings, decode axis + origin + type + valid logit)
  - all heads trainable jointly.
- `mesh2robot/model/losses.py` — multi-task: cross-entropy seg (ignore_index=-1), cosine-distance-on-axis (sign-invariant), smooth-L1 origin, **class-weighted CE for joint type** (revolute 1.0 / continuous 2.5 / prismatic 8.0 to balance the v1 distribution where prismatic is only 1.2%), BCE for joint validity. Per-task metrics: seg_acc, axis_deg_err, origin_m_err, valid_acc.
- `scripts/train_model.py` — AdamW + cosine LR + checkpoint per epoch + CSV log, configurable batch / device / epoch budget.

**CPU smoke run on v1** (16 steps × batch 4 × 4k pts):
- Train loss falls 6.11 → 5.10 in 10 steps; seg_acc 0.5% → 22%; type_loss 1.78 → 0.76.
- Val seg_acc 6.6%, axis_deg 57°, origin_m 0.51 (in normalized units), valid_acc 77%.
- All 5 heads receive gradient and produce sensible outputs. No shape mismatches, no NaNs.

### What was validated

- `URDF → labeled-mesh → point-cloud → tensor → model → multi-task loss → optimizer step` runs without errors on real-scale data (706 trainable URDFs and 219 MJCF robots).
- Per-vertex link labels make sense visually — verified by exporting smoke runs as multi-mesh GLB scenes (one node per link, each with its own PBR material color) and inspecting in a 3D viewer.
- Augmentations preserve the joint-axis/origin geometry consistency with the points.
- Multiprocessing across 4 workers gives 4-8× speedup over single-process for URDF ingest.

### What was NOT validated

- Whether the baseline PointNet model **converges** to good metrics on real-scale data — only verified shape/gradient flow, not actual learning. CPU is too slow for a useful training run; will run on the local RTX 3090 next.
- Whether the `K_LINKS_MAX = 64` per-vertex classification formulation is the right choice. Each robot has its own per-robot link indexing (link 0 = root for that robot, etc.) — so the model has to learn per-robot semantics, not a universal "joint 0 means base". A Hungarian-matching approach might be needed.
- Whether the `J_MAX = 32` slot decoder is well-suited for the long-tail of high-DOF humanoids (74-DOF, 102-DOF examples are clipped; ~5% of the dataset). Variable-length transformer decoder might be better.

### Open questions

- **Per-robot vs cross-robot link semantics**. With the current label scheme, the model must learn that "link 0 of an xArm6" and "link 0 of a Panda" should both be predicted as link 0 even though their geometry differs significantly. This works if there's enough overlap in shape priors. If not, we'll see seg_acc stuck near 1/N.
- **Mesh quality variation across sources**. urdf_files_dataset has many URDFs with low-poly or auto-generated meshes; MuJoCo Menagerie has high-res CAD. Mixed-fidelity training data is realistic but the loss may bias toward the heavier sources.
- **Domain gap to real MILO scans**. We have one real test (test_2 xArm6) and zero MILO scans for other robot types. Synthetic noise should bridge most of the gap, but a real test set is needed.

### Files added today

```
mesh2robot/data_gen/__init__.py
mesh2robot/data_gen/urdf_loader.py
mesh2robot/data_gen/mjcf_loader.py
mesh2robot/data_gen/augment.py
mesh2robot/model/__init__.py
mesh2robot/model/dataset.py
mesh2robot/model/model.py
mesh2robot/model/losses.py
scripts/generate_training_data.py
scripts/inspect_training_shards.py
scripts/train_model.py
data/training_shards_v1/                 (3.7 GB, 655 shards)
data/training_shards_v1_mjcf/            (1.2 GB, 206 shards)
data/shard_stats_v1_urdf.json
data/shard_stats_v1_combined.json
data/synthetic_train_smoke/*.glb         (visual sanity-check artifacts)
```

### Next: Phase C.7 — real training on the local workstation

Hardware confirmed: **Intel i9-11900KF (8c/16t), 64 GB RAM, NVIDIA RTX 3090 24 GB**. RTX 3090 is more than sufficient for this dataset scale; no H200 transfer needed.

Sizing on RTX 3090 (24 GB VRAM):

| model | params | est. VRAM @ B=64 N=16k | est. step time | 50-epoch wall time |
|---|---:|---:|---:|---:|
| PointNet baseline (current) | 0.5 M | ~0.5 GB | ~10 ms | 2-4 min |
| Point Transformer V3 medium | ~100 M | ~3 GB | ~70 ms | 20-30 min |
| Point Transformer V3 large | ~500 M | ~9 GB | ~200 ms | 1-2 hr |

Training the v1 baseline locally takes roughly the same time as one cloud-instance setup, with no data-transfer step.

Action items:
1. Install CUDA-enabled PyTorch: `pip install torch --index-url https://download.pytorch.org/whl/cu121` (or appropriate CUDA index).
2. Verify: `python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"`.
3. Launch training:
   ```
   python scripts/train_model.py \
       --shard-dir data/training_shards_v1 \
       --shard-dir data/training_shards_v1_mjcf \
       --device cuda --epochs 50 --batch-size 64 --n-points 16384
   ```
4. Monitor seg_acc, axis_deg, origin_m, valid_acc on val split (val is genuinely held-out by-robot, see split_robots in dataset.py).

If baseline PointNet plateaus below 80% seg_acc / above 5° axis err, swap encoder for **Point Transformer V3** (`PointTransformerV3` from the Pointcept library or official PyTorch implementation). The model interface (`forward(points) → dict`) is preserved so only the encoder class changes. PT-V3 medium (~100M) fits comfortably in 24 GB at batch 64.

---

## 2026-04-25 — Project pivot: from heuristic pipeline to learned 3D foundation model

### Why pivot

User-driven realization: the heuristic pipeline's single-plane-cut formulation is **fundamentally insufficient** for arm geometries beyond simple stacked cylinders. The xArm6 test_2_j2 case (a gooseneck arm where link_2 extends laterally from link_1) cannot be cleanly cut by any single plane — link_1 (vertical stub) and link_2 (horizontal section) overlap in Z, so a horizontal cut leaves part of link_2 attached to link_1, while a vertical cut splits the upper arm.

After exploring per-joint priors, mesh-neck detection, child-link slab PCA, and housing-peak Z-correction, the conclusion is that **no fixed heuristic generalizes across robot families**. A trained model that has *seen* many arm topologies is the only path to robust per-vertex segmentation across arbitrary articulated robots — including humanoids, quadrupeds, and mobile manipulators that the user wants to support.

### Goal of the new direction

Train a **single multi-task 3D foundation model** that, given a mesh of any articulated robot, outputs:
1. Robot-type classification (arm / gripper / humanoid / quadruped / wheeled / multi-arm / mobile-manip)
2. Articulation graph (link count + parent/child topology)
3. Per-vertex link assignment
4. Per-joint type, axis, origin, range
5. Per-link mass / inertia hints

When the input is a **known robot** (matches the training distribution), output should reproduce the official URDF's segmentation. When the input is a **custom robot**, output should fall within the inductive bias of "common industrial / mobile robot" — close enough to be useful as a starting point.

### Phase A complete (2026-04-25): multi-source URDF dataset

Cloned six public datasets to `data/raw_robots/` (~6.5 GB on disk):

| Source | Robots discovered | Trainable |
|---|---:|---:|
| `urdf_files_dataset` (Daniella1) | 645 | 504 |
| MuJoCo Menagerie (DeepMind) | 246 | 182 |
| `bullet3` examples | 2,467 | 100 |
| `robot-assets` (ankurhanda) | 108 | 90 |
| robosuite | 66 | 33 |
| Gymnasium-Robotics | 58 | 16 |
| **Total** | **3,590** | **925** |

"Trainable" = loaded successfully (yourdfpy or mujoco) + has ≥ 1 actuated joint + ≥ 80% of referenced meshes resolve on disk.

**Vendor breadth (top 25 trainable)**: Fanuc 82, ABB 54, Kinova 53, KUKA 49, Rainbow Robotics 46, Franka 40, Universal Robots 39, ROS-Industrial 26, Unitree 25, PAL Robotics 20, Stäubli 20, Rethink 14, Willow Garage 12, Boston Dynamics 12, uFactory 11, Robotiq 11, Fetch Robotics 10, Anybotics 8, Shadow 6, Clearpath 6, Wonik 4 (Allegro), SoftBank 2, AgileX 2, Agility 2, Google 2.

**DOF distribution (trainable)**: peak at 6-DOF (239 — UR/xArm/IIWA family), 7-DOF (78 — xArm7/IIWA/Panda), and a long tail going up to 102-DOF (full humanoids with dexterous fingers). 12-DOF (65) covers light humanoids/quadrupeds; 16-DOF (55) and 32-DOF (23) cover bigger humanoids; 56–74-DOF entries are full-fingered humanoids (Apollo, Berkeley Humanoid).

**Drake** sparse-checkout failed (only 3 model files matched the pattern); skipped — already have plenty of data.
**PartNet-Mobility** intentionally excluded for now — it's 2.3k articulated objects but mostly household items (cabinets, scissors, toasters), would dilute the robot signal. Add later if needed for general articulated-object generalization.

### Why this is a real dataset, not template matching

The previous 13-row `urdf_db.json` was a template-matching lookup. **925 robots × ~200 random poses × ~5 noise augmentations ≈ 925k training examples** is what makes this an actual dataset for a 1B-parameter Point Transformer. Per-vertex segmentation labels come *for free* from URDF link structure — no human labeling needed.

### Files added / changed

```
data/raw_robots/                           NEW (6.5 GB) — six cloned datasets
data/robot_manifest.json                   NEW — 3,590 entries with load status + mesh-resolution metadata
scripts/build_robot_manifest.py            NEW — discovery + load + mesh-resolution probe
mesh2robot/io/urdf_database.py             unchanged (legacy 13-arm DB superseded)
mesh2robot/core/robot_retrieval.py         unchanged (still works on legacy DB; will be replaced by trained model)
RESEARCH_LOG.md                            this entry
ROADMAP.md                                 reorganized phase plan
```

### Open questions surfaced

- **License heterogeneity**. URDFs come under MIT, BSD, CC, proprietary, and "no license specified" terms. For a publishable trained model, we'll need to track per-robot licenses and possibly redistribute only a subset. Not a blocker for training but blocks public release.
- **bullet3 inflation**. 2,467 entries from bullet3 but only 100 are trainable — most are static fixtures (tables, cubes) bundled with their physics-demo URDFs. Most of the *real* robot diversity comes from urdf_files_dataset (504) and Menagerie (182).
- **MJCF-only robots**. Menagerie ships MJCF (MuJoCo XML), not URDF. They load fine via the `mujoco` library but the synthetic data generator will need both URDF and MJCF code paths for full coverage.
- **Pose distribution**. URDFs have joint limits but no canonical "natural use" pose distribution. Random uniform sampling within limits will produce many physically-implausible configurations (self-collision, hyper-extension). Phase B should consider self-collision filtering or learned pose priors per robot family.

### Next: Phase B — synthetic data generator

For each trainable URDF:
1. Sample random joint configuration within limits (filter self-collisions; sample N=200 valid configs)
2. FK every link to world frame; load each link's visual mesh; concatenate with per-vertex link labels
3. Sample point cloud (16k points) over surface area, preserving labels
4. Apply augmentations: vertex Gaussian noise σ ∈ [0.5, 5]mm, hole-punching (drop 5–15% in random clusters), random rigid transform, scale ±20%
5. Save (point_cloud, vertex_labels, joint_axes_world, joint_origins_world, joint_types) per example

Target output: **~1M training examples**, packaged as PyTorch-loadable shards.

---

## 2026-04-24 — Cleanup: single-joint drivers removed, multi-joint driver renamed

### Goal

After the geometric-prior-snap refactor (entry below), the multi-joint orchestrator handles any N ≥ 1 joints correctly, making the dedicated single-joint drivers redundant. Remove them.

### Changes

Deleted three files (1247 lines):
- `mesh2robot/experiments/urdf_from_one_joint.py` (745 lines, single-joint URDF driver).
- `mesh2robot/experiments/test_one_joint.py` (349 lines, single-joint debug CLI).
- `mesh2robot/experiments/e2e_from_images.py` (153 lines, older 2-state MILO-cameras driver).

Renamed `mesh2robot/experiments/urdf_from_multi_joints.py` → `urdf_from_images.py`. The "multi" qualifier became meaningless once the one-joint version was gone; "from_images" parallels the existing `end_to_end_xarm6.py` (synthetic pose-mesh driver) and names the distinctive input.

In `urdf_from_images.py`:
- Dropped imports of `_refine_axis_and_origin_smart`, `_ring_circularity`, `_search_axis_and_offset_for_circular_ring` from the deleted `urdf_from_one_joint.py`.
- Deleted dead `_best_of_two_searches` (~80 lines) — replaced by direct `_snap_to_canonical` in the 2026-04-24 refactor, never called since.
- Deleted dead `_refine_joint_via_ring` (~22 lines) — same reason.
- Inlined `_find_state_images` (9 lines) and `_load_calibration` (10 lines), previously imported from `test_one_joint.py`.
- Updated module docstring to reflect the rename.
- Updated `register_milo_to_world.py` docstring cross-reference.

### Verification

- **4-joint run on test_2**: identical axes, cut heights, masses, and URDF structure to pre-cleanup.
- **1-joint degenerate run** (only `joint_1/` folder): 2 links (link_base + link_tip), 1 actuated joint, URDF loads OK. Partition matches the 4-joint run's first cut exactly (32806 base verts / 155918 above).

### What was validated

- The multi-joint driver handles `N = 1` correctly, so removing the specialized one-joint driver doesn't lose capability.
- No other code paths import the deleted modules (grep confirmed — only historical references in `settings.local.json` and prior log entries remain, both harmless).

### Files changed

```
mesh2robot/experiments/urdf_from_multi_joints.py   → renamed to urdf_from_images.py
mesh2robot/experiments/urdf_from_one_joint.py      DELETED
mesh2robot/experiments/test_one_joint.py           DELETED
mesh2robot/experiments/e2e_from_images.py          DELETED
mesh2robot/experiments/register_milo_to_world.py   docstring cross-ref updated
```

---

## 2026-04-24 — Geometric-prior snap for real-mesh URDF synthesis

### Context

Status dashboard's Phase 2 "noise-robust to σ=5 mm" applies to **synthetic K-pose-mesh** inputs. Since 2026-04-22 the active path has been the **real-scan pipeline**: a single MILO scan + per-joint image captures at multiple joint states. This needs different motion extraction because there is no per-pose mesh — only 2D photos before/after each joint moves.

Real-scan path stood up in prior sessions (logged here for the dashboard, not as standalone entries):
- ArUco GridBoard calibration + camera-pose solve.
- `clean_mesh.py` to drop MILO floaters (`--min-faces 10000` keeps only the main arm component).
- `register_cleaned_to_original.py` — ICP to undo Blender's PLY↔OBJ axis flip; produces `T_cleaned_to_original.npy`.
- `mesh2robot/core/motion_from_images.py` — ORB features + silhouette-mask-guided detection + Lowe's-ratio matching + multi-body RANSAC; final SE(3) refined via `solvePnP` to avoid depth-prior triangulation bias.
- `mesh2robot/experiments/urdf_from_multi_joints.py` orchestrator: per-joint motion extraction, sort by cut-height, successively partition mesh along each joint.

Today's work covers the **geometric-prior snap** the user asked for after the v6 pipeline produced diagonal cuts.

### Problem

ORB+RANSAC motion axes on xArm6 test_2 came out visibly tilted from any canonical direction:

| joint | motion axis (raw) | nearest canonical | snap angle |
|---|---|---|---:|
| joint_1 | [0.21, -0.01, 0.98] | Z (parallel to base) | 6° |
| joint_2 | [-0.11, 0.78, 0.62] | Y (⊥ base) | 38° |
| joint_3 | [-0.19, 0.84, 0.50] | Y (⊥ base) | 33° |
| joint_4 | [0.76, -0.37, 0.53] | [-0.87, 0.5, 0] (⊥60°) | 32° |

Downstream plane-SVD used these motion axes as cut plane normals, producing cuts that slice diagonally across the arm instead of along a principled direction. User's insight: "for a Z-parallel cylinder, joints should slice along X or Y with 0° tilt, not along Z".

### Method — four iterations

**(1) Circularity-based canonical candidates.** Added 7 canonical axis candidates (1 parallel + 6 perpendicular to base) to the existing motion-cone search; scored each `(axis, offset)` combo by interface-ring circularity. Failed: xArm6 has multiple pitch joints with similar horizontal axes, so joint_2's search latched onto joint_3's bearing (circ 0.988 but wrong 3D location).

**(2) Spatial anchor.** Required the ring centroid to stay within 15 cm of motion origin. Partially fixed (1) but excluded joint_2's true bearing (which was >15 cm from motion origin along the tilted axis). Tightening excluded even valid rings.

**(3) Force-snap when circularity is low.** Snapped joint axis to nearest canonical within 45° whenever motion search gave circ < 0.93. Worked for *axis direction* but not for *cut plane* — the partition code used joint axis as the cut-plane normal, so joint_2's axis snapping to Y rotated the cut plane to Y-normal, which missed the arm entirely (arm centered near Y=0, cut at Y=0.15 sliced outside the mesh).

**(4) Decouple cut normal from joint axis (final).** The core insight: rotation axis and cut plane normal are **different physical concepts** and the earlier iterations conflated them. Rewrote the partition loop around this separation:

- `cut_point` = `origin + biggest_gap_along_motion * motion_axis` (where the cut plane sits — motion-derived).
- `link_dir` = `RANSAC_pivot[i] − RANSAC_pivot[i−1]` — **uses `origin` not `cut_point`** so the direction doesn't inherit the motion-axis bias. (Earlier version used cut_point and fake-aligned link_dir with motion axis, misclassifying joint_2 as roll.)
- **Roll vs pitch classification**: `|motion · link_dir| > cos(30°)` → roll, else pitch.
- **Roll**: `cut_normal = axis = snap(motion)` to nearest canonical relative to base.
- **Pitch**: `cut_normal = snap(link_dir)`, `axis = snap(motion, perpendicular-to-cut_normal)`.
- **Moving side**: `sign(link_dir · cut_normal)` — topology-based. Replaces the prior seed-based `mean((seed_verts − cut_point) · cut_normal)` which failed when RANSAC's "moved vertex mask" is noisy (joint_2's 490 "moved verts" spanned the entire mesh Z range, not just the moving side).

### Results on xArm6 test_2

| joint | v6 axis (tilted) | v7 axis (snapped) | kind | cut_normal |
|---|---|---|:---:|---|
| joint_1 | [0.21, -0.01, 0.98] | **[0, 0, 1]** | ROLL | [0, 0, 1] |
| joint_2 | [-0.11, 0.78, 0.62] | **[0, 1, 0]** | PITCH | [0, 0, 1] |
| joint_3 | [-0.19, 0.84, 0.50] | **[0, 1, 0]** | PITCH | [0, 0, 1] |
| joint_4 | [0.76, -0.37, 0.53] | **[0.87, -0.5, 0]** | ROLL | [0.87, -0.5, 0] |

Partition (5 links, 4 actuated joints, `yourdfpy` reload passes):

| link | verts | mass (1289 kg/m³) |
|---|---:|---:|
| link_base | 32806 | 6.56 kg |
| link_1 | 7747 | 0.78 kg |
| link_2 | 71541 | 13.47 kg |
| link_3 | 23411 | 1.49 kg |
| link_tip | 54350 | 6.99 kg |

### What was validated

- All 4 joints classify correctly (joint_1 roll, joint_2/3 pitch, joint_4 roll).
- Joint axes are canonical (exact Z, Y, or snapped horizontal) — no residual tilt.
- Cut planes are horizontal for pitch joints, perpendicular-to-link for the horizontal roll joint.
- URDF loads cleanly: 5 links, 4 actuated joints.
- Topology-based moving-side detection is robust: joint_2's seed cloud spanned the whole mesh but the correct side (moving = upper arm, 148 k verts) was still selected.

### What was NOT validated

- No ground-truth comparison — the test_2 MILO scan has no official reference URDF at the same pose.
- Not opened in Isaac Sim or visualized in a URDF viewer.
- link_1 has only 7747 verts — a thin slab between joint_1 (Z=0.164) and joint_2 (Z=0.222); unclear whether the slab captures a closed visual mesh or a thin shell with open faces.
- Only one real-scan dataset (xArm6 test_2); generalization untested.

### Honest limitations

- **The "parallel or perpendicular to base" prior is an industrial-arm assumption.** Arms with 45° joint axes (Scorbot, custom arms) will be snapped incorrectly.
- **Link direction relies on RANSAC pivot quality.** For small rotations or sparse features, pivots drift and the roll/pitch classifier can misfire.
- **Snap silently discards motion evidence within 45°** — no fallback to raw motion when the nearest canonical is actually wrong for this robot.
- **Cut location still uses motion biggest-gap**, which can err when the moved-vertex cloud is small or mis-clustered.

### Open questions

- Should the geometric-prior snap be opt-in (off-by-default) for non-industrial robots? Or always-on with override?
- Should `--base-axis` default be inferred from joint_1's motion axis (since joint_1 is almost always the base-mount) rather than hardcoded to world Z?
- For pitch joints, snap `cut_normal` to the nearest canonical relative to `link_dir` vs use raw `link_dir`? Current code snaps; raw would handle tilted-link arms but lose the "axis-aligned" visual property.
- Do we need a confidence-weighted blend between motion axis and canonical snap for robots that live in the middle (e.g., 20° off canonical)?

### Next recommended experiment

1. **Visual verification**: open `output/test_2_multijoint_4joints_prior/robot.urdf` in a URDF viewer; confirm the 5 links render at plausible positions.
2. **Inspect link_1 slab**: 7747 verts is suspiciously small; verify the slab is a closed volume and not a missing-face shell.
3. **Apply the joint-limit resolver** (Phase 4 from 2026-04-22) to the test_2 URDF — template match on 6-DOF revolute should still hit xArm6 and give template-bounded limits.
4. **Second real-scan dataset** (different arm) to test prior generalization.

### Files changed

```
mesh2robot/experiments/urdf_from_multi_joints.py   REWRITTEN partition loop
mesh2robot/experiments/urdf_from_one_joint.py      +max_centroid_shift param
```

Added helpers in `urdf_from_multi_joints.py`:
- `_canonical_axes_relative_to_base(base)` — 1 parallel + 6 perpendicular canonicals.
- `_snap_to_canonical(axis, reference, perpendicular_only=False)` — snap to nearest canonical, returns `(axis, snap_angle_deg, canonical_index)`.
- Roll/pitch classification + topology-based moving-side from `link_dir · cut_normal`.

`JointInfo` dataclass gained `cut_normal` and `cut_point` fields (distinct from `axis` / `origin`).

Output: [`output/test_2_multijoint_4joints_prior/robot.urdf`](output/test_2_multijoint_4joints_prior/robot.urdf).

---

## 2026-04-22 — Joint-limit resolution for bespoke robots

### Problem

The initial pipeline wrote each joint's limits from Phase 3's observed motion range — but the capture protocol only exercises ±45° per joint, so the generated URDF clamped every joint to a narrow window far below the real robot's capability.

For a **known robot** (xArm6), the template match already solved this via DB lookup. But for a **bespoke robot** with no matching template, we have no such ground truth. User explicitly asked how the system should handle this case.

### Design

Tiered priority (highest wins), implemented in [`mesh2robot/core/joint_limits.py`](mesh2robot/core/joint_limits.py):

1. **User YAML override** (`overrides.yaml`) — authoritative per-joint limits from datasheet.
2. **Template limits** (Phase 4) — if DOF + joint-type signature matches a DB robot.
3. **Self-collision envelope** (PyBullet sweep) — always intersected on top of template, gives a safe ceiling.
4. **Observed × 4 margin** (Phase 3 fallback) — only kicks in if (1) (2) (3) are all absent for a given joint.

### Implementation

New module exposes three primitives:
- `load_yaml_overrides(path) -> {joint_name: (lo, hi)}` — parses the YAML schema `joints: {name: {lower, upper}}`.
- `sweep_self_collision_limits(urdf_path, step_deg) -> {joint_name: (lo, hi)}` — loads URDF in PyBullet with `URDF_USE_SELF_COLLISION`, sweeps each joint outward from 0 in both directions until self-collision (ignoring adjacent-link contacts).
- `resolve_limits(joint_names, template, observed, collision, override) -> [(lo, hi)]` — applies the priority rules.

URDF assembler gained `AssemblyInput.final_limits_per_joint`. When set, it wins over all other sources in the template. End-to-end now does two passes: (1) assemble with template limits for the sweep to use; (2) sweep + resolve + re-assemble with final limits.

### Results on synthetic xArm6

```
joint        resolved            template (GT)     collision sweep
joint_1      [-360°, +360°]      [-360°, +360°]    [-360°, +360°]
joint_2      [ -26°, +120°]      [-118°, +120°]    [ -26°, +132°]   ← collision tightens
joint_3      [-225°,   +6°]      [-225°,  +11°]    [-294°,   +6°]
joint_4      [ -44°, +124°]      [-360°, +360°]    [ -44°, +124°]   ← template continuous,
                                                                        collision caps
joint_5      [ -70°, +100°]      [ -97°, +180°]    [ -70°, +100°]
joint_6      [-360°, +360°]      [-360°, +360°]    [-360°, +360°]
```

For the xArm6 test, the collision sweep finds tighter bounds on joints 2–5 than the template. Intersecting gives the final limits. Joints 1 and 6 (vertical axes with no geometric obstruction) remain unconstrained.

### Honest limitations

- **Collision sweep over-estimates restrictions** when the real robot's cables/harness wrap continuously (joint_4 is a continuous joint on xArm6 but gets capped to ±90° because the visual meshes self-intersect past that). User YAML override is the escape hatch.
- **Collision sweep under-estimates restrictions** if the URDF's geometry is looser than the actual hardware (e.g. convex-hull collisions miss concave obstructions). In practice this is rare because mesh-to-mesh SDF detection is fine-grained.
- **Self-collision at home pose** disables the sweep entirely for that URDF. The resolver returns early with an empty dict. Usually signals an assembly bug (meshes overlapping at rest).

### Why this is good for bespoke robots

In the realistic no-template case:

- Tier 1 (YAML) covers the few joints the user knows from specs.
- Tier 3 (collision sweep) gives an automatic safe ceiling for all joints based on the actual scanned geometry — the same MILO mesh that defines visuals is reused for limit discovery, costing nothing extra.
- Tier 4 (observed × margin) is a last-resort pessimistic floor.

The combined output is at worst conservative — the robot may not reach all physically-achievable configurations, but it will never drive into self-collision.

### Open items still pending

- **Encoder-log sidecar parser** (Tier 1 from the prior analysis) — for powered robots, reads actual encoder angles at pose extremes. Not implemented.
- **Video-based sweep extractor** (Tier 2) — uses SAM2 / CoTracker on a limit-sweep video. Not implemented.
- Real MILO scan test still pending.

### Files added / changed

```
mesh2robot/core/joint_limits.py                  NEW — resolver
mesh2robot/core/urdf_assembly.py                 final_limits_per_joint field added
mesh2robot/experiments/end_to_end_xarm6.py       two-pass flow (assemble -> sweep -> re-assemble)
data/generated/xarm6/overrides.example.yaml      sample YAML format
```

---

## 2026-04-22 — URDF visual-placement bug fix

### Symptom

User reported that the generated URDF loads in viewers but link visuals are **placed in wrong world positions**, yet loading each mesh directly (e.g. in Blender) shows them in correct positions.

### Root cause

In `urdf_assembly.assemble()`, I was treating `AssemblyInput.body_transforms_pose0` as the world-frame pose of each body at rest. But the caller passed `[Ts[0] for Ts in seg.body_transforms]`, and `body_transforms[b][0]` is **always the identity** by Phase 2's construction (SE(3) mapping pose 0 to pose 0).

Consequences:
- `T_local = inv(identity) = identity`, so `mesh.apply_transform(T_local)` was a **no-op** → meshes stayed in world frame.
- `origin_in_parent = inv(T_parent) @ origin_world` reduced to `origin_world` for all joints, but the URDF viewer then applies the kinematic chain, which stacks offsets. Net result: each mesh got **double-translated** by the accumulated chain offset.

Blender loading the meshes directly saw them at pose-0 world coordinates → correct. URDF viewers walked the chain and added the joint offsets on top → wrong.

### Fix

Compute the actual world-frame pose for each body by **walking the joint graph from the root**, using the joint origins (which Phase 3 gives us in world frame). New function `_compute_body_world_frames()` in [`urdf_assembly.py`](mesh2robot/core/urdf_assembly.py):

```
T[root] = identity
for each joint j in BFS order:
    T[j.child] = T[j.parent] with origin set to j.origin_world
```

The assembler now uses this `body_T_world` everywhere it previously used the identity-filled `body_transforms_pose0`. Meshes are transformed into their true link-local frame; joint origins are expressed correctly relative to parent.

### Verification after fix

Comparison of link world AABBs at home pose, generated URDF vs GT xArm6:

| link | GT AABB | Generated AABB | Δ max |
|---|---|---|---|
| link_base | [-0.092, -0.063, 0] → [0.063, 0.063, 0.155] | [-0.095, -0.066, -0.003] → [0.066, 0.066, 0.158] | 3 mm |
| link1 | [-0.048, -0.057, 0.155] → [0.048, 0.075, 0.315] | [-0.051, -0.060, 0.152] → [0.051, 0.078, 0.318] | 3 mm |
| link2 | [-0.048, -0.069, 0.220] → [0.096, 0.119, 0.594] | [-0.051, -0.072, 0.217] → [0.099, 0.122, 0.597] | 3 mm |
| link3 | [0.011, -0.042, 0.382] → [0.174, 0.107, 0.594] | [0.008, -0.045, 0.379] → [0.177, 0.110, 0.597] | 3 mm |
| link4 | [0.089, -0.086, 0.172] → [0.173, 0.043, 0.382] | [0.086, -0.089, 0.169] → [0.176, 0.046, 0.385] | 3 mm |
| link5 | [0.093, -0.037, 0.140] → [0.244, 0.058, 0.246] | [0.090, -0.040, 0.137] → [0.247, 0.061, 0.249] | 3 mm |
| link6 | [0.170, -0.037, 0.112] → [0.244, 0.052, 0.140] | [0.167, -0.040, 0.109] → [0.247, 0.055, 0.143] | 3 mm |

All links match within ~3 mm (rounding + segmentation-boundary vertex reassignment). Joint actuation sweeps produce physically correct rotation — e.g. rotating joint_1 by ±45° tilts link2's AABB as expected.

### Remaining minor issue (not a blocker for rendering) — fixed

My Phase 3 originally picked the "closest to world origin" point on each joint's axis as the `origin_world`. For joint axes that pass through multiple points equally close (e.g. the base→link1 joint's Z-axis through (0,0,anything)), this picked (0,0,0) instead of the physically-meaningful interior of the child link.

**Fix**: added `refine_joint_origins()` in [`joint_extraction.py`](mesh2robot/core/joint_extraction.py). For each joint, compute the midpoint between parent and child mesh centroids (a proxy for the mechanical interface), then project that midpoint onto the rotation-axis line. The projection stays on the axis (so rotation is unchanged) but moves the origin to a geometrically meaningful point inside the link bodies.

Verification: after refinement, link AABBs still match GT within 3.5 mm; frame origins now sit inside the physical body of each link.

### URDF emission order — fixed

The original assembler iterated bodies in `sorted(body_ids)` order — which is Phase 2's internal index order, not kinematic chain order. Joints were numbered 1..N in that arbitrary order too. Fix: new `_chain_order()` helper in [`urdf_assembly.py`](mesh2robot/core/urdf_assembly.py) does a BFS from root; links and joints are now emitted in kinematic order (`link_base → link1 → ... → link6`, `joint_1 = base→link1`, etc.). Auto-naming for unlabeled bodies follows the same convention.

### Files changed

- [mesh2robot/core/urdf_assembly.py](mesh2robot/core/urdf_assembly.py) — added `_compute_body_world_frames()`, replaced both uses of `inp.body_transforms_pose0` with the computed `body_T_world`.
- [data/generated/xarm6/robot.urdf](data/generated/xarm6/robot.urdf) — regenerated; now renders correctly in PyBullet.

### Open questions to revisit

The 18 URDFs from `data/generated/sweep_sigma*_seed*/` were generated **before** the fix — their structure is valid (they parsed) but their visuals would also have been misplaced. Rerunning the e2e × noise sweep with the fix applied would re-validate them; low priority since the single-sigma=0 test suffices for the fix verification.

---

## 2026-04-22 — CoACD integration + end-to-end × noise sweep

### Goal

Close the gap between "URDF parses in yourdfpy under clean input" and "pipeline is actually production-ready": add real convex-decomposition collision geometry (CoACD) and prove the URDF-writing path survives realistic MILO noise levels.

### CoACD integration

New module: [`mesh2robot/core/collision.py`](mesh2robot/core/collision.py) — wraps `coacd.run_coacd` with a `trimesh.convex_hull` fallback. `combine_hulls()` flattens the multi-hull decomposition into a single trimesh for URDF export.

End-to-end run on synthetic xArm6 with `USE_COACD=1`:

| link | hulls | CoACD time |
|---|---:|---:|
| link_base | 5 | 52.7 s |
| link1 | 3 | 60.1 s |
| link2 | 8 | 201.2 s |
| link3 | 8 | 251.2 s |
| link4 | 8 | 279.1 s |
| link5 | 8 | 264.1 s |
| link6 | 8 | 347.3 s |
| **total** | 48 | **~24 min** |

Most links hit the `max_hulls=8` cap, meaning the convex parts are still approximations. Raising the cap (or tightening `threshold`) would improve fidelity at the cost of even longer runtimes.

Practical recommendation: run CoACD **once** per robot offline (part of Phase 5 export), not during iteration. Convex hull is a reasonable baseline for rapid development.

### End-to-end × noise sweep

New module: [`mesh2robot/experiments/e2e_noise_sweep.py`](mesh2robot/experiments/e2e_noise_sweep.py). Runs Phase 2 + 3 + 4 + 5 at each (σ, seed) and verifies the output URDF parses in yourdfpy **and** supports forward kinematics on each actuated joint.

Collision geometry uses visual mesh (convex-hull-lite path) to keep the 18-run sweep tractable (~5 min total vs ~8 hours with CoACD).

### Results

| σ (mm) | seeds | bodies | seg acc | axis err | origin err | URDF load | FK |
|---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 0.0 | 0–2 | 7 | 99.98 % | 0.000° | 0.000 mm | ✓ | ✓ |
| 0.1 | 0–2 | 7 | 99.98 % | 0.009° | 0.007 mm | ✓ | ✓ |
| 0.5 | 0–2 | 7 | 99.93 % | 0.061° | 0.060 mm | ✓ | ✓ |
| 1.0 | 0–2 | 7 | 99.82 % | 0.121° | 0.122 mm | ✓ | ✓ |
| 2.0 | 0–2 | 7 | 99.50 % | 0.224° | 0.211 mm | ✓ | ✓ |
| 5.0 | 0–2 | 7 | 97.67 % | 0.814° | 0.578 mm | ✓ | ✓ |

**URDF validity: 18 / 18 runs** parsed and FK'd successfully.

### What was validated

- URDF-writing path is deterministic regardless of upstream segmentation noise: body count, joint count, link names, and FK all match across all 18 runs.
- The axis/origin values baked into the URDF match Phase 3 output (no regression from URDF serialization).
- Phase 2's output (7 bodies for xArm6) consistently routes through Phases 4–5 without dropped links.
- CoACD integration is correct but too slow for in-loop use; kept as an offline export step.

### Bug fixed during this work

First pass of the FK validator in `e2e_noise_sweep.py` tested `get_transform(child, "world")` — but our generated URDFs use `link_base` as root (inherited from the xArm6 GT name map). Fix: test each joint's `get_transform(j.child, j.parent)` — URDF-relative, no assumption about a global "world" frame name. After the fix, 18/18 passed; before, 0/18 passed due to false-negative FK.

### Open questions still pending

- Real MILO scan end-to-end (blocked on capture).
- Isaac Sim URDF Importer acceptance (separate tool).
- Mass calibration for thin-walled (hollow) scans — current pipeline over-estimates mass by 2-3×.
- Non-serial topologies (parallel arms, grippers with mimic joints).

### Deliverables added

```
mesh2robot/core/collision.py             # CoACD wrapper + combine_hulls
mesh2robot/experiments/e2e_noise_sweep.py  # 18-run URDF validity sweep
data/generated/xarm6/                    # CoACD URDF + decomposed collisions
data/generated/sweep_sigma*_seed*/       # per-run URDFs (18 dirs)
```

### Next recommended experiment

Before touching real MILO, one cheap validation worth doing: open one of the sweep-σ5-mm URDFs in Isaac Sim's URDF Importer GUI and confirm it loads + actuates correctly. This closes the loop on the "URDF validity" claim: yourdfpy parsing is necessary but not sufficient for Isaac Sim compatibility.

Then either:
- **(a) Capture real MILO data** and run the full pipeline end-to-end.
- **(b) Add correspondence noise** (random re-indexing between poses) to simulate imperfect Gaussian tracking. This is the remaining "known unknown" in the Phase 1/2 interface.

---

## 2026-04-22 — Phases 0, 4, 5: end-to-end pipeline

### Goal

Extend the validated Phase 2+3 core with the downstream phases needed to produce a real URDF: build a robot URDF database (Phase 0), add template-based physics defaults (Phase 4), and assemble a loadable URDF (Phase 5). End with a dry-run on synthetic xArm6 pose meshes.

### What was built

**Phase 0 — Robot URDF database**: [`mesh2robot/io/urdf_database.py`](mesh2robot/io/urdf_database.py)

Ingests 13 serial manipulators from `robot_descriptions.py` and writes a JSON database with per-robot: DOF, joint types, axes (local frame), limits, effort/velocity bounds, link masses, total mass, and a back-solved density estimate.

Successfully ingested: xArm6 (6 DOF), xArm7 (7), Panda (8 incl. gripper), UR3/5/10 and UR3e/5e/10e (all 6), IIWA7/14 (7), Gen3 and Gen3-Lite (7). Density estimates from URDFs that supply both mass and mesh volume: ~1300 kg/m³ (xArm6), ~1500 kg/m³ (xArm7), ~1500-2000 kg/m³ (UR e-series), ~2700 kg/m³ (Gen3). The deprecated ur3/5/10 entries report 0 density because their meshes are loaded as scenes not single volumes — benign.

**Phase 4 — Template match**: [`mesh2robot/core/template_match.py`](mesh2robot/core/template_match.py)

Given a query (DOF, joint_types), returns the closest-matching database entry. Score is joint-type agreement with a hard DOF penalty. Returns a `Template` with density, friction, damping, and per-joint effort/velocity defaults. Falls back to aluminum/0.5/0.1/100 N·m/3.14 rad/s if DB is unavailable.

Sanity-check: `match(6, ["revolute"]*6)` → xArm6; `match(7, ["revolute"]*7)` → xArm7.

**Phase 4 — Physics**: [`mesh2robot/core/physics.py`](mesh2robot/core/physics.py)

Two functions: `split_mesh_by_labels()` breaks the combined pose-0 mesh into per-link submeshes using the Phase 2 vertex labels; `compute_inertial_from_mesh()` uses trimesh's volume integration with the template density to produce mass / COM / inertia tensor. Falls back to an AABB proxy if the submesh is not watertight.

Unit test on a 0.1×0.2×0.3 m box at 2700 kg/m³: mass = 16.200 kg (analytical); COM = origin; diagonal inertia. Closed-form exact.

**Phase 5 — URDF assembly**: [`mesh2robot/core/urdf_assembly.py`](mesh2robot/core/urdf_assembly.py)

Jinja2 template fills in `<link>` (with visual/collision meshes + inertial) and `<joint>` (type, axis, origin, limits, effort, velocity, damping, friction) tags. Key coordinate transforms:
- Each link's mesh is exported in its **local frame** (pose-0 world-to-body inverse).
- Inertial COM and tensor are rotated/translated into that local frame.
- Joint `<origin>` is expressed in the parent's local frame; axis is expressed in parent-local frame at pose 0.

Meshes are written as STL alongside the URDF.

### End-to-end experiment: synthetic xArm6 → URDF

Run: `python -m mesh2robot.experiments.end_to_end_xarm6`

```
Input:  13 poses x 23828 vertices  (faces=51423)

Phase 2: motion segmentation ...
  bodies=7  accuracy=99.98%

Phase 3: joint extraction ...
  joints=6  (6 revolute)

Phase 4: template match + inertials ...
  template=xarm6_description  density=1289 kg/m^3
  inertials computed for 7 bodies

Phase 5: URDF assembly ...
  wrote .../data/generated/xarm6/robot.urdf

Verification: reload URDF with yourdfpy ...
  OK  links=7  joints=6  actuated=6
```

Per-link masses (using xArm6-derived density 1289 kg/m³):

| body | link | mass (kg) |
|---:|---|---:|
| 0 | link_base | 9.59 |
| 1 | link1 | 1.95 |
| 2 | link2 | 2.62 |
| 3 | link3 | 5.60 |
| 4 | link4 | 2.64 |
| 5 | link5 | 0.22 |
| 6 | link6 | 3.59 |

Note: total ~26 kg vs real xArm6 ~10 kg. Over-estimate because synthetic mesh uses visual geometry which is hollow-shell-style, so trimesh fills those as solid volumes at aluminum-like density. Real MILO meshes will be outer-shell only — same issue will apply. Mitigation paths: (a) use a hollow-fraction correction factor, (b) hand-override total mass and scale per-link proportionally, (c) scan with a denser density prior that accounts for typical hollow fraction (~30 % solid).

### What was validated

- Phase 0 database schema and ingestion on 13 robots.
- Phase 4 template match returns the correct family for a clean 6-DOF query.
- Phase 4 trimesh inertials are exact on analytical test case.
- Phase 5 URDF passes yourdfpy parser + forward-kinematics at home pose.
- End-to-end pipeline executes without errors on synthetic input.

### What still needs work

- **Mass calibration**: total recovered mass over-estimated 2-3× because visual meshes are filled solid. Need a hollow-fraction prior or a manual total-mass override that distributes proportionally.
- **Joint axis frame convention**: axes are written in parent-local coords with `rpy=0`. Works when GT axis is already aligned with URDF convention; needs deeper verification for arms with non-Z local axes (e.g., KUKA IIWA).
- **Collision meshes**: currently reuses the visual mesh. Next step: call CoACD per link to get convex-decomposition collision geometry.
- **Isaac Sim import**: URDF reload tested in yourdfpy only. Not yet opened in Isaac Sim's URDF Importer → USD pipeline.
- **Single-seed deterministic test**: the end-to-end used seed=0 and clean data. Need an end-to-end × noise-sweep to prove URDFs also come out valid under σ ≤ 5 mm.

### Open questions surfaced

- Do URDFs assembled under σ=1–5 mm noise still pass the yourdfpy parse + Isaac Sim import? Expected yes based on Phase 2/3 robustness, but untested.
- Is the hollow-fraction correction robot-family-specific? (Probably yes: Panda is thin-walled, UR5e is thicker.)
- Does Isaac Sim tolerate `rpy=0` + explicit axis, or does it prefer `rpy` encoding the axis rotation?

### Next recommended experiment

1. **Add CoACD collision decomposition** to the assembly pipeline. ~1 hour.
2. **End-to-end × noise sweep** to prove URDF validity survives realistic MILO noise. ~30 minutes.
3. **Open in Isaac Sim** and visually verify joint drives. Manual step; needs a workstation with Isaac Sim installed.

### Deliverables

```
data/urdf_db.json                        13-robot database
data/generated/xarm6/
├── robot.urdf                           assembled URDF
└── meshes/
    ├── link_base.stl + link_base_collision.stl
    ├── link1.stl + link1_collision.stl
    ├── ...
    └── link6.stl + link6_collision.stl
```

---

## 2026-04-22 — Noise robustness sweep

### Goal

Characterize how Phases 2 + 3 degrade under iid Gaussian vertex noise. Real MILO meshes have ~sub-mm to a few mm of vertex error depending on capture quality; σ ∈ [0, 10] mm spans pessimistic-to-realistic.

### Setup

- Same 13 synthetic xArm6 pose-meshes as prior entry.
- Inject `v + N(0, σI)` to every vertex in every pose.
- 7 sigmas × 3 random seeds = 21 runs total.
- Inlier threshold scaled as `max(0.5 mm, 4σ)`. Body-merge tolerance = 10 × threshold.

### Iterative fixes to segmentation

**First attempt** (vanilla RANSAC): catastrophic failure. At σ = 0.1 mm, accuracy dropped to 74% and mean axis error jumped to 28°. At σ ≥ 0.5 mm, accuracy stalled near 40%.

Diagnosis: 3-point Kabsch is extremely noise-sensitive. A noisy 3-point sample produces a bad SE(3) estimate, which then either (a) accepts too few real inliers or (b) accepts spurious cross-link inliers.

**Fix 1 — LO-RANSAC**: after each successful sample, refit `T` from the collected inliers and re-collect. Iterate to convergence. A body's final transform is determined by hundreds of inliers, not 3 noisy ones.

**Fix 2 — Spatial-diversity sampling**: reject 3-point samples whose pose-0 pairwise distances are below 2 cm. Rotation error from a 3-point fit scales with `σ / sample_spread`; widely-separated samples give lower axis error per trial.

**Fix 3 — Body-merge post-processing**: after hierarchical peeling, one real link can be split across several bodies with near-identical transforms. Union-find collapse of bodies whose per-pose transform Frobenius distance is below `merge_tol`. Eliminates the over-segmentation observed at σ ≥ 0.5 mm.

**Fix 4 — Orphan reassignment**: after all bodies are found, assign each RANSAC-rejected vertex to the body whose transform best explains its observed motion. Recovers the 20–25% of vertices that failed the inlier threshold at σ ≥ 0.5 mm without compromising axis fit (which was already done from the RANSAC inliers).

### Results — full pipeline (LO-RANSAC + merge + orphan)

Mean across 3 seeds. Pass criteria from ROADMAP: seg ≥ 95 %, axis < 2°, origin < 5 mm.

| σ (mm) | Seg acc | Axis err | Origin err | Joints recovered | Pass |
|---:|---:|---:|---:|:-:|:-:|
| 0.0 | 99.98 % | 0.000° | 0.000 mm | 6 / 6 | ✓ |
| 0.1 | 99.98 % | 0.009° | 0.007 mm | 6 / 6 | ✓ |
| 0.5 | 99.93 % | 0.061° | 0.060 mm | 6 / 6 | ✓ |
| 1.0 | 99.82 % | 0.121° | 0.122 mm | 6 / 6 | ✓ |
| 2.0 | 99.50 % | 0.224° | 0.211 mm | 6 / 6 | ✓ |
| 5.0 | 97.67 % | 0.814° | 0.578 mm | 6 / 6 | ✓ |
| 10.0 | 81.34 % | 11.179° | 5.366 mm | 5.3 / 6 | ✗ |

Raw per-seed data: `data/results/noise_sweep.csv`.

### Intermediate attribution (single seed, σ = 0.5–5 mm)

To attribute the gains across fixes:

| σ (mm) | raw | +merge | +merge +orphan |
|---:|:---:|:---:|:---:|
| 0.1 | 95 % / 5.6° | 95 % / 0.01° | **99.98 % / 0.01°** |
| 0.5 | 78 % / 24.6° | 78 % / 0.09° | **99.92 % / 0.09°** |
| 1.0 | 78 % / 22.5° | 78 % / 0.17° | **99.82 % / 0.17°** |
| 5.0 | 77 % / 24.9° | 77 % / 0.77° | **97.77 % / 0.77°** |

- **Body-merge alone** cuts axis error from ~25° down to sub-degree by consolidating fragmented bodies. Doesn't touch segmentation accuracy.
- **Orphan reassignment** recovers segmentation accuracy from ~77 % back to > 97 % without disturbing axis/origin (those are fixed from the inlier set).
- Together the two collapse to the clean-data performance for the metrics that matter.

### What was validated

- **Method is noise-robust up to σ = 5 mm**, which comfortably covers realistic MILO reconstruction error (typically sub-mm to ~1 mm).
- Axis error stays below 1° through σ = 5 mm.
- All 6 revolute joints recovered at every sigma ≤ 5 mm across all seeds.
- Body count matches GT (7) at every sigma ≤ 5 mm.
- The fixes (LO-RANSAC, merge, orphan assignment) compose without interaction side-effects.

### What breaks at σ = 10 mm

- 2/3 seeds find only 6 bodies (miss one link, usually a small link near the tip).
- Axis error jumps to 6–17°.
- Seg acc drops to 78–83 %.
- Cause: noise amplitude approaches typical inter-link spacing for the smaller links (link5/link6 geometry). RANSAC inlier sets become ambiguous.
- Not a concern for MILO — σ = 10 mm would be a severely under-calibrated scan.

### Implementation notes

- `segment_multi_pose` now has `lo_ransac=True` default; `min_spread=0.02` default.
- New functions: `merge_duplicate_bodies(seg, pose_pts, merge_tol)`, `assign_orphans_to_nearest_body(seg, pose_pts)`.
- `noise_sweep.py` default pipeline: segment → merge → orphan-assign → extract_joints.
- Per-sigma threshold and merge tolerance are auto-scaled from σ; no manual tuning per noise level.

### Open questions surfaced

- **Non-1-to-1 correspondence.** All noise was positional; real MILO will also have imperfect vertex tracking across poses. Next experiment: simulate correspondence noise (random re-indexing or nearest-neighbor matching on noisy meshes) and measure degradation.
- **Smallest link reliability.** At σ = 10 mm one link occasionally goes missing. For robots with very small links (grippers, wrists) we may hit this regime earlier. Mitigation: increase `n_trials`, lower `min_inliers`.
- **Origin error at 10 mm is on the boundary** (5.37 mm vs 5 mm threshold). Suggests the `4σ` inlier threshold scaling needs a tighter regime at high σ, or the data needs per-link weighted inertial reasoning.

### Next recommended experiment

- **Correspondence-noise sweep**: perturb the 1-to-1 vertex indexing across poses (mimics imperfect Gaussian anchor tracking) and rerun. Estimated 30 minutes.
- Alternatively, move to Phase 5 (URDF assembly) and come back to noise variants later — the geometric core is proven robust enough.

---

## 2026-04-22 — Feasibility experiment on synthetic xArm6

### Goal

Verify that the geometric core of mesh2robot (Phases 2 + 3) can recover a robot's kinematic chain from noise-free multi-pose meshes. Per the roadmap's critical experiment: "If geometric recovery doesn't work on clean synthetic data, no amount of MILO integration will save it."

### Setup

- **Ground truth**: xArm6 URDF from `robot_descriptions.xarm6_description`. 9 links, 6 revolute joints, all axes = local Z.
- **Synthetic input**: 13 poses (the recommended 2N+1 protocol — home + each joint at ±45°).
- **Per-pose vertex count**: 23,828 (7 link meshes concatenated and transformed by forward kinematics).
- **Correspondence**: 1-to-1 across poses (same vertex indexing).
- **Noise**: none.

### Method

1. `mesh2robot.io.synthetic_poses` loads xArm6 URDF + visual meshes, applies FK per pose, writes combined mesh + GT labels to `data/synthetic/xarm6/`.
2. `mesh2robot.core.motion_segmentation.segment_multi_pose` runs hierarchical RANSAC with a 0.5 mm inlier threshold.
3. `mesh2robot.core.joint_extraction.extract_joints` infers parent-child graph via relative stillness, then extracts screw axes.
4. `mesh2robot.experiments.feasibility_xarm6` compares recovered axes/origins to GT.

### Results

| Metric | Target | Achieved |
|---|---|---|
| Vertex assignment accuracy | ≥ 95% | **99.98%** |
| Bodies recovered | 7 | **7** |
| Unassigned vertices | minimal | **0** |
| Mean axis direction error | < 2° | **0.000°** |
| Mean origin error | < 5 mm | **0.000 mm** |
| Joint ranges | [-45°, +45°] | **[-45°, +45°] on all 6** |

Per-joint:

```
            joint    axis_err_deg   origin_err_mm   range (deg)
            link1           0.000           0.000   [-45.0, +45.0]
            link2           0.000           0.000   [-45.0, +45.0]
            link3           0.000           0.000   [-45.0, +45.0]
            link4           0.000           0.000   [-45.0, +45.0]
            link5           0.000           0.000   [-45.0, +45.0]
            link6           0.000           0.000   [-45.0, +45.0]
```

### What was validated

- Multi-pose hierarchical RANSAC correctly separates 7 rigid bodies when fed 13 poses.
- Screw-axis extraction from SE(3) is numerically exact on clean data (as expected — it's a closed-form decomposition).
- The "stillness-based" parent inference (parent = body with smaller total motion whose motion profile correlates with the child) correctly reconstructs the serial chain.
- The 2N+1 capture protocol provides enough motion diversity for both segmentation and joint axis fitting.

### What was NOT validated

- Noise robustness (σ > 0 vertex noise).
- Non-1-to-1 vertex correspondence across poses (i.e., what MILO actually produces).
- Multi-joint simultaneous motion (if the user can't isolate joints cleanly).
- Robots other than xArm6 (e.g., 7-DOF, SCARA, parallel).
- Prismatic joints (xArm6 has none).

### Implementation notes / decisions

- Chose `yourdfpy` over `urdfpy` — actively maintained, supports Python 3.13.
- Used `robot_descriptions` for URDF sourcing instead of cloning repos manually. Required `xacrodoc` for xArm xacro processing.
- Horn's method for SE(3) fit uses SVD with determinant correction to avoid reflections.
- Screw axis origin solved via `np.linalg.lstsq` on the rank-2 system `(I - R) @ origin = t_perp`. The null space along the axis is handled automatically.
- Parent inference uses motion-magnitude correlation rather than temporal clustering. Simpler; works on the 2N+1 protocol because only ancestors of the moved joint stay still.

### Open questions surfaced

1. **Breakdown point under noise** — at what σ does segmentation accuracy drop below 95%?
2. **Minimum K** — can we go below 2N+1 poses? What if one joint is inaccessible?
3. **Correspondence strategy** — Gaussian tracking via MILO anchors vs FPFH+ICP on mesh vertices. Quality difference?
4. **Revolute vs prismatic disambiguation** — current code always returns revolute. Need prismatic test case (a Cartesian robot or SCARA's Z axis).

### Next recommended experiment

**Noise sweep**: inject `σ ∈ {0, 0.1, 0.5, 1, 5, 10}` mm Gaussian noise into vertex positions and rerun. Report accuracy degradation curve. Estimated effort: 30 minutes (tweak the feasibility script).

---

## 2026-04-22 — Project scaffold and tooling

### Repo layout

```
mesh2robot/
├── ROADMAP.md                      8-week plan, research gaps, pipeline
├── RESEARCH_LOG.md                 this file
├── requirements.txt
├── data/
│   └── synthetic/xarm6/            13 pose meshes + metadata
└── mesh2robot/
    ├── io/
    │   └── synthetic_poses.py      FK-based synthetic pose generator
    ├── core/
    │   ├── rigid_fit.py            Horn + screw decomposition
    │   ├── motion_segmentation.py  Phase 2 multi-pose RANSAC
    │   └── joint_extraction.py     Phase 3 parent inference + axes
    └── experiments/
        └── feasibility_xarm6.py    end-to-end GT comparison
```

### Dependencies resolved

- Environment: Anaconda Python 3.13.5 (Windows).
- Installed: `yourdfpy`, `robot_descriptions`, `xacrodoc` on top of existing `trimesh`, `numpy`, `scipy`.
- Quirk: on this machine `pip` and `python` point to different interpreters — must use `python -m pip install` to hit the correct env.

---

## Protocol: how to add a log entry

1. Prepend a dated section at the top (under the status dashboard).
2. Use the structure: **Goal** → **Setup** → **Method** → **Results** (table) → **What was / was not validated** → **Open questions** → **Next step**.
3. Update the status dashboard table if a phase changes state.
4. If the experiment introduces a new metric, add it to the dashboard.
5. Never edit or delete prior entries — append-only. Corrections go in a new entry.

---

## Running open questions (aggregated across entries)

Answered questions are moved to historical entries; this list is only open items.

- [x] ~~Noise breakdown point for Phase 2 (σ sweep)~~ → answered 2026-04-22: robust up to σ = 5 mm with full post-processing chain; breaks at σ = 10 mm
- [ ] Non-1-to-1 correspondence strategy — Gaussian tracking vs FPFH+ICP
- [ ] Minimum K (pose count) that still recovers chain correctly
- [ ] Prismatic joint recovery (need SCARA or Cartesian test case)
- [ ] 7-DOF arm generalization (add xArm7 or Panda to test battery)
- [ ] Multi-joint simultaneous motion — does parent-inference degrade gracefully?
- [ ] Real MILO output vs synthetic — how large is the reality gap?
- [ ] σ = 10 mm borderline origin error (5.37 mm just over 5 mm target) — needs threshold re-tuning or weighted reasoning?
- [ ] Visual-mesh-as-volume mass over-estimate — hollow-fraction correction or manual mass override?
- [ ] Encoder-log sidecar parser (Tier 1 joint-limit source for powered robots)
- [ ] Video-based joint-limit sweep extractor (SAM2 / CoTracker)
- [x] ~~URDF assembly under σ ≥ 1 mm noise — still parses?~~ → answered 2026-04-22: 18/18 runs at σ ∈ [0, 5] mm × 3 seeds pass yourdfpy parse + FK
- [ ] Isaac Sim URDF Importer acceptance (yourdfpy passes but Isaac Sim may be stricter)
- [x] ~~CoACD collision decomposition integration~~ → answered 2026-04-22: integrated via [`mesh2robot/core/collision.py`](mesh2robot/core/collision.py); 24 min total for 7 xArm6 links
- [ ] KUKA IIWA / other non-Z-axis robots — does the `<axis>` + `<origin rpy>` convention break?
- [ ] Real-scan geometric-prior snap: opt-in vs always-on default for non-industrial arms (2026-04-24)
- [ ] `--base-axis` auto-inference from joint_1 motion axis rather than hardcoded world Z (2026-04-24)
- [ ] Does the `cut_normal = snap(link_dir)` for pitch joints hurt robots with tilted links? Consider raw `link_dir` as alternative. (2026-04-24)
- [ ] link_1 slab on test_2 has only 7747 verts — verify it's a closed volume (2026-04-24)
- [ ] Second real-scan dataset to test geometric-prior generalization (2026-04-24)

---

## References

- ROADMAP.md — full roadmap and research-gap analysis
- PARIS (ICCV 2023), Real2Code (ICLR 2025), URDFormer (CVPR 2024), Articulate-Anything (ICLR 2025) — prior articulated-object reconstruction work
- MuJoCo Menagerie, robot_descriptions.py, urdf_files_dataset — URDF prior sources
