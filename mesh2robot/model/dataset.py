"""PyTorch Dataset for the synthetic robot training shards (Phase B output).

Each shard `.npz` packs B examples with arrays:
  - points (B, N, 3) float32                   surface points (world)
  - point_labels (B, N) int32                  per-point link index
  - joint_axes_world (B, J_max, 3) float32     padded
  - joint_origins_world (B, J_max, 3) float32  padded
  - joint_types (B, J_max) int32               padded (-1 for invalid)
  - joint_topology (B, J_max, 2) int32         (parent, child) link indices
  - joint_valid (B, J_max) bool                mask: 1 if joint exists
  - robot_idx (B,) int32                       index into shard's `names`
  - names (R,) str                             one per unique robot in shard

Per-shard J_max varies. To assemble batches across shards we re-pad to a
GLOBAL max joint count `J_MAX`, and likewise truncate/pad link labels to
the global max link count `K_LINKS_MAX`. Both bounds are conservative —
real robots in the manifest stay well under them.

Each `__getitem__` returns ONE example (the i-th of (shard_id, intra_idx)
mapping). The collate fn handles batching and ensures all tensors are
padded to (J_MAX,) and (N,) consistently.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset


# Global caps. Anymal has 22 links, Fetch 19, full humanoids ~30. Set
# generously so we don't truncate.
K_LINKS_MAX = 64
J_MAX = 32


class ShardDataset(Dataset):
    """Reads pre-packed .npz shards as a flat dataset of training examples.

    Parameters
    ----------
    shard_paths
        List of .npz shard files to consume. Each shard's example count is
        learned by inspection (no metadata file needed).
    n_points
        Number of points per example. Shards must store at least this many;
        if more, we randomly subsample per __getitem__ for augmentation.
    augment_subsample
        If True, randomly pick `n_points` of the available points each
        epoch (recommended for training). If False, take the first
        `n_points`.
    normalize
        If True (default), each example is centered + scaled so its 99th-
        percentile point radius is 1. Robust to outlier vertices in
        malformed URDFs. Joint origins are transformed identically.
    rotate_aug
        If True (default), apply a random yaw rotation per __getitem__,
        applied identically to points, joint axes, joint origins.
        Trains the model to be Z-rotation invariant.
    robot_filter
        Optional set of robot names. If provided, only examples whose
        source robot is in this set are emitted. Used to enforce a
        BY-ROBOT train/val split — without it, a 90/10 random shard
        split puts the same robot in both train and val (because each
        robot's examples are scattered across many shards), which
        contaminates the validation signal.
    """

    def __init__(
        self,
        shard_paths: Sequence[Path | str],
        n_points: int = 16384,
        augment_subsample: bool = True,
        normalize: bool = True,
        rotate_aug: bool = True,
        robot_filter: set[str] | None = None,
        in_memory: bool = False,
        verbose: bool = False,
    ) -> None:
        self.shard_paths = [Path(p) for p in shard_paths]
        self.n_points = n_points
        self.augment_subsample = augment_subsample
        self.normalize = normalize
        self.rotate_aug = rotate_aug
        self.robot_filter = robot_filter
        self.in_memory = in_memory
        self.verbose = verbose

        # Build a flat (shard_idx, intra_idx) index across all shards,
        # filtered to allowed robots when `robot_filter` is given.
        self._index: list[tuple[int, int]] = []
        # Disk-cache (LRU) when in_memory=False; full preload dict when True.
        self._cache: dict[int, dict] = {}
        self._cache_order: list[int] = []
        self._cache_size = 4

        from time import time as _t
        t0 = _t()
        for si, p in enumerate(self.shard_paths):
            if in_memory:
                # Materialize the entire shard into the persistent cache.
                with np.load(p, allow_pickle=True) as z:
                    data = {k: z[k] for k in z.files}
                self._cache[si] = data
                names = list(data["names"])
                robot_idx = data["robot_idx"]
                B = data["points"].shape[0]
            else:
                with np.load(p, allow_pickle=True) as z:
                    names = list(z["names"])
                    robot_idx = z["robot_idx"]
                    B = z["points"].shape[0]
            for j in range(B):
                if robot_filter is not None:
                    rname = str(names[int(robot_idx[j])])
                    if rname not in robot_filter:
                        continue
                self._index.append((si, j))
        if in_memory and verbose:
            mb = sum(
                sum(v.nbytes for v in s.values() if hasattr(v, "nbytes"))
                for s in self._cache.values()
            ) / 1e6
            print(f"  [ShardDataset] loaded {len(self.shard_paths)} shards "
                  f"({mb:.0f} MB) into memory in {(_t() - t0):.1f}s")

    def __len__(self) -> int:
        return len(self._index)

    def _get_shard(self, shard_idx: int) -> dict:
        if shard_idx in self._cache:
            return self._cache[shard_idx]
        # Disk path — only reachable when in_memory=False.
        z = np.load(self.shard_paths[shard_idx], allow_pickle=True)
        data = {k: z[k] for k in z.files}
        z.close()
        self._cache[shard_idx] = data
        self._cache_order.append(shard_idx)
        if len(self._cache_order) > self._cache_size:
            evict = self._cache_order.pop(0)
            del self._cache[evict]
        return data

    def __getitem__(self, idx: int) -> dict:
        shard_idx, intra_idx = self._index[idx]
        shard = self._get_shard(shard_idx)
        pts = shard["points"][intra_idx]            # (N, 3)
        labels = shard["point_labels"][intra_idx]   # (N,)
        axes = shard["joint_axes_world"][intra_idx]    # (J, 3)
        origins = shard["joint_origins_world"][intra_idx] # (J, 3)
        types = shard["joint_types"][intra_idx]     # (J,)
        topo = shard["joint_topology"][intra_idx]   # (J, 2)
        valid = shard["joint_valid"][intra_idx]     # (J,)
        # joint_limits is v3+; v1/v2 shards lack it, so fall back to zeros.
        # Loss is masked by joint_valid AND a separate has_limits flag below,
        # so legacy shards safely contribute nothing to limits training.
        if "joint_limits" in shard:
            jlimits = shard["joint_limits"][intra_idx]   # (J, 2)
            has_limits = True
        else:
            jlimits = np.zeros((axes.shape[0], 2), dtype=np.float32)
            has_limits = False

        # Subsample to n_points
        if pts.shape[0] >= self.n_points:
            if self.augment_subsample:
                idx_sub = np.random.choice(pts.shape[0], self.n_points, replace=False)
            else:
                idx_sub = np.arange(self.n_points)
            pts = pts[idx_sub]
            labels = labels[idx_sub]
        else:
            # pad by repeat (rare for our generator)
            pad = np.random.choice(pts.shape[0], self.n_points - pts.shape[0], replace=True)
            pts = np.concatenate([pts, pts[pad]], 0)
            labels = np.concatenate([labels, labels[pad]], 0)

        # Truncate/pad labels to K_LINKS_MAX. Out-of-bound link indices get
        # collapsed to "ignore" sentinel -1 — we never expect this in
        # well-formed data, but be defensive.
        labels = labels.astype(np.int64)
        labels[labels >= K_LINKS_MAX] = -1
        labels[labels < 0] = -1

        # Pad joint arrays to global J_MAX
        J = axes.shape[0]
        if J > J_MAX:
            axes = axes[:J_MAX]
            origins = origins[:J_MAX]
            types = types[:J_MAX]
            topo = topo[:J_MAX]
            valid = valid[:J_MAX]
            jlimits = jlimits[:J_MAX]
        elif J < J_MAX:
            pad_n = J_MAX - J
            axes = np.concatenate([axes, np.zeros((pad_n, 3), dtype=np.float32)], 0)
            origins = np.concatenate([origins, np.zeros((pad_n, 3), dtype=np.float32)], 0)
            types = np.concatenate([types, np.full(pad_n, -1, dtype=np.int32)], 0)
            topo = np.concatenate([topo, np.full((pad_n, 2), -1, dtype=np.int32)], 0)
            valid = np.concatenate([valid, np.zeros(pad_n, dtype=bool)], 0)
            jlimits = np.concatenate([jlimits, np.zeros((pad_n, 2), dtype=np.float32)], 0)
        # Truncate topology link indices that exceed K_LINKS_MAX
        topo[topo >= K_LINKS_MAX] = -1

        # Normalize point cloud to unit ball using 99th-percentile radius
        # (robust to outlier verts in malformed URDFs — some examples had
        # a single vertex at 160m, which would crush the rest of the
        # cloud near zero if we used max). Joint origins follow the same
        # affine transform; axes are unitless directions and don't change.
        if self.normalize:
            centroid = pts.mean(axis=0)
            pts = pts - centroid
            radii = np.linalg.norm(pts, axis=1)
            scale = float(np.percentile(radii, 99)) + 1e-8
            pts = pts / scale
            origins = (origins.astype(np.float32) - centroid) / scale
        else:
            origins = origins.astype(np.float32)

        # Z-axis rotation augmentation: random yaw, applied identically to
        # points + joint axes + joint origins (axes rotate too because
        # we want yaw-invariance, not yaw-equivariance).
        if self.rotate_aug:
            theta = float(np.random.uniform(-np.pi, np.pi))
            c, s = np.cos(theta), np.sin(theta)
            R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]],
                         dtype=np.float32)
            pts = pts @ R.T
            axes = axes.astype(np.float32) @ R.T
            origins = origins @ R.T

        return {
            "points": torch.from_numpy(np.ascontiguousarray(pts.astype(np.float32))),
            "point_labels": torch.from_numpy(labels),
            "joint_axes": torch.from_numpy(np.ascontiguousarray(axes.astype(np.float32))),
            "joint_origins": torch.from_numpy(np.ascontiguousarray(origins.astype(np.float32))),
            "joint_types": torch.from_numpy(types.astype(np.int64)),
            "joint_topology": torch.from_numpy(topo.astype(np.int64)),
            "joint_valid": torch.from_numpy(valid.astype(np.bool_)),
            "joint_limits": torch.from_numpy(np.ascontiguousarray(jlimits.astype(np.float32))),
            "has_limits": torch.tensor(has_limits, dtype=torch.bool),
        }


def collate_examples(batch: list[dict]) -> dict:
    """Default collate: stacks tensors along batch dim 0."""
    out: dict = {}
    for k in batch[0]:
        out[k] = torch.stack([b[k] for b in batch], dim=0)
    return out


def enumerate_robots(shard_paths: Sequence[Path | str]) -> dict[str, int]:
    """Walk shards once and return {robot_name: example_count}.

    Used to plan a by-robot train/val split: the caller picks robot names
    for each split, then constructs ShardDataset instances with
    `robot_filter=` set to those names.
    """
    counts: dict[str, int] = {}
    for p in shard_paths:
        with np.load(Path(p), allow_pickle=True) as z:
            names = list(z["names"])
            robot_idx = z["robot_idx"]
        for ri in robot_idx:
            rname = str(names[int(ri)])
            counts[rname] = counts.get(rname, 0) + 1
    return counts


def split_robots(
    shard_paths: Sequence[Path | str],
    val_frac: float = 0.1,
    seed: int = 0,
) -> tuple[set[str], set[str]]:
    """Split unique robots in the given shards into train/val sets.

    A robot's ALL examples go to the same split, so val truly measures
    generalization to unseen robots.
    """
    counts = enumerate_robots(shard_paths)
    robots = sorted(counts.keys())
    if not robots:
        raise ValueError("No robots found in shards")
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(robots))
    n_val = max(1, int(round(len(robots) * val_frac)))
    val_robots = {robots[i] for i in perm[:n_val]}
    train_robots = {robots[i] for i in perm[n_val:]}
    return train_robots, val_robots


def stratified_split_canonical(
    research_manifest_path: Path | str,
    shard_paths: Sequence[Path | str],
    val_frac: float = 0.1,
    seed: int = 0,
) -> tuple[set[str], set[str]]:
    """Stratified by-robot train/val split over canonical robots.

    Strata are the cross of `fidelity_class` (high/medium/low/unknown) and
    a DOF bucket (1-6, 7-12, 13+). Each cell contributes
    `round(cell_size * val_frac)` robots to val (at least 1 if cell has
    >= 2 robots), guaranteeing val sees every cell.

    The result is intersected with shard names to drop canonical robots
    whose Phase B generation failed (3 missing entries as of writing).

    Why stratified: a random by-robot split with a fresh seed sometimes
    lands all high-fidelity Menagerie humanoids in train and the val set
    becomes mostly low-fidelity bullet3 arms — making the val number
    over-optimistic relative to "the typical input the model will see
    in deployment". Stratification gives every (fidelity, DOF) cell at
    least one held-out example.
    """
    import json
    research_manifest_path = Path(research_manifest_path)
    entries = json.loads(research_manifest_path.read_text())
    canonical = [e for e in entries if e.get("canonical_train_set")]

    # Canonical robot names in BOTH legacy + new shard formats. Patched
    # v2 shards use `<source>/<parent>/<stem>`; v1 shards use
    # `<source>/<stem>`. Returning both lets the intersection with shard
    # names succeed regardless of which format the shards use.
    def names_for(e: dict) -> list[str]:
        p = Path(e["path"])
        return [
            f"{e['source']}/{p.parent.name}/{p.stem}",   # v2/new
            f"{e['source']}/{p.stem}",                    # v1/legacy
        ]

    # DOF buckets: small arms / mid (humanoid limbs, light humanoids) / heavy humanoids+hands
    def dof_bucket(dof: int) -> str:
        if dof <= 6:
            return "1-6"
        if dof <= 12:
            return "7-12"
        return "13+"

    # Filter to canonical robots that actually appear in shards. Try
    # the v2 (new) name first, fall back to v1 (legacy).
    available = set(enumerate_robots(shard_paths).keys())
    pool: list[tuple[str, dict]] = []
    for e in canonical:
        for n in names_for(e):
            if n in available:
                pool.append((n, e))
                break

    # Group into strata
    strata: dict[tuple, list[str]] = {}
    for name, e in pool:
        key = (e.get("fidelity_class") or "unknown", dof_bucket(e.get("dof", 0)))
        strata.setdefault(key, []).append(name)

    rng = np.random.default_rng(seed)
    val_robots: set[str] = set()
    train_robots: set[str] = set()
    for key, names in sorted(strata.items()):
        names_arr = np.array(sorted(names))  # sorted for reproducibility
        rng.shuffle(names_arr)
        n = len(names_arr)
        if n >= 2:
            n_val = max(1, int(round(n * val_frac)))
        else:
            n_val = 0  # singletons stay in train
        val_robots.update(names_arr[:n_val].tolist())
        train_robots.update(names_arr[n_val:].tolist())

    return train_robots, val_robots


def load_canonical_robot_names(
    research_manifest_path: Path | str,
    legacy_v1_naming: bool = True,
) -> set[str]:
    """Load the set of canonical robot names from the research manifest.

    Names match what `generate_training_data.py` writes into shards.
    The naming convention changed: v1 shards use `<source>/<stem>` (which
    conflates 21 robots whose stems collide — e.g. `robosuite/robot`
    spans 9 distinct robots). New shards use `<source>/<parent>/<stem>`.
    With `legacy_v1_naming=True` (default), this returns BOTH formats so
    the canonical filter works against existing v1 shards AND any new
    regenerated ones.
    """
    import json
    research_manifest_path = Path(research_manifest_path)
    entries = json.loads(research_manifest_path.read_text())
    names: set[str] = set()
    for e in entries:
        if not e.get("canonical_train_set"):
            continue
        path = e.get("path", "")
        if not path:
            continue
        p = Path(path)
        # New convention: source/parent_dir/stem
        names.add(f"{e['source']}/{p.parent.name}/{p.stem}")
        if legacy_v1_naming:
            # v1 convention: source/stem (kept for backward compat)
            names.add(f"{e['source']}/{p.stem}")
    return names
