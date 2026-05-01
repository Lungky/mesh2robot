"""Phase 2 — match a captured robot's geometry against the URDF database.

The user does NOT need to hint the robot type. Given:
  - K captured joint axes in world frame (snapped to canonical)
  - Their measured world-frame origins
  - The user mesh's overall AABB

we score every database entry on:
  1. Joint axis pattern match for the captured joints (lower angle error = better)
  2. Joint Z-fraction match — where each joint sits along the arm's height
     as a fraction (penalizes a robot whose joint_1 is 70% up vs ours at 17%)
  3. DOF preference (prefer dof ≥ captured_dof, penalize big gaps)

The top scorer is returned with a confidence label based on the margin over
the second-best candidate. Downstream code uses the matched reference's
joint axes/origins as a soft prior for transfer — but only when confidence
is "high" or "medium". Low-confidence (or no-candidate) results trigger the
custom/distributional-prior path instead.

The user can override retrieval with `hint_name=...` (e.g. for evaluation
or when scanning a new robot whose match is known a priori).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


DEFAULT_DB = Path(__file__).resolve().parents[2] / "data" / "urdf_db.json"


@dataclass
class MatchResult:
    record: dict | None       # the matched DB entry, or None if no match
    score: float              # combined score in [0, 1]; higher = better
    confidence: str           # "high" | "medium" | "low" | "none"
    rationale: list[str]      # human-readable scoring breakdown
    margin: float = 0.0       # gap to second-best candidate

    @property
    def name(self) -> str:
        return self.record["name"] if self.record else "<no-match>"


def load_db(path: Path | str = DEFAULT_DB) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    return json.loads(p.read_text())


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

def _axis_angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    """Angle between two axis directions, sign-invariant (axis ↔ -axis)."""
    a = a / (np.linalg.norm(a) + 1e-12)
    b = b / (np.linalg.norm(b) + 1e-12)
    cos = abs(float(a @ b))
    return float(np.rad2deg(np.arccos(np.clip(cos, -1.0, 1.0))))


def _score_axis_pattern(
    captured_axes_world: list[np.ndarray],
    ref_axes_world: list[list[float]],
) -> tuple[float, float]:
    """Score = 1 - mean(angle_error)/90°. Returns (score, mean_err_deg)."""
    K = len(captured_axes_world)
    if K == 0 or len(ref_axes_world) < K:
        return 0.0, 90.0
    errs = [
        _axis_angle_deg(np.asarray(captured_axes_world[i]),
                        np.asarray(ref_axes_world[i]))
        for i in range(K)
    ]
    mean_err = sum(errs) / K
    return max(0.0, 1.0 - mean_err / 90.0), mean_err


def _score_z_fraction(
    captured_z_fractions: list[float] | None,
    ref_z_fractions: list[float],
) -> tuple[float, float]:
    """Score = 1 - mean(|frac_diff|)/0.5 (clipped). Returns (score, mean_err)."""
    if not captured_z_fractions:
        return 0.5, math.nan       # no data → neutral score
    K = len(captured_z_fractions)
    if len(ref_z_fractions) < K:
        return 0.0, 1.0
    errs = [abs(captured_z_fractions[i] - ref_z_fractions[i]) for i in range(K)]
    mean_err = sum(errs) / K
    return max(0.0, 1.0 - mean_err / 0.5), mean_err


def _score_dof(captured_K: int, ref_dof: int) -> float:
    """Prefer DOF ≥ captured_K. Smaller gap is better."""
    if ref_dof < captured_K:
        return 0.0
    diff = ref_dof - captured_K
    return 1.0 / (1.0 + 0.15 * diff)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def match_robot(
    captured_axes_world: list[np.ndarray],
    captured_z_fractions: list[float] | None = None,
    db: list[dict] | None = None,
    db_path: Path | str | None = None,
    hint_name: str | None = None,
    weights: dict[str, float] | None = None,
) -> MatchResult:
    """Match captured joint geometry against the URDF database.

    Parameters
    ----------
    captured_axes_world
        Per-captured-joint axis direction in world frame, snapped to canonical.
    captured_z_fractions
        Per-captured-joint cut Z as fraction of total user-mesh height. Used
        to discriminate robots with similar axis patterns but different
        elbow/wrist heights (e.g. xArm6 vs UR5e).
    db, db_path
        Provide one or the other; defaults to data/urdf_db.json.
    hint_name
        If set, force-match by exact `name`. Returns confidence="hinted".
    weights
        Optional override of {"axis": 0.5, "zfrac": 0.3, "dof": 0.2}.
    """
    if db is None:
        db = load_db(db_path or DEFAULT_DB)

    if hint_name:
        for rec in db:
            if rec.get("name") == hint_name:
                return MatchResult(rec, 1.0, "hinted",
                                   [f"User-hinted: {hint_name}"], margin=1.0)
        return MatchResult(None, 0.0, "none",
                           [f"Hint {hint_name!r} not in DB"], margin=0.0)

    K = len(captured_axes_world)
    if K == 0:
        return MatchResult(None, 0.0, "none",
                           ["No captured joint axes provided"], margin=0.0)

    w = {"axis": 0.5, "zfrac": 0.3, "dof": 0.2}
    if weights:
        w.update(weights)

    rated: list[tuple[float, dict, list[str]]] = []
    for rec in db:
        if rec.get("dof", 0) == 0:
            continue
        if rec.get("total_height", 0.0) <= 0.0:
            # Broken record (mesh failed to load). Skip.
            continue

        ax_score, ax_err = _score_axis_pattern(
            captured_axes_world, rec.get("joint_axes_world", []),
        )
        zf_score, zf_err = _score_z_fraction(
            captured_z_fractions, rec.get("joint_z_fractions", []),
        )
        dof_score = _score_dof(K, rec.get("dof", 0))
        total = (w["axis"] * ax_score
                 + w["zfrac"] * zf_score
                 + w["dof"] * dof_score)

        why = [
            f"axis: mean_err={ax_err:.1f}°  → {ax_score:.2f}",
            f"zfrac: mean_err={zf_err:.3f}  → {zf_score:.2f}"
                 if not math.isnan(zf_err) else f"zfrac: -- (no input)",
            f"dof: {rec['dof']} vs K={K}  → {dof_score:.2f}",
        ]
        rated.append((total, rec, why))

    if not rated:
        return MatchResult(None, 0.0, "none",
                           ["No DB record passed filters"], margin=0.0)

    rated.sort(key=lambda r: -r[0])
    top_score, top_rec, top_why = rated[0]
    margin = top_score - rated[1][0] if len(rated) >= 2 else top_score

    if margin >= 0.10:
        confidence = "high"
    elif margin >= 0.03:
        confidence = "medium"
    else:
        confidence = "low"

    rationale = [f"matched {top_rec['name']}  score={top_score:.2f}  "
                 f"margin={margin:.2f}  conf={confidence}"]
    rationale.extend(top_why)
    if len(rated) >= 2:
        rationale.append(f"runner-up: {rated[1][1]['name']}  "
                         f"score={rated[1][0]:.2f}")
    return MatchResult(top_rec, top_score, confidence, rationale, margin)


def summarize(result: MatchResult) -> str:
    return "\n".join(["[robot retrieval]"] +
                     ["  " + line for line in result.rationale])
