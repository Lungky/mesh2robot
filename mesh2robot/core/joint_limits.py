"""Joint-limit resolution for generated URDFs.

Priority (highest wins):
  1. User YAML override              (load_yaml_overrides)
  2. Intersection of:
       a. Template limits from the URDF database (Phase 4)
       b. Self-collision sweep envelope from PyBullet
  3. Observed motion range * safety margin (fallback)

Most users of a bespoke robot have no matching template and want a mix of
(1) for a few joints they know plus (2b) for an automatic safety cap on the
rest. The resolver supports all combinations.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable

import numpy as np


# ---------------------------------------------------------------------------
# User YAML override
# ---------------------------------------------------------------------------

def load_yaml_overrides(path: str | Path) -> dict[str, tuple[float, float]]:
    """Load per-joint limit overrides from a YAML file.

    Expected schema:

        joints:
          joint_1: {lower: -3.14, upper: 3.14}
          joint_2: {lower: -1.57, upper: 2.09}

    Angles are radians. Missing joints fall through to the next priority tier.
    Returns {} if the file is absent or unparseable.
    """
    p = Path(path)
    if not p.exists():
        return {}
    try:
        import yaml
    except ImportError:
        return {}
    try:
        data = yaml.safe_load(p.read_text()) or {}
    except yaml.YAMLError:
        return {}

    out: dict[str, tuple[float, float]] = {}
    for name, val in (data.get("joints") or {}).items():
        if isinstance(val, dict) and "lower" in val and "upper" in val:
            try:
                lo = float(val["lower"])
                hi = float(val["upper"])
            except (TypeError, ValueError):
                continue
            if math.isfinite(lo) and math.isfinite(hi) and hi > lo:
                out[str(name)] = (lo, hi)
    return out


# ---------------------------------------------------------------------------
# Self-collision sweep
# ---------------------------------------------------------------------------

def sweep_self_collision_limits(
    urdf_path: str | Path,
    max_range_rad: float = 2 * math.pi,
    step_deg: float = 2.0,
    ignore_adjacent_links: bool = True,
) -> dict[str, tuple[float, float]]:
    """Per-joint safe envelope discovered by sweeping in PyBullet.

    For each revolute joint, starts at 0 and walks outward in both directions
    until a self-collision is detected (or max_range is reached). All other
    joints are held at 0. Adjacent-link contacts are ignored by default, as
    they are always touching at the joint mechanism.

    Returns {joint_name: (lower_safe, upper_safe)} in radians.
    """
    import pybullet as p  # local import — optional dependency at runtime

    step = math.radians(step_deg)
    cid = p.connect(p.DIRECT)
    try:
        rid = p.loadURDF(
            str(urdf_path),
            useFixedBase=True,
            flags=p.URDF_USE_SELF_COLLISION,
        )
        n_joints = p.getNumJoints(rid)

        def set_all_zero() -> None:
            for i in range(n_joints):
                p.resetJointState(rid, i, 0.0)

        def self_collides() -> bool:
            p.performCollisionDetection()
            for c in p.getContactPoints(rid, rid):
                link_a, link_b = c[3], c[4]
                if ignore_adjacent_links and abs(link_a - link_b) == 1:
                    continue
                return True
            return False

        # Quick sanity: if we self-collide at home, the URDF geometry is
        # already in contact. Skip the sweep (we can't distinguish ok from bad).
        set_all_zero()
        if self_collides():
            return {}

        limits: dict[str, tuple[float, float]] = {}
        for joint_idx in range(n_joints):
            info = p.getJointInfo(rid, joint_idx)
            jtype = info[2]
            if jtype != p.JOINT_REVOLUTE and jtype != p.JOINT_PRISMATIC:
                continue
            jname = info[1].decode()

            # Sweep positive
            set_all_zero()
            pos_limit = max_range_rad
            angle = step
            while angle <= max_range_rad + 1e-9:
                p.resetJointState(rid, joint_idx, angle)
                if self_collides():
                    pos_limit = angle - step
                    break
                angle += step

            # Sweep negative
            set_all_zero()
            neg_limit = -max_range_rad
            angle = -step
            while angle >= -max_range_rad - 1e-9:
                p.resetJointState(rid, joint_idx, angle)
                if self_collides():
                    neg_limit = angle + step
                    break
                angle -= step

            limits[jname] = (float(neg_limit), float(pos_limit))
            set_all_zero()

        return limits
    finally:
        p.disconnect(cid)


# ---------------------------------------------------------------------------
# Limit resolver
# ---------------------------------------------------------------------------

def resolve_limits(
    joint_names: list[str],
    template: list[tuple[float, float]] | None,
    observed: list[tuple[float, float]],
    collision: dict[str, tuple[float, float]] | None = None,
    override: dict[str, tuple[float, float]] | None = None,
    observed_margin: float = 4.0,
) -> list[tuple[float, float]]:
    """Resolve final per-joint limits from all available sources.

    - For each joint (in chain order by `joint_names`):
      1. If override has this joint → use it (user is authoritative).
      2. Else start from template limit if available and finite, else
         observed * margin.
      3. If collision envelope is available, intersect (tighten) the limits
         so the joint never drives into self-collision.
    """
    n = len(joint_names)
    if template is None:
        template = []
    collision = collision or {}
    override = override or {}

    resolved: list[tuple[float, float]] = []
    for i, jname in enumerate(joint_names):
        if jname in override:
            lo, hi = override[jname]
            resolved.append((lo, hi))
            continue

        tpl = template[i] if i < len(template) else (math.nan, math.nan)
        if math.isfinite(tpl[0]) and math.isfinite(tpl[1]) and tpl[1] > tpl[0]:
            lo, hi = tpl
        else:
            obs_lo, obs_hi = observed[i] if i < len(observed) else (0.0, 0.0)
            mid = 0.5 * (obs_lo + obs_hi)
            half = 0.5 * (obs_hi - obs_lo) * observed_margin
            lo, hi = mid - half, mid + half

        if jname in collision:
            c_lo, c_hi = collision[jname]
            lo = max(lo, c_lo)
            hi = min(hi, c_hi)
            if hi <= lo:
                # Collision envelope is incompatible with everything else —
                # this usually means self-collision at home pose. Fall back
                # to the collision envelope alone.
                lo, hi = c_lo, c_hi

        resolved.append((float(lo), float(hi)))
    return resolved


def summarize(
    joint_names: list[str],
    resolved: list[tuple[float, float]],
    template: list[tuple[float, float]] | None = None,
    collision: dict[str, tuple[float, float]] | None = None,
    override: dict[str, tuple[float, float]] | None = None,
) -> str:
    """Pretty-print each joint's final limit along with the contributing sources."""
    template = template or []
    collision = collision or {}
    override = override or {}
    lines = [f"{'joint':<12s} {'resolved (deg)':<22s} {'template':<16s} "
             f"{'collision':<16s} {'override':<12s}"]
    for i, jname in enumerate(joint_names):
        lo, hi = resolved[i]
        tpl = template[i] if i < len(template) else None
        col = collision.get(jname)
        ovr = override.get(jname)
        tpl_s = f"[{math.degrees(tpl[0]):+6.1f},{math.degrees(tpl[1]):+6.1f}]" if tpl and math.isfinite(tpl[0]) else "-"
        col_s = f"[{math.degrees(col[0]):+6.1f},{math.degrees(col[1]):+6.1f}]" if col else "-"
        ovr_s = "yes" if ovr else "-"
        res_s = f"[{math.degrees(lo):+6.1f},{math.degrees(hi):+6.1f}]"
        lines.append(f"{jname:<12s} {res_s:<22s} {tpl_s:<16s} {col_s:<16s} {ovr_s:<12s}")
    return "\n".join(lines)
