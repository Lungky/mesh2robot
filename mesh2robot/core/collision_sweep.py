"""Tier-3 joint-limit refinement via PyBullet self-collision sweep.

Given a URDF and an initial (lower, upper) prior per joint (typically
the model's predicted limits), sweep each joint through that prior
range while holding all other joints at zero, log when self-collision
first occurs on each side of zero, and return the largest contiguous
collision-free interval around the home pose.

The output is the **intersection of the model prior and the
collision-free range** — a strictly tighter (or equal) limit than the
prior. This is the "refine if the initial generated joints are
overlimit" step the user asked for.

Caveats:
  - Only checks self-collision. Doesn't catch real-world joint
    end-stops mechanical in the actual robot (those need motor specs).
  - Other-joint setpoint matters: we sweep one joint at a time with
    the rest at zero. Collisions involving multiple-joint configurations
    may exist outside the returned range. Acceptable for a Tier-3
    safety bound; the user can manually narrow further.
  - Prismatic joints in metres; revolute/continuous in radians.

Usage:
    from mesh2robot.core.collision_sweep import sweep_collision_free

    refined = sweep_collision_free(
        urdf_path="output/foo/refined/robot.urdf",
        priors=[(-3.14, 3.14), (-1.57, 1.57), ...],
        joint_types=["revolute", "revolute", ...],
        n_steps=64,
    )
    # refined: list of (lo, hi) tuples, one per actuated joint.
"""

from __future__ import annotations

import math
from contextlib import contextmanager
from pathlib import Path

import numpy as np


@contextmanager
def _pybullet_session(urdf_path: Path):
    """Spin up a headless PyBullet client, load the URDF, yield (p, body_id)."""
    import pybullet as p
    cid = p.connect(p.DIRECT)
    try:
        # Disable gravity / dynamics — we only do kinematic sweeps.
        p.setRealTimeSimulation(0, physicsClientId=cid)
        body_id = p.loadURDF(
            str(urdf_path),
            useFixedBase=True,
            flags=p.URDF_USE_SELF_COLLISION,
            physicsClientId=cid,
        )
        # Disable adjacent-link contacts. Reconstructed URDFs from real
        # scans typically have visual-mesh overlap at joint boundaries
        # (the link geometry was the user's mesh, not a clean CAD part);
        # PyBullet flags the home pose as self-colliding. Adjacent-link
        # contact is *expected* for any rotational joint anyway, so we
        # only care about NON-adjacent collisions.
        n_joints = p.getNumJoints(body_id, physicsClientId=cid)
        # Build adjacency: child link → parent link from joint.parent_index
        parent_of: dict[int, int] = {-1: -1}
        for ji in range(n_joints):
            info = p.getJointInfo(body_id, ji, physicsClientId=cid)
            parent_idx = int(info[16])   # parent link index (or -1 for base)
            parent_of[ji] = parent_idx
        # Disable each (link, its parent) and (link, its grandparent if same chain)
        # collision pair. PyBullet uses link index = joint index here, with -1 = base.
        for ji in range(n_joints):
            par = parent_of.get(ji, -1)
            p.setCollisionFilterPair(body_id, body_id, ji, par, 0,
                                     physicsClientId=cid)
            grand = parent_of.get(par, -2)
            if grand != -2 and grand != par:
                p.setCollisionFilterPair(body_id, body_id, ji, grand, 0,
                                         physicsClientId=cid)
        yield p, body_id, cid
    finally:
        p.disconnect(cid)


def _has_self_collision(p, body_id: int, cid: int) -> bool:
    """Run one collision-detection step and return True if any self-contact
    on a NON-adjacent link pair (adjacent pairs are filtered out at load time)."""
    p.performCollisionDetection(physicsClientId=cid)
    contacts = p.getContactPoints(bodyA=body_id, bodyB=body_id, physicsClientId=cid)
    return len(contacts) > 0


def _set_all_joints_to(p, body_id: int, cid: int,
                       q_per_joint: dict[int, float]) -> None:
    """Reset every revolute/prismatic joint to the value given (or 0 if absent)."""
    n = p.getNumJoints(body_id, physicsClientId=cid)
    for ji in range(n):
        info = p.getJointInfo(body_id, ji, physicsClientId=cid)
        jt = int(info[2])
        # PyBullet joint types: 0=revolute, 1=prismatic, 4=fixed (others rare)
        if jt not in (0, 1):
            continue
        q = q_per_joint.get(ji, 0.0)
        p.resetJointState(body_id, ji, targetValue=q, physicsClientId=cid)


def _actuated_joint_indices(p, body_id: int, cid: int) -> list[int]:
    """Return PyBullet joint indices in chain order (revolute + prismatic)."""
    out: list[int] = []
    n = p.getNumJoints(body_id, physicsClientId=cid)
    for ji in range(n):
        info = p.getJointInfo(body_id, ji, physicsClientId=cid)
        jt = int(info[2])
        if jt in (0, 1):
            out.append(ji)
    return out


def sweep_collision_free(
    urdf_path: Path | str,
    priors: list[tuple[float, float]],
    joint_types: list[str] | None = None,
    n_steps: int = 64,
    safety_margin: float = 0.02,
    verbose: bool = False,
) -> list[tuple[float, float]]:
    """Sweep each actuated joint and return the largest collision-free
    interval around 0 within the given prior.

    Parameters
    ----------
    urdf_path
        Path to the URDF that will be loaded into PyBullet.
    priors
        Initial (lower, upper) per actuated joint (in chain order matching
        PyBullet's enumeration). These are typically the model's predicted
        limits or the absolute physical caps.
    joint_types
        Optional, parallel to `priors`. Currently used only for logging.
    n_steps
        How many samples to test in each direction around 0.
        Higher = tighter bound but slower.
    safety_margin
        Subtract this fraction of (boundary − 0) on each side after finding
        the first collision; gives the limit some breathing room.
        Set to 0 for the raw boundary.

    Returns
    -------
    refined : list of (lower, upper) per actuated joint.
        - lower in [priors[i][0], 0]
        - upper in [0, priors[i][1]]
        - if a side collides at q=0 already, that side is set to 0 (joint
          effectively can't move that direction)
        - if no collision is found in the prior range, returns the prior
          unchanged on that side.
    """
    urdf_path = Path(urdf_path)
    refined: list[tuple[float, float]] = list(priors)

    with _pybullet_session(urdf_path) as (p, body_id, cid):
        actuated = _actuated_joint_indices(p, body_id, cid)
        if len(actuated) != len(priors):
            if verbose:
                print(f"  [sweep] WARNING: PyBullet sees {len(actuated)} actuated "
                      f"joints but priors has {len(priors)}; using min length")
        n_use = min(len(actuated), len(priors))

        for i in range(n_use):
            ji = actuated[i]
            lo_prior, hi_prior = priors[i]
            jt_str = (joint_types[i] if (joint_types and i < len(joint_types))
                      else "?")

            # Reset to home pose every iteration so we sweep one joint
            # against the canonical zero configuration of all others.
            _set_all_joints_to(p, body_id, cid, {})

            # First check home pose itself
            if _has_self_collision(p, body_id, cid):
                if verbose:
                    print(f"  [sweep] joint{i} ({jt_str}): home pose self-collides; "
                          f"keeping prior {lo_prior:.3f}..{hi_prior:.3f}")
                continue

            lo_safe = lo_prior
            hi_safe = hi_prior

            # Sweep upper side from 0 → hi_prior
            for step_idx in range(1, n_steps + 1):
                q = step_idx / n_steps * hi_prior
                p.resetJointState(body_id, ji, targetValue=q, physicsClientId=cid)
                if _has_self_collision(p, body_id, cid):
                    boundary = (step_idx - 1) / n_steps * hi_prior
                    hi_safe = boundary - safety_margin * abs(hi_prior - 0)
                    hi_safe = max(hi_safe, 0.0)
                    break

            p.resetJointState(body_id, ji, targetValue=0.0, physicsClientId=cid)

            # Sweep lower side from 0 → lo_prior
            for step_idx in range(1, n_steps + 1):
                q = step_idx / n_steps * lo_prior
                p.resetJointState(body_id, ji, targetValue=q, physicsClientId=cid)
                if _has_self_collision(p, body_id, cid):
                    boundary = (step_idx - 1) / n_steps * lo_prior
                    lo_safe = boundary + safety_margin * abs(lo_prior - 0)
                    lo_safe = min(lo_safe, 0.0)
                    break

            # Reset and apply the safe range
            p.resetJointState(body_id, ji, targetValue=0.0, physicsClientId=cid)
            refined[i] = (float(lo_safe), float(hi_safe))
            if verbose:
                print(f"  [sweep] joint{i} ({jt_str}): "
                      f"prior=[{lo_prior:.3f}, {hi_prior:.3f}] "
                      f"→ refined=[{lo_safe:.3f}, {hi_safe:.3f}]")

    return refined
