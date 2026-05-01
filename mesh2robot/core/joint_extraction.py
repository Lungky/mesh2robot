"""Phase 3: extract the kinematic chain from per-body transform trajectories.

Input: output of Phase 2 — a list of rigid bodies, each with a list of
SE(3) transforms T_body[t] for pose t = 0..K-1.

Output:
  - parent/child graph (each body's parent is the body it moves relative to)
  - joint axis and origin (world frame) connecting parent to child
  - joint type (revolute / prismatic / fixed)
  - per-joint angular range (from the supplied configurations)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mesh2robot.core.rigid_fit import screw_from_transform


@dataclass
class JointEstimate:
    parent_body: int
    child_body: int
    type: str                  # 'revolute' | 'prismatic' | 'fixed'
    axis: np.ndarray           # (3,) unit, world frame at pose 0
    origin: np.ndarray         # (3,) world frame at pose 0
    angles: list[float]        # signed rotation per pose (rad)
    lower: float
    upper: float


def _relative_transform(T_parent: np.ndarray, T_child: np.ndarray) -> np.ndarray:
    """Transform applied to the child relative to its parent."""
    return np.linalg.inv(T_parent) @ T_child


def _motion_magnitude(T: np.ndarray) -> float:
    """Proxy for how much this transform deviates from identity."""
    R = T[:3, :3]
    t = T[:3, 3]
    cos_a = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    angle = float(np.arccos(cos_a))
    return angle + float(np.linalg.norm(t))


def infer_parent_by_stillness(
    body_transforms: list[list[np.ndarray]],
) -> list[int]:
    """Greedy parent inference: a body's parent is the one that is 'most still'
    in the poses where the body itself moves.

    Works for serial chains captured with one-joint-at-a-time: the parent of
    link_k is link_{k-1}, because link_{k-1} does not move in the pose where
    joint_k moves.

    Returns parent index per body; -1 means root (world).
    """
    n_bodies = len(body_transforms)
    K = len(body_transforms[0])

    # Motion magnitudes: motion[body][pose]
    motion = np.array([
        [_motion_magnitude(body_transforms[b][t]) for t in range(K)]
        for b in range(n_bodies)
    ])

    # Total motion per body (to identify root = least-moving body)
    total_motion = motion.sum(axis=1)
    root = int(np.argmin(total_motion))

    parents = [-1] * n_bodies
    # Order bodies by increasing total motion (closer to root moves less)
    order = list(np.argsort(total_motion))
    assigned = {root}
    for b in order:
        if b == root:
            continue
        # Parent = the already-assigned body whose motion profile best matches
        # "a subset of" b's motion profile (i.e., parent moves iff child moves,
        # but parent may be still when only later joints move).
        # Equivalently: parent is the body among assigned ones with the largest
        # motion correlation with b, but smaller total motion.
        best_p, best_score = -1, -np.inf
        for p in assigned:
            if total_motion[p] >= total_motion[b]:
                continue
            # Score: inner product of motion profiles
            score = float(motion[p] @ motion[b])
            if score > best_score:
                best_score = score
                best_p = p
        parents[b] = best_p if best_p >= 0 else root
        assigned.add(b)

    return parents


def extract_joints(
    body_transforms: list[list[np.ndarray]],
    configurations: list[dict[str, float]] | None = None,
) -> list[JointEstimate]:
    """Build a kinematic chain from per-body pose transforms.

    Returns one JointEstimate per non-root body.
    """
    n_bodies = len(body_transforms)
    K = len(body_transforms[0])
    parents = infer_parent_by_stillness(body_transforms)

    joints: list[JointEstimate] = []
    for b in range(n_bodies):
        if parents[b] < 0:
            continue
        p = parents[b]
        # Relative transform of child relative to parent at each pose
        rel = [_relative_transform(body_transforms[p][t], body_transforms[b][t])
               for t in range(K)]
        # At pose 0 this is the rest transform; subsequent poses differ by the
        # joint's screw motion from rest.
        rest = rel[0]
        motions = [np.linalg.inv(rest) @ rel[t] for t in range(K)]

        # Aggregate axis estimate: choose pose with largest motion
        mags = [_motion_magnitude(m) for m in motions]
        t_best = int(np.argmax(mags))
        if mags[t_best] < 1e-5:
            # Static joint
            joints.append(JointEstimate(
                parent_body=p, child_body=b, type="fixed",
                axis=np.array([0.0, 0.0, 1.0]), origin=np.zeros(3),
                angles=[0.0] * K, lower=0.0, upper=0.0,
            ))
            continue

        screw = screw_from_transform(motions[t_best])
        # Express axis/origin in world frame at pose 0 = parent frame at pose 0
        # Actually motions[t] is in parent-local frame (rest-relative), so the
        # axis is in that frame. Lift to world using rest and T_parent(0).
        T_parent0 = body_transforms[p][0]
        T_rest_world = T_parent0 @ rest
        R_pw = T_rest_world[:3, :3]
        origin_w = T_rest_world @ np.array([*screw["origin"], 1.0])
        axis_w = R_pw @ screw["axis"]
        axis_w /= np.linalg.norm(axis_w)

        # Signed angle per pose: project the screw motion onto the estimated axis
        angles = []
        for t in range(K):
            s = screw_from_transform(motions[t])
            # Sign: dot with estimated axis direction (in local frame)
            sign = float(np.sign(s["axis"] @ screw["axis"])) if s["angle"] > 1e-6 else 0.0
            angles.append(sign * s["angle"])

        joints.append(JointEstimate(
            parent_body=p, child_body=b, type="revolute",
            axis=axis_w, origin=origin_w[:3],
            angles=angles, lower=min(angles), upper=max(angles),
        ))

    return joints


def refine_joint_origins(
    joints: list[JointEstimate],
    per_link_meshes: dict,
) -> list[JointEstimate]:
    """Pick a physically-meaningful point on each joint's axis.

    The raw `screw_from_transform` returns the axis-line point closest to the
    WORLD origin — mathematically valid but often geometrically arbitrary
    (e.g. (0,0,0) for a Z-axis joint, instead of a point inside the child
    link). This pass replaces each joint's origin with the point on its axis
    line closest to the midpoint between the parent and child mesh centroids.
    The midpoint is a proxy for the mechanical interface between parent and
    child; projecting it onto the axis snaps that interface point onto the
    kinematic line.

    Mesh rendering and kinematics are invariant to this shift (the axis-line
    is unchanged), so this is a pure convention improvement.
    """
    from dataclasses import replace
    refined: list[JointEstimate] = []
    for j in joints:
        parent_mesh = per_link_meshes.get(j.parent_body)
        child_mesh = per_link_meshes.get(j.child_body)
        if child_mesh is None:
            refined.append(j)
            continue
        child_centroid = np.asarray(child_mesh.centroid, dtype=float)
        if parent_mesh is not None:
            parent_centroid = np.asarray(parent_mesh.centroid, dtype=float)
            target = 0.5 * (parent_centroid + child_centroid)
        else:
            target = child_centroid

        axis = j.axis / (np.linalg.norm(j.axis) + 1e-12)
        p = j.origin
        # Project `target` onto the line {p + t * axis}
        t_proj = float((target - p) @ axis)
        new_origin = p + t_proj * axis
        refined.append(replace(j, origin=new_origin))
    return refined
