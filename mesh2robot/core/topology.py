"""Phase E.2 — Tree-topology inference from per-face link labels.

The original `predict_urdf_interactive.build_urdf_from_predictions`
hardcoded a serial chain (Z-sort the links, emit joint i → i+1).
That works for industrial arms but produces nonsense URDFs for
humanoids, quadrupeds, and any robot where the kinematic graph
branches off a torso/base.

This module replaces the Z-sort with a proper tree inference:

  1. Build a link-adjacency graph from `mesh.face_adjacency`. Edge
     weight = number of mesh edges shared between two link clusters.
  2. Pick a root (largest cluster by face count, or a VLM-supplied
     hint).
  3. BFS from root; for each unvisited link, parent = the
     highest-adjacency-count visited neighbour. This handles tree
     topologies (humanoid: torso → arms + legs) AND collapses to a
     path for serial robots (arm: base → link1 → link2 → ...).

The tree is exposed as a `TreeTopology` dataclass; consumers like
`extract_joints_for_tree` and the URDF assembler iterate over the
`(parent, child)` pairs to build joints.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np
import trimesh


@dataclass
class TreeTopology:
    """Parent-child relationships for a robot's link tree.

    `root` is the base / torso (no parent).
    `parent_of[child]` returns parent link id; root is NOT a key.
    `children_of[parent]` returns list of child ids; leaves have empty list.
    """
    root: int
    parent_of: dict[int, int] = field(default_factory=dict)
    children_of: dict[int, list[int]] = field(default_factory=dict)

    @property
    def all_links(self) -> set[int]:
        return {self.root} | set(self.parent_of)

    @property
    def n_joints(self) -> int:
        return len(self.parent_of)

    def is_serial(self) -> bool:
        """True when every node has ≤ 1 child (i.e., a single chain)."""
        return all(len(c) <= 1 for c in self.children_of.values())

    def chain_order(self) -> list[int] | None:
        """If serial, return the link sequence root→tip. Else None."""
        if not self.is_serial():
            return None
        order = [self.root]
        cur = self.root
        while self.children_of.get(cur):
            cur = self.children_of[cur][0]
            order.append(cur)
        return order

    def __str__(self) -> str:
        lines = [f"TreeTopology root={self.root}, {self.n_joints} joints, "
                 f"is_serial={self.is_serial()}"]
        # Walk the tree breadth-first for a readable dump
        depth = {self.root: 0}
        for child, par in self.parent_of.items():
            d = depth.get(par, 0) + 1
            depth[child] = d
        for lid in sorted(self.all_links, key=lambda x: (depth.get(x, 0), x)):
            d = depth.get(lid, 0)
            par = self.parent_of.get(lid, "·")
            n_child = len(self.children_of.get(lid, []))
            lines.append(f"  {'  ' * d}link_{lid}  (parent={par}, "
                         f"children={n_child})")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Adjacency graph
# ---------------------------------------------------------------------------

def build_link_adjacency_graph(
    mesh: trimesh.Trimesh,
    face_labels: np.ndarray,
) -> dict[int, dict[int, int]]:
    """Returns: `graph[link_a][link_b] = number of mesh edges shared between
    link_a and link_b (i.e., face-adjacency edges crossing the (a,b)
    boundary).` Symmetric; ignores edges where either side is unlabeled
    (label < 0).
    """
    adj = mesh.face_adjacency  # (E, 2) face-index pairs sharing an edge
    if len(adj) == 0:
        return {}
    la = face_labels[adj[:, 0]]
    lb = face_labels[adj[:, 1]]
    # Keep only cross-link edges with both sides labeled
    cross = (la != lb) & (la >= 0) & (lb >= 0)
    if not cross.any():
        return {}
    pairs = np.column_stack([la[cross], lb[cross]])    # (M, 2)

    graph: dict[int, dict[int, int]] = {}
    for a, b in pairs:
        a, b = int(a), int(b)
        graph.setdefault(a, {})
        graph.setdefault(b, {})
        graph[a][b] = graph[a].get(b, 0) + 1
        graph[b][a] = graph[b].get(a, 0) + 1
    return graph


# ---------------------------------------------------------------------------
# Root finder
# ---------------------------------------------------------------------------

def find_root_link(
    per_link_meshes: dict[int, trimesh.Trimesh],
    hint_root_id: int | None = None,
    prefer_lowest_z: bool = False,
) -> int:
    """Pick the root link.

    Priority:
      1. `hint_root_id` if provided and present.
      2. If `prefer_lowest_z` (good for industrial arms): the link with
         the lowest Z centroid — typically the base.
      3. Otherwise: the link with the most faces (typically the torso /
         largest body part).
    """
    if hint_root_id is not None and hint_root_id in per_link_meshes:
        return hint_root_id
    if prefer_lowest_z:
        return min(per_link_meshes.keys(),
                   key=lambda lid: float(per_link_meshes[lid].centroid[2]))
    return max(per_link_meshes.keys(),
               key=lambda lid: len(per_link_meshes[lid].faces))


# ---------------------------------------------------------------------------
# Tree inference (BFS with adjacency-weighted parent selection)
# ---------------------------------------------------------------------------

def infer_tree_topology(
    adjacency: dict[int, dict[int, int]],
    all_link_ids: set[int],
    root: int,
) -> TreeTopology:
    """BFS from root, parent-finding heuristic:

    For each newly-discovered link L (reached because some visited
    neighbour V has L as an adjacency entry), find L's BEST parent
    among ALL already-visited links — defined as the visited link
    sharing the most boundary edges with L.

    This keeps the topology accurate when limbs come close to each
    other but only one is the actual kinematic parent (e.g. a hand
    might be near the opposite shoulder geometrically but its real
    parent is its forearm via the wrist boundary).

    Disconnected links (no adjacency to any visited link) get
    attached to root with a 0-edge link — they'll appear as fixed
    joints downstream, since `find_boundary_vertices` will return
    no edges for them.
    """
    if root not in all_link_ids:
        raise ValueError(f"root {root!r} not in all_link_ids")

    parent_of: dict[int, int] = {}
    children_of: dict[int, list[int]] = {lid: [] for lid in all_link_ids}
    visited: set[int] = {root}
    queue: deque[int] = deque([root])

    while queue:
        current = queue.popleft()
        for neighbor, _count in adjacency.get(current, {}).items():
            if neighbor in visited or neighbor not in all_link_ids:
                continue
            # Find neighbor's best parent: the visited link with which
            # it shares the most boundary edges.
            best_parent = current
            best_edges = adjacency.get(neighbor, {}).get(current, 0)
            for v in visited:
                edges = adjacency.get(neighbor, {}).get(v, 0)
                if edges > best_edges:
                    best_edges = edges
                    best_parent = v
            parent_of[neighbor] = best_parent
            children_of[best_parent].append(neighbor)
            visited.add(neighbor)
            queue.append(neighbor)

    # Disconnected components — attach to root as fixed joints
    for lid in all_link_ids:
        if lid not in visited:
            parent_of[lid] = root
            children_of[root].append(lid)

    return TreeTopology(
        root=root,
        parent_of=parent_of,
        children_of=children_of,
    )


# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------

def infer_topology_auto(
    mesh: trimesh.Trimesh,
    face_labels: np.ndarray,
    per_link_meshes: dict[int, trimesh.Trimesh],
    hint_root_id: int | None = None,
    prefer_lowest_z_root: bool = False,
) -> TreeTopology:
    """One-shot helper: build adjacency graph, find root, infer tree."""
    graph = build_link_adjacency_graph(mesh, face_labels)
    root = find_root_link(
        per_link_meshes,
        hint_root_id=hint_root_id,
        prefer_lowest_z=prefer_lowest_z_root,
    )
    return infer_tree_topology(graph, set(per_link_meshes), root)
