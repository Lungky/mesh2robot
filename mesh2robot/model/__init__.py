"""mesh2robot model package — Phase C.

Multi-task 3D foundation model for robot mesh understanding:
  - per-vertex link segmentation
  - per-joint axis + origin regression
  - per-joint type classification
  - robot type classification
  - articulation graph (link/joint topology)

The dataset module loads training shards produced by `data_gen` (Phase B).
The model module defines a backbone (PointNet baseline now, Point
Transformer V3 in a future iteration) and the multi-task heads.
"""

from mesh2robot.model.dataset import (
    K_LINKS_MAX,
    J_MAX,
    ShardDataset,
    collate_examples,
)

__all__ = [
    "K_LINKS_MAX",
    "J_MAX",
    "ShardDataset",
    "collate_examples",
]
