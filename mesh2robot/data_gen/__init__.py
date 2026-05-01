"""Synthetic training-data generator for the mesh2robot foundation model.

Takes URDF / MJCF files and emits labeled training examples:
  - point_cloud (N, 3): sampled surface points
  - link_labels (N,): per-point link index
  - joint_axes_world (J, 3): unit axis per joint
  - joint_origins_world (J, 3): pivot per joint in world frame
  - joint_types (J,): integer encoding (revolute=0, prismatic=1, fixed=2, ...)
  - joint_parents (J,): parent link index per joint
  - joint_children (J,): child link index per joint
  - meta: robot family, vendor, source, config sampled, augmentation params

Used by Phase B of the research roadmap. See `urdf_loader.py` for FK +
mesh assembly and `augment.py` for noise/transform augmentations.
"""

from mesh2robot.data_gen.urdf_loader import (
    LoadedRobot,
    TrainingExample,
    JOINT_TYPE_TO_INT,
    INT_TO_JOINT_TYPE,
    load_robot,
    sample_random_config,
    articulate_and_label,
    sample_point_cloud,
)

__all__ = [
    "LoadedRobot",
    "TrainingExample",
    "JOINT_TYPE_TO_INT",
    "INT_TO_JOINT_TYPE",
    "load_robot",
    "sample_random_config",
    "articulate_and_label",
    "sample_point_cloud",
]
