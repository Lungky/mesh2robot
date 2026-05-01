"""Multi-task model — PointNet baseline + heads.

This is a baseline architecture for Phase C. It's deliberately simple:
  - PointNet encoder (per-point MLP + max-pool global) — easy to train,
    runs on CPU/GPU, no custom CUDA ops.
  - Per-vertex segmentation head: per-point MLP over (local_feat, global_feat)
  - Per-joint regression head: takes the global feature plus a learned
    "joint slot" embedding, decodes (axis_unit, origin_offset, joint_type)
    for each of `J_MAX` slots. A `joint_valid` mask in the loss tells the
    model which slots are real.

Once the training loop is validated end-to-end, swap the encoder for a
real Point Transformer V3. The interface (forward(points) → dict of
predictions) stays the same.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from mesh2robot.model.dataset import J_MAX, K_LINKS_MAX
from mesh2robot.model.encoders import (
    PointNetEncoder as _PointNetEncoder,
    PointTransformerV3Encoder,
)


# ---------------------------------------------------------------------------
# Encoder (PointNet baseline)
# ---------------------------------------------------------------------------

class PointNetEncoder(nn.Module):
    """Per-point MLP → max-pool global feature → broadcast back to per-point.

    Output:
      per_point: (B, N, F)
      global_feat: (B, F)
    """

    def __init__(self, in_dim: int = 3, feat_dim: int = 256) -> None:
        super().__init__()
        h = feat_dim
        self.mlp1 = nn.Sequential(
            nn.Linear(in_dim, 64), nn.ReLU(inplace=True),
            nn.Linear(64, 128), nn.ReLU(inplace=True),
            nn.Linear(128, h), nn.ReLU(inplace=True),
        )
        self.mlp2 = nn.Sequential(
            nn.Linear(h, h), nn.ReLU(inplace=True),
            nn.Linear(h, h),
        )
        self.feat_dim = h

    def forward(self, points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # points: (B, N, 3)
        x = self.mlp1(points)          # (B, N, F)
        g = x.amax(dim=1)              # (B, F) max-pool
        g = self.mlp2(g)               # (B, F)
        return x, g


# ---------------------------------------------------------------------------
# Heads
# ---------------------------------------------------------------------------

class SegmentationHead(nn.Module):
    """Per-point classification over K_LINKS_MAX classes."""

    def __init__(self, in_dim: int, num_classes: int = K_LINKS_MAX) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256), nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(256, 128), nn.ReLU(inplace=True),
            nn.Linear(128, num_classes),
        )

    def forward(self, per_point: torch.Tensor, global_feat: torch.Tensor) -> torch.Tensor:
        # per_point: (B, N, F_pp), global_feat: (B, F_g)
        B, N, _ = per_point.shape
        g = global_feat.unsqueeze(1).expand(B, N, -1)
        x = torch.cat([per_point, g], dim=-1)
        return self.net(x)             # (B, N, K_LINKS_MAX)


class JointHead(nn.Module):
    """Per-joint multi-output head.

    Uses J_MAX learned "slot" embeddings, each fused with the global feat
    via concatenation. Output per slot:
      - axis (3,)   — will be re-normalized to unit before loss
      - origin (3,)
      - type_logits (NUM_JOINT_TYPES,)
      - valid_logit (1,) — predicted "joint exists" (auxiliary; supervised
        from the dataset's joint_valid mask)
      - limits (2,) — (lower, upper) in radians for revolute/continuous,
        metres for prismatic, both 0 for fixed/floating/planar. Predicted
        as raw scalars; loss is masked by joint_valid AND has_limits to
        skip legacy shards that lack the field.
    """

    NUM_JOINT_TYPES = 6   # matches JOINT_TYPE_TO_INT in data_gen

    def __init__(self, global_dim: int, slot_dim: int = 64) -> None:
        super().__init__()
        self.slot = nn.Embedding(J_MAX, slot_dim)
        in_dim = global_dim + slot_dim
        self.shared = nn.Sequential(
            nn.Linear(in_dim, 256), nn.ReLU(inplace=True),
            nn.Linear(256, 256), nn.ReLU(inplace=True),
        )
        self.axis_head = nn.Linear(256, 3)
        self.origin_head = nn.Linear(256, 3)
        self.type_head = nn.Linear(256, self.NUM_JOINT_TYPES)
        self.valid_head = nn.Linear(256, 1)
        self.limits_head = nn.Linear(256, 2)

    def forward(self, global_feat: torch.Tensor) -> dict[str, torch.Tensor]:
        # global_feat: (B, F_g)
        B = global_feat.shape[0]
        slots = self.slot(torch.arange(J_MAX, device=global_feat.device))   # (J_MAX, S)
        slots = slots.unsqueeze(0).expand(B, J_MAX, -1)                     # (B, J_MAX, S)
        g = global_feat.unsqueeze(1).expand(B, J_MAX, -1)                   # (B, J_MAX, F_g)
        x = torch.cat([g, slots], dim=-1)
        h = self.shared(x)
        return {
            "axis": self.axis_head(h),                 # (B, J_MAX, 3)
            "origin": self.origin_head(h),             # (B, J_MAX, 3)
            "type_logits": self.type_head(h),          # (B, J_MAX, NUM_JOINT_TYPES)
            "valid_logit": self.valid_head(h).squeeze(-1),   # (B, J_MAX)
            "limits": self.limits_head(h),             # (B, J_MAX, 2)
        }


# ---------------------------------------------------------------------------
# Combined model
# ---------------------------------------------------------------------------

class Mesh2RobotModel(nn.Module):
    """Multi-task model with swappable encoder backbone.

    Encoders supported:
      - "pointnet" (default, baseline): per-point MLP + max-pool, ~0.5M params
      - "ptv3"    : Point Transformer V3, ~30M params, sparse-conv + serialization

    Heads stay the same — segmentation predicts per-vertex link assignment,
    joint head predicts axis/origin/type/valid for J_MAX slots from the
    global feature.
    """

    def __init__(
        self,
        feat_dim: int = 256,
        encoder: str = "pointnet",
        encoder_size: str = "small",
    ) -> None:
        super().__init__()
        self.encoder_kind = encoder
        self.encoder_size = encoder_size
        if encoder == "pointnet":
            self.encoder = _PointNetEncoder(in_dim=3, feat_dim=feat_dim)
            seg_in_dim = 2 * feat_dim         # concat(per_point, global)
            global_dim = feat_dim
        elif encoder == "ptv3":
            self.encoder = PointTransformerV3Encoder(size=encoder_size)
            seg_in_dim = self.encoder.feat_dim + self.encoder.global_feat_dim
            global_dim = self.encoder.global_feat_dim
        else:
            raise ValueError(f"Unknown encoder: {encoder!r}")
        self.seg_head = SegmentationHead(in_dim=seg_in_dim,
                                         num_classes=K_LINKS_MAX)
        self.joint_head = JointHead(global_dim=global_dim)

    def forward(self, points: torch.Tensor) -> dict[str, torch.Tensor]:
        per_point, global_feat = self.encoder(points)
        seg_logits = self.seg_head(per_point, global_feat)
        joints = self.joint_head(global_feat)
        return {
            "seg_logits": seg_logits,    # (B, N, K_LINKS_MAX)
            **joints,                     # axis, origin, type_logits, valid_logit
        }
