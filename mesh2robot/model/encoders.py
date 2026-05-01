"""Encoder backbones for the Mesh2RobotModel.

Two interchangeable encoders, both with the same external interface:
    forward(points: (B, N, 3)) -> (per_point: (B, N, F), global_feat: (B, F))

so the rest of the model (seg head, joint head) is encoder-agnostic.

Backbones:
  - PointNetEncoder           the 0.5M-param baseline (kept for ablations)
  - PointTransformerV3Encoder the SOTA upgrade, ~30M params

The PT-V3 encoder handles the (B, N, 3) -> Pointcept Point dict conversion
and back to (B, N, F).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from mesh2robot.model.ptv3.model import PointTransformerV3, Point


class PointNetEncoder(nn.Module):
    """Per-point MLP + max-pool global feature. Baseline.

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
        x = self.mlp1(points)
        g = x.amax(dim=1)
        g = self.mlp2(g)
        return x, g


# Named PT-V3 configs. Use `--encoder-size <name>` at training time;
# the chosen name gets saved in checkpoint["args"] so predict scripts
# can reconstruct the same architecture before load_state_dict.
PTV3_CONFIGS: dict[str, dict] = {
    # Small (~30M params) — fits a 24 GB 3090 at batch 16-32 × 16k pts.
    # This is the historical default and matches model_v2_ptv3_25ep.
    "small": dict(
        enc_depths=(2, 2, 2, 4, 2),
        enc_channels=(32, 64, 128, 256, 384),
        enc_num_head=(2, 4, 8, 16, 24),
        dec_depths=(2, 2, 2, 2),
        dec_channels=(64, 64, 128, 256),
        dec_num_head=(4, 4, 8, 16),
    ),
    # Base (~120M params) — wider channels + deeper stage-3/4. Targets
    # H200-class GPUs (143 GB VRAM); batch 64-96 × 16k pts typical.
    "base": dict(
        enc_depths=(2, 2, 4, 6, 2),
        enc_channels=(64, 128, 256, 384, 512),
        enc_num_head=(4, 8, 16, 24, 32),
        dec_depths=(2, 2, 4, 4),
        dec_channels=(128, 128, 256, 384),
        dec_num_head=(8, 8, 16, 24),
    ),
}


class PointTransformerV3Encoder(nn.Module):
    """Wraps Pointcept's PointTransformerV3 to produce per-point and global
    features for an entire (B, N, 3) batch, identical interface to PointNet.

    Internally:
      1. Flatten (B, N, 3) → (B*N, 3) plus a `batch` index tensor.
      2. Build a Point dict with grid_size for serialization.
      3. Run PT-V3 → per-point feat (N_total, dec_channels[0]).
      4. Reshape back to (B, N, F).
      5. Max-pool per-batch for the global feat.

    `size` ∈ PTV3_CONFIGS picks the architecture (default "small"). Any
    of the per-stage args below override the named config — useful for
    one-off experiments without adding a new config name.
    """

    def __init__(
        self,
        feat_dim: int = 64,
        grid_size: float = 0.02,    # in normalized-mesh units (~unit ball)
        enable_flash: bool = False,
        size: str = "small",
        # Per-stage overrides (None → take from PTV3_CONFIGS[size]):
        enc_depths: tuple[int, ...] | None = None,
        enc_channels: tuple[int, ...] | None = None,
        enc_num_head: tuple[int, ...] | None = None,
        dec_depths: tuple[int, ...] | None = None,
        dec_channels: tuple[int, ...] | None = None,
        dec_num_head: tuple[int, ...] | None = None,
    ) -> None:
        if size not in PTV3_CONFIGS:
            raise ValueError(f"Unknown PT-V3 size {size!r}; "
                             f"choose from {list(PTV3_CONFIGS)}")
        cfg = PTV3_CONFIGS[size]
        enc_depths = enc_depths if enc_depths is not None else cfg["enc_depths"]
        enc_channels = enc_channels if enc_channels is not None else cfg["enc_channels"]
        enc_num_head = enc_num_head if enc_num_head is not None else cfg["enc_num_head"]
        dec_depths = dec_depths if dec_depths is not None else cfg["dec_depths"]
        dec_channels = dec_channels if dec_channels is not None else cfg["dec_channels"]
        dec_num_head = dec_num_head if dec_num_head is not None else cfg["dec_num_head"]
        super().__init__()
        self.grid_size = grid_size
        self.feat_dim = dec_channels[0]   # PT-V3 output dim per point
        self.global_feat_dim = self.feat_dim
        self.backbone = PointTransformerV3(
            in_channels=3,                # we use (x, y, z) only
            cls_mode=False,
            enable_flash=enable_flash,
            enc_depths=enc_depths,
            enc_channels=enc_channels,
            enc_num_head=enc_num_head,
            enc_patch_size=(1024,) * len(enc_depths),
            dec_depths=dec_depths,
            dec_channels=dec_channels,
            dec_num_head=dec_num_head,
            dec_patch_size=(1024,) * len(dec_depths),
            shuffle_orders=True,
        )

    def forward(self, points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # points: (B, N, 3) — already normalized to unit ball by ShardDataset.
        B, N, _ = points.shape
        device = points.device

        # Flatten and build batch indices
        flat = points.reshape(B * N, 3).contiguous()
        batch = torch.arange(B, device=device).repeat_interleave(N)
        offset = torch.arange(1, B + 1, device=device) * N   # cumulative

        # PT-V3 expects integer grid coordinates (sparse-conv tensor). The
        # `Point` class derives them from coord+grid_size on first
        # serialization call, but we need to ensure positive coordinates.
        # ShardDataset normalizes points to a unit ball (≈ [-1, 1]^3), so
        # we shift them into [0, 2] by subtracting the per-batch min in the
        # forward call. The Point class does this in `serialization()`.
        data = {
            "coord": flat,
            "feat": flat,                 # use coords as features
            "batch": batch.long(),
            "offset": offset.long(),
            "grid_size": self.grid_size,
        }
        point = Point(data)
        out: Point = self.backbone(point)

        # PT-V3 returns the original Point (with .feat updated) when not
        # cls_mode. Output features are at out.feat shape (N_total, feat_dim).
        # Note: PT-V3's serialization can permute points internally; if so,
        # `out.serialized_inverse` would give us the inverse permutation.
        # In cls_mode=False with the standard config, the encoder-decoder
        # path returns features in the SAME order as input.
        per_point_flat = out.feat                 # (N_total, F)
        per_point = per_point_flat.view(B, N, -1)  # (B, N, F)

        # Global feat: max-pool over points per batch
        global_feat = per_point.amax(dim=1)
        return per_point, global_feat
