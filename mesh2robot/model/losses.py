"""Multi-task losses for the mesh2robot model.

  - segmentation: per-point cross-entropy with ignore_index=-1
  - axis regression: 1 - cos(predicted, target) on valid joints only
  - origin regression: smooth-L1 in meters on valid joints only
  - joint type: cross-entropy on valid joints only
  - valid prediction: BCE supervised by `joint_valid`

Per-task weights are tunable. We multiply by `joint_valid` to mask
padded slots so the model isn't penalized for "predicting no joint" on
padding.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class LossWeights:
    seg: float = 1.0
    axis: float = 1.0
    origin: float = 1.0
    type: float = 0.5
    valid: float = 0.5
    limits: float = 0.5


# Joint-type class weights, derived from v1 URDF dataset distribution
# (revolute 88.7%, continuous 11.1%, prismatic 1.2%, others rare).
# Inverse-frequency-style; clamped to avoid extreme weights.
# Order matches JOINT_TYPE_TO_INT: revolute=0, continuous=1, prismatic=2,
#                                   fixed=3, floating=4, planar=5.
JOINT_TYPE_CLASS_WEIGHTS = torch.tensor(
    [1.0, 2.5, 8.0, 1.0, 1.0, 1.0],
    dtype=torch.float32,
)


def compute_losses(
    pred: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    weights: LossWeights | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute multi-task loss + per-task scalar dict for logging."""
    if weights is None:
        weights = LossWeights()

    # -- segmentation --
    # seg_logits: (B, N, K)  point_labels: (B, N) with -1 = ignore
    seg_logits = pred["seg_logits"]
    seg_labels = batch["point_labels"]
    seg_loss = F.cross_entropy(
        seg_logits.reshape(-1, seg_logits.shape[-1]),
        seg_labels.reshape(-1),
        ignore_index=-1,
    )

    # -- joint validity (BCE on a fixed J_MAX-slot binary classifier) --
    valid_logit = pred["valid_logit"]            # (B, J_MAX)
    valid_target = batch["joint_valid"].float()  # (B, J_MAX)
    valid_loss = F.binary_cross_entropy_with_logits(valid_logit, valid_target)

    # Mask: only valid joints contribute to axis/origin/type losses.
    valid_mask = batch["joint_valid"]            # (B, J_MAX) bool
    n_valid = valid_mask.sum().clamp(min=1)

    # -- axis: 1 - cos. Re-normalize predicted axes. --
    pred_axis = pred["axis"]                          # (B, J_MAX, 3)
    pred_axis = pred_axis / (pred_axis.norm(dim=-1, keepdim=True) + 1e-8)
    target_axis = batch["joint_axes"]                 # (B, J_MAX, 3)
    target_norm = target_axis.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    target_axis_u = target_axis / target_norm
    cos = (pred_axis * target_axis_u).sum(dim=-1)     # (B, J_MAX)
    # Sign-invariant axis loss (a and -a are the same direction)
    axis_loss = (1.0 - cos.abs())[valid_mask].sum() / n_valid

    # -- origin: smooth-L1 over (B, J_MAX, 3) --
    origin_diff = pred["origin"] - batch["joint_origins"]    # (B, J_MAX, 3)
    origin_l1 = F.smooth_l1_loss(
        origin_diff, torch.zeros_like(origin_diff), reduction="none",
    ).sum(dim=-1)                                            # (B, J_MAX)
    origin_loss = origin_l1[valid_mask].sum() / n_valid

    # -- joint type: cross-entropy over valid slots --
    type_logits = pred["type_logits"]                # (B, J_MAX, T)
    type_target = batch["joint_types"]               # (B, J_MAX) int
    type_target_clamped = type_target.clamp(min=0)   # -1 → 0; valid_mask handles them
    type_ce = F.cross_entropy(
        type_logits.reshape(-1, type_logits.shape[-1]),
        type_target_clamped.reshape(-1),
        reduction="none",
    ).reshape(type_target.shape)                     # (B, J_MAX)
    type_loss = type_ce[valid_mask].sum() / n_valid

    # -- joint limits: smooth-L1 on (lower, upper) for valid joints with
    # has_limits=True. Legacy v1/v2 shards lacked this field; their batches
    # come through with all-zero targets and has_limits=False — those
    # examples must NOT contribute to the limits loss (the zeros are
    # uninformative, not ground-truth ±0). The mask below ensures that.
    if "limits" in pred and "joint_limits" in batch:
        pred_limits = pred["limits"]                       # (B, J_MAX, 2)
        target_limits = batch["joint_limits"]              # (B, J_MAX, 2)
        # Per-example has_limits: shape (B,) bool. Broadcast over (J_MAX, 2).
        has_limits_b = batch["has_limits"]                 # (B,) bool
        limits_per_slot_mask = valid_mask & has_limits_b.unsqueeze(-1)   # (B, J_MAX)
        n_limit = limits_per_slot_mask.sum().clamp(min=1)
        limits_l1 = F.smooth_l1_loss(
            pred_limits, target_limits, reduction="none",
        ).sum(dim=-1)                                       # (B, J_MAX)
        limits_loss = limits_l1[limits_per_slot_mask].sum() / n_limit
        limits_mae = _limits_mae(
            pred_limits, target_limits, limits_per_slot_mask,
        ).detach()
    else:
        limits_loss = torch.zeros((), device=seg_logits.device)
        limits_mae = torch.zeros((), device=seg_logits.device)

    total = (
        weights.seg * seg_loss
        + weights.axis * axis_loss
        + weights.origin * origin_loss
        + weights.type * type_loss
        + weights.valid * valid_loss
        + weights.limits * limits_loss
    )

    return total, {
        "loss/total": total.detach(),
        "loss/seg": seg_loss.detach(),
        "loss/axis": axis_loss.detach(),
        "loss/origin": origin_loss.detach(),
        "loss/type": type_loss.detach(),
        "loss/valid": valid_loss.detach(),
        "loss/limits": limits_loss.detach(),
        "metric/seg_acc": _seg_acc(seg_logits, seg_labels).detach(),
        "metric/axis_deg": _axis_deg_err(pred_axis, target_axis_u, valid_mask).detach(),
        "metric/origin_m": _origin_m_err(pred["origin"], batch["joint_origins"],
                                           valid_mask).detach(),
        "metric/valid_acc": _valid_acc(valid_logit, valid_target).detach(),
        "metric/limits_mae": limits_mae,
    }


def _seg_acc(seg_logits: torch.Tensor, seg_labels: torch.Tensor) -> torch.Tensor:
    pred_cls = seg_logits.argmax(dim=-1)
    mask = seg_labels >= 0
    if mask.sum() == 0:
        return torch.tensor(0.0, device=seg_logits.device)
    return (pred_cls[mask] == seg_labels[mask]).float().mean()


def _axis_deg_err(pred_axis_u: torch.Tensor, target_axis_u: torch.Tensor,
                   valid_mask: torch.Tensor) -> torch.Tensor:
    cos = (pred_axis_u * target_axis_u).sum(dim=-1).abs().clamp(-1, 1)
    if valid_mask.sum() == 0:
        return torch.tensor(0.0, device=pred_axis_u.device)
    deg = torch.rad2deg(torch.acos(cos))
    return deg[valid_mask].mean()


def _origin_m_err(pred_origin: torch.Tensor, target_origin: torch.Tensor,
                    valid_mask: torch.Tensor) -> torch.Tensor:
    if valid_mask.sum() == 0:
        return torch.tensor(0.0, device=pred_origin.device)
    err = (pred_origin - target_origin).norm(dim=-1)
    return err[valid_mask].mean()


def _valid_acc(valid_logit: torch.Tensor, valid_target: torch.Tensor) -> torch.Tensor:
    pred = (torch.sigmoid(valid_logit) > 0.5).float()
    return (pred == valid_target).float().mean()


def _limits_mae(
    pred_limits: torch.Tensor, target_limits: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Mean absolute error on (lower, upper) over masked slots.
    Returns a single scalar in the prediction's native units (radians for
    revolute, metres for prismatic — mixed; interpret with care)."""
    if mask.sum() == 0:
        return torch.zeros((), device=pred_limits.device)
    err = (pred_limits - target_limits).abs().sum(dim=-1) / 2.0   # avg of 2 components
    return err[mask].mean()
