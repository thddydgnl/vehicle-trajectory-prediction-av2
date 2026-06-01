from __future__ import annotations

import torch
import torch.nn.functional as F


def _validate_trajectory_shapes(pred: torch.Tensor, gt: torch.Tensor) -> None:
    if pred.shape != gt.shape:
        raise ValueError(f"pred and gt must have the same shape, got {pred.shape} and {gt.shape}")
    if pred.ndim != 3 or pred.shape[-1] != 2:
        raise ValueError(f"Expected trajectory shape [B, T, 2], got {pred.shape}")


def _apply_mask(loss: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    if mask is None:
        return loss.mean()
    if mask.shape != loss.shape[:2]:
        raise ValueError(f"mask shape must be [B, T], got {mask.shape}, expected {loss.shape[:2]}")
    mask_float = mask.to(device=loss.device, dtype=loss.dtype)
    denominator = mask_float.sum()
    if denominator <= 0:
        raise ValueError("mask must contain at least one valid trajectory step")
    return (loss * mask_float).sum() / denominator


def trajectory_mse_loss(pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    _validate_trajectory_shapes(pred, gt)
    step_loss = F.mse_loss(pred, gt, reduction="none").mean(dim=-1)
    return _apply_mask(step_loss, mask)


def trajectory_smooth_l1_loss(
    pred: torch.Tensor,
    gt: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    _validate_trajectory_shapes(pred, gt)
    step_loss = F.smooth_l1_loss(pred, gt, reduction="none").mean(dim=-1)
    return _apply_mask(step_loss, mask)


def _final_valid_indices(mask: torch.Tensor) -> torch.Tensor:
    if mask.ndim != 2:
        raise ValueError(f"mask shape must be [B, T], got {mask.shape}")
    valid_counts = mask.sum(dim=1)
    if torch.any(valid_counts <= 0):
        raise ValueError("each batch item must contain at least one valid trajectory step")
    step_indices = torch.arange(mask.shape[1], device=mask.device).expand(mask.shape[0], -1)
    return torch.where(mask, step_indices, torch.full_like(step_indices, -1)).max(dim=1).values


def endpoint_loss(pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    _validate_trajectory_shapes(pred, gt)
    if mask is not None:
        if mask.shape != pred.shape[:2]:
            raise ValueError(f"mask shape must be [B, T], got {mask.shape}, expected {pred.shape[:2]}")
        final_indices = _final_valid_indices(mask.to(device=pred.device, dtype=torch.bool))
        batch_indices = torch.arange(pred.shape[0], device=pred.device)
        return F.mse_loss(pred[batch_indices, final_indices, :], gt[batch_indices, final_indices, :])
    return F.mse_loss(pred[:, -1, :], gt[:, -1, :])


def combined_trajectory_loss(
    pred: torch.Tensor,
    gt: torch.Tensor,
    mask: torch.Tensor | None = None,
    endpoint_weight: float = 0.0,
) -> torch.Tensor:
    base_loss = trajectory_smooth_l1_loss(pred, gt, mask)
    if endpoint_weight <= 0.0:
        return base_loss
    return base_loss + endpoint_weight * endpoint_loss(pred, gt, mask)
