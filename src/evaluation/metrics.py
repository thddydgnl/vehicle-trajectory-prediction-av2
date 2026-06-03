from __future__ import annotations

import torch


def _trajectory_errors(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    if pred.shape != gt.shape:
        raise ValueError(f"pred and gt must have the same shape, got {pred.shape} and {gt.shape}")
    if pred.ndim != 3 or pred.shape[-1] != 2:
        raise ValueError(f"Expected trajectory shape [B, T, 2], got {pred.shape}")
    if pred.shape[0] == 0 or pred.shape[1] == 0:
        raise ValueError(f"Trajectory batch and time dimensions must be non-empty, got {pred.shape}")
    return torch.linalg.norm(pred - gt, dim=-1)


def _masked_mean(errors: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    if mask is None:
        return errors.mean()
    mask_float = mask.to(device=errors.device, dtype=errors.dtype)
    if mask_float.shape != errors.shape:
        raise ValueError(f"mask shape must match errors shape, got {mask_float.shape} and {errors.shape}")
    denominator = mask_float.sum()
    if denominator <= 0:
        raise ValueError("mask must contain at least one valid trajectory step")
    return (errors * mask_float).sum() / denominator


def ade(pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    """Average displacement error over all valid trajectory steps."""
    return _masked_mean(_trajectory_errors(pred, gt), mask)


def _final_valid_indices(mask: torch.Tensor) -> torch.Tensor:
    if mask.ndim != 2:
        raise ValueError(f"mask shape must be [B, T], got {mask.shape}")
    valid_counts = mask.sum(dim=1)
    if torch.any(valid_counts <= 0):
        raise ValueError("each batch item must contain at least one valid trajectory step")
    step_indices = torch.arange(mask.shape[1], device=mask.device).expand(mask.shape[0], -1)
    return torch.where(mask, step_indices, torch.full_like(step_indices, -1)).max(dim=1).values


def fde(pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    """Final displacement error averaged over the batch."""
    errors = _trajectory_errors(pred, gt)
    if mask is not None:
        if mask.shape != errors.shape:
            raise ValueError(f"mask shape must be [B, T], got {mask.shape}, expected {errors.shape}")
        final_indices = _final_valid_indices(mask.to(device=errors.device, dtype=torch.bool))
        batch_indices = torch.arange(errors.shape[0], device=errors.device)
        return errors[batch_indices, final_indices].mean()
    return errors[:, -1].mean()


def min_ade(
    pred_samples: torch.Tensor,
    gt: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Best-sample ADE for multi-sample predictions."""
    if pred_samples.ndim != 4 or pred_samples.shape[-1] != 2:
        raise ValueError(f"Expected pred_samples shape [B, K, T, 2], got {pred_samples.shape}")
    if pred_samples.shape[0] == 0 or pred_samples.shape[1] == 0 or pred_samples.shape[2] == 0:
        raise ValueError(f"Batch, sample, and time dimensions must be non-empty, got {pred_samples.shape}")
    if pred_samples.shape[0] != gt.shape[0] or pred_samples.shape[2:] != gt.shape[1:]:
        raise ValueError(f"pred_samples and gt shapes are incompatible: {pred_samples.shape}, {gt.shape}")

    errors = torch.linalg.norm(pred_samples - gt[:, None, :, :], dim=-1)
    if mask is not None:
        if mask.shape != gt.shape[:2]:
            raise ValueError(f"mask shape must be [B, T], got {mask.shape}, expected {gt.shape[:2]}")
        mask_float = mask[:, None, :].to(device=errors.device, dtype=errors.dtype)
        denominator = mask_float.sum(dim=-1)
        if torch.any(denominator <= 0):
            raise ValueError("each batch item must contain at least one valid trajectory step")
        sample_ade = (errors * mask_float).sum(dim=-1) / denominator
    else:
        sample_ade = errors.mean(dim=-1)
    return sample_ade.min(dim=1).values.mean()


def min_fde(pred_samples: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    """Best-sample FDE for multi-sample predictions."""
    if pred_samples.ndim != 4 or pred_samples.shape[-1] != 2:
        raise ValueError(f"Expected pred_samples shape [B, K, T, 2], got {pred_samples.shape}")
    if pred_samples.shape[0] == 0 or pred_samples.shape[1] == 0 or pred_samples.shape[2] == 0:
        raise ValueError(f"Batch, sample, and time dimensions must be non-empty, got {pred_samples.shape}")
    if pred_samples.shape[0] != gt.shape[0] or pred_samples.shape[2:] != gt.shape[1:]:
        raise ValueError(f"pred_samples and gt shapes are incompatible: {pred_samples.shape}, {gt.shape}")

    if mask is not None:
        if mask.shape != gt.shape[:2]:
            raise ValueError(f"mask shape must be [B, T], got {mask.shape}, expected {gt.shape[:2]}")
        final_indices = _final_valid_indices(mask.to(device=pred_samples.device, dtype=torch.bool))
        batch_indices = torch.arange(pred_samples.shape[0], device=pred_samples.device)
        final_pred = pred_samples[batch_indices, :, final_indices, :]
        final_gt = gt.to(device=pred_samples.device)[batch_indices, final_indices, :]
        final_errors = torch.linalg.norm(final_pred - final_gt[:, None, :], dim=-1)
    else:
        final_errors = torch.linalg.norm(pred_samples[:, :, -1, :] - gt[:, None, -1, :], dim=-1)
    return final_errors.min(dim=1).values.mean()


def miss_rate(
    pred: torch.Tensor,
    gt: torch.Tensor,
    threshold: float = 2.0,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Fraction of trajectories whose final displacement error exceeds threshold."""
    errors = _trajectory_errors(pred, gt)
    if mask is not None:
        if mask.shape != errors.shape:
            raise ValueError(f"mask shape must be [B, T], got {mask.shape}, expected {errors.shape}")
        final_indices = _final_valid_indices(mask.to(device=errors.device, dtype=torch.bool))
        batch_indices = torch.arange(errors.shape[0], device=errors.device)
        return (errors[batch_indices, final_indices] > threshold).to(dtype=pred.dtype).mean()
    return (errors[:, -1] > threshold).to(dtype=pred.dtype).mean()


def count_parameters(model: torch.nn.Module) -> int:
    """Count trainable parameters in a torch module."""
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
