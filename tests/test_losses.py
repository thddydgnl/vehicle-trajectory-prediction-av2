import pytest
import torch

from src.training.losses import (
    combined_trajectory_loss,
    endpoint_loss,
    trajectory_mse_loss,
    trajectory_smooth_l1_loss,
)


def test_trajectory_mse_loss_matches_expected_value() -> None:
    pred = torch.tensor([[[2.0, 0.0], [0.0, 2.0]]])
    gt = torch.zeros_like(pred)

    assert torch.isclose(trajectory_mse_loss(pred, gt), torch.tensor(2.0))


def test_trajectory_loss_respects_mask() -> None:
    pred = torch.tensor([[[2.0, 0.0], [100.0, 100.0]]])
    gt = torch.zeros_like(pred)
    mask = torch.tensor([[True, False]])

    assert torch.isclose(trajectory_smooth_l1_loss(pred, gt, mask), torch.tensor(0.75))


def test_endpoint_loss_uses_final_step() -> None:
    pred = torch.tensor([[[10.0, 10.0], [2.0, 4.0]]])
    gt = torch.zeros_like(pred)

    assert torch.isclose(endpoint_loss(pred, gt), torch.tensor(10.0))


def test_endpoint_loss_can_use_final_valid_masked_step() -> None:
    pred = torch.tensor([[[2.0, 4.0], [100.0, 100.0]]])
    gt = torch.zeros_like(pred)
    mask = torch.tensor([[True, False]])

    assert torch.isclose(endpoint_loss(pred, gt, mask), torch.tensor(10.0))


def test_combined_trajectory_loss_adds_endpoint_term() -> None:
    pred = torch.tensor([[[1.0, 1.0], [1.0, 1.0]]])
    gt = torch.zeros_like(pred)

    base = trajectory_smooth_l1_loss(pred, gt)
    endpoint = endpoint_loss(pred, gt)

    assert torch.isclose(combined_trajectory_loss(pred, gt, endpoint_weight=0.5), base + 0.5 * endpoint)


def test_loss_rejects_invalid_mask_shape() -> None:
    pred = torch.zeros((2, 3, 2))
    gt = torch.zeros_like(pred)
    mask = torch.ones((2, 1), dtype=torch.bool)

    with pytest.raises(ValueError, match="mask shape"):
        trajectory_mse_loss(pred, gt, mask)
