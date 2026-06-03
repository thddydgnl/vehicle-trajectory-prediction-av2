import torch
import pytest

from src.evaluation.metrics import ade, count_parameters, fde, min_ade, min_fde, miss_rate


def test_ade_matches_exact_value() -> None:
    pred = torch.tensor([[[0.0, 0.0], [3.0, 4.0]], [[0.0, 0.0], [0.0, 2.0]]])
    gt = torch.zeros_like(pred)

    assert torch.isclose(ade(pred, gt), torch.tensor(1.75))


def test_masked_ade_uses_only_valid_steps() -> None:
    pred = torch.tensor([[[3.0, 4.0], [100.0, 0.0]], [[0.0, 2.0], [0.0, 0.0]]])
    gt = torch.zeros_like(pred)
    mask = torch.tensor([[True, False], [True, False]])

    assert torch.isclose(ade(pred, gt, mask), torch.tensor(3.5))


def test_fde_matches_exact_value() -> None:
    pred = torch.tensor([[[0.0, 0.0], [3.0, 4.0]], [[0.0, 0.0], [0.0, 2.0]]])
    gt = torch.zeros_like(pred)

    assert torch.isclose(fde(pred, gt), torch.tensor(3.5))


def test_fde_can_use_final_valid_masked_step() -> None:
    pred = torch.tensor([[[3.0, 4.0], [100.0, 0.0]], [[0.0, 0.0], [0.0, 2.0]]])
    gt = torch.zeros_like(pred)
    mask = torch.tensor([[True, False], [True, True]])

    assert torch.isclose(fde(pred, gt, mask), torch.tensor(3.5))


def test_min_ade_chooses_best_sample_per_batch_item() -> None:
    gt = torch.zeros((2, 2, 2))
    pred_samples = torch.tensor(
        [
            [
                [[3.0, 4.0], [3.0, 4.0]],
                [[1.0, 0.0], [1.0, 0.0]],
            ],
            [
                [[0.0, 2.0], [0.0, 2.0]],
                [[0.0, 4.0], [0.0, 4.0]],
            ],
        ]
    )

    assert torch.isclose(min_ade(pred_samples, gt), torch.tensor(1.5))


def test_min_ade_respects_mask() -> None:
    gt = torch.zeros((1, 2, 2))
    pred_samples = torch.tensor([[[[5.0, 0.0], [100.0, 0.0]], [[3.0, 0.0], [200.0, 0.0]]]])
    mask = torch.tensor([[True, False]])

    assert torch.isclose(min_ade(pred_samples, gt, mask), torch.tensor(3.0))


def test_min_fde_chooses_best_final_sample() -> None:
    gt = torch.zeros((1, 2, 2))
    pred_samples = torch.tensor([[[[0.0, 0.0], [5.0, 0.0]], [[0.0, 0.0], [1.0, 0.0]]]])

    assert torch.isclose(min_fde(pred_samples, gt), torch.tensor(1.0))


def test_min_fde_can_use_final_valid_masked_step() -> None:
    gt = torch.zeros((1, 2, 2))
    pred_samples = torch.tensor([[[[4.0, 0.0], [0.0, 0.0]], [[2.0, 0.0], [100.0, 0.0]]]])
    mask = torch.tensor([[True, False]])

    assert torch.isclose(min_fde(pred_samples, gt, mask), torch.tensor(2.0))


def test_miss_rate_uses_final_displacement_threshold() -> None:
    gt = torch.zeros((3, 2, 2))
    pred = torch.tensor(
        [
            [[0.0, 0.0], [1.0, 0.0]],
            [[0.0, 0.0], [3.0, 0.0]],
            [[0.0, 0.0], [2.0, 0.0]],
        ]
    )

    assert torch.isclose(miss_rate(pred, gt, threshold=2.0), torch.tensor(1.0 / 3.0))


def test_miss_rate_can_use_final_valid_masked_step() -> None:
    gt = torch.zeros((2, 2, 2))
    pred = torch.tensor([[[3.0, 0.0], [0.0, 0.0]], [[1.0, 0.0], [4.0, 0.0]]])
    mask = torch.tensor([[True, False], [True, True]])

    assert torch.isclose(miss_rate(pred, gt, threshold=2.0, mask=mask), torch.tensor(1.0))


def test_count_parameters_counts_only_trainable_parameters() -> None:
    model = torch.nn.Sequential(torch.nn.Linear(2, 3), torch.nn.Linear(3, 1))
    for parameter in model[1].parameters():
        parameter.requires_grad = False

    assert count_parameters(model) == 9


def test_ade_rejects_invalid_mask_shape() -> None:
    pred = torch.zeros((2, 3, 2))
    gt = torch.zeros_like(pred)
    mask = torch.ones((2, 1), dtype=torch.bool)

    with pytest.raises(ValueError, match="mask shape"):
        ade(pred, gt, mask)


def test_ade_rejects_all_false_mask() -> None:
    pred = torch.zeros((2, 3, 2))
    gt = torch.zeros_like(pred)
    mask = torch.zeros((2, 3), dtype=torch.bool)

    with pytest.raises(ValueError, match="at least one valid"):
        ade(pred, gt, mask)


def test_min_ade_rejects_invalid_mask_shape() -> None:
    pred_samples = torch.zeros((2, 2, 3, 2))
    gt = torch.zeros((2, 3, 2))
    mask = torch.ones((1, 3), dtype=torch.bool)

    with pytest.raises(ValueError, match="mask shape"):
        min_ade(pred_samples, gt, mask)


def test_min_ade_rejects_all_false_batch_item_mask() -> None:
    pred_samples = torch.zeros((2, 2, 3, 2))
    gt = torch.zeros((2, 3, 2))
    mask = torch.tensor([[True, False, False], [False, False, False]])

    with pytest.raises(ValueError, match="each batch item"):
        min_ade(pred_samples, gt, mask)


def test_metrics_reject_empty_time_dimension() -> None:
    pred = torch.zeros((1, 0, 2))
    gt = torch.zeros_like(pred)

    with pytest.raises(ValueError, match="non-empty"):
        fde(pred, gt)


def test_min_metrics_reject_empty_sample_dimension() -> None:
    pred_samples = torch.zeros((1, 0, 3, 2))
    gt = torch.zeros((1, 3, 2))

    with pytest.raises(ValueError, match="non-empty"):
        min_fde(pred_samples, gt)
