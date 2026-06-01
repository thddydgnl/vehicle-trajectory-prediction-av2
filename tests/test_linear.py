import torch

from src.models.linear import LinearExtrapolation


def test_linear_extrapolation_uses_last_position_and_velocity() -> None:
    model = LinearExtrapolation(pred_len=3, dt=0.5)
    X = torch.zeros((2, 5, 6), dtype=torch.float32)
    X[0, -1, 0:2] = torch.tensor([1.0, 2.0])
    X[0, -1, 2:4] = torch.tensor([2.0, -1.0])
    X[1, -1, 0:2] = torch.tensor([0.0, 0.0])
    X[1, -1, 2:4] = torch.tensor([0.0, 4.0])

    pred = model.predict(X)

    expected = torch.tensor(
        [
            [[2.0, 1.5], [3.0, 1.0], [4.0, 0.5]],
            [[0.0, 2.0], [0.0, 4.0], [0.0, 6.0]],
        ]
    )
    assert torch.allclose(pred, expected)
