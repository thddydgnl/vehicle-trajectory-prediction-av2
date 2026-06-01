import torch

from src.models.lstm import LSTMForecast


def test_lstm_forecast_output_shape() -> None:
    model = LSTMForecast(input_dim=6, pred_len=30, hidden_dim=16, num_layers=1, dropout=0.0)
    x = torch.randn(4, 50, 6)

    pred = model(x)

    assert pred.shape == (4, 30, 2)


def test_lstm_forecast_teacher_forcing_shape() -> None:
    model = LSTMForecast(input_dim=6, pred_len=30, hidden_dim=16, num_layers=1, dropout=0.0)
    model.train()
    x = torch.randn(2, 50, 6)
    teacher_y = torch.randn(2, 30, 2)

    pred = model(x, teacher_y=teacher_y, teacher_forcing_ratio=1.0)

    assert pred.shape == (2, 30, 2)
