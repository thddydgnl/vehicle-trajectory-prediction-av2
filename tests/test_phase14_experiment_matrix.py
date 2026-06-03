from __future__ import annotations

from pathlib import Path

import pandas as pd

from scripts.run_all_evaluations import comparison_row, write_model_comparison


def test_comparison_row_fills_single_prediction_min_metrics() -> None:
    row = comparison_row(
        {
            "model": "linear",
            "ADE": 1.0,
            "FDE": 2.0,
            "Miss Rate": 0.25,
            "Latency": 0.001,
            "Parameters": 0,
        },
        data_split="val",
        target_type="mixed",
    )

    assert row["minADE"] == 1.0
    assert row["minFDE"] == 2.0
    assert row["Latency_ms"] == 1.0
    assert "single prediction" in str(row["Notes"])


def test_write_model_comparison_outputs_csv_and_markdown(tmp_path: Path) -> None:
    rows = [
        {
            "model": "linear",
            "data_split": "val",
            "target_type": "mixed",
            "ADE": 1.0,
            "FDE": 2.0,
            "minADE": 1.0,
            "minFDE": 2.0,
            "Miss_Rate": 0.25,
            "Latency_ms": 1.0,
            "Params": 0,
            "Notes": "single prediction",
        }
    ]

    paths = write_model_comparison(rows, tmp_path)

    assert paths["csv"].exists()
    assert paths["markdown"].exists()
    csv = pd.read_csv(paths["csv"])
    assert list(csv.columns) == ["model", "data_split", "target_type", "ADE", "FDE", "minADE", "minFDE", "Miss_Rate", "Latency_ms", "Params", "Notes"]
    assert "linear" in paths["markdown"].read_text(encoding="utf-8")
