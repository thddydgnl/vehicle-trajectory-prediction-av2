from __future__ import annotations

import argparse
from pathlib import Path
import pickle

import pandas as pd
import pytest

from scripts.run_all_evaluations import (
    archive_prediction,
    comparison_row,
    resolve_requested_checkpoints,
    run_all_evaluations,
    write_model_comparison,
)


def _args(tmp_path: Path, **overrides: object) -> argparse.Namespace:
    values: dict[str, object] = {
        "data": tmp_path / "missing.npz",
        "out_dir": tmp_path,
        "models": ["linear", "lstm"],
        "linear_config": Path("configs/linear.yaml"),
        "lstm_checkpoint": None,
        "transformer_checkpoint": None,
        "diffusion_direct_checkpoint": None,
        "diffusion_pca_checkpoint": None,
        "checkpoint_dir": None,
        "checkpoint_tag": None,
        "batch_size": 32,
        "data_split": "val",
        "target_type": "mixed",
        "prediction_tag": None,
        "allow_missing_models": False,
        "strict_models": False,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


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


def test_run_all_evaluations_requires_explicit_trainable_checkpoints(tmp_path: Path) -> None:
    args = _args(tmp_path)

    with pytest.raises(FileNotFoundError, match="lstm: checkpoint is required"):
        run_all_evaluations(args)

    assert not (tmp_path / "tables" / "model_comparison.csv").exists()


def test_checkpoint_dir_and_tag_resolve_phase14_checkpoint_names(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    checkpoint = checkpoint_dir / "best_lstm_phase14_smoke.pt"
    checkpoint.touch()
    args = _args(tmp_path, checkpoint_dir=checkpoint_dir, checkpoint_tag="phase14_smoke")

    checkpoints, skipped = resolve_requested_checkpoints(args)

    assert checkpoints["lstm"] == checkpoint
    assert skipped == []


def test_allow_missing_models_skips_missing_checkpoints(tmp_path: Path) -> None:
    args = _args(tmp_path, allow_missing_models=True)

    checkpoints, skipped = resolve_requested_checkpoints(args)

    assert checkpoints == {}
    assert skipped == ["lstm: checkpoint is required; pass --lstm_checkpoint or --checkpoint_dir [--checkpoint_tag]"]


def test_prediction_tag_archives_model_payload(tmp_path: Path) -> None:
    predictions_dir = tmp_path / "predictions"
    predictions_dir.mkdir()
    payload_path = predictions_dir / "lstm_val.pkl"
    with payload_path.open("wb") as f:
        pickle.dump({"pred": []}, f)

    archived = archive_prediction("lstm", tmp_path, "phase14_av2_small")

    assert archived == predictions_dir / "phase14_av2_small" / "lstm_val.pkl"
    assert archived.exists()
