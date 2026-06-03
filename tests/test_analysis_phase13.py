from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.analysis.common import collect_model_outputs, per_sample_metrics, summarize_metrics
from src.analysis.error_analysis import run_error_analysis
from src.analysis.kmeans_analysis import run_kmeans_analysis
from src.analysis.pca_analysis import run_pca_analysis


def _write_npz(path: Path, num_samples: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    obs_len = 50
    pred_len = 30
    X = np.zeros((num_samples, obs_len, 6), dtype=np.float32)
    Y = np.zeros((num_samples, pred_len, 2), dtype=np.float32)
    for idx in range(num_samples):
        speed = 0.2 + 0.02 * idx
        curve = (idx % 4 - 1.5) * 0.04
        X[idx, :, 0] = np.linspace(-obs_len + 1, 0, obs_len) * speed
        X[idx, :, 2] = speed / 0.1
        X[idx, :, 5] = 1.0
        steps = np.arange(1, pred_len + 1, dtype=np.float32)
        Y[idx, :, 0] = steps * speed
        Y[idx, :, 1] = curve * steps**1.4 + rng.normal(0.0, 0.01, size=pred_len)
    np.savez_compressed(
        path,
        X=X,
        Y=Y,
        mask_x=np.ones((num_samples, obs_len), dtype=bool),
        mask_y=np.ones((num_samples, pred_len), dtype=bool),
        object_type=np.array([idx % 2 for idx in range(num_samples)], dtype=np.int64),
        scenario_id=np.array([f"s{idx}" for idx in range(num_samples)], dtype=object),
        track_id=np.array([f"t{idx}" for idx in range(num_samples)], dtype=object),
        origin=np.zeros((num_samples, 2), dtype=np.float32),
        theta=np.zeros((num_samples,), dtype=np.float32),
    )


def _write_predictions(predictions_dir: Path, data_path: Path) -> None:
    predictions_dir.mkdir(parents=True)
    with np.load(data_path, allow_pickle=True) as data:
        gt = data["Y"].astype(np.float32)
        mask_y = data["mask_y"]
    pred = gt + 0.25
    samples = np.stack([gt + 0.25, gt + 0.1, gt - 0.1], axis=1).astype(np.float32)
    with (predictions_dir / "lstm_val.pkl").open("wb") as f:
        pickle.dump({"pred": pred, "gt": gt, "mask_y": mask_y}, f)
    with (predictions_dir / "diffusion_direct_val.pkl").open("wb") as f:
        pickle.dump({"samples": samples, "gt": gt, "mask_y": mask_y}, f)


def test_phase13_analysis_outputs(tmp_path: Path) -> None:
    train_data = tmp_path / "train.npz"
    val_data = tmp_path / "val.npz"
    predictions_dir = tmp_path / "predictions"
    out_dir = tmp_path / "outputs"
    _write_npz(train_data, num_samples=32, seed=1)
    _write_npz(val_data, num_samples=20, seed=2)
    _write_predictions(predictions_dir, val_data)

    pca_paths = run_pca_analysis(train_data, out_dir, n_components=6, data=val_data)
    kmeans_paths = run_kmeans_analysis(train_data, val_data, predictions_dir, out_dir, n_components=6, n_clusters=4)
    error_paths = run_error_analysis(val_data, predictions_dir, out_dir, top_k=3)

    for path in [*pca_paths.values(), *kmeans_paths.values(), *error_paths.values()]:
        assert path.exists()
        assert path.stat().st_size > 0

    cluster_summary = pd.read_csv(kmeans_paths["cluster_summary"])
    cluster_metrics = pd.read_csv(kmeans_paths["cluster_metrics"])
    error_summary = pd.read_csv(error_paths["summary"])
    assert {"cluster", "count", "vehicle_count", "pedestrian_count"}.issubset(cluster_summary.columns)
    assert {"cluster", "model", "ADE", "FDE", "minADE", "minFDE", "Miss Rate"}.issubset(cluster_metrics.columns)
    assert {"linear", "lstm", "diffusion_direct"}.issubset(set(cluster_metrics["model"]))
    assert {"model", "ADE", "FDE", "minADE", "minFDE", "Miss Rate"}.issubset(error_summary.columns)


def test_analysis_rejects_misaligned_prediction_payload(tmp_path: Path) -> None:
    data_path = tmp_path / "val.npz"
    predictions_dir = tmp_path / "predictions"
    _write_npz(data_path, num_samples=8, seed=3)
    predictions_dir.mkdir()
    with np.load(data_path, allow_pickle=True) as data:
        gt = data["Y"].copy()
        mask_y = data["mask_y"].copy()
        payload = {
            "pred": gt + 0.1,
            "gt": gt[::-1],
            "mask_y": mask_y,
            "scenario_id": [str(value) for value in data["scenario_id"]],
            "track_id": [str(value) for value in data["track_id"]],
        }
    with (predictions_dir / "lstm_val.pkl").open("wb") as f:
        pickle.dump(payload, f)

    data = dict(np.load(data_path, allow_pickle=True))
    with pytest.raises(ValueError, match="payload gt"):
        collect_model_outputs(data, predictions_dir)


def test_analysis_rejects_prediction_shape_mismatch_and_missing_required_model(tmp_path: Path) -> None:
    data_path = tmp_path / "val.npz"
    predictions_dir = tmp_path / "predictions"
    _write_npz(data_path, num_samples=8, seed=4)
    predictions_dir.mkdir()
    with np.load(data_path, allow_pickle=True) as data:
        payload = {"pred": data["Y"][:4].copy()}
        loaded = {key: data[key] for key in data.files}
    with (predictions_dir / "lstm_val.pkl").open("wb") as f:
        pickle.dump(payload, f)

    with pytest.raises(ValueError, match="prediction shape"):
        collect_model_outputs(loaded, predictions_dir)
    with pytest.raises(ValueError, match="Missing required"):
        collect_model_outputs(loaded, tmp_path / "empty", required_models=("transformer",))


def test_per_sample_metrics_use_best_of_k_min_miss_and_global_ade_summary() -> None:
    gt = np.zeros((2, 2, 2), dtype=np.float32)
    pred = np.array([[[3.0, 0.0], [3.0, 0.0]], [[1.0, 0.0], [1.0, 0.0]]], dtype=np.float32)
    samples = np.array(
        [
            [
                [[3.0, 0.0], [3.0, 0.0]],
                [[1.0, 0.0], [1.0, 0.0]],
            ],
            [
                [[1.0, 0.0], [1.0, 0.0]],
                [[0.5, 0.0], [0.5, 0.0]],
            ],
        ],
        dtype=np.float32,
    )
    mask = np.array([[True, False], [True, True]])

    metrics = per_sample_metrics(pred, gt, mask, samples=samples, miss_threshold=2.0)
    summary = summarize_metrics(metrics, np.array([0, 1]))

    assert np.isclose(metrics["Miss Rate"][0], 1.0)
    assert np.isclose(metrics["minMiss Rate"][0], 0.0)
    assert np.isclose(summary["ADE"], 5.0 / 3.0)
