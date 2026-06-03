from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

from src.visualization.plot_clusters import plot_kmeans_clusters
from src.visualization.plot_errors import plot_error_histograms
from src.visualization.plot_pca import plot_pca_trajectory_space
from src.visualization.plot_report_cases import plot_best_worst_cases, plot_diffusion_interesting_case
from src.visualization.plot_trajectories import plot_diffusion_samples, plot_model_overlay, plot_top_error_cases


def _write_processed_npz(path: Path, num_samples: int = 12) -> None:
    obs_len = 50
    pred_len = 30
    X = np.zeros((num_samples, obs_len, 6), dtype=np.float32)
    Y = np.zeros((num_samples, pred_len, 2), dtype=np.float32)
    for idx in range(num_samples):
        speed = 0.5 + 0.05 * idx
        X[idx, :, 0] = np.linspace(-obs_len + 1, 0, obs_len) * speed
        X[idx, :, 2] = speed / 0.1
        X[idx, :, 5] = 1.0
        Y[idx, :, 0] = np.arange(1, pred_len + 1) * speed
        Y[idx, :, 1] = 0.03 * idx * np.sin(np.linspace(0, np.pi, pred_len))
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
        gt = data["Y"]
        mask_y = data["mask_y"]
    pred = gt + 0.1
    with (predictions_dir / "lstm_val.pkl").open("wb") as f:
        pickle.dump({"pred": pred, "gt": gt, "mask_y": mask_y}, f)
    with (predictions_dir / "transformer_val.pkl").open("wb") as f:
        pickle.dump({"pred": gt + 0.05, "gt": gt, "mask_y": mask_y}, f)
    samples = np.stack([gt + offset for offset in (0.0, 0.1, -0.1)], axis=1).astype(np.float32)
    with (predictions_dir / "diffusion_direct_val.pkl").open("wb") as f:
        pickle.dump({"samples": samples, "gt": gt, "mask_y": mask_y}, f)


def test_visualization_scripts_create_pngs(tmp_path: Path) -> None:
    data_path = tmp_path / "val.npz"
    predictions_dir = tmp_path / "predictions"
    out_dir = tmp_path / "figures"
    _write_processed_npz(data_path)
    _write_predictions(predictions_dir, data_path)

    paths = [
        plot_model_overlay(data_path, predictions_dir, out_dir, num_cases=4),
        plot_diffusion_samples(data_path, predictions_dir, out_dir, num_cases=3),
        *plot_top_error_cases(data_path, predictions_dir, out_dir, num_cases=2),
        *plot_error_histograms(data_path, predictions_dir, out_dir),
        plot_pca_trajectory_space(data_path, out_dir),
        plot_kmeans_clusters(data_path, out_dir, n_clusters=3),
        *plot_best_worst_cases(data_path, predictions_dir, out_dir, num_cases=3),
        plot_diffusion_interesting_case(data_path, predictions_dir, out_dir),
    ]

    assert len(paths) >= 8
    for path in paths:
        assert path.exists()
        assert path.suffix == ".png"
        assert path.stat().st_size > 0
