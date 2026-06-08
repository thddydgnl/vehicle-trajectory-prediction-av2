from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.visualization.common import (
    available_predictions,
    load_prediction_payloads,
    load_processed_npz,
    per_sample_ade_fde,
    prediction_samples,
    save_figure,
)
from src.visualization.plot_trajectories import MODEL_COLORS


DATA = Path("data/processed/val_full.npz")
PREDICTIONS = Path("outputs/predictions/full_long_final")
OUT_DIR = Path("outputs/presentation/figures")


LABELS = {
    "linear": "Linear",
    "lstm": "LSTM",
    "transformer": "Transformer",
    "diffusion_pca": "PCA Diff.",
    "diffusion_direct": "Direct Diff.",
}


def _bounds(ax: plt.Axes, arrays: list[np.ndarray]) -> None:
    points = [arr.reshape(-1, 2) for arr in arrays if arr.size > 0]
    stacked = np.concatenate(points, axis=0)
    x_min, y_min = stacked.min(axis=0)
    x_max, y_max = stacked.max(axis=0)
    span = max(float(x_max - x_min), float(y_max - y_min), 1.0)
    pad = span * 0.14
    x_mid = float((x_min + x_max) / 2.0)
    y_mid = float((y_min + y_max) / 2.0)
    half = span / 2.0 + pad
    ax.set_xlim(x_mid - half, x_mid + half)
    ax.set_ylim(y_mid - half, y_mid + half)


def _style_axis(ax: plt.Axes) -> None:
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.22)
    ax.set_xlabel("x (m)", fontsize=10)
    ax.set_ylabel("y (m)", fontsize=10)
    ax.tick_params(labelsize=9)


def _plot_case(
    data: dict[str, np.ndarray],
    predictions: dict[str, np.ndarray],
    index: int,
    path: Path,
) -> Path:
    fig, ax = plt.subplots(figsize=(6.4, 5.1), dpi=180)
    past = data["X"][index, :, 0:2]
    future = data["Y"][index]
    mask_x = data["mask_x"][index].astype(bool)
    mask_y = data["mask_y"][index].astype(bool)

    plotted = [past[mask_x], future[mask_y]]
    ax.plot(past[mask_x, 0], past[mask_x, 1], color="#111827", linewidth=3.0, label="Past")
    ax.plot(future[mask_y, 0], future[mask_y, 1], color="#f59e0b", linewidth=3.0, label="Ground truth")

    for model in ["linear", "lstm", "transformer", "diffusion_pca", "diffusion_direct"]:
        pred = predictions.get(model)
        if pred is None or index >= pred.shape[0]:
            continue
        ax.plot(
            pred[index, :, 0],
            pred[index, :, 1],
            linestyle="--",
            linewidth=2.0,
            color=MODEL_COLORS.get(model, "#64748b"),
            label=LABELS.get(model, model),
        )
        plotted.append(pred[index])

    ax.scatter([0.0], [0.0], color="#111827", s=30, zorder=4)
    _bounds(ax, plotted)
    _style_axis(ax)
    fig.tight_layout(rect=(0.02, 0.02, 0.98, 0.98))
    saved = save_figure(fig, path)
    plt.close(fig)
    return saved


def _sample_fde(samples: np.ndarray, gt: np.ndarray, mask_y: np.ndarray) -> np.ndarray:
    valid = mask_y.astype(bool)
    final_indices = valid.shape[1] - 1 - np.argmax(valid[:, ::-1], axis=1)
    return np.linalg.norm(
        samples[np.arange(samples.shape[0]), :, final_indices] - gt[np.arange(gt.shape[0]), None, final_indices],
        axis=-1,
    )


def _plot_diffusion_case(data: dict[str, np.ndarray], payloads: dict[str, dict], path: Path) -> Path:
    model = "diffusion_pca" if "diffusion_pca" in payloads else "diffusion_direct"
    samples = prediction_samples(payloads[model])
    if samples is None:
        raise ValueError("Diffusion prediction payload does not contain samples")

    fde = _sample_fde(samples, data["Y"], data["mask_y"])
    best_fde = fde.min(axis=1)
    first_fde = fde[:, 0]
    improvement = first_fde - best_fde
    index = int(np.nanargmax(improvement))
    for candidate in np.argsort(improvement)[::-1][:800]:
        sample_points = samples[int(candidate)].reshape(-1, 2)
        data_points = np.concatenate(
            [
                data["X"][int(candidate), :, 0:2][data["mask_x"][int(candidate)].astype(bool)],
                data["Y"][int(candidate)][data["mask_y"][int(candidate)].astype(bool)],
                sample_points,
            ],
            axis=0,
        )
        span = float(np.max(data_points.max(axis=0) - data_points.min(axis=0)))
        if 8.0 < span < 32.0 and best_fde[int(candidate)] < 2.0 and first_fde[int(candidate)] > 2.0:
            index = int(candidate)
            break
    best_sample = int(np.nanargmin(fde[index]))

    fig, ax = plt.subplots(figsize=(6.4, 5.1), dpi=180)
    past = data["X"][index, :, 0:2]
    future = data["Y"][index]
    mask_x = data["mask_x"][index].astype(bool)
    mask_y = data["mask_y"][index].astype(bool)
    plotted = [past[mask_x], future[mask_y]]

    ax.plot(past[mask_x, 0], past[mask_x, 1], color="#111827", linewidth=3.0, label="Past")
    ax.plot(future[mask_y, 0], future[mask_y, 1], color="#f59e0b", linewidth=3.0, label="Ground truth")
    for sample_idx in range(min(samples.shape[1], 16)):
        is_best = sample_idx == best_sample
        ax.plot(
            samples[index, sample_idx, :, 0],
            samples[index, sample_idx, :, 1],
            color="#16a34a" if is_best else "#9333ea",
            linewidth=3.0 if is_best else 1.2,
            alpha=0.95 if is_best else 0.18,
            label="Best sample" if is_best else ("Diffusion candidates" if sample_idx == 0 else None),
        )
        plotted.append(samples[index, sample_idx])

    _bounds(ax, plotted)
    _style_axis(ax)
    fig.tight_layout(rect=(0.02, 0.02, 0.98, 0.98))
    saved = save_figure(fig, path)
    plt.close(fig)
    return saved


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = load_processed_npz(DATA)
    predictions = available_predictions(data, PREDICTIONS)
    payloads = load_prediction_payloads(PREDICTIONS)

    _, lstm_fde = per_sample_ade_fde(predictions["lstm"], data["Y"], data["mask_y"])
    future_displacement = []
    for idx in range(data["Y"].shape[0]):
        valid = data["mask_y"][idx].astype(bool)
        future = data["Y"][idx][valid]
        future_displacement.append(float(np.linalg.norm(future[-1] - future[0])) if len(future) > 1 else 0.0)
    future_displacement = np.asarray(future_displacement)

    best_index = int(np.nanargmin(lstm_fde))
    for candidate in np.argsort(lstm_fde):
        if future_displacement[int(candidate)] > 12.0 and lstm_fde[int(candidate)] < 0.25:
            best_index = int(candidate)
            break

    worst_index = int(np.nanargmax(lstm_fde))
    for candidate in np.argsort(lstm_fde)[::-1]:
        if future_displacement[int(candidate)] > 15.0:
            worst_index = int(candidate)
            break

    paths = [
        _plot_case(data, predictions, best_index, OUT_DIR / "presentation_best_case_overlay.png"),
        _plot_case(data, predictions, worst_index, OUT_DIR / "presentation_worst_case_overlay.png"),
        _plot_diffusion_case(data, payloads, OUT_DIR / "presentation_diffusion_candidates.png"),
    ]
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
