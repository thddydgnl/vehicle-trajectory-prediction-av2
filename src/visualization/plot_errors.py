from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.visualization.common import (
    available_predictions,
    load_processed_npz,
    per_sample_ade_fde,
    save_figure,
)


def _plot_histogram(
    values_by_model: dict[str, np.ndarray],
    metric_name: str,
    out_path: Path,
) -> Path:
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    for model_name, values in values_by_model.items():
        clean = values[np.isfinite(values)]
        if clean.size == 0:
            continue
        value_min = float(clean.min())
        value_max = float(clean.max())
        if value_min == value_max:
            hist_range = (value_min - 0.5, value_max + 0.5)
        else:
            margin = max((value_max - value_min) * 0.05, 1e-6)
            hist_range = (value_min - margin, value_max + margin)
        ax.hist(clean, bins=24, range=hist_range, alpha=0.45, label=model_name)
    ax.set_xlabel(f"{metric_name} (meters)")
    ax.set_ylabel("count")
    ax.set_title(f"{metric_name} Error Distribution")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    return save_figure(fig, out_path)


def plot_error_histograms(
    data_path: str | Path,
    predictions_dir: str | Path,
    out_dir: str | Path,
) -> list[Path]:
    data = load_processed_npz(data_path)
    predictions = available_predictions(data, predictions_dir)
    ade_by_model: dict[str, np.ndarray] = {}
    fde_by_model: dict[str, np.ndarray] = {}
    for model_name, pred in predictions.items():
        ade_values, fde_values = per_sample_ade_fde(pred, data["Y"], data["mask_y"])
        ade_by_model[model_name] = ade_values
        fde_by_model[model_name] = fde_values

    out = Path(out_dir)
    return [
        _plot_histogram(ade_by_model, "ADE", out / "error_histogram_ade.png"),
        _plot_histogram(fde_by_model, "FDE", out / "error_histogram_fde.png"),
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot trajectory error histograms.")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, default=Path("outputs/predictions"))
    parser.add_argument("--out_dir", type=Path, default=Path("outputs/figures"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for path in plot_error_histograms(args.data, args.predictions, args.out_dir):
        print(path)


if __name__ == "__main__":
    main()
