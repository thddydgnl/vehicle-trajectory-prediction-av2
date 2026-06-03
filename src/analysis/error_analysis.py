from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis.common import collect_model_outputs, load_processed_npz, per_sample_metrics
from src.utils.paths import ensure_dir


def run_error_analysis(
    data: str | Path,
    predictions_dir: str | Path | None,
    out_dir: str | Path,
    top_k: int = 20,
    miss_threshold: float = 2.0,
    required_models: tuple[str, ...] | None = None,
) -> dict[str, Path]:
    dataset = load_processed_npz(data)
    outputs = collect_model_outputs(dataset, predictions_dir, required_models=required_models)
    tables_dir = ensure_dir(Path(out_dir) / "tables")
    summary_rows = []
    top_rows = []

    for model_name, model_output in outputs.items():
        metrics = per_sample_metrics(
            model_output["pred"],
            dataset["Y"],
            dataset["mask_y"],
            samples=model_output.get("samples"),
            miss_threshold=miss_threshold,
        )
        summary_rows.append(
            {
                "model": model_name,
                "count": int(metrics["ADE"].shape[0]),
                "ADE": float(np.nanmean(metrics["ADE"])),
                "FDE": float(np.nanmean(metrics["FDE"])),
                "minADE": float(np.nanmean(metrics["minADE"])),
                "minFDE": float(np.nanmean(metrics["minFDE"])),
                "Miss Rate": float(np.nanmean(metrics["Miss Rate"])),
                "minMiss Rate": float(np.nanmean(metrics["minMiss Rate"])),
            }
        )
        top_indices = np.argsort(metrics["FDE"])[::-1][:top_k]
        for rank, index in enumerate(top_indices, start=1):
            top_rows.append(
                {
                    "model": model_name,
                    "rank": rank,
                    "sample_index": int(index),
                    "scenario_id": str(dataset["scenario_id"][index]),
                    "track_id": str(dataset["track_id"][index]),
                    "object_type": int(dataset["object_type"][index]),
                    "ADE": float(metrics["ADE"][index]),
                    "FDE": float(metrics["FDE"][index]),
                    "minADE": float(metrics["minADE"][index]),
                    "minFDE": float(metrics["minFDE"][index]),
                    "miss": int(metrics["Miss Rate"][index] > 0.0),
                    "min_miss": int(metrics["minMiss Rate"][index] > 0.0),
                }
            )

    summary_path = tables_dir / "error_summary.csv"
    top_path = tables_dir / "top_error_cases.csv"
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    pd.DataFrame(top_rows).to_csv(top_path, index=False)
    return {"summary": summary_path, "top_cases": top_path}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create per-model trajectory error summaries.")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, default=Path("outputs/predictions"))
    parser.add_argument("--out_dir", type=Path, default=Path("outputs"))
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--miss_threshold", type=float, default=2.0)
    parser.add_argument("--required_models", nargs="*", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = run_error_analysis(
        args.data,
        args.predictions,
        args.out_dir,
        args.top_k,
        args.miss_threshold,
        tuple(args.required_models) if args.required_models else None,
    )
    for key, path in paths.items():
        print(f"{key}: {path}")


if __name__ == "__main__":
    main()
