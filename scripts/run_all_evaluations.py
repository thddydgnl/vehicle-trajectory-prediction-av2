from __future__ import annotations

import argparse
import csv
import shutil
import sys
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.evaluate import (
    evaluate_diffusion_direct,
    evaluate_diffusion_pca,
    evaluate_linear,
    evaluate_lstm,
    evaluate_transformer,
)
from src.utils.paths import ensure_dir
from src.utils.seed import set_seed


MetricDict = dict[str, float | int | str]


MODEL_ORDER = ("linear", "lstm", "transformer", "diffusion_direct", "diffusion_pca")
PREDICTION_FILENAMES = {
    "linear": "linear_val.pkl",
    "lstm": "lstm_val.pkl",
    "transformer": "transformer_val.pkl",
    "diffusion_direct": "diffusion_direct_val.pkl",
    "diffusion_pca": "diffusion_pca_val.pkl",
}


def _metric(metrics: MetricDict, key: str, fallback: str | None = None) -> float | int | str:
    if key in metrics:
        return metrics[key]
    if fallback is not None and fallback in metrics:
        return metrics[fallback]
    return ""


def comparison_row(metrics: MetricDict, data_split: str, target_type: str) -> dict[str, float | int | str]:
    has_min_metrics = "minADE" in metrics and "minFDE" in metrics
    latency = float(metrics.get("Latency", 0.0))
    notes = "multi-sample evaluation" if has_min_metrics else "single prediction; min metrics equal ADE/FDE"
    if "num_prediction_samples" in metrics:
        notes += f"; K={metrics['num_prediction_samples']}"
    return {
        "model": str(metrics["model"]),
        "data_split": data_split,
        "target_type": target_type,
        "ADE": _metric(metrics, "ADE"),
        "FDE": _metric(metrics, "FDE"),
        "minADE": _metric(metrics, "minADE", fallback="ADE"),
        "minFDE": _metric(metrics, "minFDE", fallback="FDE"),
        "Sample_Diversity": _metric(metrics, "Sample_Diversity"),
        "Miss_Rate": _metric(metrics, "Miss Rate"),
        "Latency_ms": latency * 1000.0,
        "Params": _metric(metrics, "Parameters"),
        "Notes": notes,
    }


def write_model_comparison(rows: list[dict[str, float | int | str]], out_dir: str | Path) -> dict[str, Path]:
    tables_dir = ensure_dir(Path(out_dir) / "tables")
    csv_path = tables_dir / "model_comparison.csv"
    md_path = tables_dir / "model_comparison.md"
    columns = ["model", "data_split", "target_type", "ADE", "FDE", "minADE", "minFDE", "Sample_Diversity", "Miss_Rate", "Latency_ms", "Params", "Notes"]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    with md_path.open("w", encoding="utf-8") as f:
        f.write("| " + " | ".join(columns) + " |\n")
        f.write("| " + " | ".join(["---"] * len(columns)) + " |\n")
        for row in rows:
            f.write("| " + " | ".join(_format_markdown_cell(row[column]) for column in columns) + " |\n")
    return {"csv": csv_path, "markdown": md_path}


def _format_markdown_cell(value: float | int | str) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _checkpoint_arg(args: argparse.Namespace, model: str) -> Path | None:
    explicit_checkpoint = {
        "lstm": args.lstm_checkpoint,
        "transformer": args.transformer_checkpoint,
        "diffusion_direct": args.diffusion_direct_checkpoint,
        "diffusion_pca": args.diffusion_pca_checkpoint,
    }[model]
    if explicit_checkpoint is not None:
        return explicit_checkpoint
    if args.checkpoint_dir is not None:
        checkpoint_tag = f"_{args.checkpoint_tag}" if args.checkpoint_tag else ""
        return args.checkpoint_dir / f"best_{model}{checkpoint_tag}.pt"
    return None


def _checkpoint_message(model: str, checkpoint: Path | None) -> str:
    if checkpoint is None:
        return f"{model}: checkpoint is required; pass --{model}_checkpoint or --checkpoint_dir [--checkpoint_tag]"
    return f"{model}: missing checkpoint {checkpoint}"


def resolve_requested_checkpoints(args: argparse.Namespace) -> tuple[dict[str, Path], list[str]]:
    checkpoints: dict[str, Path] = {}
    skipped: list[str] = []
    for model in args.models:
        if model == "linear":
            continue
        checkpoint = _checkpoint_arg(args, model)
        if checkpoint is not None and checkpoint.exists():
            checkpoints[model] = checkpoint
            continue

        message = _checkpoint_message(model, checkpoint)
        if args.allow_missing_models:
            skipped.append(message)
            continue
        raise FileNotFoundError(message)
    return checkpoints, skipped


def archive_prediction(model: str, out_dir: str | Path, prediction_tag: str | None) -> Path | None:
    if not prediction_tag:
        return None
    source = Path(out_dir) / "predictions" / PREDICTION_FILENAMES[model]
    if not source.exists():
        raise FileNotFoundError(f"Expected prediction payload was not written: {source}")
    target_dir = ensure_dir(Path(out_dir) / "predictions" / prediction_tag)
    target = target_dir / source.name
    shutil.copy2(source, target)
    return target


def run_all_evaluations(args: argparse.Namespace) -> tuple[list[dict[str, float | int | str]], list[str]]:
    evaluators: dict[str, Callable[..., MetricDict]] = {
        "lstm": evaluate_lstm,
        "transformer": evaluate_transformer,
        "diffusion_direct": evaluate_diffusion_direct,
        "diffusion_pca": evaluate_diffusion_pca,
    }
    rows: list[dict[str, float | int | str]] = []
    checkpoints, skipped = resolve_requested_checkpoints(args)

    for model in args.models:
        set_seed(args.seed)
        if model == "linear":
            metrics = evaluate_linear(args.data, args.linear_config, args.out_dir)
        else:
            checkpoint = checkpoints.get(model)
            if checkpoint is None:
                continue
            metrics = evaluators[model](args.data, checkpoint, args.out_dir, batch_size=args.batch_size)
        archive_prediction(model, args.out_dir, args.prediction_tag)
        rows.append(comparison_row(metrics, args.data_split, args.target_type))

    if not rows:
        raise ValueError("No models were evaluated")
    write_model_comparison(rows, args.out_dir)
    return rows, skipped


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run available model evaluations and create a comparison table.")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, default=Path("outputs"))
    parser.add_argument("--models", nargs="+", choices=MODEL_ORDER, default=list(MODEL_ORDER))
    parser.add_argument("--linear_config", type=Path, default=Path("configs/linear.yaml"))
    parser.add_argument("--lstm_checkpoint", type=Path, default=None)
    parser.add_argument("--transformer_checkpoint", type=Path, default=None)
    parser.add_argument("--diffusion_direct_checkpoint", type=Path, default=None)
    parser.add_argument("--diffusion_pca_checkpoint", type=Path, default=None)
    parser.add_argument("--checkpoint_dir", type=Path, default=None)
    parser.add_argument("--checkpoint_tag", default=None)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--data_split", default="val")
    parser.add_argument("--target_type", default="mixed")
    parser.add_argument("--prediction_tag", default=None)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--allow_missing_models", action="store_true")
    parser.add_argument("--strict_models", action="store_true", help="Deprecated: strict checkpoint checking is now the default.")
    return parser.parse_args()


def main() -> None:
    rows, skipped = run_all_evaluations(parse_args())
    print(f"evaluated_models: {', '.join(str(row['model']) for row in rows)}")
    if skipped:
        print("skipped_models:")
        for item in skipped:
            print(f"- {item}")


if __name__ == "__main__":
    main()
