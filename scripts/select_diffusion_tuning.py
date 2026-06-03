from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import yaml

from src.utils.paths import ensure_dir


MODEL_BASELINES = {
    "diffusion_pca": "diffusion_pca_f4",
    "diffusion_direct": "diffusion_direct_f4",
}
FINAL_MODEL_NAMES = {
    "diffusion_pca": "diffusion_pca_full_long",
    "diffusion_direct": "diffusion_direct_full_long",
}
FINAL_CONFIG_FILENAMES = {
    "diffusion_pca": "full_long_diffusion_pca.yaml",
    "diffusion_direct": "full_long_diffusion_direct.yaml",
}


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in {path}")
    return data


def _load_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8-sig") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in {path}")
    return data


def _resolve(path_value: str | Path, repo_root: Path) -> Path:
    path = Path(str(path_value))
    if path.is_absolute() or (len(str(path)) > 2 and str(path)[1:3] == ":/"):
        return path
    return repo_root / path


def _float_metric(metrics: dict[str, Any] | None, key: str) -> float:
    if metrics is None or key not in metrics:
        return float("nan")
    return float(metrics[key])


def _is_finite(*values: float) -> bool:
    return all(math.isfinite(value) for value in values)


def _pct_improvement(baseline: float, value: float) -> float:
    if not math.isfinite(baseline) or baseline <= 0 or not math.isfinite(value):
        return float("nan")
    return (baseline - value) / baseline * 100.0


def _score(row: dict[str, Any]) -> float:
    min_ade = float(row["minADE"])
    min_fde = float(row["minFDE"])
    if not _is_finite(min_ade, min_fde):
        return float("inf")
    return min_ade + 0.25 * min_fde


def _markdown_cell(value: Any) -> str:
    if isinstance(value, float):
        if math.isnan(value):
            return ""
        return f"{value:.6g}"
    return str(value)


def _write_summary(rows: list[dict[str, Any]], output_dir: Path) -> dict[str, Path]:
    tables_dir = ensure_dir(output_dir / "tables")
    csv_path = tables_dir / "diffusion_tuning_summary.csv"
    md_path = tables_dir / "diffusion_tuning_summary.md"
    fieldnames = [
        "model",
        "candidate_id",
        "config",
        "hard_gate",
        "preferred_gate",
        "selected",
        "epochs_ran",
        "ADE",
        "FDE",
        "minADE",
        "minFDE",
        "Sample_Diversity",
        "minADE_improvement_pct",
        "minFDE_improvement_pct",
        "score",
        "reason",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    with md_path.open("w", encoding="utf-8") as f:
        f.write("| " + " | ".join(fieldnames) + " |\n")
        f.write("| " + " | ".join(["---"] * len(fieldnames)) + " |\n")
        for row in rows:
            f.write("| " + " | ".join(_markdown_cell(row[name]) for name in fieldnames) + " |\n")
    return {"csv": csv_path, "markdown": md_path}


def _candidate_row(model_key: str, candidate: dict[str, Any], matrix: dict[str, Any], repo_root: Path) -> dict[str, Any]:
    baselines = matrix["baselines"]
    gates = matrix["gates"]
    baseline = baselines[MODEL_BASELINES[model_key]]
    train_metrics_path = _resolve(candidate["train_metrics"], repo_root)
    eval_metrics_path = _resolve(candidate["eval_metrics"], repo_root)
    checkpoint_path = _resolve(candidate["checkpoint"], repo_root)
    train_metrics = _load_json_if_exists(train_metrics_path)
    eval_metrics = _load_json_if_exists(eval_metrics_path)

    ade = _float_metric(eval_metrics, "ADE")
    fde = _float_metric(eval_metrics, "FDE")
    min_ade = _float_metric(eval_metrics, "minADE")
    min_fde = _float_metric(eval_metrics, "minFDE")
    diversity = _float_metric(eval_metrics, "Sample_Diversity")
    epochs_ran = int(train_metrics.get("epochs_ran", 0)) if train_metrics is not None else 0
    min_ade_improvement = _pct_improvement(float(baseline["minADE"]), min_ade)
    min_fde_improvement = _pct_improvement(float(baseline["minFDE"]), min_fde)

    reasons: list[str] = []
    if train_metrics is None:
        reasons.append("missing train metrics")
    if eval_metrics is None:
        reasons.append("missing eval metrics")
    if not checkpoint_path.exists():
        reasons.append("missing checkpoint")
    if epochs_ran < int(gates["min_epochs_ran"]):
        reasons.append("too few epochs")
    if not _is_finite(ade, fde, min_ade, min_fde, diversity):
        reasons.append("non-finite metric")
    if math.isfinite(diversity) and diversity < float(gates["min_sample_diversity"]):
        reasons.append("low sample diversity")

    hard_gate = len(reasons) == 0
    preferred_improvement = float(gates["preferred_min_metric_improvement_pct"])
    preferred_gate = hard_gate and max(min_ade_improvement, min_fde_improvement) >= preferred_improvement
    row: dict[str, Any] = {
        "model": model_key,
        "candidate_id": candidate["id"],
        "config": candidate["config"],
        "hard_gate": hard_gate,
        "preferred_gate": preferred_gate,
        "selected": False,
        "epochs_ran": epochs_ran,
        "ADE": ade,
        "FDE": fde,
        "minADE": min_ade,
        "minFDE": min_fde,
        "Sample_Diversity": diversity,
        "minADE_improvement_pct": min_ade_improvement,
        "minFDE_improvement_pct": min_fde_improvement,
        "reason": "; ".join(reasons) if reasons else "passed",
    }
    row["score"] = _score(row)
    return row


def _select_candidate(rows: list[dict[str, Any]], model_key: str) -> dict[str, Any]:
    eligible = [row for row in rows if row["model"] == model_key and row["hard_gate"]]
    if not eligible:
        raise RuntimeError(f"No {model_key} candidates passed hard gates")
    preferred = [row for row in eligible if row["preferred_gate"]]
    pool = preferred or eligible
    selected = min(pool, key=lambda row: float(row["score"]))
    selected["selected"] = True
    if not preferred:
        selected["reason"] = selected["reason"] + "; selected as best hard-gate fallback"
    return selected


def _candidate_by_id(matrix: dict[str, Any], model_key: str, candidate_id: str) -> dict[str, Any]:
    for candidate in matrix["candidates"][model_key]:
        if candidate["id"] == candidate_id:
            return candidate
    raise KeyError(candidate_id)


def _write_final_config(
    matrix: dict[str, Any],
    model_key: str,
    selected: dict[str, Any],
    repo_root: Path,
) -> Path:
    run = matrix["run"]
    final_dir = Path(str(run["final_dir"]))
    generated_dir = ensure_dir(final_dir / "generated_configs")
    source_candidate = _candidate_by_id(matrix, model_key, str(selected["candidate_id"]))
    config = _load_yaml(_resolve(source_candidate["config"], repo_root))
    config["model"]["name"] = FINAL_MODEL_NAMES[model_key]
    if model_key == "diffusion_pca":
        config["model"]["codec_path"] = str(final_dir / "checkpoints" / "pca_codec.pkl").replace("\\", "/")
    config["training"]["epochs"] = int(run["final_epochs"])
    config["training"]["early_stopping_patience"] = int(run["final_early_stopping_patience"])
    config["training"]["out_dir"] = str(final_dir).replace("\\", "/")
    output_path = generated_dir / FINAL_CONFIG_FILENAMES[model_key]
    with output_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)
    return output_path


def select_and_write(matrix_path: Path, repo_root: Path) -> dict[str, Any]:
    matrix = _load_yaml(matrix_path)
    tuning_dir = Path(str(matrix["run"]["tuning_dir"]))
    rows: list[dict[str, Any]] = []
    for model_key, candidates in matrix["candidates"].items():
        for candidate in candidates:
            rows.append(_candidate_row(model_key, candidate, matrix, repo_root))

    selected = {
        "diffusion_pca": _select_candidate(rows, "diffusion_pca"),
        "diffusion_direct": _select_candidate(rows, "diffusion_direct"),
    }
    final_configs = {
        model_key: str(_write_final_config(matrix, model_key, row, repo_root))
        for model_key, row in selected.items()
    }
    summary_paths = _write_summary(rows, tuning_dir)
    selection = {
        "selected": selected,
        "final_configs": final_configs,
        "summary": {key: str(path) for key, path in summary_paths.items()},
    }
    selection_path = ensure_dir(tuning_dir / "tables") / "selected_diffusion_configs.json"
    with selection_path.open("w", encoding="utf-8") as f:
        json.dump(selection, f, indent=2)
    return selection


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select best diffusion tuning candidates and generate final configs.")
    parser.add_argument("--matrix", type=Path, default=Path("configs/full_diffusion_tuning_matrix.yaml"))
    parser.add_argument("--repo_root", type=Path, default=Path.cwd())
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selection = select_and_write(args.matrix, args.repo_root)
    print(json.dumps(selection, indent=2))


if __name__ == "__main__":
    main()
