from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


REQUIRED_COMPARISON_COLUMNS = {
    "model",
    "data_split",
    "target_type",
    "ADE",
    "FDE",
    "minADE",
    "minFDE",
    "Miss_Rate",
    "Params",
    "Notes",
}


def _fmt(value: object) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _markdown_table(df: pd.DataFrame, columns: list[str]) -> str:
    rows = df[columns].copy()
    formatted = rows.map(_fmt)
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = ["| " + " | ".join(str(value) for value in row) + " |" for row in formatted.to_numpy()]
    return "\n".join([header, separator, *body])


def _load_comparison(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Model comparison table not found: {path}")
    df = pd.read_csv(path)
    missing = sorted(REQUIRED_COMPARISON_COLUMNS.difference(df.columns))
    if missing:
        raise ValueError(f"Comparison table is missing required columns: {missing}")
    return df


def _selected_tuning_rows(path: Path | None) -> pd.DataFrame | None:
    if path is None or not path.exists():
        return None
    df = pd.read_csv(path)
    if "selected" not in df.columns:
        return None
    selected = df[df["selected"].astype(str).str.lower() == "true"].copy()
    return selected if not selected.empty else None


def build_report(
    comparison_path: str | Path,
    out_path: str | Path,
    tuning_summary_path: str | Path | None = None,
    analysis_dir: str | Path = "outputs/full_av2_analysis",
    figures_dir: str | Path = "outputs/full_av2/figures",
) -> Path:
    comparison = _load_comparison(Path(comparison_path))
    selected_tuning = _selected_tuning_rows(Path(tuning_summary_path) if tuning_summary_path else None)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    comparison_columns = [
        "model",
        "ADE",
        "FDE",
        "minADE",
        "minFDE",
        "Sample_Diversity" if "Sample_Diversity" in comparison.columns else "Miss_Rate",
        "Miss_Rate",
        "Params",
    ]
    comparison_columns = [column for column in comparison_columns if column in comparison.columns]
    best_row = comparison.sort_values("ADE", ascending=True).iloc[0]

    lines = [
        "# Phase 15 Final Report Summary",
        "",
        "## Project Problem",
        "",
        "This project predicts the next 3 seconds of vehicle/pedestrian motion from the previous 5 seconds of trajectory history. "
        "It compares Linear Extrapolation, LSTM, Transformer, Direct Diffusion, and PCA Latent Diffusion under the same processed AV2-style input format.",
        "",
        "## Dataset And Setup",
        "",
        "- Dataset: Argoverse 2 Motion Forecasting, focal vehicle/pedestrian target mix.",
        "- Input: 50 observed steps with relative position, velocity, and heading features.",
        "- Output: 30 future `(x, y)` steps in the last-observed-agent coordinate frame.",
        "- Full AV2 preprocessing and GPU training were run on Windows HOME; code, reports, and Git commits are managed from Mac.",
        "",
        "## Metrics",
        "",
        "- ADE/FDE evaluate the single saved trajectory prediction.",
        "- For diffusion models, minADE/minFDE are best-of-K metrics across generated samples; they should be interpreted as sample-set quality, not a single deterministic trajectory.",
        "- Miss Rate uses the configured FDE threshold from the evaluator.",
        "",
        "## Main Results",
        "",
        f"Best ADE model in the final comparison table: **{best_row['model']}** with ADE {_fmt(best_row['ADE'])} and FDE {_fmt(best_row['FDE'])}.",
        "",
        _markdown_table(comparison, comparison_columns),
        "",
    ]

    if selected_tuning is not None:
        tuning_columns = [
            "model",
            "candidate_id",
            "target_gate",
            "ADE",
            "FDE",
            "minADE",
            "minFDE",
            "Sample_Diversity",
            "epochs_ran",
        ]
        tuning_columns = [column for column in tuning_columns if column in selected_tuning.columns]
        lines.extend(
            [
                "## Diffusion Tuning Gate",
                "",
                "The final long run was allowed only after the selected PCA Diffusion and Direct Diffusion candidates passed the user-defined minADE/minFDE gates on real `val_full` outputs.",
                "",
                _markdown_table(selected_tuning, tuning_columns),
                "",
            ]
        )

    lines.extend(
        [
            "## Error Analysis And Figures",
            "",
            f"- Analysis tables: `{Path(analysis_dir) / 'tables'}`",
            f"- Report figures: `{Path(figures_dir)}`",
            "- Required Phase 15 figures include best-case overlays, worst-case overlays, diffusion sample interpretation, PCA, K-means, and error histograms.",
            "",
            "## Interpretation",
            "",
            "The deterministic sequence models show how much temporal modeling improves over a constant-velocity baseline. "
            "The diffusion models are evaluated separately with both single-sample ADE/FDE and best-of-K minADE/minFDE, which exposes whether the sampler can generate at least one plausible future even when an arbitrary first sample is weaker.",
            "",
            "## Limitations",
            "",
            "- The project does not use HD maps, lane graphs, traffic lights, or multi-agent interaction modeling.",
            "- Diffusion min metrics depend on the number of samples K and are not directly equivalent to deterministic ADE/FDE.",
            "- Tuning was performed on validation outputs for a school project setting, so final claims should stay scoped to this dataset split and experiment setup.",
            "",
            "## Future Work",
            "",
            "- Add map/lane context and multi-agent interaction features.",
            "- Separate vehicle and pedestrian reporting more deeply.",
            "- Explore stronger diffusion objectives, trajectory anchors, and calibration of sample diversity.",
            "",
            "## Artifact Policy",
            "",
            "Raw AV2 data, processed `.npz` files, checkpoints, logs, and prediction payloads are intentionally excluded from Git. "
            "The repository commits code, configs, tests, lightweight metrics, tables, figures, and this report summary.",
            "",
        ]
    )

    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the Phase 15 report summary from real result tables.")
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("outputs/report_summary.md"))
    parser.add_argument("--tuning_summary", type=Path, default=None)
    parser.add_argument("--analysis_dir", type=Path, default=Path("outputs/full_av2_analysis"))
    parser.add_argument("--figures_dir", type=Path, default=Path("outputs/full_av2/figures"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path = build_report(
        comparison_path=args.comparison,
        out_path=args.out,
        tuning_summary_path=args.tuning_summary,
        analysis_dir=args.analysis_dir,
        figures_dir=args.figures_dir,
    )
    print(path)


if __name__ == "__main__":
    main()
