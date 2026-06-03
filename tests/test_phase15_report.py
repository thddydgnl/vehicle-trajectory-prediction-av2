from __future__ import annotations

from pathlib import Path

import pandas as pd

from scripts.build_phase15_report import build_report


def test_build_phase15_report_from_real_tables(tmp_path: Path) -> None:
    comparison = tmp_path / "model_comparison.csv"
    tuning = tmp_path / "diffusion_tuning_summary.csv"
    out = tmp_path / "report_summary.md"
    pd.DataFrame(
        [
            {
                "model": "linear",
                "data_split": "val_full",
                "target_type": "av2_focal_mixed",
                "ADE": 1.5,
                "FDE": 3.7,
                "minADE": 1.5,
                "minFDE": 3.7,
                "Sample_Diversity": None,
                "Miss_Rate": 0.5,
                "Latency_ms": 0.02,
                "Params": 0,
                "Notes": "single prediction",
            },
            {
                "model": "transformer",
                "data_split": "val_full",
                "target_type": "av2_focal_mixed",
                "ADE": 0.9,
                "FDE": 2.2,
                "minADE": 0.9,
                "minFDE": 2.2,
                "Sample_Diversity": None,
                "Miss_Rate": 0.4,
                "Latency_ms": 0.3,
                "Params": 71420,
                "Notes": "single prediction",
            },
        ]
    ).to_csv(comparison, index=False)
    pd.DataFrame(
        [
            {
                "model": "diffusion_pca",
                "candidate_id": "pca_b",
                "target_gate": True,
                "selected": True,
                "ADE": 1.5,
                "FDE": 3.8,
                "minADE": 0.45,
                "minFDE": 0.94,
                "Sample_Diversity": 1.6,
                "epochs_ran": 15,
            }
        ]
    ).to_csv(tuning, index=False)

    path = build_report(comparison, out, tuning)

    text = path.read_text(encoding="utf-8")
    assert "Phase 15 Final Report Summary" in text
    assert "transformer" in text
    assert "best-of-K" in text
    assert "pca_b" in text
