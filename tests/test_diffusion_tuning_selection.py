from __future__ import annotations

import json
from pathlib import Path

import yaml

from scripts.select_diffusion_tuning import select_and_write


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_config(path: Path, model_name: str, architecture: str, final_dir: Path) -> None:
    payload = {
        "model": {
            "name": model_name,
            "architecture": architecture,
            "input_dim": 6,
            "pred_len": 30,
            "trajectory_dim": 60,
            "latent_dim": 12,
            "codec_path": str(final_dir / "old_codec.pkl"),
            "cond_dim": 64,
            "hidden_dim": 128,
            "diffusion_steps": 50,
            "sampling_steps": 10,
            "num_samples": 8,
        },
        "training": {
            "batch_size": 64,
            "epochs": 10,
            "learning_rate": 0.0002,
            "weight_decay": 0.0001,
            "loss": "noise_mse",
            "gradient_clip": 1.0,
            "early_stopping_patience": 6,
            "device": "cuda",
            "num_workers": 0,
            "out_dir": str(final_dir / "old"),
        },
    }
    if architecture == "diffusion_pca":
        payload["model"].pop("trajectory_dim")
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def test_select_diffusion_tuning_generates_final_configs(tmp_path: Path) -> None:
    tuning_dir = tmp_path / "tuning"
    final_dir = tmp_path / "final"
    configs_dir = tmp_path / "configs"
    configs_dir.mkdir()
    pca_config = configs_dir / "pca.yaml"
    direct_config = configs_dir / "direct.yaml"
    _write_config(pca_config, "diffusion_pca_tune_a", "diffusion_pca", final_dir)
    _write_config(direct_config, "diffusion_direct_tune_a", "diffusion_direct", final_dir)

    pca_checkpoint = tuning_dir / "checkpoints" / "best_diffusion_pca_tune_a.pt"
    direct_checkpoint = tuning_dir / "checkpoints" / "best_diffusion_direct_tune_a.pt"
    pca_checkpoint.parent.mkdir(parents=True)
    pca_checkpoint.touch()
    direct_checkpoint.touch()

    _write_json(tuning_dir / "metrics" / "diffusion_pca_tune_a_val_metrics.json", {"epochs_ran": 10})
    _write_json(tuning_dir / "metrics" / "diffusion_direct_tune_a_val_metrics.json", {"epochs_ran": 10})
    _write_json(
        tuning_dir / "evaluations" / "pca_a" / "metrics" / "diffusion_pca_eval_metrics.json",
        {"ADE": 4.0, "FDE": 8.0, "minADE": 3.5, "minFDE": 7.5, "Sample_Diversity": 0.5},
    )
    _write_json(
        tuning_dir / "evaluations" / "direct_a" / "metrics" / "diffusion_direct_eval_metrics.json",
        {"ADE": 7.0, "FDE": 14.0, "minADE": 6.5, "minFDE": 13.0, "Sample_Diversity": 0.5},
    )

    matrix = {
        "run": {
            "tuning_dir": str(tuning_dir),
            "final_dir": str(final_dir),
            "final_epochs": 30,
            "final_early_stopping_patience": 10,
            "require_target_gate_for_final": True,
        },
        "baselines": {
            "diffusion_pca_f4": {"minADE": 6.0, "minFDE": 12.0},
            "diffusion_direct_f4": {"minADE": 10.0, "minFDE": 20.0},
        },
        "gates": {
            "min_epochs_ran": 5,
            "min_sample_diversity": 0.001,
            "preferred_min_metric_improvement_pct": 10.0,
        },
        "target_gates": {
            "diffusion_pca": {"max_minADE": 4.8, "max_minFDE": 9.5},
            "diffusion_direct": {"max_minADE": 8.0, "max_minFDE": 15.0},
        },
        "candidates": {
            "diffusion_pca": [
                {
                    "id": "pca_a",
                    "model_name": "diffusion_pca_tune_a",
                    "config": str(pca_config),
                    "checkpoint": str(pca_checkpoint),
                    "train_metrics": str(tuning_dir / "metrics" / "diffusion_pca_tune_a_val_metrics.json"),
                    "eval_metrics": str(tuning_dir / "evaluations" / "pca_a" / "metrics" / "diffusion_pca_eval_metrics.json"),
                }
            ],
            "diffusion_direct": [
                {
                    "id": "direct_a",
                    "model_name": "diffusion_direct_tune_a",
                    "config": str(direct_config),
                    "checkpoint": str(direct_checkpoint),
                    "train_metrics": str(tuning_dir / "metrics" / "diffusion_direct_tune_a_val_metrics.json"),
                    "eval_metrics": str(tuning_dir / "evaluations" / "direct_a" / "metrics" / "diffusion_direct_eval_metrics.json"),
                }
            ],
        },
    }
    matrix_path = tmp_path / "matrix.yaml"
    matrix_path.write_text(yaml.safe_dump(matrix, sort_keys=False), encoding="utf-8")

    selection = select_and_write(matrix_path, tmp_path)

    assert selection["selected"]["diffusion_pca"]["candidate_id"] == "pca_a"
    assert selection["selected"]["diffusion_direct"]["candidate_id"] == "direct_a"
    assert selection["full_run_ready"] is True
    pca_final = Path(selection["final_configs"]["diffusion_pca"])
    direct_final = Path(selection["final_configs"]["diffusion_direct"])
    assert pca_final.exists()
    assert direct_final.exists()
    assert yaml.safe_load(pca_final.read_text(encoding="utf-8"))["model"]["name"] == "diffusion_pca_full_long"
    assert yaml.safe_load(direct_final.read_text(encoding="utf-8"))["training"]["epochs"] == 30
    assert (tuning_dir / "tables" / "diffusion_tuning_summary.csv").exists()


def test_select_diffusion_tuning_blocks_final_configs_when_target_gates_fail(tmp_path: Path) -> None:
    tuning_dir = tmp_path / "tuning"
    final_dir = tmp_path / "final"
    configs_dir = tmp_path / "configs"
    configs_dir.mkdir()
    pca_config = configs_dir / "pca.yaml"
    direct_config = configs_dir / "direct.yaml"
    _write_config(pca_config, "diffusion_pca_tune_a", "diffusion_pca", final_dir)
    _write_config(direct_config, "diffusion_direct_tune_a", "diffusion_direct", final_dir)

    pca_checkpoint = tuning_dir / "checkpoints" / "best_diffusion_pca_tune_a.pt"
    direct_checkpoint = tuning_dir / "checkpoints" / "best_diffusion_direct_tune_a.pt"
    pca_checkpoint.parent.mkdir(parents=True)
    pca_checkpoint.touch()
    direct_checkpoint.touch()

    _write_json(tuning_dir / "metrics" / "diffusion_pca_tune_a_val_metrics.json", {"epochs_ran": 10})
    _write_json(tuning_dir / "metrics" / "diffusion_direct_tune_a_val_metrics.json", {"epochs_ran": 10})
    _write_json(
        tuning_dir / "evaluations" / "pca_a" / "metrics" / "diffusion_pca_eval_metrics.json",
        {"ADE": 5.0, "FDE": 10.0, "minADE": 4.9, "minFDE": 9.4, "Sample_Diversity": 0.5},
    )
    _write_json(
        tuning_dir / "evaluations" / "direct_a" / "metrics" / "diffusion_direct_eval_metrics.json",
        {"ADE": 8.0, "FDE": 16.0, "minADE": 7.9, "minFDE": 15.1, "Sample_Diversity": 0.5},
    )
    matrix = {
        "run": {
            "tuning_dir": str(tuning_dir),
            "final_dir": str(final_dir),
            "final_epochs": 30,
            "final_early_stopping_patience": 10,
            "require_target_gate_for_final": True,
        },
        "baselines": {
            "diffusion_pca_f4": {"minADE": 6.0, "minFDE": 12.0},
            "diffusion_direct_f4": {"minADE": 10.0, "minFDE": 20.0},
        },
        "gates": {
            "min_epochs_ran": 5,
            "min_sample_diversity": 0.001,
            "preferred_min_metric_improvement_pct": 10.0,
        },
        "target_gates": {
            "diffusion_pca": {"max_minADE": 4.8, "max_minFDE": 9.5},
            "diffusion_direct": {"max_minADE": 8.0, "max_minFDE": 15.0},
        },
        "candidates": {
            "diffusion_pca": [
                {
                    "id": "pca_a",
                    "model_name": "diffusion_pca_tune_a",
                    "config": str(pca_config),
                    "checkpoint": str(pca_checkpoint),
                    "train_metrics": str(tuning_dir / "metrics" / "diffusion_pca_tune_a_val_metrics.json"),
                    "eval_metrics": str(tuning_dir / "evaluations" / "pca_a" / "metrics" / "diffusion_pca_eval_metrics.json"),
                }
            ],
            "diffusion_direct": [
                {
                    "id": "direct_a",
                    "model_name": "diffusion_direct_tune_a",
                    "config": str(direct_config),
                    "checkpoint": str(direct_checkpoint),
                    "train_metrics": str(tuning_dir / "metrics" / "diffusion_direct_tune_a_val_metrics.json"),
                    "eval_metrics": str(tuning_dir / "evaluations" / "direct_a" / "metrics" / "diffusion_direct_eval_metrics.json"),
                }
            ],
        },
    }
    matrix_path = tmp_path / "matrix.yaml"
    matrix_path.write_text(yaml.safe_dump(matrix, sort_keys=False), encoding="utf-8")

    selection = select_and_write(matrix_path, tmp_path)

    assert selection["full_run_ready"] is False
    assert selection["final_configs"] == {}
    assert selection["selected"]["diffusion_pca"]["target_gate"] is False
    assert selection["selected"]["diffusion_direct"]["target_gate"] is False
