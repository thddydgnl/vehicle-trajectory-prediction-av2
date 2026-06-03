from pathlib import Path

from src.utils.config import load_yaml_config


ROOT = Path(__file__).resolve().parents[1]
RUN_DIR_1EPOCH = "D:/runs/vehicle_trajectory_project/full_pilot_1epoch"
RUN_DIR_5EPOCH = "D:/runs/vehicle_trajectory_project/full_pilot_5epoch"
RUN_DIR_TUNING = "D:/runs/vehicle_trajectory_project/full_long_tuning"
RUN_DIR_FINAL = "D:/runs/vehicle_trajectory_project/full_long_final"


def test_full_pilot_configs_are_cuda_1epoch_windows_configs() -> None:
    expected = {
        "full_pilot_lstm_1epoch.yaml": ("lstm_full_pilot_1epoch", "lstm", 64),
        "full_pilot_transformer_1epoch.yaml": ("transformer_full_pilot_1epoch", "transformer", 32),
        "full_pilot_diffusion_pca_1epoch.yaml": ("diffusion_pca_full_pilot_1epoch", "diffusion_pca", 64),
        "full_pilot_diffusion_direct_1epoch.yaml": ("diffusion_direct_full_pilot_1epoch", "diffusion_direct", 64),
    }

    for filename, (model_name, architecture, batch_size) in expected.items():
        config = load_yaml_config(ROOT / "configs" / filename)
        assert config["model"]["name"] == model_name
        assert config["model"]["architecture"] == architecture
        assert config["training"]["epochs"] == 1
        assert config["training"]["batch_size"] == batch_size
        assert config["training"]["device"] == "cuda"
        assert config["training"]["num_workers"] == 0
        assert config["training"]["out_dir"] == RUN_DIR_1EPOCH


def test_full_pilot_diffusion_pca_uses_run_local_codec() -> None:
    config = load_yaml_config(ROOT / "configs" / "full_pilot_diffusion_pca_1epoch.yaml")
    assert config["model"]["codec_path"] == f"{RUN_DIR_1EPOCH}/checkpoints/pca_codec.pkl"


def test_full_pilot_configs_are_cuda_5epoch_windows_configs() -> None:
    expected = {
        "full_pilot_lstm_5epoch.yaml": ("lstm_full_pilot_5epoch", "lstm", 64),
        "full_pilot_transformer_5epoch.yaml": ("transformer_full_pilot_5epoch", "transformer", 32),
        "full_pilot_diffusion_pca_5epoch.yaml": ("diffusion_pca_full_pilot_5epoch", "diffusion_pca", 64),
        "full_pilot_diffusion_direct_5epoch.yaml": ("diffusion_direct_full_pilot_5epoch", "diffusion_direct", 64),
    }

    for filename, (model_name, architecture, batch_size) in expected.items():
        config = load_yaml_config(ROOT / "configs" / filename)
        assert config["model"]["name"] == model_name
        assert config["model"]["architecture"] == architecture
        assert config["training"]["epochs"] == 5
        assert config["training"]["batch_size"] == batch_size
        assert config["training"]["device"] == "cuda"
        assert config["training"]["num_workers"] == 0
        assert config["training"]["out_dir"] == RUN_DIR_5EPOCH
        assert config["training"]["early_stopping_patience"] == 5


def test_full_pilot_5epoch_diffusion_pca_uses_run_local_codec() -> None:
    config = load_yaml_config(ROOT / "configs" / "full_pilot_diffusion_pca_5epoch.yaml")
    assert config["model"]["codec_path"] == f"{RUN_DIR_5EPOCH}/checkpoints/pca_codec.pkl"


def test_full_long_configs_are_cuda_windows_configs() -> None:
    expected = {
        "full_long_lstm.yaml": ("lstm_full_long", "lstm", 30, 64),
        "full_long_transformer.yaml": ("transformer_full_long", "transformer", 30, 32),
    }

    for filename, (model_name, architecture, epochs, batch_size) in expected.items():
        config = load_yaml_config(ROOT / "configs" / filename)
        assert config["model"]["name"] == model_name
        assert config["model"]["architecture"] == architecture
        assert config["training"]["epochs"] == epochs
        assert config["training"]["batch_size"] == batch_size
        assert config["training"]["device"] == "cuda"
        assert config["training"]["num_workers"] == 0
        assert config["training"]["out_dir"] == RUN_DIR_FINAL


def test_diffusion_tuning_configs_are_cuda_windows_configs() -> None:
    expected = {
        "full_tune_diffusion_pca_a.yaml": ("diffusion_pca_tune_a", "diffusion_pca", 64),
        "full_tune_diffusion_pca_b.yaml": ("diffusion_pca_tune_b", "diffusion_pca", 64),
        "full_tune_diffusion_pca_c.yaml": ("diffusion_pca_tune_c", "diffusion_pca", 64),
        "full_tune_diffusion_pca_d.yaml": ("diffusion_pca_tune_d", "diffusion_pca", 64),
        "full_tune_diffusion_pca_e.yaml": ("diffusion_pca_tune_e", "diffusion_pca", 64),
        "full_tune_diffusion_pca_f.yaml": ("diffusion_pca_tune_f", "diffusion_pca", 32),
        "full_tune_diffusion_direct_a.yaml": ("diffusion_direct_tune_a", "diffusion_direct", 64),
        "full_tune_diffusion_direct_b.yaml": ("diffusion_direct_tune_b", "diffusion_direct", 32),
        "full_tune_diffusion_direct_c.yaml": ("diffusion_direct_tune_c", "diffusion_direct", 32),
        "full_tune_diffusion_direct_d.yaml": ("diffusion_direct_tune_d", "diffusion_direct", 32),
        "full_tune_diffusion_direct_e.yaml": ("diffusion_direct_tune_e", "diffusion_direct", 32),
        "full_tune_diffusion_direct_f.yaml": ("diffusion_direct_tune_f", "diffusion_direct", 16),
    }

    for filename, (model_name, architecture, batch_size) in expected.items():
        config = load_yaml_config(ROOT / "configs" / filename)
        assert config["model"]["name"] == model_name
        assert config["model"]["architecture"] == architecture
        assert config["model"]["diffusion_steps"] == 1000
        assert config["training"]["epochs"] == 15
        assert config["training"]["batch_size"] == batch_size
        assert config["training"]["device"] == "cuda"
        assert config["training"]["num_workers"] == 0
        assert config["training"]["out_dir"] == RUN_DIR_TUNING
        assert config["model"]["num_samples"] == 16
        assert config["training"]["selection_metric"] == "minADE"
        assert config["training"]["validation_num_samples"] == 8
        assert config["training"]["validation_seed"] == 1234


def test_diffusion_tuning_pca_configs_use_tuning_codec() -> None:
    expected = {
        "a": "pca_codec_12.pkl",
        "b": "pca_codec_12.pkl",
        "c": "pca_codec_12.pkl",
        "d": "pca_codec_16.pkl",
        "e": "pca_codec_24.pkl",
        "f": "pca_codec_24.pkl",
    }
    for suffix, codec_name in expected.items():
        config = load_yaml_config(ROOT / "configs" / f"full_tune_diffusion_pca_{suffix}.yaml")
        assert config["model"]["codec_path"] == f"{RUN_DIR_TUNING}/checkpoints/{codec_name}"


def test_full_long_windows_script_can_resume_completed_final_stages() -> None:
    script = (ROOT / "scripts" / "windows_full_long_experiments.ps1").read_text(encoding="utf-8")
    assert "ExpectedArtifacts" in script
    assert "Invoke-OrSkipCompleted" in script
    assert "completed_with_artifacts_after_nonzero_exit" in script
    assert "skipped_existing_artifacts" in script
    assert "metrics\\lstm_full_long_val_metrics.json" in script
    assert "metrics\\transformer_full_long_val_metrics.json" in script
