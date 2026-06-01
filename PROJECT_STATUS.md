# Vehicle Trajectory Project Status

작성일: 2026-06-01

This file is the persistent progress ledger for Codex goal mode.

## Current State

```text
Phase 0 repository setup is complete.
Project folder now contains the base Python package, utilities, tests, config/data/output directories, and documentation.
Git repository has been initialized on branch main.
GitHub remote has been configured and pushed.
Long-form workflow documents have been moved under docs/.
Windows SSH over Tailscale has been verified.
Windows NVIDIA GPU is visible, but default PyTorch is CPU-only.
Windows conda currently has only the base environment; vehicle_traj does not exist yet.
Windows AV2 data path has been created.
s5cmd v2.3.0 has been installed on Windows from the official GitHub release binary.
The first AV2 download attempt was stopped; no s5cmd process or scheduled download task is running.
```

Verified Windows access:

```text
Mac Tailscale IP: 100.75.150.25
Windows Tailscale IP: 100.87.219.58
SSH user: song
Command: ssh song@100.87.219.58 'hostname && whoami'
Output: Song / song\song
```

## Phase Progress

```text
Phase 0  Repository Setup                         complete
Phase 1  Synthetic Smoke Dataset                  complete
Phase 2  Geometry and Coordinate Transform        complete
Phase 3  Dataset and DataLoader                   complete
Phase 4  Metrics                                  complete
Phase 5  Linear Baseline                          complete
Phase 6  Common Training Pipeline                 complete
Phase 7  LSTM Encoder-Decoder                     complete
Phase 8  Transformer Encoder                      complete
Phase 9  Direct Diffusion Model                   complete
Phase 10 PCA Latent Diffusion                     complete
Phase 11 Argoverse 2 Preprocessing                pending
Phase 12 Visualization                            pending
Phase 13 PCA and K-means Analysis                 pending
Phase 14 Final Experiment Matrix                  pending
Phase 15 Final Report Assets                      pending
```

## Next Recommended Task

```text
Phase 7 through Phase 10 are complete.
Stop this goal here. Do not start Phase 11 until the user explicitly asks.
Next future task will be Phase 11 Argoverse 2 Preprocessing with real AV2 data.
```

## Latest Verified Commands

```text
ssh song@100.87.219.58 'hostname && whoami'
ssh song@100.87.219.58 'powershell -NoProfile -ExecutionPolicy Bypass -Command "python --version; where.exe python; nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader"'
ssh song@100.87.219.58 'powershell -NoProfile -ExecutionPolicy Bypass -Command "python -c \"import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 0)\""'
ssh song@100.87.219.58 'powershell -NoProfile -ExecutionPolicy Bypass -Command "conda env list"'
pytest -q
python -c "from src.utils.device import get_device; print(get_device())"
python -m src.datasets.synthetic --out_dir data/processed --num_samples 1000
pytest tests/test_synthetic_data.py -q
pytest tests/test_geometry.py -q
pytest tests/test_metrics.py -q
python -m src.evaluation.evaluate --model linear --data data/processed/val_smoke.npz --config configs/linear.yaml --out_dir outputs
pytest tests/test_losses.py -q
python -m src.training.train --config configs/lstm.yaml --max_epochs 1 --data data/processed/train_smoke.npz --val_data data/processed/val_smoke.npz
pytest tests/test_models_shape.py -q
python -m src.evaluation.evaluate --model lstm --checkpoint outputs/checkpoints/best_lstm.pt --data data/processed/val_smoke.npz --out_dir outputs
python -m src.training.train --config configs/transformer.yaml --max_epochs 1 --data data/processed/train_smoke.npz --val_data data/processed/val_smoke.npz
python -m src.evaluation.evaluate --model transformer --checkpoint outputs/checkpoints/best_transformer.pt --data data/processed/val_smoke.npz --out_dir outputs
pytest tests/test_diffusion_step.py -q
python -m src.training.train --config configs/diffusion_direct.yaml --max_epochs 1 --data data/processed/train_smoke.npz --val_data data/processed/val_smoke.npz
python -m src.evaluation.evaluate --model diffusion_direct --checkpoint outputs/checkpoints/best_diffusion_direct.pt --data data/processed/val_smoke.npz --out_dir outputs --batch_size 64
pytest tests/test_pca_latent.py -q
python -m src.analysis.pca_analysis --train_data data/processed/train_smoke.npz --out_dir outputs
python -m src.training.train --config configs/diffusion_pca.yaml --max_epochs 1 --data data/processed/train_smoke.npz --val_data data/processed/val_smoke.npz
python - <<'PY'
import torch
ckpt = torch.load('outputs/checkpoints/best_lstm_smoke.pt', map_location='cpu')
print(sorted(ckpt.keys()))
print(ckpt['metadata']['device'])
print(ckpt['metadata']['model']['architecture'])
PY
pytest -q
```

Result:

```text
Song
song\song
Python 3.13.5
C:\Users\thddy\anaconda3\python.exe
NVIDIA GeForce RTX 3080, 591.86, 10240 MiB
torch 2.10.0+cpu
torch.cuda.is_available() == False
conda envs: base only
pytest: 5 passed
get_device: mps
s5cmd: v2.3.0-991c9fb at C:\Users\thddy\bin\s5cmd\s5cmd.exe
Windows AV2 partial state: annotation parquet files downloaded; test split partially created; train/val not downloaded; INCOMPLETE_DOWNLOAD.txt marker written
Stop verification: taskkill terminated s5cmd.exe PID 27984; later tasklist found no s5cmd.exe; VehicleTrajectoryAV2Download task not present
Tailscale ping after stop: pong from song in 11ms
OpenSSH event log: publickey accepted/disconnected cleanly; no server-side OpenSSH error observed
Phase 1 synthetic generator: created ignored train_smoke.npz, val_smoke.npz, and test_smoke.npz
Phase 1 tests: tests/test_synthetic_data.py passed, full pytest passed 10 tests
Phase 2 geometry: relative/global transforms and wrap_angle tests passed; full pytest passed 16 tests
Phase 2 subagent review: no coordinate transform issues found
Phase 3 dataset/dataloader: synthetic generation passed, tests/test_synthetic_data.py passed 7 tests, full pytest passed 18 tests
Phase 4 metrics: tests/test_metrics.py passed 14 tests, full pytest passed 32 tests
Phase 4 subagent review: mask shape/all-false mask findings were fixed before commit
Phase 5 linear baseline: evaluation CLI produced ADE, FDE, Miss Rate, Latency, and Parameters on val_smoke; full pytest passed 33 tests
Phase 6 common training: tiny_regressor smoke training completed for 1 epoch on CPU; best/last checkpoint, train log, and val metrics were generated
Phase 6 tests: tests/test_losses.py passed 5 tests, metrics/loss tests passed 22 tests, full pytest passed 41 tests
Phase 6 checkpoint audit: best_lstm_smoke.pt includes model/optimizer state, trainer_config, and metadata with device=cpu and architecture=tiny_regressor
Phase 6 subagent review: mask-aware FDE/endpoint, valid-step loss aggregation, and checkpoint metadata findings were fixed before commit
Phase 7 LSTM: tests/test_models_shape.py passed 2 tests; 1-epoch synthetic smoke training completed on CPU; LSTM checkpoint evaluation produced ADE, FDE, Miss Rate, Latency, and Parameters; full pytest passed 43 tests
Phase 8 Transformer: tests/test_models_shape.py passed 4 tests; 1-epoch synthetic smoke training completed on CPU; Transformer checkpoint evaluation produced ADE, FDE, Miss Rate, Latency, and Parameters; full pytest passed 45 tests
Phase 9 direct diffusion: tests/test_diffusion_step.py passed 5 tests; 1-epoch synthetic smoke training completed on CPU; diffusion checkpoint evaluation produced ADE/FDE and minADE/minFDE with 4 samples; full pytest passed 50 tests
Phase 10 PCA latent diffusion: tests/test_pca_latent.py passed 3 tests; PCA codec fit on train_smoke and wrote ignored outputs/checkpoints/pca_codec.pkl; PCA explained variance figure generated; diffusion_pca 1-epoch synthetic smoke training completed on CPU; full pytest passed 53 tests
```

## Open External Requirements

```text
AV2 raw data is not present yet.
Full AV2 download is not complete. Do not treat the current Windows AV2 folder as a complete dataset.
For the next AV2 download attempt, do not use a long foreground SSH command; use the safe remote execution rule in docs/windows_gpu_training_only_workflow.md.
Windows vehicle_traj CUDA PyTorch environment must be created before long model training.
```

## GitHub

```text
Repository: https://github.com/thddydgnl/vehicle-trajectory-prediction-av2
Remote: origin
Default branch: main
Latest pushed branch: main
```

## Update Rule

After every completed Phase, update:

```text
1. Phase Progress
2. Next Recommended Task
3. Latest Verified Commands
4. Open External Requirements
```

Do not mark a Phase complete unless its required validation command ran and the
result is recorded.
