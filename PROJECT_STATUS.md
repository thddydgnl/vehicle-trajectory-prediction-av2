# Vehicle Trajectory Project Status

작성일: 2026-06-01

This file is the persistent progress ledger for Codex goal mode.

## Current State

```text
Phase 0 repository setup is complete.
Project folder now contains the base Python package, utilities, tests, config/data/output directories, and documentation.
Git repository has been initialized on branch main.
GitHub remote has been configured and pushed.
Windows SSH over Tailscale has been verified.
Windows NVIDIA GPU is visible, but default PyTorch is CPU-only.
Windows conda currently has only the base environment; vehicle_traj does not exist yet.
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
Phase 1  Synthetic Smoke Dataset                  pending
Phase 2  Geometry and Coordinate Transform        pending
Phase 3  Dataset and DataLoader                   pending
Phase 4  Metrics                                  pending
Phase 5  Linear Baseline                          pending
Phase 6  Common Training Pipeline                 pending
Phase 7  LSTM Encoder-Decoder                     pending
Phase 8  Transformer Encoder                      pending
Phase 9  Direct Diffusion Model                   pending
Phase 10 PCA Latent Diffusion                     pending
Phase 11 Argoverse 2 Preprocessing                pending
Phase 12 Visualization                            pending
Phase 13 PCA and K-means Analysis                 pending
Phase 14 Final Experiment Matrix                  pending
Phase 15 Final Report Assets                      pending
```

## Next Recommended Task

```text
Start Phase 1 only.
Create the synthetic smoke dataset generator and tests.
Do not implement geometry, dataloader, metrics, or models yet.
```

## Latest Verified Commands

```text
ssh song@100.87.219.58 'hostname && whoami'
ssh song@100.87.219.58 'powershell -NoProfile -ExecutionPolicy Bypass -Command "python --version; where.exe python; nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader"'
ssh song@100.87.219.58 'powershell -NoProfile -ExecutionPolicy Bypass -Command "python -c \"import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 0)\""'
ssh song@100.87.219.58 'powershell -NoProfile -ExecutionPolicy Bypass -Command "conda env list"'
pytest -q
python -c "from src.utils.device import get_device; print(get_device())"
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
```

## Open External Requirements

```text
AV2 raw data is not present yet.
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
