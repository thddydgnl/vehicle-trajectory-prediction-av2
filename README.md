# Vehicle Trajectory Prediction

Argoverse 2-style vehicle and pedestrian trajectory forecasting project. The
project predicts a future 3-second trajectory from 5 seconds of past motion and
will compare Linear Extrapolation, LSTM, Transformer, and Diffusion models under
the same data format and metrics.

## Current Status

Phase 0 is repository setup. Model, dataset generation, evaluation, and
visualization code are intentionally not implemented yet.

## Planned Scope

- Dataset format: Argoverse 2 Motion Forecasting style `.npz`
- Input: past 50 steps at 10 Hz
- Output: future 30 steps as `[30, 2]`
- Features: relative position, velocity, and heading features
- Metrics: ADE, FDE, Miss Rate, minADE/minFDE for multi-sample models

## Development

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Run tests:

```bash
pytest -q
```

Check selected device:

```bash
python -c "from src.utils.device import get_device; print(get_device())"
```

## Workflow Documents

- `GOAL_RUNBOOK.md`: goal-mode operating entry point
- `PROJECT_STATUS.md`: persistent project progress ledger
- `codex_vehicle_trajectory_project_plan.md`: phase-by-phase implementation plan
- `windows_gpu_training_only_workflow.md`: Windows GPU training-only workflow
- `WINDOWS_ENV_SETUP.md`: Windows CUDA/PyTorch setup checklist
- `github_portfolio_workflow.md`: commit/push and portfolio policy

## Data Policy

Raw AV2 data, processed `.npz` files, checkpoints, logs, and private credentials
are not committed to Git.
