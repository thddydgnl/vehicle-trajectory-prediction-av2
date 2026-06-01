# Vehicle Trajectory Prediction

Argoverse 2-style vehicle and pedestrian trajectory forecasting project. The
project predicts a future 3-second trajectory from 5 seconds of past motion and
will compare Linear Extrapolation, LSTM, Transformer, and Diffusion models under
the same data format and metrics.

## Current Status

Phase 0 repository setup is complete. Phase 1 is the next implementation step.
Model, dataset generation, evaluation, and visualization code are intentionally
not implemented yet.

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
- `docs/README.md`: document index
- `docs/codex_vehicle_trajectory_project_plan.md`: phase-by-phase implementation plan
- `docs/windows_gpu_training_only_workflow.md`: Windows AV2 data and GPU training workflow
- `docs/WINDOWS_ENV_SETUP.md`: Windows CUDA/PyTorch and AV2 download setup checklist
- `docs/github_portfolio_workflow.md`: commit/push and portfolio policy
- `docs/prompts/phase_1_to_6_goal_prompt.md`: goal prompt for Mac-only Phase 1-6 work

## Data Policy

Raw AV2 data, processed `.npz` files, checkpoints, logs, and private credentials
are not committed to Git.

The full AV2 Motion Forecasting dataset is stored on Windows by default:

```text
C:\Users\thddy\data\av2\motion-forecasting
```
