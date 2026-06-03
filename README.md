# Vehicle Trajectory Prediction

Argoverse 2-style vehicle and pedestrian trajectory forecasting project. The
project predicts a future 3-second trajectory from 5 seconds of past motion and
will compare Linear Extrapolation, LSTM, Transformer, and Diffusion models under
the same data format and metrics.

## Current Status

Phases 0 through 10 are complete on synthetic smoke data. The next major step
is Phase 11 Argoverse 2 preprocessing with real AV2 data on the Windows host.

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

## Data Policy

Raw AV2 data, processed `.npz` files, checkpoints, logs, and private credentials
are not committed to Git.

The full AV2 Motion Forecasting dataset is stored on Windows by default:

```text
D:\data\av2\motion-forecasting
```

For real AV2 preprocessing, verify this marker first:

```text
D:\data\av2\motion-forecasting\DATA_READY_FOR_PHASE11.txt
```

Default Windows training host:

```text
Primary: HOME over LAN, ssh thddy@192.168.35.17
Fallback: HOME over Tailscale, ssh thddy@100.99.63.23, only after verifying SSH works
Do not use legacy song@100.87.219.58 for AV2 data; it does not expose D:\data.
```
