# Codex Agent Instructions

This repository is a vehicle/pedestrian future trajectory prediction project.

## Required Read Order

At the start of every goal-mode turn, read:

```text
1. GOAL_RUNBOOK.md
2. PROJECT_STATUS.md
3. codex_vehicle_trajectory_project_plan.md
4. windows_gpu_training_only_workflow.md
5. WINDOWS_ENV_SETUP.md
6. github_portfolio_workflow.md
```

## Work Rules

```text
Implement one Phase/Task at a time.
Run required tests before reporting a Phase complete.
Use synthetic data when AV2 raw data is unavailable.
Do not fake metrics, figures, checkpoints, or report results.
Do not commit raw data, processed .npz files, checkpoints, logs, or secrets.
Update PROJECT_STATUS.md after each completed Phase.
Commit and push verified Phase work according to github_portfolio_workflow.md.
```

## Mac / Windows Split

```text
Mac: source of truth for code, tests, preprocessing, evaluation, visualization,
analysis, reporting, git commits, and pushes.

Windows: model training only.
```

Before using Windows, verify:

```bash
ssh song@100.87.219.58 'hostname && whoami'
```

Expected:

```text
Song
song\song
```

For GPU training, use the `vehicle_traj` conda environment from
`WINDOWS_ENV_SETUP.md`. Do not run long training with a CPU-only PyTorch build.

## Git Discipline

```text
Check git status before edits.
Stage only files related to the current Phase.
Do not revert unrelated user changes.
If GitHub remote is missing, create local commits and report push as blocked.
```

## Subagent Guidance

Use subagents for high-risk phases:

```text
Phase 2 geometry
Phase 4 metrics
Phase 6 training pipeline
Phase 7 LSTM
Phase 8 Transformer
Phase 9 diffusion
Phase 11 AV2 preprocessing
Phase 13 analysis
Phase 14 final experiments
Phase 15 final report
```

Subagents should focus on:

```text
shape contracts
data leakage
metric definitions
mask handling
checkpoint/evaluation consistency
result claims versus saved files
missing tests
```

Main Codex still owns implementation, validation, PROJECT_STATUS updates, and
Git commit/push.

## Reporting

Every completed Phase report must include:

```text
changed files
commands run
test results
Git branch/commit/push status
Windows usage, if any
subagent usage, if any
remaining risks
next recommended task
```
