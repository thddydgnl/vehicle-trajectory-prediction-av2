# Goal Prompt: Phase 1-6 Mac-Only Pipeline

작성일: 2026-06-01

아래 프롬프트는 Windows에서 AV2 다운로드가 진행 중일 때, Mac에서 먼저
Phase 1부터 Phase 6까지 synthetic data 기반 파이프라인을 진행하기 위한
Codex goal 시작 프롬프트이다.

## Copy/Paste Prompt

```text
Use goal mode for the vehicle trajectory prediction project.

Objective:
Complete Phase 1 through Phase 6 of this repository, one phase at a time,
using synthetic data only. Do not use real AV2 data for these phases.

Important current context:
- Windows may be downloading the AV2 Motion Forecasting dataset.
- Do not interrupt, stop, restart, or modify the Windows AV2 download.
- Phase 1-6 are Mac-only phases.
- Do not use Windows unless the newest user message explicitly asks for it.
- Treat the current working directory from `pwd` as the project root.
- If documentation contains older absolute paths, prefer the current project root.

Required read order before starting:
1. GOAL_RUNBOOK.md
2. AGENTS.md
3. PROJECT_STATUS.md
4. docs/codex_vehicle_trajectory_project_plan.md
5. docs/github_portfolio_workflow.md
6. docs/windows_gpu_training_only_workflow.md only for role boundaries; do not run Windows work

Initial inspection:
Run:
- pwd
- git status --short
- rg --files

Execution rules:
- Proceed strictly one Phase at a time.
- Start with the next incomplete Phase from PROJECT_STATUS.md.
- Stop after Phase 6 is complete, verified, committed, and pushed.
- Do not start Phase 7.
- Do not implement LSTM, Transformer, Diffusion, AV2 preprocessing, visualization,
  PCA/K-means, or final report assets during this goal.
- Do not fake metrics, checkpoints, figures, or training results.
- Do not commit raw data, generated .npz files, checkpoints, logs, or secrets.
- Generated synthetic .npz files may be used for smoke tests but should remain ignored.
- Update PROJECT_STATUS.md after each completed Phase.
- After each completed Phase, commit and push according to docs/github_portfolio_workflow.md.

Phase scope:

Phase 1: Synthetic Smoke Dataset
- Implement src/datasets/synthetic.py.
- Add configs/preprocess_smoke.yaml.
- Add or update tests/test_synthetic_data.py.
- Generate train_smoke.npz, val_smoke.npz, and test_smoke.npz locally.
- Required validation:
  python -m src.datasets.synthetic --out_dir data/processed --num_samples 1000
  pytest tests/test_synthetic_data.py -q

Phase 2: Geometry and Coordinate Transform
- Implement src/utils/geometry.py.
- Add tests/test_geometry.py.
- Required validation:
  pytest tests/test_geometry.py -q

Phase 3: Dataset and DataLoader
- Implement src/datasets/av2_dataset.py.
- Add or update tests for TrajectoryDataset and create_dataloader.
- Use synthetic .npz files for validation.
- Required validation:
  python -m src.datasets.synthetic --out_dir data/processed --num_samples 1000
  pytest tests/test_synthetic_data.py -q

Phase 4: Metrics
- Implement src/evaluation/metrics.py.
- Add tests/test_metrics.py with exact tensor expectations.
- Required validation:
  pytest tests/test_metrics.py -q

Phase 5: Linear Baseline
- Implement src/models/linear.py.
- Implement src/evaluation/evaluate.py for the linear model.
- Add configs/linear.yaml.
- Use val_smoke.npz for evaluation.
- Required validation:
  python -m src.evaluation.evaluate --model linear --data data/processed/val_smoke.npz --config configs/linear.yaml --out_dir outputs
- The metrics output must include ADE, FDE, Miss Rate, Latency, and Parameters when applicable.

Phase 6: Common Training Pipeline
- Implement src/training/losses.py.
- Implement src/training/trainer.py.
- Implement src/training/train.py.
- Add a tiny placeholder trainable model or minimal model path only if needed to smoke-test the common training loop.
- Do not implement the real Phase 7 LSTM architecture yet.
- Add configs/lstm.yaml only if needed as a tiny smoke config for the training CLI.
- Required validation:
  python -m src.training.train --config configs/lstm.yaml --max_epochs 1 --data data/processed/train_smoke.npz --val_data data/processed/val_smoke.npz

Subagent guidance:
- Phase 1: subagent optional.
- Phase 2: use a subagent if available to review coordinate transform correctness.
- Phase 4: use a subagent if available to review metric formulas, masks, and shape contracts.
- Phase 6: use a subagent if available to review training-loop checkpointing, validation metrics, and leakage risks.
- Main Codex remains responsible for implementation, validation, PROJECT_STATUS.md updates, commits, and pushes.

After each Phase:
1. Run required validation for that Phase.
2. Run `pytest -q` when practical, especially after shared utilities are changed.
3. Inspect `git status --short`.
4. Ensure generated data/checkpoints/logs are ignored and not staged.
5. Update PROJECT_STATUS.md with:
   - completed Phase
   - validation commands and results
   - next recommended task
   - remaining external requirements
6. Commit with the recommended message:
   - Phase 1: feat: add synthetic trajectory smoke dataset
   - Phase 2: feat: implement relative coordinate transforms
   - Phase 3: feat: add processed trajectory dataset loader
   - Phase 4: feat: implement trajectory evaluation metrics
   - Phase 5: feat: add linear extrapolation baseline
   - Phase 6: feat: add common training pipeline
7. Push the current branch to GitHub.
8. Report changed files, commands run, validation results, commit hash,
   push status, Windows usage, subagent usage, remaining risks, and next phase.

Completion criteria for this goal:
- Phase 1, 2, 3, 4, 5, and 6 are marked complete in PROJECT_STATUS.md.
- Required validation for every phase has passed.
- The repository has commits for the completed phases.
- The current branch has been pushed to GitHub.
- Phase 7 has not been started.
```

## Short Version

```text
Use goal mode. Read GOAL_RUNBOOK.md, AGENTS.md, PROJECT_STATUS.md,
docs/codex_vehicle_trajectory_project_plan.md,
docs/github_portfolio_workflow.md, and
docs/windows_gpu_training_only_workflow.md.

Complete Phase 1 through Phase 6 only, one phase at a time, using synthetic
data only. Windows may be downloading AV2 data; do not touch Windows or the
download. After each phase, run required validation, update PROJECT_STATUS.md,
commit, push, and report results. Stop after Phase 6. Do not start Phase 7.
```
