# Vehicle Trajectory Project GitHub Portfolio Workflow

작성일: 2026-06-01

목표:

```text
Codex goal 진행 중 각 Phase 결과를 GitHub에 꾸준히 commit/push하여,
차량·보행자 미래 궤적 예측 프로젝트를 포트폴리오 저장소로 남긴다.
```

핵심 원칙:

```text
작동하는 단위마다 커밋한다.
실행하지 않은 결과를 커밋하지 않는다.
데이터와 체크포인트 같은 heavy artifact는 올리지 않는다.
README와 결과 문서는 포트폴리오 관점에서 계속 개선한다.
```

## 1. Goal-Mode Rule

Codex goal 진행 중 GitHub 작업 규칙:

```text
1. Phase 시작 전 git status를 확인한다.
2. Phase 구현 후 테스트/smoke run을 실행한다.
3. 결과가 검증되면 의미 있는 단위로 commit한다.
4. 사용자가 push를 원한다고 했으므로 Phase 완료 커밋은 GitHub에 push한다.
5. 실패한 실험 결과를 성공처럼 커밋하지 않는다.
6. raw data, processed .npz, checkpoints, logs는 Git에 올리지 않는다.
7. Windows AV2/학습 Phase는 Mac commit/push -> Windows data/preprocess/train -> Mac 평가 -> 결과 commit/push 순서로 진행한다.
```

Goal turn algorithm에 추가할 Git 단계:

```text
1. Read GOAL_RUNBOOK.md.
2. Read PROJECT_STATUS.md.
3. Read docs/codex_vehicle_trajectory_project_plan.md.
4. Read docs/windows_gpu_training_only_workflow.md if Windows AV2 data or training may be needed.
5. Read docs/full_av2_training_staged_workflow.md if full AV2 preprocessing/training may be needed.
6. Read this GitHub portfolio workflow.
7. Run git status --short.
8. Implement only the current Phase/Task.
9. Run required tests and smoke commands.
10. Review generated files and .gitignore coverage.
11. Commit with a clear message.
12. Push the current branch.
13. Report commit hash, pushed branch, test results, and next Phase.
```

If the working tree contains unrelated user changes:

```text
Do not revert them.
Do not include unrelated files in the commit.
Stage only files created or modified for the current Phase.
If unrelated changes affect the same file, inspect carefully and preserve them.
```

## 2. Repository Setup

Recommended GitHub repository names:

```text
vehicle-trajectory-prediction-av2
argoverse2-trajectory-forecasting
trajectory-prediction-lstm-transformer-diffusion
```

Recommended public description:

```text
Argoverse 2 trajectory forecasting project comparing Linear, LSTM,
Transformer, and Diffusion models with ADE/FDE and trajectory visualizations.
```

Recommended topics:

```text
trajectory-prediction
autonomous-driving
argoverse2
lstm
transformer
diffusion-model
pytorch
machine-learning
```

If the repository has not been initialized:

```bash
git init
git branch -M main
```

If GitHub remote has not been configured:

```bash
git remote add origin git@github.com:<USER>/<REPO>.git
```

Use the actual GitHub URL chosen by the user.

If no remote URL has been provided yet:

```text
Create local commits after validation.
Report push as blocked by missing GitHub remote URL.
Do not stop Mac-only implementation work just because the remote is missing.
```

## 3. Commit Granularity

Commit by completed Phase, not by every tiny edit.

Recommended Phase commit messages:

```text
chore: initialize project skeleton
feat: add synthetic trajectory smoke dataset
feat: implement relative coordinate transforms
feat: add processed trajectory dataset loader
feat: implement trajectory evaluation metrics
feat: add linear extrapolation baseline
feat: add common training pipeline
feat: implement LSTM trajectory forecaster
feat: implement Transformer trajectory forecaster
feat: implement direct diffusion trajectory model
feat: add PCA latent trajectory codec
feat: add AV2 preprocessing pipeline
feat: add trajectory visualization scripts
feat: add PCA and k-means error analysis
docs: add final experiment summary
```

Use `fix:` for bug fixes:

```text
fix: correct masked ADE denominator
fix: avoid scenario leakage in split generation
fix: load checkpoint config during evaluation
```

Use `docs:` for documentation-only changes:

```text
docs: add Windows GPU training workflow
docs: document GitHub portfolio workflow
docs: update README with experiment results
```

## 4. What To Commit

Commit:

```text
src/
configs/
tests/
scripts/
README.md
AGENTS.md
docs/codex_vehicle_trajectory_project_plan.md
CODEX_TASKS.md
requirements.txt
pyproject.toml
Makefile
docs/*.md
small example figures selected for README
outputs/tables/*.md
outputs/tables/*.csv, if small
outputs/report_summary.md
```

Usually do not commit:

```text
data/raw/
data/processed/*.npz
data/processed/*.pkl
outputs/checkpoints/
outputs/predictions/
outputs/logs/
outputs/remote_runs/
windows_training_results/
large figures
videos
notebook checkpoints
```

Optional to commit:

```text
small synthetic sample metadata
small generated tables
one or two compressed/readme-friendly PNG examples
```

Before committing any generated artifact, ask:

```text
Is it small?
Is it reproducible?
Does it improve portfolio readability?
Does it avoid dataset/license problems?
```

## 5. Required .gitignore

The repository must ignore at least:

```gitignore
# Python
__pycache__/
*.py[cod]
.pytest_cache/
.ruff_cache/
.mypy_cache/
.venv/
venv/

# OS/editor
.DS_Store
.idea/
.vscode/

# Data
data/raw/
data/processed/*.npz
data/processed/*.pkl
data/processed/*.joblib
data/processed/scaler.pkl

# Training outputs
outputs/checkpoints/
outputs/predictions/
outputs/logs/
outputs/remote_runs/
windows_training_results/

# Large media
*.mp4
*.mov
*.avi
```

If a file should be preserved in an ignored directory, use `.gitkeep` or an
explicit allowlist rule.

## 6. README Portfolio Standard

The README should gradually include:

```text
1. Project overview
2. Problem definition
3. Dataset and synthetic fallback
4. Input/output format
5. Model list
6. Evaluation metrics
7. How to run tests
8. How to run preprocessing
9. How to train/evaluate each model
10. Results table
11. Qualitative trajectory examples
12. Error analysis
13. Limitations
14. Future work
```

README top section should make the project obvious:

```text
This project predicts future vehicle/pedestrian trajectories from past motion
history using Argoverse 2-style data. It compares Linear Extrapolation, LSTM,
Transformer, and Diffusion models under the same input format and metrics.
```

Do not invent final numbers. Use placeholders until real metrics exist:

```text
Results will be filled after the corresponding experiment is run.
```

## 7. Windows Training Commit Order

For Windows-training phases:

```text
1. Mac: implement model/trainer/evaluator code.
2. Mac: run unit tests and synthetic smoke where possible.
3. Mac: commit and push the code that Windows will train.
4. Mac -> Windows: sync source/config; use Windows-local AV2 processed data when available.
5. Windows: run approved AV2 preprocessing and/or training commands only.
6. Windows -> Mac: pull checkpoints/logs/metrics.
7. Mac: run evaluation and visualization.
8. Mac: commit and push result tables/report updates.
```

Do not commit Windows-only scratch output unless it has been validated on Mac.

## 8. Commands

Status:

```bash
git status --short
git branch --show-current
git remote -v
```

Review changes:

```bash
git diff --stat
git diff
```

Stage current Phase files intentionally:

```bash
git add README.md requirements.txt pyproject.toml Makefile src tests configs docs
```

Commit:

```bash
git commit -m "feat: implement trajectory evaluation metrics"
```

Push:

```bash
git push -u origin main
```

For later pushes:

```bash
git push
```

## 9. Phase Result Additions

Every Phase Result should include Git information:

```text
### Git
- Branch:
- Commit:
- Pushed: yes/no
- Remote:
```

If not pushed:

```text
### Git
- Branch:
- Commit:
- Pushed: no
- Reason:
```

## 10. Safety Rules

Never commit:

```text
API keys
SSH keys
Tailscale auth keys
private credentials
raw AV2 data
large checkpoints
large processed .npz files
private machine logs with secrets
```

If a secret is accidentally staged:

```text
Stop.
Unstage the file.
Remove the secret from the working tree.
Rotate the secret if it was committed or pushed.
```

## 11. Final Portfolio Checklist

Before considering the repository portfolio-ready:

```text
[ ] README explains the project in the first screen.
[ ] README has setup and run commands.
[ ] Tests pass.
[ ] Linear baseline runs.
[ ] At least LSTM and Transformer have trained/evaluated results.
[ ] Metrics table exists.
[ ] At least one trajectory overlay figure exists.
[ ] Limitations are honestly documented.
[ ] Raw data and checkpoints are not in Git.
[ ] The commit history shows clear project progression.
```

## 12. Final Rule

```text
Commit code and verified lightweight results.
Do not commit private data, heavy artifacts, or unverified claims.
```
