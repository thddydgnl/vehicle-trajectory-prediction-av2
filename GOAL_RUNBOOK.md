# Vehicle Trajectory Project Goal Runbook

작성일: 2026-06-01

이 문서는 Codex goal 기능으로 프로젝트를 장기 자동 진행할 때 사용하는 단일 진입점이다.

## 1. Read Order

Every goal turn must start by reading:

```text
1. AGENTS.md
2. PROJECT_STATUS.md
3. docs/codex_vehicle_trajectory_project_plan.md
4. docs/windows_gpu_training_only_workflow.md
5. docs/WINDOWS_ENV_SETUP.md
6. docs/github_portfolio_workflow.md
```

Then inspect the actual filesystem:

```bash
pwd
find . -maxdepth 2 -type f | sort
git status --short
```

## 2. Goal Objective

Project objective:

```text
Build a portfolio-ready vehicle/pedestrian future trajectory prediction project
using Argoverse 2-style data, comparing Linear, LSTM, Transformer, and
Diffusion models with consistent preprocessing, metrics, visualizations, and
final report assets.
```

Minimum presentation-ready target:

```text
Linear + LSTM + Transformer + ADE/FDE + trajectory overlay
```

Good target:

```text
Linear + LSTM + Transformer + Direct Diffusion
+ ADE/FDE/minADE/minFDE
+ trajectory overlays
+ error/PCA/K-means analysis
```

## 3. Phase Policy

Codex must proceed one Phase/Task at a time.

```text
Do not implement the whole project at once.
Do not skip tests for the current Phase.
Do not move to the next Phase after a failing required validation.
Do not report success without generated evidence.
```

Current phase must be selected from `PROJECT_STATUS.md`. If that file conflicts
with the filesystem, inspect both and update `PROJECT_STATUS.md` with the
evidence before continuing.

## 4. Mac / Windows Policy

Mac is the source-of-truth environment:

```text
code edits
docs
tests
synthetic data generation
small sample-data validation
evaluation
visualization
analysis
report writing
git commit/push
```

Windows is used for AV2 data storage, data-local AV2 preprocessing when needed,
and GPU model training:

```text
AV2 raw data download/storage
AV2 raw -> processed conversion when the raw data stays on Windows
python -m src.training.train ...
```

Before using Windows, follow `docs/windows_gpu_training_only_workflow.md` and
`docs/WINDOWS_ENV_SETUP.md`.

Verified Windows SSH:

```bash
ssh song@100.87.219.58 'hostname && whoami'
```

Expected output:

```text
Song
song\song
```

## 5. GitHub Portfolio Policy

Follow `docs/github_portfolio_workflow.md`.

At the end of every completed Phase:

```text
1. run required validation
2. update PROJECT_STATUS.md
3. update README/results docs if relevant
4. commit verified code/docs/lightweight results
5. push to GitHub
6. report commit hash and push status
```

If no GitHub remote exists:

```text
Initialize local git if needed.
Create local commits.
Do not block implementation forever.
Report push as blocked and ask for the GitHub remote URL.
```

## 6. Data And Artifact Policy

Never commit:

```text
data/raw/
data/processed/*.npz
Windows AV2 raw/processed data directories
outputs/checkpoints/
outputs/predictions/
outputs/logs/
windows_training_results/
secrets
auth keys
private credentials
```

Synthetic data may be generated locally for tests, but generated `.npz` files
remain ignored unless the user explicitly asks for a tiny tracked fixture.

## 7. Blocking Policy

Do not mark the goal blocked just because:

```text
AV2 raw data is missing
Windows is temporarily unreachable
GitHub remote is not configured
Diffusion training is slow
```

Use fallback paths:

```text
AV2 missing before Phase 11 -> continue with synthetic data
AV2 missing at Phase 11+ -> download or verify Windows-local AV2 data first
Windows unreachable -> finish Mac-only phases and record training as pending
Windows PyTorch is CPU-only -> set up vehicle_traj CUDA environment before long training
GitHub remote missing -> local commit and request remote URL
Diffusion slow -> run smoke/tiny config first
```

Only stop for user input when a required external decision cannot be safely
inferred, such as the actual GitHub remote URL or credentials.

## 8. Subagent Policy

Use subagents selectively. They are most useful for high-risk phases where
correctness, leakage prevention, shape contracts, or result integrity matter
more than raw implementation speed.

Recommended subagent phases:

```text
Phase 2  Geometry and Coordinate Transform
Phase 4  Metrics
Phase 6  Common Training Pipeline
Phase 7  LSTM Encoder-Decoder
Phase 8  Transformer Encoder
Phase 9  Direct Diffusion Model
Phase 11 Argoverse 2 Preprocessing
Phase 13 PCA and K-means Analysis
Phase 14 Final Experiment Matrix
Phase 15 Final Report Assets
```

Subagents are optional for:

```text
Phase 1  Synthetic Smoke Dataset, unless tests fail or pattern quality is uncertain
Phase 5  Linear Baseline
simple README/docs updates
small config-only edits
```

Default pattern:

```text
Main Codex:
  implement code, run commands, integrate feedback, update docs/status, commit/push

Subagent A:
  review shape contracts, tests, metrics, leakage, edge cases

Subagent B, only when needed:
  review experiment/report consistency, README/portfolio clarity, result claims
```

Subagent rules:

```text
1. Subagents advise; the main agent owns final edits and validation.
2. Do not let subagent work replace required tests.
3. Do not delay small Mac-only phases just to spawn subagents.
4. Re-trigger subagent review after repeated failures, AV2 schema uncertainty,
   metric ambiguity, diffusion sampling issues, or final report/result claims.
5. If subagent capacity is unavailable, continue with local tests and document
   the missing review as a residual risk.
```

For Phase Result, mention subagent usage:

```text
### Subagents
- Used: yes/no
- Focus:
- Findings:
```

## 9. Required Phase Result Format

After each Phase:

```text
## Phase Result

### Phase
- ...

### Changed Files
- ...

### Commands Run
- ...

### Results
- ...

### Validation
- ...

### Git
- Branch:
- Commit:
- Pushed:
- Remote:

### Windows
- Used: yes/no
- If yes, command:
- Pulled back:

### Subagents
- Used:
- Focus:
- Findings:

### Remaining Risks
- ...

### Next Recommended Task
- ...
```

## 10. Resume Rule

When resuming after context compaction or a long-running turn:

```text
1. Re-read this file.
2. Re-read PROJECT_STATUS.md.
3. Check git status.
4. Check whether any long-running command is still active.
5. Continue from the newest user request and the latest verified filesystem state.
```

Do not rely on memory alone.

## 11. Final Done Criteria

The goal is complete only when:

```text
[ ] pytest -q passes
[ ] Linear baseline runs
[ ] LSTM trained/evaluated
[ ] Transformer trained/evaluated
[ ] Diffusion smoke or full result exists, if time allows
[ ] model comparison table exists
[ ] trajectory overlay figure exists
[ ] report_summary.md exists
[ ] README is portfolio-ready
[ ] GitHub has the final pushed state
```
