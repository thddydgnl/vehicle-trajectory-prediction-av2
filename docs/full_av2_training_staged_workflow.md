# Full AV2 Staged Training Workflow

작성일: 2026-06-03

이 문서는 Phase 14 small AV2 smoke 이후, Codex goal 기능으로 full AV2
preprocessing/training을 진행할 때 따라야 하는 단계별 실행 전략이다.

## 1. Purpose

Current status as of 2026-06-04:

```text
Phase 14 small AV2 smoke is complete.
Full AV2 Stage F1 preprocessing is complete.
Full AV2 Stage F2 schema validation is complete.
Full AV2 Stage F3 1-epoch pilot is complete.
Full AV2 Stage F4 5-epoch pilot is complete.
The small AV2 results prove the pipeline works, and the full 1/5-epoch pilots
prove the full processed dataset can train/evaluate on Windows CUDA.
These are useful school-project pilot results, but still not report-ready
long-run final performance.
```

Goal-mode rule:

```text
Do not jump directly from small smoke to a 30-50 epoch long run.
Proceed through full preprocessing, 1-epoch pilot, 5-epoch pilot, then long run.
Stop at each gate if validation, CUDA, checkpoint, metric, or provenance checks fail.
```

## 2. Estimated Runtime

Use these as planning estimates, not guarantees:

```text
Full preprocessing: 2-6 hours
Full schema validation and PCA fitting: 10-40 minutes
Full 1-epoch pilot across core models: 1-3 hours
Full 5-epoch pilot across core models: 3-8 hours
Report-ready long run: 8-20 hours
```

Observed on HOME for the current compact pilot configs:

```text
Full preprocessing completed in roughly 6 hours, after retrying with parquet
read-error skipping.
Full 5-epoch pilot completed in about 13 minutes from scheduled-task start to
final comparison table generation.
Future long-run timing can still be much longer if epochs, model sizes,
sampling steps, or batch sizes are increased.
```

Main bottleneck:

```text
Full preprocessing is likely disk I/O bound because it scans roughly:
train: 199,892 scenario parquet files
val:    24,988 scenario parquet files
```

## 3. Required Preconditions

Before starting any full run:

```text
1. Read AGENTS.md, GOAL_RUNBOOK.md, PROJECT_STATUS.md, and this document.
2. Confirm git status is clean on Mac.
3. Confirm latest Mac code is committed and pushed.
4. Confirm HOME Windows repo is on latest origin/main.
5. Confirm HOME LAN SSH works.
6. Confirm vehicle_traj CUDA environment works.
7. Confirm D:\data\av2\motion-forecasting\DATA_READY_FOR_PHASE11.txt exists.
8. Confirm D:\data has enough free space.
```

Required Windows checks:

```bash
ssh thddy@192.168.35.17 'hostname && whoami'
ssh thddy@192.168.35.17 'cmd /c "cd /d C:\Users\thddy\Documents\code\vehicle_trajectory_project && git pull --ff-only origin main && git rev-parse --short HEAD"'
ssh thddy@192.168.35.17 'powershell -NoProfile -ExecutionPolicy Bypass -Command "& C:\Users\thddy\Miniconda3\Scripts\conda.exe run -n vehicle_traj python -c \"import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 0)\""'
ssh thddy@192.168.35.17 'cmd /c "if exist D:\data\av2\motion-forecasting\DATA_READY_FOR_PHASE11.txt type D:\data\av2\motion-forecasting\DATA_READY_FOR_PHASE11.txt"'
```

Expected:

```text
HOME / home\thddy
torch without +cpu
torch.cuda.is_available() == True
NVIDIA GeForce RTX 2070 SUPER
```

## 4. Stage F1 - Full Preprocessing

Purpose:

```text
Create train_full.npz and val_full.npz from Windows-local raw AV2 data.
```

Run location:

```text
Windows HOME, because raw AV2 data is stored on D:.
```

Execution rule:

```text
Do not run full preprocessing as a long foreground SSH command.
Create a Windows background PowerShell job or scheduled task that writes logs.
Record the exact command and log path in PROJECT_STATUS.md.
```

Target outputs:

```text
D:\data\vehicle_trajectory_project\processed\full\train_full.npz
D:\data\vehicle_trajectory_project\processed\full\val_full.npz
D:\data\vehicle_trajectory_project\processed\full\metadata\train_full_metadata.csv
D:\data\vehicle_trajectory_project\processed\full\metadata\val_full_metadata.csv
D:\data\vehicle_trajectory_project\processed\full\scaler.pkl
```

Preprocessing command inside the background script:

```powershell
Set-Location C:\Users\thddy\Documents\code\vehicle_trajectory_project
& C:\Users\thddy\Miniconda3\Scripts\conda.exe run -n vehicle_traj python -m src.datasets.preprocess_av2 `
  --raw_dir D:\data\av2\motion-forecasting `
  --out_dir D:\data\vehicle_trajectory_project\processed\full `
  --full `
  --splits train val `
  --target_types VEHICLE PEDESTRIAN `
  --target_mode focal
```

Pass criteria:

```text
train_full.npz exists
val_full.npz exists
metadata CSV files exist
scaler.pkl exists
preprocessing log has no traceback
```

Stop criteria:

```text
No valid samples produced
schema validation fails
disk space falls too low
SSH instability prevents log/status checks
```

## 5. Stage F2 - Full Schema Validation

Run on Windows first:

```powershell
Set-Location C:\Users\thddy\Documents\code\vehicle_trajectory_project
& C:\Users\thddy\Miniconda3\Scripts\conda.exe run -n vehicle_traj python -m src.datasets.validate_processed `
  --npz D:\data\vehicle_trajectory_project\processed\full\train_full.npz
& C:\Users\thddy\Miniconda3\Scripts\conda.exe run -n vehicle_traj python -m src.datasets.validate_processed `
  --npz D:\data\vehicle_trajectory_project\processed\full\val_full.npz
```

Then record:

```text
sample counts
scenario counts
file sizes
validation command output
```

Mac copy policy:

```text
Do not copy train_full.npz to Mac by default.
If val_full.npz is small enough for the current Mac disk budget, copying only
val_full.npz for Mac-side final evaluation is preferred.
If val_full.npz is too large, run data-local evaluation on Windows as an
explicit exception using committed Mac-authored code, then pull back only
metrics, tables, predictions if lightweight, and figures/tables needed for
reporting.
```

## 6. Stage F3 - Full 1-Epoch Pilot

Purpose:

```text
Prove the full dataset works with CUDA training and evaluation before spending
many hours on long runs.
```

Recommended model order:

```text
1. Linear evaluation
2. LSTM 1 epoch
3. Transformer 1 epoch
4. PCA codec fit on train_full
5. PCA Diffusion 1 epoch
6. Direct Diffusion 1 epoch, optional and last
```

Config rule:

```text
Create dedicated full-pilot configs on Mac, commit/push them, and pull on
Windows before running. Do not edit configs directly on Windows.
Use model names with an explicit suffix, such as:
lstm_full_pilot_1epoch
transformer_full_pilot_1epoch
diffusion_pca_full_pilot_1epoch
diffusion_direct_full_pilot_1epoch
```

Recommended training settings:

```text
device: cuda
out_dir: D:/runs/vehicle_trajectory_project/full_pilot_1epoch
epochs: 1
num_workers: 0 initially
batch_size:
  LSTM: 64
  Transformer: 32
  Diffusion PCA: 64
  Diffusion Direct: 64
```

Pass criteria:

```text
CUDA is recorded in checkpoint metadata
train log exists
best checkpoint exists
last checkpoint exists
validation metrics JSON exists
ADE/FDE are finite
no NaN or traceback in logs
```

Stop criteria:

```text
CUDA is False
PyTorch is CPU-only
GPU OOM repeats after reducing batch size
validation metrics are NaN/inf
checkpoint/evaluation shape mismatch occurs
```

## 7. Stage F4 - Full 5-Epoch Pilot

Purpose:

```text
Check whether learned models move in a useful direction before a long run.
```

Recommended model order:

```text
1. LSTM
2. Transformer
3. PCA Diffusion
4. Direct Diffusion
```

Recommended training settings:

```text
out_dir: D:/runs/vehicle_trajectory_project/full_pilot_5epoch
epochs: 5
device: cuda
same batch sizes as Stage F3 unless OOM requires reduction
```

Decision rule:

```text
Continue to long run if train/val losses are finite and generally improving.
Do not require learned models to beat Linear after only 5 epochs.
If LSTM and Transformer are still dramatically worse than Linear and not
improving, pause for config tuning before running Diffusion long runs.
```

Observed F4 result on 2026-06-04:

```text
Stage F4 passed on HOME using commit 8014ed3.
All trainable models ran exactly 5 epochs with CUDA checkpoint metadata.
Linear val_full:       ADE 1.53242 / FDE 3.79734
LSTM val_full:         ADE 1.04012 / FDE 2.60335
Transformer val_full:  ADE 0.94158 / FDE 2.41605
PCA Diffusion val_full:    ADE 6.27722 / FDE 12.32233 / minADE 6.09676 / minFDE 11.60643
Direct Diffusion val_full: ADE 10.05855 / FDE 19.44815 / minADE 9.88166 / minFDE 18.50691

LSTM and Transformer improved from Stage F3 and beat Linear after 5 epochs.
Transformer is currently the strongest pilot model.
Diffusion variants are still much worse than Linear and should be tuned before
any expensive long Diffusion run.
Lightweight F4 artifacts are stored under outputs/full_av2_5epoch_pilot.
```

Useful tuning checks before long run:

```text
learning rate
batch size
hidden size / d_model
loss function
object_type target mix
normalization/scaler handling
relative coordinate assumptions
```

## 8. Stage F5A - Diffusion Tuning Gate

Only start after Stage F4 passes and the user explicitly asks for stronger
final numbers beyond the pilot results.

Purpose:

```text
Tune PCA Diffusion and Direct Diffusion enough to produce analysis-ready final
results. They do not need to beat LSTM/Transformer, but they must be evaluated
honestly with finite metrics, sample diversity, and clear failure/success notes.
```

Candidate policy:

```text
Run a small bounded matrix before any long Diffusion run:
PCA Diffusion: 3 candidates, 10 epochs each
Direct Diffusion: 3 candidates, 10 epochs each

Vary one small set of risk-controlled parameters:
learning rate
hidden_dim / cond_dim
diffusion_steps
sampling_steps
num_samples
batch_size for OOM safety
```

Hard gate:

```text
checkpoint exists
train metrics exist
evaluation metrics exist
epochs_ran >= 5
ADE/FDE/minADE/minFDE are finite
Sample_Diversity is finite and >= 0.001
no NaN/OOM/checkpoint mismatch
```

Preferred gate:

```text
At least one of minADE or minFDE improves by 10% or more versus the Stage F4
same-model diffusion baseline.
```

Selection rule:

```text
Prefer candidates that pass the preferred gate.
If none pass the preferred gate but at least one passes the hard gate, select
the best hard-gate candidate for analysis-ready final training and clearly
label that it did not meet the improvement target.
If no candidate passes the hard gate, stop long Diffusion training and inspect
the failure before spending more GPU time.
```

Current committed tuning matrix:

```text
configs/full_diffusion_tuning_matrix.yaml
configs/full_tune_diffusion_pca_a.yaml
configs/full_tune_diffusion_pca_b.yaml
configs/full_tune_diffusion_pca_c.yaml
configs/full_tune_diffusion_direct_a.yaml
configs/full_tune_diffusion_direct_b.yaml
configs/full_tune_diffusion_direct_c.yaml
```

Automation:

```text
scripts/select_diffusion_tuning.py writes:
D:\runs\vehicle_trajectory_project\full_long_tuning\tables\diffusion_tuning_summary.csv
D:\runs\vehicle_trajectory_project\full_long_tuning\tables\diffusion_tuning_summary.md
D:\runs\vehicle_trajectory_project\full_long_tuning\tables\selected_diffusion_configs.json
D:\runs\vehicle_trajectory_project\full_long_final\generated_configs\full_long_diffusion_pca.yaml
D:\runs\vehicle_trajectory_project\full_long_final\generated_configs\full_long_diffusion_direct.yaml
```

## 9. Stage F5B - Report-Ready Long Run

Only start after Stage F5A selects analysis-ready Diffusion configs.

Recommended priority:

```text
1. LSTM: 20-30 epochs
2. Transformer: 20-50 epochs
3. PCA Diffusion: 20-50 epochs
4. Direct Diffusion: overnight run, last priority
```

Why this order:

```text
LSTM and Transformer are the clearest portfolio comparison against Linear.
PCA Diffusion is cheaper and often more stable than direct 60D diffusion.
Direct Diffusion is useful, but it is the highest-risk long run.
```

Run rule:

```text
Use Windows background-safe execution with logs.
Do not launch all long runs blindly if the earlier model fails.
After each model, validate checkpoint, metrics, and logs before starting the
next expensive model.
```

Current committed long-run configs/tooling:

```text
configs/full_long_lstm.yaml
configs/full_long_transformer.yaml
scripts/windows_full_long_experiments.ps1
scripts/windows_full_long_status.ps1
```

Windows background execution:

```text
Run scripts/windows_full_long_experiments.ps1 only from a scheduled task or
other background-safe launcher. Do not run it as a multi-hour foreground SSH
command.
If the tuning candidates have already completed and only selection/final long
run needs to resume, use scripts/windows_full_long_experiments.ps1
-ResumeAfterTuning from a background-safe launcher.
```

## 10. Stage F6 - Result Integration

Preferred integration path:

```text
1. Pull only lightweight metrics/tables/log summaries to Mac.
2. Pull checkpoints to ignored outputs/checkpoints/<run_tag>/ only if Mac-side
   evaluation is practical.
3. Pull val_full.npz only if disk budget allows.
4. Generate final comparison tables under a separate output directory, such as:
   outputs/full_av2/tables/model_comparison.csv
   outputs/full_av2/tables/model_comparison.md
5. Generate analysis outputs under:
   outputs/full_av2_analysis/
6. Keep raw data, full processed .npz, checkpoints, logs, and prediction .pkl
   files out of Git unless explicitly allowed and lightweight.
```

Evaluation command pattern with explicit checkpoints:

```bash
python scripts/run_all_evaluations.py \
  --data data/processed/val_full.npz \
  --out_dir outputs/full_av2 \
  --models linear lstm transformer diffusion_direct diffusion_pca \
  --checkpoint_dir outputs/checkpoints/full_av2_long \
  --checkpoint_tag full_av2_long \
  --batch_size 64 \
  --data_split val_full \
  --target_type av2_focal_mixed \
  --prediction_tag full_av2
```

If evaluation must run on Windows because val_full.npz is not copied to Mac,
record that exception clearly:

```text
Evaluation was run data-local on Windows using committed source at commit <hash>.
Mac performed reporting and committed only lightweight tables/metrics.
```

## 11. Reporting Requirements

After each full stage, update `PROJECT_STATUS.md` with:

```text
stage name
changed files
commands run
log paths
test/validation results
Windows usage
subagent usage if any
Git commit/push status
remaining risks
next recommended stage
```

Do not claim:

```text
full-data result before train_full.npz and val_full.npz are validated
model performance before metrics are generated from real outputs
long-run success before checkpoints and logs are verified
```

## 11. Recommended Next Goal

Use this as the next goal objective if the user wants to start FULL from small
AV2 smoke only:

```text
Proceed with Full AV2 Stage F1-F3 only:
1. Verify Mac/Windows/git/CUDA preconditions.
2. Start full AV2 preprocessing as a Windows background job.
3. Validate train_full.npz and val_full.npz when complete.
4. Create Mac-authored 1-epoch full pilot configs.
5. Commit/push configs before Windows execution.
6. Run full 1-epoch pilot for Linear, LSTM, Transformer, PCA codec, PCA Diffusion, and optionally Direct Diffusion.
7. Pull back lightweight metrics/tables.
8. Update PROJECT_STATUS.md.
9. Commit and push verified docs/configs/lightweight results.
Do not run 5-epoch or long-run training in this goal unless Stage F3 passes
and the user explicitly asks to continue.
```

As of 2026-06-04, Stage F1-F3 have passed. The next recommended goal is Stage
F4 only, or Phase 15 if the user wants to stop at the current pilot results.
