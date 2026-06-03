# Vehicle Trajectory Project Windows AV2 Data and GPU Training Workflow

작성일: 2026-06-01

목표:

```text
차량·보행자 미래 궤적 예측 프로젝트에서 Mac을 주 작업 환경으로 유지하고,
Windows 컴퓨터는 AV2 원본/학습 데이터 저장소와 GPU 실행 노드로 사용한다.
```

핵심 원칙:

```text
Mac: 코드 작성, 문서, 테스트, synthetic/sample 검증, 평가, 시각화, 리포트, Git
Windows: AV2 raw/processed 데이터 보관, 데이터가 Windows에 있을 때 AV2 전처리, GPU 학습
```

현재 확인된 접속 방식:

```text
확인일: 2026-06-02
Mac Tailscale client: /Applications/Tailscale.app
Mac Tailscale node: macbookair
Mac Tailscale IP: 100.75.150.25
Windows training host: HOME
Windows LAN IP: 192.168.35.17
Windows Tailscale node: home
Windows Tailscale IP: 100.99.63.23
SSH user: thddy
Verified command: ssh thddy@192.168.35.17 'hostname && whoami'
Verified output: HOME / home\thddy
Connection priority: LAN first, Tailscale fallback
Fallback command: ssh thddy@100.99.63.23 'hostname && whoami'
Fallback status on 2026-06-03: SSH port timed out; verify before use
Legacy secondary host: song, song@100.87.219.58, not the AV2 data host
```

Goal 진행 시 Codex가 지켜야 하는 첫 규칙:

```text
1. GOAL_RUNBOOK.md를 먼저 읽는다.
2. PROJECT_STATUS.md를 읽고 현재 Phase/Task를 확인한다.
3. docs/codex_vehicle_trajectory_project_plan.md를 읽고 Phase 요구사항을 확인한다.
4. Windows는 AV2 데이터 보관, AV2 data-local preprocessing, GPU 학습에만 사용한다.
5. Windows 접속 전 Mac 테스트와 가능한 data validation을 먼저 끝낸다.
6. Windows 작업 후 checkpoint/log/metric과 필요한 lightweight 결과만 Mac으로 회수한다.
7. 최종 평가, 그림, 표, 보고서는 Mac에서 만든다.
```

이 문서는 기존 RCS-AC/CARLA 원격 실행 문서를 이번 Argoverse 2 trajectory
prediction 프로젝트에 맞게 재구성한 것이다. CARLA 실행, sensor IO, closed-loop
eval, stage smoke gate는 이 프로젝트 범위가 아니므로 제거한다.

## 0. Goal-Mode Entry Protocol

Codex goal 기능으로 이 프로젝트를 진행할 때, 새 goal 또는 자동 재개 턴의 시작
순서는 다음과 같다.

Read order:

```text
1. /Users/song-yonghwi/Documents/vehicle_trajectory_project/GOAL_RUNBOOK.md
2. /Users/song-yonghwi/Documents/vehicle_trajectory_project/PROJECT_STATUS.md
3. /Users/song-yonghwi/Documents/vehicle_trajectory_project/docs/codex_vehicle_trajectory_project_plan.md
4. /Users/song-yonghwi/Documents/vehicle_trajectory_project/docs/windows_gpu_training_only_workflow.md
5. /Users/song-yonghwi/Documents/vehicle_trajectory_project/docs/WINDOWS_ENV_SETUP.md
6. /Users/song-yonghwi/Documents/vehicle_trajectory_project/docs/full_av2_training_staged_workflow.md
7. /Users/song-yonghwi/Documents/vehicle_trajectory_project/docs/github_portfolio_workflow.md
8. 현재 구현된 README.md, AGENTS.md, CODEX_TASKS.md가 있으면 함께 확인
9. 마지막 Phase Result 또는 outputs/report_summary.md가 있으면 진행 상태 확인
```

Goal start prompt 예시:

```text
Use goal mode for the vehicle trajectory prediction project.
Read GOAL_RUNBOOK.md first.
Read PROJECT_STATUS.md before selecting the next Phase.
Read docs/codex_vehicle_trajectory_project_plan.md for the phase specification.
Use docs/windows_gpu_training_only_workflow.md for Windows AV2 data and GPU training.
Use docs/WINDOWS_ENV_SETUP.md before any AV2 download or long Windows training run.
Use docs/full_av2_training_staged_workflow.md before full AV2 preprocessing/training.
Use docs/github_portfolio_workflow.md for commit/push and portfolio rules.
Keep Mac as the source-of-truth environment.
Use Windows only for AV2 data storage, data-local preprocessing, and GPU training.
Start with the next incomplete Phase only.
After each phase, report changed files, commands, validation results,
Git commit/push status, remaining risks, and the next recommended phase.
```

Goal turn algorithm:

```text
1. Identify the current requested Phase/Task.
2. If the task is Mac-only, do all work on Mac and do not contact Windows.
3. If the task needs real AV2 data, verify the Windows data paths first.
4. If the task includes AV2 preprocessing, finish Mac implementation/tests first,
   then run the committed preprocessing code on Windows against Windows-local data.
5. If the task includes model training, finish Mac implementation/tests first.
6. Validate train/val processed data on the machine that stores it; for Windows
   AV2 data, run validation over SSH or copy a tiny sample back to Mac.
7. Sync source/config only; avoid moving raw AV2 or large processed files to Mac.
8. Run Windows environment/CUDA preflight.
9. Run exactly the intended preprocessing or training command on Windows.
   If this is full AV2 work, follow the staged gates in
   docs/full_av2_training_staged_workflow.md.
10. Pull checkpoint/log/metric/lightweight result files back to Mac.
11. Run evaluate/visualization/analysis on Mac when the required inputs are available.
12. Commit and push verified code/results according to docs/github_portfolio_workflow.md.
13. Write Phase Result.
```

Resume rule:

```text
If Codex resumes after context compaction or a long-running goal turn, it must
re-read this section, check the filesystem state, and continue from the newest
user request and the latest completed Phase Result. It must not assume a
Windows training run succeeded unless the checkpoint/log/metric are visible on
Mac or were explicitly verified on Windows.
```

## 1. Scope

Windows에서 실행해도 되는 작업:

```text
AV2 raw data download with s5cmd
AV2 raw data inventory and integrity checks
python -m src.datasets.preprocess_av2 ... when raw AV2 stays on Windows
python -m src.datasets.validate_processed ... for Windows-local processed data
python -m src.training.train ...
```

Windows에서 실행하지 않는 작업:

```text
1. Codex 코드 수정
2. 프로젝트 문서 수정
3. synthetic data 생성, unless explicitly needed for a tiny Windows GPU smoke
4. metric evaluation for final reported numbers
5. trajectory visualization
6. PCA/K-means/error analysis
7. final report summary 생성
8. raw AV2 data 수정
```

Windows는 data/training worker이며, 프로젝트 코드와 판단의 source of truth는 Mac 폴더다.

## 2. Machine Role Split

Mac 역할:

```text
1. Codex 작업의 기준 환경
2. 코드, 설정, 테스트, 문서 수정
3. requirements/pyproject/Makefile 관리
4. synthetic smoke data 생성
5. AV2 전처리 코드 작성 및 작은 샘플 검증
6. processed .npz schema/validation logic 관리
7. unit test와 smoke test 실행
8. Linear baseline 평가
9. Windows 실행용 source/config bundle 준비
10. Windows 원격 preprocessing/training 실행
11. checkpoint, train log, validation metric, lightweight 결과 회수
12. Mac에서 최종 evaluate/visualization/analysis/report 실행
```

Windows 역할:

```text
1. Python/PyTorch GPU 환경 제공
2. AV2 raw data를 Windows local disk에 저장
3. AV2 processed train/val/test data를 Windows local disk에 저장
4. Mac에서 보낸 source/config를 받아 동일 코드로 preprocessing/training 실행
5. outputs/checkpoints/*.pt 저장
6. outputs/logs/*.csv 저장
7. outputs/metrics/*_val_metrics.json 저장
8. 학습 stdout/stderr 로그 저장
```

금지:

```text
Windows에서 코드를 직접 고치지 않는다.
Windows에서 raw AV2 data를 수정하지 않는다.
Windows에서 Mac에 커밋되지 않은 코드로 전처리 결과를 만들지 않는다.
Windows 학습 결과를 실제 회수/검증하지 않고 성공했다고 보고하지 않는다.
Mac과 Windows의 코드 버전이 다른 상태에서 학습 결과를 최종 결과로 쓰지 않는다.
```

## 3. Paths

Mac project root:

```text
/Users/song-yonghwi/Documents/vehicle_trajectory_project
```

Mac Windows-result cache:

```text
/Users/song-yonghwi/Documents/vehicle_trajectory_project/windows_training_results
```

Windows project root 권장값:

```text
C:\Users\thddy\Documents\code\vehicle_trajectory_project
```

Windows AV2 raw data:

```text
D:\data\av2\motion-forecasting
```

Windows AV2 Phase 11 ready marker:

```text
D:\data\av2\motion-forecasting\DATA_READY_FOR_PHASE11.txt
```

Windows processed training data:

```text
D:\data\vehicle_trajectory_project\processed
```

Windows training output:

```text
D:\runs\vehicle_trajectory_project
```

Remote identity:

```text
Primary SSH login: thddy@192.168.35.17
Fallback SSH login: thddy@100.99.63.23, verify before use
Mac Tailscale address: 100.75.150.25
Windows LAN address: 192.168.35.17
Windows Tailscale address: 100.99.63.23
Legacy secondary host: song@100.87.219.58, not for AV2 data
Latest check on 2026-06-03: HOME LAN sees D:\data; HOME Tailscale SSH timed out;
legacy song sees only C: and does not expose D:\data.
```

Windows 사용자명이나 경로가 달라지면 이 섹션을 먼저 수정한다.

## 4. Phase Mapping

Mac-only phases:

```text
Phase 0  Repository setup
Phase 1  Synthetic smoke dataset
Phase 2  Geometry and coordinate transform
Phase 3  Dataset/DataLoader
Phase 4  Metrics
Phase 5  Linear baseline evaluation
Phase 10 PCA codec fitting and PCA analysis
Phase 12 Visualization
Phase 13 PCA/K-means/error analysis
Phase 14 Final evaluation matrix
Phase 15 Final report assets
```

Windows-training phases:

```text
Phase 7  LSTM training
Phase 8  Transformer training
Phase 9  Direct diffusion training
Phase 10 PCA latent diffusion training, if implemented
```

Hybrid phases:

```text
Phase 6  Common training pipeline
Phase 11 AV2 preprocessing
```

Phase 6 code is written and tested on Mac. Windows only runs a tiny training
command after Mac tests pass, to verify the training loop works on the GPU
machine.

Phase 11 preprocessing code is written and tested on Mac. Because raw AV2 is
stored on Windows to protect Mac disk space, the full raw-to-processed conversion
may run on Windows with committed Mac-authored code.

## 5. Mac Preflight

Before any Windows training run:

```bash
cd /Users/song-yonghwi/Documents/vehicle_trajectory_project
pytest -q
python -m src.datasets.validate_processed --npz data/processed/train_smoke.npz
python -m src.datasets.validate_processed --npz data/processed/val_smoke.npz
```

If AV2 processed data is used:

```bash
trajwinssh thddy@192.168.35.17 'powershell -NoProfile -ExecutionPolicy Bypass -Command "Set-Location C:\Users\thddy\Documents\code\vehicle_trajectory_project; conda run -n vehicle_traj python -m src.datasets.validate_processed --npz D:\data\vehicle_trajectory_project\processed\train_small.npz"'
trajwinssh thddy@192.168.35.17 'powershell -NoProfile -ExecutionPolicy Bypass -Command "Set-Location C:\Users\thddy\Documents\code\vehicle_trajectory_project; conda run -n vehicle_traj python -m src.datasets.validate_processed --npz D:\data\vehicle_trajectory_project\processed\val_small.npz"'
```

Required Mac-side checks:

```text
1. tests pass
2. training config exists
3. train/val .npz exists
4. npz validation passes on the machine holding the data
5. no future-coordinate leakage in input feature code
6. scaler/PCA, if any, was fit on train split only
7. training command is written down before launching Windows
```

## 5A. Windows AV2 Data Download

Primary AV2 Motion Forecasting storage path:

```text
D:\data\av2\motion-forecasting
```

Recommended disk budget:

```text
Minimum: 100 GB free
Comfortable: 150 GB+ free
Current verified HOME C: free space on 2026-06-02: about 192 GB
Current verified HOME D: free space before AV2 extraction on 2026-06-02:
about 839.6 GB
```

Install `s5cmd` on Windows if missing:

```powershell
conda install s5cmd -c conda-forge -y
```

If conda metadata solving is slow, use the official GitHub release binary:

```powershell
$bin = "C:\Users\thddy\bin\s5cmd"
New-Item -ItemType Directory -Force $bin | Out-Null
$zip = "$env:TEMP\s5cmd_2.3.0_Windows-64bit.zip"
Invoke-WebRequest `
  -Uri "https://github.com/peak/s5cmd/releases/download/v2.3.0/s5cmd_2.3.0_Windows-64bit.zip" `
  -OutFile $zip
Expand-Archive -Force $zip -DestinationPath $bin
& "$bin\s5cmd.exe" version
```

Create directories:

```powershell
New-Item -ItemType Directory -Force D:\data\av2\motion-forecasting | Out-Null
New-Item -ItemType Directory -Force D:\data\vehicle_trajectory_project\processed | Out-Null
New-Item -ItemType Directory -Force D:\runs\vehicle_trajectory_project | Out-Null
```

List official public S3 objects before copying:

```powershell
s5cmd --no-sign-request ls "s3://argoverse/datasets/av2/motion-forecasting/"
```

Download AV2 Motion Forecasting directly to Windows:

```powershell
s5cmd --no-sign-request cp `
  "s3://argoverse/datasets/av2/motion-forecasting/*" `
  "D:\data\av2\motion-forecasting\"
```

If the files were downloaded as archives under `D:\data`, normalize them into
the project raw directory before Phase 11:

```powershell
$raw = "D:\data\av2\motion-forecasting"
New-Item -ItemType Directory -Force $raw, "$raw\archives", "D:\data\av2\logs" | Out-Null
Copy-Item -Force D:\data\av2_mf_*_test_annotations.parquet $raw
tar -xf D:\data\train.tar -C $raw
tar -xf D:\data\val.tar -C $raw
tar -xf D:\data\test.tar -C $raw
Move-Item -Force D:\data\train.tar "$raw\archives\train.tar"
Move-Item -Force D:\data\val.tar "$raw\archives\val.tar"
Move-Item -Force D:\data\test.tar "$raw\archives\test.tar"
Set-Content "$raw\DATA_READY_FOR_PHASE11.txt" "AV2 Motion Forecasting data organized for Phase 11"
```

For long extraction, use a detached PowerShell script and logs instead of a
foreground SSH session. The current HOME organizer convention is:

```text
Script: D:\data\av2\organize_phase11_av2.ps1
Scheduled task: VehicleTrajectoryAV2Organize
Latest log pointer: D:\data\av2\logs\latest_phase11_data_organize_log.txt
In progress marker: D:\data\av2\motion-forecasting\EXTRACTION_IN_PROGRESS.txt
Failure marker: D:\data\av2\motion-forecasting\EXTRACTION_FAILED.txt
Ready marker: D:\data\av2\motion-forecasting\DATA_READY_FOR_PHASE11.txt
```

Before Phase 11, verify:

```bash
trajwinssh thddy@192.168.35.17 'powershell -NoProfile -Command "Test-Path D:\data\av2\motion-forecasting\DATA_READY_FOR_PHASE11.txt; Get-Content D:\data\av2\motion-forecasting\DATA_READY_FOR_PHASE11.txt"'
```

Safe remote execution rule:

```text
Do not run the full AV2 download as a long foreground SSH command.
First test one small object, then run long downloads from an interactive Windows
terminal or a validated detached Windows script that writes logs to disk.
Do not leave Codex waiting on a multi-hour SSH session.
If SSH starts timing out, stop s5cmd first and verify OpenSSH/Tailscale before
restarting any download.
```

Verified stop commands:

```bash
ssh thddy@192.168.35.17 "cmd /c schtasks /End /TN VehicleTrajectoryAV2Download & schtasks /Delete /TN VehicleTrajectoryAV2Download /F & taskkill /IM s5cmd.exe /F"
ssh thddy@192.168.35.17 "cmd /c tasklist /FI \"IMAGENAME eq s5cmd.exe\""
```

Keep this data out of Git. Do not copy the full dataset to Mac by default.

## 6. SSH/LAN/Tailscale Preflight

Use LAN first while Mac and HOME are on the same Wi-Fi. Use Tailscale only as
fallback or when outside the home network. Do not use userspace `tailscaled` or
a SOCKS proxy by default.

Primary connectivity check:

```bash
nc -vz -G 8 192.168.35.17 22
trajwinssh thddy@192.168.35.17 'hostname'
trajwinssh thddy@192.168.35.17 'hostname && whoami'
```

Expected verified output:

```text
HOME
home\thddy
```

Tailscale fallback target:

```bash
nc -vz -G 8 100.99.63.23 22
trajwinssh thddy@100.99.63.23 'hostname && whoami'
```

Tailscale app path:

```bash
TAILSCALE_CLI="/Applications/Tailscale.app/Contents/MacOS/Tailscale"
```

Check Tailscale:

```bash
"$TAILSCALE_CLI" status
"$TAILSCALE_CLI" ip -4
```

SSH helpers:

```bash
trajwinssh() {
  ssh \
    -o 'ServerAliveInterval=30' \
    -o 'ServerAliveCountMax=6' \
    -o 'ControlMaster=auto' \
    -o 'ControlPersist=8h' \
    -o 'ControlPath=~/.ssh/traj-win-%r@%h:%p' \
    "$@"
}

trajwinscp() {
  scp \
    -o 'ServerAliveInterval=30' \
    -o 'ServerAliveCountMax=6' \
    -o 'ControlMaster=auto' \
    -o 'ControlPersist=8h' \
    -o 'ControlPath=~/.ssh/traj-win-%r@%h:%p' \
    "$@"
}
```

Do not use the old `song` host unless HOME is unavailable and the user explicitly
asks to fall back to the secondary Windows machine.

Legacy `song` target:

```bash
ssh song@100.87.219.58 'hostname && whoami'
```

If the HOME LAN address changes, verify the current Windows LAN IP with Windows
`ipconfig`, then update this document and `PROJECT_STATUS.md`.

LAN address pattern:

```text
Mac current LAN: 192.168.35.42
HOME current LAN: 192.168.35.17
```

After Windows reboot or tunnel reconnect, bypass stale SSH multiplexing once:

```bash
ssh \
  -o ConnectTimeout=20 \
  -o ServerAliveInterval=30 \
  -o ServerAliveCountMax=6 \
  -o ControlMaster=no \
  thddy@192.168.35.17 'hostname && whoami'
```

Legacy userspace fallback:

```text
The previous userspace Tailscale + SOCKS 1055 workflow is no longer the normal
path. If the macOS app is logged out or unavailable, fix the app login/VPN
permission first. If userspace tailscaled logs
`policy requires hardware attestation`, classify that as Tailscale
infrastructure/policy failure and do not use it for project runs.
```

## 7. Windows Environment Preflight

Current verified state:

```text
HOME default Python is 3.12.7.
HOME GPU is NVIDIA GeForce RTX 2070 SUPER, 8192 MiB.
HOME Miniconda3 is installed at C:\Users\thddy\Miniconda3.
HOME vehicle_traj exists with Python 3.12 and CUDA PyTorch 2.11.0+cu128.
HOME vehicle_traj torch.cuda.is_available() is True.
HOME s5cmd is not required for current Phase 14 work and is only needed for
future direct AV2 download or resync.
```

Run once per Windows session:

```bash
trajwinssh thddy@192.168.35.17 'powershell -NoProfile -ExecutionPolicy Bypass -Command "python --version; where.exe python; nvidia-smi"'
```

PyTorch CUDA check:

```bash
trajwinssh thddy@192.168.35.17 'powershell -NoProfile -ExecutionPolicy Bypass -Command "& C:\Users\thddy\Miniconda3\Scripts\conda.exe run -n vehicle_traj python -c \"import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 0)\""'
```

If CUDA is unavailable:

```text
Classify as Windows environment failure, not model failure.
Do not run long training on Windows CPU unless explicitly requested.
Follow docs/WINDOWS_ENV_SETUP.md before Phase 7/8/9 training.
```

## 8. Sync Policy

Preferred sync direction:

```text
Mac source/config -> Windows data/training worker -> Mac lightweight result pullback
```

Allowed to sync to Windows:

```text
src/
configs/
tests/
requirements.txt
pyproject.toml
Makefile
README.md
AGENTS.md
GOAL_RUNBOOK.md
PROJECT_STATUS.md
docs/codex_vehicle_trajectory_project_plan.md
docs/windows_gpu_training_only_workflow.md
docs/WINDOWS_ENV_SETUP.md
docs/github_portfolio_workflow.md
CODEX_TASKS.md
data/processed/*.npz
data/processed/*.pkl
data/processed/*.csv
outputs/checkpoints/pca_codec.pkl, if PCA latent diffusion uses it
```

Do not sync to Windows by default:

```text
data/raw/
large AV2 processed files when already present on Windows
outputs/figures/
outputs/predictions/
outputs/tables/
large notebooks
Mac-only cache directories
```

Rationale:

```text
Windows only needs enough to train. Raw data, final analysis, and figures stay
out of Git; large AV2 raw/processed data stays on Windows to protect Mac disk
space, while result interpretation remains centralized on Mac.
```

## 9. Preparing Windows Project Directory

Create the Windows project root once:

```bash
trajwinssh thddy@192.168.35.17 'powershell -NoProfile -ExecutionPolicy Bypass -Command "New-Item -ItemType Directory -Force C:\Users\thddy\Documents\code\vehicle_trajectory_project | Out-Null"'
```

Recommended sync method for small/medium project state:

```bash
cd /Users/song-yonghwi/Documents/vehicle_trajectory_project
rsync -avz --delete \
  --exclude '.git/' \
  --exclude '.venv/' \
  --exclude '__pycache__/' \
  --exclude '.pytest_cache/' \
  --exclude 'data/raw/' \
  --exclude 'data/processed/*.npz' \
  --exclude 'data/processed/*.pkl' \
  --exclude 'outputs/figures/' \
  --exclude 'outputs/predictions/' \
  --exclude 'outputs/tables/' \
  --exclude 'windows_training_results/' \
  -e "ssh -o ServerAliveInterval=30 -o ServerAliveCountMax=6" \
  ./ thddy@192.168.35.17:'C:/Users/thddy/Documents/code/vehicle_trajectory_project/'
```

If `rsync` over SSH is unavailable on Windows, create a tarball on Mac and copy
it with `scp`, then extract on Windows:

```bash
cd /Users/song-yonghwi/Documents/vehicle_trajectory_project
tar \
  --exclude='.git' \
  --exclude='.venv' \
  --exclude='__pycache__' \
  --exclude='.pytest_cache' \
  --exclude='data/raw' \
  --exclude='data/processed/*.npz' \
  --exclude='data/processed/*.pkl' \
  --exclude='outputs/figures' \
  --exclude='outputs/predictions' \
  --exclude='outputs/tables' \
  --exclude='windows_training_results' \
  -czf /tmp/vehicle_trajectory_project_training_bundle.tgz .

trajwinscp /tmp/vehicle_trajectory_project_training_bundle.tgz \
  thddy@192.168.35.17:'C:\Users\thddy\Documents\code\vehicle_trajectory_project_training_bundle.tgz'

trajwinssh thddy@192.168.35.17 'powershell -NoProfile -ExecutionPolicy Bypass -Command "Set-Location C:\Users\thddy\Documents\code; Remove-Item -Recurse -Force vehicle_trajectory_project_sync_tmp -ErrorAction SilentlyContinue; New-Item -ItemType Directory vehicle_trajectory_project_sync_tmp | Out-Null; tar -xzf vehicle_trajectory_project_training_bundle.tgz -C vehicle_trajectory_project_sync_tmp; robocopy vehicle_trajectory_project_sync_tmp vehicle_trajectory_project /MIR /XD data\raw outputs\figures outputs\predictions outputs\tables windows_training_results; if ($LASTEXITCODE -le 7) { exit 0 } else { exit $LASTEXITCODE }"'
```

## 10. Windows Dependency Setup

Run after initial sync or requirements change:

```bash
trajwinssh thddy@192.168.35.17 'powershell -NoProfile -ExecutionPolicy Bypass -Command "Set-Location C:\Users\thddy\Documents\code\vehicle_trajectory_project; conda run -n vehicle_traj python -m pip install -U pip; conda run -n vehicle_traj python -m pip install -r requirements.txt"'
```

If CUDA-specific PyTorch is needed, install the correct wheel on Windows
manually or with the official PyTorch command for that GPU/CUDA version. Record
the exact command in the phase result.

## 11. Windows Training Commands

General pattern:

```powershell
Set-Location C:\Users\thddy\Documents\code\vehicle_trajectory_project
$env:PYTHONPATH="C:\Users\thddy\Documents\code\vehicle_trajectory_project;$env:PYTHONPATH"
conda run -n vehicle_traj python -m src.training.train `
  --config configs\<model>.yaml `
  --data data\processed\train_smoke.npz `
  --val_data data\processed\val_smoke.npz
```

For Windows-local AV2 data:

```powershell
conda run -n vehicle_traj python -m src.training.train `
  --config configs\<model>_av2.yaml `
  --data D:\data\vehicle_trajectory_project\processed\train_small.npz `
  --val_data D:\data\vehicle_trajectory_project\processed\val_small.npz `
  --output_dir D:\runs\vehicle_trajectory_project\<run_name>
```

Tiny GPU smoke after Phase 6:

```bash
trajwinssh thddy@192.168.35.17 'powershell -NoProfile -ExecutionPolicy Bypass -Command "Set-Location C:\Users\thddy\Documents\code\vehicle_trajectory_project; $env:PYTHONPATH=\"C:\Users\thddy\Documents\code\vehicle_trajectory_project;$env:PYTHONPATH\"; conda run -n vehicle_traj python -m src.training.train --config configs/lstm.yaml --max_epochs 1 --data data\processed\train_smoke.npz --val_data data\processed\val_smoke.npz"'
```

LSTM training:

```bash
trajwinssh thddy@192.168.35.17 'powershell -NoProfile -ExecutionPolicy Bypass -Command "Set-Location C:\Users\thddy\Documents\code\vehicle_trajectory_project; $env:PYTHONPATH=\"C:\Users\thddy\Documents\code\vehicle_trajectory_project;$env:PYTHONPATH\"; conda run -n vehicle_traj python -m src.training.train --config configs/lstm.yaml --data data\processed\train_smoke.npz --val_data data\processed\val_smoke.npz"'
```

Transformer training:

```bash
trajwinssh thddy@192.168.35.17 'powershell -NoProfile -ExecutionPolicy Bypass -Command "Set-Location C:\Users\thddy\Documents\code\vehicle_trajectory_project; $env:PYTHONPATH=\"C:\Users\thddy\Documents\code\vehicle_trajectory_project;$env:PYTHONPATH\"; conda run -n vehicle_traj python -m src.training.train --config configs/transformer.yaml --data data\processed\train_smoke.npz --val_data data\processed\val_smoke.npz"'
```

Direct diffusion training:

```bash
trajwinssh thddy@192.168.35.17 'powershell -NoProfile -ExecutionPolicy Bypass -Command "Set-Location C:\Users\thddy\Documents\code\vehicle_trajectory_project; $env:PYTHONPATH=\"C:\Users\thddy\Documents\code\vehicle_trajectory_project;$env:PYTHONPATH\"; conda run -n vehicle_traj python -m src.training.train --config configs/diffusion_direct.yaml --data data\processed\train_smoke.npz --val_data data\processed\val_smoke.npz"'
```

For AV2 small data, replace smoke paths:

```text
D:\data\vehicle_trajectory_project\processed\train_small.npz
D:\data\vehicle_trajectory_project\processed\val_small.npz
```

## 12. Long Run Logging

For longer training, use a timestamped run directory:

```powershell
Set-Location C:\Users\thddy\Documents\code\vehicle_trajectory_project
$run = Get-Date -Format "yyyyMMdd_HHmmss"
$logDir = "D:\runs\vehicle_trajectory_project\$run"
New-Item -ItemType Directory -Force $logDir | Out-Null
$env:PYTHONPATH="C:\Users\thddy\Documents\code\vehicle_trajectory_project;$env:PYTHONPATH"
conda run -n vehicle_traj python -m src.training.train `
  --config configs\lstm.yaml `
  --data data\processed\train_smoke.npz `
  --val_data data\processed\val_smoke.npz `
  *> "$logDir\train_stdout_stderr.log"
```

Every long run should record:

```text
1. hostname
2. date/time
3. git commit or source bundle checksum, if available
4. python version
5. torch version
6. CUDA availability and GPU name
7. exact training command
8. config path
9. train/val data paths
10. output checkpoint paths
```

## 13. Pullback Policy

Pull back from Windows to Mac:

```text
outputs/checkpoints/best_*.pt
outputs/checkpoints/last_*.pt
outputs/logs/*.csv
outputs/metrics/*.json
outputs/remote_runs/**/*.log
outputs/remote_runs/**/*.json
```

Do not pull back by default:

```text
data/processed/*.npz
data/raw/
large temporary tensors
large debug dumps
```

Example pullback:

```bash
mkdir -p /Users/song-yonghwi/Documents/vehicle_trajectory_project/windows_training_results

trajwinscp -r \
  thddy@192.168.35.17:'C:\Users\thddy\Documents\code\vehicle_trajectory_project\outputs\checkpoints' \
  /Users/song-yonghwi/Documents/vehicle_trajectory_project/windows_training_results/

trajwinscp -r \
  thddy@192.168.35.17:'C:\Users\thddy\Documents\code\vehicle_trajectory_project\outputs\logs' \
  /Users/song-yonghwi/Documents/vehicle_trajectory_project/windows_training_results/

trajwinscp -r \
  thddy@192.168.35.17:'C:\Users\thddy\Documents\code\vehicle_trajectory_project\outputs\metrics' \
  /Users/song-yonghwi/Documents/vehicle_trajectory_project/windows_training_results/
```

After pullback, copy or move only the chosen checkpoint into Mac `outputs/`:

```text
outputs/checkpoints/best_lstm.pt
outputs/checkpoints/best_transformer.pt
outputs/checkpoints/best_diffusion_direct.pt
```

## 14. Mac Validation After Windows Training

After checkpoint pullback, evaluation returns to Mac:

```bash
cd /Users/song-yonghwi/Documents/vehicle_trajectory_project

python -m src.evaluation.evaluate \
  --model lstm \
  --checkpoint outputs/checkpoints/best_lstm.pt \
  --data data/processed/val_smoke.npz \
  --out_dir outputs
```

Transformer:

```bash
python -m src.evaluation.evaluate \
  --model transformer \
  --checkpoint outputs/checkpoints/best_transformer.pt \
  --data data/processed/val_smoke.npz \
  --out_dir outputs
```

Diffusion:

```bash
python -m src.evaluation.evaluate \
  --model diffusion_direct \
  --checkpoint outputs/checkpoints/best_diffusion_direct.pt \
  --data data/processed/val_smoke.npz \
  --out_dir outputs
```

Mac then owns:

```text
outputs/predictions/
outputs/metrics/
outputs/tables/
outputs/figures/
outputs/report_summary.md
```

## 15. Failure Classification

Classify failures before changing code.

Connectivity failure:

```text
SSH timeout
Tailscale unavailable
stale SSH control socket
Windows sleeping/offline
long-running foreground SSH download tying up the remote shell
```

Windows environment failure:

```text
python missing
pip install failure
torch import failure
CUDA unavailable
GPU OOM before training begins
```

Sync failure:

```text
missing src/config/data file on Windows
wrong project root
old code version
train/val .npz missing
path quoting failure
```

Remote execution failure:

```text
long s5cmd process was started through foreground SSH
detached process inherited SSH stdio and kept the session open
large directory listing or download output made the command appear stuck
Scheduled Task was not verified before assuming the download was detached
```

Training-code failure:

```text
shape mismatch
loss NaN
checkpoint not saved
metric computation failure during validation loop
dataloader error from valid .npz
```

Model-performance issue:

```text
training completes but ADE/FDE is poor
validation does not improve
model overfits
diffusion samples have low diversity
```

Only training-code and model-performance issues should normally lead to model
or trainer changes. Connectivity, environment, and sync failures should be
fixed operationally first.

## 16. Result Note Template

Use this format after each Windows training run:

```text
## Windows Training Result

### Model
- ...

### Mac Source State
- Project root:
- Commit/checksum, if available:
- Config:
- Train data:
- Val data:

### Windows Environment
- Hostname:
- Python:
- Torch:
- CUDA:
- GPU:

### Command
- ...

### Pulled Back
- checkpoints:
- logs:
- metrics:

### Mac Validation
- evaluation command:
- ADE:
- FDE:
- Miss Rate:
- minADE/minFDE, if diffusion:

### Failure Class
- none / connectivity / environment / sync / training-code / performance

### Next Action
- ...
```

## 17. Minimal Operating Checklist

Before Windows training:

```text
[ ] Mac tests pass
[ ] processed train/val data validated on Mac
[ ] config checked
[ ] Windows reachable
[ ] Windows CUDA available
[ ] source/data synced to Windows
[ ] exact command recorded
```

After Windows training:

```text
[ ] checkpoint exists on Windows
[ ] train log exists on Windows
[ ] checkpoint/log/metric pulled back to Mac
[ ] Mac evaluation executed
[ ] predictions saved on Mac
[ ] metrics table saved on Mac
[ ] result note written
```

## 18. Final Rule

```text
Windows에서 하는 일은 AV2 데이터 보관, 데이터가 Windows에 있을 때의 전처리,
그리고 GPU 학습뿐이다. 프로젝트 판단, 평가, 그림, 표, 보고서는 모두 Mac에서 만든다.
```
