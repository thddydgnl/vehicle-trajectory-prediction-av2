# Windows GPU Training and AV2 Data Environment Setup

작성일: 2026-06-01

이 문서는 Windows 컴퓨터를 차량 궤적 예측 프로젝트의 AV2 데이터 저장소와 GPU
학습 노드로 쓰기 위한 환경 설정 체크리스트다.

## 1. Current Verified State

Verified from Mac:

```bash
ssh thddy@192.168.35.17 'hostname && whoami'
ssh thddy@100.99.63.23 'hostname && whoami'
```

Output:

```text
HOME
home\thddy
```

Connection priority:

```text
Primary: HOME LAN, thddy@192.168.35.17
Fallback: HOME Tailscale, thddy@100.99.63.23
Legacy secondary host: song, song@100.87.219.58
```

Windows home path:

```text
C:\Users\thddy
```

Windows AV2/data paths:

```text
Raw AV2: C:\Users\thddy\data\av2\motion-forecasting
Processed: C:\Users\thddy\data\vehicle_trajectory_project\processed
Runs: C:\Users\thddy\runs\vehicle_trajectory_project
```

Python / GPU preflight:

```text
Python 3.12.7
C:\Users\thddy\AppData\Local\Programs\Python\Python312\python.exe
NVIDIA GeForce RTX 2070 SUPER
Driver 591.86
GPU memory 8192 MiB
C drive free space about 192 GB
```

Current PyTorch state:

```text
HOME vehicle_traj environment not created yet.
HOME conda is not installed or not on PATH yet.
HOME s5cmd is missing at C:\Users\thddy\bin\s5cmd\s5cmd.exe.
```

Conclusion:

```text
HOME SSH and NVIDIA GPU are available.
HOME is not yet ready for long GPU training because vehicle_traj does not exist.
s5cmd is required for direct AV2 download on HOME.
```

## 2. Required Fix

Create a dedicated Python environment for this project with a CUDA-enabled
PyTorch build.

Important:

```text
HOME default Python is 3.12.7, but long training should still use a dedicated
vehicle_traj environment rather than the global Python install.
Use Python 3.12 unless the project intentionally moves to another supported
version.
```

Reason:

```text
The official PyTorch Windows install docs currently state Windows support for
Python 3.9-3.12. Use the official PyTorch selector before installing.
```

Reference:

```text
https://docs.pytorch.org/get-started/locally/
https://pytorch.org/get-started/
```

## 3. Recommended Conda Environment

If `conda` is missing on HOME, install Miniconda or Anaconda first, then reopen
PowerShell so `conda` is on PATH.

Run on Windows PowerShell:

```powershell
conda create -n vehicle_traj python=3.12 -y
conda activate vehicle_traj
python -m pip install -U pip
```

Install project dependencies after the repository is synced:

```powershell
Set-Location C:\Users\thddy\Documents\code\vehicle_trajectory_project
conda activate vehicle_traj
python -m pip install -r requirements.txt
```

Then install CUDA-enabled PyTorch using the current official selector:

```text
Go to https://pytorch.org/get-started/locally/
Select:
  PyTorch Build: Stable
  OS: Windows
  Package: Pip or Conda
  Language: Python
  Compute Platform: CUDA version supported by the selector
Run the generated command inside the vehicle_traj environment.
```

Example only, verify with the official selector before use:

```powershell
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

If the selector recommends a different CUDA wheel, use the selector output.

## 4. Verification Commands

Run on Windows:

```powershell
conda activate vehicle_traj
python --version
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no cuda')"
nvidia-smi
```

Expected:

```text
Python 3.12.x
torch version without +cpu
True
NVIDIA GeForce RTX 2070 SUPER
```

Run from Mac:

```bash
ssh thddy@192.168.35.17 'powershell -NoProfile -ExecutionPolicy Bypass -Command "conda activate vehicle_traj; python -c \"import torch; print(torch.__version__); print(torch.cuda.is_available())\""'
```

If `conda activate` does not work in non-interactive PowerShell, use:

```powershell
& C:\Users\thddy\anaconda3\Scripts\activate vehicle_traj
```

or:

```powershell
conda run -n vehicle_traj python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

## 5. Training Command Pattern

Once verified, Windows training commands should use the environment explicitly:

```bash
ssh thddy@192.168.35.17 'powershell -NoProfile -ExecutionPolicy Bypass -Command "Set-Location C:\Users\thddy\Documents\code\vehicle_trajectory_project; conda run -n vehicle_traj python -m src.training.train --config configs/lstm.yaml --data data\processed\train_smoke.npz --val_data data\processed\val_smoke.npz"'
```

Do not run long training until:

```text
torch.cuda.is_available() == True
```

## 6. AV2 Download Tooling

Install `s5cmd` on Windows if it is missing:

```powershell
conda install s5cmd -c conda-forge -y
```

If conda is slow, install the official release binary instead:

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

Create the Windows data directories:

```powershell
New-Item -ItemType Directory -Force C:\Users\thddy\data\av2\motion-forecasting | Out-Null
New-Item -ItemType Directory -Force C:\Users\thddy\data\vehicle_trajectory_project\processed | Out-Null
New-Item -ItemType Directory -Force C:\Users\thddy\runs\vehicle_trajectory_project | Out-Null
```

List and download AV2 Motion Forecasting:

```powershell
s5cmd --no-sign-request ls "s3://argoverse/datasets/av2/motion-forecasting/"
s5cmd --no-sign-request cp `
  "s3://argoverse/datasets/av2/motion-forecasting/*" `
  "C:\Users\thddy\data\av2\motion-forecasting\"
```

Do not copy the full AV2 dataset to Mac unless explicitly requested.

Do not run the full AV2 download as a multi-hour foreground SSH command. Use an
interactive Windows terminal or a detached script whose log file is verified
before leaving it unattended.

## 7. Failure Classification

Classify as Windows environment failure:

```text
default Python is 3.13 and PyTorch selector does not support it
conda is missing or not on PATH
torch version ends with +cpu
torch.cuda.is_available() is False
nvidia-smi works but PyTorch cannot see CUDA
conda environment activation fails
s5cmd missing or cannot access the public AV2 S3 path
long-running s5cmd foreground SSH session makes SSH appear unresponsive
```

Do not change model code to fix these issues. Fix the Windows environment first.
