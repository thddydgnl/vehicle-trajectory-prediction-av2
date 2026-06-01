# Windows GPU Training Environment Setup

작성일: 2026-06-01

이 문서는 Windows 컴퓨터를 차량 궤적 예측 프로젝트의 GPU 학습 노드로 쓰기 위한
환경 설정 체크리스트다.

## 1. Current Verified State

Verified from Mac:

```bash
ssh song@100.87.219.58 'hostname && whoami'
```

Output:

```text
Song
song\song
```

Windows home path:

```text
C:\Users\thddy
```

Python / GPU preflight:

```text
Python 3.13.5
C:\Users\thddy\anaconda3\python.exe
NVIDIA GeForce RTX 3080
Driver 591.86
GPU memory 10240 MiB
```

Current PyTorch state:

```text
torch 2.10.0+cpu
torch.cuda.is_available() == False
```

Conclusion:

```text
Windows SSH and NVIDIA GPU are available.
Windows is not yet ready for GPU training because the active PyTorch install is CPU-only.
```

## 2. Required Fix

Create a dedicated Python environment for this project with a CUDA-enabled
PyTorch build.

Important:

```text
Do not use the current default Python 3.13 environment for GPU training.
Use Python 3.12 unless official PyTorch docs confirm the selected PyTorch build
supports Python 3.13 on Windows.
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
NVIDIA GeForce RTX 3080
```

Run from Mac:

```bash
ssh song@100.87.219.58 'powershell -NoProfile -ExecutionPolicy Bypass -Command "conda activate vehicle_traj; python -c \"import torch; print(torch.__version__); print(torch.cuda.is_available())\""'
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
ssh song@100.87.219.58 'powershell -NoProfile -ExecutionPolicy Bypass -Command "Set-Location C:\Users\thddy\Documents\code\vehicle_trajectory_project; conda run -n vehicle_traj python -m src.training.train --config configs/lstm.yaml --data data\processed\train_smoke.npz --val_data data\processed\val_smoke.npz"'
```

Do not run long training until:

```text
torch.cuda.is_available() == True
```

## 6. Failure Classification

Classify as Windows environment failure:

```text
default Python is 3.13 and PyTorch selector does not support it
torch version ends with +cpu
torch.cuda.is_available() is False
nvidia-smi works but PyTorch cannot see CUDA
conda environment activation fails
```

Do not change model code to fix these issues. Fix the Windows environment first.
