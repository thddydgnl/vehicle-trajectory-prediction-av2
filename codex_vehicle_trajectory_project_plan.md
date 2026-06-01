---
title: "Codex용 차량·보행자 미래 궤적 예측 프로젝트 단계별 구현 계획서"
subtitle: "Argoverse 2 기반 LSTM vs Transformer vs Diffusion 비교 실험"
author: "프로젝트 구현 계획"
date: "2026-06-01"
lang: ko-KR
mainfont: "Noto Sans CJK KR"
CJKmainfont: "Noto Sans CJK KR"
monofont: "NanumGothicCoding"
geometry: margin=20mm
fontsize: 10pt
toc: true
toc-depth: 3
colorlinks: true
linkcolor: blue
urlcolor: blue
---

# 0. 문서 사용 방법

이 문서는 OpenAI Codex가 프로젝트를 단계별로 구현할 수 있도록 만든 실행 계획서이다.  
프로젝트 루트에는 이 파일을 `codex_vehicle_trajectory_project_plan.md`로 저장하고, 필요하면 아래 섹션을 분리해 `AGENTS.md`와 `CODEX_TASKS.md`로 저장한다.

권장 사용 방식은 다음과 같다.

1. 프로젝트 저장소 루트에 `codex_vehicle_trajectory_project_plan.md`를 저장한다.
2. Codex에게 이 문서를 먼저 읽게 한다.
3. 한 번에 전체 구현을 맡기지 않고, Phase 또는 Task 단위로 하나씩 진행한다.
4. 각 단계가 끝날 때마다 테스트와 smoke run을 실행한다.
5. 실패한 단계는 다음 단계로 넘어가지 않고 먼저 수정한다.

Codex에게 처음 줄 프롬프트는 다음과 같다.

```text
Read GOAL_RUNBOOK.md first.
Read PROJECT_STATUS.md before selecting the next phase.
Read codex_vehicle_trajectory_project_plan.md for the phase specification.
Do not implement the whole project at once.
Start with Phase 0 only.
After implementation, run the required tests and report:
1. changed files
2. commands executed
3. test results
4. remaining risks
5. Git commit/push status
6. next recommended phase
```

# 1. 프로젝트 고정 사양

## 1.1 프로젝트 이름

**Argoverse 2 기반 차량·보행자 미래 궤적 예측: LSTM, Transformer, Diffusion 비교 분석**

## 1.2 프로젝트 목표

자율주행 환경에서 주변 차량과 보행자의 과거 움직임을 보고 미래 위치를 예측한다. 단순히 하나의 모델만 구현하는 것이 아니라, 같은 데이터와 같은 지표에서 다음 모델들을 공정하게 비교한다.

1. Linear Extrapolation Baseline
2. LSTM Encoder-Decoder
3. Transformer Encoder
4. Conditional Diffusion Trajectory Generator
5. 선택 구현: PCA latent diffusion
6. 선택 분석: K-means trajectory pattern analysis

이 프로젝트의 목표는 논문 수준의 state-of-the-art 모델을 완벽히 재현하는 것이 아니다. 목표는 수업 프로젝트에서 구현 가능하면서도 난이도, 독창성, 분석력을 보여줄 수 있는 비교 실험형 프로젝트를 완성하는 것이다.

## 1.3 입력과 출력

| 항목 | 고정값 |
|---|---|
| Dataset | Argoverse 2 Motion Forecasting |
| Sampling rate | 10Hz |
| Observation length | 과거 5초 = 50 step |
| Prediction length | 미래 3초 = 30 step |
| Main target | focal vehicle |
| Extended target | vehicle + pedestrian |
| Coordinate system | 마지막 관측 시점 기준 상대좌표 + heading 정렬 |
| Input feature | rel_x, rel_y, velocity_x, velocity_y, sin_heading, cos_heading |
| Output | future trajectory `[30, 2]` |

## 1.4 모델 비교 범위

필수 구현 범위:

1. Linear baseline
2. LSTM
3. Transformer
4. ADE/FDE 평가
5. trajectory overlay 시각화

권장 구현 범위:

1. Diffusion direct 60D trajectory model
2. minADE/minFDE 평가
3. PCA 2D visualization
4. K-means cluster-wise error analysis

시간이 충분할 때 추가할 범위:

1. PCA latent diffusion
2. vehicle vs pedestrian 별도 결과
3. direct diffusion vs PCA latent diffusion 비교
4. 실패 사례 top-k 자동 저장

구현하지 않는 범위:

1. HD map / lane graph 사용
2. traffic light 정보 사용
3. multi-agent joint prediction
4. Argoverse leaderboard 제출용 SOTA 재현

위 항목은 발표에서 향후 연구로만 언급한다.

# 2. Codex 작업 고정 규칙

이 섹션은 필요하면 프로젝트 루트의 `AGENTS.md`로 그대로 분리한다.

## 2.1 Codex 기본 작업 원칙

Codex는 다음 규칙을 항상 지켜야 한다.

1. 요청받은 Phase 또는 Task만 구현한다.
2. 다음 Phase의 코드를 미리 크게 만들지 않는다.
3. 각 Phase마다 테스트를 추가하거나 갱신한다.
4. 모든 스크립트는 프로젝트 루트에서 실행 가능해야 한다.
5. 절대경로를 하드코딩하지 않는다.
6. raw data는 수정하지 않는다.
7. 생성 파일은 `data/processed/` 또는 `outputs/` 아래에 저장한다.
8. 결과가 없는데 성공했다고 말하지 않는다.
9. AV2 raw data가 없으면 synthetic data로 pipeline을 먼저 검증한다.
10. 매 작업 종료 시 변경 파일, 실행 명령, 결과, 남은 위험을 보고한다.

## 2.2 데이터 누수 방지 규칙

다음 규칙은 반드시 지켜야 한다.

1. scaler는 train set으로만 fit한다.
2. PCA는 train set의 future trajectory로만 fit한다.
3. input feature 계산에 future coordinate를 사용하지 않는다.
4. 같은 `scenario_id`가 train과 validation에 동시에 들어가지 않게 한다.
5. validation/test set을 보고 hyperparameter를 과도하게 튜닝하지 않는다.
6. 데이터 split 이후에 train/val/test가 섞이지 않게 파일명을 명확히 한다.

## 2.3 좌표 변환 규칙

모든 모델의 입력과 출력은 마지막 관측 시점을 기준으로 상대좌표화한다.

```python
p0 = position[obs_len - 1]
theta = heading[obs_len - 1]

p_rel = R(-theta) @ (p_global - p0)
p_global_recovered = R(theta) @ p_rel + p0
```

검증 기준:

1. 마지막 관측 지점은 변환 후 거의 `(0, 0)`이어야 한다.
2. inverse transform을 적용하면 원래 global coordinate를 복원해야 한다.
3. 좌표 단위는 meter를 유지한다.

## 2.4 공통 데이터 포맷

전처리 결과는 `.npz`로 저장한다.

```text
X: float32 [N, 50, F]
Y: float32 [N, 30, 2]
mask_x: bool [N, 50]
mask_y: bool [N, 30]
object_type: int64 [N]
scenario_id: str/object [N]
track_id: str/object [N]
origin: float32 [N, 2]
theta: float32 [N]
```

권장 feature 순서:

```text
X[..., 0] = rel_x
X[..., 1] = rel_y
X[..., 2] = velocity_x
X[..., 3] = velocity_y
X[..., 4] = sin_heading
X[..., 5] = cos_heading
```

선택 feature:

```text
object_type_onehot_vehicle
object_type_onehot_pedestrian
```

## 2.5 평가 지표

필수 지표:

1. ADE
2. FDE
3. Miss Rate
4. Latency
5. Parameters

Diffusion 또는 multi-sample 모델에서 추가할 지표:

1. minADE
2. minFDE
3. sample diversity

shape 규칙:

```text
single prediction:
pred: [B, T, 2]
gt: [B, T, 2]

multi-sample prediction:
pred_samples: [B, K, T, 2]
gt: [B, T, 2]
```

## 2.6 구현 순서

모델은 반드시 아래 순서로 구현한다.

```text
Linear -> LSTM -> Transformer -> Direct Diffusion -> PCA Latent Diffusion
```

분석은 아래 순서로 구현한다.

```text
metrics -> trajectory overlay -> error histogram -> PCA -> K-means -> report summary
```

# 3. 최종 저장소 구조

Codex는 최종적으로 다음 구조를 만들거나 유지한다.

```text
trajectory-project/
  AGENTS.md
  codex_vehicle_trajectory_project_plan.md
  CODEX_TASKS.md
  README.md
  requirements.txt
  pyproject.toml
  Makefile

  configs/
    preprocess_smoke.yaml
    preprocess_small.yaml
    linear.yaml
    lstm.yaml
    transformer.yaml
    diffusion_direct.yaml
    diffusion_pca.yaml
    analysis.yaml

  data/
    raw/
      av2_motion_forecasting/
    processed/
      train_smoke.npz
      val_smoke.npz
      test_smoke.npz
      train_small.npz
      val_small.npz
      test_small.npz
      scaler.pkl
      metadata.csv

  src/
    __init__.py

    datasets/
      __init__.py
      av2_dataset.py
      preprocess_av2.py
      synthetic.py
      validate_processed.py

    models/
      __init__.py
      linear.py
      lstm.py
      transformer.py
      diffusion.py
      pca_latent.py

    training/
      __init__.py
      losses.py
      trainer.py
      train.py

    evaluation/
      __init__.py
      metrics.py
      evaluate.py
      latency.py

    analysis/
      __init__.py
      pca_analysis.py
      kmeans_analysis.py
      error_analysis.py

    visualization/
      __init__.py
      plot_trajectories.py
      plot_errors.py
      plot_pca.py
      plot_clusters.py

    utils/
      __init__.py
      config.py
      device.py
      seed.py
      paths.py
      logging.py
      geometry.py
      io.py

  tests/
    test_geometry.py
    test_metrics.py
    test_synthetic_data.py
    test_models_shape.py
    test_diffusion_step.py

  notebooks/
    01_data_check.ipynb
    02_trajectory_visualization.ipynb
    03_error_analysis.ipynb

  outputs/
    checkpoints/
    predictions/
    metrics/
    figures/
    tables/
    logs/
```

`data/raw/av2_motion_forecasting/` is optional on Mac and should only hold a
tiny local sample if needed. The primary full AV2 Motion Forecasting dataset is
stored on Windows:

```text
C:\Users\thddy\data\av2\motion-forecasting
```

Primary Windows processed data path:

```text
C:\Users\thddy\data\vehicle_trajectory_project\processed
```

# 4. 단계별 구현 계획

## Phase 0. Repository Setup

### 목표

프로젝트 기본 구조를 만들고, CPU/CUDA/MPS 환경에서 실행 가능한 최소 골격을 만든다.

### 생성 파일

```text
README.md
requirements.txt
pyproject.toml
Makefile
configs/
src/
tests/
outputs/
src/utils/config.py
src/utils/device.py
src/utils/seed.py
src/utils/paths.py
src/utils/logging.py
src/utils/io.py
```

### 필수 구현

다음 함수를 구현한다.

```python
load_yaml_config(path: str) -> dict
set_seed(seed: int) -> None
get_device() -> torch.device
ensure_dir(path: str | Path) -> Path
save_json(obj: dict, path: str | Path) -> None
load_json(path: str | Path) -> dict
```

### requirements.txt 기본 패키지

```text
torch
numpy
pandas
pyarrow
scikit-learn
matplotlib
tqdm
pyyaml
joblib
pytest
```

`av2` 패키지는 설치 가능한 환경일 때만 추가한다. 설치되어 있지 않아도 synthetic pipeline은 동작해야 한다.

### 완료 기준

```bash
python -c "from src.utils.device import get_device; print(get_device())"
pytest -q
```

---

## Phase 1. Synthetic Smoke Dataset

### 목표

Argoverse 2 raw data 없이도 전체 pipeline을 개발할 수 있도록 synthetic trajectory dataset을 만든다.

### 이유

AV2 data parser는 복잡할 수 있다. 처음부터 AV2 parser에 막히면 모델과 평가 코드를 만들 수 없다. 따라서 synthetic data로 shape, dataloader, metrics, training, visualization을 먼저 검증한다.

### 생성 파일

```text
src/datasets/synthetic.py
configs/preprocess_smoke.yaml
tests/test_synthetic_data.py
```

### synthetic trajectory 패턴

1. straight motion
2. slowdown
3. left turn
4. right turn
5. stop-and-go
6. pedestrian-like random direction change

### 출력 파일

```text
data/processed/train_smoke.npz
data/processed/val_smoke.npz
data/processed/test_smoke.npz
```

### CLI

```bash
python -m src.datasets.synthetic \
  --out_dir data/processed \
  --num_samples 1000 \
  --obs_len 50 \
  --pred_len 30 \
  --seed 42
```

### 완료 기준

```bash
python -m src.datasets.synthetic --out_dir data/processed --num_samples 1000
pytest tests/test_synthetic_data.py -q
```

---

## Phase 2. Geometry and Coordinate Transform

### 목표

상대좌표 변환과 inverse transform을 구현한다.

### 생성 파일

```text
src/utils/geometry.py
tests/test_geometry.py
```

### 필수 함수

```python
rotation_matrix(theta: float) -> np.ndarray

to_relative_coords(
    positions: np.ndarray,
    origin: np.ndarray,
    theta: float,
) -> np.ndarray

to_global_coords(
    rel_positions: np.ndarray,
    origin: np.ndarray,
    theta: float,
) -> np.ndarray

wrap_angle(angle: np.ndarray | float) -> np.ndarray | float
```

### 완료 기준

```bash
pytest tests/test_geometry.py -q
```

테스트해야 할 내용:

1. translation이 정확히 적용된다.
2. rotation이 정확히 적용된다.
3. inverse transform으로 원래 좌표를 복원한다.
4. 마지막 관측점이 상대좌표에서 `(0, 0)`에 가깝다.

---

## Phase 3. Dataset and DataLoader

### 목표

전처리된 `.npz` 파일을 PyTorch Dataset으로 읽는 클래스를 만든다.

### 생성 파일

```text
src/datasets/av2_dataset.py
tests/test_synthetic_data.py
```

### 필수 클래스

```python
class TrajectoryDataset(torch.utils.data.Dataset):
    def __init__(self, npz_path: str | Path):
        ...

    def __len__(self) -> int:
        ...

    def __getitem__(self, idx: int) -> dict:
        ...
```

각 item은 다음 key를 포함해야 한다.

```python
{
    "X": torch.FloatTensor,          # [50, F]
    "Y": torch.FloatTensor,          # [30, 2]
    "mask_x": torch.BoolTensor,      # [50]
    "mask_y": torch.BoolTensor,      # [30]
    "object_type": torch.LongTensor, # scalar
    "scenario_id": str,
    "track_id": str,
    "origin": torch.FloatTensor,     # [2]
    "theta": torch.FloatTensor,      # scalar
}
```

### helper 함수

```python
create_dataloader(
    npz_path: str | Path,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 0,
) -> DataLoader
```

### 완료 기준

```bash
python -m src.datasets.synthetic --out_dir data/processed --num_samples 1000
pytest tests/test_synthetic_data.py -q
```

---

## Phase 4. Metrics

### 목표

모델 학습보다 먼저 평가 지표를 구현한다.

### 생성 파일

```text
src/evaluation/metrics.py
tests/test_metrics.py
```

### 필수 함수

```python
ade(pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor
fde(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor
min_ade(pred_samples: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor
min_fde(pred_samples: torch.Tensor, gt: torch.Tensor) -> torch.Tensor
miss_rate(pred: torch.Tensor, gt: torch.Tensor, threshold: float = 2.0) -> torch.Tensor
count_parameters(model: torch.nn.Module) -> int
```

### 완료 기준

```bash
pytest tests/test_metrics.py -q
```

테스트는 작은 tensor를 사용해 exact expected value를 검증한다.

---

## Phase 5. Linear Baseline

### 목표

학습이 필요 없는 직선 extrapolation baseline을 구현한다.

### 생성 파일

```text
src/models/linear.py
src/evaluation/evaluate.py
configs/linear.yaml
```

### 필수 클래스

```python
class LinearExtrapolation:
    def __init__(self, pred_len: int = 30, dt: float = 0.1):
        ...

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        ...
```

### 계산식

```python
p_last = X[:, -1, 0:2]
v_last = X[:, -1, 2:4]

for k in range(pred_len):
    y_hat[:, k, :] = p_last + v_last * ((k + 1) * dt)
```

### 평가 CLI

```bash
python -m src.evaluation.evaluate \
  --model linear \
  --data data/processed/val_smoke.npz \
  --config configs/linear.yaml \
  --out_dir outputs
```

### 출력 파일

```text
outputs/predictions/linear_val.pkl
outputs/metrics/linear_val_metrics.json
outputs/tables/linear_val_metrics.csv
```

### 완료 기준

```bash
python -m src.evaluation.evaluate \
  --model linear \
  --data data/processed/val_smoke.npz \
  --config configs/linear.yaml \
  --out_dir outputs
```

결과에 ADE, FDE, Miss Rate, Latency가 포함되어야 한다.

---

## Phase 6. Common Training Pipeline

### 목표

LSTM, Transformer, Diffusion에서 재사용 가능한 학습 loop를 만든다.

### 생성 파일

```text
src/training/losses.py
src/training/trainer.py
src/training/train.py
```

### trainer 기능

1. train loop
2. validation loop
3. checkpoint save
4. best model tracking by validation ADE
5. gradient clipping
6. early stopping
7. CSV 또는 JSON logging
8. device selection
9. reproducible seed

### loss 함수

```python
trajectory_mse_loss(pred, gt, mask=None)
trajectory_smooth_l1_loss(pred, gt, mask=None)
endpoint_loss(pred, gt)
combined_trajectory_loss(pred, gt, mask=None, endpoint_weight=0.0)
```

### 출력 파일

```text
outputs/checkpoints/best_{model_name}.pt
outputs/checkpoints/last_{model_name}.pt
outputs/logs/{model_name}_train_log.csv
outputs/metrics/{model_name}_val_metrics.json
```

### 완료 기준

```bash
python -m src.training.train \
  --config configs/lstm.yaml \
  --max_epochs 1 \
  --data data/processed/train_smoke.npz \
  --val_data data/processed/val_smoke.npz
```

이 단계에서는 tiny model 또는 placeholder model로 training loop만 검증해도 된다.

---

## Phase 7. LSTM Encoder-Decoder

### 목표

첫 번째 deep learning trajectory prediction model을 구현한다.

### 생성 파일

```text
src/models/lstm.py
configs/lstm.yaml
tests/test_models_shape.py
```

### 필수 클래스

```python
class LSTMForecast(torch.nn.Module):
    def __init__(
        self,
        input_dim: int,
        pred_len: int = 30,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        ...

    def forward(
        self,
        x: torch.Tensor,
        teacher_y: torch.Tensor | None = None,
        teacher_forcing_ratio: float = 0.0,
    ) -> torch.Tensor:
        ...
```

### shape

```text
input:  x [B, 50, F]
output: pred [B, 30, 2]
```

### config 기본값

```yaml
model:
  name: lstm
  input_dim: 6
  hidden_dim: 128
  num_layers: 2
  dropout: 0.1
  pred_len: 30

training:
  batch_size: 64
  epochs: 30
  learning_rate: 0.001
  weight_decay: 0.0001
  optimizer: adamw
  loss: smooth_l1
  gradient_clip: 1.0
  early_stopping_patience: 7
  seed: 42
```

### 완료 기준

```bash
pytest tests/test_models_shape.py -q
python -m src.training.train --config configs/lstm.yaml --max_epochs 1 \
  --data data/processed/train_smoke.npz \
  --val_data data/processed/val_smoke.npz
python -m src.evaluation.evaluate --model lstm \
  --checkpoint outputs/checkpoints/best_lstm.pt \
  --data data/processed/val_smoke.npz \
  --out_dir outputs
```

---

## Phase 8. Transformer Encoder

### 목표

과거 50 step 전체에 self-attention을 적용하는 Transformer regression model을 구현한다.

### 생성 파일

```text
src/models/transformer.py
configs/transformer.yaml
tests/test_models_shape.py
```

### 필수 클래스

```python
class PositionalEncoding(torch.nn.Module):
    ...

class TransformerForecast(torch.nn.Module):
    def __init__(
        self,
        input_dim: int,
        pred_len: int = 30,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
    ):
        ...

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ...
```

### shape

```text
input:  x [B, 50, F]
output: pred [B, 30, 2]
```

### config 기본값

```yaml
model:
  name: transformer
  input_dim: 6
  d_model: 128
  nhead: 4
  num_layers: 3
  dim_feedforward: 256
  dropout: 0.1
  pred_len: 30

training:
  batch_size: 32
  epochs: 50
  learning_rate: 0.0003
  weight_decay: 0.0001
  optimizer: adamw
  loss: smooth_l1
  gradient_clip: 1.0
  early_stopping_patience: 10
  seed: 42
```

### 완료 기준

```bash
pytest tests/test_models_shape.py -q
python -m src.training.train --config configs/transformer.yaml --max_epochs 1 \
  --data data/processed/train_smoke.npz \
  --val_data data/processed/val_smoke.npz
python -m src.evaluation.evaluate --model transformer \
  --checkpoint outputs/checkpoints/best_transformer.pt \
  --data data/processed/val_smoke.npz \
  --out_dir outputs
```

---

## Phase 9. Direct Diffusion Model

### 목표

미래 trajectory `[30, 2]`를 60차원 vector로 flatten하여 conditional diffusion을 구현한다.

```text
Y: [B, 30, 2]
x0: [B, 60]
cond: encoded past trajectory [B, cond_dim]
```

### 생성 파일

```text
src/models/diffusion.py
configs/diffusion_direct.yaml
tests/test_diffusion_step.py
```

### 필수 component

```python
class SinusoidalTimeEmbedding(torch.nn.Module):
    ...

class TrajectoryConditionEncoder(torch.nn.Module):
    ...

class DiffusionDenoiser(torch.nn.Module):
    ...

class GaussianDiffusionTrajectory(torch.nn.Module):
    ...
```

### 필수 method

```python
q_sample(x0, t, noise)
predict_noise(xt, t, cond)
training_loss(X, Y)
sample(X, num_samples=6, num_steps=None)
```

### 학습 목표

```python
x0 = Y.view(B, 60)
noise = torch.randn_like(x0)
t = torch.randint(0, num_diffusion_steps, (B,))
xt = sqrt_alpha_bar[t] * x0 + sqrt_one_minus_alpha_bar[t] * noise
cond = encoder(X)
noise_hat = denoiser(xt, t, cond)
loss = mse(noise_hat, noise)
```

### sampling output

```text
samples: [B, K, 30, 2]
```

### config 기본값

```yaml
model:
  name: diffusion_direct
  input_dim: 6
  pred_len: 30
  trajectory_dim: 60
  cond_dim: 128
  hidden_dim: 256
  diffusion_steps: 100
  sampling_steps: 50
  beta_start: 0.0001
  beta_end: 0.02
  num_samples: 6

training:
  batch_size: 64
  epochs: 50
  learning_rate: 0.0003
  weight_decay: 0.0001
  optimizer: adamw
  loss: noise_mse
  gradient_clip: 1.0
  early_stopping_patience: 10
  seed: 42
```

### 완료 기준

```bash
pytest tests/test_diffusion_step.py -q
python -m src.training.train --config configs/diffusion_direct.yaml --max_epochs 1 \
  --data data/processed/train_smoke.npz \
  --val_data data/processed/val_smoke.npz
python -m src.evaluation.evaluate --model diffusion_direct \
  --checkpoint outputs/checkpoints/best_diffusion_direct.pt \
  --data data/processed/val_smoke.npz \
  --out_dir outputs
```

평가 결과에는 ADE/FDE와 minADE/minFDE를 구분해서 저장한다.

---

## Phase 10. PCA Latent Diffusion

### 목표

미래 궤적 60D를 PCA latent 12D로 압축한 뒤, latent space에서 diffusion을 학습한다.

```text
Y_flat: [N, 60]
z: [N, 12]
```

### 생성 파일

```text
src/models/pca_latent.py
configs/diffusion_pca.yaml
src/analysis/pca_analysis.py
```

### 필수 클래스

```python
class PCATrajectoryCodec:
    def __init__(self, n_components: int = 12):
        ...

    def fit(self, Y_train: np.ndarray) -> None:
        ...

    def transform(self, Y: np.ndarray) -> np.ndarray:
        ...

    def inverse_transform(self, z: np.ndarray) -> np.ndarray:
        ...

    def save(self, path: str | Path) -> None:
        ...

    @classmethod
    def load(cls, path: str | Path) -> "PCATrajectoryCodec":
        ...
```

### 완료 기준

```bash
python -m src.analysis.pca_analysis \
  --train_data data/processed/train_smoke.npz \
  --out_dir outputs
python -m src.training.train --config configs/diffusion_pca.yaml --max_epochs 1 \
  --data data/processed/train_smoke.npz \
  --val_data data/processed/val_smoke.npz
```

출력:

```text
outputs/checkpoints/pca_codec.pkl
outputs/figures/pca_explained_variance.png
```

---

## Phase 11. Argoverse 2 Preprocessing

### 목표

Argoverse 2 Motion Forecasting scenario를 읽고 프로젝트 공통 `.npz` 포맷으로 변환한다.

### 생성 파일

```text
src/datasets/preprocess_av2.py
src/datasets/validate_processed.py
configs/preprocess_small.yaml
```

### raw data 위치

Primary Windows location:

```text
C:\Users\thddy\data\av2\motion-forecasting
```

Optional Mac sample location:

```text
data/raw/av2_motion_forecasting/
```

Codex는 실제 실행할 machine의 local directory를 inspect한 뒤 파일 구조를 판단해야 한다.
AV2 file layout을 확인하지 않고 확정적으로 가정하지 않는다. Mac 용량을 보호하기
위해 full AV2 preprocessing은 Windows-local raw data를 대상으로 실행할 수 있다.

### target selection 옵션

```text
--target_mode focal
--target_mode scored
--target_mode all_supported
```

기본값:

```text
target_mode: focal
```

지원 object type:

```text
VEHICLE
PEDESTRIAN
```

### 전처리 절차

1. scenario를 load한다.
2. `scenario_id`를 읽는다.
3. `focal_track_id`를 읽는다.
4. track을 반복한다.
5. target track을 선택한다.
6. object state를 timestep 기준으로 정렬한다.
7. 과거 50 step을 만든다.
8. 미래 30 step을 만든다.
9. required step이 부족한 track은 제외하거나 mask 처리한다.
10. 마지막 관측 위치와 heading 기준으로 상대좌표 변환한다.
11. 입력 feature를 만든다.
12. object type id를 저장한다.
13. train/val/test split을 만든다.
14. scaler를 train split에만 fit한다.
15. `metadata.csv`를 저장한다.

### CLI

```bash
python -m src.datasets.preprocess_av2 \
  --raw_dir data/raw/av2_motion_forecasting \
  --out_dir data/processed \
  --num_scenarios 100 \
  --target_types VEHICLE PEDESTRIAN \
  --obs_len 50 \
  --pred_len 30 \
  --target_mode focal \
  --seed 42
```

Windows full/small AV2 run pattern:

```powershell
Set-Location C:\Users\thddy\Documents\code\vehicle_trajectory_project
conda run -n vehicle_traj python -m src.datasets.preprocess_av2 `
  --raw_dir C:\Users\thddy\data\av2\motion-forecasting `
  --out_dir C:\Users\thddy\data\vehicle_trajectory_project\processed `
  --num_scenarios 100 `
  --target_types VEHICLE PEDESTRIAN `
  --obs_len 50 `
  --pred_len 30 `
  --target_mode focal `
  --seed 42
```

### validation CLI

```bash
python -m src.datasets.validate_processed --npz data/processed/train_small.npz
```

For Windows-local AV2 processed data:

```powershell
conda run -n vehicle_traj python -m src.datasets.validate_processed `
  --npz C:\Users\thddy\data\vehicle_trajectory_project\processed\train_small.npz
```

검증 항목:

1. shape
2. dtype
3. no NaN
4. no Inf
5. final observed relative position near `(0, 0)`
6. mask shape
7. object type value
8. scenario_id split leakage

### 완료 기준

```bash
python -m src.datasets.preprocess_av2 \
  --raw_dir data/raw/av2_motion_forecasting \
  --out_dir data/processed \
  --num_scenarios 100
python -m src.datasets.validate_processed --npz data/processed/train_small.npz
```

If using Windows-local AV2 data, the equivalent completion commands must run on
Windows over SSH with the Windows paths above.

AV2 raw data가 없으면 성공으로 보고하지 않는다. Phase 11 전이면 synthetic data
pipeline으로 계속 진행하고, Phase 11 이상이면 Windows-local AV2 download/verification을
먼저 완료해야 한다고 명확히 보고한다.

---

## Phase 12. Visualization

### 목표

최종 발표에 사용할 그림을 자동 생성한다.

### 생성 파일

```text
src/visualization/plot_trajectories.py
src/visualization/plot_errors.py
src/visualization/plot_pca.py
src/visualization/plot_clusters.py
```

### 필수 figure

```text
outputs/figures/trajectory_overlay_linear_lstm_transformer.png
outputs/figures/trajectory_overlay_diffusion_samples.png
outputs/figures/error_histogram_ade.png
outputs/figures/error_histogram_fde.png
outputs/figures/pca_trajectory_space.png
outputs/figures/kmeans_clusters.png
outputs/figures/top_error_cases/
```

### trajectory overlay 포함 요소

1. past trajectory
2. ground truth future
3. Linear prediction
4. LSTM prediction
5. Transformer prediction
6. Diffusion samples, if available

### CLI

```bash
python -m src.visualization.plot_trajectories \
  --data data/processed/val_smoke.npz \
  --predictions outputs/predictions \
  --out_dir outputs/figures \
  --num_cases 20
```

### 완료 기준

최소 5개 이상의 PNG figure가 `outputs/figures/` 아래에 생성되어야 한다.

---

## Phase 13. PCA and K-means Analysis

### 목표

미래 trajectory pattern을 PCA와 K-means로 분석하고, cluster별 model error를 비교한다.

### 생성 파일

```text
src/analysis/pca_analysis.py
src/analysis/kmeans_analysis.py
src/analysis/error_analysis.py
configs/analysis.yaml
```

### PCA 분석

```text
Y: [N, 30, 2]
Y_flat: [N, 60]
```

수행할 작업:

1. PCA 2D visualization
2. PCA explained variance plot
3. PCA latent representation 저장

### K-means 분석

권장 pipeline:

```text
Y_flat [N, 60] -> PCA 12D -> K-means
```

시도할 cluster 수:

```text
k = 3
k = 5
```

기본값:

```text
k = 5
```

### 출력 파일

```text
outputs/tables/cluster_summary.csv
outputs/tables/cluster_metrics.csv
outputs/figures/kmeans_clusters.png
```

### cluster metric

각 cluster와 각 model에 대해 다음을 계산한다.

```text
count
ADE
FDE
minADE
minFDE
Miss Rate
```

### 완료 기준

```bash
python -m src.analysis.pca_analysis \
  --data data/processed/val_smoke.npz \
  --out_dir outputs
python -m src.analysis.kmeans_analysis \
  --data data/processed/val_smoke.npz \
  --predictions outputs/predictions \
  --out_dir outputs
```

결과에는 어떤 모델이 어떤 trajectory pattern에서 실패하는지 보여주는 표가 포함되어야 한다.

---

## Phase 14. Final Experiment Matrix

### 목표

같은 데이터, 같은 입력, 같은 지표로 모델을 공정하게 비교한다.

### 실험 목록

| ID | 실험 | 비교 모델 | 목적 |
|---|---|---|---|
| E1 | Linear vs LSTM | Linear, LSTM | sequence learning이 단순 직선보다 나은지 확인 |
| E2 | LSTM vs Transformer | LSTM, Transformer | attention이 도움이 되는지 확인 |
| E3 | Transformer vs Diffusion | Transformer, Diffusion | single prediction과 multi-sample generation 비교 |
| E4 | Relative vs Global | LSTM relative/global | 상대좌표 전처리 효과 확인 |
| E5 | Vehicle vs Pedestrian | same model | 객체 종류별 난이도 차이 분석 |
| E6 | Direct vs PCA Diffusion | direct, PCA diffusion | PCA 압축 효과 확인 |
| E7 | K-means cluster metrics | all models | trajectory pattern별 약점 분석 |

### 최종 비교표

```text
outputs/tables/model_comparison.csv
outputs/tables/model_comparison.md
```

필수 column:

```text
model
data_split
target_type
ADE
FDE
minADE
minFDE
Miss_Rate
Latency_ms
Params
Notes
```

### 완료 기준

```bash
python -m src.evaluation.evaluate \
  --all_models \
  --data data/processed/val_small.npz \
  --out_dir outputs
```

`--all_models` 구현이 복잡하면 다음 script를 만든다.

```bash
python scripts/run_all_evaluations.py
```

---

## Phase 15. Final Report Assets

### 목표

발표자료에 바로 넣을 수 있는 표, 그림, 요약 문서를 생성한다.

### 최종 산출물

```text
outputs/tables/model_comparison.md
outputs/tables/cluster_metrics.csv
outputs/figures/trajectory_overlay_best_cases.png
outputs/figures/trajectory_overlay_worst_cases.png
outputs/figures/diffusion_samples_interesting_case.png
outputs/figures/pca_trajectory_space.png
outputs/figures/kmeans_clusters.png
outputs/figures/error_histogram_fde.png
outputs/report_summary.md
```

### `report_summary.md` 포함 내용

1. Project problem
2. Dataset
3. Input/output setting
4. Models
5. Evaluation metrics
6. Main results
7. Error analysis
8. Vehicle vs pedestrian comparison
9. Diffusion sample interpretation
10. Limitations
11. Future work

### 완료 기준

```bash
python -m src.analysis.error_analysis --out_dir outputs
```

프로젝트는 다음 조건을 만족하면 발표 가능한 상태이다.

1. Linear baseline runs.
2. LSTM runs.
3. Transformer runs.
4. Diffusion generates multiple future trajectories.
5. Metrics table is created.
6. At least five presentation figures are created.
7. Final report summary is created.

# 5. Codex에게 순서대로 줄 작업 프롬프트

이 섹션은 필요하면 `CODEX_TASKS.md`로 따로 저장한다.  
Codex에게는 아래 Task를 하나씩만 지시한다.

## Task 0. Initialize Repository

```text
Read codex_vehicle_trajectory_project_plan.md.
Start Phase 0 only.
Create the repository skeleton, utility modules, requirements.txt,
pyproject.toml, Makefile, and minimal pytest setup.
Do not implement data preprocessing or models yet.

After implementation, run:
pytest -q
python -c "from src.utils.device import get_device; print(get_device())"

Report changed files, commands run, test results, and next task.
```

## Task 1. Create Synthetic Smoke Dataset

```text
Read codex_vehicle_trajectory_project_plan.md.
Implement Phase 1 only.
Create a synthetic trajectory generator that saves train_smoke.npz,
val_smoke.npz, and test_smoke.npz with the required data format.

Generate patterns:
- straight
- slowdown
- left turn
- right turn
- stop-and-go
- pedestrian-like random direction change

Add tests for shape, dtype, masks, and required keys.

Run:
python -m src.datasets.synthetic --out_dir data/processed --num_samples 1000
pytest tests/test_synthetic_data.py -q
```

## Task 2. Implement Coordinate Geometry

```text
Read codex_vehicle_trajectory_project_plan.md.
Implement Phase 2 only.
Create geometry utilities for rotation_matrix, to_relative_coords,
to_global_coords, and wrap_angle.

Add tests for inverse coordinate recovery and final observed point near zero.
Run:
pytest tests/test_geometry.py -q
```

## Task 3. Implement Dataset Loader

```text
Read codex_vehicle_trajectory_project_plan.md.
Implement Phase 3 only.
Create TrajectoryDataset and create_dataloader for processed .npz files.
Use synthetic smoke data for tests.

Run:
python -m src.datasets.synthetic --out_dir data/processed --num_samples 1000
pytest tests/test_synthetic_data.py -q
```

## Task 4. Implement Metrics

```text
Read codex_vehicle_trajectory_project_plan.md.
Implement Phase 4 only.
Create ADE, FDE, minADE, minFDE, Miss Rate, and parameter count metrics.
Add exact-value unit tests using small tensors.

Run:
pytest tests/test_metrics.py -q
```

## Task 5. Implement Linear Baseline

```text
Read codex_vehicle_trajectory_project_plan.md.
Implement Phase 5 only.
Create LinearExtrapolation and evaluation CLI for the linear model.
Use synthetic smoke validation data.

Run:
python -m src.datasets.synthetic --out_dir data/processed --num_samples 1000
python -m src.evaluation.evaluate --model linear \
  --data data/processed/val_smoke.npz \
  --config configs/linear.yaml \
  --out_dir outputs

Report ADE, FDE, Miss Rate, and Latency.
```

## Task 6. Implement Common Training Pipeline

```text
Read codex_vehicle_trajectory_project_plan.md.
Implement Phase 6 only.
Create the common training loop, loss functions, checkpoint saving,
logging, early stopping, and config loading.
Do not implement Transformer or Diffusion yet.

Run:
pytest -q
```

## Task 7. Implement LSTM Model

```text
Read codex_vehicle_trajectory_project_plan.md.
Implement Phase 7 only.
Create LSTMForecast with encoder-decoder structure.
Connect it to the common training pipeline.
Add shape tests.

Run:
pytest tests/test_models_shape.py -q
python -m src.training.train --config configs/lstm.yaml --max_epochs 1 \
  --data data/processed/train_smoke.npz \
  --val_data data/processed/val_smoke.npz
python -m src.evaluation.evaluate --model lstm \
  --checkpoint outputs/checkpoints/best_lstm.pt \
  --data data/processed/val_smoke.npz \
  --out_dir outputs
```

## Task 8. Implement Transformer Model

```text
Read codex_vehicle_trajectory_project_plan.md.
Implement Phase 8 only.
Create PositionalEncoding and TransformerForecast.
Connect it to training and evaluation.
Add shape tests.

Run:
pytest tests/test_models_shape.py -q
python -m src.training.train --config configs/transformer.yaml --max_epochs 1 \
  --data data/processed/train_smoke.npz \
  --val_data data/processed/val_smoke.npz
python -m src.evaluation.evaluate --model transformer \
  --checkpoint outputs/checkpoints/best_transformer.pt \
  --data data/processed/val_smoke.npz \
  --out_dir outputs
```

## Task 9. Implement Direct Diffusion

```text
Read codex_vehicle_trajectory_project_plan.md.
Implement Phase 9 only.
Create conditional 60D trajectory diffusion.
The model must encode X, flatten Y into 60D, add DDPM noise,
predict noise, and sample K future trajectories.

Add tests for one training step and sample shape.

Run:
pytest tests/test_diffusion_step.py -q
python -m src.training.train --config configs/diffusion_direct.yaml --max_epochs 1 \
  --data data/processed/train_smoke.npz \
  --val_data data/processed/val_smoke.npz
python -m src.evaluation.evaluate --model diffusion_direct \
  --checkpoint outputs/checkpoints/best_diffusion_direct.pt \
  --data data/processed/val_smoke.npz \
  --out_dir outputs
```

## Task 10. Implement PCA Latent Diffusion

```text
Read codex_vehicle_trajectory_project_plan.md.
Implement Phase 10 only.
Create PCATrajectoryCodec.
Fit PCA only on training future trajectories.
Add optional diffusion_pca model path if direct diffusion already works.

Run:
python -m src.analysis.pca_analysis \
  --train_data data/processed/train_smoke.npz \
  --out_dir outputs
python -m src.training.train --config configs/diffusion_pca.yaml --max_epochs 1 \
  --data data/processed/train_smoke.npz \
  --val_data data/processed/val_smoke.npz
```

## Task 11. Implement Visualization

```text
Read codex_vehicle_trajectory_project_plan.md.
Implement Phase 12 only.
Create plotting scripts for trajectory overlay, diffusion samples,
ADE/FDE histogram, PCA plot, and K-means clusters.
Use matplotlib only.

Run:
python -m src.visualization.plot_trajectories \
  --data data/processed/val_smoke.npz \
  --predictions outputs/predictions \
  --out_dir outputs/figures \
  --num_cases 10
```

## Task 12. Implement K-means and Error Analysis

```text
Read codex_vehicle_trajectory_project_plan.md.
Implement Phase 13 only.
Create PCA and K-means analysis over future trajectories.
Create cluster-wise metric tables for all available model predictions.

Run:
python -m src.analysis.pca_analysis \
  --data data/processed/val_smoke.npz \
  --out_dir outputs
python -m src.analysis.kmeans_analysis \
  --data data/processed/val_smoke.npz \
  --predictions outputs/predictions \
  --out_dir outputs
```

## Task 13. Implement AV2 Preprocessing

```text
Read codex_vehicle_trajectory_project_plan.md.
Implement Phase 11 only.
Inspect the available raw Argoverse 2 data directory before assuming exact filenames.
Create AV2 preprocessing script that outputs the required .npz format.

Run first on 100 scenarios:
python -m src.datasets.preprocess_av2 \
  --raw_dir data/raw/av2_motion_forecasting \
  --out_dir data/processed \
  --num_scenarios 100 \
  --target_types VEHICLE PEDESTRIAN \
  --obs_len 50 \
  --pred_len 30 \
  --target_mode focal \
  --seed 42

Then validate:
python -m src.datasets.validate_processed --npz data/processed/train_small.npz

If raw AV2 data is unavailable, report that clearly and do not fake success.
```

## Task 14. Run Final Experiments

```text
Read codex_vehicle_trajectory_project_plan.md.
Run final experiments using the available processed dataset.
Required models:
- linear
- lstm
- transformer
- diffusion_direct if trained successfully

Generate:
outputs/tables/model_comparison.csv
outputs/tables/model_comparison.md
outputs/figures/error_histogram_fde.png
outputs/figures/trajectory_overlay_best_cases.png
outputs/figures/trajectory_overlay_worst_cases.png

Report:
1. model comparison table
2. best and worst cases
3. whether LSTM beats Linear
4. whether Transformer beats LSTM
5. whether Diffusion creates useful multiple futures
```

## Task 15. Generate Final Report Summary

```text
Read codex_vehicle_trajectory_project_plan.md.
Create outputs/report_summary.md.
The report must include problem definition, dataset, preprocessing,
models, metrics, results table, trajectory visualizations, error analysis,
PCA/K-means interpretation, limitations, and future work.
Do not invent results. Use only saved metrics and figures.
```

# 6. 실행 순서 요약

가장 안전한 진행 순서는 다음과 같다.

```text
Task 0  repository setup
Task 1  synthetic smoke dataset
Task 2  coordinate transform
Task 3  Dataset/DataLoader
Task 4  metrics
Task 5  linear baseline
Task 6  common training pipeline
Task 7  LSTM
Task 8  Transformer
Task 9  direct diffusion
Task 10 PCA latent diffusion
Task 11 visualization
Task 12 K-means/error analysis
Task 13 AV2 preprocessing
Task 14 final experiments
Task 15 final report summary
```

중요한 전략은 AV2 preprocessing을 너무 먼저 하지 않는 것이다. AV2 data structure에서 막히면 전체 프로젝트가 멈출 수 있으므로, synthetic data로 model, metric, training, visualization pipeline을 먼저 완성하고 그 다음 AV2 parser를 붙이는 방식이 안정적이다. Full AV2 raw data는 Mac 용량을 쓰지 않도록 Windows에 직접 저장하고, 필요한 경우 Windows에서 raw-to-processed conversion까지 실행한다.

# 7. 구현 성공 기준

최소 발표 가능 버전:

```text
Linear + LSTM + Transformer + ADE/FDE + trajectory overlay
```

좋은 발표 버전:

```text
Linear + LSTM + Transformer + Diffusion
+ ADE/FDE/minADE/minFDE
+ diffusion samples
+ K-means cluster-wise error analysis
```

가장 좋은 발표 버전:

```text
AV2 real data
+ vehicle/pedestrian separated results
+ direct diffusion vs PCA latent diffusion
+ success/failure case study
+ map-free limitation analysis
```

프로젝트는 다음 7개가 완료되면 발표 가능한 상태로 본다.

1. Linear baseline 결과표 생성
2. LSTM 학습 및 평가 완료
3. Transformer 학습 및 평가 완료
4. Diffusion sample trajectory 생성
5. ADE/FDE/minADE/minFDE 비교표 생성
6. 예측 궤적 overlay 그림 생성
7. PCA/K-means 또는 오류 사례 분석 그림 생성

# 8. 최종 발표 스토리라인

발표는 다음 흐름으로 구성한다.

1. 자율주행차는 현재 위치뿐 아니라 주변 객체의 미래 행동을 예측해야 한다.
2. Argoverse 2 Motion Forecasting 데이터의 구조와 선택 이유를 설명한다.
3. global coordinate를 relative coordinate로 변환하는 이유를 그림으로 설명한다.
4. Linear baseline으로 단순 직선 예측의 한계를 보여준다.
5. LSTM으로 순차 데이터 학습을 설명한다.
6. Transformer로 attention이 과거 step 중 중요한 정보를 선택하는 방식을 설명한다.
7. Diffusion으로 여러 가능한 미래 후보를 생성하는 장점을 설명한다.
8. ADE/FDE 표와 trajectory overlay 그림을 함께 제시한다.
9. 회전, 정지, 보행자 등 어려운 사례를 보여준다.
10. HD map, traffic light, agent interaction, 실시간성 한계를 제시한다.

# 9. 리스크 관리

| 리스크 | 위험도 | 대응 전략 |
|---|---:|---|
| AV2 데이터 구조가 복잡함 | 높음 | 처음 100개 scenario로 parser 검증 후 확장 |
| Mac에서 Diffusion 학습이 느림 | 높음 | Windows RTX 3080에서 GPU 학습, PCA latent diffusion, 작은 subset 사용 |
| Mac 저장공간 부족 | 높음 | Full AV2 raw/processed data는 Windows에 저장하고 Mac에는 샘플/결과만 회수 |
| Transformer가 LSTM보다 성능이 안 좋을 수 있음 | 중 | 성능만 보지 말고 오류 유형과 시각화 분석 포함 |
| Diffusion sampling 결과가 불안정함 | 중상 | direct 60D와 PCA 12D를 모두 시도하고 minADE로 평가 |
| 데이터 누락/mask 처리 오류 | 높음 | 전처리 검증 script와 assert 함수 작성 |
| 시간 부족 | 높음 | Linear+LSTM+Transformer를 필수로 완료하고 Diffusion은 단순 버전으로 제한 |

# 10. Codex 보고 형식

각 작업 후 Codex는 다음 형식으로 보고해야 한다.

```text
## Phase Result

### Changed Files
- ...

### Commands Run
- ...

### Results
- ...

### Validation
- ...

### Remaining Risks
- ...

### Next Recommended Task
- ...
```

# 11. 금지 사항

Codex는 다음 행동을 하지 않는다.

1. raw AV2 data 수정
2. 결과 파일을 임의로 만들어 성공한 것처럼 보고
3. future coordinate를 input feature로 사용
4. train/validation data leakage 발생
5. 한 번에 모든 모델 구현
6. test 없이 다음 단계로 이동
7. hardcoded absolute path 사용
8. 평가 지표 정의 변경
9. Diffusion 결과를 ADE/FDE만으로 평가하고 minADE/minFDE를 누락
10. 발표용 수치를 실제 실험 없이 추정해서 작성

# 12. 최종 체크리스트

프로젝트 완료 전 확인한다.

- [ ] `pytest -q`가 통과한다.
- [ ] synthetic data에서 Linear evaluation이 동작한다.
- [ ] LSTM forward shape가 `[B, 30, 2]`이다.
- [ ] Transformer forward shape가 `[B, 30, 2]`이다.
- [ ] Diffusion sample shape가 `[B, K, 30, 2]`이다.
- [ ] ADE/FDE/minADE/minFDE가 저장된다.
- [ ] 모델별 prediction 파일이 저장된다.
- [ ] trajectory overlay figure가 생성된다.
- [ ] model comparison table이 생성된다.
- [ ] cluster-wise metric table이 생성된다.
- [ ] `outputs/report_summary.md`가 생성된다.
- [ ] 발표에서 사용할 best/worst case figure가 있다.

# 13. 최종 결론 문장 예시

본 프로젝트는 자율주행 환경에서 주변 차량과 보행자의 미래 궤적을 예측하는 문제를 다루었다. Argoverse 2 데이터셋을 이용하여 과거 5초의 움직임으로 미래 3초의 궤적을 예측하도록 문제를 정의하고, Linear baseline, LSTM, Transformer, Diffusion 모델을 동일한 조건에서 비교하였다.

LSTM은 순차 데이터의 기본 패턴을 학습하는 데 효과적이지만, 복잡한 상호작용이나 회전 구간에서는 한계가 있었다. Transformer는 attention을 통해 과거 시점의 중요도를 반영할 수 있어 일부 장면에서 개선이 기대된다. Diffusion은 여러 가능한 미래 후보를 생성할 수 있어 보행자나 갈림길 상황처럼 불확실성이 큰 장면을 설명하는 데 유리하다.

한계로는 HD map, traffic light, 주변 agent interaction을 충분히 반영하지 못했다는 점이 있다. 향후 연구에서는 lane graph, traffic signal, multi-agent joint prediction을 추가해 실제 자율주행 planning에 더 가까운 예측 모델로 확장할 수 있다.
