# Vehicle Trajectory Project GPT Q&A Reference

작성일: 2026-06-08

이 문서는 이 저장소의 코드, 설정, 테스트, 문서, 현재 lightweight 결과 파일을 바탕으로 만든 GPT 질의응답용 상세 요약이다. 이 파일 하나만 GPT에게 제공해도 프로젝트의 목적, 입력 데이터 전처리, 모델 구조, 학습/평가 절차, 결과 해석, 한계와 예상 질문에 답할 수 있도록 작성했다.

주의:

- raw AV2 데이터, processed `.npz`, checkpoint, prediction pickle, training log는 원칙적으로 Git에 올리지 않는다.
- 이 저장소의 Git 추적 대상은 코드, 설정, 테스트, 문서, 작은 표/그림, lightweight metrics/results 중심이다.
- 현재 Mac 로컬에는 분석 재현을 위한 ignored `.npz`, checkpoint, prediction payload가 일부 존재하지만, 이들은 커밋 대상이 아니다.
- 이 문서는 새 실험 결과를 만들지 않고 현재 코드와 파일 상태를 읽어 정리한 문서다.

## 1. 프로젝트 한 줄 요약

Argoverse 2 Motion Forecasting 데이터를 사용해 차량/보행자의 과거 5초 궤적을 보고 미래 3초 궤적을 예측하는 프로젝트다. Linear, LSTM, Transformer, Direct Diffusion, PCA Latent Diffusion을 같은 전처리 포맷과 같은 평가 지표에서 비교한다.

## 2. 문제 정의

목표:

- 입력: 타깃 agent의 과거 5초 움직임.
- 출력: 같은 agent의 미래 3초 `(x, y)` 궤적.
- sampling: 10Hz.
- observation length: 50 step.
- prediction length: 30 step.
- 좌표계: 마지막 관측 시점 기준 상대좌표, 마지막 heading 기준 정렬.
- 주요 target: AV2 focal track.
- 확장 target: vehicle + pedestrian.

이 프로젝트는 leaderboard SOTA 재현이 목적이 아니다. 데이터 전처리, 학습, 평가, 시각화, 오류 분석을 end-to-end로 직접 구현하고, 모델별 장단점을 공정하게 비교하는 포트폴리오/수업 프로젝트다.

## 3. 저장소 역할 분리

Mac 역할:

- source of truth.
- 코드, 문서, 테스트, synthetic/sample 검증.
- 평가, 시각화, 분석, 리포트 작성.
- Git commit/push.

Windows HOME 역할:

- AV2 raw data 저장.
- Windows-local AV2 preprocessing.
- CUDA GPU 학습.
- 표/metric/checkpoint/log 생성 후 lightweight 결과를 Mac으로 회수.

검증된 Windows 정보:

- primary host: `thddy@192.168.35.17`.
- expected SSH check: `HOME` / `home\thddy`.
- CUDA 환경: `vehicle_traj` conda environment, CUDA PyTorch, NVIDIA GeForce RTX 2070 SUPER.
- raw AV2 path: `D:\data\av2\motion-forecasting`.
- processed path: `D:\data\vehicle_trajectory_project\processed`.
- run path: `D:\runs\vehicle_trajectory_project`.

## 4. 핵심 디렉터리와 파일

프로젝트 루트:

- `README.md`: 프로젝트 개요와 final result 요약.
- `PROJECT_STATUS.md`: phase 진행 상태, Windows/AV2/실험 provenance ledger.
- `GOAL_RUNBOOK.md`: 장기 goal 진행 규칙.
- `AGENTS.md`: Codex 작업 규칙.
- `requirements.txt`, `pyproject.toml`: Python 의존성.
- `Makefile`: `pytest -q`, device check, clean.

코드:

- `src/datasets/`: synthetic data, AV2 preprocessing, processed dataset loader, schema validation.
- `src/utils/`: geometry, seed, paths, JSON/YAML, device, logging.
- `src/models/`: Linear baseline, LSTM, Transformer, Direct Diffusion, PCA Latent Diffusion.
- `src/training/`: loss functions, common trainer, train CLI.
- `src/evaluation/`: ADE/FDE/minADE/minFDE/miss rate/latency/params evaluation.
- `src/analysis/`: PCA, K-means, error analysis.
- `src/visualization/`: trajectory overlay, error histogram, PCA plot, cluster plot, report-case figures.
- `scripts/`: all-model evaluation, diffusion tuning selection, Phase 15 report generation, Windows pilot/long-run orchestration.
- `configs/`: preprocessing, model, pilot, tuning, long-run YAML configs.
- `tests/`: unit/smoke tests for shape contracts, metrics, preprocessing, training, analysis, visualization, report building.

결과:

- `outputs/report_summary.md`: Phase 15 final report summary.
- `outputs/full_av2/tables/model_comparison.csv`: final full AV2 comparison table.
- `outputs/full_av2/metrics/`: final lightweight metrics JSON.
- `outputs/full_av2/figures/`: final visualization figures.
- `outputs/full_av2_analysis/tables/`: PCA/K-means/error analysis tables.
- `outputs/full_av2_tuning/tables/`: diffusion tuning summary and selected configs.

Ignored/local-only examples:

- `data/processed/*.npz`.
- `outputs/checkpoints/*`.
- `outputs/predictions/*`.
- `outputs/logs/*`.

## 5. 입력 데이터 전처리 전체 흐름

전처리 코드는 `src/datasets/preprocess_av2.py`가 담당한다.

### 5.1 raw AV2 입력 위치와 파일 구조

Windows raw AV2 standard path:

```text
D:\data\av2\motion-forecasting
```

예상 split 구조:

```text
D:\data\av2\motion-forecasting\train\<scenario_id>\scenario_<scenario_id>.parquet
D:\data\av2\motion-forecasting\val\<scenario_id>\scenario_<scenario_id>.parquet
D:\data\av2\motion-forecasting\test\<scenario_id>\scenario_<scenario_id>.parquet
```

Phase 11 이상 real AV2 작업 전 ready marker:

```text
D:\data\av2\motion-forecasting\DATA_READY_FOR_PHASE11.txt
```

### 5.2 parquet에서 요구하는 컬럼

전처리 코드는 각 scenario parquet에 다음 컬럼이 있어야 한다고 검사한다.

```text
track_id
object_type
timestep
position_x
position_y
heading
velocity_x
velocity_y
scenario_id
focal_track_id
observed
```

`object_category`는 필수 컬럼은 아니지만 `target_mode=scored`일 때 scored track 필터링에 사용된다.

### 5.3 전처리 설정

`PreprocessConfig` 기본값:

- `num_scenarios=100`: small preprocessing이면 split별 앞 100개 scenario만 처리.
- `num_scenarios=None`: full preprocessing이면 전체 scenario 처리.
- `target_types=("VEHICLE", "PEDESTRIAN")`.
- `obs_len=50`.
- `pred_len=30`.
- `target_mode="focal"`.
- `splits=("train", "val")`.
- `max_read_errors=1000`.

`configs/preprocess_small.yaml`:

- raw: `D:/data/av2/motion-forecasting`.
- out: `D:/data/vehicle_trajectory_project/processed/small`.
- split: train/val.
- target mode: focal.
- target types: VEHICLE, PEDESTRIAN.
- output: `train_small.npz`, `val_small.npz`.

Full preprocessing command pattern:

```powershell
python -m src.datasets.preprocess_av2 `
  --raw_dir D:\data\av2\motion-forecasting `
  --out_dir D:\data\vehicle_trajectory_project\processed\full `
  --full `
  --splits train val `
  --target_types VEHICLE PEDESTRIAN `
  --target_mode focal
```

### 5.4 target track 선택 방식

전처리의 target selection 함수는 `_candidate_track_ids`다.

지원 mode:

- `focal`: parquet의 `focal_track_id` 하나만 target으로 사용한다.
- `scored`: `object_type`이 target type에 포함되고 `object_category >= 2`인 track을 사용한다.
- `all_supported`: 지원 object type 전체 track을 후보로 사용한다.

현재 small/full AV2 결과는 `target_mode=focal`, `target_types=VEHICLE PEDESTRIAN` 설정이다. 즉 scenario마다 focal track이 vehicle 또는 pedestrian이면 sample이 만들어지고, 지원 object type이 아니면 제외된다.

### 5.5 한 track sample을 만드는 세부 과정

함수: `_build_track_sample`.

입력:

- 한 scenario dataframe.
- target `track_id`.
- target type set.
- `obs_len=50`.
- `pred_len=30`.

처리 순서:

1. 해당 `track_id` row만 추출하고 `timestep`으로 정렬한다.
2. `object_type`을 lower-case로 정규화한다.
3. object type이 `vehicle` 또는 `pedestrian`이 아니면 제외한다.
4. 전체 길이 `total_len = 50 + 30 = 80`을 만든다.
5. `positions[80,2]`, `velocities[80,2]`, `headings[80]`, `mask[80]`를 0으로 초기화한다.
6. timestep 0부터 79까지 순회한다.
7. 해당 timestep row가 있으면 position, velocity, heading을 채우고 mask를 True로 둔다.
8. `observed` 값이 `t < obs_len` 조건과 일치하지 않으면 sample을 버린다.
9. 마지막 관측 timestep인 49가 존재하지 않으면 sample을 버린다.
10. 관측 구간에 유효 timestep이 하나도 없으면 sample을 버린다.
11. 미래 구간 30 step이 모두 유효하지 않으면 sample을 버린다.
12. 마지막 관측 위치와 heading으로 상대좌표 변환을 한다.
13. model input `X[50,6]`, target `Y[30,2]`를 만든다.
14. masked 관측/미래 값은 0으로 채운다.

중요한 sample rejection 조건:

- target track이 비어 있음.
- object type 미지원.
- `observed` flag가 과거/미래 구간과 불일치.
- 마지막 관측 시점이 missing.
- 미래 30 step 중 하나라도 missing.
- split 전체에서 valid sample이 0개.

### 5.6 좌표 변환 방식

좌표 변환은 `src/utils/geometry.py`의 `to_relative_coords`와 `rotation_matrix`를 사용한다.

정의:

```text
origin = positions[obs_len - 1]
theta = heading[obs_len - 1]
rel_position = R(-theta) @ (position_global - origin)
global_position = R(theta) @ rel_position + origin
```

코드상 행벡터 연산:

```text
rel_positions = (positions - origin) @ rotation_matrix(-theta).T
```

검증되는 성질:

- 마지막 관측 위치 `X[:, -1, 0:2]`는 거의 `(0, 0)`이어야 한다.
- inverse transform을 적용하면 global coordinate를 복원해야 한다.
- 좌표 단위는 meter를 유지한다.
- heading은 `wrap_angle(headings - theta)`로 `[-pi, pi)` 범위로 정규화한다.

### 5.7 velocity와 heading feature

AV2 parquet의 global velocity도 마지막 heading 기준 frame으로 회전한다.

```text
rel_velocities = velocities @ rotation_matrix(-theta).T
```

heading은 직접 angle 값을 넣지 않고 sine/cosine으로 넣는다.

```text
rel_heading = wrap_angle(headings - theta)
sin_heading = sin(rel_heading)
cos_heading = cos(rel_heading)
```

angle을 sin/cos로 쓰는 이유:

- heading의 주기성 문제를 줄인다.
- `pi`와 `-pi`가 실제로는 가깝지만 수치상 멀어지는 discontinuity를 피한다.

### 5.8 최종 model input X feature 순서

전처리 후 `X` shape:

```text
X: float32 [N, 50, 6]
```

feature order:

```text
X[..., 0] = rel_x
X[..., 1] = rel_y
X[..., 2] = velocity_x
X[..., 3] = velocity_y
X[..., 4] = sin_heading
X[..., 5] = cos_heading
```

`Y` shape:

```text
Y: float32 [N, 30, 2]
Y[..., 0] = future rel_x
Y[..., 1] = future rel_y
```

### 5.9 processed `.npz` 공통 schema

모든 synthetic/AV2 processed 파일은 다음 키를 사용한다.

```text
X: float32 [N, 50, 6]
Y: float32 [N, 30, 2]
mask_x: bool [N, 50]
mask_y: bool [N, 30]
object_type: int64 [N]
scenario_id: object/string [N]
track_id: object/string [N]
origin: float32 [N, 2]
theta: float32 [N]
```

Synthetic data에는 추가로:

```text
pattern: object/string [N]
```

object type mapping:

```text
vehicle -> 0
pedestrian -> 1
```

### 5.10 scaler 저장 방식

전처리 시 train split에 대해서만 scaler를 fit한다.

`_save_scaler`:

- `X[mask_x]`처럼 관측 mask가 True인 timestep feature만 사용한다.
- mean/std를 계산한다.
- std는 최소 `1e-6`으로 clamp한다.
- 저장 위치: `<out_dir>/scaler.pkl`.
- feature order도 함께 저장한다.

현재 모델 학습 코드가 이 scaler를 직접 적용해 normalization을 수행하지는 않는다. 다만 train-only fit 원칙과 feature statistics provenance를 남긴다.

### 5.11 full AV2 preprocessing 결과

`PROJECT_STATUS.md`와 현재 local ignored `.npz` shape 기준:

- full preprocessing 완료 시각: 2026-06-03 23:57 KST.
- Windows HOME output:
  - `D:\data\vehicle_trajectory_project\processed\full\train_full.npz`
  - `D:\data\vehicle_trajectory_project\processed\full\val_full.npz`
- validation passed:
  - train_full: 189,541 samples.
  - val_full: 23,706 samples.
- unreadable train parquet 25개를 skip했다.
- skip은 `PermissionError`, `OSError`, PyArrow read error 등 read error types에 한정된다.
- `max_read_errors` 기본값은 split당 1000개라 너무 많은 파일이 조용히 누락되는 것을 막는다.

현재 Mac local data shape:

```text
train_full.npz: X=(189541,50,6), Y=(189541,30,2)
val_full.npz:   X=(23706,50,6),  Y=(23706,30,2)
train_small.npz / val_small.npz: each 95 samples
train_smoke.npz: 700 samples
val_smoke.npz: 150 samples
test_smoke.npz: 150 samples
```

### 5.12 schema validation

검증 코드는 `src/datasets/validate_processed.py`다.

검사 항목:

- required keys 존재.
- shape 일치:
  - `X`: `[N, 50, 6]`
  - `Y`: `[N, 30, 2]`
  - `mask_x`: `[N, 50]`
  - `mask_y`: `[N, 30]`
  - `origin`: `[N, 2]`
  - `theta`: `[N]`
- dtype 일치:
  - `X`, `Y`, `origin`, `theta`: float32
  - `mask_x`, `mask_y`: bool
  - `object_type`: int64
- sample count가 0이면 fail.
- NaN/Inf 없음.
- `X[:, -1, 0:2]`가 0에 가까워야 함.
- `object_type`은 0 또는 1만 허용.
- `mask_x[:, -1]`는 모든 sample에서 True여야 함.
- `mask_y`는 모든 미래 timestep이 True여야 함.
- masked X/Y payload는 0이어야 함.

검증 명령 예:

```bash
python -m src.datasets.validate_processed --npz data/processed/val_full.npz
```

## 6. Synthetic smoke data

코드: `src/datasets/synthetic.py`.

목적:

- AV2 raw data가 없어도 전체 pipeline을 테스트한다.
- geometry, loader, model, loss, metrics, visualization을 빠르게 검증한다.

패턴:

- `straight`
- `slowdown`
- `left_turn`
- `right_turn`
- `stop_and_go`
- `pedestrian_random`

split:

- 기본 `num_samples=1000`.
- train 70%, val 15%, test 15%.
- 현재 local:
  - train 700.
  - val 150.
  - test 150.

Synthetic도 마지막 관측 시점 기준 상대좌표를 사용하며, 마지막 observed position이 원점이 되도록 만든다.

생성 명령:

```bash
python -m src.datasets.synthetic --out_dir data/processed --num_samples 1000
```

## 7. Dataset/DataLoader

코드: `src/datasets/av2_dataset.py`.

`TrajectoryDataset`:

- `.npz` 파일을 `allow_pickle=True`로 로드한다.
- required keys가 없으면 `KeyError`.
- 모든 key의 첫 dimension이 `N`과 같아야 한다.
- `__getitem__`은 tensor와 string metadata를 반환한다.

반환 batch:

```text
X: torch.float32 [B,50,6]
Y: torch.float32 [B,30,2]
mask_x: torch.bool [B,50]
mask_y: torch.bool [B,30]
object_type: torch.long [B]
scenario_id: list[str]
track_id: list[str]
origin: torch.float32 [B,2]
theta: torch.float32 [B]
```

`create_dataloader`:

- batch size, shuffle, num_workers를 받아 PyTorch DataLoader 생성.

## 8. 모델 구조

### 8.1 Linear Extrapolation

코드: `src/models/linear.py`.

방식:

- 마지막 관측 위치 `p_last = X[:, -1, 0:2]`.
- 마지막 관측 velocity `v_last = X[:, -1, 2:4]`.
- `dt=0.1`.
- 미래 step `1..30`에 대해:

```text
pred[t] = p_last + v_last * (t * dt)
```

특징:

- 학습 parameter 없음.
- constant velocity baseline.
- 최종 결과에서 parameters = 0.

### 8.2 LSTM Encoder-Decoder

코드: `src/models/lstm.py`.

구조:

- encoder LSTM: input `[B,50,6]`.
- decoder LSTM: input은 2D 좌표.
- decoder 첫 입력은 마지막 관측 위치 `X[:, -1, 0:2]`, 즉 상대좌표 원점.
- 30 step을 autoregressive하게 예측한다.
- output linear layer가 hidden state를 `(x,y)`로 변환한다.
- teacher forcing 지원:
  - training 중 `teacher_y`가 있고 ratio가 0보다 크면 일부 step에 ground truth를 decoder input으로 사용 가능.
  - 현재 일반 configs에서는 teacher forcing ratio를 명시적으로 사용하지 않는다.

대표 full long config:

- `configs/full_long_lstm.yaml`.
- hidden_dim 128.
- num_layers 2.
- dropout 0.1.
- epochs 30.
- batch_size 64.
- device cuda.

### 8.3 Transformer Encoder

코드: `src/models/transformer.py`.

구조:

- input projection: 6D feature를 `d_model`로 projection.
- sinusoidal positional encoding.
- TransformerEncoder.
- 마지막 encoded token `encoded[:, -1, :]`를 pooling으로 사용.
- MLP가 `pred_len * 2`를 직접 출력하고 `[B,30,2]`로 reshape.

대표 full long config:

- `configs/full_long_transformer.yaml`.
- d_model 128.
- nhead 4.
- num_layers 3.
- dim_feedforward 256.
- epochs 30.
- batch_size 32.
- device cuda.

### 8.4 Direct Diffusion

코드: `src/models/diffusion.py`.

핵심 구성:

- `TrajectoryConditionEncoder`: observed trajectory `X`를 GRU로 encode해 condition vector 생성.
- `SinusoidalTimeEmbedding`: diffusion timestep embedding.
- `DiffusionDenoiser`: noisy flattened future trajectory와 condition, timestep embedding을 받아 noise를 예측하는 MLP.
- `GaussianDiffusionTrajectory`: future trajectory를 60D vector `[30*2]`로 flatten해 Gaussian diffusion 수행.

학습:

1. `Y`를 `[B,60]`으로 flatten.
2. random noise 생성.
3. random timestep `t` sample.
4. forward diffusion:

```text
x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise
```

5. denoiser가 `noise_hat` 예측.
6. loss: MSE(noise_hat, noise).

샘플링:

- `reverse_diffusion_timesteps(diffusion_steps, sampling_steps)`로 skipped-step reverse schedule 생성.
- 예: diffusion_steps=10, sampling_steps=4이면 `[9,6,3,0]`.
- 각 step에서 predicted noise로 `x0`를 추정하고 다음 sampled timestep으로 이동한다.
- output shape:

```text
samples: [B, K, 30, 2]
forward(): samples[:,0] -> [B,30,2]
```

최종 long run은 K=16 sample evaluation을 사용한다.

### 8.5 PCA Latent Diffusion

코드: `src/models/pca_latent.py`.

목적:

- 60D future trajectory를 train-set PCA latent로 압축한 뒤 latent space에서 diffusion을 수행한다.
- direct 60D diffusion보다 더 낮은 차원에서 생성하므로 학습이 쉬울 수 있다.

`PCATrajectoryCodec`:

- train split의 `Y_train [N,30,2]`만 사용해 PCA fit.
- `Y_train.reshape(N,60)`에 PCA 적용.
- latent mean/std도 train latent로 계산한다.
- 기본 `normalize_latent=True`.
- `transform`: PCA latent로 변환 후 train-fitted mean/std로 normalize.
- `inverse_transform`: normalize 해제 후 PCA inverse로 `[N,30,2]` 복원.

데이터 누수 방지:

- PCA는 validation/test가 아니라 train future trajectory로만 fit한다.
- final 분석에서도 `pca_analysis`는 `--train_data train_full.npz`로 fit하고 `--data val_full.npz`는 transform 대상이다.

`PCALatentDiffusionTrajectory`:

- checkpoint/config의 `codec_path`에서 train-fitted codec load.
- `encode_y`: torch에서 PCA projection과 latent normalization 수행.
- `decode_z`: latent를 trajectory로 복원.
- diffusion은 latent dimension에서 수행.

대표 final tuning selection:

- selected PCA Diffusion: `pca_b`.
- latent_dim 12.
- diffusion_steps 1000.
- sampling_steps 75.
- num_samples 16.
- learning_rate 0.0002.

## 9. Loss functions

코드: `src/training/losses.py`.

지원 loss:

- `trajectory_mse_loss`: step별 x/y MSE 평균 후 mask-aware 평균.
- `trajectory_smooth_l1_loss`: Smooth L1 loss를 mask-aware 평균.
- `endpoint_loss`: final valid timestep의 endpoint MSE.
- `combined_trajectory_loss`: smooth L1 + endpoint_weight * endpoint loss.

Diffusion 모델은 model 자체의 `training_loss(X,Y)`를 사용하므로 `noise_mse` 개념으로 denoising noise MSE를 최적화한다.

## 10. Training pipeline

코드:

- `src/training/train.py`.
- `src/training/trainer.py`.

### 10.1 train CLI

명령 예:

```bash
python -m src.training.train \
  --config configs/lstm.yaml \
  --data data/processed/train_smoke.npz \
  --val_data data/processed/val_smoke.npz
```

처리:

1. YAML config load.
2. seed 설정.
3. device resolve:
   - `auto`: CUDA, MPS, CPU 순서.
   - 명시 config: `cpu` 또는 `cuda`.
4. train dataset 첫 sample로 `input_dim`, `obs_len` 확인.
5. architecture에 따라 model build.
6. train/val DataLoader 생성.
7. Trainer 실행.

지원 architecture:

- `tiny_regressor`
- `lstm`
- `transformer`
- `diffusion_direct`
- `diffusion_pca`

### 10.2 common Trainer

Trainer 특징:

- AdamW optimizer.
- gradient clipping.
- early stopping.
- checkpoint 저장:
  - `outputs/checkpoints/best_<model_name>.pt`
  - `outputs/checkpoints/last_<model_name>.pt`
- log 저장:
  - `outputs/logs/<model_name>_train_log.csv`
- metrics 저장:
  - `outputs/metrics/<model_name>_val_metrics.json`

checkpoint에는 다음이 들어간다.

```text
epoch
model_state_dict
optimizer_state_dict
metrics
model_name
trainer_config
metadata
```

metadata에는 다음 provenance가 들어간다.

```text
config_path
train_data
val_data
seed
device
model config
training config
```

### 10.3 diffusion validation

일반 deterministic 모델:

- `pred = model(X)`.
- ADE/FDE 계산.

Diffusion 모델:

- training loss는 denoising loss.
- validation에서 `validation_num_samples > 1`이면 `model.sample`로 multi-sample 생성.
- 첫 sample로 ADE/FDE 계산.
- 전체 sample set으로 minADE/minFDE/Sample Diversity 계산.
- `validation_seed`가 있으면 validation sampling을 deterministic하게 만든다.

## 11. Evaluation pipeline

코드:

- `src/evaluation/metrics.py`.
- `src/evaluation/evaluate.py`.
- `scripts/run_all_evaluations.py`.

### 11.1 metrics

단일 예측:

```text
pred: [B,T,2]
gt: [B,T,2]
mask: [B,T]
```

ADE:

- valid timestep displacement error 평균.

FDE:

- 각 sample의 final valid timestep displacement error 평균.

Miss Rate:

- final displacement error가 threshold보다 큰 sample 비율.
- 기본 threshold는 2.0m.

Multi-sample:

```text
pred_samples: [B,K,T,2]
gt: [B,T,2]
```

minADE:

- 각 sample item마다 K개 prediction 중 ADE가 가장 낮은 sample 선택.
- batch 평균.

minFDE:

- 각 item마다 K개 prediction 중 final error가 가장 낮은 sample 선택.
- batch 평균.

Sample Diversity:

- K개 sample 사이의 평균 pairwise displacement.
- 0이면 모든 sample이 동일하다는 뜻.
- 너무 낮으면 diffusion collapse 의심.

Parameter count:

- trainable parameter 수만 계산.

### 11.2 evaluate CLI

Linear:

```bash
python -m src.evaluation.evaluate \
  --model linear \
  --data data/processed/val_smoke.npz \
  --config configs/linear.yaml \
  --out_dir outputs
```

Trainable model:

```bash
python -m src.evaluation.evaluate \
  --model lstm \
  --checkpoint outputs/checkpoints/best_lstm.pt \
  --data data/processed/val_smoke.npz \
  --out_dir outputs
```

Output:

- metrics JSON.
- metrics CSV.
- prediction pickle.

Prediction pickle structure:

Deterministic:

```text
pred: [N,30,2]
gt: [N,30,2]
mask_y: [N,30]
scenario_id: list[str]
track_id: list[str]
```

Diffusion:

```text
samples: [N,K,30,2]
pred: [N,30,2]  # first sample
gt: [N,30,2]
mask_y: [N,30]
scenario_id: list[str]
track_id: list[str]
```

### 11.3 all-model evaluation

`scripts/run_all_evaluations.py`:

- model order: linear, lstm, transformer, diffusion_direct, diffusion_pca.
- explicit checkpoint required for trainable models.
- `--checkpoint_dir` + `--checkpoint_tag`로 checkpoint naming 자동 resolve.
- missing checkpoint는 기본적으로 fail.
- `--allow_missing_models`를 쓰면 missing model skip 가능.
- `--prediction_tag`가 있으면 prediction payload를 subdirectory로 archive.
- `tables/model_comparison.csv`와 `.md` 생성.

Single prediction 모델은 table에서:

```text
minADE = ADE
minFDE = FDE
```

Diffusion 모델은:

```text
Notes = multi-sample evaluation; K=...
```

## 12. Analysis and visualization

### 12.1 PCA analysis

코드: `src/analysis/pca_analysis.py`.

절차:

1. `train_data`에서 full-horizon `Y`만 가져온다.
2. train future trajectory로 PCA codec fit.
3. codec 저장: `<out_dir>/checkpoints/pca_codec.pkl`.
4. explained variance figure 저장.
5. `data` split을 train-fitted PCA로 transform.
6. `pca_latent.csv` 저장.
7. `pca_trajectory_space.png` 생성.

### 12.2 K-means analysis

코드: `src/analysis/kmeans_analysis.py`.

절차:

1. train future trajectories로 PCA fit.
2. train latent로 KMeans fit.
3. target split future trajectories를 같은 PCA로 transform.
4. KMeans label 예측.
5. cluster별 count, mean final point, vehicle/pedestrian count 저장.
6. prediction payload가 있으면 cluster별 model metrics 계산.
7. `required_models`가 빠져 있으면 fail.

최종 full AV2 cluster summary:

```text
cluster 0: 4302 samples, mean_final_x 33.70, vehicle 4302, pedestrian 0
cluster 1: 5479 samples, mean_final_x 13.41, vehicle 5463, pedestrian 16
cluster 2: 6806 samples, mean_final_x 3.13,  vehicle 5253, pedestrian 1553
cluster 3: 5291 samples, mean_final_x 23.32, vehicle 5288, pedestrian 3
cluster 4: 1828 samples, mean_final_x 48.04, vehicle 1828, pedestrian 0
```

해석:

- cluster 2는 pedestrian 비율이 가장 높고 짧은 이동 거리/느린 motion을 많이 포함한다.
- cluster 4는 final x가 가장 길어 빠르게 직진하는 차량 궤적이 많다.

### 12.3 Error analysis

코드: `src/analysis/error_analysis.py`.

출력:

- `error_summary.csv`: model별 ADE/FDE/minADE/minFDE/Miss Rate.
- `top_error_cases.csv`: model별 FDE worst top-k sample.

분석 코드의 integrity check:

- prediction shape가 data `Y`와 다르면 fail.
- payload의 `gt`가 현재 data `Y`와 다르면 fail.
- payload의 `mask_y`가 현재 data `mask_y`와 다르면 fail.
- payload `scenario_id`, `track_id` order가 현재 data와 다르면 fail.
- required model prediction이 없으면 fail.

### 12.4 Visualization

코드:

- `plot_trajectories.py`: past/future/prediction overlay, diffusion sample overlay, top error cases.
- `plot_errors.py`: ADE/FDE histogram.
- `plot_pca.py`: future trajectory PCA scatter.
- `plot_clusters.py`: PCA + K-means cluster plot.
- `plot_report_cases.py`: best/worst cases and interesting diffusion case.

최종 주요 figures:

- `outputs/full_av2/figures/trajectory_overlay_best_cases.png`
- `outputs/full_av2/figures/trajectory_overlay_worst_cases.png`
- `outputs/full_av2/figures/diffusion_samples_interesting_case.png`
- `outputs/full_av2/figures/trajectory_overlay_linear_lstm_transformer.png`
- `outputs/full_av2/figures/trajectory_overlay_diffusion_samples.png`
- `outputs/full_av2/figures/error_histogram_ade.png`
- `outputs/full_av2/figures/error_histogram_fde.png`
- `outputs/full_av2/figures/pca_trajectory_space.png`
- `outputs/full_av2/figures/kmeans_clusters.png`

## 13. Full AV2 staged workflow

문서:

- `docs/full_av2_training_staged_workflow.md`.
- `scripts/windows_full_pilot_1epoch.ps1`.
- `scripts/windows_full_pilot_5epoch.ps1`.
- `scripts/windows_full_long_experiments.ps1`.

Stage:

1. F1 full preprocessing.
2. F2 schema validation.
3. F3 1-epoch pilot.
4. F4 5-epoch pilot.
5. F5A diffusion tuning gate.
6. F5B report-ready all-model long run.
7. Phase 15 report assets on Mac.

이 프로젝트는 바로 30-50 epoch long run으로 뛰지 않고, full preprocessing과 pilot gate를 통과한 뒤 long run을 진행했다.

F3/F4 pilot 목적:

- full dataset이 CUDA 학습에서 shape/runtime 문제 없이 작동하는지 확인.
- train/val loss와 finite metric/checkpoint 확인.
- LSTM/Transformer가 Linear 대비 개선되는지 확인.
- Diffusion은 5epoch pilot에서 약했기 때문에 F5A tuning gate를 별도로 수행했다.

F5A diffusion tuning gate:

- user-selected target:
  - PCA Diffusion: minADE < 4.8, minFDE < 9.5.
  - Direct Diffusion: minADE < 8.0, minFDE < 15.0.
- hard gate:
  - checkpoint 존재.
  - train/eval metrics 존재.
  - minimum epochs.
  - finite metrics.
  - sample diversity minimum.
- selected:
  - PCA Diffusion `pca_b`: minADE 0.45693, minFDE 0.94410.
  - Direct Diffusion `direct_f`: minADE 0.72301, minFDE 1.45575.
- both target gates passed.
- `full_run_ready=true`.

F5B final long run:

- Windows HOME.
- repo commit: `1e511e3`.
- completed: 2026-06-04 06:48:13 KST.
- status: complete.
- all required final artifacts present.
- complete marker exists.

## 14. Final full AV2 결과

데이터:

- split: `val_full`.
- target type label: `av2_focal_mixed`.
- sample count: 23,706.
- prediction horizon: 30 steps / 3 seconds.
- metrics are in meters.

Final comparison:

| Model | ADE | FDE | minADE | minFDE | Sample Diversity | Miss Rate | Params | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Linear | 1.53242 | 3.79734 | 1.53242 | 3.79734 | | 0.59630 | 0 | single prediction |
| LSTM | 0.79166 | 2.05703 | 0.79166 | 2.05703 | | 0.38163 | 401,666 | single prediction |
| Transformer | 0.85148 | 2.18351 | 0.85148 | 2.18351 | | 0.40007 | 406,332 | single prediction |
| PCA Diffusion | 1.43391 | 3.66399 | 0.41746 | 0.85825 | 1.54089 | 0.62415 | 222,988 | K=16 |
| Direct Diffusion | 1.92816 | 4.91142 | 0.50093 | 0.98452 | 2.31048 | 0.72674 | 531,132 | K=16 |

핵심 해석:

- 단일 trajectory ADE/FDE 기준 best model은 LSTM이다.
- Transformer도 Linear보다 훨씬 좋지만 LSTM보다 약간 뒤진다.
- PCA Diffusion과 Direct Diffusion은 첫 sample ADE/FDE 기준으로는 LSTM/Transformer보다 약하다.
- 그러나 diffusion은 K=16 generated futures 중 best-of-K를 보는 minADE/minFDE에서 매우 낮다.
- PCA Diffusion이 best-of-K minADE/minFDE 기준으로 가장 좋다.
- Diffusion min metrics는 deterministic ADE/FDE와 같은 방식으로 직접 비교하면 안 된다. "여러 후보 중 좋은 후보를 생성할 수 있는가"를 보여주는 별도 지표다.

최종 report:

- `outputs/report_summary.md`.
- `outputs/full_av2/tables/model_comparison.md`.
- `outputs/full_av2_analysis/tables/error_summary.csv`.
- `outputs/full_av2_analysis/tables/cluster_metrics.csv`.

## 15. 테스트 커버리지

대표 테스트:

- `tests/test_geometry.py`: rotation, relative/global inverse, angle wrapping.
- `tests/test_synthetic_data.py`: synthetic schema, reproducibility, DataLoader batch.
- `tests/test_preprocess_av2.py`: mock AV2 preprocessing, YAML config, read-error skipping, invalid mask rejection, schema validation.
- `tests/test_metrics.py`: ADE/FDE/minADE/minFDE/sample diversity/miss rate, invalid mask handling.
- `tests/test_losses.py`: MSE/SmoothL1/endpoint/combined loss.
- `tests/test_models_shape.py`: LSTM/Transformer output shape.
- `tests/test_diffusion_step.py`: diffusion embedding, denoiser, training loss, sample shape, skipped timesteps.
- `tests/test_pca_latent.py`: PCA codec fit/transform/inverse/save/load, latent normalization, PCA diffusion shape.
- `tests/test_trainer_diffusion_validation.py`: diffusion validation min metrics and diversity recording.
- `tests/test_phase14_experiment_matrix.py`: all-model evaluation table, checkpoint resolution, prediction archive, PCA codec fallback.
- `tests/test_analysis_phase13.py`: PCA/KMeans/error outputs, prediction alignment rejection, required model checks.
- `tests/test_visualization.py`: PNG generation for all visualization scripts.
- `tests/test_full_pilot_configs.py`: Windows CUDA config structure and long-run script resumability.
- `tests/test_phase15_report.py`: report generation from comparison/tuning tables.

전체 테스트 명령:

```bash
pytest -q
```

## 16. 주요 명령 모음

Install:

```bash
python -m pip install -r requirements.txt
```

Device check:

```bash
make device
```

Generate synthetic:

```bash
python -m src.datasets.synthetic --out_dir data/processed --num_samples 1000
```

Validate processed data:

```bash
python -m src.datasets.validate_processed --npz data/processed/val_smoke.npz
```

Train LSTM smoke:

```bash
python -m src.training.train \
  --config configs/lstm.yaml \
  --max_epochs 1 \
  --data data/processed/train_smoke.npz \
  --val_data data/processed/val_smoke.npz
```

Evaluate Linear:

```bash
python -m src.evaluation.evaluate \
  --model linear \
  --data data/processed/val_smoke.npz \
  --config configs/linear.yaml \
  --out_dir outputs
```

Run all evaluations:

```bash
python scripts/run_all_evaluations.py \
  --data data/processed/val_full.npz \
  --out_dir outputs/full_av2 \
  --models linear lstm transformer diffusion_direct diffusion_pca \
  --checkpoint_dir outputs/checkpoints \
  --checkpoint_tag full_long \
  --batch_size 64 \
  --data_split val_full \
  --target_type av2_focal_mixed \
  --prediction_tag full_long_final
```

Analysis:

```bash
python -m src.analysis.pca_analysis \
  --train_data data/processed/train_full.npz \
  --data data/processed/val_full.npz \
  --out_dir outputs/full_av2_analysis \
  --n_components 12

python -m src.analysis.kmeans_analysis \
  --train_data data/processed/train_full.npz \
  --data data/processed/val_full.npz \
  --predictions outputs/predictions/full_long_final \
  --out_dir outputs/full_av2_analysis \
  --required_models linear lstm transformer diffusion_pca diffusion_direct

python -m src.analysis.error_analysis \
  --data data/processed/val_full.npz \
  --predictions outputs/predictions/full_long_final \
  --out_dir outputs/full_av2_analysis \
  --top_k 20 \
  --required_models linear lstm transformer diffusion_pca diffusion_direct
```

Visualization:

```bash
python -m src.visualization.plot_report_cases \
  --data data/processed/val_full.npz \
  --predictions outputs/predictions/full_long_final \
  --out_dir outputs/full_av2/figures \
  --reference_model lstm \
  --diffusion_model diffusion_pca \
  --num_cases 6
```

Build report summary:

```bash
python scripts/build_phase15_report.py \
  --comparison outputs/full_av2/tables/model_comparison.csv \
  --tuning_summary outputs/full_av2_tuning/tables/diffusion_tuning_summary.csv \
  --out outputs/report_summary.md \
  --analysis_dir outputs/full_av2_analysis \
  --figures_dir outputs/full_av2/figures
```

Windows SSH preflight:

```bash
ssh thddy@192.168.35.17 'hostname && whoami'
```

Expected:

```text
HOME
home\thddy
```

## 17. 자주 나올 수 있는 질문과 답변

### Q1. 입력 데이터를 구체적으로 어떻게 전처리했나?

AV2 scenario parquet에서 focal track을 선택하고, timestep 0-49를 과거 관측, timestep 50-79를 미래 target으로 사용했다. 각 timestep의 global position, velocity, heading을 가져온 뒤 마지막 관측 시점의 position을 origin, 마지막 heading을 theta로 삼아 전체 80-step 좌표를 상대좌표로 변환했다. model input은 과거 50 step의 `[rel_x, rel_y, velocity_x, velocity_y, sin_heading, cos_heading]`이고, output target은 미래 30 step의 relative `(x,y)`다. 마지막 관측 위치가 원점이 되도록 검증하고, 미래 30 step이 모두 존재하지 않는 track은 제외했다.

### Q2. 왜 마지막 관측 시점 기준 상대좌표를 사용했나?

global map 좌표를 그대로 쓰면 scene 위치와 방향에 따라 같은 motion pattern도 전혀 다른 숫자로 보인다. 마지막 관측 위치를 원점으로 만들고 heading을 정렬하면 모델이 "지금 agent 기준으로 앞으로 어떻게 움직이는가"를 배우게 된다. 이는 translation/rotation variation을 줄이고 trajectory forecasting에서 흔히 쓰이는 canonicalization이다.

### Q3. input feature에 future 정보가 들어갔나?

아니다. `X`는 timestep 0-49의 position, velocity, heading에서만 만들어진다. `Y`는 timestep 50-79의 future relative position이다. 다만 좌표 변환 기준인 origin/theta는 마지막 관측 timestep 49에서만 가져온다. 미래 좌표는 target 생성에만 사용된다.

### Q4. 데이터 누수를 어떻게 막았나?

- scaler는 train split의 valid observed `X`로만 fit한다.
- PCA codec은 train split의 future `Y`로만 fit한다.
- validation `Y`는 PCA transform/evaluation 대상으로만 사용한다.
- prediction payload 분석 시 `gt`, `mask_y`, `scenario_id`, `track_id` alignment를 검사한다.
- train/val은 AV2 split directory 자체가 분리되어 있고 output filename도 `train_full`, `val_full`처럼 구분된다.

### Q5. mask는 어떻게 처리했나?

관측 mask `mask_x`와 미래 mask `mask_y`를 저장한다. 전처리에서는 마지막 관측 timestep이 반드시 valid여야 하고, 미래 30 step은 모두 valid해야 sample을 채택한다. masked payload는 0으로 채운다. metrics/loss는 mask-aware로 구현되어 final valid timestep과 valid-step average를 사용한다.

### Q6. object type은 어떻게 처리했나?

지원 object type은 vehicle과 pedestrian이다. mapping은 `vehicle=0`, `pedestrian=1`이다. full AV2 결과는 focal target이 vehicle 또는 pedestrian인 sample들로 구성된다. cluster analysis를 보면 대부분 vehicle이고, cluster 2에 pedestrian sample이 가장 많이 포함된다.

### Q7. 왜 map/lane information을 쓰지 않았나?

이 프로젝트는 모델 비교형 수업/포트폴리오 프로젝트로 범위를 제한했다. HD map, lane graph, traffic light, multi-agent interaction은 구현 범위에서 제외했고 future work로 남겼다. 그래서 결과는 "trajectory history only" 모델 비교로 해석해야 한다.

### Q8. Linear baseline은 정확히 무엇인가?

마지막 관측 위치와 velocity를 사용한 constant-velocity extrapolation이다. parameter가 없고 학습하지 않는다. 미래 `t` step은 `p_last + v_last * t * dt`로 예측한다.

### Q9. LSTM과 Transformer의 차이는?

LSTM은 과거 sequence를 recurrent encoder로 요약하고, decoder가 30 step을 autoregressive하게 출력한다. Transformer는 과거 sequence 전체에 positional encoding과 self-attention을 적용한 뒤 마지막 encoded token으로 미래 60D를 직접 regression한다.

### Q10. Diffusion은 무엇을 예측하나?

Direct Diffusion은 future trajectory `[30,2]`를 flatten한 60D vector를 diffusion target으로 둔다. 학습 시에는 trajectory에 noise를 섞은 뒤 noise를 예측하도록 훈련한다. 추론 시에는 random noise에서 시작해 denoising 과정을 거쳐 여러 개의 미래 trajectory sample을 생성한다.

### Q11. PCA Diffusion은 Direct Diffusion과 무엇이 다른가?

PCA Diffusion은 future trajectory 60D를 train-set PCA latent로 압축하고, 그 latent 공간에서 diffusion을 수행한다. 생성된 latent는 PCA inverse transform으로 다시 `[30,2]` trajectory가 된다. full final 결과에서는 PCA Diffusion이 best-of-K minADE/minFDE 기준으로 가장 좋았다.

### Q12. 왜 diffusion의 ADE/FDE와 minADE/minFDE를 따로 봐야 하나?

Diffusion은 여러 가능한 미래를 생성하는 multi-sample 모델이다. 첫 번째 sample 하나만 보면 deterministic model처럼 ADE/FDE를 계산할 수 있지만, 그 sample이 반드시 가장 좋은 후보는 아니다. minADE/minFDE는 K개 후보 중 ground truth에 가장 가까운 후보를 보는 best-of-K 지표다. 따라서 diffusion의 sample-set quality를 보여주지만, deterministic model의 single prediction ADE/FDE와 직접 동일선상에서 비교하면 안 된다.

### Q13. 최종적으로 어떤 모델이 가장 좋았나?

단일 예측 ADE/FDE 기준으로는 LSTM이 가장 좋았다.

```text
LSTM: ADE 0.79166, FDE 2.05703
Transformer: ADE 0.85148, FDE 2.18351
Linear: ADE 1.53242, FDE 3.79734
```

best-of-K 기준으로는 PCA Diffusion이 가장 좋았다.

```text
PCA Diffusion: minADE 0.41746, minFDE 0.85825
Direct Diffusion: minADE 0.50093, minFDE 0.98452
```

### Q14. Miss Rate가 diffusion에서 높은 이유는?

현재 table의 Miss Rate는 diffusion의 첫 sample `pred=samples[:,0]`로 계산된다. 첫 sample이 약하면 Miss Rate가 높게 나올 수 있다. best-of-K 관점에서는 `minMiss Rate`를 별도로 볼 수 있고, final error analysis에서는 PCA Diffusion minMiss Rate가 0.0967, Direct Diffusion minMiss Rate가 0.1176으로 크게 낮다.

### Q15. full AV2 preprocessing 중 손상/읽기 실패 파일은 어떻게 처리했나?

read error type이 `PermissionError`, `OSError`, PyArrow exception이면 해당 parquet을 skip하고 metadata에 기록한다. full run에서는 unreadable train parquet 25개가 skip됐다. `max_read_errors` 기본값이 1000이라 skip이 과도해지면 RuntimeError로 중단한다.

### Q16. full AV2 결과가 실제 데이터 기반인가?

그렇다. `PROJECT_STATUS.md`, `outputs/full_av2/metadata/status.json`, `outputs/full_av2/tables/model_comparison.csv`, `outputs/report_summary.md`에 따르면 full AV2 long run은 Windows HOME에서 repo commit `1e511e3`로 완료됐고, all required artifacts가 존재했다. final metrics는 `val_full` 23,706 sample 평가에서 나온 값이다.

### Q17. 왜 Windows에서 학습하고 Mac에서 보고서를 만들었나?

AV2 raw/processed data와 GPU 학습은 Windows HOME의 D drive와 NVIDIA GPU를 사용했다. Mac은 코드, 문서, 테스트, 평가, 시각화, 분석, Git의 source of truth로 유지했다. 큰 raw data/checkpoint/log는 Git에 올리지 않고, lightweight 결과만 Mac으로 회수했다.

### Q18. 평가 결과를 fake하지 않았다는 근거는?

- final comparison table과 metrics JSON이 존재한다.
- Windows status JSON에 complete status, repo head, run path, artifact 존재 여부가 기록되어 있다.
- prediction payload는 `gt`, `mask_y`, `scenario_id`, `track_id` alignment를 검사하도록 analysis code가 작성되어 있다.
- PROJECT_STATUS에 단계별 commands/results가 기록되어 있다.
- 테스트가 preprocessing/metrics/analysis alignment failure case까지 검증한다.

### Q19. 이 프로젝트의 한계는?

- HD map/lane graph/traffic light를 사용하지 않는다.
- multi-agent interaction modeling이 없다.
- focal target 중심이며, vehicle/pedestrian 별도 결과는 깊게 분리하지 않았다.
- diffusion min metrics는 K에 의존한다.
- validation split에서 tuning gate를 사용했으므로 결과 claim은 이 실험 setup과 split에 한정해야 한다.
- raw AV2와 checkpoint가 Git에 없기 때문에 완전 재현에는 Windows data path와 ignored artifacts가 필요하다.

### Q20. 발표에서 강조할 포인트는?

- 같은 processed schema와 metrics에서 5개 모델을 비교했다.
- 전처리에서 last-observed coordinate frame을 사용해 motion canonicalization을 했다.
- Linear 대비 LSTM/Transformer가 크게 개선됐다.
- Diffusion은 single-sample은 약하지만 best-of-K에서는 다양한 plausible future 후보를 생성할 수 있음을 보여준다.
- 데이터 누수 방지를 위해 scaler/PCA는 train-only로 fit했다.
- analysis code가 prediction alignment를 검증해 결과 누락/불일치를 줄인다.

## 18. GPT가 답변할 때 지켜야 할 claim boundary

강하게 말해도 되는 것:

- full AV2 final comparison에서 LSTM이 single prediction ADE/FDE best다.
- PCA Diffusion이 best-of-K minADE/minFDE best다.
- 전처리는 마지막 관측 위치/heading 기준 상대좌표를 사용한다.
- input feature는 6D: rel position, velocity, sin/cos heading.
- output은 future relative trajectory `[30,2]`.
- full validation sample count는 23,706이다.
- train_full sample count는 189,541이다.

조심해서 말해야 하는 것:

- leaderboard 성능이나 SOTA claim.
- real-world deployment readiness.
- map-aware 모델보다 좋다는 claim.
- diffusion이 deterministic model보다 전반적으로 우수하다는 claim.
- validation tuning 이후 test generalization claim.

말하면 안 되는 것:

- raw data나 checkpoint가 Git에 포함되어 있다고 말하기.
- AV2 test leaderboard 제출을 했다고 말하기.
- metrics가 없는데 새 결과가 있다고 말하기.
- map/lane/multi-agent interaction을 사용했다고 말하기.

## 19. 파일별 빠른 답변 지도

전처리 질문:

- `src/datasets/preprocess_av2.py`
- `src/datasets/validate_processed.py`
- `src/utils/geometry.py`
- `configs/preprocess_small.yaml`
- `tests/test_preprocess_av2.py`

데이터 schema/DataLoader 질문:

- `src/datasets/av2_dataset.py`
- `tests/test_synthetic_data.py`
- `tests/test_preprocess_av2.py`

Synthetic data 질문:

- `src/datasets/synthetic.py`
- `configs/preprocess_smoke.yaml`

모델 질문:

- `src/models/linear.py`
- `src/models/lstm.py`
- `src/models/transformer.py`
- `src/models/diffusion.py`
- `src/models/pca_latent.py`

학습 질문:

- `src/training/train.py`
- `src/training/trainer.py`
- `src/training/losses.py`
- model configs under `configs/`.

평가 질문:

- `src/evaluation/metrics.py`
- `src/evaluation/evaluate.py`
- `scripts/run_all_evaluations.py`

분석/시각화 질문:

- `src/analysis/common.py`
- `src/analysis/pca_analysis.py`
- `src/analysis/kmeans_analysis.py`
- `src/analysis/error_analysis.py`
- `src/visualization/*.py`

최종 결과 질문:

- `outputs/report_summary.md`
- `outputs/full_av2/tables/model_comparison.csv`
- `outputs/full_av2_analysis/tables/error_summary.csv`
- `outputs/full_av2_analysis/tables/cluster_metrics.csv`
- `outputs/full_av2_tuning/tables/selected_diffusion_configs.json`
- `PROJECT_STATUS.md`

Windows/GPU/실험 provenance 질문:

- `docs/windows_gpu_training_only_workflow.md`
- `docs/WINDOWS_ENV_SETUP.md`
- `docs/full_av2_training_staged_workflow.md`
- `scripts/windows_full_long_experiments.ps1`
- `outputs/full_av2/metadata/status.json`

Git/artifact policy 질문:

- `AGENTS.md`
- `GOAL_RUNBOOK.md`
- `docs/github_portfolio_workflow.md`
- `.gitignore`

