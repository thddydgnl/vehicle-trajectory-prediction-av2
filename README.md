# Argoverse 2 차량·보행자 미래 궤적 예측

Argoverse 2 Motion Forecasting 데이터를 기반으로 주변 차량과 보행자의 과거
5초 움직임을 보고 미래 3초 궤적을 예측하는 포트폴리오 프로젝트입니다.

Linear Extrapolation, LSTM, Transformer, Direct Diffusion, PCA Latent Diffusion을
같은 데이터 포맷과 같은 평가 지표에서 비교합니다. 목표는 leaderboard 제출용
SOTA 재현이 아니라, 데이터 전처리부터 모델 학습, 평가, 시각화, 분석까지 이어지는
end-to-end trajectory forecasting 파이프라인을 직접 구현하고 검증하는 것입니다.

## 프로젝트 개요

| 항목 | 내용 |
| --- | --- |
| Dataset | Argoverse 2 Motion Forecasting |
| Task | 과거 5초로 미래 3초 궤적 예측 |
| Sampling | 10Hz |
| Input | 50 step, relative position/velocity/heading features |
| Output | 30 step future trajectory, `[30, 2]` |
| Target | focal agent 중심, vehicle/pedestrian 지원 |
| Metrics | ADE, FDE, Miss Rate, minADE, minFDE, latency, parameter count |
| Models | Linear, LSTM, Transformer, Direct Diffusion, PCA Diffusion |

## 현재 상태

Phase 0부터 Phase 14까지 구현과 검증이 완료되었습니다.

- Synthetic smoke 데이터 파이프라인 완료
- Geometry, dataset loader, metrics, training loop 완료
- LSTM, Transformer, Direct Diffusion, PCA Latent Diffusion 구현 완료
- AV2 raw 데이터 준비와 Phase 11 전처리 완료
- 시각화, PCA/K-means/error analysis 완료
- Phase 14 final experiment matrix와 Windows GPU smoke/pilot 실험 완료
- Phase 15 final report assets는 다음 단계입니다

주의: 아래 full AV2 결과는 report-ready long run이 아니라 5-epoch pilot 결과입니다.
장기 학습 F5 결과는 아직 완료로 주장하지 않습니다.

## Full AV2 5-Epoch Pilot 결과

검증 데이터: `val_full`, target: `av2_focal_mixed`

결과 파일:
`outputs/full_av2_5epoch_pilot/tables/model_comparison.md`

| Model | ADE | FDE | minADE | minFDE | Miss Rate | Params |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Linear | 1.53242 | 3.79734 | 1.53242 | 3.79734 | 0.59631 | 0 |
| LSTM | 1.04012 | 2.60335 | 1.04012 | 2.60335 | 0.47920 | 35,970 |
| Transformer | 0.94158 | 2.41605 | 0.94158 | 2.41605 | 0.43453 | 71,420 |
| PCA Diffusion | 6.27722 | 12.32230 | 6.09676 | 11.60640 | 0.93622 | 46,156 |
| Direct Diffusion | 10.05860 | 19.44820 | 9.88166 | 18.50690 | 0.90462 | 55,420 |

현재 pilot 기준으로는 Transformer가 가장 낮은 ADE/FDE를 보였고, diffusion 계열은
추가 튜닝이 필요한 상태입니다. diffusion 결과는 실패로 숨기지 않고, 같은 지표와
같은 데이터에서 비교 대상으로 기록합니다.

## 주요 기능

- AV2 scenario parquet 기반 전처리
- 마지막 관측 시점 기준 상대좌표 변환
- train-only scaler/PCA fitting으로 데이터 누수 방지
- `.npz` 공통 데이터 포맷
- mask-aware ADE/FDE/minADE/minFDE
- 공통 PyTorch training/evaluation pipeline
- Windows GPU 원격 학습 workflow
- trajectory overlay, error histogram, PCA, K-means 분석
- lightweight 결과만 Git에 커밋하는 데이터/아티팩트 정책

## 저장소 구조

```text
configs/                 모델, 전처리, full AV2 pilot/long-run 설정
docs/                    goal runbook, Windows workflow, staged training 문서
scripts/                 Windows 원격 실험 및 평가 실행 스크립트
src/
  analysis/              PCA, K-means, error analysis
  datasets/              synthetic, AV2 preprocessing, validation, dataset loader
  evaluation/            metrics, evaluation CLI
  models/                linear, LSTM, Transformer, diffusion models
  training/              trainer, losses, train CLI
  visualization/         trajectory/error/PCA/cluster plotting
tests/                   unit/smoke tests
outputs/tables/          lightweight comparison tables
outputs/figures/         lightweight figures
outputs/full_av2_*       full AV2 pilot의 lightweight 결과
```

## 빠른 시작

의존성 설치:

```bash
python -m pip install -r requirements.txt
```

전체 테스트:

```bash
pytest -q
```

Synthetic smoke 데이터 생성:

```bash
python -m src.datasets.synthetic --out_dir data/processed --num_samples 1000
```

Synthetic smoke 학습 예시:

```bash
python -m src.training.train \
  --config configs/lstm.yaml \
  --max_epochs 1 \
  --data data/processed/train_smoke.npz \
  --val_data data/processed/val_smoke.npz
```

평가 예시:

```bash
python -m src.evaluation.evaluate \
  --model linear \
  --data data/processed/val_smoke.npz \
  --config configs/linear.yaml \
  --out_dir outputs
```

## AV2와 Windows GPU Workflow

Mac은 코드, 문서, 테스트, 평가, 시각화, Git의 source of truth입니다.
Windows HOME 머신은 AV2 데이터 저장, Windows-local preprocessing, GPU 학습 노드로
사용합니다.

기본 Windows 경로:

```text
Raw AV2:   D:\data\av2\motion-forecasting
Processed: D:\data\vehicle_trajectory_project\processed
Runs:      D:\runs\vehicle_trajectory_project
```

Phase 11 이상의 real AV2 작업 전에는 다음 marker를 확인합니다.

```text
D:\data\av2\motion-forecasting\DATA_READY_FOR_PHASE11.txt
```

자세한 절차는 다음 문서를 따릅니다.

- `GOAL_RUNBOOK.md`
- `PROJECT_STATUS.md`
- `docs/windows_gpu_training_only_workflow.md`
- `docs/WINDOWS_ENV_SETUP.md`
- `docs/full_av2_training_staged_workflow.md`
- `docs/github_portfolio_workflow.md`

## 데이터와 아티팩트 정책

다음 파일은 Git에 올리지 않습니다.

- raw AV2 데이터
- processed `.npz`
- checkpoints
- prediction payloads
- training logs
- secrets, credentials

Git에는 코드, 설정, 테스트, 문서, 작은 표/그림, lightweight metrics만 남깁니다.
실험 결과를 포트폴리오로 보여주되, 재현성과 저장소 용량을 해치지 않는 방향을
기본 원칙으로 둡니다.

## 다음 단계

1. Stage F5A diffusion tuning gate 실행
2. Gate 통과 시 Stage F5B all-model long run 실행
3. lightweight 결과 회수
4. Phase 15 final report assets 작성

완료되지 않은 장기 실험 결과는 README나 보고서에서 성공으로 주장하지 않습니다.
