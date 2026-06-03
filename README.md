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

Phase 0부터 Phase 15까지 구현과 검증이 완료되었습니다.

- Synthetic smoke 데이터 파이프라인 완료
- Geometry, dataset loader, metrics, training loop 완료
- LSTM, Transformer, Direct Diffusion, PCA Latent Diffusion 구현 완료
- AV2 raw 데이터 준비와 Phase 11 전처리 완료
- 시각화, PCA/K-means/error analysis 완료
- Phase 14 final experiment matrix와 Windows GPU smoke/pilot 실험 완료
- Full AV2 diffusion tuning gate와 all-model long run 완료
- Phase 15 final report assets 완료

최종 full AV2 long run은 Windows HOME GPU에서 실행했고, Mac에서는 lightweight
metrics, tables, figures, report만 회수해 검증했습니다. raw data, processed
`.npz`, checkpoints, logs, prediction payloads는 Git에 올리지 않습니다.

## Full AV2 Long Run 결과

검증 데이터: `val_full`, target: `av2_focal_mixed`

결과 파일:
`outputs/full_av2/tables/model_comparison.md`

| Model | ADE | FDE | minADE | minFDE | Sample Diversity | Miss Rate | Params |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Linear | 1.53242 | 3.79734 | 1.53242 | 3.79734 |  | 0.59630 | 0 |
| LSTM | 0.79166 | 2.05703 | 0.79166 | 2.05703 |  | 0.38163 | 401,666 |
| Transformer | 0.85148 | 2.18351 | 0.85148 | 2.18351 |  | 0.40007 | 406,332 |
| PCA Diffusion | 1.43391 | 3.66399 | 0.41746 | 0.85825 | 1.54089 | 0.62415 | 222,988 |
| Direct Diffusion | 1.92816 | 4.91142 | 0.50093 | 0.98452 | 2.31048 | 0.72674 | 531,132 |

LSTM이 단일 예측 ADE/FDE 기준으로 가장 좋은 결과를 냈습니다. Diffusion 모델은
첫 번째 단일 샘플 ADE/FDE로는 LSTM/Transformer보다 약하지만, `K=16` 생성 샘플의
best-of-K 지표인 minADE/minFDE에서는 PCA Diffusion과 Direct Diffusion이 매우 낮은
값을 보였습니다. 이 결과는 diffusion이 여러 가능한 미래 중 좋은 후보를 생성할
수 있음을 보여주지만, deterministic 모델과 같은 방식으로 직접 비교하면 안 됩니다.

최종 보고서:

- `outputs/report_summary.md`
- `outputs/full_av2/tables/model_comparison.md`
- `outputs/full_av2_analysis/tables/cluster_metrics.csv`
- `outputs/full_av2_analysis/tables/error_summary.csv`
- `outputs/full_av2/figures/trajectory_overlay_best_cases.png`
- `outputs/full_av2/figures/trajectory_overlay_worst_cases.png`
- `outputs/full_av2/figures/diffusion_samples_interesting_case.png`

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
outputs/full_av2*        full AV2 pilot/long-run lightweight 결과
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

## 남은 개선 아이디어

현재 저장소는 수업 발표와 포트폴리오 제출 가능한 상태입니다. 이후 개선한다면
map/lane context, vehicle/pedestrian 별도 리포트, diffusion calibration, 더 긴
학습/반복 실험으로 확장할 수 있습니다.
