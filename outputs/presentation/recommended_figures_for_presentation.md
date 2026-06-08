# 발표 자료에 꼭 넣으면 좋은 그림

이 문서는 최종 발표본 `vehicle_trajectory_forecasting_presentation_final.pptx`에 사용한 발표용 그림과, 추가로 넣기 좋은 프로젝트 산출물을 정리한 것이다.

최종 PPT에는 그림 내부 제목/범례가 겹치지 않도록 아래 발표용 clean figure를 사용했다.

## 1순위: 반드시 넣기

| PPT 위치 | 추천 그림 | 넣는 이유 |
|---|---|---|
| 1장 또는 10장 | `outputs/presentation/figures/presentation_best_case_overlay.png` | 실제 궤적과 예측 궤적이 잘 맞은 사례를 보여줘 프로젝트 주제를 직관적으로 설명할 수 있음 |
| 10장 | `outputs/presentation/figures/presentation_worst_case_overlay.png` | 어려운 사례를 함께 보여줘 결과를 과장하지 않고 분석했다는 인상을 줄 수 있음 |
| 10장 | `outputs/presentation/figures/presentation_diffusion_candidates.png` | Diffusion이 여러 미래 후보를 생성한다는 차별점을 설명하기 좋음 |

## 2순위: 시간이 있으면 넣기

| PPT 위치 | 추천 그림 | 넣는 이유 |
|---|---|---|
| 결과 또는 부록 | `outputs/full_av2/figures/error_histogram_fde.png` | 모델별 최종 위치 오차 분포를 보여줘 ADE/FDE 표를 보완할 수 있음 |
| 결과 또는 부록 | `outputs/full_av2/figures/error_histogram_ade.png` | 전체 궤적 평균 오차 분포를 보여줘 모델별 안정성을 설명하기 좋음 |
| 데이터/분석 또는 부록 | `outputs/full_av2/figures/pca_trajectory_space.png` | 궤적 패턴이 PCA 공간에서 어떻게 분포하는지 보여줄 수 있음 |
| 데이터/분석 또는 부록 | `outputs/full_av2/figures/kmeans_clusters.png` | K-means 기반 trajectory pattern 분석을 시각적으로 보여줄 수 있음 |

## 발표에서 가장 추천하는 조합

5분 발표라면 그림을 너무 많이 넣기보다 아래 3개만 넣는 것을 추천한다.

1. `presentation_best_case_overlay.png`
   - 주제와 예측 성공 사례 설명

2. `presentation_worst_case_overlay.png`
   - 고찰과 한계 설명

3. `presentation_diffusion_candidates.png`
   - Diffusion 모델의 독창성 설명

## 넣는 방식

- 1장 오른쪽 그림 영역에는 `presentation_best_case_overlay.png`를 넣었다.
- 8장에는 Diffusion 튜닝 전후 비교를 PPT 내부 막대 그래프로 넣었다.
- 9장에는 Miss Rate와 파라미터 수 비교를 PPT 내부 막대/표로 넣었다.
- 10장 왼쪽 카드에는 `presentation_best_case_overlay.png`를 넣었다.
- 10장 가운데 카드에는 `presentation_worst_case_overlay.png`를 넣었다.
- 10장 오른쪽 카드에는 `presentation_diffusion_candidates.png`를 넣었다.
- 7장에는 이미 결과 표가 있으므로 그림을 추가한다면 `error_histogram_fde.png`를 작은 보조 그림으로 넣는 정도가 적당하다.

## 발표 멘트 예시

```text
왼쪽은 예측이 잘 된 사례이고, 가운데는 상대적으로 어려웠던 실패 사례입니다.
오른쪽은 Diffusion 모델이 하나의 미래가 아니라 여러 가능한 미래 후보를 생성하는 모습입니다.
이를 통해 LSTM/Transformer는 단일 예측 성능이 좋고, Diffusion은 후보 생성 관점에서 의미가 있음을 확인했습니다.
```
