# Phase 15 Final Report Summary

## Project Problem

This project predicts the next 3 seconds of vehicle/pedestrian motion from the previous 5 seconds of trajectory history. It compares Linear Extrapolation, LSTM, Transformer, Direct Diffusion, and PCA Latent Diffusion under the same processed AV2-style input format.

## Dataset And Setup

- Dataset: Argoverse 2 Motion Forecasting, focal vehicle/pedestrian target mix.
- Input: 50 observed steps with relative position, velocity, and heading features.
- Output: 30 future `(x, y)` steps in the last-observed-agent coordinate frame.
- Full AV2 preprocessing and GPU training were run on Windows HOME; code, reports, and Git commits are managed from Mac.

## Metrics

- ADE/FDE evaluate the single saved trajectory prediction.
- For diffusion models, minADE/minFDE are best-of-K metrics across generated samples; they should be interpreted as sample-set quality, not a single deterministic trajectory.
- Miss Rate uses the configured FDE threshold from the evaluator.

## Main Results

Best ADE model in the final comparison table: **lstm** with ADE 0.7917 and FDE 2.0570.

| model | ADE | FDE | minADE | minFDE | Sample_Diversity | Miss_Rate | Params |
| --- | --- | --- | --- | --- | --- | --- | --- |
| linear | 1.5324 | 3.7973 | 1.5324 | 3.7973 |  | 0.5963 | 0 |
| lstm | 0.7917 | 2.0570 | 0.7917 | 2.0570 |  | 0.3816 | 401666 |
| transformer | 0.8515 | 2.1835 | 0.8515 | 2.1835 |  | 0.4001 | 406332 |
| diffusion_pca | 1.4339 | 3.6640 | 0.4175 | 0.8583 | 1.5409 | 0.6241 | 222988 |
| diffusion_direct | 1.9282 | 4.9114 | 0.5009 | 0.9845 | 2.3105 | 0.7267 | 531132 |

## Diffusion Tuning Gate

The final long run was allowed only after the selected PCA Diffusion and Direct Diffusion candidates passed the user-defined minADE/minFDE gates on real `val_full` outputs.

| model | candidate_id | target_gate | ADE | FDE | minADE | minFDE | Sample_Diversity | epochs_ran |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| diffusion_pca | pca_b | True | 1.5422 | 3.8595 | 0.4569 | 0.9441 | 1.6574 | 15 |
| diffusion_direct | direct_f | True | 4.1821 | 9.8780 | 0.7230 | 1.4558 | 5.5345 | 15 |

## Error Analysis And Figures

- Analysis tables: `outputs/full_av2_analysis/tables`
- Report figures: `outputs/full_av2/figures`
- Required Phase 15 figures include best-case overlays, worst-case overlays, diffusion sample interpretation, PCA, K-means, and error histograms.

## Interpretation

The deterministic sequence models show how much temporal modeling improves over a constant-velocity baseline. The diffusion models are evaluated separately with both single-sample ADE/FDE and best-of-K minADE/minFDE, which exposes whether the sampler can generate at least one plausible future even when an arbitrary first sample is weaker.

## Limitations

- The project does not use HD maps, lane graphs, traffic lights, or multi-agent interaction modeling.
- Diffusion min metrics depend on the number of samples K and are not directly equivalent to deterministic ADE/FDE.
- Tuning was performed on validation outputs for a school project setting, so final claims should stay scoped to this dataset split and experiment setup.

## Future Work

- Add map/lane context and multi-agent interaction features.
- Separate vehicle and pedestrian reporting more deeply.
- Explore stronger diffusion objectives, trajectory anchors, and calibration of sample diversity.

## Artifact Policy

Raw AV2 data, processed `.npz` files, checkpoints, logs, and prediction payloads are intentionally excluded from Git. The repository commits code, configs, tests, lightweight metrics, tables, figures, and this report summary.
