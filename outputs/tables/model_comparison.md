| model | data_split | target_type | ADE | FDE | minADE | minFDE | Miss_Rate | Latency_ms | Params | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| linear | val_smoke | synthetic_mixed | 3.7614 | 8.52468 | 3.7614 | 8.52468 | 0.826667 | 0.143627 | 0 | single prediction; min metrics equal ADE/FDE |
| lstm | val_smoke | synthetic_mixed | 8.51447 | 17.3557 | 8.51447 | 17.3557 | 1 | 4.00009 | 401666 | single prediction; min metrics equal ADE/FDE |
| transformer | val_smoke | synthetic_mixed | 8.33797 | 17.6097 | 8.33797 | 17.6097 | 1 | 0.466539 | 406332 | single prediction; min metrics equal ADE/FDE |
| diffusion_direct | val_smoke | synthetic_mixed | 9.96528 | 18.5409 | 9.77279 | 17.4611 | 0.986667 | 0.892321 | 70588 | multi-sample evaluation; K=4 |
| diffusion_pca | val_smoke | synthetic_mixed | 4.49675 | 9.38416 | 4.34296 | 8.74119 | 0.886667 | 1.40448 | 58252 | multi-sample evaluation; K=4 |
