| model | data_split | target_type | ADE | FDE | minADE | minFDE | Sample_Diversity | Miss_Rate | Latency_ms | Params | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| linear | val_full | av2_focal_mixed | 1.53242 | 3.79734 | 1.53242 | 3.79734 |  | 0.596305 | 0.0191954 | 0 | single prediction; min metrics equal ADE/FDE |
| lstm | val_full | av2_focal_mixed | 0.791665 | 2.05703 | 0.791665 | 2.05703 |  | 0.381633 | 0.196536 | 401666 | single prediction; min metrics equal ADE/FDE |
| transformer | val_full | av2_focal_mixed | 0.85148 | 2.18351 | 0.85148 | 2.18351 |  | 0.400068 | 0.0453469 | 406332 | single prediction; min metrics equal ADE/FDE |
| diffusion_pca | val_full | av2_focal_mixed | 1.43391 | 3.66399 | 0.417465 | 0.858253 | 1.54089 | 0.624146 | 0.926293 | 222988 | multi-sample evaluation; K=16 |
| diffusion_direct | val_full | av2_focal_mixed | 1.92816 | 4.91142 | 0.500931 | 0.984522 | 2.31048 | 0.726736 | 0.970593 | 531132 | multi-sample evaluation; K=16 |
