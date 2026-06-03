| model | data_split | target_type | ADE | FDE | minADE | minFDE | Miss_Rate | Latency_ms | Params | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| linear | val_small | av2_focal_mixed | 1.46284 | 3.4918 | 1.46284 | 3.4918 | 0.515789 | 0.177512 | 0 | single prediction; min metrics equal ADE/FDE |
| lstm | val_small | av2_focal_mixed | 10.7344 | 21.0111 | 10.7344 | 21.0111 | 0.873684 | 8.12172 | 35970 | single prediction; min metrics equal ADE/FDE |
| transformer | val_small | av2_focal_mixed | 10.1987 | 19.4198 | 10.1987 | 19.4198 | 0.863158 | 2.72306 | 71420 | single prediction; min metrics equal ADE/FDE |
| diffusion_direct | val_small | av2_focal_mixed | 10.9464 | 21.1416 | 10.7912 | 20.1243 | 0.894737 | 0.966818 | 55420 | multi-sample evaluation; K=4 |
| diffusion_pca | val_small | av2_focal_mixed | 7.10332 | 13.7734 | 6.91784 | 12.9242 | 0.947368 | 1.30593 | 46156 | multi-sample evaluation; K=4 |
