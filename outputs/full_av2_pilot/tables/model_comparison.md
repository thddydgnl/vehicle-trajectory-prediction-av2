| model | data_split | target_type | ADE | FDE | minADE | minFDE | Miss_Rate | Latency_ms | Params | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| linear | val_full | av2_focal_mixed | 1.53242 | 3.79734 | 1.53242 | 3.79734 | 0.596305 | 0.0191419 | 0 | single prediction; min metrics equal ADE/FDE |
| lstm | val_full | av2_focal_mixed | 1.3741 | 3.95133 | 1.3741 | 3.95133 | 0.594153 | 0.102535 | 35970 | single prediction; min metrics equal ADE/FDE |
| transformer | val_full | av2_focal_mixed | 1.08678 | 2.70783 | 1.08678 | 2.70783 | 0.511305 | 0.0372091 | 71420 | single prediction; min metrics equal ADE/FDE |
| diffusion_pca | val_full | av2_focal_mixed | 6.28801 | 12.3061 | 6.10756 | 11.5999 | 0.935544 | 0.0828328 | 46156 | multi-sample evaluation; K=4 |
| diffusion_direct | val_full | av2_focal_mixed | 10.1179 | 19.4552 | 9.94257 | 18.4612 | 0.908589 | 0.0963579 | 55420 | multi-sample evaluation; K=4 |
