| model | data_split | target_type | ADE | FDE | minADE | minFDE | Miss_Rate | Latency_ms | Params | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| linear | val_full | av2_focal_mixed | 1.53242 | 3.79734 | 1.53242 | 3.79734 | 0.596305 | 0.0193513 | 0 | single prediction; min metrics equal ADE/FDE |
| lstm | val_full | av2_focal_mixed | 1.04012 | 2.60335 | 1.04012 | 2.60335 | 0.479204 | 0.101707 | 35970 | single prediction; min metrics equal ADE/FDE |
| transformer | val_full | av2_focal_mixed | 0.94158 | 2.41605 | 0.94158 | 2.41605 | 0.434531 | 0.0376241 | 71420 | single prediction; min metrics equal ADE/FDE |
| diffusion_pca | val_full | av2_focal_mixed | 6.27722 | 12.3223 | 6.09676 | 11.6064 | 0.936219 | 0.0819716 | 46156 | multi-sample evaluation; K=4 |
| diffusion_direct | val_full | av2_focal_mixed | 10.0586 | 19.4482 | 9.88166 | 18.5069 | 0.904623 | 0.0955518 | 55420 | multi-sample evaluation; K=4 |
