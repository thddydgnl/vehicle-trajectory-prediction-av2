# Vehicle Trajectory Project Status

작성일: 2026-06-01

This file is the persistent progress ledger for Codex goal mode.

## Current State

```text
Phase 0 repository setup is complete.
Project folder now contains the base Python package, utilities, tests, config/data/output directories, and documentation.
Git repository has been initialized on branch main.
GitHub remote has been configured and pushed.
Long-form workflow documents have been moved under docs/.
Primary Windows training host has changed to HOME.
HOME SSH works over LAN.
Use HOME LAN first for Windows work.
HOME Tailscale SSH was documented as fallback, but port 22 timed out on 2026-06-03; re-verify before use.
HOME NVIDIA GPU is visible.
HOME Miniconda3 is installed at C:\Users\thddy\Miniconda3.
HOME vehicle_traj conda environment exists with Python 3.12 and CUDA PyTorch 2.11.0+cu128.
HOME vehicle_traj torch.cuda.is_available() is True on NVIDIA GeForce RTX 2070 SUPER.
HOME s5cmd is still only needed for future direct AV2 download or resync.
The previous song AV2 download attempt was stopped; do not treat the song AV2 folder as complete.
HOME AV2 Motion Forecasting archives are stored under D:\datasets\argoverse.
HOME AV2 extraction/organization completed successfully.
D:\data\av2\motion-forecasting\DATA_READY_FOR_PHASE11.txt exists.
HOME D:\data was re-verified on 2026-06-03 over LAN SSH.
Legacy song@100.87.219.58 is not the AV2 data host; it currently exposes only C: over SSH.
HOME Windows data/training folder layout is ready for Phase 11 preprocessing and later GPU training.
Phase 11 preprocessing code has been implemented, tested on Mac, committed, and pushed.
HOME Windows code root has been cloned from GitHub and updated to commit 9561eae.
HOME small AV2 preprocessing completed successfully on 2026-06-03.
D:\data\vehicle_trajectory_project\processed\small\train_small.npz and val_small.npz passed schema validation.
Phase 12 visualization code has been implemented and tested on Mac.
Synthetic and small AV2 trajectory/error/PCA/K-means smoke figures were generated successfully.
Phase 13 PCA/K-means/error analysis code has been implemented and tested on Mac.
Phase 13 analysis now validates prediction/data alignment and can require expected model predictions to avoid silent model omission.
Phase 14 final experiment matrix tooling has been implemented, tested, committed, and pushed.
Phase 14 Windows small AV2 GPU smoke training completed for LSTM, Transformer, Direct Diffusion, and PCA Diffusion.
Phase 14 all-model evaluation was rerun on Windows with explicit checkpoint_dir/checkpoint_tag and prediction_tag, then regenerated on Mac from ignored copied small checkpoints to keep Mac as the evaluation source of truth.
outputs/tables/model_comparison.csv and model_comparison.md now contain real val_small AV2 smoke results for Linear, LSTM, Transformer, Direct Diffusion, and PCA Diffusion.
Full AV2 preprocessing and Stage F3/F4 pilot training/evaluation have completed. Phase 14 results remain small AV2 smoke results; full AV2 pilot results are stored separately under outputs/full_av2_pilot and outputs/full_av2_5epoch_pilot.
Full AV2 staged training workflow is documented in docs/full_av2_training_staged_workflow.md.
Full AV2 preprocessing background retry on 2026-06-03 reached train file index 16609 and failed on a corrupted/unreadable parquet file with Windows WinError 1392.
AV2 preprocessing now skips logged parquet read errors including PermissionError, OSError, and PyArrow read errors, with max_read_errors defaulting to 1000 per split to avoid silently dropping too much data.
Full AV2 preprocessing was restarted on HOME at 2026-06-03T17:47+09:00 using commit f9a6a14; by 2026-06-03T18:18+09:00 it was still running at train 20617/199887, had skipped 3 unreadable parquet files, and had passed the previous failure at train 16609.
Full AV2 preprocessing completed on HOME by 2026-06-03T23:57+09:00. Outputs exist at D:\data\vehicle_trajectory_project\processed\full\train_full.npz and val_full.npz; schema validation passed with 189,541 train samples and 23,706 val samples. The run skipped 25 unreadable train parquet files.
Full AV2 Stage F3 completed on HOME by 2026-06-04T00:16+09:00 using commit 8c70cf1. CUDA preflight passed, Linear/LSTM/Transformer/PCA Diffusion/Direct Diffusion all produced finite val_full metrics, checkpoints, and comparison tables. Full pilot comparison tables and metrics were copied to outputs/full_av2_pilot without raw data, processed .npz files, checkpoints, logs, or prediction payloads.
Full AV2 Stage F4 completed on HOME by 2026-06-04T00:46+09:00 using commit 8014ed3. CUDA preflight passed, LSTM/Transformer/PCA Diffusion/Direct Diffusion each ran exactly 5 epochs with CUDA checkpoint metadata, and final comparison tables were generated from real val_full outputs. Lightweight F4 metrics, tables, and PCA figures were copied to outputs/full_av2_5epoch_pilot without raw data, processed .npz files, checkpoints, logs, or prediction payloads.
Full AV2 Stage F5A/F5B tooling has been prepared for diffusion tuning gates, all-model long runs, and Phase 15 final assets. Do not claim F5 long-run results until Windows outputs are verified and lightweight result artifacts are copied back to Mac.
Current user-selected F5 gate: do not start all-model FULL RUN until PCA Diffusion reaches minADE < 4.8 and minFDE < 9.5, and Direct Diffusion reaches minADE < 8.0 and minFDE < 15.0 on real val_full evaluation outputs.
F5A tuning tooling has been strengthened on Mac: skipped-step diffusion sampling now jumps to the actual next sampled timestep, PCA latent diffusion normalizes train-fitted latents, diffusion validation/checkpoint selection can use deterministic multi-sample minADE/minFDE, and the selector blocks final long-run config generation when the absolute target gates fail.
Full AV2 Stage F5A tuning completed on HOME using real val_full outputs. Selected PCA Diffusion candidate pca_b reached minADE 0.45693081617355347 and minFDE 0.9440990686416626. Selected Direct Diffusion candidate direct_f reached minADE 0.7230075597763062 and minFDE 1.4557510614395142. Both target gates passed and selected_diffusion_configs.json reported full_run_ready=true.
Full AV2 Stage F5B all-model long run completed on HOME at 2026-06-04T06:48:13+09:00 using repo commit 1e511e3. Windows status.json reports status=complete, exit_code=0, all required final artifacts present, and FULL_LONG_EXPERIMENTS_COMPLETE.txt exists.
Final val_full comparison: Linear ADE 1.5324182510375977 / FDE 3.7973430156707764; LSTM ADE 0.7916645407676697 / FDE 2.0570261478424072; Transformer ADE 0.8514801263809204 / FDE 2.183514356613159; PCA Diffusion ADE 1.4339096546173096 / FDE 3.6639866828918457 / minADE 0.41746464371681213 / minFDE 0.8582534193992615; Direct Diffusion ADE 1.9281558990478516 / FDE 4.91142463684082 / minADE 0.5009313821792603 / minFDE 0.9845224618911743.
Lightweight F5B metrics, tables, figures, tuning summaries, status.json, and complete marker were copied to outputs/full_av2 and outputs/full_av2_tuning. Full .npz files and prediction payloads were copied to ignored local paths only for Mac-side Phase 15 analysis and must not be committed.
Phase 15 report assets were generated on Mac: outputs/report_summary.md, final model comparison tables, full AV2 PCA/K-means/error analysis tables, and presentation figures including best/worst overlays and diffusion sample interpretation.
```

Verified Windows access:

```text
Mac Tailscale IP: 100.75.150.25
Primary Windows host: HOME
HOME LAN IP: 192.168.35.17
HOME Tailscale IP: 100.99.63.23
SSH user: thddy
Primary command: ssh thddy@192.168.35.17 'hostname && whoami'
Fallback command: ssh thddy@100.99.63.23 'hostname && whoami'
Fallback status: timed out on 2026-06-03, verify before use
Output: HOME / home\thddy
Legacy secondary Windows host: song, song@100.87.219.58, not for AV2 data
```

## Phase Progress

```text
Phase 0  Repository Setup                         complete
Phase 1  Synthetic Smoke Dataset                  complete
Phase 2  Geometry and Coordinate Transform        complete
Phase 3  Dataset and DataLoader                   complete
Phase 4  Metrics                                  complete
Phase 5  Linear Baseline                          complete
Phase 6  Common Training Pipeline                 complete
Phase 7  LSTM Encoder-Decoder                     complete
Phase 8  Transformer Encoder                      complete
Phase 9  Direct Diffusion Model                   complete
Phase 10 PCA Latent Diffusion                     complete
Phase 11 Argoverse 2 Preprocessing                complete
Phase 12 Visualization                            complete
Phase 13 PCA and K-means Analysis                 complete
Phase 14 Final Experiment Matrix                  complete
Phase 15 Final Report Assets                      complete
```

## Next Recommended Task

```text
Phase 15 is complete with final full AV2 long-run results and report assets.
Next recommended task: use the repository for school presentation/report preparation, or optionally extend analysis with vehicle-vs-pedestrian breakdowns and map/lane context as future work.
Do not run additional long Windows training unless the user explicitly asks for a new experiment.
Keep raw AV2 data, processed .npz files, checkpoints, logs, and prediction payloads ignored and out of Git.
```

## Latest Verified Commands

```text
ssh thddy@192.168.35.17 'hostname && whoami'
ssh thddy@100.99.63.23 'hostname && whoami'
ssh -o ConnectTimeout=8 thddy@192.168.35.17 'cmd /c "dir D:\data"'
ssh -o ConnectTimeout=8 thddy@192.168.35.17 'cmd /c "dir D:\data\av2\motion-forecasting"'
ssh -o ConnectTimeout=8 thddy@192.168.35.17 'cmd /c "type D:\data\av2\motion-forecasting\DATA_READY_FOR_PHASE11.txt"'
ssh -o ConnectTimeout=8 thddy@100.99.63.23 'hostname && whoami'
ssh song@100.87.219.58 'powershell -NoProfile -ExecutionPolicy Bypass -Command "Get-CimInstance Win32_LogicalDisk | Select-Object DeviceID,DriveType,FileSystem,FreeSpace,Size,VolumeName | ConvertTo-Json -Depth 3"'
ssh thddy@192.168.35.17 'powershell -NoProfile -Command "hostname; whoami; python --version; where.exe python; nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader"'
ssh thddy@192.168.35.17 'powershell -NoProfile -Command "conda --version; conda env list; if (Test-Path ''C:\Users\thddy\bin\s5cmd\s5cmd.exe'') { & ''C:\Users\thddy\bin\s5cmd\s5cmd.exe'' version } else { ''s5cmd-missing'' }"'
ssh thddy@192.168.35.17 'powershell -NoProfile -Command "Get-ChildItem D:\data"'
scp /tmp/organize_phase11_av2.ps1 thddy@192.168.35.17:'D:/data/av2/organize_phase11_av2.ps1'
ssh thddy@192.168.35.17 'powershell -NoProfile -Command "schtasks /Create /TN VehicleTrajectoryAV2Organize /SC ONCE /ST <HH:mm> /TR \"powershell.exe -NoProfile -ExecutionPolicy Bypass -File D:\data\av2\organize_phase11_av2.ps1\" /F; schtasks /Run /TN VehicleTrajectoryAV2Organize"'
ssh thddy@192.168.35.17 'powershell -NoProfile -ExecutionPolicy Bypass -File D:\datasets\argoverse\finalize_training_folder_structure.ps1'
ssh thddy@192.168.35.17 'powershell -NoProfile -Command "Get-Content D:\data\av2\motion-forecasting\DATA_READY_FOR_PHASE11.txt; Get-ChildItem D:\data\vehicle_trajectory_project\processed; Get-ChildItem D:\runs\vehicle_trajectory_project"'
pytest -q
python -c "from src.utils.device import get_device; print(get_device())"
python -m src.datasets.synthetic --out_dir data/processed --num_samples 1000
pytest tests/test_synthetic_data.py -q
pytest tests/test_geometry.py -q
pytest tests/test_metrics.py -q
python -m src.evaluation.evaluate --model linear --data data/processed/val_smoke.npz --config configs/linear.yaml --out_dir outputs
pytest tests/test_losses.py -q
python -m src.training.train --config configs/lstm.yaml --max_epochs 1 --data data/processed/train_smoke.npz --val_data data/processed/val_smoke.npz
pytest tests/test_models_shape.py -q
python -m src.evaluation.evaluate --model lstm --checkpoint outputs/checkpoints/best_lstm.pt --data data/processed/val_smoke.npz --out_dir outputs
python -m src.training.train --config configs/transformer.yaml --max_epochs 1 --data data/processed/train_smoke.npz --val_data data/processed/val_smoke.npz
python -m src.evaluation.evaluate --model transformer --checkpoint outputs/checkpoints/best_transformer.pt --data data/processed/val_smoke.npz --out_dir outputs
pytest tests/test_diffusion_step.py -q
python -m src.training.train --config configs/diffusion_direct.yaml --max_epochs 1 --data data/processed/train_smoke.npz --val_data data/processed/val_smoke.npz
python -m src.evaluation.evaluate --model diffusion_direct --checkpoint outputs/checkpoints/best_diffusion_direct.pt --data data/processed/val_smoke.npz --out_dir outputs --batch_size 64
pytest tests/test_pca_latent.py -q
python -m src.analysis.pca_analysis --train_data data/processed/train_smoke.npz --out_dir outputs
python -m src.training.train --config configs/diffusion_pca.yaml --max_epochs 1 --data data/processed/train_smoke.npz --val_data data/processed/val_smoke.npz
python - <<'PY'
import torch
ckpt = torch.load('outputs/checkpoints/best_lstm_smoke.pt', map_location='cpu')
print(sorted(ckpt.keys()))
print(ckpt['metadata']['device'])
print(ckpt['metadata']['model']['architecture'])
PY
pytest -q
pytest tests/test_preprocess_av2.py -q
pytest -q
ssh thddy@192.168.35.17 'cmd /c "if not exist C:\Users\thddy\Documents\code\vehicle_trajectory_project\.git git clone https://github.com/thddydgnl/vehicle-trajectory-prediction-av2.git C:\Users\thddy\Documents\code\vehicle_trajectory_project && cd /d C:\Users\thddy\Documents\code\vehicle_trajectory_project && git checkout main && git pull --ff-only origin main && git rev-parse --short HEAD"'
ssh thddy@192.168.35.17 'cmd /c "cd /d C:\Users\thddy\Documents\code\vehicle_trajectory_project && python -m pip install numpy pandas pyarrow joblib tqdm pyyaml"'
ssh thddy@192.168.35.17 'cmd /c "cd /d C:\Users\thddy\Documents\code\vehicle_trajectory_project && python -m src.datasets.preprocess_av2 --config configs\preprocess_small.yaml"'
ssh thddy@192.168.35.17 'cmd /c "cd /d C:\Users\thddy\Documents\code\vehicle_trajectory_project && python -m src.datasets.validate_processed --npz D:\data\vehicle_trajectory_project\processed\small\train_small.npz"'
ssh thddy@192.168.35.17 'cmd /c "cd /d C:\Users\thddy\Documents\code\vehicle_trajectory_project && python -m src.datasets.validate_processed --npz D:\data\vehicle_trajectory_project\processed\small\val_small.npz"'
pytest tests/test_visualization.py -q
python -m src.visualization.plot_trajectories --data data/processed/val_smoke.npz --predictions outputs/predictions --out_dir outputs/figures --num_cases 10
python -m src.visualization.plot_errors --data data/processed/val_smoke.npz --predictions outputs/predictions --out_dir outputs/figures
python -m src.visualization.plot_pca --data data/processed/val_smoke.npz --out_dir outputs/figures
python -m src.visualization.plot_clusters --data data/processed/val_smoke.npz --out_dir outputs/figures --n_clusters 5
scp thddy@192.168.35.17:'D:/data/vehicle_trajectory_project/processed/small/train_small.npz' data/processed/train_small.npz
scp thddy@192.168.35.17:'D:/data/vehicle_trajectory_project/processed/small/val_small.npz' data/processed/val_small.npz
python -m src.datasets.validate_processed --npz data/processed/train_small.npz
python -m src.datasets.validate_processed --npz data/processed/val_small.npz
python -m src.visualization.plot_trajectories --data data/processed/val_small.npz --predictions outputs/predictions --out_dir outputs/figures/av2_small --num_cases 10
python -m src.visualization.plot_errors --data data/processed/val_small.npz --predictions outputs/predictions --out_dir outputs/figures/av2_small
python -m src.visualization.plot_pca --data data/processed/val_small.npz --out_dir outputs/figures/av2_small
python -m src.visualization.plot_clusters --data data/processed/val_small.npz --out_dir outputs/figures/av2_small --n_clusters 5
pytest tests/test_analysis_phase13.py -q
python -m src.analysis.pca_analysis --train_data data/processed/train_smoke.npz --data data/processed/val_smoke.npz --out_dir outputs --n_components 12
python -m src.analysis.kmeans_analysis --train_data data/processed/train_smoke.npz --data data/processed/val_smoke.npz --predictions outputs/predictions --out_dir outputs --n_components 12 --n_clusters 5 --required_models linear
python -m src.analysis.error_analysis --data data/processed/val_smoke.npz --predictions outputs/predictions --out_dir outputs --top_k 10 --required_models linear
python -m src.analysis.pca_analysis --train_data data/processed/train_small.npz --data data/processed/val_small.npz --out_dir outputs/av2_small_analysis --n_components 12
python -m src.analysis.kmeans_analysis --train_data data/processed/train_small.npz --data data/processed/val_small.npz --predictions outputs/predictions/phase14_av2_small --out_dir outputs/av2_small_analysis --n_components 12 --n_clusters 5 --required_models linear lstm transformer diffusion_direct diffusion_pca
python -m src.analysis.error_analysis --data data/processed/val_small.npz --predictions outputs/predictions/phase14_av2_small --out_dir outputs/av2_small_analysis --top_k 10 --required_models linear lstm transformer diffusion_direct diffusion_pca
pytest tests/test_phase14_experiment_matrix.py tests/test_metrics.py -q
python scripts/run_all_evaluations.py --data data/processed/val_smoke.npz --out_dir /tmp/phase14_eval_smoke --models linear lstm transformer diffusion_direct diffusion_pca --checkpoint_dir outputs/checkpoints --batch_size 64 --data_split val_smoke --target_type synthetic_mixed --prediction_tag synthetic_smoke
pytest -q
ssh -o ConnectTimeout=12 thddy@192.168.35.17 'hostname && whoami'
ssh -o ConnectTimeout=12 thddy@192.168.35.17 'cmd /c "cd /d C:\Users\thddy\Documents\code\vehicle_trajectory_project && git status --short --branch && git pull --ff-only origin main && git rev-parse --short HEAD"'
ssh -o ConnectTimeout=12 thddy@192.168.35.17 'powershell -NoProfile -ExecutionPolicy Bypass -Command "& C:\Users\thddy\Miniconda3\Scripts\conda.exe run -n vehicle_traj python -c \"import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 0)\""'
ssh -o ConnectTimeout=12 thddy@192.168.35.17 'powershell -NoProfile -ExecutionPolicy Bypass -Command "Set-Location C:\Users\thddy\Documents\code\vehicle_trajectory_project; & C:\Users\thddy\Miniconda3\Scripts\conda.exe run -n vehicle_traj python -m pytest tests/test_phase14_experiment_matrix.py tests/test_metrics.py -q"'
ssh -o ConnectTimeout=12 thddy@192.168.35.17 'powershell -NoProfile -ExecutionPolicy Bypass -Command "Set-Location C:\Users\thddy\Documents\code\vehicle_trajectory_project; & C:\Users\thddy\Miniconda3\Scripts\conda.exe run -n vehicle_traj python scripts/run_all_evaluations.py --data D:/data/vehicle_trajectory_project/processed/small/val_small.npz --out_dir D:/runs/vehicle_trajectory_project/phase14_smoke --models linear lstm transformer diffusion_direct diffusion_pca --checkpoint_dir D:/runs/vehicle_trajectory_project/phase14_smoke/checkpoints --checkpoint_tag phase14_smoke --batch_size 64 --data_split val_small --target_type av2_focal_mixed --prediction_tag phase14_av2_small"'
scp thddy@192.168.35.17:'D:/runs/vehicle_trajectory_project/phase14_smoke/tables/model_comparison.csv' outputs/tables/model_comparison.csv
scp thddy@192.168.35.17:'D:/runs/vehicle_trajectory_project/phase14_smoke/tables/model_comparison.md' outputs/tables/model_comparison.md
scp thddy@192.168.35.17:'D:/runs/vehicle_trajectory_project/phase14_smoke/tables/*_metrics.csv' outputs/tables/
scp thddy@192.168.35.17:'D:/runs/vehicle_trajectory_project/phase14_smoke/metrics/*_metrics.json' outputs/metrics/
scp thddy@192.168.35.17:'D:/runs/vehicle_trajectory_project/phase14_smoke/predictions/phase14_av2_small/*.pkl' outputs/predictions/phase14_av2_small/
scp thddy@192.168.35.17:'D:/runs/vehicle_trajectory_project/phase14_smoke/checkpoints/best_lstm_phase14_smoke.pt' outputs/checkpoints/phase14_av2_small/
scp thddy@192.168.35.17:'D:/runs/vehicle_trajectory_project/phase14_smoke/checkpoints/best_transformer_phase14_smoke.pt' outputs/checkpoints/phase14_av2_small/
scp thddy@192.168.35.17:'D:/runs/vehicle_trajectory_project/phase14_smoke/checkpoints/best_diffusion_direct_phase14_smoke.pt' outputs/checkpoints/phase14_av2_small/
scp thddy@192.168.35.17:'D:/runs/vehicle_trajectory_project/phase14_smoke/checkpoints/best_diffusion_pca_phase14_smoke.pt' outputs/checkpoints/phase14_av2_small/
scp thddy@192.168.35.17:'D:/runs/vehicle_trajectory_project/phase14_smoke/checkpoints/pca_codec.pkl' outputs/checkpoints/phase14_av2_small/
python scripts/run_all_evaluations.py --data data/processed/val_small.npz --out_dir outputs --models linear lstm transformer diffusion_direct diffusion_pca --checkpoint_dir outputs/checkpoints/phase14_av2_small --checkpoint_tag phase14_smoke --batch_size 64 --data_split val_small --target_type av2_focal_mixed --prediction_tag phase14_av2_small
python -m src.analysis.kmeans_analysis --train_data data/processed/train_small.npz --data data/processed/val_small.npz --predictions outputs/predictions/phase14_av2_small --out_dir outputs/av2_small_analysis --n_components 12 --n_clusters 5 --required_models linear lstm transformer diffusion_direct diffusion_pca
python -m src.analysis.error_analysis --data data/processed/val_small.npz --predictions outputs/predictions/phase14_av2_small --out_dir outputs/av2_small_analysis --top_k 10 --required_models linear lstm transformer diffusion_direct diffusion_pca
pytest tests/test_full_pilot_configs.py tests/test_phase14_experiment_matrix.py -q
pytest -q
ssh -o ConnectTimeout=12 thddy@192.168.35.17 'powershell -NoProfile -ExecutionPolicy Bypass -File D:\runs\vehicle_trajectory_project\summarize_full_pilot_5epoch_status.ps1'
ssh -o ConnectTimeout=12 thddy@192.168.35.17 'powershell -NoProfile -ExecutionPolicy Bypass -Command "Set-Location C:\Users\thddy\Documents\code\vehicle_trajectory_project; & C:\Users\thddy\Miniconda3\Scripts\conda.exe run -n vehicle_traj python D:\runs\vehicle_trajectory_project\verify_full_pilot_5epoch_outputs.py"'
scp thddy@192.168.35.17:'D:/runs/vehicle_trajectory_project/full_pilot_5epoch/tables/model_comparison.csv' outputs/full_av2_5epoch_pilot/tables/
scp thddy@192.168.35.17:'D:/runs/vehicle_trajectory_project/full_pilot_5epoch/tables/model_comparison.md' outputs/full_av2_5epoch_pilot/tables/
scp thddy@192.168.35.17:'D:/runs/vehicle_trajectory_project/full_pilot_5epoch/tables/*_metrics.csv' outputs/full_av2_5epoch_pilot/tables/
scp thddy@192.168.35.17:'D:/runs/vehicle_trajectory_project/full_pilot_5epoch/metrics/*_metrics.json' outputs/full_av2_5epoch_pilot/metrics/
scp thddy@192.168.35.17:'D:/runs/vehicle_trajectory_project/full_pilot_5epoch/figures/*.png' outputs/full_av2_5epoch_pilot/figures/
ssh -o ConnectTimeout=12 thddy@192.168.35.17 'hostname && whoami'
ssh -o ConnectTimeout=12 thddy@192.168.35.17 'cmd /c "cd /d C:\Users\thddy\Documents\code\vehicle_trajectory_project && git status --short --branch && git rev-parse --short HEAD"'
ssh -o ConnectTimeout=12 thddy@192.168.35.17 'powershell -NoProfile -ExecutionPolicy Bypass -Command "Get-Content D:\runs\vehicle_trajectory_project\full_long_goal\status.json | ConvertFrom-Json | ConvertTo-Json -Depth 8; if (Test-Path D:\runs\vehicle_trajectory_project\full_long_goal\FULL_LONG_EXPERIMENTS_COMPLETE.txt) { Get-Content D:\runs\vehicle_trajectory_project\full_long_goal\FULL_LONG_EXPERIMENTS_COMPLETE.txt } else { ''NO_COMPLETE_MARKER'' }"'
ssh -o ConnectTimeout=12 thddy@192.168.35.17 'powershell -NoProfile -ExecutionPolicy Bypass -Command "Get-Content D:\runs\vehicle_trajectory_project\full_long_final\tables\model_comparison.csv"'
scp -o ConnectTimeout=12 thddy@192.168.35.17:'D:/runs/vehicle_trajectory_project/full_long_final/tables/*' outputs/full_av2/tables/
scp -o ConnectTimeout=12 thddy@192.168.35.17:'D:/runs/vehicle_trajectory_project/full_long_final/metrics/*' outputs/full_av2/metrics/
scp -o ConnectTimeout=12 thddy@192.168.35.17:'D:/runs/vehicle_trajectory_project/full_long_final/figures/*.png' outputs/full_av2/figures/
scp -o ConnectTimeout=12 thddy@192.168.35.17:'D:/runs/vehicle_trajectory_project/full_long_tuning/tables/*' outputs/full_av2_tuning/tables/
scp -o ConnectTimeout=12 thddy@192.168.35.17:'D:/runs/vehicle_trajectory_project/full_long_goal/status.json' outputs/full_av2/metadata/
scp -o ConnectTimeout=12 thddy@192.168.35.17:'D:/runs/vehicle_trajectory_project/full_long_goal/FULL_LONG_EXPERIMENTS_COMPLETE.txt' outputs/full_av2/metadata/
scp -o ConnectTimeout=12 thddy@192.168.35.17:'D:/data/vehicle_trajectory_project/processed/full/train_full.npz' data/processed/train_full.npz
scp -o ConnectTimeout=12 thddy@192.168.35.17:'D:/data/vehicle_trajectory_project/processed/full/val_full.npz' data/processed/val_full.npz
scp -o ConnectTimeout=12 thddy@192.168.35.17:'D:/runs/vehicle_trajectory_project/full_long_final/predictions/full_long_final/*.pkl' outputs/predictions/full_long_final/
python -m src.datasets.validate_processed --npz data/processed/train_full.npz
python -m src.datasets.validate_processed --npz data/processed/val_full.npz
python -m src.visualization.plot_trajectories --data data/processed/val_full.npz --predictions outputs/predictions/full_long_final --out_dir outputs/full_av2/figures --num_cases 12
python -m src.visualization.plot_errors --data data/processed/val_full.npz --predictions outputs/predictions/full_long_final --out_dir outputs/full_av2/figures
python -m src.visualization.plot_pca --data data/processed/val_full.npz --out_dir outputs/full_av2/figures --max_points 5000
python -m src.visualization.plot_clusters --data data/processed/val_full.npz --out_dir outputs/full_av2/figures --n_clusters 5 --max_points 5000
python -m src.visualization.plot_report_cases --data data/processed/val_full.npz --predictions outputs/predictions/full_long_final --out_dir outputs/full_av2/figures --reference_model lstm --diffusion_model diffusion_pca --num_cases 6
python -m src.analysis.pca_analysis --train_data data/processed/train_full.npz --data data/processed/val_full.npz --out_dir outputs/full_av2_analysis --n_components 12 --max_export_rows 5000
python -m src.analysis.kmeans_analysis --train_data data/processed/train_full.npz --data data/processed/val_full.npz --predictions outputs/predictions/full_long_final --out_dir outputs/full_av2_analysis --n_components 12 --n_clusters 5 --required_models linear lstm transformer diffusion_pca diffusion_direct
python -m src.analysis.error_analysis --data data/processed/val_full.npz --predictions outputs/predictions/full_long_final --out_dir outputs/full_av2_analysis --top_k 20 --required_models linear lstm transformer diffusion_pca diffusion_direct
python scripts/build_phase15_report.py --comparison outputs/full_av2/tables/model_comparison.csv --tuning_summary outputs/full_av2_tuning/tables/diffusion_tuning_summary.csv --out outputs/report_summary.md --analysis_dir outputs/full_av2_analysis --figures_dir outputs/full_av2/figures
```

Result:

```text
HOME
home\thddy
HOME LAN SSH: verified
HOME Tailscale SSH: timed out on 2026-06-03; re-verify before use
Python 3.12.7
C:\Users\thddy\AppData\Local\Programs\Python\Python312\python.exe
NVIDIA GeForce RTX 2070 SUPER, 591.86, 8192 MiB
HOME C drive free space: about 192 GB
HOME conda: not found on PATH
HOME s5cmd: missing at C:\Users\thddy\bin\s5cmd\s5cmd.exe
HOME AV2 archive inventory under D:\datasets\argoverse: train.tar, val.tar, test.tar, av2_mf_focal_test_annotations.parquet, av2_mf_multi_test_annotations.parquet
HOME D drive free space after completed AV2 layout check: about 776 GB
HOME D:\data contains av2 and vehicle_trajectory_project directories
HOME AV2 raw standard path: D:\data\av2\motion-forecasting
HOME AV2 raw path contains train, val, test, archives, test annotation parquet files, and DATA_READY_FOR_PHASE11.txt
HOME AV2 organizer script: D:\data\av2\organize_phase11_av2.ps1
HOME AV2 organizer scheduled task: VehicleTrajectoryAV2Organize
HOME AV2 organizer background log: D:\data\av2\logs\phase11_data_organize_bg_20260602_221953.log
HOME AV2 organizer status: completed successfully on HOME.
HOME AV2 Phase 11 marker: D:\data\av2\motion-forecasting\DATA_READY_FOR_PHASE11.txt exists.
HOME AV2 ready marker completed timestamp: 2026-06-03T08:18:02.9223133+09:00
HOME AV2 source archive path: D:\datasets\argoverse
HOME AV2 split inventory: train scenario parquet files 199892, val scenario parquet files 24988, test scenario parquet files 24984
HOME AV2 map inventory: train map json files 199892, val map json files 24988, test map json files 24984
HOME standard processed path prepared: D:\data\vehicle_trajectory_project\processed with small, full, metadata, and cache subdirectories
HOME standard runs path prepared: D:\runs\vehicle_trajectory_project with checkpoints, logs, metrics, predictions, figures, and remote_runs subdirectories
HOME Windows code root prepared: C:\Users\thddy\Documents\code\vehicle_trajectory_project
pytest: 5 passed
get_device: mps
song s5cmd: v2.3.0-991c9fb at C:\Users\thddy\bin\s5cmd\s5cmd.exe
song AV2 partial state: annotation parquet files downloaded; test split partially created; train/val not downloaded; INCOMPLETE_DOWNLOAD.txt marker written
song host latest check: only C: is exposed over SSH; D:\data is not available there
song stop verification: taskkill terminated s5cmd.exe PID 27984; later tasklist found no s5cmd.exe; VehicleTrajectoryAV2Download task not present
Phase 1 synthetic generator: created ignored train_smoke.npz, val_smoke.npz, and test_smoke.npz
Phase 1 tests: tests/test_synthetic_data.py passed, full pytest passed 10 tests
Phase 2 geometry: relative/global transforms and wrap_angle tests passed; full pytest passed 16 tests
Phase 2 subagent review: no coordinate transform issues found
Phase 3 dataset/dataloader: synthetic generation passed, tests/test_synthetic_data.py passed 7 tests, full pytest passed 18 tests
Phase 4 metrics: tests/test_metrics.py passed 14 tests, full pytest passed 32 tests
Phase 4 subagent review: mask shape/all-false mask findings were fixed before commit
Phase 5 linear baseline: evaluation CLI produced ADE, FDE, Miss Rate, Latency, and Parameters on val_smoke; full pytest passed 33 tests
Phase 6 common training: tiny_regressor smoke training completed for 1 epoch on CPU; best/last checkpoint, train log, and val metrics were generated
Phase 6 tests: tests/test_losses.py passed 5 tests, metrics/loss tests passed 22 tests, full pytest passed 41 tests
Phase 6 checkpoint audit: best_lstm_smoke.pt includes model/optimizer state, trainer_config, and metadata with device=cpu and architecture=tiny_regressor
Phase 6 subagent review: mask-aware FDE/endpoint, valid-step loss aggregation, and checkpoint metadata findings were fixed before commit
Phase 7 LSTM: tests/test_models_shape.py passed 2 tests; 1-epoch synthetic smoke training completed on CPU; LSTM checkpoint evaluation produced ADE, FDE, Miss Rate, Latency, and Parameters; full pytest passed 43 tests
Phase 8 Transformer: tests/test_models_shape.py passed 4 tests; 1-epoch synthetic smoke training completed on CPU; Transformer checkpoint evaluation produced ADE, FDE, Miss Rate, Latency, and Parameters; full pytest passed 45 tests
Phase 9 direct diffusion: tests/test_diffusion_step.py passed 5 tests; 1-epoch synthetic smoke training completed on CPU; diffusion checkpoint evaluation produced ADE/FDE and minADE/minFDE with 4 samples; full pytest passed 50 tests
Phase 10 PCA latent diffusion: tests/test_pca_latent.py passed 3 tests; PCA codec fit on train_smoke and wrote ignored outputs/checkpoints/pca_codec.pkl; PCA explained variance figure generated; diffusion_pca 1-epoch synthetic smoke training completed on CPU; full pytest passed 53 tests
Phase 11 AV2 preprocessing implementation: tests/test_preprocess_av2.py passed 5 tests; full pytest passed 58 tests
Phase 11 subagent review: config loading, fixed future horizon, observed flag handling, and masked payload validation findings were fixed before Windows execution
Phase 11 GitHub source sync: Mac pushed commit 143dd6b, then pushed status commit 9561eae; HOME cloned/pulled main at 9561eae
Phase 11 Windows dependency check: numpy, pandas, pyarrow, joblib, tqdm, and pyyaml available on HOME global Python 3.12 after installing pyarrow
Phase 11 Windows small preprocessing: generated D:\data\vehicle_trajectory_project\processed\small\train_small.npz and val_small.npz
Phase 11 schema validation: train_small.npz passed with 95 samples / 95 scenarios; val_small.npz passed with 95 samples / 95 scenarios
Phase 11 output sizes: train_small.npz about 131,957 bytes; val_small.npz about 132,217 bytes; scaler.pkl about 445 bytes; metadata directory created
Phase 11 full preprocessing decision: technically unblocked after small validation, but not started in this turn; run full as a Windows background job only after reviewing small visualizations or just before real AV2 training
Phase 12 visualization implementation: added trajectory overlay, diffusion sample overlay, top-error case, ADE/FDE histogram, PCA trajectory space, and K-means cluster plotting scripts
Phase 12 tests: tests/test_visualization.py passed; full pytest passed 59 tests
Phase 12 synthetic smoke: generated trajectory_overlay_linear_lstm_transformer.png, trajectory_overlay_diffusion_samples.png, error_histogram_ade.png, error_histogram_fde.png, pca_trajectory_space.png, kmeans_clusters.png, and top_error_cases PNG files
Phase 12 small AV2 smoke: copied ignored train_small.npz/val_small.npz to Mac, validated both files, and generated the same figure set under outputs/figures/av2_small
Phase 12 visual QA: representative synthetic and small AV2 trajectory overlay PNGs were inspected and showed nonblank trajectory plots with readable layout
Phase 12 GitHub: visualization work pushed on main; use git log --oneline for the authoritative latest commit hash
Phase 13 analysis implementation: added common prediction/metric utilities, expanded PCA analysis, added K-means cluster analysis, added error summary/top-case analysis, and added configs/analysis.yaml
Phase 13 tests: tests/test_analysis_phase13.py passed 4 tests; full pytest passed 63 tests
Phase 13 subagent review: no train-data leakage found; prediction/data alignment, required-model checks, best-of-K minMiss Rate, and global masked ADE aggregation findings were fixed
Phase 13 synthetic smoke: generated outputs/tables/cluster_summary.csv, cluster_metrics.csv, error_summary.csv, pca_latent.csv, top_error_cases.csv, and refreshed PCA/K-means figures
Phase 13 small AV2 smoke: generated the same analysis outputs under outputs/av2_small_analysis using ignored Mac copies of train_small.npz and val_small.npz
Phase 13 result scope: the original Phase 13 committed small AV2 analysis tables were Linear-only; Phase 14 regenerated current small AV2 analysis tables with all five model prediction payloads under outputs/predictions/phase14_av2_small
Phase 14 tooling: scripts/run_all_evaluations.py writes model_comparison.csv/md and now requires explicit trainable-model checkpoints by default to avoid stale checkpoint use
Phase 14 tooling: prediction_tag archives evaluation payloads under outputs/predictions/<tag> so small AV2 analysis does not accidentally consume top-level synthetic prediction payloads
Phase 14 metric fix: diffusion minFDE evaluation is mask-aware; saved val_small results are numerically unaffected because val_small has full future masks
Phase 14 tests: tests/test_phase14_experiment_matrix.py and tests/test_metrics.py passed 23 tests on Mac and Windows; full Mac pytest passed 70 tests
Phase 14 Windows environment: Miniconda3 and vehicle_traj were created on HOME; vehicle_traj reports torch 2.11.0+cu128, torch.cuda.is_available() True, NVIDIA GeForce RTX 2070 SUPER
Phase 14 Windows repo sync: HOME pulled main to a4f77bd before the final strict small AV2 evaluation rerun
Phase 14 Windows small training validation metrics: LSTM ADE 10.734397888183594 / FDE 21.01107406616211; Transformer ADE 10.198661804199219 / FDE 19.419828414916992; Direct Diffusion ADE 11.003629684448242 / FDE 21.149229049682617; PCA Diffusion ADE 7.114823818206787 / FDE 13.674212455749512
Phase 14 Mac final strict val_small comparison: Linear ADE 1.462841272354126 / FDE 3.491798162460327; LSTM ADE 10.734396934509277 / FDE 21.011072158813477; Transformer ADE 10.198661804199219 / FDE 19.419828414916992; Direct Diffusion ADE 10.946355819702148 / FDE 21.14156723022461 / minADE 10.79123306274414 / minFDE 20.124284744262695; PCA Diffusion ADE 7.103316307067871 / FDE 13.77344036102295 / minADE 6.917838096618652 / minFDE 12.924206733703613
Phase 14 Mac evaluation note: loading the Windows-trained PCA codec on Mac emitted a scikit-learn version warning, but evaluation completed; keep the copied codec/checkpoints ignored and do not commit them
Phase 14 all-model analysis: outputs/av2_small_analysis/tables/cluster_metrics.csv and error_summary.csv were regenerated from tagged phase14_av2_small prediction payloads with required_models linear, lstm, transformer, diffusion_direct, diffusion_pca
Phase 14 subagent review: stale checkpoint defaults, prediction provenance, mask-aware minFDE, permissive missing-model handling, and stale PROJECT_STATUS findings were addressed before final Phase 14 completion
Phase 14 result scope: complete for small AV2 smoke matrix; full AV2 Stage F1-F3 are also complete as optional follow-up pilot work, but report-ready 5-epoch/long-run results have not been run and should not be claimed
Full AV2 Stage F3 configs/tooling: committed in 8c70cf1 with dedicated 1-epoch configs and Windows background/status scripts
Full AV2 Stage F3 Windows run: HOME pulled 8c70cf1, verified CUDA torch 2.11.0+cu128 on NVIDIA GeForce RTX 2070 SUPER, and ran scripts/windows_full_pilot_1epoch.ps1 as scheduled task VehicleTrajectoryFullPilot1Epoch
Full AV2 Stage F3 validation: Windows verification confirmed complete status, CUDA checkpoint metadata, finite ADE/FDE, exactly 1 epoch for each trainable model, PCA codec existence, and model_comparison.csv rows for Linear/LSTM/Transformer/PCA Diffusion/Direct Diffusion
Full AV2 Stage F3 val_full metrics: Linear ADE 1.5324182510375977 / FDE 3.7973430156707764; LSTM ADE 1.3740959167480469 / FDE 3.951328992843628; Transformer ADE 1.0867810249328613 / FDE 2.707834482192993; PCA Diffusion ADE 6.28801155090332 / FDE 12.30606460571289 / minADE 6.107557773590088 / minFDE 11.599884986877441; Direct Diffusion ADE 10.117935180664062 / FDE 19.455183029174805 / minADE 9.942566871643066 / minFDE 18.461170196533203
Full AV2 Stage F3 result artifacts copied to Mac: outputs/full_av2_pilot/tables/model_comparison.csv, outputs/full_av2_pilot/tables/model_comparison.md, and lightweight metrics CSV/JSON files. Checkpoints, logs, prediction payloads, and full .npz files remain Windows-local or ignored.
Full AV2 Stage F4 configs/tooling: committed in 8014ed3 with dedicated 5-epoch configs and Windows background/status scripts
Full AV2 Stage F4 Windows run: HOME pulled 8014ed3, verified CUDA torch 2.11.0+cu128 on NVIDIA GeForce RTX 2070 SUPER, and ran scripts/windows_full_pilot_5epoch.ps1 as scheduled task VehicleTrajectoryFullPilot5Epoch
Full AV2 Stage F4 validation: Windows verification confirmed complete status, CUDA checkpoint metadata, finite ADE/FDE, exactly 5 epochs for each trainable model, PCA codec/figures existence, and model_comparison.csv rows for Linear/LSTM/Transformer/PCA Diffusion/Direct Diffusion
Full AV2 Stage F4 val_full metrics: Linear ADE 1.5324182510375977 / FDE 3.7973430156707764; LSTM ADE 1.0401222705841064 / FDE 2.6033456325531006; Transformer ADE 0.9415797591209412 / FDE 2.416048526763916; PCA Diffusion ADE 6.277219772338867 / FDE 12.322331428527832 / minADE 6.096756935119629 / minFDE 11.606433868408203; Direct Diffusion ADE 10.058550834655762 / FDE 19.448150634765625 / minADE 9.881662368774414 / minFDE 18.506914138793945
Full AV2 Stage F4 result artifacts copied to Mac: outputs/full_av2_5epoch_pilot/tables/model_comparison.csv, outputs/full_av2_5epoch_pilot/tables/model_comparison.md, lightweight metrics CSV/JSON files, and PCA figures. Checkpoints, logs, prediction payloads, and full .npz files remain Windows-local or ignored.
Full AV2 Stage F4 decision note: LSTM and Transformer improved versus Stage F3 and beat Linear on val_full after 5 epochs; Transformer is currently the strongest pilot model. Diffusion variants remain much worse than Linear and should be tuned before any expensive long Diffusion run.
Full AV2 Stage F5A tuning gate: selected PCA Diffusion pca_b and Direct Diffusion direct_f; both passed the user target gates with full_run_ready=true.
Full AV2 Stage F5B Windows run: HOME repo was clean on main at 1e511e3; status.json reported status=complete and exit_code=0; FULL_LONG_EXPERIMENTS_COMPLETE.txt exists.
Full AV2 Stage F5B final comparison: LSTM is best single-trajectory ADE/FDE at ADE 0.7916645407676697 / FDE 2.0570261478424072. PCA Diffusion has the best best-of-K min metrics at minADE 0.41746464371681213 / minFDE 0.8582534193992615. Direct Diffusion final metrics are ADE 1.9281558990478516 / FDE 4.91142463684082 / minADE 0.5009313821792603 / minFDE 0.9845224618911743.
Phase 15 Mac integration: train_full.npz and val_full.npz validated on Mac with 189,541 train samples and 23,706 val samples; final figures, analysis tables, and outputs/report_summary.md were generated from real full AV2 outputs.
```

## Open External Requirements

```text
HOME AV2 raw data is complete at D:\data\av2\motion-forecasting.
HOME DATA_READY_FOR_PHASE11.txt exists and split counts have been verified.
HOME small processed AV2 data is available at D:\data\vehicle_trajectory_project\processed\small.
Mac has ignored lightweight copies of train_small.npz and val_small.npz for visualization/analysis smoke checks.
HOME needs s5cmd only for future direct AV2 download or resync; current archives were already downloaded manually under D:\datasets\argoverse.
HOME Miniconda3 and the vehicle_traj CUDA PyTorch environment are available for GPU training.
Full AV2 preprocessing, schema validation, 1-epoch pilot, 5-epoch pilot, F5A tuning gate, F5B all-model long run, and Phase 15 report asset generation are complete.
Mac currently has ignored local copies of train_full.npz, val_full.npz, and full_long_final prediction payloads only for analysis reproducibility; do not commit them.
For full AV2 preprocessing, long AV2 download/extraction, or GPU training attempts, do not use a long foreground SSH command; use the safe remote execution rule in docs/windows_gpu_training_only_workflow.md.
```

## GitHub

```text
Repository: https://github.com/thddydgnl/vehicle-trajectory-prediction-av2
Remote: origin
Default branch: main
Latest pushed branch: main
Latest pushed commit: use git log --oneline -1 for the authoritative current commit
```

## Update Rule

After every completed Phase, update:

```text
1. Phase Progress
2. Next Recommended Task
3. Latest Verified Commands
4. Open External Requirements
```

Do not mark a Phase complete unless its required validation command ran and the
result is recorded.
