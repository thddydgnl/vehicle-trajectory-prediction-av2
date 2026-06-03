param(
  [switch]$SkipFinalDirectDiffusion
)

$ErrorActionPreference = "Stop"

$Repo = "C:\Users\thddy\Documents\code\vehicle_trajectory_project"
$Conda = "C:\Users\thddy\Miniconda3\Scripts\conda.exe"
$TrainData = "D:\data\vehicle_trajectory_project\processed\full\train_full.npz"
$ValData = "D:\data\vehicle_trajectory_project\processed\full\val_full.npz"
$TuningRunDir = "D:\runs\vehicle_trajectory_project\full_long_tuning"
$FinalRunDir = "D:\runs\vehicle_trajectory_project\full_long_final"
$RunRoot = "D:\runs\vehicle_trajectory_project\full_long_goal"
$LogDir = Join-Path $RunRoot "logs"
$StatusPath = Join-Path $RunRoot "status.json"
$LatestLogPath = Join-Path $RunRoot "latest_full_long_log.txt"
$CompleteMarker = Join-Path $RunRoot "FULL_LONG_EXPERIMENTS_COMPLETE.txt"
$FailedMarker = Join-Path $RunRoot "FULL_LONG_EXPERIMENTS_FAILED.txt"

New-Item -ItemType Directory -Force $RunRoot, $LogDir, $TuningRunDir, $FinalRunDir | Out-Null
Remove-Item -Force $CompleteMarker, $FailedMarker -ErrorAction SilentlyContinue

$Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$Log = Join-Path $LogDir "full_long_experiments_$Timestamp.log"
Set-Content -Path $LatestLogPath -Value $Log -Encoding UTF8

function Get-RepoHead {
  try {
    Push-Location $Repo
    $Head = (& git rev-parse --short HEAD).Trim()
    Pop-Location
    return $Head
  } catch {
    return "unknown"
  }
}

function Test-Artifact {
  param([string]$Path)
  return Test-Path $Path
}

function Write-Status {
  param(
    [string]$Status,
    [string]$Step,
    [int]$ExitCode = -1,
    [string]$Message = ""
  )
  $Artifacts = [ordered]@{
    tuning_pca_summary = Test-Artifact (Join-Path $TuningRunDir "tables\diffusion_tuning_summary.csv")
    tuning_selection = Test-Artifact (Join-Path $TuningRunDir "tables\selected_diffusion_configs.json")
    final_linear_metrics = Test-Artifact (Join-Path $FinalRunDir "metrics\linear_val_metrics.json")
    final_lstm_best = Test-Artifact (Join-Path $FinalRunDir "checkpoints\best_lstm_full_long.pt")
    final_lstm_metrics = Test-Artifact (Join-Path $FinalRunDir "metrics\lstm_full_long_val_metrics.json")
    final_transformer_best = Test-Artifact (Join-Path $FinalRunDir "checkpoints\best_transformer_full_long.pt")
    final_transformer_metrics = Test-Artifact (Join-Path $FinalRunDir "metrics\transformer_full_long_val_metrics.json")
    final_pca_codec = Test-Artifact (Join-Path $FinalRunDir "checkpoints\pca_codec.pkl")
    final_diffusion_pca_best = Test-Artifact (Join-Path $FinalRunDir "checkpoints\best_diffusion_pca_full_long.pt")
    final_diffusion_pca_metrics = Test-Artifact (Join-Path $FinalRunDir "metrics\diffusion_pca_full_long_val_metrics.json")
    final_diffusion_direct_best = Test-Artifact (Join-Path $FinalRunDir "checkpoints\best_diffusion_direct_full_long.pt")
    final_diffusion_direct_metrics = Test-Artifact (Join-Path $FinalRunDir "metrics\diffusion_direct_full_long_val_metrics.json")
    final_model_comparison_csv = Test-Artifact (Join-Path $FinalRunDir "tables\model_comparison.csv")
    final_model_comparison_md = Test-Artifact (Join-Path $FinalRunDir "tables\model_comparison.md")
  }
  $StatusObject = [ordered]@{
    status = $Status
    step = $Step
    updated_at = (Get-Date -Format o)
    exit_code = $ExitCode
    message = $Message
    repo = $Repo
    repo_head = Get-RepoHead
    train_data = $TrainData
    val_data = $ValData
    tuning_run_dir = $TuningRunDir
    final_run_dir = $FinalRunDir
    run_root = $RunRoot
    log = $Log
    skip_final_direct_diffusion = [bool]$SkipFinalDirectDiffusion
    artifacts = $Artifacts
  }
  $StatusObject | ConvertTo-Json -Depth 5 | Set-Content -Path $StatusPath -Encoding UTF8
}

function Invoke-CondaPython {
  param(
    [string]$Step,
    [string[]]$PythonArgs
  )
  Write-Status -Status "running" -Step $Step -Message "started"
  Add-Content -Path $Log -Value ""
  Add-Content -Path $Log -Value "[$(Get-Date -Format o)] START $Step"
  & $Conda run --no-capture-output -n vehicle_traj python @PythonArgs 2>&1 | Tee-Object -FilePath $Log -Append
  $ExitCode = $LASTEXITCODE
  if ($ExitCode -ne 0) {
    throw "$Step failed with exit code $ExitCode"
  }
  Add-Content -Path $Log -Value "[$(Get-Date -Format o)] END $Step"
  Write-Status -Status "running" -Step $Step -ExitCode 0 -Message "completed"
}

function Invoke-DiffusionCandidate {
  param(
    [string]$ModelKey,
    [string]$CandidateId,
    [string]$ConfigPath,
    [string]$CheckpointPath
  )
  Invoke-CondaPython "train_$CandidateId" @(
    "-m", "src.training.train",
    "--config", $ConfigPath,
    "--data", $TrainData,
    "--val_data", $ValData
  )

  $EvalOut = Join-Path $TuningRunDir "evaluations\$CandidateId"
  if ($ModelKey -eq "diffusion_pca") {
    Invoke-CondaPython "eval_$CandidateId" @(
      "scripts/run_all_evaluations.py",
      "--data", $ValData,
      "--out_dir", $EvalOut,
      "--models", "diffusion_pca",
      "--diffusion_pca_checkpoint", $CheckpointPath,
      "--batch_size", "64",
      "--data_split", "val_full",
      "--target_type", "av2_focal_mixed",
      "--prediction_tag", "full_long_tuning_$CandidateId"
    )
  } else {
    Invoke-CondaPython "eval_$CandidateId" @(
      "scripts/run_all_evaluations.py",
      "--data", $ValData,
      "--out_dir", $EvalOut,
      "--models", "diffusion_direct",
      "--diffusion_direct_checkpoint", $CheckpointPath,
      "--batch_size", "64",
      "--data_split", "val_full",
      "--target_type", "av2_focal_mixed",
      "--prediction_tag", "full_long_tuning_$CandidateId"
    )
  }
}

try {
  Write-Status -Status "running" -Step "preflight" -Message "Full long experiments started"
  Add-Content -Path $Log -Value "[$(Get-Date -Format o)] Starting full long experiments"
  Add-Content -Path $Log -Value "Repo: $Repo"
  Add-Content -Path $Log -Value "TrainData: $TrainData"
  Add-Content -Path $Log -Value "ValData: $ValData"
  Add-Content -Path $Log -Value "TuningRunDir: $TuningRunDir"
  Add-Content -Path $Log -Value "FinalRunDir: $FinalRunDir"

  if (-not (Test-Path $Repo)) { throw "Repo not found: $Repo" }
  if (-not (Test-Path $Conda)) { throw "Conda not found: $Conda" }
  if (-not (Test-Path $TrainData)) { throw "Missing train data: $TrainData" }
  if (-not (Test-Path $ValData)) { throw "Missing val data: $ValData" }

  Set-Location $Repo
  & git status --short --branch 2>&1 | Tee-Object -FilePath $Log -Append
  & git rev-parse --short HEAD 2>&1 | Tee-Object -FilePath $Log -Append

  Invoke-CondaPython "cuda_preflight" @(
    "-c",
    "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no cuda'); raise SystemExit(0 if torch.cuda.is_available() else 1)"
  )

  Invoke-CondaPython "tuning_pca_codec_fit" @(
    "-m", "src.analysis.pca_analysis",
    "--train_data", $TrainData,
    "--data", $ValData,
    "--out_dir", $TuningRunDir,
    "--n_components", "12",
    "--max_export_rows", "5000"
  )

  Invoke-DiffusionCandidate "diffusion_pca" "pca_a" "configs/full_tune_diffusion_pca_a.yaml" (Join-Path $TuningRunDir "checkpoints\best_diffusion_pca_tune_a.pt")
  Invoke-DiffusionCandidate "diffusion_pca" "pca_b" "configs/full_tune_diffusion_pca_b.yaml" (Join-Path $TuningRunDir "checkpoints\best_diffusion_pca_tune_b.pt")
  Invoke-DiffusionCandidate "diffusion_pca" "pca_c" "configs/full_tune_diffusion_pca_c.yaml" (Join-Path $TuningRunDir "checkpoints\best_diffusion_pca_tune_c.pt")
  Invoke-DiffusionCandidate "diffusion_direct" "direct_a" "configs/full_tune_diffusion_direct_a.yaml" (Join-Path $TuningRunDir "checkpoints\best_diffusion_direct_tune_a.pt")
  Invoke-DiffusionCandidate "diffusion_direct" "direct_b" "configs/full_tune_diffusion_direct_b.yaml" (Join-Path $TuningRunDir "checkpoints\best_diffusion_direct_tune_b.pt")
  Invoke-DiffusionCandidate "diffusion_direct" "direct_c" "configs/full_tune_diffusion_direct_c.yaml" (Join-Path $TuningRunDir "checkpoints\best_diffusion_direct_tune_c.pt")

  Invoke-CondaPython "select_diffusion_tuning" @(
    "scripts/select_diffusion_tuning.py",
    "--matrix", "configs/full_diffusion_tuning_matrix.yaml",
    "--repo_root", $Repo
  )

  Invoke-CondaPython "final_linear_evaluation" @(
    "scripts/run_all_evaluations.py",
    "--data", $ValData,
    "--out_dir", $FinalRunDir,
    "--models", "linear",
    "--batch_size", "64",
    "--data_split", "val_full",
    "--target_type", "av2_focal_mixed",
    "--prediction_tag", "full_long_final"
  )

  Invoke-CondaPython "final_lstm_training" @(
    "-m", "src.training.train",
    "--config", "configs/full_long_lstm.yaml",
    "--data", $TrainData,
    "--val_data", $ValData
  )

  Invoke-CondaPython "final_transformer_training" @(
    "-m", "src.training.train",
    "--config", "configs/full_long_transformer.yaml",
    "--data", $TrainData,
    "--val_data", $ValData
  )

  Invoke-CondaPython "final_pca_codec_fit" @(
    "-m", "src.analysis.pca_analysis",
    "--train_data", $TrainData,
    "--data", $ValData,
    "--out_dir", $FinalRunDir,
    "--n_components", "12",
    "--max_export_rows", "5000"
  )

  $PcaFinalConfig = Join-Path $FinalRunDir "generated_configs\full_long_diffusion_pca.yaml"
  $DirectFinalConfig = Join-Path $FinalRunDir "generated_configs\full_long_diffusion_direct.yaml"

  Invoke-CondaPython "final_diffusion_pca_training" @(
    "-m", "src.training.train",
    "--config", $PcaFinalConfig,
    "--data", $TrainData,
    "--val_data", $ValData
  )

  if (-not $SkipFinalDirectDiffusion) {
    Invoke-CondaPython "final_diffusion_direct_training" @(
      "-m", "src.training.train",
      "--config", $DirectFinalConfig,
      "--data", $TrainData,
      "--val_data", $ValData
    )
  }

  $EvalModels = @("linear", "lstm", "transformer", "diffusion_pca")
  if (-not $SkipFinalDirectDiffusion) {
    $EvalModels += "diffusion_direct"
  }

  $FinalEvalArgs = @(
    "scripts/run_all_evaluations.py",
    "--data", $ValData,
    "--out_dir", $FinalRunDir,
    "--models"
  ) + $EvalModels + @(
    "--checkpoint_dir", (Join-Path $FinalRunDir "checkpoints"),
    "--checkpoint_tag", "full_long",
    "--batch_size", "64",
    "--data_split", "val_full",
    "--target_type", "av2_focal_mixed",
    "--prediction_tag", "full_long_final"
  )
  Invoke-CondaPython "final_comparison_evaluation" $FinalEvalArgs

  Set-Content -Path $CompleteMarker -Value "Completed full long experiments at $(Get-Date -Format o). Log: $Log" -Encoding UTF8
  Write-Status -Status "complete" -Step "complete" -ExitCode 0 -Message "Full long experiments complete"
  exit 0
} catch {
  $Message = $_.Exception.Message
  Add-Content -Path $Log -Value "[$(Get-Date -Format o)] FAILED: $Message"
  Set-Content -Path $FailedMarker -Value "Failed full long experiments at $(Get-Date -Format o). Log: $Log. Error: $Message" -Encoding UTF8
  Write-Status -Status "failed" -Step "failed" -ExitCode 1 -Message $Message
  exit 1
}
