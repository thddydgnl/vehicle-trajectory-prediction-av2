param(
  [switch]$SkipDirectDiffusion
)

$ErrorActionPreference = "Stop"

$Repo = "C:\Users\thddy\Documents\code\vehicle_trajectory_project"
$Conda = "C:\Users\thddy\Miniconda3\Scripts\conda.exe"
$TrainData = "D:\data\vehicle_trajectory_project\processed\full\train_full.npz"
$ValData = "D:\data\vehicle_trajectory_project\processed\full\val_full.npz"
$RunDir = "D:\runs\vehicle_trajectory_project\full_pilot_5epoch"
$LogDir = Join-Path $RunDir "logs"
$StatusPath = Join-Path $RunDir "status.json"
$LatestLogPath = Join-Path $RunDir "latest_full_pilot_log.txt"
$CompleteMarker = Join-Path $RunDir "FULL_PILOT_5EPOCH_COMPLETE.txt"
$FailedMarker = Join-Path $RunDir "FULL_PILOT_5EPOCH_FAILED.txt"

New-Item -ItemType Directory -Force $RunDir, $LogDir | Out-Null
Remove-Item -Force $CompleteMarker, $FailedMarker -ErrorAction SilentlyContinue

$Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$Log = Join-Path $LogDir "full_pilot_5epoch_$Timestamp.log"
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
    linear_metrics = Test-Artifact (Join-Path $RunDir "metrics\linear_val_metrics.json")
    lstm_best = Test-Artifact (Join-Path $RunDir "checkpoints\best_lstm_full_pilot_5epoch.pt")
    lstm_metrics = Test-Artifact (Join-Path $RunDir "metrics\lstm_full_pilot_5epoch_val_metrics.json")
    transformer_best = Test-Artifact (Join-Path $RunDir "checkpoints\best_transformer_full_pilot_5epoch.pt")
    transformer_metrics = Test-Artifact (Join-Path $RunDir "metrics\transformer_full_pilot_5epoch_val_metrics.json")
    pca_codec = Test-Artifact (Join-Path $RunDir "checkpoints\pca_codec.pkl")
    diffusion_pca_best = Test-Artifact (Join-Path $RunDir "checkpoints\best_diffusion_pca_full_pilot_5epoch.pt")
    diffusion_pca_metrics = Test-Artifact (Join-Path $RunDir "metrics\diffusion_pca_full_pilot_5epoch_val_metrics.json")
    diffusion_direct_best = Test-Artifact (Join-Path $RunDir "checkpoints\best_diffusion_direct_full_pilot_5epoch.pt")
    diffusion_direct_metrics = Test-Artifact (Join-Path $RunDir "metrics\diffusion_direct_full_pilot_5epoch_val_metrics.json")
    model_comparison_csv = Test-Artifact (Join-Path $RunDir "tables\model_comparison.csv")
    model_comparison_md = Test-Artifact (Join-Path $RunDir "tables\model_comparison.md")
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
    run_dir = $RunDir
    log = $Log
    skip_direct_diffusion = [bool]$SkipDirectDiffusion
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

try {
  Write-Status -Status "running" -Step "preflight" -Message "Full AV2 5-epoch pilot started"
  Add-Content -Path $Log -Value "[$(Get-Date -Format o)] Starting Full AV2 5-epoch pilot"
  Add-Content -Path $Log -Value "Repo: $Repo"
  Add-Content -Path $Log -Value "TrainData: $TrainData"
  Add-Content -Path $Log -Value "ValData: $ValData"
  Add-Content -Path $Log -Value "RunDir: $RunDir"

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

  Invoke-CondaPython "linear_evaluation" @(
    "scripts/run_all_evaluations.py",
    "--data", $ValData,
    "--out_dir", $RunDir,
    "--models", "linear",
    "--batch_size", "64",
    "--data_split", "val_full",
    "--target_type", "av2_focal_mixed",
    "--prediction_tag", "full_pilot_5epoch"
  )

  Invoke-CondaPython "lstm_training" @(
    "-m", "src.training.train",
    "--config", "configs/full_pilot_lstm_5epoch.yaml",
    "--data", $TrainData,
    "--val_data", $ValData
  )

  Invoke-CondaPython "transformer_training" @(
    "-m", "src.training.train",
    "--config", "configs/full_pilot_transformer_5epoch.yaml",
    "--data", $TrainData,
    "--val_data", $ValData
  )

  Invoke-CondaPython "pca_codec_fit" @(
    "-m", "src.analysis.pca_analysis",
    "--train_data", $TrainData,
    "--data", $ValData,
    "--out_dir", $RunDir,
    "--n_components", "12",
    "--max_export_rows", "5000"
  )

  Invoke-CondaPython "diffusion_pca_training" @(
    "-m", "src.training.train",
    "--config", "configs/full_pilot_diffusion_pca_5epoch.yaml",
    "--data", $TrainData,
    "--val_data", $ValData
  )

  if (-not $SkipDirectDiffusion) {
    Invoke-CondaPython "diffusion_direct_training" @(
      "-m", "src.training.train",
      "--config", "configs/full_pilot_diffusion_direct_5epoch.yaml",
      "--data", $TrainData,
      "--val_data", $ValData
    )
  }

  $EvalModels = @("linear", "lstm", "transformer", "diffusion_pca")
  if (-not $SkipDirectDiffusion) {
    $EvalModels += "diffusion_direct"
  }

  $FinalEvalArgs = @(
    "scripts/run_all_evaluations.py",
    "--data", $ValData,
    "--out_dir", $RunDir,
    "--models"
  ) + $EvalModels + @(
    "--checkpoint_dir", (Join-Path $RunDir "checkpoints"),
    "--checkpoint_tag", "full_pilot_5epoch",
    "--batch_size", "64",
    "--data_split", "val_full",
    "--target_type", "av2_focal_mixed",
    "--prediction_tag", "full_pilot_5epoch"
  )
  Invoke-CondaPython "final_comparison_evaluation" $FinalEvalArgs

  Set-Content -Path $CompleteMarker -Value "Completed Full AV2 5-epoch pilot at $(Get-Date -Format o). Log: $Log" -Encoding UTF8
  Write-Status -Status "complete" -Step "complete" -ExitCode 0 -Message "Full AV2 5-epoch pilot complete"
  exit 0
} catch {
  $Message = $_.Exception.Message
  Add-Content -Path $Log -Value "[$(Get-Date -Format o)] FAILED: $Message"
  Set-Content -Path $FailedMarker -Value "Failed Full AV2 5-epoch pilot at $(Get-Date -Format o). Log: $Log. Error: $Message" -Encoding UTF8
  Write-Status -Status "failed" -Step "failed" -ExitCode 1 -Message $Message
  exit 1
}
