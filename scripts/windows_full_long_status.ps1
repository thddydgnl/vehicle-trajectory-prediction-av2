$ErrorActionPreference = "Continue"

$RunRoot = "D:\runs\vehicle_trajectory_project\full_long_goal"
$TuningRunDir = "D:\runs\vehicle_trajectory_project\full_long_tuning"
$FinalRunDir = "D:\runs\vehicle_trajectory_project\full_long_final"
$StatusPath = Join-Path $RunRoot "status.json"
$LatestLogPath = Join-Path $RunRoot "latest_full_long_log.txt"

Write-Host "STATUS"
if (Test-Path $StatusPath) {
  Get-Content $StatusPath
} else {
  Write-Host "missing status.json"
}

Write-Host ""
Write-Host "LATEST_LOG"
if (Test-Path $LatestLogPath) {
  $Log = Get-Content $LatestLogPath -ErrorAction SilentlyContinue
  Write-Host $Log
  if (Test-Path $Log) {
    Get-Item $Log | Select-Object FullName,Length,LastWriteTime | ConvertTo-Json -Depth 3
    Write-Host "LOG_TAIL"
    Get-Content $Log -Tail 80
  }
} else {
  Write-Host "missing latest log pointer"
}

Write-Host ""
Write-Host "TUNING_SUMMARY"
$TuningSummary = Join-Path $TuningRunDir "tables\diffusion_tuning_summary.csv"
if (Test-Path $TuningSummary) {
  Get-Content $TuningSummary
} else {
  Write-Host "missing diffusion_tuning_summary.csv"
}

Write-Host ""
Write-Host "FINAL_COMPARISON"
$FinalComparison = Join-Path $FinalRunDir "tables\model_comparison.csv"
if (Test-Path $FinalComparison) {
  Get-Content $FinalComparison
} else {
  Write-Host "missing final model_comparison.csv"
}

Write-Host ""
Write-Host "PROCESSES"
Get-CimInstance Win32_Process |
  Where-Object { $_.CommandLine -match "full_long|full_tune|src.training.train|run_all_evaluations|pca_analysis|select_diffusion_tuning" } |
  Select-Object ProcessId,CommandLine |
  ConvertTo-Json -Depth 3
