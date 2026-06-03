$ErrorActionPreference = "Continue"

$RunDir = "D:\runs\vehicle_trajectory_project\full_pilot_1epoch"
$StatusPath = Join-Path $RunDir "status.json"
$LatestLogPath = Join-Path $RunDir "latest_full_pilot_log.txt"

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
    Get-Content $Log -Tail 40
  }
} else {
  Write-Host "missing latest log pointer"
}

Write-Host ""
Write-Host "ARTIFACTS"
if (Test-Path $RunDir) {
  Get-ChildItem $RunDir -Recurse -File -ErrorAction SilentlyContinue |
    Where-Object { $_.FullName -match "\\(metrics|tables|checkpoints|logs)\\" } |
    Select-Object FullName,Length,LastWriteTime |
    Sort-Object FullName |
    ConvertTo-Json -Depth 3
}

Write-Host ""
Write-Host "PROCESSES"
Get-CimInstance Win32_Process |
  Where-Object { $_.CommandLine -match "full_pilot_1epoch|src.training.train|run_all_evaluations|pca_analysis" } |
  Select-Object ProcessId,CommandLine |
  ConvertTo-Json -Depth 3
