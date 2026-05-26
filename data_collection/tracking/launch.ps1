$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$configPath = Join-Path $scriptDir "config\default.yaml"

$candidateExecutables = @(
    (Join-Path $scriptDir "build\Release\read_franka_ee_pose.exe"),
    (Join-Path $scriptDir "build\read_franka_ee_pose.exe"),
    (Join-Path $scriptDir "build\Release\read_franka_ee_pose"),
    (Join-Path $scriptDir "build\read_franka_ee_pose"),
    (Join-Path $scriptDir "src\read_franka_ee_pose.exe"),
    (Join-Path $scriptDir "src\read_franka_ee_pose")
)

$binaryPath = $candidateExecutables | Where-Object { Test-Path $_ } | Select-Object -First 1

if (-not $binaryPath) {
    throw "Cannot find read_franka_ee_pose binary. Checked build/, build/Release/, and src/."
}

if (-not (Test-Path $configPath)) {
    throw "Cannot find default config file: $configPath"
}

Write-Host "[INFO] Launching: $binaryPath"
Write-Host "[INFO] Config: $configPath"

& $binaryPath $configPath
