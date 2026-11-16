param([Parameter(Mandatory=$true)][string]$SpecFile)
$ErrorActionPreference="Stop"
$projectRoot=Split-Path $PSScriptRoot -Parent
Set-Location $projectRoot
$spec=Get-Content (Join-Path $projectRoot $SpecFile) -Raw -Encoding UTF8 | ConvertFrom-Json
$log=Join-Path $projectRoot $spec.host_log
$arguments=@{Mode=$spec.mode;Task="task3";Epochs=[int]$spec.epochs;Minutes=[int]$spec.minutes;Config=$spec.config}
if ($spec.run_directory) {$arguments.RunDirectory=$spec.run_directory}
@{monitor_pid=$PID;identity=[Security.Principal.WindowsIdentity]::GetCurrent().Name;started=(Get-Date).ToString("o")} |
    ConvertTo-Json | Set-Content (Join-Path $projectRoot $spec.host_info) -Encoding UTF8
try {
    & (Join-Path $projectRoot "scripts\run_experiments.ps1") @arguments *>&1 |
        Out-File -FilePath $log -Encoding UTF8
    exit 0
} catch {
    $_ | Out-String | Add-Content $log -Encoding UTF8
    exit 1
}
