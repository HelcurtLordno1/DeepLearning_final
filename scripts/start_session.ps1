param(
    [ValidateSet("Develop","Evaluate")][string]$Mode="Develop",
    [ValidateRange(1,4)][int]$Epochs=2,
    [ValidateRange(2,15)][int]$Minutes=15,
    [string]$Config="task3/config/aligned.json",
    [string]$RunDirectory
)
$ErrorActionPreference="Stop"
$projectRoot=Split-Path $PSScriptRoot -Parent
Set-Location $projectRoot
$folder=Join-Path $projectRoot "run\safety"
New-Item -ItemType Directory -Force $folder | Out-Null
$stamp=Get-Date -Format "yyyyMMdd_HHmmss_fff"
$relativeSpec="run/safety/${stamp}_job.json"
$record=@{started=(Get-Date).ToString("o");mode=$Mode;epochs=$Epochs;minutes=$Minutes;
          config=$Config;run_directory=$RunDirectory;host_log="run/safety/${stamp}_host.log";
          host_info="run/safety/${stamp}_host_info.json"}
$record | ConvertTo-Json | Set-Content (Join-Path $projectRoot $relativeSpec) -Encoding UTF8
# WMI creates a native Windows process independently of the API's terminal job.
# All arguments are read from JSON; no user path is interpolated into shell code.
$command="powershell.exe -NoProfile -NonInteractive -ExecutionPolicy Bypass -File scripts\session_host.ps1 -SpecFile $relativeSpec"
$result=Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine=$command;CurrentDirectory=$projectRoot}
if ($result.ReturnValue -ne 0) {throw "Could not start native monitored session: $($result.ReturnValue)"}
$record.monitor_pid=[int]$result.ProcessId
$record | ConvertTo-Json | Set-Content "$folder\active_session.json" -Encoding UTF8
$record | ConvertTo-Json
