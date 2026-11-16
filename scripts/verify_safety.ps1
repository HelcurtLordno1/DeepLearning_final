# Verify the actual guard functions with a sleeping CPU-only process.
$ErrorActionPreference = "Stop"
$projectRoot = Split-Path $PSScriptRoot -Parent
Set-Location $projectRoot
$tokens = $null
$parseErrors = $null
$ast = [System.Management.Automation.Language.Parser]::ParseFile(
    (Join-Path $projectRoot "scripts\run_experiments.ps1"), [ref]$tokens, [ref]$parseErrors)
if ($parseErrors.Count) { throw ($parseErrors | Out-String) }
$names = @("Get-LimitReason", "Test-OwnedProcess", "Stop-OwnedRun")
foreach ($definition in $ast.FindAll({param($node) $node -is [System.Management.Automation.Language.FunctionDefinitionAst]}, $false)) {
    if ($definition.Name -in $names) { . ([scriptblock]::Create($definition.Extent.Text)) }
}
$sample = [pscustomobject]@{free_ram_gib=20;free_commit_gib=20;free_disk_gib=60;gpu_temperature_c=50;gpu_memory_mib=500}
if (Get-LimitReason $sample) { throw "An ordinary sample was rejected." }
foreach ($case in @(@("free_ram_gib",7),@("free_commit_gib",7),@("free_disk_gib",19),@("gpu_temperature_c",75),@("gpu_memory_mib",6144))) {
    $saved = $sample.($case[0]); $sample.($case[0]) = $case[1]
    if (-not (Get-LimitReason $sample)) { throw "A resource threshold was not enforced." }
    $sample.($case[0]) = $saved
}
$child = $null
try {
    $child = Start-Process powershell.exe -ArgumentList '-NoProfile -Command "Start-Sleep -Seconds 30"' -PassThru -WindowStyle Hidden
    if (-not (Test-OwnedProcess $child.Id $child.Id)) { throw "Own child not identified." }
    if (Test-OwnedProcess $PID $child.Id) { throw "Unrelated process incorrectly identified as ours." }
    Stop-OwnedRun
    if (-not $child.HasExited) { throw "Own child was not stopped." }
} finally {
    Stop-OwnedRun
}
$key = [System.BitConverter]::ToString([System.Security.Cryptography.SHA256]::Create().ComputeHash(
    [System.Text.Encoding]::UTF8.GetBytes($projectRoot.ToLowerInvariant()))).Replace("-", "").Substring(0,16)
$mutex = [System.Threading.Mutex]::new($false, "Local\FinalProjectGpu_$key")
$held = $mutex.WaitOne(0)
try {
    if (-not $held) { throw "A monitored run is already active; abort this test." }
    $probe = Start-Process powershell.exe -ArgumentList '-NoProfile -ExecutionPolicy Bypass -File scripts\run_experiments.ps1 -Mode Check' `
        -WorkingDirectory $projectRoot -Wait -PassThru -NoNewWindow `
        -RedirectStandardOutput "run\safety\duplicate_monitor_stdout.log" -RedirectStandardError "run\safety\duplicate_monitor_stderr.log"
    $output = Get-Content "run\safety\duplicate_monitor_stderr.log" -Raw
    if ($probe.ExitCode -eq 0 -or $output -notmatch "already active") { throw "Duplicate monitor was not blocked." }
} finally {
    if ($held) { $mutex.ReleaseMutex() }; $mutex.Dispose()
}
@{powershell_syntax=$true;resource_thresholds=$true;owned_child_stopped=$true;
  unrelated_process_not_targeted=$true;duplicate_monitor_blocked=$true;training_started=$false} |
  ConvertTo-Json | Set-Content "run\safety\powershell_guard_checks.json" -Encoding UTF8
Write-Host "PowerShell guard checks: PASS. No training started."
