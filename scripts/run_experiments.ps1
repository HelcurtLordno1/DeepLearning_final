param(
    [ValidateSet("Check", "Train", "Develop", "Evaluate")][string]$Mode = "Check",
    [ValidateSet("task1", "task2", "task3")][string]$Task,
    [ValidateRange(1, 10)][int]$Epochs = 2,
    [ValidateRange(1, 15)][int]$Minutes = 5,
    [ValidateSet(42, 123, 2026)][int]$Seed = 42,
    [switch]$Improve,
    [string]$Config = "task3/config/aligned.json",
    [string]$RunDirectory
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path $PSScriptRoot -Parent
Set-Location $projectRoot
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
$env:OMP_NUM_THREADS = "2"
$env:MKL_NUM_THREADS = "2"
[Console]::OutputEncoding = New-Object System.Text.UTF8Encoding($false)
$OutputEncoding = [Console]::OutputEncoding
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"
$folder = Join-Path $projectRoot "run\safety"
New-Item -ItemType Directory -Force $folder | Out-Null
$stamp = Get-Date -Format "yyyyMMdd_HHmmss_fff"
$prefix = Join-Path $folder $stamp
$child = $null
$ownsMutex = $false
$oldMonitor = $env:FINAL_PROJECT_MONITOR_PID

function Get-HardwareWarnings([datetime]$Since) {
    $eventErrors = @()
    $events = @(Get-WinEvent -FilterHashtable @{LogName="System"; StartTime=$Since; Level=1,2,3} `
        -ErrorAction SilentlyContinue -ErrorVariable +eventErrors | Where-Object {
        $_.ProviderName -in @("Microsoft-Windows-WHEA-Logger", "Display", "nvlddmkm", "Microsoft-Windows-Resource-Exhaustion-Detector") -or
        ($_.ProviderName -match "disk|stornvme|storahci|iaStor" -and $_.Id -in @(7,11,51,129,153))
    })
    if (@($eventErrors | Where-Object {$_.FullyQualifiedErrorId -notmatch "NoMatchingEventsFound"}).Count) {
        throw "Cannot read Windows hardware event logs; training will not start."
    }
    return @($events | Select-Object @{n="time";e={$_.TimeCreated.ToString("o")}}, Id, RecordId, ProviderName, Message)
}

function Get-Resources {
    $os = Get-CimInstance Win32_OperatingSystem
    $gpuText = & nvidia-smi.exe --id=0 --query-gpu=temperature.gpu,utilization.gpu,memory.used,power.draw --format=csv,noheader,nounits
    if ($LASTEXITCODE -ne 0) { throw "Cannot read GPU telemetry." }
    $values = ([string]$gpuText).Split(",").Trim()
    if ($values.Count -ne 4) { throw "Unexpected GPU telemetry." }
    $drive = [System.IO.DriveInfo]::new([System.IO.Path]::GetPathRoot($projectRoot))
    return [pscustomobject]@{
        time=(Get-Date).ToString("o"); free_ram_gib=[math]::Round($os.FreePhysicalMemory/1MB,2)
        free_commit_gib=[math]::Round($os.FreeVirtualMemory/1MB,2)
        free_disk_gib=[math]::Round($drive.AvailableFreeSpace/1GB,2)
        gpu_temperature_c=[int]$values[0]; gpu_utilization_percent=[int]$values[1]
        gpu_memory_mib=[int]$values[2]
        gpu_power_w=[double]::Parse($values[3], [cultureinfo]::InvariantCulture)
    }
}

function Get-LimitReason($sample) {
    if ($sample.free_ram_gib -lt 8) { return "Less than 8 GiB physical RAM free." }
    if ($sample.free_commit_gib -lt 8) { return "Less than 8 GiB commit headroom." }
    if ($sample.free_disk_gib -lt 20) { return "Less than 20 GiB disk space free." }
    if ($sample.gpu_temperature_c -ge 75) { return "GPU reached the conservative 75 C stop threshold." }
    if ($sample.gpu_memory_mib -ge 6144) { return "Global GPU memory reached 6 GiB." }
    return $null
}

function Get-TrainingProcesses {
    # This WDDM driver lists Explorer/browser windows as C+G too. That list
    # cannot establish CUDA activity. Conservatively exclude other ML runtimes.
    return @(Get-CimInstance Win32_Process | Where-Object {
        $_.Name -match '^(python|pythonw|matlab|julia|ollama|llama-server)\.exe$' -and
        $_.CommandLine -notmatch 'run-jedi-language-server\.py' -and
        $_.CommandLine -notmatch '[\\/]jedilsp[\\/]jedi[\\/]inference[\\/]compiled[\\/]subprocess[\\/]__main__\.py'
    } | ForEach-Object {[int]$_.ProcessId})
}

function Test-OwnedProcess([int]$ProcessIdToCheck, [int]$RootProcessId) {
    for ($depth=0; $depth -lt 8 -and $ProcessIdToCheck -gt 0; $depth++) {
        if ($ProcessIdToCheck -eq $RootProcessId) { return $true }
        $info = Get-CimInstance Win32_Process -Filter "ProcessId = $ProcessIdToCheck"
        if (-not $info) { return $false }
        $ProcessIdToCheck = [int]$info.ParentProcessId
    }
    return $false
}

function Stop-OwnedRun {
    if ($null -ne $child -and -not $child.HasExited) {
        & taskkill.exe /PID $child.Id /T /F | Out-Null
        $child.WaitForExit()
    }
}

$key = [System.BitConverter]::ToString([System.Security.Cryptography.SHA256]::Create().ComputeHash(
    [System.Text.Encoding]::UTF8.GetBytes($projectRoot.ToLowerInvariant()))).Replace("-", "").Substring(0,16)
$mutex = [System.Threading.Mutex]::new($false, "Local\FinalProjectGpu_$key")
try {
    try { $ownsMutex = $mutex.WaitOne(0) } catch [System.Threading.AbandonedMutexException] { $ownsMutex = $true }
    if (-not $ownsMutex) { throw "Another monitored project run is already active." }
    # A single WDDM sample also includes a momentary desktop/browser redraw.
    # Require the median of three samples to be idle; do not bypass sustained load.
    $startupSamples = @()
    for ($probeIndex=0; $probeIndex -lt 3; $probeIndex++) {
        $startupSamples += Get-Resources
        if ($probeIndex -lt 2) { Start-Sleep -Seconds 1 }
    }
    $sample = $startupSamples[-1]
    $startupGpuMedian = @($startupSamples.gpu_utilization_percent | Sort-Object)[1]
    $warnings = @(Get-HardwareWarnings (Get-Date).AddMinutes(-30))
    $otherCompute = @(Get-TrainingProcesses)
    $reasons = @()
    $limit = Get-LimitReason $sample
    if ($limit) { $reasons += $limit }
    if ($warnings.Count) { $reasons += "Hardware/resource warnings occurred in the last 30 minutes." }
    if ($otherCompute.Count) { $reasons += "Another Python/ML process is active." }
    if ($startupGpuMedian -gt 30) { $reasons += "GPU is busy across the startup samples." }
    $reason = if ($reasons.Count) { $reasons -join " " } else { $null }
    $report = [ordered]@{mode=$Mode; task=$Task; improvement=[bool]$Improve; resources=$sample;
        startup_samples=$startupSamples;startup_gpu_median=$startupGpuMedian;hardware_warnings=$warnings;
        existing_training_pids=$otherCompute; training_allowed=($null -eq $reason); reason=$reason}
    $report | ConvertTo-Json -Depth 7 | Set-Content "$prefix`_preflight.json" -Encoding UTF8
    $sample | Format-List | Out-Host
    if ($Mode -eq "Check") {
        if ($reason) { Write-Host "Training blocked: $reason" } else { Write-Host "Preflight passed. No training was started." }
        return
    }
    if ($reason) { throw "Training blocked: $reason See $prefix`_preflight.json" }
    if (-not $Task) { throw "Specify one task, for example -Task task3." }
    if ($Improve -and $Task -ne "task3") { throw "-Improve applies to Task 3 only." }
    if ($Mode -in @("Develop", "Evaluate") -and $Task -ne "task3") { throw "This mode applies to Task 3 only." }
    if ($Mode -eq "Develop" -and (Test-Path "run\task3\final_selection.json")) {
        $frozen = Get-Content "run\task3\final_selection.json" -Raw -Encoding UTF8 | ConvertFrom-Json
        $requested = Get-Content $Config -Raw -Encoding UTF8 | ConvertFrom-Json
        foreach ($entry in $frozen.checkpoints) {
            $frozenConfig = Get-Content (Join-Path $entry.run "config.json") -Raw -Encoding UTF8 | ConvertFrom-Json
            if ($requested.name -eq $frozenConfig.name) { throw "This run name is frozen for test. Use a distinct config/name for a new experiment." }
        }
    }
    if (-not (Test-Path $python)) { throw "The project's Windows environment is missing." }

    # One short, validation-only run. No automatic sweep or notebook retraining.
    $env:FINAL_PROJECT_MONITOR_PID = [string]$PID
    $arguments = @("-u", "$Task\code\$Task.py", "--train", "--epochs", "$Epochs", "--seed", "$Seed")
    if ($Improve) { $arguments += "--improve" }
    if ($Mode -eq "Develop") {
        $softMinutes = [math]::Max(1, $Minutes-1)
        $arguments = @("-u", "task3\code\train_advanced.py", "--config", $Config,
                       "--session-epochs", "$Epochs", "--soft-minutes", "$softMinutes")
        if ($RunDirectory) { $arguments += @("--run", ('"'+$RunDirectory+'"')) }
    }
    if ($Mode -eq "Evaluate") {
        if (-not $RunDirectory) { throw "Specify -RunDirectory for the locked checkpoint." }
        $arguments = @("-u", "scripts\evaluate_task3.py", "--run", ('"'+$RunDirectory+'"'))
    }
    $start = Get-Date
    $lastEventCheck = $start
    $child = Start-Process -FilePath $python -ArgumentList $arguments -WorkingDirectory $projectRoot `
        -PassThru -NoNewWindow -RedirectStandardOutput "$prefix`_stdout.log" -RedirectStandardError "$prefix`_stderr.log"
    # Windows PowerShell 5 may otherwise lose ExitCode after Refresh/HasExited.
    $childHandle = $child.Handle
    Write-Host "Started $Task; at most $Epochs epochs / $Minutes minutes. Logs: $prefix"
    while (-not $child.HasExited) {
        $sample = Get-Resources
        $sample | ConvertTo-Json -Compress | Add-Content "$prefix`_telemetry.jsonl" -Encoding UTF8
        $reason = Get-LimitReason $sample
        $checkTime = Get-Date
        $newWarnings = @(Get-HardwareWarnings $lastEventCheck)
        $lastEventCheck = $checkTime
        if ($newWarnings.Count) {
            $newWarnings | ConvertTo-Json -Depth 5 | Set-Content "$prefix`_hardware_stop.json" -Encoding UTF8
            $reason = "New Windows hardware/resource warning during this run."
        }
        foreach ($computeId in @(Get-TrainingProcesses)) {
            if (-not (Test-OwnedProcess $computeId $child.Id)) { $reason = "A competing Python/ML process started." }
        }
        if (((Get-Date)-$start).TotalMinutes -ge $Minutes) { $reason = "Session time limit reached." }
        if ($reason) { throw $reason }
        Start-Sleep -Seconds 5
        $child.Refresh()
    }
    $child.WaitForExit()
    if ($child.ExitCode -ne 0) { throw "Python failed. Read $prefix`_stderr.log" }
    Write-Host "Finished one monitored session. No further run was queued."
    @{status="complete";mode=$Mode;task=$Task;python_exit_code=$child.ExitCode;finished=(Get-Date).ToString("o")} |
        ConvertTo-Json | Set-Content "$prefix`_session.json" -Encoding UTF8
} catch {
    Stop-OwnedRun
    $_.Exception.Message | Set-Content "$prefix`_stopped.txt" -Encoding UTF8
    @{status="stopped";mode=$Mode;task=$Task;reason=$_.Exception.Message;finished=(Get-Date).ToString("o")} |
        ConvertTo-Json | Set-Content "$prefix`_session.json" -Encoding UTF8
    throw
} finally {
    # Also stop our child tree if Ctrl+C interrupts the PowerShell monitor.
    Stop-OwnedRun
    $env:FINAL_PROJECT_MONITOR_PID = $oldMonitor
    if ($ownsMutex) { $mutex.ReleaseMutex() }
    $mutex.Dispose()
}
