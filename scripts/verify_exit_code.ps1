$ErrorActionPreference="Stop"
$projectRoot=Split-Path $PSScriptRoot -Parent
Set-Location $projectRoot
$process=Start-Process powershell.exe -ArgumentList '-NoProfile -NonInteractive -Command "Start-Sleep -Milliseconds 300; exit 0"' `
    -PassThru -WindowStyle Hidden -RedirectStandardOutput "run/safety/exitcode_probe_stdout.log" `
    -RedirectStandardError "run/safety/exitcode_probe_stderr.log"
$processHandle=$process.Handle
while (-not $process.HasExited) {Start-Sleep -Milliseconds 100; $process.Refresh()}
$process.WaitForExit()
if ($null -eq $process.ExitCode -or $process.ExitCode -ne 0) {throw "Native exit-code capture failed."}
@{captured_exit_code=$process.ExitCode;handle_cached=$true;passed=$true;training_started=$false} |
    ConvertTo-Json | Set-Content "run/safety/exitcode_checks.json" -Encoding UTF8
Write-Host "Native exit-code check PASS. No training started."
