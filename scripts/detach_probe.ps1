$projectRoot=Split-Path $PSScriptRoot -Parent
Start-Sleep -Seconds 10
(Get-Date).ToString("o") | Set-Content (Join-Path $projectRoot "run\safety\detached_probe_completed.txt") -Encoding UTF8
