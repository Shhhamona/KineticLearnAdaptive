# PowerShell script to run multiple active learning experiments in parallel
# Each experiment uses different k-multiplicative factors

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Running 3 Active Learning Experiments in Parallel" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

# Define experiment configurations
$experiments = @(
    @{
        Name = "Experiment_1_KMult_1.001"
        KMultFactor = 1.001
        NSamples = 2500
        LokiVersion = "v2"
        LogFile = "results\\logs\\exp1_kmult_g_1.001.log"
    },
    @{
        Name = "Experiment_2_KMult_1.001"
        KMultFactor = 1.001
        NSamples = 2525
        LokiVersion = "v3"
        LogFile = "results\\logs\\exp2_kmult_1.001.log"
    },
    @{
        Name = "Experiment_3_KMult_1.0025"
        KMultFactor = 1.0025
        NSamples = 2550
        LokiVersion = "v4"
        LogFile = "results\\logs\\exp3_kmult_1.0025.log"
    },
    @{
        Name = "Experiment_4_KMult_1.0025"
        KMultFactor = 1.0025
        NSamples = 2575
        LokiVersion = "v5"
        LogFile = "results\\logs\\exp4_kmult_1.0025.log"
    }
)

# Create logs directory if it doesn't exist
$logsDir = "results\logs"
if (-not (Test-Path $logsDir)) {
    New-Item -ItemType Directory -Path $logsDir -Force | Out-Null
    Write-Host "Created logs directory: $logsDir" -ForegroundColor Green
}

# Get the current directory (needed for background jobs)
$workingDir = Get-Location

# Array to hold job objects
$jobs = @()

# Launch each experiment as a background job
foreach ($exp in $experiments) {
    Write-Host "`n[$(Get-Date -Format 'HH:mm:ss')] Launching: $($exp.Name)" -ForegroundColor Yellow
    Write-Host "  K-Multiplicative Factor: $($exp.KMultFactor)" -ForegroundColor Gray
    Write-Host "  Samples per iteration: $($exp.NSamples)" -ForegroundColor Gray
    Write-Host "  LoKI Version: $($exp.LokiVersion)" -ForegroundColor Gray
    Write-Host "  Log file: $($exp.LogFile)" -ForegroundColor Gray
    
    # Build command with full path to Python in venv
    $pythonExe = Join-Path $workingDir ".venv\Scripts\python.exe"
    $scriptPath = Join-Path $workingDir "active_learning_train.py"
    $command = "`"$pythonExe`" `"$scriptPath`" --n-samples-per-iteration $($exp.NSamples) --k-mult-factor $($exp.KMultFactor) --loki-version $($exp.LokiVersion)"
    
    # Convert log file to absolute path
    $absoluteLogFile = Join-Path $workingDir $exp.LogFile
    
    # Start background job
    $job = Start-Job -ScriptBlock {
        param($cmd, $logFile, $expName, $workDir)
        
        # Change to the working directory
        Set-Location $workDir
        
        # Set UTF-8 encoding for Python output with emojis
        $env:PYTHONIOENCODING = "utf-8"
        
        # Redirect output to log file
        $timestamp = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
        "[$timestamp] Starting experiment: $expName" | Out-File -FilePath $logFile -Encoding UTF8
        "Working Directory: $workDir" | Out-File -FilePath $logFile -Append -Encoding UTF8
        "Command: $cmd" | Out-File -FilePath $logFile -Append -Encoding UTF8
        "" | Out-File -FilePath $logFile -Append -Encoding UTF8
        
        # Execute command and capture output
        try {
            & cmd /c "chcp 65001 >nul && $cmd 2>&1" | Out-File -FilePath $logFile -Append -Encoding UTF8
        } catch {
            "ERROR: $_" | Out-File -FilePath $logFile -Append -Encoding UTF8
        }
        
        $endTime = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
        "" | Out-File -FilePath $logFile -Append -Encoding UTF8
        "[$endTime] Experiment completed: $expName" | Out-File -FilePath $logFile -Append -Encoding UTF8
        
    } -ArgumentList $command, $absoluteLogFile, $exp.Name, $workingDir
    
    $jobs += @{
        Job = $job
        Name = $exp.Name
        LogFile = $absoluteLogFile
    }
}

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "All experiments launched!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "`nMonitoring progress (press Ctrl+C to stop monitoring, jobs will continue)...`n" -ForegroundColor Yellow

# Monitor job progress
$allCompleted = $false
while (-not $allCompleted) {
    Start-Sleep -Seconds 5
    
    $completedCount = 0
    Write-Host "`r[$(Get-Date -Format 'HH:mm:ss')] Status:" -ForegroundColor Cyan
    
    foreach ($jobInfo in $jobs) {
        $job = $jobInfo.Job
        $status = $job.State
        
        $statusColor = switch ($status) {
            "Running" { "Yellow" }
            "Completed" { "Green" }
            "Failed" { "Red" }
            default { "Gray" }
        }
        
        Write-Host "  $($jobInfo.Name): $status" -ForegroundColor $statusColor
        
        if ($status -eq "Completed" -or $status -eq "Failed") {
            $completedCount++
        }
    }
    
    if ($completedCount -eq $jobs.Count) {
        $allCompleted = $true
    }
}

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "All experiments completed!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan

# Display final results
Write-Host "`nFinal Status:" -ForegroundColor Cyan
foreach ($jobInfo in $jobs) {
    $job = $jobInfo.Job
    Write-Host "`n--- $($jobInfo.Name) ---" -ForegroundColor Yellow
    Write-Host "Status: $($job.State)" -ForegroundColor $(if ($job.State -eq "Completed") { "Green" } else { "Red" })
    Write-Host "Log file: $($jobInfo.LogFile)" -ForegroundColor Gray
    
    # Receive job output (this also cleans up the job)
    if ($job.State -eq "Failed") {
        Write-Host "Error details:" -ForegroundColor Red
        Receive-Job -Job $job
    }
}

# Clean up jobs
Write-Host "`nCleaning up background jobs..." -ForegroundColor Gray
foreach ($jobInfo in $jobs) {
    Remove-Job -Job $jobInfo.Job -Force
}

Write-Host "`n✅ All done! Check the log files for detailed output." -ForegroundColor Green
Write-Host "Log directory: $logsDir`n" -ForegroundColor Gray
