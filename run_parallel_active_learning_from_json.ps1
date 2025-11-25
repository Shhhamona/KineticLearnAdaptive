# PowerShell script to run multiple active learning experiments in parallel
# Reads experiment configurations from JSON files

param(
    [string]$ConfigDir = "adaptive_learning_setups\500_iteration_040_shrink\iteration6",
    [switch]$DryRun
)

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Parallel Active Learning Runner" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

# Check if config directory exists
if (-not (Test-Path $ConfigDir)) {
    Write-Host "ERROR: Configuration directory not found: $ConfigDir" -ForegroundColor Red
    exit 1
}

# Get all JSON files from the config directory
$configFiles = Get-ChildItem -Path $ConfigDir -Filter "*.json"

if ($configFiles.Count -eq 0) {
    Write-Host "ERROR: No JSON configuration files found in: $ConfigDir" -ForegroundColor Red
    exit 1
}

Write-Host "`nFound $($configFiles.Count) configuration file(s)" -ForegroundColor Green
Write-Host "Configuration directory: $ConfigDir`n" -ForegroundColor Gray

# Load experiments from JSON files
$experiments = @()
foreach ($configFile in $configFiles) {
    try {
        $config = Get-Content -Path $configFile.FullName -Raw | ConvertFrom-Json
        
        # Extract parameters
        $params = $config.parameters
        
        # Build experiment object
        $expName = if ($config.experiment_name) { $config.experiment_name } else { $configFile.BaseName }
        
        $experiment = @{
            Name = $expName
            ConfigFile = $configFile.FullName
            NSamples = $params.n_samples_per_iteration
            KMultFactor = $params.k_mult_factor
            KCenter = $params.k_center  # Array of 3 values
            LokiVersion = $params.loki_version
            LogFile = "results\logs\$($expName).log"
            Description = $config.description
        }
        
        $experiments += $experiment
        
        Write-Host "Loaded: $expName" -ForegroundColor Yellow
        Write-Host "  Samples/iteration: $($experiment.NSamples)" -ForegroundColor Gray
        Write-Host "  K mult factor: $($experiment.KMultFactor)" -ForegroundColor Gray
        Write-Host "  K center: [$($experiment.KCenter[0]), $($experiment.KCenter[1]), $($experiment.KCenter[2])]" -ForegroundColor Gray
        Write-Host "  LoKI version: $($experiment.LokiVersion)" -ForegroundColor Gray
        if ($config.predicted_k_info) {
            Write-Host "  Source: Seed $($config.predicted_k_info.seed) (MSE: $($config.predicted_k_info.mse_original))" -ForegroundColor Gray
        }
        Write-Host ""
        
    } catch {
        Write-Host "ERROR: Failed to load config from $($configFile.Name): $_" -ForegroundColor Red
    }
}

if ($experiments.Count -eq 0) {
    Write-Host "ERROR: No valid experiments loaded" -ForegroundColor Red
    exit 1
}

# If dry run, just display what would be run and exit
if ($DryRun) {
    Write-Host "`n========================================" -ForegroundColor Cyan
    Write-Host "DRY RUN - Commands that would be executed:" -ForegroundColor Yellow
    Write-Host "========================================`n" -ForegroundColor Cyan
    
    foreach ($exp in $experiments) {
        Write-Host "$($exp.Name):" -ForegroundColor Yellow
        $kCenterStr = "$($exp.KCenter[0]) $($exp.KCenter[1]) $($exp.KCenter[2])"
        Write-Host "  python active_learning_train.py --n-samples-per-iteration $($exp.NSamples) --k-mult-factor $($exp.KMultFactor) --k-center $kCenterStr --loki-version $($exp.LokiVersion)" -ForegroundColor Gray
        Write-Host ""
    }
    
    Write-Host "Use without --DryRun to actually execute the experiments`n" -ForegroundColor Green
    exit 0
}

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
Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "Launching Experiments" -ForegroundColor Cyan
Write-Host "========================================`n" -ForegroundColor Cyan

foreach ($exp in $experiments) {
    Write-Host "[$(Get-Date -Format 'HH:mm:ss')] Launching: $($exp.Name)" -ForegroundColor Yellow
    Write-Host "  K center: [$($exp.KCenter[0]), $($exp.KCenter[1]), $($exp.KCenter[2])]" -ForegroundColor Gray
    Write-Host "  K mult factor: $($exp.KMultFactor)" -ForegroundColor Gray
    Write-Host "  Samples/iteration: $($exp.NSamples)" -ForegroundColor Gray
    Write-Host "  LoKI version: $($exp.LokiVersion)" -ForegroundColor Gray
    Write-Host "  Log file: $($exp.LogFile)" -ForegroundColor Gray
    
    # Build command with full path to Python in venv
    $pythonExe = Join-Path $workingDir ".venv\Scripts\python.exe"
    $scriptPath = Join-Path $workingDir "active_learning_train.py"
    
    # Format k-center values for command line
    $kCenterStr = "$($exp.KCenter[0]) $($exp.KCenter[1]) $($exp.KCenter[2])"
    
    $command = "`"$pythonExe`" `"$scriptPath`" --n-samples-per-iteration $($exp.NSamples) --k-mult-factor $($exp.KMultFactor) --k-center $kCenterStr --loki-version $($exp.LokiVersion)"
    
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
        ConfigFile = $exp.ConfigFile
    }
    
    Write-Host ""
}

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "All experiments launched!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "`nMonitoring progress (press Ctrl+C to stop monitoring, jobs will continue)...`n" -ForegroundColor Yellow

# Monitor job progress
$allCompleted = $false
while (-not $allCompleted) {
    Start-Sleep -Seconds 10
    
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
    
    Write-Host ""
}

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "All experiments completed!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan

# Display final results
Write-Host "`nFinal Status:" -ForegroundColor Cyan
foreach ($jobInfo in $jobs) {
    $job = $jobInfo.Job
    Write-Host "`n--- $($jobInfo.Name) ---" -ForegroundColor Yellow
    Write-Host "Status: $($job.State)" -ForegroundColor $(if ($job.State -eq "Completed") { "Green" } else { "Red" })
    Write-Host "Config: $($jobInfo.ConfigFile)" -ForegroundColor Gray
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
