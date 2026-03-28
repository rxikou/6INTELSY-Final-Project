#!/usr/bin/env pwsh
# Reproducibility script for Fake News Detection project
# Usage: .\run.ps1 [target]

param([string]$Target = "all")

$PROJECT_ROOT = $PSScriptRoot
Set-Location $PROJECT_ROOT

$DEFAULT_VENV_DIR = Join-Path $PROJECT_ROOT "venv"
$SHORT_VENV_DIR = Join-Path $env:LOCALAPPDATA "fnmd-venv"
$PATH_LIMIT_PROBE = Join-Path $DEFAULT_VENV_DIR "Lib\site-packages\transformers\models\audio_spectrogram_transformer\configuration_audio_spectrogram_transformer.py"

# Use a short venv location on Windows if site-packages paths are likely to exceed MAX_PATH.
if ($PATH_LIMIT_PROBE.Length -ge 250) {
    $VENV_DIR = $SHORT_VENV_DIR
} else {
    $VENV_DIR = $DEFAULT_VENV_DIR
}

$VENV_PYTHON = Join-Path $VENV_DIR "Scripts\python.exe"
$PYTHON = $null

function Get-PythonVersion {
    param([string]$PythonExe)
    $ver = & $PythonExe -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>$null
    return $ver
}

function Assert-StepSuccess {
    param([string]$StepName)
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Step failed: $StepName"
        exit 1
    }
}

if (Test-Path $VENV_PYTHON) {
    $PYTHON = $VENV_PYTHON
} else {
    $pyLauncher = Get-Command py -ErrorAction SilentlyContinue
    if ($pyLauncher) {
        $PYTHON = $pyLauncher.Source
    } else {
        $pythonCmd = Get-Command python -ErrorAction SilentlyContinue
        if ($pythonCmd) {
            $PYTHON = $pythonCmd.Source
        }
    }
}

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Fake News Detection - Project 5" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

if ($Target -eq "help") {
    Write-Host "Usage: .\run.ps1 [target]`n" -ForegroundColor Yellow
    Write-Host "Available targets:"
    Write-Host "  all         - Full reproducibility pipeline"
    Write-Host "  install     - Install dependencies"
    Write-Host "  baseline    - Train baseline model"
    Write-Host "  train-cnn   - Train CNN model"
    Write-Host "  train-bert  - Train DistilBERT model"
    Write-Host "  eval        - Evaluate all models"
    Write-Host "  rl-agent    - Train RL agent"
    Write-Host "  results     - Generate results"
    Write-Host "  clean       - Clean artifacts`n"
    exit 0
}

Write-Host "Starting: $Target`n" -ForegroundColor Yellow

if ($Target -eq "all" -or $Target -eq "repro" -or $Target -eq "install") {
    Write-Host "[1/6] Installing dependencies..." -ForegroundColor Cyan

    if ($VENV_DIR -ne $DEFAULT_VENV_DIR) {
        Write-Host "Using short virtual environment path: $VENV_DIR" -ForegroundColor Yellow
    }

    if (-not (Test-Path $VENV_PYTHON)) {
        if (-not $PYTHON) {
            Write-Error "Python was not found. Install Python 3.10+ and re-run .\run.ps1 install"
            exit 1
        }

        Write-Host "Creating virtual environment at $VENV_DIR ..." -ForegroundColor Yellow
        if ((Split-Path $PYTHON -Leaf) -ieq "py.exe" -or (Split-Path $PYTHON -Leaf) -ieq "py") {
            # Prefer Python 3.11 for compatibility with pinned dependencies.
            & $PYTHON -3.11 -m venv $VENV_DIR
            if ($LASTEXITCODE -ne 0) {
                & $PYTHON -3.12 -m venv $VENV_DIR
            }
            if ($LASTEXITCODE -ne 0) {
                & $PYTHON -3.10 -m venv $VENV_DIR
            }
        } else {
            & $PYTHON -m venv $VENV_DIR
        }

        if ($LASTEXITCODE -ne 0 -or -not (Test-Path $VENV_PYTHON)) {
            Write-Error "Failed to create virtual environment."
            exit 1
        }
    }

    $PYTHON = $VENV_PYTHON
    $pythonVersion = Get-PythonVersion -PythonExe $PYTHON
    if (-not $pythonVersion) {
        Write-Error "Could not determine Python version from $PYTHON"
        exit 1
    }

    if ($pythonVersion -notin @("3.10", "3.11", "3.12")) {
        Write-Host "Detected unsupported venv Python version: $pythonVersion. Recreating venv..." -ForegroundColor Yellow
        Remove-Item $VENV_DIR -Recurse -Force -ErrorAction SilentlyContinue

        $pyLauncher = Get-Command py -ErrorAction SilentlyContinue
        if (-not $pyLauncher) {
            Write-Error "Unsupported Python version detected in venv: $pythonVersion. Install Python 3.10-3.12 and ensure 'py' launcher is available."
            exit 1
        }

        & $pyLauncher.Source -3.12 -m venv $VENV_DIR
        if ($LASTEXITCODE -ne 0) {
            & $pyLauncher.Source -3.11 -m venv $VENV_DIR
        }
        if ($LASTEXITCODE -ne 0) {
            & $pyLauncher.Source -3.10 -m venv $VENV_DIR
        }
        if ($LASTEXITCODE -ne 0 -or -not (Test-Path $VENV_PYTHON)) {
            Write-Error "Failed to recreate virtual environment with Python 3.10-3.12."
            exit 1
        }

        $PYTHON = $VENV_PYTHON
        $pythonVersion = Get-PythonVersion -PythonExe $PYTHON
        if ($pythonVersion -notin @("3.10", "3.11", "3.12")) {
            Write-Error "Unsupported Python version still detected in venv: $pythonVersion"
            exit 1
        }
    }

    & $PYTHON -m pip install --upgrade pip setuptools wheel
    Assert-StepSuccess -StepName "Upgrade pip/setuptools/wheel"

    & $PYTHON -m pip install --no-cache-dir -r requirements.txt
    Assert-StepSuccess -StepName "Install dependencies"
}

if (-not $PYTHON) {
    Write-Error "Python executable not found. Run .\run.ps1 install first or install Python 3.10+."
    exit 1
}

$pythonVersion = Get-PythonVersion -PythonExe $PYTHON
if (-not $pythonVersion) {
    Write-Error "Could not determine Python version from $PYTHON"
    exit 1
}

if ($pythonVersion -notin @("3.10", "3.11", "3.12")) {
    Write-Error "Unsupported Python version detected: $pythonVersion. Recreate venv with Python 3.10, 3.11, or 3.12."
    exit 1
}

if ($Target -eq "all" -or $Target -eq "repro" -or $Target -eq "train-cnn") {
    Write-Host "[2/6] Training CNN model..." -ForegroundColor Cyan
    & $PYTHON src\train.py
    Assert-StepSuccess -StepName "Training CNN model"
}

if ($Target -eq "all" -or $Target -eq "repro" -or $Target -eq "train-bert") {
    Write-Host "[3/6] Training DistilBERT model..." -ForegroundColor Cyan
    & $PYTHON src\train_bert_v2.py
    Assert-StepSuccess -StepName "Training DistilBERT model"
}

if ($Target -eq "all" -or $Target -eq "repro" -or $Target -eq "eval") {
    Write-Host "[4/6] Running evaluation..." -ForegroundColor Cyan
    & $PYTHON src\eval.py
    Assert-StepSuccess -StepName "Running evaluation"
}

if ($Target -eq "all" -or $Target -eq "repro" -or $Target -eq "rl-agent") {
    Write-Host "[5/6] Training RL agent..." -ForegroundColor Cyan
    & $PYTHON src\rl_integration.py
    Assert-StepSuccess -StepName "Training RL agent"
}

if ($Target -eq "all" -or $Target -eq "repro" -or $Target -eq "results") {
    Write-Host "[6/6] Generating results..." -ForegroundColor Cyan
    & $PYTHON src\generate_results_quick.py
    Assert-StepSuccess -StepName "Generating results"
}

if ($Target -eq "clean") {
    Write-Host "Cleaning artifacts..." -ForegroundColor Yellow
    Remove-Item "experiments\logs" -Recurse -Force -ErrorAction SilentlyContinue
    Remove-Item "experiments\results\*" -Recurse -Force -ErrorAction SilentlyContinue
    Write-Host "✓ Clean complete`n" -ForegroundColor Green
}

if ($Target -eq "all" -or $Target -eq "repro") {
    Write-Host "`n========================================" -ForegroundColor Cyan
    Write-Host "✓ Reproducibility pipeline complete!" -ForegroundColor Green
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host "See experiments/results/ for outputs`n" -ForegroundColor Yellow
}
