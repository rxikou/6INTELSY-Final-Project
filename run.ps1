#!/usr/bin/env pwsh
# Reproducibility script for Fake News Detection project
# Usage: .\run.ps1 [target]

param([string]$Target = "all")

$PYTHON = ".\venv\Scripts\python"
$PIP = ".\venv\Scripts\pip"

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
    Write-Host "  rl-agent    - Train RL agent"
    Write-Host "  eval        - Evaluate all models"
    Write-Host "  results     - Generate results"
    Write-Host "  clean       - Clean artifacts`n"
    exit 0
}

Write-Host "Starting: $Target`n" -ForegroundColor Yellow

if ($Target -eq "all" -or $Target -eq "repro" -or $Target -eq "install") {
    Write-Host "[1/8] Installing dependencies..." -ForegroundColor Cyan
    & $PIP install -q -r requirements.txt
}

if ($Target -eq "all" -or $Target -eq "repro" -or $Target -eq "train-cnn") {
    Write-Host "[3/8] Training CNN model..." -ForegroundColor Cyan
    & $PYTHON src\train.py
}

if ($Target -eq "all" -or $Target -eq "repro" -or $Target -eq "train-bert") {
    Write-Host "[4/8] Training DistilBERT model..." -ForegroundColor Cyan
    & $PYTHON src\train_bert_v2.py
}

if ($Target -eq "all" -or $Target -eq "repro" -or $Target -eq "rl-agent") {
    Write-Host "[5/8] Training RL agent..." -ForegroundColor Cyan
    & $PYTHON src\rl_integration.py
}

if ($Target -eq "all" -or $Target -eq "repro" -or $Target -eq "eval") {
    Write-Host "[6/8] Running evaluation..." -ForegroundColor Cyan
    & $PYTHON src\eval.py
}

if ($Target -eq "all" -or $Target -eq "repro" -or $Target -eq "results") {
    Write-Host "[7/8] Generating results..." -ForegroundColor Cyan
    & $PYTHON src\generate_results_quick.py
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
