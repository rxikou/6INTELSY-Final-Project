.PHONY: help install data baseline train-cnn train-bert rl-agent eval results repro clean

PYTHON := .\venv\Scripts\python
PIP := .\venv\Scripts\pip

help:
	@echo ========================================
	@echo Fake News Detection - Project 5
	@echo ========================================
	@echo Available targets:
	@echo   make install      - Install dependencies
	@echo   make data         - Prepare data splits
	@echo   make baseline     - Train baseline model
	@echo   make train-cnn    - Train CNN model
	@echo   make train-bert   - Train DistilBERT model
	@echo   make rl-agent     - Train RL agent
	@echo   make eval         - Evaluate all models
	@echo   make results      - Generate results tables/plots
	@echo   make repro        - Full reproducibility (all targets)
	@echo   make clean        - Clean artifacts
	@echo ========================================

install:
	@echo Installing dependencies...
	$(PIP) install -q -r requirements.txt

data:
	@echo Preparing data splits...
	@if exist "data\cleaned_train.csv" (echo Data already prepared) else ($(PYTHON) data\get_data.py)

baseline: install data
	@echo Running baseline model (Logistic Regression)...
	$(PYTHON) src\baseline_model.py

train-cnn: install data
	@echo Training CNN model...
	$(PYTHON) src\train.py

train-bert: install data
	@echo Training DistilBERT model (v2 optimized)...
	$(PYTHON) src\train_bert_v2.py

rl-agent: install data
	@echo Training RL agent for threshold optimization...
	$(PYTHON) src\rl_agent.py

eval: install train-cnn train-bert
	@echo Running evaluation on all models...
	$(PYTHON) src\eval.py

results: train-cnn train-bert rl-agent
	@echo Generating results, tables, and plots...
	$(PYTHON) src\generate_results_quick.py

repro: install data baseline train-cnn train-bert rl-agent eval results
	@echo ========================================
	@echo Reproducibility pipeline complete!
	@echo ========================================
	@echo All models trained and results generated.
	@echo See experiments/results/ for output tables and plots.

clean:
	@echo Cleaning artifacts...
	@if exist "experiments\logs" rmdir /s /q experiments\logs
	@if exist "experiments\results\*.png" del experiments\results\*.png
	@if exist "*.pt" del *.pt
	@echo Clean complete.

.DEFAULT_GOAL := help
