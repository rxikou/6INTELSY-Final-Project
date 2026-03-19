#!/bin/bash

echo "Installing dependencies..."
pip install -r requirements.txt

echo "Running baseline model..."
python src/baseline_model.py

echo "Running CNN prototype..."
python src/train.py

echo "Testing DistilBERT..."
python src/test_bert.py

echo "All done!"