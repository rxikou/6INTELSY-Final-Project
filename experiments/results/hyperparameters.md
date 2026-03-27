# Hyperparameter Comparison

| Hyperparameter      | Pure CNN   | DistilBERT   |
|:--------------------|:-----------|:-------------|
| Batch Size          | 64         | 32           |
| Learning Rate       | 0.001      | 1.5e-5       |
| Max Sequence Length | 128        | 128          |
| Epochs              | 30         | 20           |
| Warmup Ratio        | 10%        | 15%          |
| Weight Decay        | 0.0005     | 0.05         |
| Dropout Rate        | 0.08       | 0.2          |
| Label Smoothing     | N/A        | 0.15         |