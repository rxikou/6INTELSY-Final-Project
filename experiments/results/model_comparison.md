# Model Performance Comparison

| Model                  |   Test Accuracy |   Test Macro-F1 |   Validation Accuracy |   Validation Macro-F1 | Training Time (approx)   |   Parameters | Architecture             |
|:-----------------------|----------------:|----------------:|----------------------:|----------------------:|:-------------------------|-------------:|:-------------------------|
| Pure CNN               |          0.62   |        nan      |                0.6214 |              nan      | ~45 min                  |      792,326 | Multi-kernel CNN (3,5,7) |
| DistilBERT (Optimized) |          0.6289 |          0.6125 |                0.6251 |                0.6115 | ~90 min                  |   67,217,026 | DistilBERT + Dense Head  |

**Best Model**: DistilBERT with **62.89% test accuracy**
