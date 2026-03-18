# Model Card: Hybrid Fake News Classifier (v0.1)

## Model Details
* **Model Architecture:** A hybrid deep learning pipeline integrating a compact transformer (DistilBERT) for deep semantic contextualization, a 1D-Convolutional Neural Network (CNN) for local linguistic feature extraction, and a Reinforcement Learning (RL) agent designed to dynamically optimize the classification threshold.
* **Version:** 0.1 (Baseline Checkpoint)
* **Frameworks:** PyTorch, Hugging Face `transformers`, Scikit-Learn, and `gymnasium`.
* **Resource Constraint:** Designed to train and evaluate strictly within a 90-minute window on a single mid-range GPU.

## Intended Use
* **Primary Use Case:** Automated binary text classification of short news claims and social media statements into "Real" (1) or "Fake" (0) categories.
* **Out-of-Scope Uses:** This model is not intended for long-form document processing, deep systemic fact-checking without human oversight, or generating text. It should not be used as an absolute arbiter of truth.

## Training Data
* **Dataset:** The official LIAR dataset, sourced directly from UC Santa Barbara.
* **Preprocessing:** The original six-point truth scale was collapsed into a binary target. Missing text rows were dropped, and text was standardized to lowercase. 
* **Data Splits:** The data is split into explicit `train`, `val`, and `test` sets to prevent data leakage during the RL threshold tuning phase.

## Evaluation Data & Metrics
* **Metrics:** The primary optimization metric is the **Macro F1-Score**, chosen specifically to account for any class imbalances and ensure the model performs equally well on both minority and majority classes. Accuracy is logged as a secondary baseline metric.
* **Baseline Performance:** Initial Scikit-Learn Logistic Regression (TF-IDF) baselines have been established to benchmark the upcoming deep learning improvements.

## Caveats and Recommendations
* **Geographical and Temporal Bias:** The LIAR dataset is heavily US-centric and spans specific political eras. The model's vocabulary and contextual embeddings may not generalize well to international news or future novel events without periodic retraining.