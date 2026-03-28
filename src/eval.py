"""
Evaluation module for trained models - generates metrics, confusion matrices, and analysis.
Usage: python eval.py
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    confusion_matrix, classification_report, accuracy_score, 
    precision_score, recall_score, f1_score, roc_auc_score, roc_curve, auc
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from collections import Counter
import pickle

# Add src to path
import sys
sys.path.insert(0, str(Path(__file__).parent))

from models.cnn_model import CNNTextClassifier
from models.distilbert_model import DistilBERTClassifier

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

RESULTS_DIR = Path(__file__).parent.parent / "experiments" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = ["FAKE", "REAL"]


def load_cnn_model(checkpoint_path="best_cnn.pt"):
    """Load CNN model with vocabulary and embeddings."""
    print("[CNN] Loading model and vocabulary...")
    
    # Try to load vocabulary
    vocab_paths = ["vocab.pkl", "data/vocab.pkl", "../vocab.pkl"]
    vocab = None
    for path in vocab_paths:
        if Path(path).exists():
            with open(path, "rb") as f:
                vocab = pickle.load(f)
            print(f"[CNN] Loaded vocab from {path}")
            break
    
    if vocab is None:
        print("[CNN] WARNING: vocab.pkl not found. Creating empty vocab for inference.")
        print("[CNN] Skipping CNN evaluation - model won't be loaded correctly.")
        raise FileNotFoundError("vocab.pkl not found. Train CNN first with train.py")
    
    vocab_size = len(vocab)
    embed_dim = 100
    
    # Load embeddings if they exist
    try:
        embeddings = np.load("embeddings.npy")
        embeddings_tensor = torch.from_numpy(embeddings).float()
    except:
        print("[CNN] Using random embeddings (training not completed properly)")
        embeddings_tensor = torch.randn(vocab_size + 1, embed_dim) * 0.01
    
    # Create model
    model = CNNTextClassifier(
        vocab_size=vocab_size + 1,
        embeddings=embeddings_tensor,
        embed_dim=100,
        num_classes=2,
        num_filters=128,
        kernel_sizes=[3, 5, 7],
        hidden_dim=128,
        dropout_rate=0.08,
        use_batch_norm=True
    )
    
    # Load checkpoint
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    
    return model, vocab


def load_bert_model(checkpoint_path="best_bert.pt"):
    """Load DistilBERT model."""
    print("[BERT] Loading model and tokenizer...")
    from transformers import DistilBertTokenizer
    
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    
    model = DistilBERTClassifier(num_classes=2, dropout_rate=0.2, label_smoothing=0.15)
    state_dict = torch.load(checkpoint_path, map_location=device)

    # Remove unexpected keys safely
    if "criterion.weight" in state_dict:
        del state_dict["criterion.weight"]

    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    
    return model, tokenizer


def encode_cnn(texts, vocab, max_len=100):
    """Encode texts for CNN."""
    X = []
    for text in texts:
        tokens = str(text).split()
        encoded = [vocab.get(word, 0) for word in tokens][:max_len]
        encoded += [0] * (max_len - len(encoded))
        X.append(encoded)
    return torch.tensor(X, dtype=torch.long).to(device)


def create_mask(x, pad_idx=0):
    """Create attention mask for CNN."""
    return (x != pad_idx).float().to(device)


def predict_cnn(model, X, vocab):
    """Get predictions from CNN model."""
    X_encoded = encode_cnn(X, vocab)
    masks = create_mask(X_encoded)
    
    with torch.no_grad():
        logits = model(X_encoded, mask=masks)
        probs = torch.softmax(logits, dim=1)
        preds = logits.argmax(dim=1)
    
    return preds.cpu().numpy(), probs.cpu().numpy()


def predict_bert(model, texts, tokenizer):
    """Get predictions from DistilBERT model."""
    from torch.utils.data import DataLoader, TensorDataset
    
    # Tokenize
    encodings = tokenizer(
        texts.tolist(),
        truncation=True,
        padding=True,
        max_length=128,
    )
    
    # Create dataset and loader
    input_ids = torch.tensor(encodings["input_ids"]).to(device)
    attention_mask = torch.tensor(encodings["attention_mask"]).to(device)
    dataset = TensorDataset(input_ids, attention_mask)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    all_preds = []
    all_probs = []
    
    with torch.no_grad():
        for batch_input_ids, batch_attention_mask in loader:
            outputs = model(batch_input_ids, attention_mask=batch_attention_mask)

            # Handle dict output safely
            if isinstance(outputs, dict):
                logits = outputs["logits"]
            else:
                logits = outputs
                
            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    return np.array(all_preds), np.array(all_probs)

def predict_baseline(X_train, y_train, X_test):
    vectorizer = TfidfVectorizer(max_features=5000)

    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_vec, y_train)

    preds = model.predict(X_test_vec)
    probs = model.predict_proba(X_test_vec)

    return preds, probs


def evaluate_model(y_true, y_pred, y_probs, model_name="Model"):
    """Calculate and return evaluation metrics."""
    metrics = {
        "Model": model_name,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "Recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "Macro-F1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "Weighted-F1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }
    
    # Per-class metrics
    report = classification_report(y_true, y_pred, target_names=CLASS_NAMES, output_dict=True)
    
    return metrics, report


def save_confusion_matrix(y_true, y_pred, model_name):
    """Generate and save confusion matrix visualization."""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title(f"Confusion Matrix - {model_name}")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    
    save_path = RESULTS_DIR / f"confusion_matrix_{model_name.lower().replace(' ', '_')}.png"
    plt.savefig(save_path, dpi=100)
    plt.close()
    print(f"[OK] Saved: {save_path}")
    
    return cm


def save_error_analysis(y_true, y_pred, texts, model_name):
    """Save misclassified samples for error analysis."""
    errors = []
    for i, (true, pred, text) in enumerate(zip(y_true, y_pred, texts)):
        if true != pred:
            errors.append({
                "Index": i,
                "Text": str(text)[:200],
                "True Label": CLASS_NAMES[true],
                "Predicted Label": CLASS_NAMES[pred],
            })
    
    error_df = pd.DataFrame(errors)
    save_path = RESULTS_DIR / f"error_analysis_{model_name.lower().replace(' ', '_')}.csv"
    error_df.to_csv(save_path, index=False)
    print(f"[OK] Saved: {save_path} ({len(errors)} errors)")
    
    return error_df


def generate_roc_curve(y_true, y_probs, model_name):
    """Generate and save ROC curve."""
    fpr, tpr, _ = roc_curve(y_true, y_probs[:, 1])
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"ROC (AUC = {roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {model_name}")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    save_path = RESULTS_DIR / f"roc_curve_{model_name.lower().replace(' ', '_')}.png"
    plt.savefig(save_path, dpi=100)
    plt.close()
    print(f"[OK] Saved: {save_path}")
    
    return roc_auc


def main():
    """Main evaluation pipeline."""
    print("\n" + "="*60)
    print("EVALUATION PIPELINE")
    print("="*60 + "\n")
    
    # Load test data
    print("Loading test data...")
    data_dir = Path(__file__).parent.parent / "data"
    test_df = pd.read_csv(data_dir / "cleaned_test.csv")
    y_test = test_df["label"].values
    X_test = test_df["statement"]
    
    print(f"Test set size: {len(test_df)}")
    print(f"Class distribution: {dict(zip(CLASS_NAMES, np.bincount(y_test)))}\n")

    # Baseline Evaluation
    print("\n" + "-"*60)
    print("Evaluating Baseline Model (Logistic Regression)")
    print("-"*60)
    try:
        train_df = pd.read_csv(data_dir / "cleaned_train.csv")
        X_train = train_df["statement"]
        y_train = train_df["label"].values

        baseline_preds, baseline_probs = predict_baseline(X_train, y_train, X_test)

        baseline_metrics, baseline_report = evaluate_model(
            y_test, baseline_preds, baseline_probs, "Baseline (LogReg)"
        )

        print(f"Accuracy: {baseline_metrics['Accuracy']:.4f}")
        print(f"Macro-F1: {baseline_metrics['Macro-F1']:.4f}")

        save_confusion_matrix(y_test, baseline_preds, "Baseline_LogReg")
        save_error_analysis(y_test, baseline_preds, X_test, "Baseline_LogReg")
        generate_roc_curve(y_test, baseline_probs, "Baseline_LogReg")

        print("[OK] Baseline evaluation complete")

    except Exception as e:
        print(f"[ERROR] Error evaluating baseline: {e}")
        baseline_metrics = None
    
    # Evaluate CNN
    print("\n" + "-"*60)
    print("Evaluating CNN Model")
    print("-"*60)
    try:
        cnn_model, vocab = load_cnn_model("best_cnn.pt")
        cnn_preds, cnn_probs = predict_cnn(cnn_model, X_test, vocab)
        cnn_metrics, cnn_report = evaluate_model(y_test, cnn_preds, cnn_probs, "Pure CNN")
        
        print(f"Accuracy: {cnn_metrics['Accuracy']:.4f}")
        print(f"Macro-F1: {cnn_metrics['Macro-F1']:.4f}")
        
        save_confusion_matrix(y_test, cnn_preds, "Pure_CNN")
        save_error_analysis(y_test, cnn_preds, X_test, "Pure_CNN")
        generate_roc_curve(y_test, cnn_probs, "Pure_CNN")
        print("[OK] CNN evaluation complete")
        
    except Exception as e:
        print(f"[ERROR] Error evaluating CNN: {e}")
        cnn_metrics = None
    
    # Evaluate DistilBERT
    print("\n" + "-"*60)
    print("Evaluating DistilBERT Model")
    print("-"*60)
    try:
        bert_model, tokenizer = load_bert_model("best_bert.pt")
        bert_preds, bert_probs = predict_bert(bert_model, X_test.reset_index(drop=True), tokenizer)
        bert_metrics, bert_report = evaluate_model(y_test, bert_preds, bert_probs, "DistilBERT")
        
        print(f"Accuracy: {bert_metrics['Accuracy']:.4f}")
        print(f"Macro-F1: {bert_metrics['Macro-F1']:.4f}")
        
        save_confusion_matrix(y_test, bert_preds, "DistilBERT")
        save_error_analysis(y_test, bert_preds, X_test, "DistilBERT")
        generate_roc_curve(y_test, bert_probs, "DistilBERT")
        print("[OK] DistilBERT evaluation complete")
        
    except Exception as e:
        print(f"[ERROR] Error evaluating DistilBERT: {e}")
        bert_metrics = None
    
    # Save detailed metrics
    metrics_list = []

    if baseline_metrics:
        metrics_list.append(baseline_metrics)
    if cnn_metrics:
        metrics_list.append(cnn_metrics)
    if bert_metrics:
        metrics_list.append(bert_metrics)

    if metrics_list:
        metrics_df = pd.DataFrame(metrics_list)
        save_path = RESULTS_DIR / "eval_metrics_comparison.csv"
        metrics_df.to_csv(save_path, index=False)

        print(f"\n[OK] Saved: {save_path}")
        print("\nMetrics Comparison:")
        print(metrics_df.to_string(index=False))
    
    print("\n" + "="*60)
    print("[DONE] Evaluation complete!")
    print("="*60)


if __name__ == "__main__":
    main()
