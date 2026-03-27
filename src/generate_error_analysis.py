"""
Error Analysis & Confusion Matrix Generation
Generates detailed error analysis for trained models.
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report
from transformers import DistilBertTokenizer

import sys
sys.path.insert(0, str(Path(__file__).parent))
from models.distilbert_model import DistilBERTClassifier
from torch.utils.data import DataLoader, TensorDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULTS_DIR = Path(__file__).parent.parent / "experiments" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CLASS_NAMES = ["FAKE", "REAL"]


def load_test_data():
    """Load test data."""
    data_dir = Path(__file__).parent.parent / "data"
    test_df = pd.read_csv(data_dir / "cleaned_test.csv")
    return test_df


def evaluate_distilbert():
    """Evaluate DistilBERT model with error analysis."""
    print("\n" + "="*70)
    print("ERROR ANALYSIS & CONFUSION MATRIX GENERATION")
    print("="*70 + "\n")
    
    # Load data
    print("[1/4] Loading test data...")
    test_df = load_test_data()
    y_test = test_df["label"].values
    X_test = test_df["statement"].values
    
    print(f"Test set: {len(test_df)} samples")
    print(f"Classes: FAKE={np.sum(y_test==0)}, REAL={np.sum(y_test==1)}\n")
    
    # Load model
    print("[2/4] Loading DistilBERT model...")
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    model = DistilBERTClassifier(num_classes=2, dropout_rate=0.2, label_smoothing=0.15)
    
    try:
        state = torch.load("best_bert.pt", map_location=device)
        # Remove criterion from state_dict if it exists
        if "criterion.weight" in state:
            del state["criterion.weight"]
        model.load_state_dict(state, strict=False)
    except Exception as e:
        print(f"[ERROR] Could not load best_bert.pt: {e}")
        print("[INFO] Attempting to load with strict=False...")
        model.load_state_dict(torch.load("best_bert.pt", map_location=device), strict=False)
    
    model.to(device)
    model.eval()
    print("[OK] Model loaded\n")
    
    # Tokenize test data
    print("[3/4] Running predictions...")
    encodings = tokenizer(
        X_test.tolist(),
        truncation=True,
        padding=True,
        max_length=128,
        return_tensors="pt"
    )
    
    # Create dataloader
    dataset = TensorDataset(
        encodings["input_ids"].to(device),
        encodings["attention_mask"].to(device)
    )
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    # Get predictions
    all_preds = []
    all_probs = []
    with torch.no_grad():
        for input_ids, attention_mask in loader:
            output = model(input_ids, attention_mask=attention_mask)
            logits = output["logits"] if isinstance(output, dict) else output
            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    y_pred = np.array(all_preds)
    y_probs = np.array(all_probs)
    
    # Calculate accuracy
    accuracy = (y_pred == y_test).mean()
    print(f"Accuracy: {accuracy:.4f}\n")
    
    # Generate Confusion Matrix
    print("[4/4] Generating visualizations...")
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, 
                cbar_kws={"label": "Count"})
    plt.title("Confusion Matrix - DistilBERT", fontsize=14, fontweight='bold')
    plt.ylabel("True Label", fontsize=12)
    plt.xlabel("Predicted Label", fontsize=12)
    plt.tight_layout()
    
    cm_path = RESULTS_DIR / "error_analysis_confusion_matrix.png"
    plt.savefig(cm_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved confusion matrix: {cm_path}")
    
    # Generate misclassification analysis
    errors = []
    error_indices = np.where(y_pred != y_test)[0]
    
    for idx in error_indices:
        errors.append({
            "Index": idx,
            "Original_Text": X_test[idx][:150] if len(X_test[idx]) > 150 else X_test[idx],
            "True_Label": CLASS_NAMES[y_test[idx]],
            "Predicted_Label": CLASS_NAMES[y_pred[idx]],
            "Confidence": f"{y_probs[idx].max():.4f}",
            "Correct_Class_Prob": f"{y_probs[idx][y_test[idx]]:.4f}"
        })
    
    error_df = pd.DataFrame(errors)
    error_csv = RESULTS_DIR / "error_analysis_misclassified_samples.csv"
    error_df.to_csv(error_csv, index=False)
    print(f"[OK] Saved error analysis: {error_csv} ({len(errors)} errors)")
    
    # Classification Report
    report = classification_report(y_test, y_pred, target_names=CLASS_NAMES, output_dict=True)
    report_text = classification_report(y_test, y_pred, target_names=CLASS_NAMES)
    
    report_path = RESULTS_DIR / "error_analysis_classification_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("Classification Report - DistilBERT\n")
        f.write("="*60 + "\n\n")
        f.write(report_text)
        f.write("\n\n" + "="*60 + "\n")
        f.write(f"Total Errors: {len(errors)} / {len(y_test)}")
        f.write(f"\nError Rate: {len(errors)/len(y_test):.2%}\n")
    
    print(f"[OK] Saved classification report: {report_path}\n")
    
    # Per-class error breakdown
    print("Per-Class Analysis:")
    print("-" * 60)
    for i, class_name in enumerate(CLASS_NAMES):
        class_mask = y_test == i
        class_errors = np.sum((y_test == i) & (y_pred != i))
        class_total = np.sum(class_mask)
        print(f"{class_name:10s}: {class_total:4d} samples, {class_errors:3d} errors ({100*class_errors/class_total:5.2f}%)")
    print("-" * 60 + "\n")
    
    # Generate detailed metrics CSV
    metrics_data = {
        "Metric": ["Total Samples", "Correct Predictions", "Errors", "Accuracy", "Precision (Macro)", "Recall (Macro)", "F1 (Macro)"],
        "Value": [
            len(y_test),
            np.sum(y_pred == y_test),
            len(errors),
            f"{accuracy:.4f}",
            f"{report['macro avg']['precision']:.4f}",
            f"{report['macro avg']['recall']:.4f}",
            f"{report['macro avg']['f1-score']:.4f}",
        ]
    }
    
    metrics_df = pd.DataFrame(metrics_data)
    metrics_path = RESULTS_DIR / "error_analysis_metrics_summary.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"[OK] Saved metrics summary: {metrics_path}\n")
    
    print("="*70)
    print("[DONE] Error analysis complete!")
    print("="*70)
    print("\nGenerated files:")
    print(f"  - {cm_path.name}")
    print(f"  - {error_csv.name}")
    print(f"  - {report_path.name}")
    print(f"  - {metrics_path.name}\n")


if __name__ == "__main__":
    evaluate_distilbert()
