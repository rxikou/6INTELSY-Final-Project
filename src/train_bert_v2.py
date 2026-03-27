import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset
from transformers import DistilBertTokenizer, get_cosine_schedule_with_warmup

from models.distilbert_model import DistilBERTClassifier

# ── Device ────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ── Hyper-parameters (Full fine-tuning + Strong Regularization) ────
MAX_LENGTH    = 128     # optimized: shorter sequences improve focus
BATCH_SIZE    = 32      # larger batch for stable training
EPOCHS        = 20      # more epochs for small dataset
PATIENCE      = 6       # allow more training before stopping
WARMUP_RATIO  = 0.15    # longer warmup for small datasets

# Stronger regularization for small datasets (prevent overfitting)
DROPOUT       = 0.2     # increased dropout to reduce overfitting
LABEL_SMOOTH  = 0.15    # increased label smoothing for noise robustness
WEIGHT_DECAY  = 0.05    # strong L2 regularization

# Uniform learning rate for full fine-tuning
LR_DEFAULT    = 1.5e-5

# ── Data ──────────────────────────────────────────────────────────
df      = pd.read_csv("data/cleaned_train.csv")
test_df = pd.read_csv("data/cleaned_test.csv")

train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["label"])

# ── Class weights ─────────────────────────────────────────────────
label_counts = train_df["label"].value_counts().sort_index().values
class_weights = torch.tensor(
    len(train_df) / (len(label_counts) * label_counts),
    dtype=torch.float,
).to(device)
print(f"Class weights: {class_weights.cpu().numpy().round(3)}")


# ── Dataset ───────────────────────────────────────────────────────
class NewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.encodings = tokenizer(
            texts.tolist(),
            truncation=True,
            padding=True,
            max_length=MAX_LENGTH,
        )
        self.labels = labels.tolist()

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


# ── Tokenizer & loaders ───────────────────────────────────────────
print("Loading tokenizer…")
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

train_dataset = NewsDataset(train_df["statement"], train_df["label"], tokenizer)
val_dataset   = NewsDataset(val_df["statement"],   val_df["label"],   tokenizer)
test_dataset  = NewsDataset(test_df["statement"],  test_df["label"],  tokenizer)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0, pin_memory=False)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=False)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=False)

# ── Model ─────────────────────────────────────────────────────────
print("Loading model…")
num_classes = df["label"].nunique()
model = DistilBERTClassifier(
    num_classes=num_classes,
    dropout_rate=DROPOUT,
    label_smoothing=LABEL_SMOOTH,
)
# Inject class weights into the loss directly on top of label smoothing
model.criterion = torch.nn.CrossEntropyLoss(
    weight=class_weights,
    label_smoothing=LABEL_SMOOTH,
)
model.to(device)

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total     = sum(p.numel() for p in model.parameters())
print(f"Trainable parameters: {trainable:,} / {total:,}")


# ── Optimizer (Full fine-tuning + strong regularization) ──────────
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LR_DEFAULT,
    weight_decay=WEIGHT_DECAY,
)

total_steps  = len(train_loader) * EPOCHS
warmup_steps = int(total_steps * WARMUP_RATIO)

scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps,
)


# ── Evaluation helper ─────────────────────────────────────────────
def evaluate(loader):
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0.0
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out   = model(**batch)
            total_loss += out["loss"].item()
            preds = out["logits"].argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(batch["labels"].cpu().numpy())
    acc  = (np.array(all_preds) == np.array(all_labels)).mean()
    f1   = f1_score(all_labels, all_preds, average="macro")
    loss = total_loss / len(loader)
    return acc, f1, loss


# ── Training loop ─────────────────────────────────────────────────
print("\nTraining DistilBERT with Full Fine-Tuning + Strong Regularization…\n")

best_val_f1  = 0.0
best_epoch   = 0
patience_ctr = 0

for epoch in range(EPOCHS):
    model.train()
    total_train_loss = 0.0

    for batch in train_loader:
        batch  = {k: v.to(device) for k, v in batch.items()}
        out    = model(**batch)
        loss   = out["loss"]
        loss.backward()
        total_train_loss += out["loss"].item()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    avg_train_loss = total_train_loss / len(train_loader)
    val_acc, val_f1, val_loss = evaluate(val_loader)

    print(
        f"Epoch {epoch+1:2d}/{EPOCHS}  "
        f"train_loss: {avg_train_loss:.4f}  "
        f"val_loss: {val_loss:.4f}  "
        f"val_acc: {val_acc:.4f}  "
        f"val_f1: {val_f1:.4f}"
    )

    # Save best checkpoint based on macro-F1
    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        best_epoch  = epoch + 1
        patience_ctr = 0
        torch.save(model.state_dict(), "best_bert.pt")
        print(f"  ✓ New best saved (F1 {best_val_f1:.4f})")
    else:
        patience_ctr += 1
        if patience_ctr >= PATIENCE:
            print(f"\nEarly stopping at epoch {epoch+1} (no F1 improvement for {PATIENCE} epochs)")
            break

# ── Test evaluation ───────────────────────────────────────────────
print(f"\nBest epoch: {best_epoch}  |  Best val F1: {best_val_f1:.4f}")

model.load_state_dict(torch.load("best_bert.pt", map_location=device))
test_acc, test_f1, _ = evaluate(test_loader)

print(f"\nTest accuracy : {test_acc:.4f}")
print(f"Test macro-F1 : {test_f1:.4f}")
