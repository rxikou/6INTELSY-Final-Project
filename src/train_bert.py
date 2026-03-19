import pandas as pd
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split

# LOAD DATA
df = pd.read_csv("data/cleaned_train.csv")
test_df = pd.read_csv("data/cleaned_test.csv")

# SPLIT TRAIN / VALIDATION
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

# DATASET CLASS
class NewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.encodings = tokenizer(
            texts.tolist(),
            truncation=True,
            padding=True,
            max_length=128
        )
        self.labels = labels.tolist()

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

# TOKENIZER
print("Loading tokenizer...")
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

# DATASETS
train_dataset = NewsDataset(train_df["statement"], train_df["label"], tokenizer)
val_dataset = NewsDataset(val_df["statement"], val_df["label"], tokenizer)
test_dataset = NewsDataset(test_df["statement"], test_df["label"], tokenizer)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8)
test_loader = DataLoader(test_dataset, batch_size=8)

# MODEL
print("Loading model...")
model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=2
)

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5)  # slightly lower LR

# TRAINING WITH VALIDATION
print("\nTraining DistilBERT with validation...\n")

epochs = 2  # safe upgrade

for epoch in range(epochs):

    # TRAIN
    model.train()
    total_loss = 0

    for batch in train_loader:
        optimizer.zero_grad()

        outputs = model(**batch)
        loss = outputs.loss

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)

    # VALIDATION
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in val_loader:
            outputs = model(**batch)
            preds = outputs.logits.argmax(dim=1)

            correct += (preds == batch["labels"]).sum().item()
            total += len(batch["labels"])

    val_acc = correct / total

    print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}, Val Acc: {val_acc:.4f}")

# FINAL TEST EVALUATION
print("\nEvaluating DistilBERT...\n")

model.eval()
correct = 0
total = 0

with torch.no_grad():
    for batch in test_loader:
        outputs = model(**batch)
        preds = outputs.logits.argmax(dim=1)

        correct += (preds == batch["labels"]).sum().item()
        total += len(batch["labels"])

accuracy = correct / total

print(f"DistilBERT Test Accuracy: {accuracy:.4f}")