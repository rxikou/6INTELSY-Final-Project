import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader

from models.cnn_model import CNNTextClassifier

# LOAD DATA
train_df = pd.read_csv("data/cleaned_train.csv")
test_df = pd.read_csv("data/cleaned_test.csv")

# BUILD VOCABULARY
def build_vocab(texts, max_size=5000):
    words = " ".join(texts).split()
    most_common = Counter(words).most_common(max_size)
    vocab = {word: i+1 for i, (word, _) in enumerate(most_common)}
    return vocab

def encode(text, vocab, max_len=100):  # increased length
    tokens = text.split()
    encoded = [vocab.get(word, 0) for word in tokens][:max_len]
    encoded += [0] * (max_len - len(encoded))
    return encoded

print("Building vocabulary...")
vocab = build_vocab(train_df["statement"])
vocab_size = len(vocab)

# ENCODE DATA
X = torch.tensor([encode(text, vocab) for text in train_df["statement"]])
y = torch.tensor(train_df["label"].values)

X_test = torch.tensor([encode(text, vocab) for text in test_df["statement"]])
y_test = torch.tensor(test_df["label"].values)

# TRAIN / VALIDATION SPLIT
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# DATALOADER
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# MODEL SETUP
model = CNNTextClassifier(vocab_size + 1, 100, 2)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005)  # lower LR

# TRAINING LOOP
print("\nTraining CNN with validation...\n")

epochs = 8

for epoch in range(epochs):

    model.train()
    total_loss = 0

    for batch_X, batch_y in train_loader:

        optimizer.zero_grad()

        outputs = model(batch_X)

        loss = criterion(outputs, batch_y)

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)

    # VALIDATION
    model.eval()
    with torch.no_grad():
        val_preds = model(X_val).argmax(dim=1)
        val_acc = (val_preds == y_val).float().mean()

    print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Val Acc: {val_acc:.4f}")

# FINAL TEST EVALUATION
print("\nEvaluating CNN on test set...\n")

with torch.no_grad():
    preds = model(X_test).argmax(dim=1)
    accuracy = (preds == y_test).float().mean()

print(f"CNN Test Accuracy: {accuracy.item():.4f}")