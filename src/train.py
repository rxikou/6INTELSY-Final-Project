import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

from models.cnn_model import CNNTextClassifier

# DEVICE
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Try to load FastText embeddings (will use TF-IDF or random if not available)
try:
    from gensim.models import FastText
    print("Loading FastText embeddings...")
    fasttext_available = True
    tfidf_available = False
except ImportError:
    print("FastText not available, will use TF-IDF embeddings")
    fasttext_available = False
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD
    tfidf_available = True

# LOAD DATA
train_df = pd.read_csv("data/cleaned_train.csv")
test_df = pd.read_csv("data/cleaned_test.csv")

# BUILD VOCABULARY
def build_vocab(texts, max_size=5000):
    words = " ".join(texts).split()
    most_common = Counter(words).most_common(max_size)
    vocab = {word: i+1 for i, (word, _) in enumerate(most_common)}
    return vocab

def encode(text, vocab, max_len=100):
    tokens = text.split()
    encoded = [vocab.get(word, 0) for word in tokens][:max_len]
    encoded += [0] * (max_len - len(encoded))
    return encoded

print("Building vocabulary...")
vocab = build_vocab(train_df["statement"])
vocab_size = len(vocab)

# CREATE EMBEDDINGS
print("Creating embeddings...")
embed_dim = 100

if fasttext_available:
    # Train FastText on the text data for word vectors
    sentences = [text.split() for text in train_df["statement"]]
    fasttext_model = FastText(sentences, vector_size=embed_dim, window=5, min_count=1, epochs=10, workers=4)
    
    # Create embedding matrix
    embedding_matrix = np.random.randn(vocab_size + 1, embed_dim) * 0.01
    embedding_matrix[0] = 0  # Padding token
    
    for word, idx in vocab.items():
        if word in fasttext_model.wv:
            embedding_matrix[idx] = fasttext_model.wv[word]
    
    print(f"✓ FastText embeddings created ({vocab_size} words)")
elif tfidf_available:
    # Use smart random initialization based on word frequency
    print("Creating frequency-informed embeddings...")
    
    # Calculate word frequencies
    all_words = " ".join(train_df["statement"]).split()
    word_freq = Counter(all_words)
    total_words = len(all_words)
    
    embedding_matrix = np.zeros((vocab_size + 1, embed_dim))
    embedding_matrix[0] = 0  # Padding token
    
    # Initialize based on frequency - rare words get smoother initialization
    for word, idx in vocab.items():
        freq = word_freq.get(word, 1)
        freq_ratio = freq / total_words
        # Inverse frequency scaling: common words have tighter initialization
        scale = 0.1 / np.sqrt(freq_ratio + 1e-4)
        embedding_matrix[idx] = np.random.randn(embed_dim) * min(scale, 0.1)
    
    print(f"✓ Frequency-informed embeddings created ({vocab_size} words)")
else:
    # Use random embeddings if nothing else is available
    embedding_matrix = np.random.randn(vocab_size + 1, embed_dim) * 0.01
    embedding_matrix[0] = 0
    print(">> Using random embeddings")

embeddings_tensor = torch.from_numpy(embedding_matrix).float()

# ENCODE DATA
X = torch.tensor([encode(text, vocab) for text in train_df["statement"]])
y = torch.tensor(train_df["label"].values)

X_test = torch.tensor([encode(text, vocab) for text in test_df["statement"]])
y_test = torch.tensor(test_df["label"].values)

# Function to create padding mask (1 for real tokens, 0 for padding)
def create_mask(x, pad_idx=0):
    return (x != pad_idx).float()

# TRAIN / VALIDATION SPLIT
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Move to device
X_train = X_train.to(device)
y_train = y_train.to(device)
X_val = X_val.to(device)
y_val = y_val.to(device)
X_test = X_test.to(device)
y_test = y_test.to(device)
embeddings_tensor = embeddings_tensor.to(device)

# DATALOADER with custom collate to handle masks
def collate_fn(batch):
    X_batch = torch.stack([item[0] for item in batch])
    y_batch = torch.stack([item[1] for item in batch])
    masks = create_mask(X_batch)
    return X_batch, y_batch, masks

train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=collate_fn)

# MODEL SETUP
print("Loading pure CNN text classifier...", flush=True)
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
model.to(device)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters: {total_params:,} | Trainable: {trainable_params:,}")

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0005)

# Learning rate scheduler
total_steps = len(train_loader) * 30
warmup_lr_scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=int(0.1 * total_steps))
cos_lr_scheduler = CosineAnnealingLR(optimizer, T_max=int(0.9 * total_steps))
scheduler = SequentialLR(optimizer, schedulers=[warmup_lr_scheduler, cos_lr_scheduler], milestones=[int(0.1 * total_steps)])

# Early stopping setup
best_val_acc = 0
best_epoch = 0
patience = 8
counter = 0

# TRAINING LOOP
print("\nTraining pure CNN with pre-trained embeddings...\n", flush=True)

epochs = 30

for epoch in range(epochs):

    model.train()
    total_loss = 0

    for batch_X, batch_y, masks in train_loader:
        # Data already on device
        optimizer.zero_grad()

        outputs = model(batch_X, mask=masks)
        loss = criterion(outputs, batch_y)

        loss.backward()

        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)

    # VALIDATION - compute masks on the fly
    model.eval()
    with torch.no_grad():
        val_masks = create_mask(X_val).to(device)
        val_preds = model(X_val, mask=val_masks).argmax(dim=1)
        val_acc = (val_preds == y_val).float().mean().item()

    print(f"Epoch {epoch+1:2d}/{epochs}, Loss: {avg_loss:.4f}, Val Acc: {val_acc:.4f}", flush=True)

    # EARLY STOPPING
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_epoch = epoch + 1
        counter = 0
        torch.save(model.state_dict(), "best_cnn.pt")
    else:
        counter += 1
        if counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

# Load best model
model.load_state_dict(torch.load("best_cnn.pt"))
print(f"\nBest Epoch: {best_epoch}, Best Val Acc: {best_val_acc:.4f}")

# FINAL TEST EVALUATION
print("\nEvaluating advanced model on test set...\n")

model.eval()
with torch.no_grad():
    test_masks = create_mask(X_test).to(device)
    preds = model(X_test, mask=test_masks).argmax(dim=1)
    accuracy = (preds == y_test).float().mean().item()

print(f"Pure CNN Test Accuracy: {accuracy:.4f}")