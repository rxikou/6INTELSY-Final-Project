import torch
import torch.nn as nn
import torch.optim as optim

from models.cnn_model import CNNTextClassifier

# Dummy dataset parameters
vocab_size = 5000
embed_dim = 100
num_classes = 2

# Create model
model = CNNTextClassifier(vocab_size, embed_dim, num_classes)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Create fake training data
inputs = torch.randint(0, vocab_size, (32, 50))
labels = torch.randint(0, num_classes, (32,))

print("Starting training...\n")

for epoch in range(5):

    optimizer.zero_grad()

    outputs = model(inputs)

    loss = criterion(outputs, labels)

    loss.backward()

    optimizer.step()

    print(f"Epoch {epoch+1} Loss: {loss.item():.4f}")