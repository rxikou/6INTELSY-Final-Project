import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNTextClassifier(nn.Module):

    def __init__(self, vocab_size, embed_dim, num_classes):
        super(CNNTextClassifier, self).__init__()

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        # Convolution layers
        self.conv1 = nn.Conv1d(embed_dim, 100, kernel_size=3)
        self.conv2 = nn.Conv1d(100, 100, kernel_size=3)

        # Dropout layer (NEW)
        self.dropout = nn.Dropout(0.5)

        # Fully connected layer
        self.fc = nn.Linear(100, num_classes)

    def forward(self, x):

        # Convert word indices to embeddings
        x = self.embedding(x)

        # Change shape for CNN
        x = x.permute(0, 2, 1)

        # Apply convolution + activation
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        # Global max pooling
        x = torch.max(x, dim=2)[0]

        # Apply dropout (NEW)
        x = self.dropout(x)

        # Final classification
        x = self.fc(x)

        return x