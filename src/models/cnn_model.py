import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNTextClassifier(nn.Module):

    def __init__(self, vocab_size, embed_dim, num_classes):
        super(CNNTextClassifier, self).__init__()

        # Convert words to vectors
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        # Convolution layers
        self.conv1 = nn.Conv1d(embed_dim, 100, kernel_size=3)
        self.conv2 = nn.Conv1d(100, 100, kernel_size=3)

        # Final classification layer
        self.fc = nn.Linear(100, num_classes)

    def forward(self, x):

        # x shape: (batch_size, sequence_length)

        x = self.embedding(x)

        # change shape for CNN
        x = x.permute(0, 2, 1)

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        # Global max pooling
        x = torch.max(x, dim=2)[0]

        x = self.fc(x)

        return x