import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNTextClassifier(nn.Module):
    """
    Pure CNN text classifier with multi-kernel convolutions and pre-trained embeddings.
    
    Args:
        vocab_size (int): Vocabulary size
        embeddings (torch.Tensor): Pre-trained embedding matrix [vocab_size, embed_dim]
        embed_dim (int): Embedding dimension
        num_classes (int): Number of classification classes
        num_filters (int): Number of filters per kernel size
        kernel_sizes (list): List of kernel sizes
        hidden_dim (int): Not used (kept for backward compatibility)
        dropout_rate (float): Dropout probability
        use_batch_norm (bool): Whether to use batch normalization
    """

    def __init__(
        self,
        vocab_size,
        embeddings=None,
        embed_dim=100,
        num_classes=2,
        num_filters=100,
        kernel_sizes=None,
        hidden_dim=128,
        dropout_rate=0.3,
        use_batch_norm=True,
    ):
        super(CNNTextClassifier, self).__init__()
        
        if kernel_sizes is None:
            kernel_sizes = [3, 5, 7]

        # Embedding layer
        if embeddings is not None:
            self.embedding = nn.Embedding.from_pretrained(embeddings, freeze=False)
        else:
            self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        # Multi-kernel convolution layers (for local features)
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, num_filters, kernel_size=k, padding=k//2)
            for k in kernel_sizes
        ])
        
        # Batch normalization for conv outputs
        self.use_batch_norm = use_batch_norm
        if use_batch_norm:
            self.bn_layers = nn.ModuleList([
                nn.BatchNorm1d(num_filters)
                for _ in kernel_sizes
            ])
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout_rate)

        # Fully connected layers
        cnn_output_dim = len(kernel_sizes) * num_filters
        self.fc1 = nn.Linear(cnn_output_dim, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x, mask=None):
        """
        Forward pass.
        
        Args:
            x: Input token indices [batch_size, seq_length]
            mask: Padding mask [batch_size, seq_length], optional (not used in pure CNN)
        
        Returns:
            logits: Classification logits [batch_size, num_classes]
        """
        # Convert word indices to embeddings [batch_size, seq_length, embed_dim]
        x = self.embedding(x)

        # Change shape for CNN [batch_size, embed_dim, seq_length]
        x = x.permute(0, 2, 1)

        # Apply multi-kernel convolutions in parallel
        conv_outputs = []
        for i, conv in enumerate(self.convs):
            # Apply convolution and ReLU
            conv_out = F.relu(conv(x))  # [batch_size, num_filters, seq_length]
            
            # Apply batch normalization if enabled
            if self.use_batch_norm:
                conv_out = self.bn_layers[i](conv_out)
            
            conv_outputs.append(conv_out)

        # Concatenate all conv outputs [batch_size, total_filters, seq_length]
        x = torch.cat(conv_outputs, dim=1)
        
        # Global average pooling over sequence dimension [batch_size, total_filters]
        x = F.adaptive_avg_pool1d(x, 1).squeeze(-1)

        # Apply dropout
        x = self.dropout(x)

        # Fully connected layers with dropout
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x