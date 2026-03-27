import torch
import torch.nn as nn
from transformers import DistilBertModel


class DistilBERTClassifier(nn.Module):

    def __init__(self, num_classes: int, dropout_rate: float = 0.1, label_smoothing: float = 0.1):
        super().__init__()

        self.bert = DistilBertModel.from_pretrained("distilbert-base-uncased")

        # CLS (768) + mean pool (768) = 1536-dim input to the head
        hidden_size = self.bert.config.hidden_size  # 768
        combined_size = hidden_size * 2             # 1536

        self.classifier = nn.Sequential(
            nn.Linear(combined_size, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout_rate),

            nn.Linear(512, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout_rate),

            nn.Linear(128, num_classes),
        )

        # Label smoothing regularizes noisy datasets (news labels often have inter-annotator noise)
        self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

        self._init_classifier_weights()

    def _init_classifier_weights(self):
        """Kaiming init on linear layers; ones/zeros on LayerNorm."""
        for module in self.classifier.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def _pool(self, last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Combine CLS token with attention-masked mean pooling.
        Both vectors are L2-normalised before concatenation so neither dominates.
        """
        cls_output = last_hidden_state[:, 0]  # (B, 768)

        # Mean pool: expand mask to match hidden dim, zero out padding tokens
        mask_expanded = attention_mask.unsqueeze(-1).float()  # (B, L, 1)
        sum_hidden = (last_hidden_state * mask_expanded).sum(dim=1)     # (B, 768)
        count = mask_expanded.sum(dim=1).clamp(min=1e-9)                # (B, 1)
        mean_output = sum_hidden / count                                 # (B, 768)

        # L2-normalise both before concat to keep scales balanced
        cls_norm  = nn.functional.normalize(cls_output,  p=2, dim=-1)
        mean_norm = nn.functional.normalize(mean_output, p=2, dim=-1)

        return torch.cat([cls_norm, mean_norm], dim=-1)  # (B, 1536)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> dict:
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        pooled = self._pool(outputs.last_hidden_state, attention_mask)
        logits = self.classifier(pooled)

        loss = None
        if labels is not None:
            loss = self.criterion(logits, labels)

        return {"loss": loss, "logits": logits}