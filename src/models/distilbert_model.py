from transformers import DistilBertModel
import torch.nn as nn

class DistilBERTClassifier(nn.Module):

    def __init__(self, num_classes):
        super(DistilBERTClassifier, self).__init__()

        self.bert = DistilBertModel.from_pretrained("distilbert-base-uncased")

        self.dropout = nn.Dropout(0.3)

        self.classifier = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

        # ADD THIS (for loss computation)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, input_ids, attention_mask, labels=None):

        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        cls_output = outputs.last_hidden_state[:, 0]

        x = self.dropout(cls_output)
        logits = self.classifier(x)

        # MAKE IT COMPATIBLE WITH HF STYLE
        loss = None
        if labels is not None:
            loss = self.criterion(logits, labels)

        return {"loss": loss, "logits": logits}

