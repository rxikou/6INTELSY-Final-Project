from transformers import DistilBertModel
import torch.nn as nn

class DistilBERTClassifier(nn.Module):

    def __init__(self, num_classes):
        super(DistilBERTClassifier, self).__init__()

        # Load pretrained DistilBERT
        self.bert = DistilBertModel.from_pretrained("distilbert-base-uncased")

        # Classification layer
        self.classifier = nn.Linear(768, num_classes)

    def forward(self, input_ids, attention_mask):

        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # CLS token representation
        cls_output = outputs.last_hidden_state[:, 0]

        logits = self.classifier(cls_output)

        return logits