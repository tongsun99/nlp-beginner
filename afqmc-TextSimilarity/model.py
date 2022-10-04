import logging
import torch

from torch import nn 
from transformers import AutoModel


class SemanticSimilarityNet(nn.Module):
    """Bert base model for semantic similarity classification.
    
    Inputs: Two sentences
    Output: Are they similar
    """

    def __init__(self, model_name):
        super(SemanticSimilarityNet, self).__init__()
        self.bert_encoder = AutoModel.from_pretrained(model_name)
        self.classifier = nn.Linear(768, 2)
    
    def forward(self, input_ids, attention_mask, token_type_ids):
        bert_outputs = self.bert_encoder(input_ids, attention_mask, token_type_ids)
        # [batch_size, seq_len, hid_dim]
        bert_output = bert_outputs[0]
        # [batch_size, hid_dim] -> [batch_size, 2]
        logits = self.classifier(bert_output[:, 0, :])
        logging.debug(f"logits shape {logits.shape}")
        return logits
