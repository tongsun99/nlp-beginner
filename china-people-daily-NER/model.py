from torch import nn
from transformers import BertPreTrainedModel, BertModel, AutoConfig

class BertForNER(BertPreTrainedModel):
    def __init__(self, config):
        """继承BertPreTrainedModel, 利用config初始化

        :param config: 
        """
        super().__init__(config)
        self.bert = BertModel(config)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.init_weights()
    
    def forward(self, x):
        """

        :param input_ids: [batch_size, seq_len]
        :return logits: [batch_size, seq_len, config.num_labels]
        """
        bert_output = self.bert(**x)[0] #   [batch_size, seq_len, hid_dim]
        logits = self.classifier(bert_output)   # [batch_size, seq_len, config.num_labels]
        return logits