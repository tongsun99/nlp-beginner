# In[1]
# import
import logging
import torch

from transformers.models.auto.tokenization_auto import AutoTokenizer
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - [%(filename)s:%(lineno)d] - %(message)s',
                    datefmt="%m/%d/%Y %H:%M:%S",
                    level=logging.DEBUG,
                    )

# In[2]
# dataset
from torch.utils.data.dataset import Dataset

id2label = {0: 'O', 1: 'B-LOC', 2: 'I-LOC', 3: 'B-ORG', 4: 'I-ORG', 5: 'B-PER', 6: 'I-PER'}
label2id = {'O': 0, 'B-LOC': 1, 'I-LOC': 2, 'B-ORG': 3, 'I-ORG': 4, 'B-PER': 5, 'I-PER': 6}

class ChinaPeopleDailyNER(Dataset):
    """cpdNER dataset
    download dataset here: http://s3.bmio.net/kashgari/china-people-daily-ner-corpus.tar.gz

    :attribute encoding["input_ids"]: [batch_size, seq_len]
    :attribute labels: [batch_size, seq_len]
    """
    def __init__(self, tokenizer, filename):
        sents, labels = self.read_data(filename)
        encoding, token_labels = self.tokenize_data(sents, labels, tokenizer)
        labels = torch.tensor(token_labels)
        self.encoding, self.labels = encoding, labels

    def read_data(self, filename):
        """read and preprocess data file, at char-level

        :param filename: e.g. "data/china-people-daily-ner-corpus/example.dev"
        :return sents: e.g. ["厦门海钓在福建", "中国警察"]
        :return labels: e.g. [
                                [(0, 2, "LOC", "厦门"), (5, 7, "LOC", "福建")]
                                [(0, 2, "LOC", "中国")]  
                            ]
        """
        with open(filename, 'r') as f:
            sents = []; labels = []
            for i, line in enumerate(f.read().split("\n\n")):  # line: "厦 o\n门 o\n ..."
                char_and_tag_s = line.split("\n") # char_and_tags [厦 o, 门 o, ...] 
                sent = ""; tags = [] # EXPECT: sent: "厦门海钓"; tags: [(0, 2, LOC)]

                j = 0; tag = (0, 0, "TYPE")
                while j < len(char_and_tag_s):
                    char, tag = char_and_tag_s[j].split(" ")
                    sent += char 
                    if tag == "O": j += 1; continue 
                    elif tag.startswith("B-"): 
                        # double pointer
                        st = j; ed = j + 1 
                        while ed < len(char_and_tag_s) and char_and_tag_s[ed].split(" ")[1] != "O": 
                            sent += char_and_tag_s[ed].split(" ")[0]
                            ed += 1
                        
                        tags.append((st, ed, tag[2:], sent[st:ed]))
                        j = ed

                sents.append(sent)
                labels.append(tags)
        return sents, labels 
    
    def tokenize_data(self, sents, labels, tokenizer):
        """preprocess data, from char-level to token_level

        :param sents: e.g. ["厦门海钓在福建", "中国警察"] 
        :param labels: e.g. [
                                [(0, 2, "LOC", "厦门"), (5, 7, "LOC", "福建")]
                                [(0, 2, "LOC", "中国")]  
                            ]
        :param tokenizer: 
        :return enocding:
        :return token_labels: e.g [[-100, 2, 3, ..., 2, 3, ...],
                                   [-100, 2, 3, 0, 0, ...]]
                             2-"B-LOC"; 3-"I-LOC"
                             每个元素对应一个token, 例如: 1999在char-level中对应4个元素, 在token-level对应一个元素
                             注意, [CLS], [SEP], [PAD]会映射到-100, 为了后面方便处理loss_fn以及输出
        """
        # TODO maybe return_offsets_mapping can help
        encoding = tokenizer(sents, padding=True, truncation=True, return_tensors="pt")
        token_labels = []
        for i, (sent, tags) in enumerate(zip(sents, labels)):
            token_tags = [0] * len(encoding["input_ids"][0])
            # TODO too slow!
            for j, input_id in enumerate(encoding["input_ids"][i].tolist()):
                if input_id == tokenizer.convert_tokens_to_ids("[CLS]") or \
                    input_id == tokenizer.convert_tokens_to_ids("[SEP]") or \
                    input_id == tokenizer.convert_tokens_to_ids("[PAD]"):
                        token_tags[j] = -100

            for st, ed, tag, tag_in_sent in tags:
                # st 对应的 token index; ed - 1(防止char_to_token == None)对应的 token index
                token_st = encoding.char_to_token(i, st)
                token_ed = encoding.char_to_token(i, ed - 1)
                token_tags[token_st] = label2id[f"B-{tag}"]
                if token_ed >= token_st + 1:
                    token_tags[token_st + 1 : token_ed + 1] =  [label2id[f"I-{tag}"]] * (token_ed - token_st)
            token_labels.append(token_tags)
        return encoding, token_labels
         
    def __getitem__(self, idx):
        return (
            {
                "input_ids": self.encoding["input_ids"][idx],
                "token_type_ids": self.encoding["token_type_ids"][idx],
                "attention_mask": self.encoding["attention_mask"][idx]
            },
            self.labels[idx]
        )

    def __len__(self):
        return self.labels.shape[0]

tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
demo_dataset = ChinaPeopleDailyNER(tokenizer, "data/china-people-daily-ner-corpus/example.demo")
train_dataset = ChinaPeopleDailyNER(tokenizer, "data/china-people-daily-ner-corpus/example.train")
dev_dataset = ChinaPeopleDailyNER(tokenizer, "data/china-people-daily-ner-corpus/example.dev")
test_dataset = ChinaPeopleDailyNER(tokenizer, "data/china-people-daily-ner-corpus/example.test")
# In[3]
# dataloader
from torch.utils.data.dataloader import DataLoader
demo_dataloader = DataLoader(demo_dataset, batch_size=1, shuffle=False)
train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
dev_dataloader = DataLoader(dev_dataset, batch_size=4, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=False)

# test
batch_x, batch_y = next(iter(demo_dataloader))
# In[4]
# Define model
from torch import nn
from transformers import BertPreTrainedModel, BertModel, AutoConfig

class BertForNER(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.init_weights()
    
    def forward(self, x):
        """BertForNer

        :param input_ids: [batch_size, seq_len]
        :return logits: [batch_size, seq_len, config.num_labels]
        """
        bert_output = self.bert(**x)[0] #   [batch_size, seq_len, hid_dim]
        logits = self.classifier(bert_output)   # [batch_size, seq_len, config.num_labels]
        return logits

# test
checkpoint = "bert-base-chinese"
config = AutoConfig.from_pretrained(checkpoint)
config.num_labels = len(id2label)
model = BertForNER(config)

batch_x, batch_y = next(iter(demo_dataloader))
logging.info(f"batch_x input_ids shape: {batch_x['input_ids'].shape}")
logits = model(batch_x)

for i in range(1):
    logging.info(f"sent {i}: {tokenizer.convert_ids_to_tokens(batch_x['input_ids'].tolist()[i])}")
    logging.info(f"logits {i}: {logits[i]}")        

# In[5]:
# Loss function
loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
logging.info(loss_fn(logits.permute(0, 2, 1), batch_y))
# In[6]:
# Predict labels
predicted_logits, predicted_labels = torch.max(logits, dim=2)
logging.debug(logits)
logging.debug(predicted_logits)
logging.debug(predicted_labels)
logging.debug(batch_y.shape)
logging.debug(predicted_labels.shape)
# In[7]:
# about seqeval
from seqeval.metrics import classification_report
from seqeval.scheme import IOB2

y_true = [['O', 'O', 'O', 'B-LOC', 'I-LOC', 'I-LOC', 'B-LOC', 'O'], ['B-PER', 'I-PER', 'O']]
y_pred = [['O', 'O', 'B-LOC', 'I-LOC', 'I-LOC', 'I-LOC', 'B-LOC', 'O'], ['B-PER', 'I-PER', 'O']]

print(classification_report(y_true, y_pred, mode='strict', scheme=IOB2))
# In[8]:
def compare(pred_labels, true_labels):
    """use seqeval predicted

    :param pred_labels: [batch_size, seq_len]
    :param true_labels: [batch_size, seq_len]
    :param input_ids: [batch_size, seq_len]
    :return pred_labels_seq: [batch_size, seq_len]
    :return true_labels_seq: [batch_size, seq_len]
    """    
    true_labels_seq = [[id2label[tag] for tag in true_label if tag != -100 ] for true_label in true_labels.tolist()]
    pred_labels_seq = [[id2label[pred_tag] for true_tag, pred_tag in zip(true_label, pred_label) if true_tag != -100] for true_label, pred_label in zip(true_labels.tolist(), pred_labels.tolist())]
    return pred_labels_seq, true_labels_seq

# test
pred_labels_seq, true_labels_seq = compare(predicted_labels, batch_y)
logging.debug(f"pred_labels_seq: {pred_labels_seq}, true_lables_seq: {true_lables_seq}")
logging.debug(len(pred_labels_seq[0]))
logging.debug(len(true_labels_seq[0]))
print(classification_report(true_labels_seq, pred_labels_seq,
                          mode="strict", scheme=IOB2))
