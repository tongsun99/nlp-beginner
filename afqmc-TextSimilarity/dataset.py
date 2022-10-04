import json
import logging
import torch

from torch.utils.data import Dataset


class AFQMCDataset(Dataset):
    """AFQMC train set and dev set

    AFQMC' dataset download https://storage.googleapis.com/cluebenchmark/tasks/afqmc_public.zip
    submit examples download https://storage.googleapis.com/cluebenchmark/tasks/clue_submit_examples.zip
    """

    def __init__(self, filename, tokenizer):
        sent1s = []; sent2s = []; labels = []
        with open(filename, "r") as f:
            for line in f:
                data = json.loads(line)
                sent1s.append(data["sentence1"])
                sent2s.append(data["sentence2"])
                labels.append(int(data["label"]))
        self.tokenized_output = tokenizer(sent1s, sent2s, padding=True, truncation=True, return_tensors="pt")
        self.labels = torch.tensor(labels)

    def __getitem__(self, idx):
        return (self.tokenized_output["input_ids"][idx],
                self.tokenized_output["attention_mask"][idx],
                self.tokenized_output["token_type_ids"][idx],
                self.labels[idx])

    def __len__(self):
        return len(self.labels)


class AFQMCDatasetNOLABEL(Dataset):
    """AFQMC test set
    
    no labels, you should submit your answer to https://www.cluebenchmarks.com/submit.html
    """

    def __init__(self, filename, tokenizer):
        sent1s = []; sent2s = []
        with open(filename, "r") as f:
            for line in f:
                data = json.loads(line)
                sent1s.append((data["sentence1"]))
                sent2s.append((data["sentence2"]))
        self.tokenized_output = tokenizer(sent1s, sent2s, padding=True, truncation=True, return_tensors="pt")

    def __getitem__(self, idx):
        return (self.tokenized_output["input_ids"][idx],
                self.tokenized_output["attention_mask"][idx],
                self.tokenized_output["token_type_ids"][idx],
                )

    def __len__(self):
        return len(self.tokenized_output["input_ids"])
