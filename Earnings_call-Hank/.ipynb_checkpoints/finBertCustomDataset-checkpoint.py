import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import random
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import torch.nn.functional as F

class CustomDataset(Dataset):
    def __init__(self, mode, df, specify, args, method):
        assert mode in ["train", "test"]
        self.mode = mode
        self.df = df.reset_index(drop=True)  # 確保索引連續
        self.specify = specify
        self.method = method  # 新增 method 参数
        if self.mode != 'test':
            self.label = df['three_class_label'].apply(lambda x: x if x != -1 else 2).reset_index(drop=True)  # 將 -1 轉換為 2 並重置索引
        self.tokenizer = AutoTokenizer.from_pretrained(args["config"])
        self.max_len = args["max_len"]
        self.num_class = args["num_class"]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        sentence = str(self.df[self.specify][index])
        ids, mask, token_type_ids = self.tokenize(sentence)

        if self.mode == "test":
            return torch.tensor(ids, dtype=torch.long), torch.tensor(mask, dtype=torch.long), \
                   torch.tensor(token_type_ids, dtype=torch.long)
        else:
            if self.num_class > 2:
                return torch.tensor(ids, dtype=torch.long), torch.tensor(mask, dtype=torch.long), \
                       torch.tensor(token_type_ids, dtype=torch.long), torch.tensor(self.label[index], dtype=torch.long)
            else:
                return torch.tensor(ids, dtype=torch.long), torch.tensor(mask, dtype=torch.long), \
                       torch.tensor(token_type_ids, dtype=torch.long), torch.tensor(self.label[index], dtype=torch.float)

    def one_hot_label(self, label):
        return F.one_hot(torch.tensor(label), num_classes=self.num_class)

    def tokenize(self, input_text):
        """
        以前、後、隨機的方式進行tokenzize，max_length是512tokens
        """
        if self.method == "first":
            inputs = self.tokenizer(input_text, max_length=self.max_len, padding="max_length", truncation=True, return_tensors="pt")
        elif self.method == "last":
            tokens = self.tokenizer(input_text, return_tensors="pt", truncation=False)["input_ids"].squeeze(0).tolist()
            tokens = tokens[-self.max_len:]
            inputs = self.tokenizer.decode(tokens, return_tensors="pt", truncation=False)
        elif self.method == "random":
            tokens = self.tokenizer(input_text, return_tensors="pt", truncation=False)["input_ids"].squeeze(0).tolist()
            if len(tokens) > self.max_len:
                start = random.randint(0, len(tokens) - self.max_len)
                tokens = tokens[start:start + self.max_len]
            inputs = self.tokenizer.decode(tokens, return_tensors="pt", truncation=False)

        input_ids = inputs["input_ids"].squeeze(0)
        attention_mask = inputs["attention_mask"].squeeze(0)
        token_type_ids = inputs["token_type_ids"].squeeze(0) if "token_type_ids" in inputs else None

        return input_ids, attention_mask, token_type_ids


