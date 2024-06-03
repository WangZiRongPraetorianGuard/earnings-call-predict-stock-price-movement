import torch
import torch.nn as nn
import time
import random
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, recall_score, precision_score
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from finBertCustomDataset import CustomDataset
from finBERTClassifier import FinBERTClassifier

class FinBERTTrainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = FinBERTClassifier.from_pretrained(args['config'], args).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args['learning_rate'], betas=(0.9, 0.999), eps=1e-9)
        self.loss_fct = nn.CrossEntropyLoss()
        
    def load_data(self, file_path, method):
        df = pd.read_csv(file_path)
        # 將 -1 轉換為 2
        # df['three_class_label'] = df['three_class_label'].apply(lambda x: x if x != -1 else 2)
        train_df, temp_df = train_test_split(df, random_state=1111, train_size=0.8)
        val_df, test_df = train_test_split(temp_df, random_state=1111, train_size=0.5)

        # 重設索引
        train_df = train_df.reset_index(drop=True)
        val_df = val_df.reset_index(drop=True)
        test_df = test_df.reset_index(drop=True)
        
        train_dataset = CustomDataset('train', train_df, 'paragraphs', self.args, method)
        val_dataset = CustomDataset('val', val_df, 'paragraphs', self.args, method)
        test_dataset = CustomDataset('test', test_df, 'paragraphs', self.args, method)
        
        train_loader = DataLoader(train_dataset, batch_size=self.args['batch_size'], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.args['batch_size'], shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=self.args['batch_size'], shuffle=False)
        
        return train_loader, val_loader, test_loader

    def cal_metrics(self, pred, labels):
        acc = (pred == labels).sum().item() / len(labels)
        f1 = f1_score(labels.cpu(), pred.cpu(), average='weighted')
        rec = recall_score(labels.cpu(), pred.cpu(), average='weighted')
        prec = precision_score(labels.cpu(), pred.cpu(), average='weighted')
        return acc, f1, rec, prec

    def get_pred(self, logits):
        return torch.argmax(logits, dim=1)

    def evaluate(self, val_loader):
        val_loss, val_acc, val_f1, val_rec, val_prec = 0.0, 0.0, 0.0, 0.0, 0.0
        step_count = 0
        self.model.eval()
        with torch.no_grad():
            loop = tqdm(val_loader, total=len(val_loader), leave=True, colour='blue')
            for data in loop:
                ids, masks, token_type_ids, labels = [t.to(self.device) for t in data]
                logits = self.model(input_ids=ids, token_type_ids=token_type_ids, attention_mask=masks)
                loss = self.loss_fct(logits, labels)
                val_loss += loss.item()
                preds = self.get_pred(logits)
                acc, f1, rec, prec = self.cal_metrics(preds, labels)
                val_acc += acc
                val_f1 += f1
                val_rec += rec
                val_prec += prec
                step_count += 1

        val_loss /= step_count
        val_acc /= step_count
        val_f1 /= step_count
        val_rec /= step_count
        val_prec /= step_count

        return val_loss, val_acc, val_f1, val_rec, val_prec


    def train(self, train_loader, val_loader):
        metrics = ['loss', 'acc', 'f1', 'rec', 'prec']
        mode = ['train_', 'val_']
        record = {s + m: [] for s in mode for m in metrics}

        for epoch in range(self.args["epochs"]):
            st_time = time.time()
            train_loss, train_acc, train_f1, train_rec, train_prec = 0.0, 0.0, 0.0, 0.0, 0.0
            step_count = 0

            self.model.train()
            for data in train_loader:
                ids, masks, token_type_ids, labels = [t.to(self.device) for t in data]
                self.optimizer.zero_grad()
                logits = self.model(input_ids=ids, token_type_ids=token_type_ids, attention_mask=masks)
                loss = self.loss_fct(logits, labels)
                loss.backward()
                self.optimizer.step()

                preds = self.get_pred(logits)
                acc, f1, rec, prec = self.cal_metrics(preds, labels)
                train_loss += loss.item()
                train_acc += acc
                train_f1 += f1
                train_rec += rec
                train_prec += prec
                step_count += 1

            train_loss /= step_count
            train_acc /= step_count
            train_f1 /= step_count
            train_rec /= step_count
            train_prec /= step_count

            print(f'[epoch {epoch + 1}] cost time: {time.time() - st_time:.4f} s')
            print('         loss     acc     f1      rec    prec')
            print(f'train | {train_loss:.4f}, {train_acc:.4f}, {train_f1:.4f}, {train_rec:.4f}, {train_prec:.4f}')

            # 评估模型在验证集上的表现
            val_loss, val_acc, val_f1, val_rec, val_prec = self.evaluate(val_loader)
            print(f'val   | {val_loss:.4f}, {val_acc:.4f}, {val_f1:.4f}, {val_rec:.4f}, {val_prec:.4f}\n')

            # 记录每个训练周期的指标
            record['train_loss'].append(train_loss)
            record['train_acc'].append(train_acc)
            record['train_f1'].append(train_f1)
            record['train_rec'].append(train_rec)
            record['train_prec'].append(train_prec)

            record['val_loss'].append(val_loss)
            record['val_acc'].append(val_acc)
            record['val_f1'].append(val_f1)
            record['val_rec'].append(val_rec)
            record['val_prec'].append(val_prec)

        return record
    
    def Softmax(x):
        return torch.exp(x) / torch.exp(x).sum()
    
    def predict_one(self, query):
        # 實例化一个 tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.args["config"])
        # Tokenize input query
        inputs = tokenizer.encode_plus(
            query,
            add_special_tokens=True,
            truncation=True,
            max_length=self.args["max_len"],
            padding='max_length',
            return_tensors='pt'
        )

        # Move input tensors to the appropriate device
        input_ids = inputs['input_ids'].to(self.device)
        attention_mask = inputs['attention_mask'].to(self.device)
        token_type_ids = inputs['token_type_ids'].to(self.device) if 'token_type_ids' in inputs else None

        # Perform inference
        self.model.eval()
        with torch.no_grad():
            logits = self.model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)

        # Calculate probabilities and predicted class
        probs = torch.softmax(logits, dim=-1).squeeze()
        pred = torch.argmax(probs).item()

        return probs, pred
    
    def predict(self, data_loader):
        total_probs, total_pred = [], []
        self.model.eval()
        with torch.no_grad():
            for data in tqdm(data_loader, total=len(data_loader), leave=True, colour='green'):
                input_ids, attention_mask, token_type_ids = [t.to(self.device) for t in data]

                logits = self.model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
                probs = torch.softmax(logits, dim=-1)  # get each class-probs
                preds = torch.argmax(probs, dim=-1)  # get the prediction for each sample in the batch

                total_probs.extend(probs.cpu().numpy())
                total_pred.extend(preds.cpu().numpy())

        return total_probs, total_pred

    