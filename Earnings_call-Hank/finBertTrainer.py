import torch
import torch.nn as nn
import time
import random
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, recall_score, precision_score
from torch.utils.data import DataLoader
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
        df['three_class_label'] = df['three_class_label'].apply(lambda x: x if x != -1 else 2)
        train_df, test_df = train_test_split(df, random_state=1111, train_size=0.8)

        # 重設索引
        train_df = train_df.reset_index(drop=True)
        test_df = test_df.reset_index(drop=True)
        
        train_dataset = CustomDataset('train', train_df, 'paragraphs', self.args, method)
        test_dataset = CustomDataset('test', test_df, 'paragraphs', self.args, method)
        
        train_loader = DataLoader(train_dataset, batch_size=self.args['batch_size'], shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=self.args['batch_size'], shuffle=False)
        
        return train_loader, test_loader

    def cal_metrics(self, pred, labels):
        acc = (pred == labels).sum().item() / len(labels)
        f1 = f1_score(labels.cpu(), pred.cpu(), average='weighted')
        rec = recall_score(labels.cpu(), pred.cpu(), average='weighted')
        prec = precision_score(labels.cpu(), pred.cpu(), average='weighted')
        return acc, f1, rec, prec

    def get_pred(self, logits):
        return torch.argmax(logits, dim=1)

    def test(self, test_loader):
        self.model.eval()
        test_loss, test_acc, test_f1, test_rec, test_prec = 0.0, 0.0, 0.0, 0.0, 0.0
        step_count = 0
        with torch.no_grad():
            for data in test_loader:
                ids, masks, token_type_ids, labels = [t.to(self.device) for t in data]
                logits = self.model(input_ids=ids, token_type_ids=token_type_ids, attention_mask=masks)
                loss = self.loss_fct(logits, labels)
                acc, f1, rec, prec = self.cal_metrics(self.get_pred(logits), labels)
                test_loss += loss.item()
                test_acc += acc
                test_f1 += f1
                test_rec += rec
                test_prec += prec
                step_count += 1
        test_loss = test_loss / step_count
        test_acc = test_acc / step_count
        test_f1 = test_f1 / step_count
        test_rec = test_rec / step_count
        test_prec = test_prec / step_count
        return test_loss, test_acc, test_f1, test_rec, test_prec

    def train(self, train_loader):
        metrics = ['loss', 'acc', 'f1', 'rec', 'prec']
        record = {m: [] for m in metrics}
        
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

                acc, f1, rec, prec = self.cal_metrics(self.get_pred(logits), labels)
                train_loss += loss.item()
                train_acc += acc
                train_f1 += f1
                train_rec += rec
                train_prec += prec
                step_count += 1

            train_loss = train_loss / step_count
            train_acc = train_acc / step_count
            train_f1 = train_f1 / step_count
            train_rec = train_rec / step_count
            train_prec = train_prec / step_count

            print('[epoch %d] cost time: %.4f s' % (epoch + 1, time.time() - st_time))
            print('         loss     acc     f1      rec    prec')
            print('train | %.4f, %.4f, %.4f, %.4f, %.4f' % (train_loss, train_acc, train_f1, train_rec, train_prec))

            record['loss'].append(train_loss)
            record['acc'].append(train_acc)
            record['f1'].append(train_f1)
            record['rec'].append(train_rec)
            record['prec'].append(train_prec)
        
        return record