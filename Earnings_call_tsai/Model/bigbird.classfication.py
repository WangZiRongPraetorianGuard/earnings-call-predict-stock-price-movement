import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BigBirdTokenizer, BigBirdForSequenceClassification, Trainer, TrainingArguments
from transformers import EvalPrediction
import torch
from torch.utils.data import Dataset
import numpy as np

# 讀取CSV文件
df = pd.read_csv(r'C:\Users\user\earnings-call-predict-stock-price-movement\Earnings_call_tsai\Preprocessing\preprocessed_0522_price_volatility_binary_label_ver2_big.csv')

# 將標籤從1和-1轉換為0和1
df['label'] = df['label'].map({1: 1, -1: 0})

# 檢查標籤範圍
print("Unique labels in the dataset:", df['label'].unique())

# 分割數據集為訓練集和測試集
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# 加載Tokenizer和模型
tokenizer = BigBirdTokenizer.from_pretrained('google/bigbird-roberta-base')
model = BigBirdForSequenceClassification.from_pretrained('google/bigbird-roberta-base', num_labels=2)  # 假設是二分類

# 創建自定義Dataset類
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=4096):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
            padding='max_length'  # 填充到最大長度
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# 創建Dataset對象
train_dataset = TextDataset(
    texts=train_df['paragraphs'].to_list(),
    labels=train_df['label'].to_list(),
    tokenizer=tokenizer,
    max_length=4096
)

test_dataset = TextDataset(
    texts=test_df['paragraphs'].to_list(),
    labels=test_df['label'].to_list(),
    tokenizer=tokenizer,
    max_length=4096
)

# 定義計算準確率的函數
def compute_metrics(p: EvalPrediction):
    preds = np.argmax(p.predictions, axis=1)
    return {"accuracy": (preds == p.label_ids).mean()}

# 定義訓練參數
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=2,  # 調整batch size以適應大長度文本
    per_device_eval_batch_size=2,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)

# 創建Trainer對象
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics
)

# 開始訓練
trainer.train()

# 保存模型
model.save_pretrained('./saved_model')
tokenizer.save_pretrained('./saved_model')

# 評估模型準確率
metrics = trainer.evaluate()
print(f"Accuracy: {metrics['eval_accuracy']}")
