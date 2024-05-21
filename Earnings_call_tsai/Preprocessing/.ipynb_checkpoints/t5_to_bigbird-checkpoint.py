import os
from transformers import BigBirdTokenizer, BigBirdForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import torch
import random

# 加載T5處理後的文本並準備數據集
def load_data_from_directory(directory_path):
    texts = []
    labels = []
    for filename in os.listdir(directory_path):
        if filename.endswith("_t5_processed.txt"):
            file_path = os.path.join(directory_path, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()
                texts.append(text)
                labels.append(random.choice([0, 1]))  # 隨機標籤0或1
    return texts, labels

def main():
    # 設置資料夾路徑
    t5_processed_directory = r'D:\EarningsCall_finalProject\Earnings_call\Datasets\transcript_text_preprocessing_t5'  # 替換為您的T5處理後資料夾路徑

    # 加載T5處理後的文本和標籤
    texts, labels = load_data_from_directory(t5_processed_directory)

    # 加載BigBird tokenizer和模型
    model_name = 'google/bigbird-roberta-base'
    tokenizer = BigBirdTokenizer.from_pretrained(model_name)
    model = BigBirdForSequenceClassification.from_pretrained(model_name, num_labels=2)

    # 使用datasets庫準備數據集
    encodings = tokenizer(texts, truncation=True, padding=True, max_length=512)
    dataset = Dataset.from_dict({
        'input_ids': encodings['input_ids'],
        'attention_mask': encodings['attention_mask'],
        'labels': labels
    })

    # 設置訓練參數
    training_args = TrainingArguments(
        output_dir='./results',          # 輸出目錄
        num_train_epochs=3,              # 訓練的總輪次
        per_device_train_batch_size=4,   # 每個設備的訓練batch大小
        warmup_steps=500,                # 進行學習率預熱的步數
        weight_decay=0.01,               # 學習率衰減
        logging_dir='./logs',            # 日誌目錄
        logging_steps=10,
    )

    # 創建Trainer實例
    trainer = Trainer(
        model=model,                         # 預訓練的模型
        args=training_args,                  # 訓練參數
        train_dataset=dataset,               # 訓練數據集
        eval_dataset=dataset,                # 驗證數據集（這裡使用相同的數據集，您應該拆分數據集進行驗證）
    )

    # 開始訓練
    trainer.train()

if __name__ == "__main__":
    main()
