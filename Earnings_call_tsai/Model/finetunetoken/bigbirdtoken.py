import torch
from transformers import BertTokenizer, BigBirdTokenizer, BigBirdForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from torch.utils.data import TensorDataset

# 加載微調後的FinBERT tokenizer
finbert_tokenizer = BertTokenizer.from_pretrained('./new_finbert_tokenizer')

# 加載BigBird tokenizer和分類模型
bigbird_tokenizer = BigBirdTokenizer.from_pretrained('google/bigbird-roberta-base')
model = BigBirdForSequenceClassification.from_pretrained('google/bigbird-roberta-base')

# 假設您的文本為`text`
text = "example text"
tokens = finbert_tokenizer.tokenize(text)
token_ids = finbert_tokenizer.convert_tokens_to_ids(tokens)
bigbird_token_ids = bigbird_tokenizer.convert_tokens_to_ids(tokens)

input_ids = torch.tensor(bigbird_token_ids).unsqueeze(0)  # batch size 1

# 將輸入張量傳入模型進行分類
outputs = model(input_ids)
logits = outputs.logits

# 獲取分類結果
predicted_class = torch.argmax(logits, dim=1).item()
print(f"Predicted class: {predicted_class}")

# 假設您的數據存儲在一個列表中，每個元素是一個元組(text, label)
data = [("example text 1", 0), ("example text 2", 1), ...]
texts, labels = zip(*data)

# 將數據切分為訓練集和測試集
train_texts, test_texts, train_labels, test_labels = train_test_split(texts, labels, test_size=0.2, random_state=42)

def encode_texts(tokenizer, texts):
    input_ids = []
    for text in texts:
        tokens = tokenizer.tokenize(text)
        token_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_ids.append(token_ids)
    return input_ids

# 使用BigBird tokenizer對訓練集和測試集進行編碼
train_input_ids = encode_texts(bigbird_tokenizer, train_texts)
test_input_ids = encode_texts(bigbird_tokenizer, test_texts)

# 創建TensorDataset
train_dataset = TensorDataset(torch.tensor(train_input_ids, dtype=torch.long), torch.tensor(train_labels, dtype=torch.long))
test_dataset = TensorDataset(torch.tensor(test_input_ids, dtype=torch.long), torch.tensor(test_labels, dtype=torch.long))

# 訓練參數設置
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
)

# 創建Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset
)

# 開始訓練
trainer.train()

# 在測試集上進行預測和評估
predictions = trainer.predict(test_dataset).predictions
pred_labels = torch.argmax(torch.tensor(predictions), dim=1).numpy()
accuracy = accuracy_score(test_labels, pred_labels)

print(f"Accuracy: {accuracy}")
