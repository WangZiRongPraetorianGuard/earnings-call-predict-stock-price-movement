from transformers import BertTokenizer, BertForPreTraining, BertConfig, LineByLineTextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments

# 加載原始的FinBERT tokenizer和模型
original_tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-pretrain')
model = BertForPreTraining.from_pretrained('yiyanghkust/finbert-pretrain')

# 假設您的數據集存儲在'train.txt'中
dataset = LineByLineTextDataset(
    tokenizer=original_tokenizer,
    file_path='train.txt',
    block_size=128
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=original_tokenizer,
    mlm=True,
    mlm_probability=0.15
)

training_args = TrainingArguments(
    output_dir='./results',
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_device_train_batch_size=8,
    save_steps=10_000,
    save_total_limit=2,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset
)

# 開始訓練
trainer.train()

# 保存微調後的tokenizer
new_tokenizer_path = './new_finbert_tokenizer'
original_tokenizer.save_pretrained(new_tokenizer_path)
