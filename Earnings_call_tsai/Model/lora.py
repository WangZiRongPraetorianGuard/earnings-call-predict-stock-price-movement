import torch
import torch.nn as nn
import pandas as pd
from transformers import BigBirdTokenizer, BigBirdForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import loralib as lora
import gc

class DoubleLayerTransformer:
    def __init__(self, bigbird_model_name='google/bigbird-roberta-base', device='cpu'):
        self.bigbird_tokenizer = BigBirdTokenizer.from_pretrained(bigbird_model_name)
        self.bigbird_model = BigBirdForSequenceClassification.from_pretrained(bigbird_model_name).to(device)
        self.device = device
        self.add_lora()

    def add_lora(self):
        lora_modules = []
        for name, module in self.bigbird_model.named_modules():
            if isinstance(module, nn.Linear):
                lora_modules.append((name, module))
        
        for name, module in lora_modules:
            lora_module = lora.Linear(module.in_features, module.out_features, r=8).to(self.device)
            lora_module.weight.data = module.weight.data.to(self.device)
            if module.bias is not None:
                lora_module.bias = module.bias.to(self.device)
            setattr(self.bigbird_model, name, lora_module)

    def tokenize_with_bigbird(self, text):
        tokens = self.bigbird_tokenizer.tokenize(text)
        token_ids = self.bigbird_tokenizer.convert_tokens_to_ids(tokens)
        return token_ids

    def process_document(self, document):
        # 使用BigBird进行tokenization
        bigbird_token_ids = self.tokenize_with_bigbird(document)
        if len(bigbird_token_ids) == 0:
            return torch.empty(0)
        
        # 确保输入的长度不超过BigBird的最大长度4096
        if len(bigbird_token_ids) > 4096:
            bigbird_token_ids = bigbird_token_ids[:4096]
        
        # 构建输入张量
        input_ids = torch.tensor(bigbird_token_ids).unsqueeze(0).to(self.device)  # batch size 1
        attention_mask = torch.ones(input_ids.shape, device=self.device)

        return input_ids, attention_mask

def process_documents(documents, transformer, max_length=4096):
    inputs = []
    masks = []
    for i, document in enumerate(documents):
        print(f"Processing document {i+1}/{len(documents)}")
        input_ids, attention_mask = transformer.process_document(document)
        if input_ids.size(0) != 0:  # 确认input_ids不是空的
            # Padding to max_length
            padding_length = max_length - input_ids.size(1)
            if padding_length > 0:
                input_ids = torch.cat([input_ids, torch.zeros((1, padding_length), dtype=torch.long).to(transformer.device)], dim=1)
                attention_mask = torch.cat([attention_mask, torch.zeros((1, padding_length), dtype=torch.long).to(transformer.device)], dim=1)
            inputs.append(input_ids)
            masks.append(attention_mask)
        # 清除中间变量以释放内存
        del input_ids, attention_mask
        gc.collect()
    return torch.cat(inputs), torch.cat(masks)

def train_model(model, optimizer, criterion, train_inputs, train_masks, train_labels, batch_size=8):
    model.train()
    total_loss = 0
    for i in range(0, len(train_inputs), batch_size):
        input_batch = train_inputs[i:i+batch_size].to(model.device)
        mask_batch = train_masks[i:i+batch_size].to(model.device)
        label_batch = torch.tensor(train_labels[i:i+batch_size]).to(model.device)
        
        optimizer.zero_grad()
        outputs = model(input_ids=input_batch, attention_mask=mask_batch, labels=label_batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
    return total_loss / len(train_inputs)

def evaluate_model(model, test_inputs, test_masks, test_labels, batch_size=8):
    model.eval()
    predictions = []
    with torch.no_grad():
        for i in range(0, len(test_inputs), batch_size):
            input_batch = test_inputs[i:i+batch_size].to(model.device)
            mask_batch = test_masks[i:i+batch_size].to(model.device)
            outputs = model(input_ids=input_batch, attention_mask=mask_batch)
            logits = outputs.logits
            predictions.extend(torch.argmax(logits, dim=1).cpu().numpy())
    
    accuracy = accuracy_score(test_labels, predictions)
    return accuracy

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 读取CSV文件
    df = pd.read_csv(r"C:\Users\user\earnings-call-predict-stock-price-movement\0522_price_volatility_ver4.csv")
    
    # 将'1_day_change_rate'转换为二进制标签
    df['label'] = df['1_day_change_rate'].apply(lambda x: 1 if x > 0 else 0)
    
    # 假设CSV文件有一列名为'paragraphs'，包含整个文档
    documents = df['paragraphs'].tolist()
    labels = df['label'].tolist()
    
    # 将数据拆分为训练集和测试集
    train_documents, test_documents, train_labels, test_labels = train_test_split(documents, labels, test_size=0.2, random_state=42)
    
    transformer = DoubleLayerTransformer(device=device)
    
    # 处理所有训练文档
    print("Processing training documents...")
    train_inputs, train_masks = process_documents(train_documents, transformer)
    
    # 处理所有测试文档
    print("Processing testing documents...")
    test_inputs, test_masks = process_documents(test_documents, transformer)
    
    # 检查是否有足够的输入进行分类
    if train_inputs.size(0) == 0 or test_inputs.size(0) == 0:
        print("Not enough inputs for classification")
        return
    
    # 定义优化器和损失函数
    optimizer = AdamW(transformer.bigbird_model.parameters(), lr=2e-5)
    criterion = nn.CrossEntropyLoss()
    
    # 训练模型
    print("Training model...")
    train_loss = train_model(transformer.bigbird_model, optimizer, criterion, train_inputs, train_masks, train_labels)
    print(f"Train loss: {train_loss:.4f}")
    
    # 评估模型
    print("Evaluating model...")
    test_accuracy = evaluate_model(transformer.bigbird_model, test_inputs, test_masks, test_labels)
    print(f"Test Accuracy: {test_accuracy:.4f}")
    
    # 保存LoRA层参数
    torch.save(transformer.bigbird_model.state_dict(), 'bigbird_with_lora.pth')

if __name__ == "__main__":
    main()
