import torch
import pandas as pd
from transformers import AutoModel, BertTokenizer, BigBirdTokenizer, BigBirdForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import gc

class DoubleLayerTransformer:
    def __init__(self, finbert_model_name='yiyanghkust/finbert-tone', bigbird_model_name='google/bigbird-roberta-base', device='cpu'):
        self.finbert_tokenizer = BertTokenizer.from_pretrained(finbert_model_name)
        self.bigbird_tokenizer = BigBirdTokenizer.from_pretrained(bigbird_model_name)
        self.finbert_model = AutoModel.from_pretrained(finbert_model_name).to(device)
        self.bigbird_model = BigBirdForSequenceClassification.from_pretrained(bigbird_model_name).to(device)
        self.device = device

    def tokenize_with_finbert(self, text):
        tokens = self.finbert_tokenizer.tokenize(text)
        token_ids = self.finbert_tokenizer.convert_tokens_to_ids(tokens)
        return tokens, token_ids

    def tokenize_with_bigbird(self, tokens):
        bigbird_token_ids = self.bigbird_tokenizer.convert_tokens_to_ids(tokens)
        return bigbird_token_ids

    def process_document(self, document):
        # 使用FinBERT进行tokenization
        tokens, finbert_token_ids = self.tokenize_with_finbert(document)
        if len(finbert_token_ids) == 0:
            return torch.empty(0)
        
        # 使用BigBird进行进一步的tokenization
        bigbird_token_ids = self.tokenize_with_bigbird(tokens)
        if len(bigbird_token_ids) == 0:
            return torch.empty(0)
        
        # 确保输入的长度不超过BigBird的最大长度4096
        if len(bigbird_token_ids) > 4096:
            bigbird_token_ids = bigbird_token_ids[:4096]
        
        # 构建输入张量
        input_ids = torch.tensor(bigbird_token_ids).unsqueeze(0).to(self.device)  # batch size 1
        attention_mask = torch.ones(input_ids.shape, device=self.device)

        # 使用BigBird模型进行分类
        outputs = self.bigbird_model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits

        # 获取分类结果
        predicted_class = torch.argmax(logits, dim=1).detach().cpu()
        return predicted_class

def process_documents(documents, transformer):
    predictions = []
    for i, document in enumerate(documents):
        print(f"Processing document {i+1}/{len(documents)}")
        prediction = transformer.process_document(document)
        if prediction.size(0) != 0:  # 确认prediction不是空的
            predictions.append(prediction)
        # 清除中间变量以释放内存
        del prediction
        gc.collect()
    return torch.cat(predictions) if predictions else torch.empty(0)

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
    train_predictions = process_documents(train_documents, transformer)
    
    # 处理所有测试文档
    print("Processing testing documents...")
    test_predictions = process_documents(test_documents, transformer)
    
    # 检查是否有足够的预测结果进行分类
    if train_predictions.size(0) == 0 or test_predictions.size(0) == 0:
        print("Not enough predictions for classification")
        return
    
    # 计算准确率
    train_accuracy = accuracy_score(train_labels[:len(train_predictions)], train_predictions)
    test_accuracy = accuracy_score(test_labels[:len(test_predictions)], test_predictions)
    print(f"Train Accuracy: {train_accuracy:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")

if __name__ == "__main__":
    main()
