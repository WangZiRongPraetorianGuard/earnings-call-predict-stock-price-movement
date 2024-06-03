import torch
import pandas as pd
from transformers import AutoModel, AutoTokenizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import gc

class DoubleLayerTransformer:
    def __init__(self, bert_model_name='yiyanghkust/finbert-tone', device='cpu'):
        self.tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
        self.bert_model = AutoModel.from_pretrained(bert_model_name).to(device)
        self.context_model = AutoModel.from_pretrained('bert-base-uncased', add_pooling_layer=False).to(device)
        self.device = device

    def chunk_text(self, text, max_length=512):
        tokens = self.tokenizer.tokenize(text)
        chunks = [' '.join(tokens[i:i+max_length]) for i in range(0, len(tokens), max_length)]
        print(f"Chunked text into {len(chunks)} chunks.")  # 输出分块数量
        return chunks

    def encode_chunks(self, chunks):
        chunk_vectors = []
        for i, chunk in enumerate(chunks):
            try:
                print(f"Encoding chunk {i+1}/{len(chunks)}")
                inputs = self.tokenizer(chunk, return_tensors='pt', truncation=True, padding='max_length', max_length=512).to(self.device)
                outputs = self.bert_model(**inputs)
                chunk_vector = outputs.last_hidden_state[:, 0, :].detach().cpu()  # [CLS] token representation
                print(f"Chunk vector shape: {chunk_vector.shape}")  # 输出每个chunk的向量形状
                chunk_vectors.append(chunk_vector)
            except Exception as e:
                print(f"Error encoding chunk {i+1}: {e}")
        return torch.cat(chunk_vectors, dim=0) if chunk_vectors else torch.empty(0)
    
    def integrate_context(self, chunk_vectors):
        try:
            chunk_vectors = chunk_vectors.to(self.device)
            position_ids = torch.arange(chunk_vectors.size(0), dtype=torch.long).unsqueeze(0).to(self.device)
            outputs = self.context_model(inputs_embeds=chunk_vectors.unsqueeze(0), position_ids=position_ids)
            document_vector = outputs.last_hidden_state.mean(dim=1).detach().cpu()  # Average pooling
            print(f"Document vector shape after integration: {document_vector.shape}")  # 输出整合后的文档向量形状
            return document_vector
        except Exception as e:
            print(f"Error integrating context: {e}")
            return torch.empty(0)
    
    def process_document(self, document):
        chunks = self.chunk_text(document)
        chunk_vectors = self.encode_chunks(chunks)
        if chunk_vectors.size(0) == 0:
            return torch.empty(0)  # 如果编码失败，返回一个空张量
        document_vector = self.integrate_context(chunk_vectors)
        return document_vector

def process_documents(documents, transformer):
    vectors = []
    for i, document in enumerate(documents):
        print(f"Processing document {i+1}/{len(documents)}")
        document_vector = transformer.process_document(document)
        if document_vector.size(0) != 0:  # 确认document_vector不是空的
            vectors.append(document_vector)
        # 清除中间变量以释放内存
        del document_vector
        gc.collect()
    return vectors

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
    train_vectors = process_documents(train_documents, transformer)
    
    # 处理所有测试文档
    print("Processing testing documents...")
    test_vectors = process_documents(test_documents, transformer)
    
    # 检查是否有足够的向量进行分类
    if not train_vectors or not test_vectors:
        print("Not enough vectors for classification")
        return
    
    # 将向量列表转换为张量
    train_vectors = torch.cat(train_vectors)
    test_vectors = torch.cat(test_vectors)
    
    # 展平成2D张量
    train_vectors = train_vectors.view(train_vectors.size(0), -1)
    test_vectors = test_vectors.view(test_vectors.size(0), -1)
    print(f"Train vectors shape: {train_vectors.shape}")  # 输出训练向量的形状
    print(f"Test vectors shape: {test_vectors.shape}")  # 输出测试向量的形状
    
    # 使用逻辑回归进行分类
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(random_state=42)
    clf.fit(train_vectors.detach().numpy(), train_labels[:len(train_vectors)])
    
    # 预测测试集的标签
    test_predictions = clf.predict(test_vectors.detach().numpy())
    
    # 计算准确率
    accuracy = accuracy_score(test_labels[:len(test_vectors)], test_predictions)
    print(f"Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    main()
