import torch
import torch.nn as nn
from transformers import BertModel

class DocumentClassifier(nn.Module):
    def __init__(self, num_labels, emb_dim=384):
        super(DocumentClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(emb_dim, num_labels)

    def forward(self, embeddings):
        pooled_output = embeddings.mean(dim=1)  # 沿句子维度取平均
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits

# 假设有两个标签
num_labels = 2

# 初始化模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DocumentClassifier(num_labels=num_labels, emb_dim=384).to(device)

# 加载数据
embeddings_file = r'D:\AIproject\Earnings_call\Datasets\Transcripts\stacked_transcripts_embeddings.pt'
stacked_embeddings = torch.load(embeddings_file, map_location=device)

# 假设你有现成的标签数据，将它们加载或生成，这里生成的是0和1的随机标签
labels = torch.randint(0, num_labels, (stacked_embeddings.size(0),)).to(device)

# 训练模型
model.train()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# 训练循环
num_epochs = 20  # 设置合适的训练周期
for epoch in range(num_epochs):
    optimizer.zero_grad()  # 重置梯度
    logits = model(stacked_embeddings)  # 前向传播
    loss = criterion(logits, labels)  # 计算损失
    loss.backward()  # 反向传播
    optimizer.step()  # 更新权重
    print(f"Epoch {epoch + 1}, Training loss: {loss.item()}")
