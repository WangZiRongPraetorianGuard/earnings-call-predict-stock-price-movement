import os
import re
import torch
from sentence_transformers import SentenceTransformer

# 初始化 SBERT 模型
model = SentenceTransformer('all-MiniLM-L6-v2')

def split_into_sentences(text):
    """
    使用正则表达式分割文本成句子，并清理每个句子
    """
    sentences = re.split(r'(?<=[.!?]) +', text)
    return [sentence.strip() for sentence in sentences if sentence.strip()]

def embed_sentences(sentences):
    """
    使用 SBERT 嵌入每个句子，返回张量列表
    """
    if not sentences:
        return torch.tensor([])  # 返回空张量
    return model.encode(sentences, convert_to_tensor=True)

def process_and_save_embeddings(folder_path, max_sentences=360):
    all_embeddings = []
    num_files = 0

    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            num_files += 1
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()

            sentences = split_into_sentences(text)
            if len(sentences) > max_sentences:
                sentences = sentences[:max_sentences]  # 截断
            elif len(sentences) < max_sentences:
                sentences.extend([""] * (max_sentences - len(sentences)))  # 填充

            embeddings = embed_sentences(sentences)
            all_embeddings.append(embeddings)

            # 打印每个文件的句子数量以验证截断和填充
            print(f"Processed {filename}: {len(sentences)} sentences after padding/truncation.")
            print(f"Embedding shape for {filename}: {embeddings.shape}")

    # 将所有文件的嵌入堆叠成一个张量
    stacked_embeddings = torch.stack(all_embeddings)
    print(f"Final stacked embeddings shape: {stacked_embeddings.shape}")

    # 保存处理后的嵌入
    torch.save(stacked_embeddings, os.path.join(folder_path, 'stacked_transcripts_embeddings.pt'))
    print(f"Embeddings saved successfully. Total files processed: {num_files}")

# 执行嵌入处理和保存
folder_path = r'D:\AIproject\Earnings_call\Datasets\Transcripts'
process_and_save_embeddings(folder_path)
