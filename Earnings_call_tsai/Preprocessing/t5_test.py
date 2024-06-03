import os
import pandas as pd
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
import re
import string
from tqdm import tqdm
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from contractions import fix

def sorted_nicely(l):
    """Sort the given list in the way that humans expect."""
    def convert(text):
        return int(text) if text.isdigit() else text.lower()
    def alphanum_key(key):
        return [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)

def preprocess_text(text):
    # 1. 轉換為小寫
    text = text.lower()

    # 2. 展開縮寫
    text = fix(text)

    # 3. 移除標籤、標點符號、數字
    text = re.sub(r'<.*?>', '', text)  # 移除HTML標籤
    text = text.translate(str.maketrans('', '', string.punctuation))  # 移除標點符號

    # 4. 分詞
    words = word_tokenize(text)

    # 5. 停用詞移除
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]

    # 6. 詞形還原
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]

    return ' '.join(words)

def process_paragraphs_with_t5(paragraphs, tokenizer, model, device, max_length=150, min_length=30, min_char_length=280):
    cleaned_paragraphs = []
    total_word_count = 0  # 用於存儲總單字數
    for paragraph in tqdm(paragraphs, desc="Processing paragraphs"):
        text = preprocess_text(paragraph)
        if len(text) >= min_char_length:  # 檢查字元數
            inputs = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=512, truncation=True).to(device)
            summary_ids = model.generate(
                inputs,
                max_length=max_length,     # 設置生成文本的最大長度
                min_length=min_length,     # 設置生成文本的最小長度
                length_penalty=2.0,
                num_beams=4,
                early_stopping=True
            )
            output = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            cleaned_paragraphs.append(output)
            total_word_count += len(output.split())  # 累計單字數
    return cleaned_paragraphs, total_word_count

def process_csv_with_t5(input_csv_path, output_directory, max_length=150, min_length=30, min_char_length=280):
    # 加載T5模型和tokenizer
    model_name = "t5-base"
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)

    # 檢查是否有可用的 GPU，如果有則使用 GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    df = pd.read_csv(input_csv_path)

    all_cleaned_paragraphs = []
    for i, row in tqdm(df.iterrows(), total=len(df), desc="Processing rows"):
        paragraphs = row['paragraphs'].split('\n')
        cleaned_paragraphs, total_word_count = process_paragraphs_with_t5(paragraphs, tokenizer, model, device, max_length, min_length, min_char_length)
        all_cleaned_paragraphs.append('\n'.join(cleaned_paragraphs))

        # 保存前十行處理結果到單獨的 TXT 文件
        if i < 10:
            txt_output_path = os.path.join(output_directory, f'cleaned_paragraphs_row_{i+1}.txt')
            with open(txt_output_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(cleaned_paragraphs))
            print(f'Processed paragraphs for row {i+1} saved as {txt_output_path}')

    df['cleaned_paragraphs'] = all_cleaned_paragraphs
    output_csv_path = os.path.join(output_directory, 'cleaned_' + os.path.basename(input_csv_path))
    df.to_csv(output_csv_path, index=False)
    print(f'Processed CSV saved as {output_csv_path}')
    print(f'Total word count: {total_word_count}')

# 使用範例
input_csv_path = r'C:\Users\user\earnings-call-predict-stock-price-movement\Earnings_call_tsai\Datasets\20240531_nasdaq_three_class_label.csv'  # 替換為您的CSV文件路徑
output_directory = r'D:\EarningsCall_finalProject\Earnings_call\Datasets\transcript_text_preprocessing_t5'  # 替換為您的T5處理後資料夾路徑

# 設置生成文本的最大長度和最小長度
process_csv_with_t5(input_csv_path, output_directory, max_length=100, min_length=30, min_char_length=280)
