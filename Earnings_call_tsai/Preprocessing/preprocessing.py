import os
import pandas as pd
import re
import string
from tqdm import tqdm
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from contractions import fix

def preprocess_text(text):
    # 1. 轉換為小寫
    text = text.lower()

    # 2. 展開縮寫
    text = fix(text)

    # 3. 分詞


 

    # 保留换行符和标点符号
    return ' '.join(words).replace(' .', '.').replace(' ,', ',')

def preprocess_csv(input_csv_path, output_csv_path):
    df = pd.read_csv(input_csv_path)

    cleaned_texts = []
    for text in tqdm(df['paragraphs'], desc="Preprocessing texts"):
        cleaned_texts.append(preprocess_text(text))

    df['cleaned_paragraphs'] = cleaned_texts
    df.to_csv(output_csv_path, index=False)
    print(f'Preprocessed CSV saved as {output_csv_path}')

# 使用範例
input_csv_path = r'C:\Users\user\earnings-call-predict-stock-price-movement\Earnings_call_tsai\Datasets\20240531_nasdaq_three_class_label.csv'  # 替換為您的CSV文件路徑
output_csv_path = r'D:\EarningsCall_finalProject\Earnings_call\Datasets\transcript_text_preprocessing_t5\preprocessed.csv'  # 替換為您的預處理後的CSV文件路徑

preprocess_csv(input_csv_path, output_csv_path)
