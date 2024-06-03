import os
import pandas as pd

def convert_cleaned_paragraphs_to_txt(input_csv_path, output_txt_path):
    # 讀取 CSV 文件
    df = pd.read_csv(input_csv_path)
    
    # 獲取第一行的 cleaned_paragraphs 列內容
    cleaned_paragraphs = df.loc[0, 'cleaned_paragraphs']
    
    # 將 cleaned_paragraphs 內容寫入 TXT 文件
    with open(output_txt_path, 'w', encoding='utf-8') as f:
        f.write(cleaned_paragraphs)
    
    print(f'Cleaned paragraphs from the first row saved as {output_txt_path}')

# 使用範例
input_csv_path = r'D:\EarningsCall_finalProject\Earnings_call\Datasets\transcript_text_preprocessing_t5\preprocessed.csv'  # 替換為您的CSV文件路徑
output_txt_path = r'D:\EarningsCall_finalProject\Earnings_call\Datasets\transcript_text_preprocessing_t5\cleaned_paragraphs_first_row.txt'  # 替換為您的TXT文件路徑

convert_cleaned_paragraphs_to_txt(input_csv_path, output_txt_path)
