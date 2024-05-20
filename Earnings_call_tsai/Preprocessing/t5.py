import os
from transformers import T5Tokenizer, T5ForConditionalGeneration
import re

def sorted_nicely(l):
    """ Sort the given list in the way that humans expect."""
    def convert(text):
        return int(text) if text.isdigit() else text.lower()
    def alphanum_key(key):
        return [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)

def clean_text(text):
    """簡單清理文本，去除多餘的空格和重複的字符"""
    text = re.sub(r'\s+', ' ', text)  # 去除多餘的空格
    text = re.sub(r'(.)\1{2,}', r'\1', text)  # 去除連續重複的字符
    return text

def process_with_t5(input_directory, output_directory, max_length=150, min_length=30):
    # 加載T5模型和tokenizer
    model_name = "t5-base"
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    files = sorted_nicely([f for f in os.listdir(input_directory) if f.endswith(".txt")])

    for filename in files:
        file_path = os.path.join(input_directory, filename)
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
            text = clean_text(text)  # 清理輸入文本
            # 使用T5進行處理
            inputs = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=512, truncation=True)
            summary_ids = model.generate(
                inputs,
                    max_length=max_length,     # 設置生成文本的最大長度
                    min_length=min_length,     # 設置生成文本的最小長度
                    length_penalty=2.0,
                    num_beams=4,
                    early_stopping=True
            )
            output = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

            new_filename = filename.replace(".txt", "_t5_processed.txt")
            new_file_path = os.path.join(output_directory, new_filename)
            with open(new_file_path, 'w', encoding='utf-8') as new_file:
                new_file.write(output)
            print(f'Processed {filename} with T5 and saved as {new_filename} in {output_directory}')

# 使用範例
input_directory = r'D:\EarningsCall_finalProject\Earnings_call\Datasets\transcript_text_preprocessing'  # 替換為您的預處理後資料夾路徑
output_directory = r'D:\EarningsCall_finalProject\Earnings_call\Datasets\transcript_text_preprocessing_t5'  # 替換為您的T5處理後資料夾路徑

# 設置生成文本的最大長度和最小長度
process_with_t5(input_directory, output_directory, max_length=1000, min_length=30)
