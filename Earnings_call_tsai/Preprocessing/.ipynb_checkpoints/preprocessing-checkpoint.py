import os
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from contractions import fix

# 確保已下載所需的NLTK資源
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def preprocess_text(text):
    # 1. 轉換為小寫
    text = text.lower()

    # 2. 展開縮寫
    text = fix(text)

    # 3. 移除標籤、標點符號、數字
    text = re.sub(r'<.*?>', '', text)  # 移除HTML標籤
    text = text.translate(str.maketrans('', '', string.punctuation))  # 移除標點符號
    text = re.sub(r'\d+', '', text)  # 移除數字

    # 4. 分詞
    words = word_tokenize(text)

    # 5. 停用詞移除
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]

    # 6. 詞形還原
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]

    return ' '.join(words)

def preprocess_files_in_directory(input_directory, output_directory):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
        
    for filename in os.listdir(input_directory):
        if filename.endswith(".txt"):
            file_path = os.path.join(input_directory, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()
                processed_text = preprocess_text(text)
                new_filename = filename.replace(".txt", "_preprocessed.txt")
                new_file_path = os.path.join(output_directory, new_filename)
                with open(new_file_path, 'w', encoding='utf-8') as new_file:
                    new_file.write(processed_text)
                print(f'Processed {filename} and saved as {new_filename} in {output_directory}')

# 使用範例
input_directory = r'D:\EarningsCall_finalProject\transcript_texts'  # 替換為您的原始資料夾路徑
output_directory = r'D:\EarningsCall_finalProject\transcript_text_preprocessing'  # 替換為您的新資料夾路徑
preprocess_files_in_directory(input_directory, output_directory)
