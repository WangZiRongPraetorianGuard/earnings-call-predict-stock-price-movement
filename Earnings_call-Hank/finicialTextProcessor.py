import re
import csv
import itertools
import numpy as np
import nltk
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from contractions import fix
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer

# 確保已下載所需的NLTK資源
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

class FinancialTextProcessor:
    def __init__(self, lm_dict_filepath):
        self.lm_words = self.load_masterdictionary(lm_dict_filepath)
        
    def load_masterdictionary(self, file_path):
        """
        讀入Loughran-McDonald master dictionary

        """
        master_dictionary = {}
        with open(file_path, newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                word = row['Word'].upper()
                master_dictionary[word] = row
        return set(master_dictionary.keys())
    
    def process_paragraphs(self, paragraphs):
        """
        過濾小於100字符之段落
        
        參數 : 
            paragraphs : 尚未用換行符號處理好之段落 

        返回值 :
            filtered_text : 大於100字符之子文本

        """
        paragraphs = paragraphs.split("\n\n")

        # 找出每個段落長度少於 100 字符的索引
        short_paragraphs = [index for index, paragraph in enumerate(paragraphs) if len(paragraph) < 100]

        # 顯示對應索引的段落
        # for index in short_paragraphs:
        #     print(f"Index: {index}, len: {len(paragraphs[index])}, Paragraph: {paragraphs[index]}")

        filtered_paragraphs = [paragraph for index, paragraph in enumerate(paragraphs) if index not in short_paragraphs]

        # 顯示結果
        index_lengths = [len(paragraph) for paragraph in filtered_paragraphs]
        # print("Total paragraphs:", len(index_lengths))
        # print("Total characters:", sum(index_lengths))
        
        return filtered_paragraphs
    
    def preprocess_text(self, text):
        """
        進行轉小寫、展開縮寫、分詞、停用詞移除處理
        """
        # 去除逐字稿的前後介紹部分
        # text = re.sub(r'^.*?image source: the motley fool\.', '', text, flags=re.DOTALL | re.IGNORECASE)
        # text = re.sub(r'all earnings call transcripts.*$', '', text, flags=re.DOTALL | re.IGNORECASE)

        # 轉換為小寫
        text = text.lower()
        
        # 展開縮寫
        text = fix(text)
        
        # 移除標點符號
        text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
        
        # 分詞
        words = word_tokenize(text)
        
        # 停用詞移除
        stop_words = set(stopwords.words('english'))
        words = [word for word in words if word not in stop_words]
        
        return words
    
    def count_lm_words_in_paragraphs(self, paragraphs):
        """
        計算 LM 字典中每個段落的單字數並傳回排序結果
        """
        counts = []
        
        for index, paragraph in enumerate(paragraphs):
            words = self.preprocess_text(paragraph)
            lm_word_count = sum(word.upper() in self.lm_words for word in words)
            counts.append((index, lm_word_count))
        
        # 根據Index(x[0])或LM Word Count(x[1])排序
        # sorted_counts = sorted(counts, key=lambda x: x[1], reverse=True)
        sorted_counts = sorted(counts, key=lambda x: x[0])
        
        for index, count in sorted_counts:
            print(f"Index: {index}, LM Word Count: {count}")
        
        return sorted_counts
    
    def visualize_lm_word_counts(self, counts):
        """
        使用長條圖視覺化 LM 字數統計
        """
        # 计算每个 LM Word Count 出现的次数
        lm_count_values = [count for index, count in counts]
        count_freq = {}
        
        for count in lm_count_values:
            if count in count_freq:
                count_freq[count] += 1
            else:
                count_freq[count] = 1
        
        # 按 LM Word Count 从小到大排序
        sorted_count_freq = sorted(count_freq.items())
        
        # 提取 x 和 y 轴数据
        x_values = [item[0] for item in sorted_count_freq]
        y_values = [item[1] for item in sorted_count_freq]
        
        # 绘制长条图
        plt.bar(x_values, y_values, color='skyblue')
        plt.xlabel('LM Word Count')
        plt.ylabel('Number of Paragraphs')
        plt.title('Frequency of LM Word Counts in Paragraphs')
        plt.show()

    def get_top_lm_word_count_indices(self, counts, filter_text, top_n=75):
        """
        選擇 LM Word Count 最多的前 top_n 個段落的索引，并返回這些索引及其包含的單詞總數

        參數：
            counts: 包含每个段落 LM Word Count 的列表，格式为 [(index, lm_word_count), ...]
            filter_text: 過濾過後的段落集合
            top_n: 要选择的前 n 个段落，默认为 10。

        返回值：
            selected_indices: 前 top_n 个段落的索引列表。
            total_word_count: 前 top_n 个段落中的单词总数。
        """
        # 按 LM Word Count 从高到低排序
        sorted_counts = sorted(counts, key=lambda x: x[0], reverse=False)
        
        # 选择前 top_n 个段落的索引
        selected_indices = [index for index, _ in sorted_counts[:top_n]]
        print(selected_indices)
        
        # 计算前 top_n 个段落中的单词总数
        total_word_count = sum(len(paragraph.split()) for index, paragraph in enumerate(filter_text) if index in selected_indices)
        print(total_word_count)
        
        return selected_indices, total_word_count
    
    def process_and_create_denoised_df(self, df):
        """
        對原資料集每一列['paragraph']進行處理，創建包含去噪段落和標籤的新數據集
        """
        import pandas as pd
        new_paragraphs = []
        new_labels = []

        row_counts = 0

        for idx, row in df.iterrows():
            if row_counts >= 1:
                break
            filtered_text = self.process_paragraphs(row['paragraphs'])
            counts = self.count_lm_words_in_paragraphs(filtered_text)
            top_indices, _ = self.get_top_lm_word_count_indices(counts, filtered_text)

            # 將前10段落組成一個子文本
            sub_text = ' '.join([filtered_text[i] for i in top_indices])
            new_paragraphs.append(sub_text)
            new_labels.append(row['three_class_label'])
            
            row_counts += 1
        
        denoised_df = pd.DataFrame({'paragraphs': new_paragraphs, 'three_class_label': new_labels})
        return denoised_df

# Example usage
if __name__ == '__main__':
    lm_dict_filepath = r'C:\Users\Hank\earnings-call-predict-stock-price-movement\preprocessing_code\Loughran-McDonald_MasterDictionary_1993-2023.csv'
    import pandas as pd

    df = pd.read_csv(r'C:\Users\Hank\earnings-call-predict-stock-price-movement\preprocessing_code\dataset\20240531_nasdaq_three_class_label.csv')
    
    processor = FinancialTextProcessor(lm_dict_filepath)
    
    denoised_df = processor.process_and_create_denoised_df(df)
    print(denoised_df.head())
