import time
import os
import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# 创建目录来保存文本文件
if not os.path.exists("transcript_texts"):
    os.makedirs("transcript_texts")

# 初始化 WebDriver
driver = webdriver.Chrome()

# 请求网页
base_url = "https://www.fool.com/earnings-call-transcripts/"
driver.get(base_url)
time.sleep(10)

# 用于跟踪已处理的链接
processed_urls = set()

def scrape_and_save_transcripts(transcript_links):
    for i, link in enumerate(transcript_links):
        transcript_url = link.get_attribute("href")
        if transcript_url in processed_urls:
            continue

        processed_urls.add(transcript_url)
        print(f"Attempting to process URL {len(processed_urls)}: {transcript_url}")

        if not transcript_url:
            print("No URL found or URL is None.")
            continue

        try:
            transcript_response = requests.get(transcript_url)
            if transcript_response.status_code == 200:
                transcript_soup = BeautifulSoup(transcript_response.text, "html.parser")
                
                # 查找所有具有指定类的文本元素
                text_elements = transcript_soup.find_all(class_="max-w-full")
                if not text_elements:
                    print("No text elements found with the specified class.")
                    continue

                # 提取文本内容
                transcript_text = "\n".join(element.text.strip() for element in text_elements)
                if transcript_text:
                    print(f"Extracted some text for URL {len(processed_urls)}, saving...")

                # 将文本内容写入文件
                with open(f"transcript_texts/transcript_{len(processed_urls)}.txt", "w", encoding="utf-8") as f:
                    f.write(transcript_text)
                    print(f"Transcript {len(processed_urls)} successfully saved.")
            else:
                print(f"Failed to fetch URL {len(processed_urls)}: Status code {transcript_response.status_code}")
        except Exception as e:
            print(f"Error processing URL {len(processed_urls)}: {e}")

# 循环抓取最多1000个链接
while len(processed_urls) < 1000:
    # 使用 CSS 选择器找到所有的 <a> 元素
    transcript_links = driver.find_elements(By.CSS_SELECTOR, ".text-gray-1100 a")
    
    # 抓取当前页面的文本内容并保存到文件中
    scrape_and_save_transcripts(transcript_links)
    
    # 点击“Load More”按钮加载更多链接
    try:
        load_more_button = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.XPATH, "//span[contains(text(),'Load More')]")))
        driver.execute_script("arguments[0].click();", load_more_button)
        time.sleep(3)  # 等待页面加载
    except Exception as e:
        print("Error:", e)
        break

print("All transcripts saved.")
driver.quit()
