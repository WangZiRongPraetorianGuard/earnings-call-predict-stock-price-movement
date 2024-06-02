import pandas as pd

df = pd.read_csv(r"C:\Users\user\earnings-call-predict-stock-price-movement\Earnings_call_tsai\Datasets\After_t5_preprocessing.csv")


print(df['paragraphs'].head(10))


