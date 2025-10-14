import numpy as np
import pandas as pd
from feature_engineering import create_df


for ticker in ["AAPL", "GOOG", "MSFT", "TSM", "AVGO", "META", "NVDA", "ORCL", "TCEHY"]:
    df = create_df(ticker)
    df = df.ffill()
    df = df.dropna(axis=1, how='all')
    df.sort_index(inplace=True)
    df_d = df.describe()
    df_d.to_csv(f'{ticker}_description.csv')