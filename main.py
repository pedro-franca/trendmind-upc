import pickle
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import feature_engineering as fe
from LSTM_training import forecast_future
import matplotlib.pyplot as plt

for ticker in ["AAPL", "GOOG", "MSFT", "TSM", "AVGO", "META", "NVDA", "ORCL", "TCEHY"]:
    with open(f"pkl/forecast_returns_{ticker}.pkl", "rb") as f:
        model = pickle.load(f)

    df = yf.download(ticker, start="2025-08-25", end="2025-10-17", interval="1d")
    df = df.droplevel('Ticker', axis=1)
    df["Close_lag1"] = df["Close"].shift(1)
    df["PCT_change"] = (df["Close"] - df["Open"]) / df["Open"] * 100.0
    df["returns"] = df["Close"].pct_change()

    new_data = fe.lt_tech_news()
    df = df.merge(new_data, left_index=True, right_index=True, how="left")
    df = df[['returns', 'PCT_change', 'weighted_tech_sentiment']]

    df = df.ffill()
    df = df.dropna()
    print(df.head())


    preds = []
    for i in range(10, len(df)-1):  # t+1
        data = df.iloc[i-10:i, :]
        print(data)
        print("-----")
        print(df.iloc[i, 0])
        scaler = MinMaxScaler(feature_range=(0,1))
        scaled_data = scaler.fit_transform(data)
        forecast_df = forecast_future(model, data, scaler, feature_columns=['returns', 'PCT_change', 'weighted_tech_sentiment'], n_future=1, n_past=10)
        predicted_price = forecast_df["Predicted_returns"]
        preds.append(predicted_price.values[0])
        print(f"Day {i-9} predicted price: {predicted_price.values[0]}")

    print(preds)

    df_test = yf.download(ticker, start="2025-09-10", end="2025-10-15", interval="1d")['Close']
    df_test['returns'] = df_test.pct_change()
    df_test = df_test['returns'].values[1:]  # skip NaN

    plt.plot(df_test, label='Actual Returns')
    plt.plot(preds, label='Predicted Returns', marker='*')
    # plot a red line at y=0
    plt.axhline(0, color='red', linestyle='--', linewidth=1)
    plt.xlabel('Days')
    plt.ylabel('Returns')
    plt.title(f'{ticker} Stock Returns Prediction')
    plt.legend()
    plt.show()