# LSTM_training.py
import pandas as pd
import yfinance as yf
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from feature_selection import select_features
from feature_engineering import create_df

# --- Load and transform data ---
def load_transform_data(ticker, target_col='Close'):
    df = create_df(ticker)
    # df = df[['Close','Open','High','Low','Volume','weighted_sentiment']]
    df = select_features(df, target_col=target_col)
    df = df.ffill()
    df = df.dropna(axis=1, how='all')
    df.sort_index(inplace=True)
    print(df.head())  # Optional preview
    return df

# --- Data prep ---
def prepare_lstm_data(df, target_col='Close', sequence_length=60):
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(df[[target_col]])

    X, y = [], []
    for i in range(sequence_length, len(data_scaled)):
        X.append(data_scaled[i-sequence_length:i])
        y.append(data_scaled[i])
    return np.array(X), np.array(y), scaler

# --- Build model with given hyperparameters ---
def build_model(input_shape, use_gru=False):
    model = Sequential()
    if use_gru:
        model.add(GRU(120, activation="tanh", input_shape=input_shape))
    else:
        model.add(LSTM(120, activation="tanh", input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(Dense(1))  # output layer
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# --- Train ---
def train_model(df, ticker="AAPL", sequence_length=60, use_gru=False):
    X, y, scaler = prepare_lstm_data(df, target_col='Close', sequence_length=sequence_length)

    model = build_model((X.shape[1], X.shape[2]), use_gru=use_gru)

    early_stop = EarlyStopping(monitor="loss", patience=10, restore_best_weights=True)

    model.fit(
        X, y,
        epochs=100,  # capped at 100
        batch_size=30,
        verbose=1,
        callbacks=[early_stop]
    )
    return model, scaler

def forecast_future(model, scaler, df, target_col='Close', sequence_length=60, days_ahead=7):
    data_scaled = scaler.transform(df[[target_col]])
    last_seq = data_scaled[-sequence_length:].reshape(1, sequence_length, 1)

    preds = []
    seq = last_seq.copy()
    for _ in range(days_ahead):
        pred_scaled = model.predict(seq, verbose=0)  # shape (1, 1)
        preds.append(pred_scaled[0, 0])

        # reshape to (1, 1, 1) so it matches sequence dimensions
        pred_scaled_reshaped = pred_scaled.reshape(1, 1, 1)

        # drop oldest step, append new prediction
        seq = np.append(seq[:, 1:, :], pred_scaled_reshaped, axis=1)

    return scaler.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()

# --- Example run ---
if __name__ == "__main__":
    tickers = ["AAPL", "MSFT"]
    for ticker in tickers:
        print(f"--- Training and forecasting for {ticker} ---")
        y_test = yf.download(ticker, start="2025-07-01", end="2025-08-13", interval="1d")['Close'].values
        df_no = yf.download(ticker, start="2022-01-01", end="2025-07-01", interval="1d")[['Close']]
        df = load_transform_data(ticker, target_col='Close')
        print(df.head())
        # You can try different lookbacks: 10, 12, 14, 16, 18, 20
        lookback = 60
        days_ahead = 30
        model, scaler = train_model(df, ticker, sequence_length=lookback, use_gru=False)
        model_no, scaler_no = train_model(df_no, ticker, sequence_length=lookback, use_gru=False)

        future_preds = forecast_future(model, scaler, df, sequence_length=lookback, days_ahead=days_ahead)
        future_preds_no = forecast_future(model_no, scaler_no, df_no, sequence_length=lookback, days_ahead=days_ahead)
        print(f"Next 3 predicted closes for {ticker} with lookback {lookback}:")
        print(future_preds)
        print(f"Next 3 predicted closes for {ticker} without feature engineering:")
        print(future_preds_no)

        metrics = {
            "MSE_with_FE": mean_squared_error(y_test, future_preds),
            "MAE_with_FE": mean_absolute_error(y_test, future_preds),
            "R2_with_FE": r2_score(y_test, future_preds),
            "MSE_no_FE": mean_squared_error(y_test, future_preds_no),
            "MAE_no_FE": mean_absolute_error(y_test, future_preds_no),
            "R2_no_FE": r2_score(y_test, future_preds_no),
        }
        print(f"{ticker} Metrics:", metrics)

        plt.figure(figsize=(12,6))
        plt.plot(range(len(y_test)), y_test, label='Actual Close', marker='o')
        plt.plot(range(len(future_preds)), future_preds, label='Predicted Close (with FE)', marker='x')
        plt.plot(range(len(future_preds_no)), future_preds_no, label='Predicted Close (no FE)', marker='s')
        plt.title(f"{ticker} Close Price Prediction")
        plt.xlabel("Days Ahead")
        plt.ylabel("Price")
        plt.legend()
        plt.grid()
        plt.savefig(f'{ticker}_prediction.pdf')


