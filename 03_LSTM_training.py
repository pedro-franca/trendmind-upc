import duckdb
import pandas as pd
import numpy as np
import tensorflow as tf
import ast
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# --- Load data from DuckDB ---
def load_data(ticker, db_path=None):
    if db_path is None:
        db_path = f"./data/{ticker}_cleaned.duckdb"
    con = duckdb.connect(db_path)
    df = con.execute("SELECT * FROM stock_data ORDER BY Date").df()
    con.close()

    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    df = df.sort_index()
    df = df.drop(columns=['ticker'], errors='ignore')  # Drop ticker if exists
    
    return df

# --- Preprocess data for LSTM ---
def prepare_lstm_data(df, target_col='Close', sequence_length=60):
    # Only use Close for prediction; scale to [0, 1]
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(df[[target_col]])

    X, y = [], []
    for i in range(sequence_length, len(data_scaled)):
        X.append(data_scaled[i-sequence_length:i])
        y.append(data_scaled[i])
    X, y = np.array(X), np.array(y)

    return X, y, scaler

# --- Build a simple LSTM model ---
def build_lstm_model(input_shape):
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape),
        LSTM(50),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# --- Train the model ---
def train_lstm(ticker):
    df = load_data(ticker)
    df 
    print(df.head())  # Optional preview
    if 'Close' not in df.columns:
        raise ValueError("Expected 'Close' column in data.")

    X, y, scaler = prepare_lstm_data(df, target_col='Close')

    model = build_lstm_model((X.shape[1], X.shape[2]))
    model.fit(X, y, epochs=20, batch_size=32, verbose=1)

    return model, scaler

# --- Main ---
if __name__ == "__main__":
    ticker = "AAPL"
    try:
        model, scaler = train_lstm(ticker)
        print(f"✅ LSTM model trained for {ticker}")
    except Exception as e:
        print(f"❌ Error training model for {ticker}: {e}")
