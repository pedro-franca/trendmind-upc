import duckdb
import pandas as pd
import numpy as np
import tensorflow as tf
import ast
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

    
# --- Load data from DuckDB ---
def load_transform_data(ticker, db_path=None):
    low_ticker = ticker.lower()
    if db_path is None:
        db_yf_path = f"./data/{ticker}_yf_cleaned.duckdb"
        db_news_path = f"./data/{ticker}_news_cleaned.duckdb"
    
    con_yf = duckdb.connect(db_yf_path)
    yf_df = con_yf.execute("SELECT * FROM stock_data ORDER BY Date").df()
    con_yf.close()

    yf_df['date'] = pd.to_datetime(yf_df['Date'])
    yf_df['date'] = yf_df['date'].dt.date
    yf_df = yf_df.drop(columns=['ticker'], errors='ignore')  # Drop ticker if exists

    con_news = duckdb.connect(db_news_path)
    news_df = con_news.execute("SELECT * FROM news_data ORDER BY time_published").df()
    con_news.close()

    news_df['date'] = pd.to_datetime(news_df['time_published'],  format="%Y%m%dT%H%M%S")
    news_df["date"] = news_df["date"].dt.date
    news_df[f"{low_ticker}_sentiment_score"] = pd.to_numeric(news_df[f"{low_ticker}_sentiment_score"], errors='coerce')
    news_df[f"{low_ticker}_relevance_score"] = pd.to_numeric(news_df[f"{low_ticker}_relevance_score"], errors='coerce')

    sentiment_daily = news_df.groupby("date").apply(
        lambda g: (g[f"{low_ticker}_sentiment_score"] * g[f"{low_ticker}_relevance_score"]).sum() / g[f"{low_ticker}_relevance_score"].sum()
        ).reset_index(name="weighted_sentiment")

    merged_df = pd.merge(yf_df, sentiment_daily, on="date", how="left")
    merged_df["weighted_sentiment"].ffill(inplace=True)  # Forward fill missing sentiment values
    
    return merged_df

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
    df = load_transform_data(ticker)
    print(df.head())  # Optional preview
    if 'Close' not in df.columns:
        raise ValueError("Expected 'Close' column in data.")

    X, y, scaler = prepare_lstm_data(df, target_col='Close')

    model = build_lstm_model((X.shape[1], X.shape[2]))
    model.fit(X, y, epochs=20, batch_size=32, verbose=1)

    return model, scaler

# --- Main ---
if __name__ == "__main__":
    ticker = "ORCL"
    try:
        model, scaler = train_lstm(ticker)
        print(f"✅ LSTM model trained for {ticker}")
    except Exception as e:
        print(f"❌ Error training model for {ticker}: {e}")
