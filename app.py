import keras
import joblib
from sklearn.preprocessing import MinMaxScaler
import streamlit as st
import pickle
import pandas as pd
import duckdb
from pymongo import MongoClient
import plotly.express as px
from datetime import datetime, timedelta
from LSTM_training import create_df, forecast_future
from LSTM_training_megagrid import combined_loss

# ---------- CONFIG ----------
st.set_page_config(page_title="Stock Price Predictor", layout="wide")

feature_cols = {'AAPL': ['returns', 'PCT_change', 'weighted_tech_sentiment', 'Volume', 'Return_lag1'],
                'AVGO': ['returns', 'PCT_change', 'weighted_tech_sentiment', 'ADI', 'HL_PCT', '%K_14', 'Volume'],
                'GOOG': ['returns', 'PCT_change', 'weighted_tech_sentiment', '%K_14', 'crude_oil', 'gold', 'Return_lag1', 'commodity_index', 'Close', '%D', 'default_spread', 'usd_index'],
                'META': ['returns', 'PCT_change', 'weighted_tech_sentiment'],
                'MSFT': ['returns', 'PCT_change', 'weighted_tech_sentiment', 'Volume', 'HL_PCT', 'Close_lag1', '%K_14', '3m', 'RSI_14', 'commodity_index', 'Return_lag1', 'crude_oil'],
                'NVDA': ['returns', 'PCT_change', 'weighted_tech_sentiment', '%K_14', 'ADI', 'HL_PCT', 'pe', 'evebitda_x', 'evebit_x', 'Close_lag1', 'Volume', 'Close'],
                'ORCL': ['returns', 'PCT_change', 'weighted_tech_sentiment', 'HL_PCT', '%K_14', 'ADI', 'Volume', 'Close_lag1', '3m', 'Close', 'SMA_5', 'spread_baa_10y'],
                'TCEHY': ['returns', 'PCT_change', 'weighted_tech_sentiment', '%K_14', 'RSI_14', 'ADI', 'MACD(12,26,9)'],
                'TSM': ['returns', 'PCT_change', 'weighted_tech_sentiment', 'ADI', '%K_14', 'RSI_14', 'MACD(12,26,9)', 'Volume', '%D', 'debtc', 'term_spread', 'commodity_index']}

days_past = {'AAPL': 10,
                'AVGO': 10,
                'GOOG': 20,
                'META': 40,
                'MSFT': 20,
                'NVDA': 10,
                'ORCL': 10,
                'TCEHY': 60,
                'TSM': 60}

# ---------- TITLE ----------
st.title("📈 TrendMind")

# ---------- INPUT ----------
ticker = st.text_input("Enter a stock ticker (e.g., AAPL, MSFT):").upper()

if ticker:
    # ---------- LOAD DATA ----------
    df = create_df(ticker)
    cols = feature_cols[ticker]
    df = df[cols]

    # ---------- PLOT PRICE HISTORY ----------
    fig = px.line(df, x=df.index[-50:], y=df["Close"].values[-50:], title=f"{ticker} Closing Prices")
    fig.update_layout(xaxis_title="Date", yaxis_title="Price")

    # ---------- LOAD MODEL ----------
    model = keras.models.load_model(f"artifacts/best_model_{ticker}_4.keras", custom_objects={'combined_loss': combined_loss})
    try:

        # ---------- PREPARE INPUT ----------
        scaler = joblib.load(f'artifacts/best_scaler_{ticker}_4.joblib')
        scaled_data = scaler.fit_transform(df)
        forecast_df = forecast_future(model, df, scaler, feature_columns=['Close', 'PCT_change','weighted_tech_sentiment'], n_future=1, n_past=days_past[ticker])
        predicted_price = forecast_df["Predicted_returns"]

        # ---------- DISPLAY PREDICTION ----------
        # Generate future dates (same as before)
        future_dates = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), 
                             periods=len(forecast_df), freq="D")

        fig.add_scatter(x=future_dates, y=predicted_price, mode='markers+lines', name='Predicted Price', marker=dict(symbol='star', size=10, color='red'))
        st.plotly_chart(fig, use_container_width=True)

    except FileNotFoundError:
        st.error(f"No model found for ticker {ticker}")
