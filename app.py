from sklearn.preprocessing import MinMaxScaler
import streamlit as st
import pickle
import pandas as pd
import duckdb
from pymongo import MongoClient
import plotly.express as px
from datetime import datetime, timedelta
from LSTM_training import create_df, forecast_future

# ---------- CONFIG ----------
st.set_page_config(page_title="Stock Price Predictor", layout="wide")

# ---------- TITLE ----------
st.title("📈 LSTM Stock Price Predictor")

# ---------- INPUT ----------
ticker = st.text_input("Enter a stock ticker (e.g., AAPL, MSFT):").upper()

if ticker:
    # ---------- LOAD DATA ----------
    df = create_df(ticker)
    df = df[['Close', 'PCT_change','weighted_tech_sentiment']]

    # ---------- PLOT PRICE HISTORY ----------
    fig = px.line(df, x=df.index[-50:], y=df["Close"].values[-50:], title=f"{ticker} Closing Prices")
    fig.update_layout(xaxis_title="Date", yaxis_title="Price")

    # ---------- LOAD MODEL ----------
    model_path = f"pkl/forecast_{ticker}.pkl"
    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)

        # ---------- PREPARE INPUT ----------
        scaler = MinMaxScaler(feature_range=(0,1))
        scaled_data = scaler.fit_transform(df)
        forecast_df = forecast_future(model, df, scaler, feature_columns=['Close', 'PCT_change','weighted_tech_sentiment'], n_future=5, n_past=10)
        predicted_price = forecast_df["Close_t1"]

        # ---------- DISPLAY PREDICTION ----------
        # Generate future dates (same as before)
        future_dates = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), 
                             periods=len(forecast_df), freq="D")

        fig.add_scatter(x=future_dates, y=predicted_price, mode='markers+lines', name='Predicted Price', marker=dict(symbol='star', size=10, color='red'))
        st.plotly_chart(fig, use_container_width=True)

    except FileNotFoundError:
        st.error(f"No model found for ticker {ticker}")
