import streamlit as st
import pickle
import pandas as pd
import duckdb
from pymongo import MongoClient
import plotly.express as px
from datetime import datetime, timedelta
from LSTM_training import create_df, forecast_pipeline

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
    st.plotly_chart(fig, use_container_width=True)

    # ---------- LOAD MODEL ----------
    model_path = f"pkl/forecast_{ticker}.pkl"
    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)

        # ---------- PREPARE INPUT ----------
        last_10 = df["Close"].values[-10:].reshape(1, -1, 1)  # adjust for your model
        predicted_price = model.predict(last_10)[0][0]

        # ---------- DISPLAY PREDICTION ----------
        st.subheader(f"Next-Day Predicted Price for {ticker}: ${predicted_price:.2f}")
        st.write(f"Last updated: {df.index.max().strftime('%Y-%m-%d')}")


    except FileNotFoundError:
        st.error(f"No model found for ticker {ticker}")