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

def fetch_latest_news(ticker, cutoff_date=datetime(2025, 8, 25), limit=5):
    """
    Fetch latest `limit` news items where:
      - time_published <= cutoff_date
      - ticker exists in ticker_sentiment.ticker
    """
    mongo_uri = "mongodb+srv://pedrochfr:trendmind@cluster0.5j733.mongodb.net/"
    MONGO_DB = "financial_news"
    MONGO_COLLECTION = "tech_news"

    client = MongoClient(mongo_uri)
    db = client[MONGO_DB]
    collection = db[MONGO_COLLECTION]

    # Convert cutoff_date (YYYY-MM-DD) → comparable format
    cutoff_str = datetime.strptime(cutoff_date, "%Y-%m-%d").strftime("%Y%m%dT%H%M%S")

    query = {
        "time_published": {"$lte": cutoff_str},
        "ticker_sentiment.ticker": ticker.upper()
    }

    cursor = (
        collection.find(query)
        .sort("time_published", -1)
        .limit(limit)
    )

    results = list(cursor)
    client.close()
    return results

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
    open = df.iloc[-1]['Open']
    high = df.iloc[-1]['High']
    low = df.iloc[-1]['Low']
    volume = df.iloc[-1]['Volume']
    close = df.iloc[-1]['Close']
    cols = feature_cols[ticker]
    df = df[cols]

    # ---------- PLOT PRICE HISTORY ----------
    fig = px.line(df, x=df.index[-50:], y=df["returns"].values[-50:]*100, title=f"{ticker} Returns")
    fig.update_layout(xaxis_title="Date", yaxis_title="Price Returns (%)")

    # ---------- LOAD MODEL ----------
    model = keras.models.load_model(f"artifacts/best_model_{ticker}_4.keras", custom_objects={'combined_loss': combined_loss})
    try:

        # ---------- PREPARE INPUT ----------
        scaler = joblib.load(f'artifacts/best_scaler_{ticker}_4.joblib')
        scaled_data = scaler.fit_transform(df)
        forecast_df = forecast_future(model, df, scaler, feature_columns=cols, n_future=1, n_past=days_past[ticker])
        predicted_price = forecast_df["Predicted_returns"]*100

        # ---------- DISPLAY PREDICTION ----------
        # Generate future dates (same as before)
        future_dates = pd.date_range(start=df.index[-1] + pd.tseries.offsets.BDay(1), 
                             periods=len(forecast_df), freq="D")

        fig.add_scatter(x=future_dates, y=predicted_price, mode='markers+lines', name='Predicted Price', marker=dict(symbol='star', size=10, color='red'))
        # add horizontal line at y=0
        fig.add_hline(y=0, line_dash="dash", line_color="gray")
        st.plotly_chart(fig, use_container_width=True)

        st.divider()
        
        st.subheader("📊 Last Day's OHLCV Data")

        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Open", f"${open:,.2f}")
        col2.metric("High", f"${high:,.2f}")
        col3.metric("Low", f"${low:,.2f}")
        col4.metric("Close", f"${close:,.2f}")
        col5.metric("Volume", f"{volume:,.0f}")

        st.divider()
        st.markdown("### ") 

        cutoff_date = datetime(2025, 8, 25).strftime("%Y-%m-%d")
        news_data = fetch_latest_news(
            ticker,
            cutoff_date=cutoff_date,
            limit=5
        )
        if not news_data:
            st.warning(f"No news found for {ticker} before {cutoff_date}.")
        else:
            # Convert to DataFrame for nice display
            df = pd.DataFrame([
                {
                    "Title": item.get("title"),
                    "Published": datetime.strptime(item["time_published"], "%Y%m%dT%H%M%S").strftime("%Y-%m-%d %H:%M"),
                    "Source": item.get("source"),
                    "Summary": item.get("summary", "No summary available."),
                    "Sentiment": item.get("overall_sentiment_label"),
                    "Sentiment Score": item.get("overall_sentiment_score"),
                    "URL": item.get("url")
                }
                for item in news_data
            ])

            # Display nicely
            st.subheader(f"📰 Latest {ticker.upper()} News")
            for _, row in df.iterrows():
                st.markdown(f"### [{row['Title']}]({row['URL']})")
                st.markdown(
                    f"**Published:** {row['Published']}  |  "
                    f"**Source:** {row['Source']}  |  "
                    f"**Sentiment:** {row['Sentiment']} ({row['Sentiment Score']:.3f})"
                )
                st.markdown(f"📝 *{row['Summary']}*")
                st.markdown("---")

    except FileNotFoundError:
        st.error(f"No model found for ticker {ticker}")
