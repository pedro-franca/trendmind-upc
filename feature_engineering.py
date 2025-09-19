import duckdb
import pandas as pd
import numpy as np
from pymongo import MongoClient

# --- L&T yf data from DuckDB ---
def lt_yf_data(ticker, db_path=None):
    if db_path is None:
        db_yf_path = f"./data/{ticker}_yf_clean.duckdb"
    
    con_yf = duckdb.connect(db_yf_path)
    yf_df = con_yf.execute("SELECT * FROM stock_data ORDER BY Date").df()
    con_yf.close()

    yf_df['date'] = pd.to_datetime(yf_df['Date'])
    yf_df['date'] = yf_df['date'].dt.date
    yf_df = yf_df.drop(columns=['ticker'], errors='ignore')  # Drop ticker if exists
 
 # TODO: techinal features - Edward Achong
    return yf_df

# --- L&T news data from MongoDB ---
def lt_news_data(ticker):
    low_ticker = ticker.lower()
    MONGO_DB = "financial_news"
    MONGO_COLLECTION = f"{ticker}_news"

    client = MongoClient("mongodb://localhost:27017/")
    db = client[MONGO_DB]
    collection = db[MONGO_COLLECTION]

    df = pd.DataFrame(list(collection.find()))

    exploded_df = df.explode('ticker_sentiment').reset_index(drop=True)

    # Now each row has a single ticker_sentiment dict; normalize it into separate columns
    ticker_df = pd.json_normalize(exploded_df['ticker_sentiment'])

    # Concatenate with the original DataFrame
    exploded_df = pd.concat([exploded_df.drop(columns=['ticker_sentiment']), ticker_df], axis=1)

    # Filter for ticker 'ORCL' and select relevant columns
    news_df = exploded_df[exploded_df['ticker'] == ticker][[
        'relevance_score',
        'ticker_sentiment_score',
        'time_published'
    ]].rename(columns={
        'relevance_score': 'orcl_relevance_score',
        'ticker_sentiment_score': 'orcl_sentiment_score'
    })

    news_df['date'] = pd.to_datetime(news_df['time_published'],  format="%Y%m%dT%H%M%S")
    news_df["date"] = news_df["date"].dt.date
    news_df[f"{low_ticker}_sentiment_score"] = pd.to_numeric(news_df[f"{low_ticker}_sentiment_score"], errors='coerce')
    news_df[f"{low_ticker}_relevance_score"] = pd.to_numeric(news_df[f"{low_ticker}_relevance_score"], errors='coerce')

    sentiment_df = news_df.groupby("date").apply(
        lambda g: (g[f"{low_ticker}_sentiment_score"] * g[f"{low_ticker}_relevance_score"]).sum() / g[f"{low_ticker}_relevance_score"].sum()
        ).reset_index(name="weighted_sentiment")
    
    print(sentiment_df.head())  # Optional preview
         
    return sentiment_df

def lt_sec_data(ticker, db_path=None):
    
    con_sec = duckdb.connect(db_path)
    sec_df = con_sec.execute("SELECT * FROM stock_data").df()
    con_sec.close()

    sec_df = sec_df.drop('concept', axis=1).set_index('label').T
    sec_df = sec_df.rename_axis('date')
    sec_df.index = pd.to_datetime(sec_df.index)

    print(sec_df.head())  # Optional preview

    return sec_df


if __name__ == "__main__":
    ticker = "ORCL"
    lt_sec_data(ticker, db_path=f"./data/{ticker}_10k_clean.duckdb")