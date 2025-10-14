import duckdb
import pandas as pd
import numpy as np
import os
from datetime import datetime
from pymongo import MongoClient

mongo_uri = "mongodb+srv://pedrochfr:trendmind@cluster0.5j733.mongodb.net/"

# --- Load data from DuckDB ---
def load_data(db_path):
    if not os.path.exists(db_path):
        print(f"Warning: Delta path {db_path} does not exist.")
        df = pd.DataFrame() 
    else:
        table = db_path.split("/")[-1].replace(".duckdb", "")
        con = duckdb.connect(db_path)
        df = con.execute(f"SELECT * FROM {table}").df()
        con.close()
        # Ensure Date is datetime and set as index
        df['Date'] = pd.to_datetime(df['Date'])
        df['Date'] = df['Date'].dt.date
        df = df.set_index('Date').sort_index()
        df = df.drop(columns=['ticker'], errors='ignore')  # Drop ticker if exists

    return df

####### Advanced Technical Financial Variables #######
### Momentum
def momentum_rsi(data, window=14):
    delta = data["Close"].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    data[f"RSI_{window}"] = rsi

    # make sure Date stays as index
    if "Date" in data.columns:
        data = data.set_index("Date")

    return data

def momentum_stochasticK_oscillator(data, window=14):
    low_min = data["Low"].rolling(window=window).min()
    high_max = data["High"].rolling(window=window).max()

    data[f"%K_{window}"] = ((data["Close"] - low_min) / (high_max - low_min)) * 100
    #data["%D"] = data["%K"].rolling(window=3).mean()

    # make sure Date stays as index
    if "Date" in data.columns:
        data = data.set_index("Date")

    return data

def momentum_stochasticD_oscillator(data, window=14):
    data["%D"] = data["%K_14"].rolling(window=3).mean()

    # make sure Date stays as index
    if "Date" in data.columns:
        data = data.set_index("Date")

    return data

def momentum_return(data, window=1, lag=1):
    returns = data["Close"].pct_change(periods=window)

    if lag:
        returns = returns.shift(lag)   # move values back by 1 day
    data[f"Return_lag{1 if lag else 0}"] = returns

        # make sure Date stays as index
    if "Date" in data.columns:
        data = data.set_index("Date")

    return data

### Trend
def trend_moving_average(data, window=5):
    data[f"SMA_{window}"] = data["Close"].rolling(window=window).mean()

    # make sure Date stays as index
    if "Date" in data.columns:
        data = data.set_index("Date")

    return data

def trend_sma_crossover(data, short_window=10, long_window=50):
    data_s = data["Close"].rolling(window=short_window).mean()
    data_l = data["Close"].rolling(window=long_window).mean()

    # Difference between short and long SMA
    diff = data_s - data_l

    # Detect crossovers
    cross_up = (diff.shift(1) <= 0) & (diff > 0)   # short crosses above long
    cross_dn = (diff.shift(1) >= 0) & (diff < 0)   # short crosses below long

    # Assign +1 for bullish cross, -1 for bearish cross, 0 otherwise
    cross = np.where(cross_up, 1, np.where(cross_dn, -1, 0)).astype(np.int8)

    data[f"SMA({short_window},{long_window})"] = cross

    # keep Date as index if it exists
    if "Date" in data.columns:
        data = data.set_index("Date")

    return data

def trend_exponential_moving_average(data, window=10):
    data[f"EMA_{window}"] = data["Close"].ewm(span=window, adjust=False).mean()

            # make sure Date stays as index
    if "Date" in data.columns:
        data = data.set_index("Date")

    return data

def trend_macd(data, short_window=12, long_window=26, signal_window=9):

    data_short = data["Close"].ewm(span=short_window, adjust=False).mean()
    data_long = data["Close"].ewm(span=long_window, adjust=False).mean()
    data_macd = data_short - data_long
    data_signal = data_macd.ewm(span=signal_window, adjust=False).mean()

    data_histogram = data_macd - data_signal

    ##STATE
    #data[f"MACD({short_window},{long_window},{signal_window})"] = np.where(data_macd > data_signal, 1, 0)

    ##CROSSINGS##

    cross_up = (data_histogram.shift(1) <= 0) & (data_histogram > 0)   # crosses from <=0 to >0
    cross_dn = (data_histogram.shift(1) >= 0) & (data_histogram < 0)   # crosses from >=0 to <0

    cross = np.where(cross_up, 1, np.where(cross_dn, -1, 0)).astype(np.int8)
    data[f"MACD({short_window},{long_window},{signal_window})"] = cross

    # make sure Date stays as index
    if "Date" in data.columns:
        data = data.set_index("Date")

    return data

### Volume
def volume_accumulation_distribution_index(data, corr_window=14, corr_threshold=0.20):
    close = data["Close"]
    high  = data["High"]
    low   = data["Low"]
    vol   = data["Volume"]

    # 1) Money Flow Multiplier
    denom = high - low
    mfm = np.where(denom != 0, ((close - low) - (high - close)) / denom, 0.0)
    mfm = pd.Series(mfm, index=data.index, name="MFM")

    # 2) Money Flow Volume
    mfv = mfm * vol
    mfv = pd.Series(mfv, index=data.index, name="MFV")

    # 3) Accumulation/Distribution Line (cumulative)
    adl = mfv.cumsum()
    adl.name = "ADL"


    # --- Relation-based signal (not a crossover) ---
    price_ret = close.pct_change()
    adl_diff  = adl.diff()

    corr = price_ret.rolling(window=corr_window, min_periods=corr_window).corr(adl_diff)

    signal = np.where((corr >  corr_threshold) & (adl_diff > 0),  1,
              np.where((corr < -corr_threshold) & (adl_diff < 0), -1, 0)).astype(np.int8)

    data["ADI"] = signal

    # Make sure Date stays as index
    if "Date" in data.columns:
        data = data.set_index("Date")

    return data

def desperate_vars(df):
    df["HL_PCT"] = (df["High"] - df["Low"]) / df["Close"] * 100.0
    df["PCT_change"] = (df["Close"] - df["Open"]) / df["Open"] * 100.0
    # lag close by 1 day
    df["Close_lag1"] = df["Close"].shift(1)
    if "Date" in df.columns:
        df = df.set_index("Date")
    df = df[['HL_PCT', 'PCT_change', 'Close_lag1']]
    return df

# --- Join all technical variables ---
def join_tch_vars(df):
    data = momentum_rsi(df, window=14)
    data = momentum_stochasticK_oscillator(data, window=14)
    data = momentum_stochasticD_oscillator(data, window=14)
    data = momentum_return(data, window=1, lag=1)
    data = trend_moving_average(data, window=5)
    data = trend_sma_crossover(data, short_window=10, long_window=50)
    data = trend_sma_crossover(data, short_window=50, long_window=200)
    data = trend_exponential_moving_average(data, window=10)
    data = trend_macd(data, short_window=12, long_window=26, signal_window=9)
    data = volume_accumulation_distribution_index(data, corr_window=14, corr_threshold=0.20)
    return data

# --- L&T news data from MongoDB ---
def lt_news_data(ticker):
    low_ticker = ticker.lower()
    MONGO_DB = "financial_news"
    MONGO_COLLECTION = f"{ticker}_news"

    client = MongoClient(mongo_uri)
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
        'relevance_score': f"{low_ticker}_relevance_score",
        'ticker_sentiment_score': f"{low_ticker}_sentiment_score"
    })

    news_df['Date'] = pd.to_datetime(news_df['time_published'],  format="%Y%m%dT%H%M%S")
    news_df["Date"] = news_df["Date"].dt.date
    news_df[f"{low_ticker}_sentiment_score"] = pd.to_numeric(news_df[f"{low_ticker}_sentiment_score"], errors='coerce')
    news_df[f"{low_ticker}_relevance_score"] = pd.to_numeric(news_df[f"{low_ticker}_relevance_score"], errors='coerce')

    sentiment_df = news_df.groupby("Date").apply(
        lambda g: (g[f"{low_ticker}_sentiment_score"] * g[f"{low_ticker}_relevance_score"]).sum() / g[f"{low_ticker}_relevance_score"].sum()
        ).reset_index(name="weighted_sentiment")
    
    sentiment_df = sentiment_df.set_index('Date').sort_index()
         
    return sentiment_df

def lt_tech_news():
    MONGO_DB = "financial_news"
    MONGO_COLLECTION = "tech_news"

    client = MongoClient(mongo_uri)
    db = client[MONGO_DB]
    collection = db[MONGO_COLLECTION]

    # Fetch all documents
    docs = list(collection.find())

    rows = []

    for item in docs:
        # Extract publication date (YYYY-MM-DD)
        if "time_published" not in item:
            continue
        try:
            date = datetime.strptime(item["time_published"][:8], "%Y%m%d").date()
        except Exception:
            continue

        # Find Technology topic relevance
        tech_relevance = 0
        for t in item.get("topics", []):
            if t.get("topic", "").lower() == "technology":
                tech_relevance = float(t.get("relevance_score", 0))
                break

        # Skip if not tech-related
        if tech_relevance == 0:
            continue

        sentiment = item.get("overall_sentiment_score", 0)

        # Weighted contribution
        rows.append({
            "date": date,
            "sentiment": sentiment,
            "relevance": tech_relevance
        })

    # Convert to DataFrame
    df_rows = pd.DataFrame(rows)

    if df_rows.empty:
        print("No technology-related news found.")
        return pd.DataFrame(columns=["date", "weighted_tech_sentiment"])

    # Compute weighted daily sentiment
    weighted_df = (
        df_rows.groupby("date")
        .apply(lambda x: (x["sentiment"] * x["relevance"]).sum() / x["relevance"].sum())
        .reset_index(name="weighted_tech_sentiment")
    )

    weighted_df.rename(columns={'date': 'Date'}, inplace=True)
    weighted_df.set_index('Date', inplace=True)
    return weighted_df

def lt_sec_data(db_path):
    if os.path.exists(db_path):
    
        table = db_path.split("/")[-1].replace(".duckdb", "")

        con_sec = duckdb.connect(db_path)
        sec_df = con_sec.execute(f"SELECT * FROM {table}").df()
        con_sec.close()

        sec_df = sec_df.drop('concept', axis=1).set_index('label').T
        sec_df = sec_df.rename_axis('Date')
        sec_df.index = pd.to_datetime(sec_df.index)

    else:
        print(f"Warning: Delta path {db_path} does not exist.")
        sec_df = pd.DataFrame() 

    return sec_df

def lt_treasury_data(db_path="./data/treasury_yield.duckdb"):
    df = load_data(db_path)
    df["term_spread"]    = df["10y"] - df["3m"]  # 10Y - 3M
    df["default_spread"] = df["10y"] - df["5y"]  # 10Y - 5Y
    return df

def lt_fred_daily_data(db_path="./data/fred_daily.duckdb"):
    df = load_data(db_path)
    df["spread_baa_aaa"] = df["DBAA"]  - df["DAAA"]
    df["spread_baa_10y"] = df["DBAA"]  - df["DGS10"]
    return df

def create_df(ticker):
    df = load_data(db_path=f"./data/{ticker}_yf.duckdb")
    dp_df = desperate_vars(df)
    # df_tech = join_tch_vars(df)

    sec_10q_df = lt_sec_data(db_path=f"./data/{ticker}_10q.duckdb")
    sec_10k_df = lt_sec_data(db_path=f"./data/{ticker}_10k.duckdb")
    sec_10k_df = sec_10k_df.add_suffix('_10k')
    sec_df = pd.concat([sec_10q_df, sec_10k_df]).drop_duplicates().sort_index()

    news_df = lt_news_data(ticker)
    tech_news_df = lt_tech_news()

    instruments_df = load_data(db_path=f"./data/instruments.duckdb")

    d_market_df = load_data(db_path=f"./data/{ticker}_daily_market.duckdb")

    q_fund_df = load_data(db_path=f"./data/{ticker}_quarterly_fundamentals.duckdb")

    treasury_df = lt_treasury_data(db_path="./data/treasury_yields.duckdb")

    fred_daily_df = lt_fred_daily_data(db_path="./data/fred_daily.duckdb")

    # Merge all dataframes on date

    # if not sec_df.empty:
    #     data = df.merge(sec_df, how='left', left_index=True, right_index=True)
    # else:
    data = df.copy()
    data = data.merge(news_df, how='left', left_index=True, right_index=True)
    data = data.merge(tech_news_df, how='left', left_index=True, right_index=True)
    # data = data.merge(instruments_df, how='left', left_index=True, right_index=True)
    # data = data.merge(d_market_df, how='left', left_index=True, right_index=True)
    # data = data.merge(q_fund_df, how='left', left_index=True, right_index=True)
    # data = data.merge(treasury_df, how='left', left_index=True, right_index=True)
    # data = data.merge(fred_daily_df, how='left', left_index=True, right_index=True)
    data = data.dropna(subset=['Close'])  # Drop rows where target is NaN
    data = data.ffill()
    data = data.dropna(axis=1, how='all')  # Drop columns that are all NaN

    #if 'weighted_sentiment' in data.columns:
    #    data = data.dropna(subset=['weighted_sentiment'])

    # data = data.drop(columns=['Open','High','Low','Volume'], errors='ignore')
    print(data.shape)

    return data

# def cross_corr(a, b, lag=10):
#     return [np.corrcoef(a[:-i], b[i:])[0, 1] for i in range(1, lag)]

if __name__ == "__main__":
    ticker = 'AAPL'
    df = create_df(ticker)
    df = df.ffill()
    print(df.columns)
