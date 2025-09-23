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

    yf_df['Date'] = pd.to_datetime(yf_df['Date'])
    yf_df['Date'] = yf_df['Date'].dt.date
    yf_df = yf_df.drop(columns=['ticker'], errors='ignore')  # Drop ticker if exists
 
    return yf_df

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

def momentum_log_return(data, window=1, lag=1):
    # log return: log(P_t / P_{t-window})
    log_ret = np.log(data["Close"] / data["Close"].shift(window))

    # shift the result to avoid lookahead bias
    if lag:
        log_ret = log_ret.shift(lag)

    # add column with clear name
    data[f"LogReturn_lag{lag}"] = log_ret

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

# --- Join all technical variables ---
def join_tch_vars(df):
    data = momentum_rsi(df, window=14)
    data = momentum_stochasticK_oscillator(data, window=14)
    data = momentum_stochasticD_oscillator(data, window=14)
    data = momentum_return(data, window=1, lag=1)
    data = momentum_log_return(data, window=1, lag=1)
    data = trend_moving_average(data, window=5)
    data = trend_sma_crossover(data, short_window=10, long_window=50)
    data = trend_exponential_moving_average(data, window=10)
    data = trend_macd(data, short_window=12, long_window=26, signal_window=9)
    data = volume_accumulation_distribution_index(data, corr_window=14, corr_threshold=0.20)
    return data


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
    try:
        df = lt_yf_data(ticker)
        print(f"✅ Data loaded and transformed for {ticker}.")
        print(df.head())  # Optional preview
        df = join_tch_vars(df)
        print(f"✅ Technical variables added for {ticker}.")
        print(df.head())  # Optional preview
    except Exception as e:
        print(f"❌ Error loading data for {ticker}: {e}")