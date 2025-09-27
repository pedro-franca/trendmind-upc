import time
import pandas as pd
import requests
import yfinance as yf
import os
from deltalake import DeltaTable
from deltalake.writer import write_deltalake
import nasdaqdatalink as ndl
from edgar.financials import XBRLS
from edgar import set_identity, Company


##api_keys
set_identity("pedrochfr@gmail.com")
ndl.ApiConfig.api_key = "Xnd9ezf-sU1GuNmnjHcs"
FRED_API_KEY = "5d43b13c3b19f8c93900198f5b6dabc2"

# Define features to fetch from SHARADAR SF1 and DAILY datasets
quarterly_features = [
    "ticker","calendardate","reportperiod",
    "assets","assetsc","assetsnc",
    "bvps","capex","cashneq","cashnequsd","consolinc","cor",
    "currentratio","de","debt","debtc","debtnc","debtusd",
    "deferredrev","depamor","deposits","divyield","dps",
    "ebit","ebitda","ebitdamargin","ebitdausd","ebitusd","ebt",
    "eps","epsdil","epsusd","equity","equityavg","equityusd",
    "ev","evebit","evebitda","fcf","fcfps","gp",
    "grossmargin","intangibles","intexp","invcap"
]

daily_features = ['date',
  'ev',
  'evebit',
  'evebitda',
  'marketcap',
  'pb',
  'pe',
  'ps',
  'ticker']

# --- Fetch non ticker specific data ---
def fetch_instruments():
    DELTA_OUTPUT_PATH = f"./data/instruments"
    df = pd.DataFrame()
    tickers = ["CL=F", "GLD", "^BCOM", "DX-Y.NYB"]
    for ticker in tickers:
        data = yf.download(ticker, start="2020-01-01", end="2023-08-25", interval="1d")
        data = data.droplevel('Ticker', axis=1)[['Close']]
        data.rename(columns={'Close': ticker}, inplace=True)
        df = pd.concat([df, data], axis=1)

    df.columns = ['crude_oil','gold','commodity_index','usd_index']

    os.makedirs(DELTA_OUTPUT_PATH, exist_ok=True)
    write_deltalake(DELTA_OUTPUT_PATH, df, mode="overwrite")

    print(f"✅ Exported {len(df)} rows to {DELTA_OUTPUT_PATH}")

def fetch_treasury_yields():
    DELTA_OUTPUT_PATH = f"./data/treasury_yields"
    df = pd.DataFrame()
    tickers = ["^TNX", "^IRX", "^FVX", "^TYX"]  # 10-year, 3-month, 5-year, and 30-year treasury yield
    for ticker in tickers:
        data = yf.download(ticker, start="2020-01-01", end="2023-08-25", interval="1d")
        data = data.droplevel('Ticker', axis=1)[['Close']]
        data.rename(columns={'Close': ticker}, inplace=True)
        df = pd.concat([df, data], axis=1)

    df.columns = ['10y','3m','5y','30y']

    os.makedirs(DELTA_OUTPUT_PATH, exist_ok=True)
    write_deltalake(DELTA_OUTPUT_PATH, df, mode="overwrite")

    print(f"✅ Exported {len(df)} rows to {DELTA_OUTPUT_PATH}")

def fetch_fred_daily(start="2020-01-01", end="2025-08-23", retries=4):
    BASE_URL = "https://api.stlouisfed.org/fred/series/observations"
    data = pd.DataFrame()
    for series_id in ["DBAA", "DAAA","DGS10"]:
        params = {
            "series_id": series_id,
            "api_key": FRED_API_KEY,
            "file_type": "json",
            "observation_start": start,
            "observation_end": end,
        }
        for i in range(retries):
            try:
                r = requests.get(BASE_URL, params=params, timeout=60)
                r.raise_for_status()
                obs = r.json().get("observations", [])
                df = pd.DataFrame(obs, columns=["date","value"])
                df["date"]  = pd.to_datetime(df["date"], errors="coerce")
                df["value"] = pd.to_numeric(df["value"], errors="coerce")  # '.' -> NaN
                df = df.dropna(subset=["date"]).set_index("date").sort_index()
                df = df["value"].rename(series_id)
            except requests.RequestException:
                if i == retries - 1:
                    raise
                time.sleep(2 ** i)  # Exponential backoff
        data = pd.concat([data, df], axis=1)
    data = data.reset_index().rename(columns={"index": "Date"})

    DELTA_OUTPUT_PATH = f"./data/fred_daily"
    os.makedirs(DELTA_OUTPUT_PATH, exist_ok=True)
    write_deltalake(DELTA_OUTPUT_PATH, data, mode="overwrite")
    print(f"✅ Exported {len(data)} rows to {DELTA_OUTPUT_PATH}")


# --- Fetch and store data functions ---
def get_yf_data(ticker):
    """
    Load financial data for a given ticker.
    
    Args:
        ticker (str): The stock ticker symbol.
        
    Returns:
        pd.DataFrame: DataFrame containing financial data.
    """
    DELTA_OUTPUT_PATH = f"./data/{ticker}_yf"
    # Fetch data from Yahoo Finance
    df = yf.download(ticker, start="2020-01-01", end="2025-08-23", interval="1d")
    df = df.droplevel('Ticker', axis=1)
    df['ticker'] = ticker
    if df.empty:
        raise ValueError(f"No data found for ticker: {ticker}")

    os.makedirs(DELTA_OUTPUT_PATH, exist_ok=True)
    write_deltalake(DELTA_OUTPUT_PATH, df, mode="overwrite")

    print(f"✅ Exported {len(df)} rows to {DELTA_OUTPUT_PATH}")

def get_10k_data(ticker):
    """
    Placeholder function to load 10-K data for a given ticker.
    
    Args:
        ticker (str): The stock ticker symbol.
    """
    c = Company(ticker)
    filings = c.get_filings(form="10-K").latest(5)
    if not filings:
        print(f"Warning: No 10-K filings found for ticker: {ticker}")
    else:
        xbs = XBRLS.from_filings(filings)
        income_statement = xbs.statements.income_statement()
        ten_k_df = income_statement.to_dataframe()
        DELTA_OUTPUT_PATH = f"./data/{ticker}_10k"
        os.makedirs(DELTA_OUTPUT_PATH, exist_ok=True)
        write_deltalake(DELTA_OUTPUT_PATH, ten_k_df, mode="overwrite")
        
        print(f"✅ Exported {len(ten_k_df)} rows to {DELTA_OUTPUT_PATH}")

def get_10q_data(ticker):
    """
    Placeholder function to load 10-Q data for a given ticker.
    
    Args:
        ticker (str): The stock ticker symbol.
    """
    c = Company(ticker)
    filings = c.get_filings(form="10-Q").latest(10)
    if not filings:
        print(f"Warning: No 10-Q filings found for ticker: {ticker}")
    else:
        xbs = XBRLS.from_filings(filings)
        income_statement = xbs.statements.income_statement()
        ten_q_df = income_statement.to_dataframe()
        DELTA_OUTPUT_PATH = f"./data/{ticker}_10q"
        os.makedirs(DELTA_OUTPUT_PATH, exist_ok=True)
        write_deltalake(DELTA_OUTPUT_PATH, ten_q_df, mode="overwrite")

        print(f"✅ Exported {len(ten_q_df)} rows to {DELTA_OUTPUT_PATH}")

def get_quarterly_fundamentals(ticker: str, since="2000-01-01"):
    if ticker == "GOOG":
        ticker_adapt = "GOOGL"  # Adjust for NASDAQ listing
    else:
        ticker_adapt = ticker
    df = ndl.get_table(
        "SHARADAR/SF1",
        ticker=ticker_adapt,
        dimension="ARQ",  # Quarterly as-reported
        calendardate={"gte": since},
        qopts={"columns": quarterly_features},
        paginate=True
    )

    df = df.reset_index(drop=True)
    df = df.drop(columns=["reportperiod"]).rename(columns={"calendardate": "Date"})
    df = df.dropna(axis=1, how="all")
    if df.empty:
        print(f"No quarterly fundamentals found for ticker: {ticker}")
    else:
        DELTA_OUTPUT_PATH = f"./data/{ticker}_quarterly_fundamentals"
        os.makedirs(DELTA_OUTPUT_PATH, exist_ok=True)
        write_deltalake(DELTA_OUTPUT_PATH, df, mode="overwrite")

        print(f"✅ Exported {len(df)} rows to {DELTA_OUTPUT_PATH}")

def get_daily_market_data(ticker: str, since="2000-01-01"):
    if ticker == "GOOG":
        ticker_adapt = "GOOGL"  # Adjust for NASDAQ listing
    else:
        ticker_adapt = ticker
    df = ndl.get_table(
        "SHARADAR/DAILY",
        ticker=ticker_adapt,
        date={"gte": since},
        qopts={"columns": daily_features},
        paginate=True
    )
    df = df.reset_index(drop=True)
    df = df.dropna(axis=1, how="all")
    if df.empty:
        print(f"No daily market data found for ticker: {ticker}")
    else:
        DELTA_OUTPUT_PATH = f"./data/{ticker}_daily_market"
        os.makedirs(DELTA_OUTPUT_PATH, exist_ok=True)
        write_deltalake(DELTA_OUTPUT_PATH, df, mode="overwrite")

        print(f"✅ Exported {len(df)} rows to {DELTA_OUTPUT_PATH}")

