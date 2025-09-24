import pandas as pd
import yfinance as yf
import os
from deltalake import DeltaTable
from deltalake.writer import write_deltalake

from edgar.financials import XBRLS
from edgar import set_identity, Company

set_identity("pedrochfr@gmail.com")


def load_yf_data(ticker):
    """
    Load financial data for a given ticker.
    
    Args:
        ticker (str): The stock ticker symbol.
        
    Returns:
        pd.DataFrame: DataFrame containing financial data.
    """
    DELTA_OUTPUT_PATH = f"./data/{ticker}_yfinance"
    # Fetch data from Yahoo Finance
    df = yf.download(ticker, start="2020-01-01", end="2023-08-25")
    df = df.droplevel('Ticker', axis=1)
    df['ticker'] = ticker
    if df.empty:
        raise ValueError(f"No data found for ticker: {ticker}")

    os.makedirs(DELTA_OUTPUT_PATH, exist_ok=True)
    write_deltalake(DELTA_OUTPUT_PATH, df, mode="overwrite")

    print(f"✅ Exported {len(df)} rows to {DELTA_OUTPUT_PATH}")

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

def load_10k_data(ticker):
    """
    Placeholder function to load 10-K data for a given ticker.
    
    Args:
        ticker (str): The stock ticker symbol.
    """
    c = Company(ticker)
    filings = c.get_filings(form="10-K").latest(5)
    xbs = XBRLS.from_filings(filings)
    income_statement = xbs.statements.income_statement()
    ten_k_df = income_statement.to_dataframe()

    DELTA_OUTPUT_PATH = f"./data/{ticker}_10k"
    os.makedirs(DELTA_OUTPUT_PATH, exist_ok=True)
    write_deltalake(DELTA_OUTPUT_PATH, ten_k_df, mode="overwrite")
    
    print(f"✅ Exported {len(ten_k_df)} rows to {DELTA_OUTPUT_PATH}")

def load_10q_data(ticker):
    """
    Placeholder function to load 10-Q data for a given ticker.
    
    Args:
        ticker (str): The stock ticker symbol.
    """
    c = Company(ticker)
    filings = c.get_filings(form="10-Q").latest(10)
    xbs = XBRLS.from_filings(filings)
    income_statement = xbs.statements.income_statement()
    ten_q_df = income_statement.to_dataframe()

    DELTA_OUTPUT_PATH = f"./data/{ticker}_10q"
    os.makedirs(DELTA_OUTPUT_PATH, exist_ok=True)
    write_deltalake(DELTA_OUTPUT_PATH, ten_q_df, mode="overwrite")

    print(f"✅ Exported {len(ten_q_df)} rows to {DELTA_OUTPUT_PATH}")


if __name__ == "__main__":
    # Example usage
    ticker = "ORCL"
    try:
        load_yf_data(ticker)
        fetch_instruments()
        load_10k_data(ticker)
        load_10q_data(ticker)
        yf_data = DeltaTable(f"./data/{ticker}_yfinance")
        ten_k_data = DeltaTable(f"./data/{ticker}_10k")
        ten_q_data = DeltaTable(f"./data/{ticker}_10q")
        instruments_data = DeltaTable(f"./data/instruments")

        print(f"Data loaded for {ticker}:")
        print(yf_data.to_pandas().head())
        print(ten_k_data.to_pandas().head())
        print(ten_q_data.to_pandas().head())
        print(instruments_data.to_pandas().head())

    except ValueError as e:
        print(e)