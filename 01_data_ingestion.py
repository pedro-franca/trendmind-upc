import yfinance as yf
import os
from deltalake import DeltaTable
from deltalake.writer import write_deltalake



def load_financial_data(ticker):
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



if __name__ == "__main__":
    # Example usage
    ticker = "ORCL"
    try:
        load_financial_data(ticker)
        yf_data = DeltaTable(f"./data/{ticker}_yfinance")
        print(f"Data loaded for {ticker}:")
        print(yf_data.to_pandas().head())

    except ValueError as e:
        print(e)