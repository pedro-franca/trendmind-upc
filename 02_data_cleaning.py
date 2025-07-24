import duckdb
import pandas as pd
from deltalake import DeltaTable

# --- Step 1: Read from Delta Lake ---
def read_df(ticker):
    delta_path = f"./data/{ticker}_yfinance"
    dt = DeltaTable(delta_path)
    df = dt.to_pandas()
    return df

#### If we want to perform any cleaning, we can do it here.
# For example, let's drop rows with any NaN values.
def clean_df(df):
    df_cleaned = df.dropna()
    return df_cleaned

# --- Step 2: Write to DuckDB ---
def export_to_duckdb(ticker):
    df = read_df(ticker)
    df = clean_df(df)
    print(df.head())  # Optional preview

    output_path = f"./data/{ticker}_cleaned.duckdb"
    con = duckdb.connect(database=output_path, read_only=False)
    con.execute("CREATE OR REPLACE TABLE stock_data AS SELECT * FROM df")
    con.close()
    print(f"✅ Data exported to {output_path}")

if __name__ == "__main__":
    ticker = "AAPL"
    try:
        export_to_duckdb(ticker)
        print(f"✅ Data exported for {ticker}.")
    except Exception as e:
        print(f"❌ Error processing {ticker}: {e}")
