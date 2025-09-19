import json
import duckdb
import pandas as pd
from deltalake import DeltaTable
from pymongo import MongoClient


# --- Read yf from Delta Lake ---
def read_yf_df(ticker):
    delta_path = f"./data/{ticker}_yfinance"
    dt = DeltaTable(delta_path)
    df = dt.to_pandas()
    return df


#### If we want to perform any cleaning, we can do it here.
# For example, let's drop rows with any NaN values.
def clean_df(df):
    df_cleaned = df.dropna()
    return df_cleaned

# --- Write to DuckDB ---
def export_to_duckdb(ticker):
    yf_df = read_yf_df(ticker)
    yf_df = clean_df(yf_df)

    print(yf_df.head()) # Optional preview

    output_yf_path = f"./data/{ticker}_yf_cleaned.duckdb"

    # Export yf_df to DuckDB
    con = duckdb.connect(database=output_yf_path, read_only=False)
    con.execute("CREATE OR REPLACE TABLE stock_data AS SELECT * FROM yf_df")
    con.close()
    print(f"✅ Data exported to {output_yf_path}")

# --- Write news to MongoDB ---
def export_news_to_mongodb(ticker):

    json_path = f"./data/news/{ticker}_filtered.json"

    with open(json_path, 'r') as f:
        json_data = json.load(f)
    
    MONGO_DB = "financial_news"
    MONGO_COLLECTION = f"{ticker}_news"

    if isinstance(json_data, dict):
        json_data = [json_data]
    
    # Connect to MongoDB (make sure MongoDB is running and accessible)
    client = MongoClient("mongodb://localhost:27017/")
    db = client[MONGO_DB]
    collection = db[MONGO_COLLECTION]

    collection.insert_many(json_data)
    print(f"✅ Inserted {len(json_data)} news docs into MongoDB {MONGO_DB}.{MONGO_COLLECTION}")



if __name__ == "__main__":
    ticker = "ORCL"
    try:
        export_to_duckdb(ticker)
        print(f"✅ Data exported for {ticker}.")
        export_news_to_mongodb(ticker)
        print(f"✅ News exported for {ticker}.")
    except Exception as e:
        print(f"❌ Error processing {ticker}: {e}")
