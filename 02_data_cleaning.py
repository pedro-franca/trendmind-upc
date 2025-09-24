import json
import duckdb
import pandas as pd
from deltalake import DeltaTable
from pymongo import MongoClient


# --- Read df from Delta Lake ---
def read_df(delta_path=None):
    dt = DeltaTable(delta_path)
    df = dt.to_pandas()
    return df


#### If we want to perform any cleaning, we can do it here.
# For example, let's drop rows with any NaN values.
def clean_df(df):
    df_cleaned = df.dropna()
    return df_cleaned

# --- Write to DuckDB ---
def export_to_duckdb(delta_path=None, output_path=None):
    df = read_df(delta_path)
   # df = clean_df(df)

    print(df.head()) # Optional preview
    table = delta_path.split("/")[-1].replace(".delta", "")
    # Export df to DuckDB
    con = duckdb.connect(database=output_path, read_only=False)
    con.execute(f"CREATE OR REPLACE TABLE {table} AS SELECT * FROM df")
    con.close()
    print(f"✅ Data exported to {output_path}")

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
        export_to_duckdb(delta_path=f"./data/{ticker}_yf", output_path=f"./data/{ticker}_yf.duckdb")
        print(f"✅ yf Data exported for {ticker}.")
        export_to_duckdb(delta_path=f"./data/{ticker}_10k", output_path=f"./data/{ticker}_10k.duckdb")
        print(f"✅ 10-K Data exported for {ticker}.")
        export_to_duckdb(delta_path=f"./data/{ticker}_10q", output_path=f"./data/{ticker}_10q.duckdb")
        print(f"✅ 10-Q Data exported for {ticker}.")
        export_news_to_mongodb(ticker)
        print(f"✅ News exported for {ticker}.")
        export_to_duckdb(delta_path=f"./data/instruments", output_path=f"./data/instruments.duckdb")
        print(f"✅ Instruments Data exported.")
    except Exception as e:
        print(f"❌ Error processing {ticker}: {e}")
