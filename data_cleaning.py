import json
import duckdb
import pandas as pd
import os
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
    if 'date' in df.columns:
        df.rename(columns={'date': 'Date'}, inplace=True)
    return df

# --- Write to DuckDB ---
def export_to_duckdb(delta_path=None, output_path=None):
    if os.path.exists(delta_path): 
        df = read_df(delta_path)
        df = clean_df(df)

        print(df.head()) # Optional preview
        table = delta_path.split("/")[-1].replace(".delta", "")
        # Export df to DuckDB
        con = duckdb.connect(database=output_path, read_only=False)
        con.execute(f"CREATE OR REPLACE TABLE {table} AS SELECT * FROM df")
        con.close()
        print(f"✅ Data exported to {output_path}")
    else:
        print(f"Delta path {delta_path} does not exist.")

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
    export_to_duckdb(delta_path="./data/fred_daily", output_path="./data/fred_daily.duckdb")