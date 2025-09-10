from delta import configure_spark_with_delta_pip
import pandas as pd
import numpy as np
import json
import yfinance as yf
import os
from deltalake import DeltaTable
from deltalake.writer import write_deltalake
from pyspark.sql import SparkSession


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
    df = yf.download(ticker, start="2020-01-01", end="2025-08-25")
    df = df.droplevel('Ticker', axis=1)
    df['ticker'] = ticker
    if df.empty:
        raise ValueError(f"No data found for ticker: {ticker}")

    os.makedirs(DELTA_OUTPUT_PATH, exist_ok=True)
    write_deltalake(DELTA_OUTPUT_PATH, df, mode="overwrite")

    print(f"✅ Exported {len(df)} rows to {DELTA_OUTPUT_PATH}")

def load_ticker_json(ticker):
    """
    Load ticker data from a JSON file.
    
    Args:
        ticker (str): The stock ticker symbol.
    Returns:
        pd.DataFrame: DataFrame containing ticker data.
    """
    json_path = "./data/merged_deduped_news_v1.json"

    if not os.path.exists(json_path):
        raise ValueError(f"JSON file not found at path: {json_path}")

    data = json.load(open(json_path, "r", encoding="utf-8"))
    print(f"Total feed items in JSON: {len(data['feed'])}")
    ticker_feeds = []
    for item in data['feed']:
        if "ticker_sentiment" in item:
            for ticker_info in item["ticker_sentiment"]:
                if ticker_info.get("ticker") == ticker:
                    ticker_feeds.append(item)
                    break # Add break to avoid adding the same item multiple times if 'ORCL' appears multiple times in ticker_sentiment
    print(f"Total feed items for {ticker}: {len(ticker_feeds)}")

        # Save filtered JSON temporarily
    temp_json_path = f"./data/{ticker}_filtered.json"
    with open(temp_json_path, "w", encoding="utf-8") as f:
        json.dump(ticker_feeds, f, ensure_ascii=False, indent=2)

    local_jars = "./jars/delta-spark_2.13-4.0.0.jar,./jars/delta-storage-4.0.0.jar,./jars/antlr4-runtime-4.13.1.jar"


    spark = SparkSession.builder \
    .appName("JSON to Delta") \
    .config("spark.jars", local_jars) \
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
    .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
    .getOrCreate()

    # Read JSON into Spark DataFrame
    df = spark.read.option("multiline", "true").json(temp_json_path)

    # Inspect schema (optional, useful for debugging)
    df.printSchema()
    df.show(5)

    # Define Delta output path
    delta_path = f"./data/{ticker}_news"

    # Write DataFrame as Delta
    df.repartition(10).write.format("delta").mode("overwrite").save(delta_path)

    print(f"JSON data written to Delta at {delta_path}")



if __name__ == "__main__":
    # Example usage
    ticker = "ORCL"
    try:
        load_financial_data(ticker)
        load_ticker_json(ticker)
        yf_data = DeltaTable(f"./data/{ticker}_yfinance")
        news_data = DeltaTable(f"./data/{ticker}_news")
        print(f"Data loaded for {ticker}:")
        print(yf_data.to_pandas().head())
        print(news_data.to_pandas().head())

    except ValueError as e:
        print(e)