from delta import configure_spark_with_delta_pip
import duckdb
import pandas as pd
from deltalake import DeltaTable
from pyspark.sql.functions import explode, col
from pyspark.sql import SparkSession

# --- Step 1: Read from Delta Lake ---
def read_yf_df(ticker):
    delta_path = f"./data/{ticker}_yfinance"
    dt = DeltaTable(delta_path)
    df = dt.to_pandas()
    return df

def read_news_df(ticker):
    delta_path = f"./data/{ticker}_news"
    dt = DeltaTable(delta_path)
    df = dt.to_pandas()
    # Explode the 'ticker_sentiment' column (which is a list of dicts) into separate rows
    exploded_df = df.explode('ticker_sentiment').reset_index(drop=True)

    # Now each row has a single ticker_sentiment dict; normalize it into separate columns
    ticker_df = pd.json_normalize(exploded_df['ticker_sentiment'])

    # Concatenate with the original DataFrame
    exploded_df = pd.concat([exploded_df.drop(columns=['ticker_sentiment']), ticker_df], axis=1)

    # Filter for ticker 'ORCL' and select relevant columns
    sentiment_df = exploded_df[exploded_df['ticker'] == ticker][[
        'relevance_score',
        'ticker_sentiment_score',
        'time_published'
    ]].rename(columns={
        'relevance_score': 'orcl_relevance_score',
        'ticker_sentiment_score': 'orcl_sentiment_score'
    })
    print(sentiment_df.head())  # Optional preview
    return sentiment_df

#### If we want to perform any cleaning, we can do it here.
# For example, let's drop rows with any NaN values.
def clean_df(df):
    df_cleaned = df.dropna()
    return df_cleaned

# --- Step 2: Write to DuckDB ---
def export_to_duckdb(ticker):
    yf_df = read_yf_df(ticker)
    yf_df = clean_df(yf_df)
    news_df = read_news_df(ticker)
    news_df = clean_df(news_df)
    print(yf_df.head()) # Optional preview
    print(news_df.head())  # Optional preview

    output_yf_path = f"./data/{ticker}_yf_cleaned.duckdb"
    output_news_path = f"./data/{ticker}_news_cleaned.duckdb"
    # Export yf_df to DuckDB
    con = duckdb.connect(database=output_yf_path, read_only=False)
    con.execute("CREATE OR REPLACE TABLE stock_data AS SELECT * FROM yf_df")
    con.close()
    print(f"✅ Data exported to {output_yf_path}")
    # Export news_df to DuckDB
    con = duckdb.connect(database=output_news_path, read_only=False)
    con.execute("CREATE OR REPLACE TABLE news_data AS SELECT * FROM news_df")
    con.close()
    print(f"✅ Data exported to {output_news_path}")

if __name__ == "__main__":
    ticker = "ORCL"
    try:
        export_to_duckdb(ticker)
        print(f"✅ Data exported for {ticker}.")
    except Exception as e:
        print(f"❌ Error processing {ticker}: {e}")
