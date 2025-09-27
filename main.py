import data_ingestion as di
from deltalake import DeltaTable
import data_cleaning as dc
import feature_engineering as fe
import feature_selection as fs
from LSTM_training import train_test_lstm


def main(ticker):
    # 1. Load raw data
    di.get_yf_data(ticker)
    di.get_10k_data(ticker)
    di.get_10q_data(ticker)
    di.get_daily_market_data(ticker)
    di.get_quarterly_fundamentals(ticker)

    # 2. Clean / preprocess / Export to DuckDB/MongoDB
    dc.export_to_duckdb(delta_path=f"./data/{ticker}_yf", output_path=f"./data/{ticker}_yf.duckdb")
    dc.export_to_duckdb(delta_path=f"./data/{ticker}_10k", output_path=f"./data/{ticker}_10k.duckdb")
    dc.export_to_duckdb(delta_path=f"./data/{ticker}_10q", output_path=f"./data/{ticker}_10q.duckdb")
    dc.export_to_duckdb(delta_path=f"./data/{ticker}_daily_market", output_path=f"./data/{ticker}_daily_market.duckdb")
    dc.export_to_duckdb(delta_path=f"./data/{ticker}_quarterly_fundamentals", output_path=f"./data/{ticker}_quarterly_fundamentals.duckdb")
    dc.export_news_to_mongodb(ticker)

    # 3. Train model
    model, scaler, metrics = train_test_lstm(ticker, threshold=0.6, target_col='Close')
    print("Model training completed with metrics:", metrics)

if __name__ == "__main__":
    ticker = "ORCL"
    try:
        main(ticker)
    except Exception as e:
        print(f"Error occurred: {e}")
