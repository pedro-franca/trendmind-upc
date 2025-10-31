import pickle
from time import time
import joblib
import keras
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import feature_engineering as fe
import matplotlib.pyplot as plt
from LSTM_training_megagrid import combined_loss, forecast_future
from sklearn.metrics import mean_absolute_error, mean_squared_error

feature_cols = {'AAPL': ['returns', 'PCT_change', 'weighted_tech_sentiment', 'Volume', 'Return_lag1'],
                'AVGO': ['returns', 'PCT_change', 'weighted_tech_sentiment', 'ADI', 'HL_PCT', '%K_14', 'Volume'],
                'GOOG': ['returns', 'PCT_change', 'weighted_tech_sentiment', '%K_14', 'crude_oil', 'gold', 'Return_lag1', 'commodity_index', 'Close', '%D', 'default_spread', 'usd_index'],
                'META': ['returns', 'PCT_change', 'weighted_tech_sentiment'],
                'MSFT': ['returns', 'PCT_change', 'weighted_tech_sentiment', 'Volume', 'HL_PCT', 'Close_lag1', '%K_14', '3m', 'RSI_14', 'commodity_index', 'Return_lag1', 'crude_oil'],
                'NVDA': ['returns', 'PCT_change', 'weighted_tech_sentiment', '%K_14', 'ADI', 'HL_PCT', 'pe', 'evebitda_x', 'evebit_x', 'Close_lag1', 'Volume', 'Close'],
                'ORCL': ['returns', 'PCT_change', 'weighted_tech_sentiment', 'HL_PCT', '%K_14', 'ADI', 'Volume', 'Close_lag1', '3m', 'Close', 'SMA_5', 'spread_baa_10y'],
                'TCEHY': ['returns', 'PCT_change', 'weighted_tech_sentiment', '%K_14', 'RSI_14', 'ADI', 'MACD(12,26,9)'],
                'TSM': ['returns', 'PCT_change', 'weighted_tech_sentiment', 'ADI', '%K_14', 'RSI_14', 'MACD(12,26,9)', 'Volume', '%D', 'debtc', 'term_spread', 'commodity_index']}

days_past = {'AAPL': 10,
                'AVGO': 10,
                'GOOG': 20,
                'META': 40,
                'MSFT': 20,
                'NVDA': 10,
                'ORCL': 10,
                'TCEHY': 60,
                'TSM': 60}

metrics_results = []
preds_results = []
for ticker in ["AAPL", "GOOG", "MSFT", "TSM", "AVGO", "META", "NVDA", "ORCL", "TCEHY"]:
    # with open(f"pkl/forecast_returns_{ticker}_univ.pkl", "rb") as f:
    #     model = pickle.load(f)
    model = keras.models.load_model(f"artifacts/best_model_{ticker}_4.keras", custom_objects={'combined_loss': combined_loss})

    df = yf.download(ticker, start="2025-02-25", end="2025-10-20", interval="1d")
    df = df.droplevel('Ticker', axis=1)
    dp_df = fe.desperate_vars(df)
    df_tech = fe.join_tch_vars(df)
    df["returns"] = df["Close"].pct_change()

    news_df = fe.lt_news_data(ticker)
    tech_news_df_train = fe.lt_tech_news(collection_name="tech_news")
    tech_news_df_test = fe.lt_tech_news(collection_name="tech_sept_news")
    tech_news_df = pd.concat([tech_news_df_train, tech_news_df_test]).drop_duplicates().sort_index()
    d_market_df = fe.load_data(db_path=f"./data/{ticker}_daily_market.duckdb")
    q_fund_df = fe.load_data(db_path=f"./data/{ticker}_quarterly_fundamentals.duckdb")
    instruments_df = fe.load_data(db_path="./data/instruments.duckdb")
    treasury_df = fe.lt_treasury_data(db_path="./data/treasury_yields.duckdb")
    fred_daily_df = fe.lt_fred_daily_data(db_path="./data/fred_daily.duckdb")
    sec_10q_df = fe.lt_sec_data(db_path=f"./data/{ticker}_10q.duckdb")
    sec_10k_df = fe.lt_sec_data(db_path=f"./data/{ticker}_10k.duckdb")
    sec_10k_df = sec_10k_df.add_suffix('_10k')
    sec_df = pd.concat([sec_10q_df, sec_10k_df]).drop_duplicates().sort_index()

    if not sec_df.empty:
        df = df.merge(sec_df, how='left', left_index=True, right_index=True)
    else:
        df = df.copy()
    df = df.merge(news_df, how='left', left_index=True, right_index=True)
    df = df.merge(tech_news_df, how='left', left_index=True, right_index=True)
    df = df.merge(d_market_df, left_index=True, right_index=True, how="left")
    df = df.merge(q_fund_df, left_index=True, right_index=True, how="left")
    df = df.merge(instruments_df, left_index=True, right_index=True, how="left")
    df = df.merge(treasury_df, left_index=True, right_index=True, how="left")
    df = df.merge(fred_daily_df, left_index=True, right_index=True, how="left")
    
    cols = feature_cols[ticker]
    # cols = ['returns'] #, 'PCT_change', 'weighted_tech_sentiment']
    df = df[cols]

    df = df.ffill()

    n_past = days_past[ticker]
    if n_past == 40:
        start_date = "2025-06-27"
    elif n_past == 20:
        start_date = "2025-07-26"
    elif n_past == 10:
        start_date = "2025-08-11"
    elif n_past == 60:
        start_date = "2025-05-29"

    df = df[df.index >= start_date]
    # df = df.dropna()

    preds = []
    for i in range(n_past, len(df)-1):  # t+1
        data = df.iloc[i-n_past:i, :]
        if i == n_past:
            print('Last day data:')
            print(data.iloc[-1])
        scaler = joblib.load(f'artifacts/best_scaler_{ticker}_4.joblib')
        # scaler = joblib.load(f"pkl/forecast_scaler_{ticker}_univ.joblib")
        scaled_data = scaler.transform(data)
        forecast_df = forecast_future(model, data, scaler, feature_columns=df.columns.to_list(), n_future=1, n_past=n_past)
        predicted_price = forecast_df["Predicted_returns"]
        preds.append(predicted_price.values[0])
        print(f"Day {i-n_past} predicted price: {predicted_price.values[0]}")

    print(preds)
    
    df_test = yf.download(ticker, start="2025-08-22", end="2025-10-20", interval="1d")['Close']
    df_test['returns'] = df_test.pct_change()
    df_test = df_test['returns'].values[1:]  # skip NaN
    df_test = df_test[:23]
    print("Test prices:", df_test)
    preds = preds[:23]
    print("Predicted prices:", preds)

    # compute MAE, RMSE, MAPE
    mae = mean_absolute_error(df_test, preds)
    rmse = np.sqrt(mean_squared_error(df_test, preds))
    mape = np.mean(np.abs(df_test - preds) / np.abs(df_test)) * 100

    print(f"Metrics for {ticker} - MAE: {mae}, RMSE: {rmse}, MAPE: {mape}%")

    dir_acc = np.mean(np.sign(df_test) == np.sign(preds)) * 100
    print(f"Directional Accuracy for {ticker}: {dir_acc}")

    metrics_results.append({
        'Ticker': ticker,
        'MAE': mae,
        'RMSE': rmse,
        'MAPE (%)': mape,
        'Directional Accuracy (%)': dir_acc
    })

    preds_results.append({
        'Ticker': ticker,
        'Actual Returns': df_test, "Sum of actual Returns": np.sum(df_test),
        'Predicted Returns': preds, "Sum of predicted Returns": np.sum(preds)
    })

    plt.plot(df_test, label='Actual Returns')
    plt.plot(preds, label='Predicted Returns', marker='*')
    # plot a red line at y=0
    plt.axhline(0, color='red', linestyle='--', linewidth=1)
    plt.xlabel('Days')
    plt.ylabel('Returns')
    plt.title(f'{ticker} Stock Returns Prediction')
    plt.legend()
    # plt.savefig(f'data/prediction_returns_4_{ticker}.png')
    # plt.show()

# metrics_df = pd.DataFrame(metrics_results)
# metrics_df.to_csv('metrics_iter3.csv', index=False)
preds_df = pd.DataFrame(preds_results)
preds_df.to_csv('preds_iter4.csv', index=False)