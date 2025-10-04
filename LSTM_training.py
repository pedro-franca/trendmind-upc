import numpy as np
import pandas as pd
import joblib
import yfinance as yf
import matplotlib.pyplot as plt
import duckdb
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from feature_engineering import create_df



# -------------------------
# 1. Load Data
# -------------------------
# Example: assume df has OHLCV columns
# Replace with your data loading step
def load_transform_data(ticker):
    df = create_df(ticker)
    df = df.ffill()
    df = df.dropna(axis=1, how='all')
    df.sort_index(inplace=True)
    return df

# -------------------------
# 2. Feature Selection
# -------------------------
def select_features(df, target_col="Close", method="rf", top_k=10, corr_threshold=0.2):
    """
    Select features automatically using different methods:
      - 'corr': correlation with target
      - 'rf'  : RandomForest importance
    """
    if method == "corr":
        corr = df.corr()[target_col].drop(target_col)
        selected = corr[abs(corr) > corr_threshold].index.tolist()
    
    elif method == "rf":
        X_fs = df.drop(columns=[target_col])
        y_fs = df[target_col]
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X_fs, y_fs)
        importances = pd.Series(rf.feature_importances_, index=X_fs.columns)
        selected = importances.sort_values(ascending=False).head(top_k).index.tolist()
    
    else:
        raise ValueError("method must be 'corr', 'rf'")
    
    if 'Close_lag1' not in selected:
        selected = selected + ['Close_lag1']
    
    return selected + [target_col]  # always include target


def create_sequences(data, feature_columns, n_past=60):
    """
    Build sequences to predict only tomorrow's Close (t+1)
    """
    X, y = [], []
    close_idx = feature_columns.index("Close")
    
    for i in range(n_past, len(data)-1):  # t+1
        X.append(data[i-n_past:i, :])
        y.append(data[i, close_idx])  # only t+1
    
    return np.array(X), np.array(y)



# -------------------------
# 6. Forecast Function
# -------------------------
def forecast_future(model, data, scaler, feature_columns, n_future=30, n_past=60):
    close_idx = feature_columns.index("Close")
    last_sequence = data[feature_columns].values[-n_past:]
    last_sequence_scaled = scaler.transform(last_sequence)
    input_seq = last_sequence_scaled.reshape(1, n_past, len(feature_columns))
    
    preds_scaled = []
    
    for _ in range(n_future):
        next_pred_scaled = model.predict(input_seq, verbose=0)[0,0]  # scalar
        preds_scaled.append(next_pred_scaled)
        
        # build next row
        next_row_scaled = input_seq[0, -1, :].copy()
        next_row_scaled[close_idx] = next_pred_scaled
        
        # roll input
        input_seq = np.append(input_seq[:,1:,:], next_row_scaled.reshape(1,1,len(feature_columns)), axis=1)
    
    # inverse transform to original scale
    preds_scaled_array = np.array(preds_scaled).reshape(-1,1)
    preds_full = np.repeat(preds_scaled_array, len(feature_columns), axis=1)
    preds = scaler.inverse_transform(preds_full)[:, close_idx]
    
    forecast_df = pd.DataFrame({"Close_t1": preds})
    return forecast_df

def forecast_pipeline(df):
    feature_columns = select_features(df, target_col="Close", method="rf", top_k=3, corr_threshold=0.8)
    print("Auto-selected features:", feature_columns)
    data = df[feature_columns].copy()
    print(data.columns)
    data = data.dropna()
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(data)
    print("Scaled data shape:", scaled_data.shape)
    print(np.isnan(scaled_data).sum(), np.isinf(scaled_data).sum())
    X, y = create_sequences(scaled_data, feature_columns, n_past=10)
    print("X shape:", X.shape, "y shape:", y.shape)
    model = Sequential()
    model.add(LSTM(120, activation="tanh", input_shape=(X.shape[1], X.shape[2])))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer=Adam(learning_rate=0.0001), loss="mse")
    early_stop = EarlyStopping(monitor="loss", patience=10, restore_best_weights=True)
    model.fit(
        X, y,
        epochs=100,
        batch_size=32,
        validation_split=0.1,
        verbose=1,
        callbacks=[early_stop]
    )
    forecast_df = forecast_future(
        model=model,
        data=df,
        scaler=scaler,
        feature_columns=feature_columns,
        n_future=1,
        n_past=10
    )
    return model, forecast_df

def evaluate_models(true_prices, pred_model1, pred_model2):
    results = {}
    
    # Converte para arrays 1D
    true_prices = np.array(true_prices).ravel()
    pred_model1 = np.array(pred_model1).ravel()
    pred_model2 = np.array(pred_model2).ravel()
    
    # Garante mesmo tamanho (corta se necessário)
    min_len = min(len(true_prices), len(pred_model1), len(pred_model2))
    true_prices = true_prices[:min_len]
    pred_model1 = pred_model1[:min_len]
    pred_model2 = pred_model2[:min_len]

    for name, preds in {"Model1": pred_model1, "Model2": pred_model2}.items():
        # Métricas de regressão
        mae = mean_absolute_error(true_prices, preds)
        rmse = np.sqrt(mean_squared_error(true_prices, preds))
        mape = np.mean((true_prices - preds) / true_prices) * 100

        # Direcionalidade
      #  true_dir = np.sign(np.diff(true_prices))
      #  pred_dir = np.sign(np.diff(preds))
      #  dir_acc = np.mean(true_dir == pred_dir) * 100

        # Backtest simples
     #   returns = np.sign(np.diff(preds)) * np.diff(true_prices)
      #  cum_return = np.sum(returns)

        results[name] = {
            "MAE": mae,
            "RMSE": rmse,
            "MAPE (%)": mape
        }

    return pd.DataFrame(results)


# -------------------------
# 8. Plot
# -------------------------
if __name__ == "__main__":
    data = pd.DataFrame()
    for ticker in ["AAPL", "GOOG", "MSFT", "TSM", "AVGO", "META", "NVDA", "ORCL", "TCEHY"]:
        print(f"Processing {ticker}...")

        y_test = yf.download(ticker, start="2025-08-25", end="2025-09-29", interval="1d")['Close'].values

        df = load_transform_data(ticker)
        # drop columns High, Low, Volume
        #cols = [c for c in ['Close', 'Close_lag1', 'PCT_change' ,'SMA_5', 'ev_x', 'EMA_10', 'weighted_sentiment'] if c in df_all.columns]
        model, forecast_df = forecast_pipeline(df)
        forecast_close_t1 = forecast_df["Close_t1"]

       # joblib.dump(model, f"forecast_{ticker}.pkl")

        df_no = df[["Close","Close_lag1"]].copy()
        model, forecast_df_no = forecast_pipeline(df_no)
        forecast_close_no_t1 = forecast_df_no["Close_t1"]


        plt.figure(figsize=(12,6))
        plt.plot(df.index[-50:], df["Close"].values[-50:], label="History", linewidth=2)

        # Generate future dates
        future_dates = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), 
                                    periods=len(forecast_df), freq="D")

        # Plot both forecasts
        plt.plot(future_dates, forecast_df["Close_t1"], label="Forecast (t+1)", marker='*')

        plt.plot(future_dates, forecast_df_no["Close_t1"], label="Forecast NO FE (t+1)", marker='+')

        # Plot actual test prices if available
        plt.plot(future_dates[:len(y_test)], y_test, label="Actual", markerfacecolor='none', alpha=0.7, marker='o')

        plt.title(f"{ticker} Price Forecast t+1")
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
       # plt.savefig(f'prediction_{ticker}.png')
        plt.show()


        results = evaluate_models(y_test, forecast_close_t1, forecast_close_no_t1)
        # results.to_csv(f"results_{ticker}.csv")
        print(results)

        forecast_df.columns = [f'Close_t1_{ticker}']
        data = pd.concat([data,forecast_df], axis=1)

   # output_path = "./data/predictions.duckdb"
   # con = duckdb.connect(database=output_path, read_only=False)
   # con.execute(f"CREATE OR REPLACE TABLE predictions AS SELECT * FROM data")
   # con.close()
   # print(f"✅ Data exported to {output_path}")