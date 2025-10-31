import numpy as np
import pandas as pd
import joblib
from sklearn.feature_selection import mutual_info_regression
import yfinance as yf
import matplotlib.pyplot as plt
import duckdb
# from pmdarima import auto_arima
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Conv1D, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers, regularizers
from feature_engineering import create_df



# -------------------------
# 1. Load Data
# -------------------------
def load_transform_data(ticker):
    df = create_df(ticker)
    df = df.ffill()
    df = df.dropna(axis=1, how='all')
    df.sort_index(inplace=True)
    df.index = pd.to_datetime(df.index, errors='coerce')
    return df



def create_sequences(data, feature_columns, n_past=60):
    """
    Build sequences to predict only tomorrow's Close (t+1)
    """
    X, y = [], []
    close_idx = feature_columns.index("returns")
    
    for i in range(n_past, len(data)-1):  # t+1
        X.append(data[i-n_past:i, :])
        y.append(data[i, [close_idx]])  # only t+1

    return np.array(X), np.array(y)



# -------------------------
# 6. Forecast Function
# -------------------------
def forecast_future(model, data, scaler, feature_columns, n_future=7, n_past=10):
    """
    Predicts the next `n_future` days of the target column (e.g., 'Close') using
    a trained multivariate LSTM that outputs multiple steps ahead.
    """
    close_idx = feature_columns.index("returns")
    
    # Get the last n_past days of all features
    last_sequence = data[feature_columns].values[-n_past:]
    last_sequence_scaled = scaler.transform(last_sequence)
    
    print(last_sequence)
    # Reshape for model input: (1, timesteps, features)
    input_seq = last_sequence_scaled.reshape(1, n_past, len(feature_columns))
    
    # Predict next n_future days (all at once)
    preds_scaled = model.predict(input_seq, verbose=0)[0]  # shape: (n_future,)
    
    # Expand to full feature shape so we can inverse transform
    preds_scaled_array = np.array(preds_scaled).reshape(-1, 1)
    preds_full = np.repeat(preds_scaled_array, len(feature_columns), axis=1)
    preds = scaler.inverse_transform(preds_full)[:, close_idx]
    
    # Return as DataFrame
    forecast_df = pd.DataFrame({
        "Day": np.arange(1, len(preds) + 1),
        "Predicted_returns": preds
    })
    
    return forecast_df

def forecast_pipeline(df):
    feature_columns = df.columns.tolist()
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
    return scaler, model, forecast_df

def evaluate_models(true_prices, pred_model1):
    results = {}
    
    # Converte para arrays 1D
    true_prices = np.array(true_prices).ravel()
    pred_model1 = np.array(pred_model1).ravel()
    
    # Garante mesmo tamanho (corta se necessário)
    min_len = min(len(true_prices), len(pred_model1))
    true_prices = true_prices[:min_len]
    pred_model1 = pred_model1[:min_len]


    for name, preds in {"Model1": pred_model1}.items():
        # Métricas de regressão
        mae = mean_absolute_error(true_prices, preds)
        rmse = np.sqrt(mean_squared_error(true_prices, preds))
        mape = np.mean(np.abs(true_prices - preds) / true_prices) * 100

        # # Direcionalidade de retornos
        # direction_true = np.sign(np.diff(true_prices))
        # direction_pred = np.sign(np.diff(preds))
        # dir_acc = np.mean(direction_true == direction_pred) * 100

        # # Backtest simples
        # Close = np.sign(np.diff(preds)) * np.diff(true_prices)
        # cum_return = np.sum(Close)

        results[name] = {
            "MAE": mae,
            "RMSE": rmse,
            "MAPE (%)": mape
            # "Direcionalidade (%)": dir_acc,
            # "Retorno Acumulado": cum_return
        }

    return pd.DataFrame(results)


# -------------------------
# 8. Plot
# -------------------------
if __name__ == "__main__":
    data = pd.DataFrame()
    for ticker in ["AAPL", "GOOG", "MSFT", "TSM", "AVGO", "META", "NVDA", "ORCL", "TCEHY"]:
        print(f"Processing {ticker}...")

        # y_test = yf.download(ticker, start="2025-08-22", end="2025-08-26", interval="1d")['Close']
        # y_test['returns'] = y_test.pct_change()
        # y_test = y_test['returns'].values[1:]  # skip NaN
        # # y_t = yf.download(ticker, start="2025-08-22", end="2025-08-23", interval="1d")["Close"].values
        # print("Test prices:", y_test)

        df = load_transform_data(ticker)
        cols = [c for c in ['returns'] if c in df.columns]
        df1 = df[cols]
        # print(df.index[-1])
        scaler, model, forecast_df = forecast_pipeline(df1)
        forecast_returns_t1 = forecast_df["Predicted_returns"]
        print("Forecasted prices:", forecast_returns_t1.values)

        joblib.dump(model, f"./pkl/forecast_returns_{ticker}_univ.pkl")
        joblib.dump(scaler, f"./pkl/forecast_scaler_{ticker}_univ.joblib")


        # plt.figure(figsize=(12,6))
        # plt.plot(df.index[-50:], df["returns"].values[-50:], label="History", linewidth=2)

        # # Generate future dates
        # future_dates = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), 
        #                             periods=len(forecast_df), freq="D")

        # # Plot both forecasts
        # plt.plot(future_dates, forecast_df["Predicted_returns"], label="Forecast", marker='*')

        # plt.plot(future_dates, forecast_df_no["Close_t1"], label="Forecast NO FE (t+1)", marker='+')

        # Plot actual test prices if available
        # plt.plot(future_dates[:len(y_test)], y_test, label="Actual", markerfacecolor='none', alpha=0.7, marker='o')

        # plt.title(f"{ticker} Returns Forecast t+1")
        # plt.xlabel("Date")
        # plt.ylabel("Returns")
        # plt.legend()
        # plt.grid(True, alpha=0.3)
        # plt.tight_layout()
        # plt.savefig(f'prediction_Close_{ticker}.png')
        # plt.show()


        # results = evaluate_models(y_test, forecast_close_t1) # , forecast_close_no_t1)
        # # results.to_csv(f"results_{ticker}.csv")
        # results.rename(columns={'Model1': ticker},inplace=True)
        # print(results)



    #     data = pd.concat([data,results], axis=1)
    # # data.to_csv('RF_final_model.csv')
    # print(data)

   # output_path = "./data/predictions.duckdb"
   # con = duckdb.connect(database=output_path, read_only=False)
   # con.execute(f"CREATE OR REPLACE TABLE predictions AS SELECT * FROM data")
   # con.close()
   # print(f"✅ Data exported to {output_path}")