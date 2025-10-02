import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from feature_engineering import create_df
import yfinance as yf


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

df = load_transform_data("GOOG")


target_col = "Close"

# -------------------------
# 2. Feature Selection
# -------------------------
def select_features(df, target_col="Close", method="rf", top_k=5, corr_threshold=0.2):
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
    
    return selected + [target_col]  # always include target


# Choose selection method: "corr", "rf"
feature_columns = select_features(df, target_col="Close", method="rf", top_k=5, corr_threshold=0.1)
print("Auto-selected features:", feature_columns)
print(df[feature_columns].head())


# -------------------------
# 3. Prepare Data
# -------------------------
data = df[feature_columns].copy()

scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(data)

def create_sequences(data, n_past=60, n_future=1):
    X, y = [], []
    for i in range(n_past, len(data)-n_future+1):
        X.append(data[i-n_past:i, :])
        y.append(data[i+n_future-1, :])  # predict all features
    return np.array(X), np.array(y)

n_past = 10
n_future = 1

X, y = create_sequences(scaled_data, n_past, n_future)

# -------------------------
# 4. Build LSTM Model
# -------------------------
input_shape = (n_past, len(feature_columns))

model = Sequential()

model.add(LSTM(120, activation="tanh", input_shape=input_shape))
model.add(Dropout(0.2))
model.add(Dense(1))  # predict Close only
model.compile(optimizer="adam", loss="mean_squared_error")
model.summary()

early_stop = EarlyStopping(monitor="loss", patience=10, restore_best_weights=True)

# -------------------------
# 5. Train
# -------------------------
model.fit(
        X, y,
        epochs=100,  # capped at 100
        batch_size=32,
        validation_split=0.1,
        verbose=1,
        callbacks=[early_stop]
    )

# -------------------------
# 6. Forecast Function
# -------------------------
def forecast_future(model, data, scaler, feature_columns, n_future=30, n_past=60):
    last_sequence = data[feature_columns].values[-n_past:]
    last_sequence_scaled = scaler.transform(last_sequence)
    input_seq = last_sequence_scaled.reshape(1, n_past, len(feature_columns))
    
    forecast_scaled = []
    
    for _ in range(n_future):
        # Predict only Close (1 value)
        next_pred_scaled = model.predict(input_seq, verbose=0)
        next_close_scaled = next_pred_scaled[0][0]
        
        # Take the last row of the sequence (scaled features)
        next_row_scaled = input_seq[0, -1, :].copy()
        
        # Update Close column with the prediction
        next_row_scaled[feature_columns.index("Close")] = next_close_scaled
        
        # Save this row
        forecast_scaled.append(next_row_scaled)
        
        # Update input sequence
        input_seq = np.append(input_seq[:,1:,:], next_row_scaled.reshape(1,1,len(feature_columns)), axis=1)
    
    forecast_scaled = np.array(forecast_scaled)
    forecast = scaler.inverse_transform(forecast_scaled)
    forecast_df = pd.DataFrame(forecast, columns=feature_columns)
    return forecast, forecast_df

# -------------------------
# 7. Run Forecast
# -------------------------
forecast, forecast_df = forecast_future(
    model=model,
    data=df,
    scaler=scaler,
    feature_columns=feature_columns,
    n_future=10,
    n_past=n_past
)

forecast_close = forecast_df["Close"]


def forecast_pipeline(df):
    feature_columns = select_features(df, target_col="Close", method="rf", top_k=5, corr_threshold=0.1)
    print("Auto-selected features:", feature_columns)
    if len(feature_columns)>3 and 'weighted_sentiment' not in feature_columns:
        feature_columns = feature_columns + ['weighted_sentiment']
    data = df[feature_columns].copy()
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(data)
    X, y = create_sequences(scaled_data, n_past=60, n_future=1)
    model = Sequential()
    model.add(LSTM(120, activation="tanh", input_shape=(60, len(feature_columns))))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer="adam", loss="mean_squared_error")
    early_stop = EarlyStopping(monitor="loss", patience=10, restore_best_weights=True)
    model.fit(
        X, y,
        epochs=100,
        batch_size=32,
        validation_split=0.1,
        verbose=1,
        callbacks=[early_stop]
    )
    forecast, forecast_df = forecast_future(
        model=model,
        data=df,
        scaler=scaler,
        feature_columns=feature_columns,
        n_future=10,
        n_past=60
    )
    return forecast_df

# -------------------------
# 8. Plot
# -------------------------
if __name__ == "__main__":
    ticker = "MSFT"

    y_test = yf.download(ticker, start="2025-08-25", end="2025-08-29", interval="1d")['Close'].values

    df = load_transform_data(ticker)
    forecast_df = forecast_pipeline(df)
    forecast_close = forecast_df["Close"]

    df_no = df[["Close","Open"]].copy()
    forecast_df_no = forecast_pipeline(df_no)
    forecast_close_no = forecast_df_no["Close"]

    plt.figure(figsize=(10,5))
    plt.plot(df.index[-50:], df["Close"].values[-50:], label="History")

    future_dates = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=len(forecast_close), freq="D")
    plt.plot(future_dates, forecast_close, label="Forecast")
    plt.plot(future_dates, forecast_close_no, label="Forecast (No Features)", linestyle='--')
    plt.plot(future_dates[:len(y_test)], y_test, label="Actual")

    plt.legend()
    plt.show()