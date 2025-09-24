# LSTM_training.py
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from feature_selection import create_df, select_features

# --- Load and transform data ---
def load_transform_data(ticker):
    df = create_df(ticker)
    df = select_features(df, target_col='Close', threshold=0.6)
    df = df.ffill().dropna()
    return df

# --- Preprocess data for LSTM ---
def prepare_lstm_data(df, target_col='Close', sequence_length=60):
    # Only use Close for prediction; scale to [0, 1]
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(df[[target_col]])

    X, y = [], []
    for i in range(sequence_length, len(data_scaled)):
        X.append(data_scaled[i-sequence_length:i])
        y.append(data_scaled[i])
    X, y = np.array(X), np.array(y)

    return X, y, scaler

# --- Build a simple LSTM model ---
def build_lstm_model(input_shape):
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape),
        LSTM(50),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# --- Train the model ---
def train_lstm(ticker):
    df = load_transform_data(ticker)
    print(df.head())  # Optional preview
    if 'Close' not in df.columns:
        raise ValueError("Expected 'Close' column in data.")

    X, y, scaler = prepare_lstm_data(df, target_col='Close')

    model = build_lstm_model((X.shape[1], X.shape[2]))
    model.fit(X, y, epochs=20, batch_size=32, verbose=1)

    return model, scaler


# --- Train/test split + evaluation ---
def train_test_lstm(ticker, sequence_length=60, test_size=0.2):
    # Load + preprocess
    df = load_transform_data(ticker)
    if 'Close' not in df.columns:
        raise ValueError("Expected 'Close' column in data.")
    
    X, y, scaler = prepare_lstm_data(df, target_col='Close', sequence_length=sequence_length)

    # Train/test split
    split_idx = int(len(X) * (1 - test_size))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # Build + train
    model = build_lstm_model((X.shape[1], X.shape[2]))
    model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=1)

    # Predictions
    y_pred = model.predict(X_test)

    # Inverse transform to original scale
    y_test_rescaled = scaler.inverse_transform(y_test.reshape(-1, 1))
    y_pred_rescaled = scaler.inverse_transform(y_pred)

    # Metrics
    mse = mean_squared_error(y_test_rescaled, y_pred_rescaled)
    mae = mean_absolute_error(y_test_rescaled, y_pred_rescaled)
    r2 = r2_score(y_test_rescaled, y_pred_rescaled)

    print(f"MSE: {mse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R²: {r2:.4f}")

    # Plot predictions vs actual
    plt.figure(figsize=(10,5))
    plt.plot(y_test_rescaled, label="Actual")
    plt.plot(y_pred_rescaled, label="Predicted")
    plt.legend()
    plt.title(f"{ticker} - LSTM Predictions vs Actual")
    plt.show()

    return model, scaler, (mse, mae, r2)

# --- Main ---
if __name__ == "__main__":
    ticker = "ORCL"
    try:
        model, scaler, metrics = train_test_lstm(ticker)
        print(f"✅ Done training & evaluating for {ticker}")
    except Exception as e:
        print(f"❌ Error: {e}")

