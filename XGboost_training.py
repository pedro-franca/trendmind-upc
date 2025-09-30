import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from feature_engineering import create_df
from xgboost import XGBRegressor
from feature_selection import select_features



# --- Load and transform data ---
def load_transform_data(ticker, target_col='Close'):
    df = create_df(ticker)
    # df = df[['Close','Open','High','Low','Volume','weighted_sentiment']]
    df = select_features(df, target_col=target_col)
    df = df.ffill()
    df.sort_index(inplace=True)
    print(df.head())  # Optional preview
    return df

# -------------------------------
# Create lag features
# -------------------------------
def create_lag_features(df, target_col="Close", lags=5):
    df = df.copy()
    for lag in range(1, lags + 1):
        df[f"{target_col}_lag{lag}"] = df[target_col].shift(lag)
    return df

# -------------------------------
# Train and forecast next 7 days
# -------------------------------
def train_xgb_forecast(data, target_col="Close", forecast_horizon=7, lags=5):
    # --- Create lag features ---
    df = create_lag_features(data, target_col, lags=lags)

    # --- Scale data ---
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df)
    df_scaled = pd.DataFrame(scaled, columns=df.columns, index=df.index)

    # --- Split features/target ---
    X = df_scaled.drop(columns=[target_col])
    y = df_scaled[target_col]

    # --- Train model ---
    model = XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    model.fit(X, y)

    # --- Start rolling forecast ---
    last_known = df_scaled.iloc[[-1]].copy()  # last row
    future_preds = []

    for step in range(forecast_horizon):
        # Predict next value
        X_input = last_known.drop(columns=[target_col])
        y_pred = model.predict(X_input)[0]
        future_preds.append(y_pred)

        # Build next row with shifted lags
        new_row = last_known.copy()
        new_row[target_col] = y_pred
        for lag in range(1, lags + 1):
            if lag == 1:
                new_row[f"{target_col}_lag{lag}"] = last_known[target_col].values
            else:
                new_row[f"{target_col}_lag{lag}"] = last_known[f"{target_col}_lag{lag-1}"].values

        last_known = new_row

    # --- Inverse scale predictions ---
    dummy = np.zeros((forecast_horizon, df.shape[1]))
    target_idx = df.columns.get_loc(target_col)
    for i, p in enumerate(future_preds):
        dummy[i, target_idx] = p
    inv_preds = scaler.inverse_transform(dummy)[:, target_idx]

    # --- Print results ---
    print(f"\nNext {forecast_horizon} day predictions:")
    for i, val in enumerate(inv_preds, 1):
        print(f"Day +{i}: {val:.2f}")

    return model, scaler, inv_preds

# -------------------------------
# Example usage
# -------------------------------
if __name__ == "__main__":
    ticker = "AAPL"
    df = load_transform_data(ticker, target_col='Close')
    model, scaler, predictions = train_xgb_forecast(df, target_col="Close", forecast_horizon=7)
