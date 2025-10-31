import numpy as np
import pandas as pd
import joblib
import yfinance as yf
import matplotlib.pyplot as plt
import duckdb
#from pmdarima import auto_arima
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Conv1D, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from feature_engineering import create_df
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.losses import Huber
from sklearn.feature_selection import mutual_info_regression
import os

# Control TF threading on CPU (tune if you feel UI lag)
tf.config.threading.set_intra_op_parallelism_threads(4)
tf.config.threading.set_inter_op_parallelism_threads(2)
os.environ["OMP_NUM_THREADS"] = "4"


def combined_loss(y_true, y_pred):
    huber = Huber(delta=1.0)
    huber_loss = huber(y_true, y_pred)
    mae_loss = tf.reduce_mean(tf.abs(y_true - y_pred))
    return 0.7 * huber_loss + 0.3 * mae_loss

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





# -------------------------
# 2. Feature Selection
# -------------------------
def select_features(
    df: pd.DataFrame,
    target_col: str = "returns",
    method: str = "corr",           # "manual" | "corr" | "rf" | "mi"
    top_k: int = 10,
    must_include = ['weighted_tech_sentiment'],   # columns we always keep if present
    fs_train_frac: float = 0.8,              # compute FS on train part only (avoid leakage)
    random_state: int = 42,
) -> list[str]:
    """
    Return an ordered list of feature names to use (features + target at the end).
    - method:
        "manual":   caller handles columns externally; just returns must_include + target if present
        "corr":     absolute Pearson correlation with target (ranked desc)
        "rf":       co feature_importances_ (ranked desc)
        "mi":       Mutual information with target (ranked desc)  <-- NEW
    - top_k: number of features to pick (ignored for 'manual')
    - fs_train_frac: fraction of data used for FS (time-ordered)
    """
    # if must_include is None:
    # must_include = ['weighted_tech_sentiment']

    if target_col not in df.columns:
        raise ValueError(f"target_col='{target_col}' not found in df.columns")

    # Always work on a clean copy
    df_ = df.dropna().copy()

    # Time-aware split for FS to reduce leakage
    n = len(df_)
    split = max(1, int(n * fs_train_frac))
    df_train = df_.iloc[:split]

    # Build candidate feature set (exclude target)
    candidate_feats = [c for c in df_train.columns if c != target_col]

    # Manual mode: just return (must_include ∩ columns) + target
    if method == "manual":
        cols = [c for c in must_include if c in df.columns]
        if target_col not in cols:
            cols = cols + [target_col]
        return cols

    # Compute scores by method
    if method == "corr":
        corr = df_train[candidate_feats + [target_col]].corr()[target_col].dropna().drop(labels=[target_col], errors='ignore')
        scores = corr.abs().sort_values(ascending=False)

    elif method == "rf":
        X = df_train[candidate_feats].values
        y = df_train[target_col].values
        rf = RandomForestRegressor(n_estimators=300, random_state=random_state, n_jobs=-1)
        rf.fit(X, y)
        scores = pd.Series(rf.feature_importances_, index=candidate_feats).sort_values(ascending=False)

    elif method == "mi":
        # Mutual information handles non-linear relations; robust for mixed scales
        X = df_train[candidate_feats].values
        y = df_train[target_col].values
        mi = mutual_info_regression(X, y, random_state=random_state)
        scores = pd.Series(mi, index=candidate_feats).sort_values(ascending=False)

    else:
        raise ValueError("method must be one of: 'manual', 'corr', 'rf', 'mi'")

    # Take the top_k ranked features
    ranked = list(scores.index[:top_k])

    # Ensure must-have columns are included if present (dedupe while preserving order)
    prefixed = [c for c in must_include if c in df.columns]
    chosen = prefixed + [c for c in ranked if c not in prefixed]

    # Append target at the end
    if target_col not in chosen:
        chosen.append(target_col)

    return chosen



def build_featured_df(
    df: pd.DataFrame,
    method: str,
    top_k: int,
    manual_cols: list[str],
    target_col: str = "returns",
    fs_train_frac: float = 0.8,
) -> pd.DataFrame:
    """
    Returns a reduced df using the selected feature method.
    - In 'manual' mode, uses exactly manual_cols (if present) + target.
    - For other methods, selects top_k according to the method.
    """
    cols = select_features(
        df=df,
        target_col=target_col,
        method=method,
        top_k=top_k,
        must_include=manual_cols,   # we tend to always keep sentiment, etc.
        fs_train_frac=fs_train_frac,
    )
    # Keep only the intersecting columns (robust if some are missing)
    cols = [c for c in cols if c in df.columns]
    return df[cols]



# -------------------------------------------------------------------------------------------------------------------------




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
    
    # print(last_sequence)
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



#############################################################################################################S


def forecast_pipeline(df,
                      # ---- Tunables (keep defaults for now; we can tune later) ----
                      n_past: int = 20,       # lookback window used by create_sequences
                      units: int = 64,        # LSTM hidden size
                      depth: int = 2,         # number of stacked LSTM layers
                      dropout: float = 0.1,   # dropout + recurrent_dropout
                      lr: float = 5e-4,       # Adam learning rate
                      l2_reg: float = 1e-5,   # L2 regularization (safe "weight decay")
                      batch_size: int = 64,
                      max_epochs: int = 150,
                      patience: int = 8,
                      min_lr: float = 1e-5,
                      first_activation: str = "tanh"   # <— NEW: "relu" or "selu" or "tanh"
                      ):
    """
    Improved LSTM training pipeline (REGRESSION) for next-day prediction.

    Keeps your existing data flow (MinMax scaling on full data, create_sequences, forecast_future),
    but upgrades the model and training loop for better stability and generalization.

    Notes:
      - Still uses validation_split=0.1 (like your original) to minimize pipeline changes.
      - We can switch to a proper time-based split/walk-forward in a later step.
      - Compatible with Python 3.13 (no tensorflow-addons).
    """
    ############################################### Test_2 NEw FUNCTION ###############################################
    def combined_loss(y_true, y_pred):
        huber = Huber(delta=1.0)
        huber_loss = huber(y_true, y_pred)
        mae_loss = tf.reduce_mean(tf.abs(y_true - y_pred))
        return 0.7 * huber_loss + 0.3 * mae_loss
    ############################################### Test_2 NEw FUNCTION ###############################################


    # For reproducibility (best effort)
    np.random.seed(42)
    tf.random.set_seed(42)

    # -------------------------------------------------------------------------
    # 1) Select and scale features (same behavior as your original function)
    # -------------------------------------------------------------------------
    # feature_columns = select_features(df, target_col="Close", method="rf", top_k=4, corr_threshold=0.8)  #  <- For Feature Selection
    feature_columns = df.columns.tolist()   # <- Este o el anterior si no se quiere feature selection
    
    print("Auto-selected features:", feature_columns)
    data = df[feature_columns].copy()
    print("Columns used:", list(data.columns))

    # Drop rows with NaNs (you already did this; keep same behavior)
    data = data.dropna()

    # MinMax fit on entire dataset (as you already had).
    # Later, we may move scaler fitting to train-only to be stricter for time-series.
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data.values.astype("float32"))   # <- use .values


    

    print("Scaled data shape:", scaled_data.shape)
    print("NaNs in scaled:", np.isnan(scaled_data).sum(), "Infs in scaled:", np.isinf(scaled_data).sum())

    # -------------------------------------------------------------------------
    # 2) Build sequences
    #    IMPORTANT: n_past == lookback. You were hardcoding n_past=10; now it's a parameter.
    # -------------------------------------------------------------------------
    X, y = create_sequences(scaled_data, feature_columns, n_past=n_past)
    # X shape -> (samples, lookback, n_features), y shape -> (samples, 1) or (samples,)
    print("X shape:", X.shape, "y shape:", y.shape)

    # Guard: ensure y is 2D for Keras metrics consistency
    if y.ndim == 1:
        y = y.reshape(-1, 1)

    n_samples, lookback, n_features = X.shape

    # -------------------------------------------------------------------------
    # 3) Build improved LSTM (no addons; Python 3.13 friendly)
    #    - Huber loss (robust to outliers)
    #    - LayerNormalization between LSTM blocks
    #    - dropout + recurrent_dropout
    #    - L2 regularization (kernel/recurrent/bias)
    #    - Adam optimizer with gradient clipping
    # -------------------------------------------------------------------------
    inputs = keras.Input(shape=(lookback, n_features))
    x = inputs

    # choose initializer for the first layer based on activation
    if first_activation.lower() == "relu":
        first_init = "he_normal"
    elif first_activation.lower() == "selu":
        first_init = "lecun_normal"
    else:
        first_activation = "tanh"  # fallback
        first_init = "glorot_uniform"

    for i in range(depth):
        act = first_activation if i == 0 else "tanh"   # only the first layer tries relu/selu
        init = first_init       if i == 0 else "glorot_uniform"

        x = layers.LSTM(
            units,
            activation=act,                           # <— activation added
            return_sequences=(i < depth - 1),
            dropout=dropout,
            recurrent_dropout=dropout,
            kernel_regularizer=regularizers.l2(l2_reg),
            recurrent_regularizer=regularizers.l2(l2_reg),
            bias_regularizer=regularizers.l2(l2_reg),
            kernel_initializer=init                   # <— match activation for stability
        )(x)
        x = layers.LayerNormalization()(x)


    # Dropout head
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(
        1,
        name="y",
        kernel_regularizer=regularizers.l2(l2_reg),
        bias_regularizer=regularizers.l2(l2_reg),
    )(x)

    model = keras.Model(inputs, outputs)

    # Adam with gradient clipping (prevents exploding gradients)
    optimizer = keras.optimizers.Adam(learning_rate=lr, clipnorm=1.0)

    model.compile(
        optimizer=optimizer,
        loss=combined_loss,        # better than MSE for outliers
        metrics=[
            keras.metrics.MeanAbsoluteError(name="mae"),
            keras.metrics.RootMeanSquaredError(name="rmse"),
        ],
    )



    # -------------------------------------------------------------------------
    # 4) Callbacks: EarlyStopping + ReduceLROnPlateau + ModelCheckpoint
    # -------------------------------------------------------------------------
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=patience,
            restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=max(patience // 3, 4),
            min_lr=min_lr
        ),
        keras.callbacks.ModelCheckpoint(
            filepath="pkl/best_lstm.keras",    # SavedModel format; reload in app if needed
            monitor="val_loss",
            save_best_only=True
        ),
    ]

    # -------------------------------------------------------------------------
    # 5) Train
    #    Keeping your original validation strategy: validation_split=0.1.
    #    (Later we will swap to time-aware splits / walk-forward.)
    # -------------------------------------------------------------------------
    history = model.fit(
        X, y,
        epochs=max_epochs,
        batch_size=batch_size,
        validation_split=0.10,
        verbose=1,
        callbacks=callbacks
    )

    # Print last validation metrics (if available)
    if "val_mae" in history.history and "val_rmse" in history.history:
        print(f"✅ Last val metrics -> MAE={history.history['val_mae'][-1]:.4f}  RMSE={history.history['val_rmse'][-1]:.4f}")

    # # -------------------------------------------------------------------------
    # # 6) Save final model (best checkpoint already saved). This is optional redundancy.
    # # -------------------------------------------------------------------------
    # try:
    #     model.save("pkl/final_lstm.keras")
    # except Exception as e:
    #     print("Model save warning:", e)

    # -------------------------------------------------------------------------
    # 7) Forecast next step using your existing helper (kept unchanged)
    #    IMPORTANT: we pass the same n_past the model trained with.
    # -------------------------------------------------------------------------
    forecast_df = forecast_future(
        model=model,
        data=df,
        scaler=scaler,
        feature_columns=feature_columns,
        n_future=1,
        n_past=n_past
    )

    return scaler, feature_columns, model, forecast_df



# def evaluate_models(true_prices, pred_model1): #, pred_model2):
#     results = {}
    
#     # Converte para arrays 1D
#     true_prices = np.array(true_prices).ravel()
#     pred_model1 = np.array(pred_model1).ravel()
#     #pred_model2 = np.array(pred_model2).ravel()
    
#     # Garante mesmo tamanho (corta se necessário)
#     min_len = min(len(true_prices), len(pred_model1)) #, len(pred_model2))
#     true_prices = true_prices[:min_len]
#     pred_model1 = pred_model1[:min_len]
#    # pred_model2 = pred_model2[:min_len]

#     for name, preds in {"Model1": pred_model1}.items(): # , "Model2": pred_model2}.items():
#         # Métricas de regressão
#         mae = mean_absolute_error(true_prices, preds)
#         rmse = np.sqrt(mean_squared_error(true_prices, preds))
#         mape = np.mean(np.abs(true_prices - preds) / true_prices) * 100

#         # Direcionalidade
#       #  true_dir = np.sign(np.diff(true_prices))
#       #  pred_dir = np.sign(np.diff(preds))
#       #  dir_acc = np.mean(true_dir == pred_dir) * 100

#         # Backtest simples
#      #   Close = np.sign(np.diff(preds)) * np.diff(true_prices)
#       #  cum_return = np.sum(Close)

#         results[name] = {
#             "MAE": mae,
#             "RMSE": rmse,
#             "MAPE (%)": mape
#         }

#     return pd.DataFrame(results)


def score_once(df1, y_test, n_past, first_activation):
    
    # Train + forecast 1 step using your pipeline with given params
    scaler, feature_columns, model, forecast_df = forecast_pipeline(df1, n_past=n_past, first_activation=first_activation)
    y_pred = forecast_df["Predicted_returns"].values[:len(y_test)]

    # Metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    # Directional accuracy for t+1: sign of prediction vs actual
    dir_acc = float(np.sign(y_pred[0]) == np.sign(y_test[0])) if len(y_pred) > 0 and len(y_test) > 0 else np.nan

    return {"MAE": mae, "RMSE": rmse, "DIR_ACC": dir_acc}







# -------------------------
# 8. Plot
# -------------------------


def get_test_returns(ticker: str, start="2025-08-22", end="2025-10-20", interval="1d"):
    """Robust yfinance fetch → next-day returns array."""
    df = yf.download(
        ticker,
        start=start,
        end=end,
        interval=interval,
        auto_adjust=False,   # keep both Close and Adj Close
        progress=False,
    )
    if df is None or df.empty:
        raise ValueError(f"No data from yfinance for {ticker}")

    close_col = "Close" if "Close" in df.columns else ("Adj Close" if "Adj Close" in df.columns else None)
    if close_col is None:
        raise ValueError(f"'Close' not found for {ticker}. Columns={list(df.columns)}")

    return df[close_col].pct_change().dropna().values  # returns array




# ===========================
# 9. Grid Runner (10 / 20 / 40 lookbacks × tanh / relu / selu activations)
# ===========================

if __name__ == "__main__":
    import os
    os.makedirs("artifacts", exist_ok=True)



    # --- PERMUTATION GRID CONFIG --- TICKERS = ["AAPL", "GOOG", "MSFT", "TSM", "AVGO", "META", "NVDA", "ORCL", "TCEHY"]
    TICKERS = ["AAPL", "GOOG", "MSFT", "TSM", "AVGO", "META", "NVDA", "ORCL", "TCEHY"]
    GRID_N_PAST = [10, 20, 40, 60]       # ← now includes 10
    GRID_ACT = ["tanh", "relu", "selu"]
    SAVE_PLOTS_BEST = True          # change to True to save only best plot per ticker
    N_FUTURE = 1                     # t+1 horizon
    GRID_FS_METHODS = ["manual", "corr", "rf", "mi"]   # dataset method
    GRID_TOPK = [10]                          # ignored for 'manual'
    MANUAL_COLS = ['returns', 'PCT_change', 'weighted_tech_sentiment']




    def score_once(df1, n_past, first_activation, ticker=None):
        import feature_engineering as fe
        """Train + forecast with given params; return MAE, RMSE, directional accuracy."""
        scaler, feature_columns, model, forecast_df = forecast_pipeline(df1, n_past=n_past, first_activation=first_activation)
        
        df = yf.download(ticker, start="2025-02-25", end="2025-10-20", interval="1d")
        df = df.droplevel('Ticker', axis=1)
        dp_df = fe.desperate_vars(df)
        df_tech = fe.join_tch_vars(df)
        df["returns"] = df["Close"].pct_change()

        news_df = fe.lt_news_data(ticker)
        tech_news_df = fe.lt_tech_news()
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
        
        cols = feature_columns
        df = df[cols]

        df = df.ffill()

        if n_past == 40:
            start_date = "2025-06-27"
        elif n_past == 20:
            start_date = "2025-07-26"
        elif n_past == 10:
            start_date = "2025-08-11"
        elif n_past == 60:
            start_date = "2025-05-29"

        df = df[df.index >= start_date]

        preds = []
        for i in range(n_past, len(df)-1):  # t+1
            data = df.iloc[i-n_past:i, :]
            if i == n_past:
                print('Last day data:')
                print(data.iloc[-1])
            scaled_data = scaler.transform(data)
            forecast_df = forecast_future(model, data, scaler, feature_columns=df.columns.to_list(), n_future=1, n_past=n_past)
            predicted_price = forecast_df["Predicted_returns"]
            preds.append(predicted_price.values[0])
            print(f"Day {i-n_past} predicted price: {predicted_price.values[0]}")

        print(preds)
        
        df_test = yf.download(ticker, start="2025-08-22", end="2025-10-20", interval="1d")['Close']
        df_test['returns'] = df_test.pct_change()
        df_test = df_test['returns'].values[1:]  # skip NaN
        df_test = df_test[:len(preds)]
        print("Test prices:", df_test)

        # compute MAE, RMSE, MAPE
        mae = mean_absolute_error(df_test, preds)
        rmse = np.sqrt(mean_squared_error(df_test, preds))
        mape = np.mean(np.abs(df_test - preds) / np.abs(df_test)) * 100
        dir_acc = np.mean(np.sign(df_test) == np.sign(preds)) * 100

        return scaler, feature_columns, model, {"MAE": mae, "RMSE": rmse, "MAPE": mape, "DIR_ACC": dir_acc}, df_test, preds

    rows = []

    for ticker in TICKERS:
        print(f"\n===== Processing {ticker} =====")

        df = load_transform_data(ticker)
        # quick guard
        if "returns" not in df.columns:
            print(f"[WARN] No 'returns' in {ticker}; skipping.")
            continue

        best_rec, best_plot_payload = None, None

        for fs_method in GRID_FS_METHODS:
            for topk in GRID_TOPK if fs_method != "manual" else [len(MANUAL_COLS)]:
                # Build the dataset for this run
                df_run = build_featured_df(
                    df=df,
                    method=fs_method,
                    top_k=topk,
                    manual_cols=MANUAL_COLS,
                    target_col="returns",
                    fs_train_frac=0.8,
                )

                # safety
                if df_run.shape[1] < 2:  # target only → skip
                    print(f"[SKIP] {ticker} ({fs_method}, topk={topk}) -> not enough features")
                    continue

                for npast in GRID_N_PAST:
                    for act in GRID_ACT:
                        print(f"-> {ticker} | fs={fs_method} topk={topk} | n_past={npast} act={act}")
                        try:
                            scaler, feature_columns, model, res, df_test, preds = score_once(df_run, n_past=npast, first_activation=act, ticker=ticker)
                            row = {
                                "ticker": ticker,
                                "fs_method": fs_method,
                                "top_k": topk,
                                "n_past": npast,
                                "activation": act,
                                "DIR_ACC": res["DIR_ACC"],
                                "MAE": res["MAE"],
                                "RMSE": res["RMSE"],
                                "MAPE": res["MAPE"],
                                # optionally: number of features used in this run
                                "n_features": df_run.shape[1]-1,   # excluding target
                                "features": feature_columns
                            }
                            rows.append(row)

                            # best picking logic (same as you had)
                            if best_rec is None:
                                best_rec = row; best_plot_payload = (df_run, preds, df_test, npast, act)
                                best_model = model
                                model.save(f"artifacts/best_model_{ticker}_4.keras")
                            else:
                                better = (
                                    (row["DIR_ACC"] > best_rec["DIR_ACC"]) or
                                    (row["DIR_ACC"] == best_rec["DIR_ACC"] and row["RMSE"] < best_rec["RMSE"]) or
                                    (row["DIR_ACC"] == best_rec["DIR_ACC"] and row["RMSE"] == best_rec["RMSE"] and row["MAE"] < best_rec["MAE"])
                                )
                                if better:
                                    best_rec = row; best_plot_payload = (df_run, preds, df_test, npast, act)
                                    best_model = model
                                    model.save(f"artifacts/best_model_{ticker}_4.keras")
                                    best_scaler = scaler
                                    joblib.dump(best_scaler, f"artifacts/best_scaler_{ticker}_4.joblib")

                        except Exception as e:
                            print("Combo failed:", e)
                            rows.append({
                                "ticker": ticker, "fs_method": fs_method, "top_k": topk,
                                "n_past": npast, "activation": act, "DIR_ACC": np.nan,
                                "MAE": np.nan, "RMSE": np.nan, "MAPE": np.nan, "n_features": np.nan, "features": None
                            })





        # --- optional single plot for best combo per ticker
        if SAVE_PLOTS_BEST and best_rec and best_plot_payload:
            df_best, y_pred_best, y_test_best, npast_best, act_best = best_plot_payload
            try:
                plt.plot(df_test, label='Actual Returns', marker='o', markerfacecolor='none', alpha=0.7)
                plt.plot(preds, label='Predicted Returns', marker='*')
                # plot a red line at y=0
                plt.axhline(0, color='red', linestyle='--', linewidth=1)
                plt.xlabel('Days')
                plt.ylabel('Returns')
                plt.title(f'{ticker} Stock Returns Prediction')
                plt.legend()
                plt.savefig(f'artifacts/prediction_returns_4_{ticker}.png')
                # plt.show()
            except Exception as e:
                print("[Plot Warning]", e)





    # --- results summary tables ---
    results_df = pd.DataFrame(rows)
    results_df.to_csv("artifacts/grid_results_raw_4.csv", index=False)

    def pick_best(df_sub):
        return df_sub.sort_values(by=["DIR_ACC","RMSE","MAE","MAPE"], ascending=[False, True, True, True]).head(1)

    best_per_ticker = results_df.groupby("ticker", as_index=False).apply(pick_best).reset_index(drop=True)
    best_per_ticker.to_csv("artifacts/best_per_ticker_4.csv", index=False)

    overall = (results_df
            .groupby(["fs_method", "top_k", "n_past", "activation"], as_index=False)
            .agg({"RMSE": "median", "MAE": "median", "MAPE": "median", "DIR_ACC": "median"})
            .sort_values(by=["DIR_ACC", "RMSE", "MAE", "MAPE"], ascending=[False, True, True, True]))
    overall.to_csv("artifacts/overall_ranking_4.csv", index=False)




#### TEST 2 CODE #################
# if __name__ == "__main__":
#     data = pd.DataFrame()
#     for ticker in ["AAPL", "GOOG", "MSFT", "TSM", "AVGO", "META", "NVDA", "ORCL", "TCEHY"]:
#         print(f"Processing {ticker}...")

#         y_test = yf.download(ticker, start="2025-08-22", end="2025-08-26", interval="1d")['Close'] #<- start-25/end-26
#         #y_test = yf.download(ticker, start="2025-08-22", end="2025-08-26", interval="1d")['Close'].values #<- Close del 25/08
#         y_test['returns'] = y_test.pct_change()
#         y_test = y_test['returns'].values[1:]  # skip NaN
        
#         # y_t = yf.download(ticker, start="2025-08-22", end="2025-08-23", interval="1d")["Close"].values
#         print("Test prices:", y_test)

#         df = load_transform_data(ticker)
#         # drop columns High, Low, Volume
#         cols = [c for c in ['returns', 'PCT_change', 'weighted_tech_sentiment'] if c in df.columns] #<-Columns Selection
#         df1 = df[cols]
#         # print(df.index[-1])
#         model, forecast_df = forecast_pipeline(df1)
#         forecast_returns_t1 = forecast_df["Predicted_returns"]
#         print("Forecasted prices:", forecast_returns_t1.values)

#         #joblib.dump(model, f"./pkl/forecast_returns_{ticker}.pkl")

#         # df_no = df[["Close","Close_lag1"]].copy()
#         # model, forecast_df_no = forecast_pipeline(df_no)
#         # forecast_Close_no_t1 = forecast_df_no["Close_t1"]


#         plt.figure(figsize=(12,6))
#         plt.plot(df.index[-50:], df["returns"].values[-50:], label="History", linewidth=2)

#         # Generate future dates
#         future_dates = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), 
#                                     periods=len(forecast_df), freq="D")

#         # Plot both forecasts
#         plt.plot(future_dates, forecast_df["Predicted_returns"], label="Forecast", marker='*')

#         # plt.plot(future_dates, forecast_df_no["Close_t1"], label="Forecast NO FE (t+1)", marker='+')

#         # Plot actual test prices if available
#         plt.plot(future_dates[:len(y_test)], y_test, label="Actual", markerfacecolor='none', alpha=0.7, marker='o')

#         plt.title(f"{ticker} Returns Forecast t+1")
#         plt.xlabel("Date")
#         plt.ylabel("Returns")
#         plt.legend()
#         plt.grid(True, alpha=0.3)
#         plt.tight_layout()
#         plt.savefig(f'prediction_Close_{ticker}.png')
#         plt.show()


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