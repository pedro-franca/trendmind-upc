from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import pandas as pd
from feature_engineering import create_df

def fsae_feature_importance(X, bottleneck_dim=10, epochs=50, batch_size=32):
    """
    Train an autoencoder and rank features by reconstruction error.
    Lower error = more important feature.
    """
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    n_features = X.shape[1]

    # Simple symmetric autoencoder
    autoencoder = Sequential([
        Dense(64, activation='relu', input_shape=(n_features,)),
        Dense(bottleneck_dim, activation='relu'),
        Dense(64, activation='relu'),
        Dense(n_features, activation='linear')
    ])
    autoencoder.compile(optimizer='adam', loss='mse')
    autoencoder.fit(X_scaled, X_scaled, epochs=epochs, batch_size=batch_size, verbose=0)

    # Reconstruction
    X_recon = autoencoder.predict(X_scaled, verbose=0)
    errors = ((X_scaled - X_recon) ** 2).mean(axis=0)

    # Lower error → better represented feature
    fsae_importance = pd.Series(-errors, index=X.columns).sort_values(ascending=False)
    return fsae_importance

def select_features(df, target_col='Close'):

    # 1. Random Forest importance
    X = df.drop(columns=[target_col])
    y = df[target_col]
    rf = RandomForestRegressor(n_estimators=300, random_state=42)
    rf.fit(X, y)
    rf_importance = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
    top_rf = set(rf_importance.head(20).index)
    print(f"Top 20 features by Random Forest: {list(top_rf)}")

    # 2. FSAE importance
    fsae_importance = fsae_feature_importance(X)
    top_fsae = set(fsae_importance.head(20).index)
    print(f"Top 20 features by FSAE: {list(top_fsae)}")

    # 4. Consensus
    consensus_features = set(top_rf).union(top_fsae)
    print(f"Consensus selected {len(consensus_features)} features: {list(consensus_features)}")

    return df[list(consensus_features) + [target_col]]


if __name__ == "__main__":
    ticker = "AAPL"
    df = create_df(ticker)

    print("Original features:", df.columns.tolist())
    df_selected = select_features(df, target_col='Close')
    print("Selected features:", df_selected.columns.tolist())
    print(df_selected.head())