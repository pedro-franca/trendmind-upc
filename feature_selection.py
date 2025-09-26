from sklearn.ensemble import RandomForestRegressor
import feature_engineering as fe
import pandas as pd


def select_features(df, target_col='Close', threshold=0.1):
    corr_matrix = df.corr()
    target_corr = corr_matrix[target_col].abs()
    selected_features = target_corr[target_corr > threshold].index.tolist()
    selected_features.remove(target_col)  # Remove target column from features
    print(f"Selected {len(selected_features)} features based on correlation threshold {threshold}:")
    print(selected_features)

    X = df.drop(columns=[target_col])
    y = df[target_col]
    rf = RandomForestRegressor(n_estimators=300, random_state=42)
    rf.fit(X, y)

    rf_importance = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)

    top_rf = set(rf_importance.head(20).index)

    consensus_features = set(selected_features).union(top_rf)

    print(f"Consensus selected {len(consensus_features)} features:")
    print(consensus_features)

    return df[list(consensus_features) + [target_col]]
