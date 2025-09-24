from sklearn.ensemble import RandomForestRegressor
import feature_engineering as fe
import pandas as pd

def create_df(ticker):
    df = fe.lt_yf_data(db_path=f"./data/{ticker}_yf.duckdb")
    df = fe.join_tch_vars(df)
    print(df.head())  # Optional preview

    sec_df = fe.lt_sec_data(db_path=f"./data/{ticker}_10q.duckdb")

    news_df = fe.lt_news_data(ticker)

    instruments_df = fe.lt_yf_data(db_path=f"./data/instruments.duckdb")
    print(instruments_df.head())  # Optional preview

    # Merge all dataframes on date
    df = df.merge(sec_df, how='left', left_index=True, right_index=True)
    df = df.merge(news_df, how='left', left_index=True, right_index=True)
    df = df.merge(instruments_df, how='left', left_index=True, right_index=True)
    df = df.ffill()

    if 'weighted_sentiment' in df.columns:
        df = df.dropna(subset=['weighted_sentiment'])

    return df

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

if __name__ == "__main__":
    ticker = "ORCL"
    df = create_df(ticker)
    df_selected = select_features(df, target_col='Close', threshold=0.6)
    print(df_selected.head()) 