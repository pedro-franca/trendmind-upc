import pandas as pd
import numpy as np
from feature_engineering import load_data
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

data = pd.DataFrame()
for ticker in ["AAPL", "GOOG", "MSFT", "TSM", "AVGO", "META", "NVDA", "ORCL", "TCEHY"]:
    df = load_data(db_path=f"./data/{ticker}_yf.duckdb")
    df = df[['Close']].rename(columns={'Close': f'Close_{ticker}'})
    data = pd.concat([data, df], axis=1)
    data.sort_index(inplace= True)

indice_df = pd.read_csv('data/W1TEC.csv')
# convert date column to datetime
indice_df['Date'] = pd.to_datetime(indice_df['Date'], format="%d/%m/%Y")
indice_df['Indice'] = pd.to_numeric(indice_df['Indice'])
indice_df.set_index('Date', inplace=True)
df = pd.concat([indice_df,data], axis=1)
df = df.dropna()

target_col = "Indice"
feature_cols = [col for col in df.columns if col != target_col]

X = df[feature_cols]
y = df[target_col]

# ---------------------------
# 2. Train/test split
# ---------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False  # time series → no random shuffle
)

# ---------------------------
# 3. Train XGBoost
# ---------------------------
model = xgb.XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
model.fit(X_train, y_train)

# ---------------------------
# 4. Evaluate
# ---------------------------
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"RMSE: {rmse:.2f}")
print(f"R²: {r2:.3f}")

# ---------------------------
# 5. Feature importance
# ---------------------------
import matplotlib.pyplot as plt

xgb.plot_importance(model, importance_type="weight", height=0.6)
plt.show()

# ---------------------------
# 6. Forecast future values (optional)
# ---------------------------
# Example: if you want to predict based on the last row
last_row = X.iloc[[-1]]
future_pred = model.predict(last_row)[0]
print(f"Predicted Indice for next row: {future_pred:.2f}")
