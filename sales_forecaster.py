# sales_forecaster.py
# Sales Volume Forecaster: Predict tomorrow's quantity for each product & city
# Uses: Lag features (1,7,30), rolling mean (7,30), XGBoost, time-series split

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings("ignore")
import sys

# ----------------------------
# 1. Load and Clean Data
# ----------------------------
try:
    df = pd.read_csv('Sales-Data-Analysis.csv')
except FileNotFoundError:
    print("Error: File ' Sales-Data-Analysis.csv' not found in the current directory.")
    sys.exit(1)

# Clean column names and data
df.columns = df.columns.str.strip()
df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y', errors='coerce')
df['Quantity'] = pd.to_numeric(df['Quantity'], errors='coerce')
df['Price'] = pd.to_numeric(df['Price'], errors='coerce')

# Drop invalid rows
df.dropna(subset=['Date', 'Quantity', 'Price'], inplace=True)

# ----------------------------
# 2. Aggregate Daily Sales by Product & City
# ----------------------------
daily_sales = df.groupby(['Date', 'Product', 'City'])['Quantity'].sum().reset_index()

# Pivot: each (Product, City) combo becomes a time series
pivot_df = daily_sales.pivot_table(
    index='Date',
    columns=['Product', 'City'],
    values='Quantity',
    fill_value=0
)

# Flatten column names
pivot_df.columns = [f"{product}_{city}" for product, city in pivot_df.columns]
pivot_df.sort_index(inplace=True)

# ----------------------------
# 3. Feature Engineering: Lag & Rolling Mean
# ----------------------------
def add_lag_rolling_features(df, lags=[1, 7, 30], windows=[7, 30]):
    df_out = df.copy()
    for col in df.columns:
        for lag in lags:
            df_out[f'{col}_lag{lag}'] = df[col].shift(lag)
        for window in windows:
            df_out[f'{col}_rollmean{window}'] = df[col].rolling(window).mean().shift(1)
    return df_out

features_df = add_lag_rolling_features(pivot_df)
targets_df = pivot_df.shift(-1)  # Tomorrow's quantity

# Align and drop NaN
features_df = features_df.iloc[:-1]
targets_df = targets_df.iloc[:-1]

# ----------------------------
# 4. Train Models (One per Product-City)
# ----------------------------
models = {}
predictions = {}
forecast_tomorrow = {}

print("Training XGBoost models for each product-city...\n")

for col in targets_df.columns:
    X = features_df[[c for c in features_df.columns if col in c]]
    y = targets_df[col]

    # Time-series split
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    if len(X_train) == 0 or len(X_test) == 0:
        print(f"Skipping {col}: Not enough data after split.")
        continue

    # Train model
    model = xgb.XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    models[col] = model
    predictions[col] = {'mae': mae, 'rmse': rmse}

    print(f"{col:30} MAE: {mae:6.2f}, RMSE: {rmse:6.2f}")

# ----------------------------
# 5. Forecast for Tomorrow
# ----------------------------
last_features = features_df.iloc[[-1]]  # Most recent row

for col in targets_df.columns:
    if col not in models:
        continue
    model = models[col]
    X_input = last_features[[c for c in last_features.columns if col in c]]
    pred = model.predict(X_input)[0]
    forecast_tomorrow[col] = max(0, round(pred, 2))  # No negative sales

# Convert forecast to structured DataFrame
forecast_list = []
for key, value in forecast_tomorrow.items():
    try:
        product, city = key.split('_', 1)
        forecast_list.append({'Product': product, 'City': city, 'Predicted_Quantity': value})
    except:
        # Fallback in case of unexpected column name
        forecast_list.append({'Product': 'Unknown', 'City': 'Unknown', 'Predicted_Quantity': value})

forecast_df = pd.DataFrame(forecast_list)
forecast_df.sort_values(by=['City', 'Product'], inplace=True)

# ----------------------------
# 6. Output Results
# ----------------------------
print("\n" + "="*50)
print("PREDICTED SALES FOR TOMORROW")
print("="*50)
print(forecast_df.to_string(index=False))

# Optional: Save to CSV
forecast_df.to_csv('predicted_sales_tomorrow.csv', index=False)
print(f"\n✅ Forecast saved to 'predicted_sales_tomorrow.csv'")
print(f"📅 Forecast Date: {pd.to_datetime('today') + pd.Timedelta(days=1):%Y-%m-%d}")

# Optional: Print model performance summary
print(f"\n📊 Average MAE across series: {np.mean([v['mae'] for v in predictions.values()]):.2f}")