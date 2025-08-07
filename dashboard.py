# dashboard.py
# Streamlit Dashboard for Sales Volume Forecaster

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import joblib  # if you want to save models
import os

# ----------------------------
# 1. Load and Prepare Data
# ----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv('Sales-Data-Analysis.csv')
    df.columns = df.columns.str.strip()
    df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y', errors='coerce')
    df['Quantity'] = pd.to_numeric(df['Quantity'], errors='coerce')
    df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
    df.dropna(subset=['Date', 'Quantity', 'Product', 'City'], inplace=True)
    return df

df = load_data()

# ----------------------------
# 2. Forecast Logic (Embedded)
# ----------------------------
@st.cache_data
def get_forecast():
    # Aggregate daily sales
    daily_sales = df.groupby(['Date', 'Product', 'City'])['Quantity'].sum().reset_index()

    # Pivot
    pivot_df = daily_sales.pivot_table(
        index='Date',
        columns=['Product', 'City'],
        values='Quantity',
        fill_value=0
    )
    pivot_df.columns = [f"{p}_{c}" for p, c in pivot_df.columns]
    pivot_df.sort_index(inplace=True)

    # Feature engineering
    def add_lag_rolling_features(df, lags=[1, 7, 30], windows=[7, 30]):
        df_out = df.copy()
        for col in df.columns:
            for lag in lags:
                df_out[f'{col}_lag{lag}'] = df[col].shift(lag)
            for window in windows:
                df_out[f'{col}_rollmean{window}'] = df[col].rolling(window).mean().shift(1)
        return df_out

    features_df = add_lag_rolling_features(pivot_df)
    targets_df = pivot_df.shift(-1)

    features_df = features_df.iloc[:-1]
    targets_df = targets_df.iloc[:-1]

    # Forecast container
    forecast_tomorrow = {}

    for col in targets_df.columns:
        X = features_df[[c for c in features_df.columns if col in c]]
        y = targets_df[col]

        if len(X) < 10:
            continue

        # Use last 80% as test, train on earlier
        split_idx = int(0.8 * len(X))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        # XGBoost
        from xgboost import XGBRegressor
        model = XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
        model.fit(X_train, y_train)

        # Predict tomorrow
        X_last = X.iloc[[-1]]
        pred = model.predict(X_last)[0]
        forecast_tomorrow[col] = max(0, round(pred, 2))

    # Convert to DataFrame
    forecast_list = []
    for key, val in forecast_tomorrow.items():
        try:
            product, city = key.split('_', 1)
            forecast_list.append({'Product': product, 'City': city, 'Predicted_Quantity': val})
        except:
            forecast_list.append({'Product': 'Unknown', 'City': 'Unknown', 'Predicted_Quantity': val})
    return pd.DataFrame(forecast_list)

# Get forecast
try:
    forecast_df = get_forecast()
except Exception as e:
    st.error(f"Forecast generation failed: {e}")
    forecast_df = pd.DataFrame(columns=['Product', 'City', 'Predicted_Quantity'])

# ----------------------------
# 3. Streamlit App Layout
# ----------------------------
st.set_page_config(page_title="Sales Volume Forecaster", layout="wide")

st.title("🍔 Sales Volume Forecaster")
st.markdown("Predicting tomorrow's sales quantity for each product and city using **XGBoost + Lag & Rolling Features**")

# Tabs
tab1, tab2, tab3 = st.tabs(["📊 Forecast Dashboard", "📈 Trend Explorer", "📁 Data Viewer"])

# ----------------------------
# Tab 1: Forecast Dashboard
# ----------------------------
with tab1:
    st.header("Predicted Sales for Tomorrow")
    if not forecast_df.empty:
        st.dataframe(forecast_df.style.format({"Predicted_Quantity": "{:.1f}"}), use_container_width=True)

        # Summary KPIs
        total_pred = forecast_df['Predicted_Quantity'].sum()
        avg_pred = forecast_df['Predicted_Quantity'].mean()
        num_forecasts = len(forecast_df)

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Predicted Units", f"{int(total_pred):,}")
        col2.metric("Average per Product-City", f"{avg_pred:.1f}")
        col3.metric("Number of Forecasts", num_forecasts)

        # Bar chart: Top 10 predictions
        top10 = forecast_df.nlargest(10, 'Predicted_Quantity')
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(data=top10, x='Predicted_Quantity', y='Product', hue='City', dodge=False, ax=ax)
        ax.set_title("Top 10 Predicted Sales (Product-City)")
        ax.set_xlabel("Predicted Quantity")
        st.pyplot(fig)
    else:
        st.warning("No forecast data available.")

# ----------------------------
# Tab 2: Trend Explorer
# ----------------------------
with tab2:
    st.header("Sales Trend Explorer")

    product_list = df['Product'].unique().tolist()
    city_list = df['City'].unique().tolist()

    col1, col2 = st.columns(2)
    with col1:
        selected_product = st.selectbox("Select Product", options=product_list)
    with col2:
        selected_city = st.selectbox("Select City", options=city_list)

    filtered_data = df[(df['Product'] == selected_product) & (df['City'] == selected_city)]
    daily_qty = filtered_data.groupby('Date')['Quantity'].sum().reset_index()

    if not daily_qty.empty:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(daily_qty['Date'], daily_qty['Quantity'], marker='o', linewidth=2, markersize=4)
        ax.set_title(f"Daily Sales Trend: {selected_product} in {selected_city}")
        ax.set_ylabel("Quantity Sold")
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        st.pyplot(fig)

        # Show last forecast for this combo
        pred_row = forecast_df[
            (forecast_df['Product'] == selected_product) &
            (forecast_df['City'] == selected_city)
        ]
        if not pred_row.empty:
            st.success(f"Predicted Quantity for Tomorrow: **{pred_row['Predicted_Quantity'].values[0]:.1f}**")
        else:
            st.info("No forecast available for this combination.")
    else:
        st.info("No data available for the selected product and city.")

# ----------------------------
# Tab 3: Raw Data Viewer
# ----------------------------
with tab3:
    st.header("Raw Sales Data")
    st.dataframe(df, use_container_width=True)

    # Download button
    @st.cache_data
    def convert_df_to_csv(_df):
        return _df.to_csv(index=False).encode('utf-8')

    csv = convert_df_to_csv(df)
    st.download_button(
        label="Download Full Data as CSV",
        data=csv,
        file_name='sales_data_cleaned.csv',
        mime='text/csv',
    )

# ----------------------------
# Footer
# ----------------------------
st.markdown("---")
st.markdown("💬 **Model**: XGBoost | 📅 **Features**: Lag (1,7,30) + Rolling Mean (7,30) | 🧠 **Forecast for tomorrow**")