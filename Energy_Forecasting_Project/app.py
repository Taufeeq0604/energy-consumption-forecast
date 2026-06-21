import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
import os

# ------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------

st.set_page_config(
    page_title="Hourly Energy Consumption Forecasting",
    page_icon="⚡",
    layout="wide"
)

# ------------------------------------------------
# FILE PATHS
# ------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(BASE_DIR, "PJMW_MW_Hourly.xlsx")
METRICS_PATH = os.path.join(BASE_DIR, "metrics.pkl")
PRED_PATH = os.path.join(BASE_DIR, "predictions.csv")
IMPORTANCE_PATH = os.path.join(BASE_DIR, "feature_importance.csv")

# ------------------------------------------------
# LOAD DATA
# ------------------------------------------------

@st.cache_data
def load_data():

    df = pd.read_excel(DATA_PATH)

    df["Datetime"] = pd.to_datetime(df["Datetime"])

    df = df.sort_values("Datetime")

    return df

df = load_data()

# ------------------------------------------------
# LOAD MODEL
# ------------------------------------------------

MODEL_PATH = os.path.join(BASE_DIR, "energy_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")

@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

@st.cache_resource
def load_scaler():
    with open(SCALER_PATH, "rb") as f:
        return pickle.load(f)

model = load_model()
scaler = load_scaler()

@st.cache_resource
def load_metrics():

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    metrics_path = os.path.join(
        BASE_DIR,
        "metrics.pkl"
    )

    with open(metrics_path, "rb") as f:
        metrics = pickle.load(f)

    return metrics

metrics = load_metrics()
# ------------------------------------------------
# TITLE
# ------------------------------------------------

st.title("⚡ Hourly Energy Consumption Forecasting")

st.markdown("""
This dashboard analyzes PJM Hourly Energy Consumption data
and evaluates the performance of the XGBoost forecasting model.
""")

# ------------------------------------------------
# SIDEBAR
# ------------------------------------------------

st.sidebar.header("Project Information")

st.sidebar.success("Model: XGBoost Regressor")

st.sidebar.info("""
Dataset:
PJM Hourly Energy Consumption

Target Variable:
PJMW_MW
""")

# ------------------------------------------------
# DATASET OVERVIEW
# ------------------------------------------------

st.header("📊 Dataset Overview")

col1, col2, col3 = st.columns(3)

col1.metric(
    "Total Records",
    f"{len(df):,}"
)

col2.metric(
    "Start Date",
    str(df["Datetime"].min().date())
)

col3.metric(
    "End Date",
    str(df["Datetime"].max().date())
)

st.dataframe(df.head())

# ------------------------------------------------
# HISTORICAL TREND
# ------------------------------------------------

st.header("📈 Historical Energy Consumption")

fig = px.line(
    df,
    x="Datetime",
    y="PJMW_MW",
    title="Energy Consumption Over Time"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# ------------------------------------------------
# FEATURE ENGINEERING FOR EDA
# ------------------------------------------------

df["hour"] = df["Datetime"].dt.hour
df["month"] = df["Datetime"].dt.month
df["dayofweek"] = df["Datetime"].dt.day_name()

# ------------------------------------------------
# HOURLY TREND
# ------------------------------------------------

st.header("⏰ Hourly Consumption Pattern")

hourly_avg = (
    df.groupby("hour")["PJMW_MW"]
    .mean()
    .reset_index()
)

fig_hour = px.bar(
    hourly_avg,
    x="hour",
    y="PJMW_MW",
    title="Average Energy Consumption by Hour"
)

st.plotly_chart(
    fig_hour,
    use_container_width=True
)

# ------------------------------------------------
# MONTHLY TREND
# ------------------------------------------------

st.header("📅 Monthly Consumption Pattern")

monthly_avg = (
    df.groupby("month")["PJMW_MW"]
    .mean()
    .reset_index()
)

fig_month = px.bar(
    monthly_avg,
    x="month",
    y="PJMW_MW",
    title="Average Energy Consumption by Month"
)

st.plotly_chart(
    fig_month,
    use_container_width=True
)

# ------------------------------------------------
# DAY OF WEEK
# ------------------------------------------------

st.header("📆 Day Wise Consumption")

day_avg = (
    df.groupby("dayofweek")["PJMW_MW"]
    .mean()
    .reindex([
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday"
    ])
    .reset_index()
)

fig_day = px.bar(
    day_avg,
    x="dayofweek",
    y="PJMW_MW",
    title="Average Consumption by Day"
)

st.plotly_chart(
    fig_day,
    use_container_width=True
)

# ------------------------------------------------
# MODEL METRICS
# ------------------------------------------------

st.header("🎯 Model Performance")

with open(METRICS_PATH, "rb") as f:
    metrics = pickle.load(f)

col1, col2, col3 = st.columns(3)

col1.metric(
    "MAE",
    f"{metrics['MAE']:.2f}"
)

col2.metric(
    "RMSE",
    f"{metrics['RMSE']:.2f}"
)

col3.metric(
    "R² Score",
    f"{metrics['R2']:.4f}"
)

# ------------------------------------------------
# ACTUAL VS PREDICTED
# ------------------------------------------------

st.header("📈 Actual vs Predicted")

pred_df = pd.read_csv(PRED_PATH)

fig_pred = go.Figure()

fig_pred.add_trace(
    go.Scatter(
        y=pred_df["Actual"][:1000],
        mode="lines",
        name="Actual"
    )
)

fig_pred.add_trace(
    go.Scatter(
        y=pred_df["Predicted"][:1000],
        mode="lines",
        name="Predicted"
    )
)

fig_pred.update_layout(
    title="Actual vs Predicted Values"
)

st.plotly_chart(
    fig_pred,
    use_container_width=True
)

# ------------------------------------------------
# ERROR DISTRIBUTION
# ------------------------------------------------

st.header("📊 Prediction Error Distribution")

pred_df["Error"] = (
    pred_df["Actual"]
    - pred_df["Predicted"]
)

fig_error = px.histogram(
    pred_df,
    x="Error",
    nbins=50,
    title="Prediction Error Distribution"
)

st.plotly_chart(
    fig_error,
    use_container_width=True
)

# ------------------------------------------------
# FEATURE IMPORTANCE
# ------------------------------------------------

st.header("⭐ Feature Importance")

importance_df = pd.read_csv(
    IMPORTANCE_PATH
)

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=True
)

fig_imp = px.bar(
    importance_df,
    x="Importance",
    y="Feature",
    orientation="h",
    title="XGBoost Feature Importance"
)

st.plotly_chart(
    fig_imp,
    use_container_width=True
)

# ------------------------------------------------
# TOP FEATURES
# ------------------------------------------------

st.subheader("Top Important Features")

st.dataframe(
    importance_df.sort_values(
        by="Importance",
        ascending=False
    )
)
# =====================================
# FUTURE FORECAST USING XGBOOST
# =====================================

st.header("🔮 Future Energy Forecast")

forecast_days = st.sidebar.slider(
    "Forecast Days",
    1,
    30,
    7
)

future_hours = forecast_days * 24

history = df.copy()

future_predictions = []

last_time = pd.to_datetime(history["Datetime"].max())

for i in range(future_hours):

    future_time = last_time + pd.Timedelta(hours=1)

    feature_row = pd.DataFrame({
        'hour': [future_time.hour],
        'dayofweek': [future_time.dayofweek],
        'month': [future_time.month],
        'quarter': [future_time.quarter],
        'year': [future_time.year],
        'dayofyear': [future_time.dayofyear],
        'rolling_30day': [history['PJMW_MW'].tail(24*30).mean()],
        'lag24': [history['PJMW_MW'].iloc[-24]],
        'lag48': [history['PJMW_MW'].iloc[-48]],
        'lag168': [history['PJMW_MW'].iloc[-168]],
        'rolling24_mean': [history['PJMW_MW'].tail(24).mean()],
        'rolling168_mean': [history['PJMW_MW'].tail(168).mean()]
    })

    feature_scaled = scaler.transform(feature_row)

    pred = model.predict(feature_scaled)[0]

    future_predictions.append(pred)

    history = pd.concat([
        history,
        pd.DataFrame({
            "Datetime": [future_time],
            "PJMW_MW": [pred]
        })
    ], ignore_index=True)

    last_time = future_time

forecast_df = pd.DataFrame({
    "Datetime": pd.date_range(
        start=df["Datetime"].max() + pd.Timedelta(hours=1),
        periods=future_hours,
        freq="h"
    ),
    "Forecast_MW": future_predictions
})

fig_forecast = go.Figure()

fig_forecast.add_trace(
    go.Scatter(
        x=df["Datetime"].tail(500),
        y=df["PJMW_MW"].tail(500),
        mode="lines",
        name="Historical"
    )
)

fig_forecast.add_trace(
    go.Scatter(
        x=forecast_df["Datetime"],
        y=forecast_df["Forecast_MW"],
        mode="lines",
        name="Forecast"
    )
)

fig_forecast.update_layout(
    title=f"{forecast_days}-Day Energy Forecast",
    xaxis_title="Date",
    yaxis_title="Energy Consumption (MW)"
)

st.plotly_chart(
    fig_forecast,
    use_container_width=True
)

st.subheader("📋 Forecast Values")

st.dataframe(
    forecast_df,
    use_container_width=True
)

csv = forecast_df.to_csv(index=False)

st.download_button(
    "⬇ Download Forecast CSV",
    csv,
    "future_forecast.csv",
    "text/csv"
)

# =====================================
# MODEL EVALUATION METRICS
# =====================================

st.header("📊 Model Evaluation")

col1, col2, col3 = st.columns(3)

col1.metric(
    "MAE",
    f"{metrics['MAE']:.2f}"
)

col2.metric(
    "RMSE",
    f"{metrics['RMSE']:.2f}"
)

col3.metric(
    "R² Score",
    f"{metrics['R2']:.4f}"
)

st.info("""
**MAE (Mean Absolute Error)** → Average prediction error.

**RMSE (Root Mean Squared Error)** → Penalizes larger errors.

**R² Score** → Measures how well the model explains the variance in energy consumption.
""")

# ------------------------------------------------
# DOWNLOAD PREDICTIONS
# ------------------------------------------------

st.header("⬇ Download Predictions")

csv = pred_df.to_csv(index=False)

st.download_button(
    label="Download Prediction Results",
    data=csv,
    file_name="prediction_results.csv",
    mime="text/csv"
)

# ------------------------------------------------
# PROJECT SUMMARY
# ------------------------------------------------

st.header("📌 Project Summary")

st.success("""
Model Used : XGBoost Regressor

Features Used:
• hour
• dayofweek
• month
• quarter
• year
• dayofyear
• rolling_30day
• lag24
• lag48
• lag168
• rolling24_mean
• rolling168_mean

Objective:
Forecast hourly energy consumption using PJM energy data
and identify seasonal, hourly and long-term demand patterns.
""")

st.markdown("---")
st.markdown("Developed using Streamlit, XGBoost and PJM Energy Dataset")
