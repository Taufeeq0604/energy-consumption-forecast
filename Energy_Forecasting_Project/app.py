import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import plotly.express as px
import plotly.graph_objects as go

# --------------------------------
# Page Config
# --------------------------------
st.set_page_config(
    page_title="Hourly Energy Forecast",
    page_icon="⚡",
    layout="wide"
)

# --------------------------------
# Load Dataset
# --------------------------------
@st.cache_data
def load_data():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(BASE_DIR, "PJMW_MW_Hourly.xlsx")

    df = pd.read_excel(file_path)
    df.columns = ["Datetime", "PJMW_MW"]

    df["Datetime"] = pd.to_datetime(df["Datetime"])
    df.set_index("Datetime", inplace=True)

    return df


# --------------------------------
# Load Model
# --------------------------------
@st.cache_resource
def load_model():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(BASE_DIR, "energy_model.pkl")

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    return model


# --------------------------------
# Load data & model
# --------------------------------
df = load_data()
model = load_model()

st.title("⚡ Hourly Energy Consumption Forecasting")

st.markdown("Forecasting electricity demand using Machine Learning")

# --------------------------------
# Sidebar
# --------------------------------
st.sidebar.header("Forecast Settings")

forecast_days = st.sidebar.slider(
    "Forecast Horizon (Days)",
    1,
    30,
    7
)

# --------------------------------
# Basic EDA
# --------------------------------
st.subheader("📊 Dataset Overview")

col1, col2, col3 = st.columns(3)

col1.metric("Total Records", len(df))
col2.metric("Start Date", str(df.index.min().date()))
col3.metric("End Date", str(df.index.max().date()))

# --------------------------------
# Historical Plot
# --------------------------------
st.subheader("📈 Historical Data")

fig = px.line(df, y="PJMW_MW", title="Energy Consumption Over Time")
st.plotly_chart(fig, use_container_width=True)

# --------------------------------
# Feature Engineering (IMPORTANT)
# --------------------------------
df["hour"] = df.index.hour
df["day"] = df.index.day
df["month"] = df.index.month
df["lag_1"] = df["PJMW_MW"].shift(1)
df["lag_24"] = df["PJMW_MW"].shift(24)
df = df.dropna()

# --------------------------------
# Forecasting
# --------------------------------
st.subheader("🔮 Forecast")

future_dates = pd.date_range(
    start=df.index.max(),
    periods=forecast_days * 24,
    freq="h"
)

last_data = df.copy()
forecast_list = []

for i in range(len(future_dates)):

    current_time = future_dates[i]

    input_df = pd.DataFrame({
        "hour": [current_time.hour],
        "day": [current_time.day],
        "month": [current_time.month],
        "lag_1": [last_data["PJMW_MW"].iloc[-1]],
        "lag_24": [last_data["PJMW_MW"].iloc[-24]]
    })

    pred = model.predict(input_df)[0]
    forecast_list.append(pred)

    # update dataset for next step
    new_row = pd.DataFrame({"PJMW_MW": [pred]}, index=[current_time])
    last_data = pd.concat([last_data, new_row])

# --------------------------------
# Forecast DataFrame
# --------------------------------
forecast_df = pd.DataFrame({
    "Datetime": future_dates,
    "Forecast_MW": forecast_list
})

# --------------------------------
# Plot Forecast
# --------------------------------
fig_forecast = go.Figure()

fig_forecast.add_trace(go.Scatter(
    x=df.index[-500:],
    y=df["PJMW_MW"][-500:],
    mode="lines",
    name="Historical"
))

fig_forecast.add_trace(go.Scatter(
    x=forecast_df["Datetime"],
    y=forecast_df["Forecast_MW"],
    mode="lines",
    name="Forecast"
))

fig_forecast.update_layout(
    title="Energy Demand Forecast",
    xaxis_title="Time",
    yaxis_title="MW"
)

st.plotly_chart(fig_forecast, use_container_width=True)

# --------------------------------
# Forecast Table
# --------------------------------
st.subheader("📋 Forecast Data")
st.dataframe(forecast_df)

# --------------------------------
# Download
# --------------------------------
csv = forecast_df.to_csv(index=False)

st.download_button(
    "⬇ Download Forecast",
    csv,
    "forecast.csv",
    "text/csv"
)

# --------------------------------
# Footer
# --------------------------------
st.markdown("---")
st.markdown("Built using Streamlit + Machine Learning ⚡")
