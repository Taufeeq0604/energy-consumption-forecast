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
# Load Data
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


df = load_data()
model = load_model()

st.title("⚡ Hourly Energy Consumption Forecasting")

st.sidebar.header("Forecast Settings")

forecast_days = st.sidebar.slider(
    "Forecast Horizon (Days)",
    1,
    30,
    7
)

# --------------------------------
# Feature Engineering
# --------------------------------
df["hour"] = df.index.hour
df["day"] = df.index.day
df["month"] = df.index.month
df["lag_1"] = df["PJMW_MW"].shift(1)
df["lag_24"] = df["PJMW_MW"].shift(24)

df = df.dropna()

# --------------------------------
# Forecast Dates
# --------------------------------
future_dates = pd.date_range(
    start=df.index.max(),
    periods=forecast_days * 24,
    freq="h"
)

# --------------------------------
# Recursive Forecasting
# --------------------------------
last_data = df.copy()
forecast_list = []

feature_order = model.get_booster().feature_names

for i in range(len(future_dates)):

    current_time = future_dates[i]

    input_df = pd.DataFrame([{
        "hour": current_time.hour,
        "day": current_time.day,
        "month": current_time.month,
        "lag_1": float(last_data["PJMW_MW"].iloc[-1]),
        "lag_24": float(last_data["PJMW_MW"].iloc[-24])
    }])

    # 🔥 FIX: ensure correct feature order
    input_df = input_df[feature_order]

    pred = model.predict(input_df)[0]
    forecast_list.append(pred)

    # update dataset
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
# Plot
# --------------------------------
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=df.index[-500:],
    y=df["PJMW_MW"][-500:],
    mode="lines",
    name="Historical"
))

fig.add_trace(go.Scatter(
    x=forecast_df["Datetime"],
    y=forecast_df["Forecast_MW"],
    mode="lines",
    name="Forecast"
))

fig.update_layout(
    title="Energy Forecast",
    xaxis_title="Time",
    yaxis_title="MW"
)

st.plotly_chart(fig, use_container_width=True)

# --------------------------------
# Output Table
# --------------------------------
st.subheader("Forecast Data")
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
