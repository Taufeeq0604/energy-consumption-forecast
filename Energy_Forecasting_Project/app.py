import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go

# --------------------------------
# Page Configuration
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
    df = pd.read_excel("PJMW_MW_Hourly.xlsx")

    df.columns = ['Datetime', 'PJMW_MW']

    df['Datetime'] = pd.to_datetime(df['Datetime'])

    df.set_index('Datetime', inplace=True)

    return df

df = load_data()

# --------------------------------
# Load Model
# --------------------------------

@st.cache_resource
def load_model():
    with open("energy_model.pkl", "rb") as f:
        model = pickle.load(f)

    return model

model = load_model()

# --------------------------------
# Title
# --------------------------------

st.title("⚡ Hourly Energy Consumption Forecasting")

st.markdown("""
This dashboard analyzes PJM Hourly Energy Consumption and forecasts future electricity demand using Machine Learning.
""")

# --------------------------------
# Sidebar
# --------------------------------

st.sidebar.header("Forecast Settings")

forecast_days = st.sidebar.slider(
    "Forecast Horizon (Days)",
    min_value=1,
    max_value=30,
    value=7
)

# --------------------------------
# Dataset Overview
# --------------------------------

st.subheader("📊 Dataset Overview")

col1, col2, col3 = st.columns(3)

col1.metric(
    "Total Records",
    f"{len(df):,}"
)

col2.metric(
    "Start Date",
    str(df.index.min().date())
)

col3.metric(
    "End Date",
    str(df.index.max().date())
)

# --------------------------------
# Historical Consumption
# --------------------------------

st.subheader("📈 Historical Energy Consumption")

fig = px.line(
    df,
    y="PJMW_MW",
    title="Hourly Energy Consumption"
)

st.plotly_chart(fig, use_container_width=True)

# --------------------------------
# Hourly Trend
# --------------------------------

st.subheader("⏰ Average Consumption by Hour")

hourly_avg = (
    df.groupby(df.index.hour)["PJMW_MW"]
    .mean()
)

fig_hour = px.bar(
    x=hourly_avg.index,
    y=hourly_avg.values,
    labels={
        "x": "Hour",
        "y": "Average MW"
    }
)

st.plotly_chart(fig_hour, use_container_width=True)

# --------------------------------
# Monthly Trend
# --------------------------------

st.subheader("📅 Average Consumption by Month")

monthly_avg = (
    df.groupby(df.index.month)["PJMW_MW"]
    .mean()
)

fig_month = px.bar(
    x=monthly_avg.index,
    y=monthly_avg.values,
    labels={
        "x": "Month",
        "y": "Average MW"
    }
)

st.plotly_chart(fig_month, use_container_width=True)

# --------------------------------
# Forecast Section
# --------------------------------

st.subheader("🔮 Energy Forecast")

future_dates = pd.date_range(
    start=df.index.max(),
    periods=forecast_days * 24,
    freq="H"
)

# Temporary Forecast Placeholder
# Replace with actual XGBoost forecasting logic

forecast_values = np.repeat(
    df['PJMW_MW'].tail(24).mean(),
    len(future_dates)
)

forecast_df = pd.DataFrame({
    "Datetime": future_dates,
    "Forecast_MW": forecast_values
})

# --------------------------------
# Forecast Plot
# --------------------------------

fig_forecast = go.Figure()

fig_forecast.add_trace(
    go.Scatter(
        x=df.index[-500:],
        y=df['PJMW_MW'][-500:],
        mode='lines',
        name='Historical'
    )
)

fig_forecast.add_trace(
    go.Scatter(
        x=forecast_df['Datetime'],
        y=forecast_df['Forecast_MW'],
        mode='lines',
        name='Forecast'
    )
)

fig_forecast.update_layout(
    title="Future Energy Forecast",
    xaxis_title="Date",
    yaxis_title="Energy Consumption (MW)"
)

st.plotly_chart(
    fig_forecast,
    use_container_width=True
)

# --------------------------------
# Forecast Table
# --------------------------------

st.subheader("📋 Forecast Values")

st.dataframe(forecast_df)

# --------------------------------
# Download Forecast
# --------------------------------

csv = forecast_df.to_csv(index=False)

st.download_button(
    label="⬇ Download Forecast CSV",
    data=csv,
    file_name="energy_forecast.csv",
    mime="text/csv"
)

# --------------------------------
# Footer
# --------------------------------

st.markdown("---")

st.markdown(
    """
    Developed using **Streamlit**, **XGBoost**, and **PJM Energy Consumption Data**
    """
)