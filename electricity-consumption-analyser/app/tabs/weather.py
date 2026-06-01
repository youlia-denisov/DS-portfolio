import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


def render_weather(load_weather_data):
    st.header("Weather & Electricity")
    st.markdown(
        "Does your electricity use go up when it's hot? Does rain make a difference? "
        "This tab lines up your hourly consumption with weather data to find out."
    )

    df_weather = load_weather_data()
    if df_weather is None:
        st.warning("Weather data not found. Run the pipeline with weather enabled, then refresh.")
        st.code("run_pipeline(run_weather=True)", language="python")
        return

    _render_correlation_metrics(df_weather)
    st.divider()
    _render_dual_axis_trend(df_weather)
    _render_temp_band_chart(df_weather)


def _render_correlation_metrics(df_weather):
    corr_temp = df_weather["kWh"].corr(df_weather["temperature_c"])
    corr_hum  = df_weather["kWh"].corr(df_weather["humidity_pct"])
    corr_prec = df_weather["kWh"].corr(df_weather["precipitation_mm"])

    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("Corr: kWh vs Temperature",   f"{corr_temp:.3f}")
    with m2:
        st.metric("Corr: kWh vs Humidity",      f"{corr_hum:.3f}")
    with m3:
        st.metric("Corr: kWh vs Precipitation", f"{corr_prec:.3f}")

    st.caption(
        "**Correlation** measures how closely two things move together, on a scale from -1 to +1. "
        "+1 means they rise and fall perfectly in sync (hotter → always more electricity). "
        "-1 means the opposite (hotter → always less electricity). "
        "0 means no relationship at all. Values around ±0.3 are considered a moderate link; ±0.6 is strong."
    )


def _render_dual_axis_trend(df_weather):
    st.subheader("Daily Trend: Consumption & Temperature")
    daily_weather = (
        df_weather.groupby("date")
        .agg(daily_kWh=("kWh", "sum"), avg_temp=("temperature_c", "mean"))
        .reset_index()
    )
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=daily_weather["date"], y=daily_weather["daily_kWh"],
        name="Daily kWh", line=dict(color="#e55c30"), yaxis="y1",
    ))
    fig.add_trace(go.Scatter(
        x=daily_weather["date"], y=daily_weather["avg_temp"],
        name="Avg Temp (°C)", line=dict(color="#4a90d9", dash="dot"), yaxis="y2",
    ))
    fig.update_layout(
        title="Daily Consumption vs Average Temperature",
        xaxis_title="Date",
        yaxis=dict(title="Daily kWh", side="left"),
        yaxis2=dict(title="Avg Temperature (°C)", side="right", overlaying="y", showgrid=False),
        legend=dict(x=0.01, y=0.99),
    )
    st.plotly_chart(fig, width="stretch")


def _render_temp_band_chart(df_weather):
    st.subheader("Average Consumption by Temperature Band")
    df_weather = df_weather.copy()
    df_weather["temp_band"] = pd.cut(
        df_weather["temperature_c"],
        bins=[-10, 5, 10, 15, 20, 25, 30, 50],
        labels=["<5°C", "5-10°C", "10-15°C", "15-20°C", "20-25°C", "25-30°C", ">30°C"],
    )
    temp_band_agg = (
        df_weather.groupby("temp_band", observed=True)["kWh"]
        .agg(avg_kWh="mean", count="count")
        .reset_index()
    )
    st.plotly_chart(
        px.bar(temp_band_agg, x="temp_band", y="avg_kWh",
               text=temp_band_agg["count"].apply(lambda n: f"n={n}"),
               title="Avg Hourly Consumption by Temperature Band",
               labels={"temp_band": "Temperature Band", "avg_kWh": "Avg kWh"},
               color="avg_kWh", color_continuous_scale="RdYlBu_r"),
        width="stretch",
    )
