import streamlit as st
import pandas as pd
import plotly.express as px


# Simple view: metrics + daily trend + avg heatmap, with plain-language captions.
def render_simple_overview(daily_totals, date_col, daily_value_col, df_clean,
                           consumption_col, safe_mean, safe_max, WEEKDAY_ORDER):
    st.header("Your Electricity at a Glance")
    st.markdown("For the best results, make sure you have at least 30 weeks of measurements for the most accurate insights. " \
    "The charts below will update as you add more data.")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        total_kwh = pd.to_numeric(daily_totals[daily_value_col], errors="coerce").sum()
        st.metric("Total used", f"{total_kwh:,.1f} kWh")
    with col2:
        st.metric("Days tracked", len(daily_totals))
    with col3:
        st.metric("Typical day", f"{safe_mean(daily_totals[daily_value_col]):.1f} kWh")
    with col4:
        st.metric("Busiest day", f"{safe_max(daily_totals[daily_value_col]):.1f} kWh")

    st.plotly_chart(
        px.line(daily_totals, x=date_col, y=daily_value_col,
                title="Daily electricity use over time",
                labels={daily_value_col: "kWh", date_col: "Date"}),
        width="stretch",
    )
    st.caption(
        "Each point is one day's total electricity use. "
        "Peaks show days you used noticeably more than usual."
    )

    st.divider()
    st.subheader("When do you use the most electricity?")
    st.markdown(
        "The chart below shows your **average electricity use by hour of the day and day of the week**. "
        "Darker red means that time slot is typically busier — "
        "you're running more appliances or heating/cooling."
    )

    if {"hour", "weekday"}.issubset(df_clean.columns):
        pivot_mean = df_clean.pivot_table(
            index="weekday", columns="hour", values=consumption_col, aggfunc="mean"
        ).reindex([d for d in WEEKDAY_ORDER if d in df_clean["weekday"].unique()])

        st.plotly_chart(
            px.imshow(
                pivot_mean,
                title="Average electricity use — day of week vs. hour of day",
                color_continuous_scale="RdYlGn_r",
                labels={"color": "Avg kWh", "x": "Hour of day", "y": "Day of week"},
                aspect="auto",
            ),
            width="stretch",
        )
        st.caption(
            "**How to read this:** each cell is one combination of hour and day. "
            "Dark red = you consistently use a lot of electricity at that time. "
            "Green = your quietest slots. "
            "Use this to shift high-energy tasks (laundry, dishwasher) to cheaper off-peak hours."
        )
    else:
        st.info("Hour and weekday columns not found — run the pipeline to generate them.")


def render_overview(daily_totals, date_col, daily_value_col, df_clean,
                    consumption_col, safe_mean, safe_max):
    st.header("Overview")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        total_kwh = pd.to_numeric(daily_totals[daily_value_col], errors="coerce").sum()
        st.metric("Total Consumption", f"{total_kwh:,.1f} kWh")
    with col2:
        st.metric("Days Analyzed", len(daily_totals))
    with col3:
        st.metric("Avg Daily", f"{safe_mean(daily_totals[daily_value_col]):.1f} kWh")
    with col4:
        st.metric("Peak Day", f"{safe_max(daily_totals[daily_value_col]):.1f} kWh")

    st.plotly_chart(
        px.line(daily_totals, x=date_col, y=daily_value_col, title="Daily Consumption Trend"),
        width="stretch",
    )
