import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
import sys

st.set_page_config(page_title="Electricity Dashboard", page_icon="⚡", layout="wide")
st.title("⚡ Electricity Consumption Analysis Dashboard")
st.markdown("### Smart analysis of your household electricity usage")
st.markdown("**In general, it is recommended to use at least 30 weeks of data for the best statistical analysis.**")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from pipeline import run_pipeline

try:
    from config import PROCESSED_DIR, TABLE_DIR, WEEKDAY_ORDER, TARIFF
except ImportError:
    PROCESSED_DIR = Path("data/processed")
    TABLE_DIR = Path("data/tables")
    HTML_DIR = Path("data/html")
    REPORT_DIR = Path("data/reports")
    WEEKDAY_ORDER = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
    TARIFF = 0.666

from src.discount_analysis import _hours_from_restriction, extract_weekdays, add_offer_eligibility

# SIDEBAR
with st.sidebar:
    st.header("Controls")

    if st.button("Run Full Pipeline"):
        with st.spinner("Running full analysis..."):
            try:
                run_pipeline(run_weather=False)
                st.success("Pipeline completed!")
                st.rerun()
            except Exception as e:
                st.error(f"Pipeline error: {e}")
    st.info("First time? Click **Run Full Pipeline** above.")

    st.divider()
    st.subheader("Settings")

    # Tariff override — used in Discounts and Calculator tabs
    sidebar_tariff = st.number_input(
        "Electricity tariff (₪/kWh)",
        min_value=0.10,
        max_value=4.00,
        value=float(TARIFF),
        step=0.01,
        format="%.3f",
        help="Default is taken from config.py. Change here to model a different rate.",
    )

    # Smart meter selection — passed to Discounts and Calculator
    sidebar_smart_meter = st.radio(
        "Do you have a smart meter?",
        ["Unknown", "Yes", "No"],
        index=0,
        help="Smart meters are required for time-of-use plans. 'Unknown' shows all plans.",
        horizontal=True,
    )
    sidebar_has_sm = {"Yes": True, "No": False, "Unknown": None}[sidebar_smart_meter]

    # Customer type — filters plans shown in the Calculator tab
    st.divider()
    st.subheader("Customer type")
    _offers_path = ROOT / "data" / "external" / "electricity_discount_offers.csv"
    _all_customer_types = ["All"]
    if _offers_path.exists():
        _ct = pd.read_csv(_offers_path)["customer_type"].dropna().unique().tolist()
        _all_customer_types = sorted(set(_ct))
    sidebar_customer_types = st.multiselect(
        "Which offers apply to you?",
        options=_all_customer_types,
        default=["All"],
        help=(
            "Plans marked 'All' are always included. "
            "Select additional types if you qualify (e.g. you are a Cellcom subscriber)."
        ),
    )
    # Always include "All" so universal plans are never hidden
    if "All" not in sidebar_customer_types:
        sidebar_customer_types = ["All"] + sidebar_customer_types

    st.divider()
    # Mode toggle: Simple hides analyst-focused tabs and uses friendlier labels.
    view_mode = st.radio(
        "View mode",
        ["Simple", "Analyst"],
        index=0,
        horizontal=True,
        help="Simple: overview, discounts, and savings calculator. Analyst: all tabs.",
    )

# DATA LOADERS
@st.cache_data
def load_data():
    try:
        df_clean = pd.read_csv(PROCESSED_DIR / "cleaned_consumption.csv")
        hourly = pd.read_csv(PROCESSED_DIR / "weekly_hourly_stats.csv")
        daily = pd.read_csv(PROCESSED_DIR / "daily_stats.csv")
        daily_totals = pd.read_csv(PROCESSED_DIR / "daily_totals.csv")
        scenarios = pd.read_csv(TABLE_DIR / "discount_scenarios.csv")
        return df_clean, hourly, daily, daily_totals, scenarios
    except Exception as e:
        st.error("Data files not found. Please run the pipeline first!")
        st.info(f"Error: {e}")
        return None, None, None, None, None

@st.cache_data
def load_weather_data():
    path = PROCESSED_DIR / "consumption_with_weather.csv"
    return pd.read_csv(path) if path.exists() else None

@st.cache_data
def load_clustering_data():
    c = PROCESSED_DIR / "cleaned_consumption_clustered.csv"
    s = PROCESSED_DIR / "cluster_rank_summary.csv"
    return (pd.read_csv(c) if c.exists() else None), (pd.read_csv(s) if s.exists() else None)

@st.cache_data
def load_report():
    p = ROOT / "reports" / "summary_report.md"
    return p.read_text(encoding="utf-8") if p.exists() else None

df_clean, hourly, daily, daily_totals, scenarios = load_data()

# Show dataset summary in sidebar once data is loaded
if df_clean is not None and "date" in df_clean.columns:
    with st.sidebar:
        st.divider()
        st.subheader("Dataset")
        dates = pd.to_datetime(df_clean["date"], errors="coerce").dropna()
        date_range = f"{dates.min().date()} to {dates.max().date()}"
        st.caption(
            f"Dates: **{date_range}**  \n"
            f"{len(df_clean):,} readings over {dates.dt.date.nunique()} days"
        )

if df_clean is None:
    st.markdown("---")
    st.subheader("Welcome! Let's get started.")
    st.markdown(
        "It looks like this is your first time here, or the data files haven't been generated yet. "
        "Here's what to do:\n\n"
        "**Step 1** — Click **Run Full Pipeline** in the left sidebar. "
        "This processes your raw electricity data and creates all the files the dashboard needs. "
        "It only takes a minute.\n\n"
        "**Step 2** — Once it finishes, the dashboard will reload automatically and all tabs will be available."
    )
    st.info("The **Run Full Pipeline** button is in the sidebar on the left.")
    st.stop()

def safe_mean(s):
    return pd.to_numeric(s, errors="coerce").mean()

def safe_max(s):
    return pd.to_numeric(s, errors="coerce").max()

consumption_col = next(
    (c for c in df_clean.columns if any(w in c.lower() for w in ["kwh", "kwatt", "consumption"])),
    df_clean.columns[1],
)
date_col = next((c for c in daily_totals.columns if "date" in c.lower()), daily_totals.columns[0])  # type: ignore
daily_value_col = next(
    (c for c in daily_totals.columns if any(w in c.lower() for w in ["kwh", "kwatt", "daily"])),  # type: ignore
    daily_totals.columns[1],  # type: ignore
)

from tabs import (
    render_overview, render_simple_overview, render_hourly, render_trends,
    render_clustering, render_discounts, render_calculator, render_weather,
    render_behavior_profile, render_report, render_about, render_outlier_methods,
)

_offers_csv = pd.read_csv(PROCESSED_DIR.parent / "external" / "electricity_discount_offers.csv")

if view_mode == "Simple":
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Overview", "Best deals", "Calculate my savings", "Report", "About",
    ])

    with tab1:
        render_simple_overview(daily_totals, date_col, daily_value_col, df_clean,
                               consumption_col, safe_mean, safe_max, WEEKDAY_ORDER)
    with tab2:
        render_discounts(scenarios, PROCESSED_DIR, WEEKDAY_ORDER, sidebar_tariff,
                         add_offer_eligibility, extract_weekdays, _hours_from_restriction)
    with tab3:
        render_calculator(df_clean, _offers_csv, sidebar_tariff,
                          has_smart_meter=sidebar_has_sm,
                          customer_types=sidebar_customer_types)
    with tab4:
        render_report(load_report, ROOT, simple=True)
    with tab5:
        render_about(df_clean, daily_totals, hourly)

else:
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10, tab11 = st.tabs([
        "Overview", "Hourly Patterns", "Behaviour Profile", "Trends & Outliers",
        "Outlier Methods", "Clustering", "Weather", "Discounts",
        "Savings Calculator", "Report", "About",
    ])

    with tab1:
        render_overview(daily_totals, date_col, daily_value_col, df_clean, consumption_col,
                        safe_mean, safe_max)
    with tab2:
        render_hourly(df_clean, hourly, consumption_col, WEEKDAY_ORDER)
    with tab3:
        render_behavior_profile(df_clean)
    with tab4:
        render_trends(df_clean, consumption_col)
    with tab5:
        from config import FIGURE_DIR
        render_outlier_methods(df_clean, consumption_col, FIGURE_DIR)
    with tab6:
        render_clustering(load_clustering_data, WEEKDAY_ORDER)
    with tab7:
        render_weather(load_weather_data)
    with tab8:
        render_discounts(scenarios, PROCESSED_DIR, WEEKDAY_ORDER, sidebar_tariff,
                         add_offer_eligibility, extract_weekdays, _hours_from_restriction)
    with tab9:
        render_calculator(df_clean, _offers_csv, sidebar_tariff,
                          has_smart_meter=sidebar_has_sm,
                          customer_types=sidebar_customer_types)
    with tab10:
        render_report(load_report, ROOT)
    with tab11:
        render_about(df_clean, daily_totals, hourly)
