"""
Tab rendering functions for the Electricity Dashboard.
Split into a separate module to keep file sizes manageable.
"""
import re
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go


CLUSTER_PALETTE = {0: "#2ca25f", 1: "#fee08b", 2: "#fdae61", 3: "#d73027"}
CLUSTER_LABELS = {0: "Low use", 1: "Medium-low", 2: "Medium-high", 3: "High use"}

# Shared CSS: larger explanation text, applied once per session via st.markdown
_EXPLANATION_CSS = """
<style>
/* Make inline explanatory paragraphs more readable */
section[data-testid="stMain"] .stMarkdown p,
section[data-testid="stMain"] .stCaptionContainer p {
    font-size: 1.05rem !important;
    line-height: 1.75 !important;
}
</style>
"""


# OVERVIEW 
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


#  HOURLY PATTERNS 
def render_hourly(df_clean, hourly, consumption_col, WEEKDAY_ORDER):
    st.markdown(_EXPLANATION_CSS, unsafe_allow_html=True)
    st.header("Hourly Consumption Patterns")

    if {"hour", "weekday"}.issubset(df_clean.columns):
        col1, col2 = st.columns(2)

        pivot_mean = df_clean.pivot_table(
            index="weekday", columns="hour", values=consumption_col, aggfunc="mean"
        ).reindex([d for d in WEEKDAY_ORDER if d in df_clean["weekday"].unique()])

        pivot_std = df_clean.pivot_table(
            index="weekday", columns="hour", values=consumption_col, aggfunc="std"
        ).reindex([d for d in WEEKDAY_ORDER if d in df_clean["weekday"].unique()])

        with col1:
            st.plotly_chart(
                px.imshow(pivot_mean, title="Mean Usage by Weekday & Hour",
                          color_continuous_scale="RdYlGn_r", labels={"color": "Avg kWh"}),
                width="stretch",
            )
        with col2:
            st.plotly_chart(
                px.imshow(pivot_std, title="Variability by Weekday & Hour",
                          color_continuous_scale="Blues", labels={"color": "Std kWh"}),
                width="stretch",
            )
        st.caption(
            "**Left chart — Average usage:** darker red = you typically use more electricity at that time. "
            "**Right chart — Variability (standard deviation):** darker blue = your usage at that time varies a lot "
            "from week to week. A pale cell means you're very consistent; a dark cell means some weeks are very "
            "different from others."
        )


# TRENDS & OUTLIERS
def render_trends(df_clean, consumption_col):
    st.header("Trends & Outliers")

    col1, col2 = st.columns(2)
    with col1:
        daily_sum = df_clean.groupby("date")[consumption_col].sum().reset_index(name="daily_kWh")
        daily_sum["rolling_7d"] = daily_sum["daily_kWh"].rolling(7, min_periods=1).mean()
        st.plotly_chart(
            px.line(daily_sum, x="date", y=["daily_kWh", "rolling_7d"],
                    title="Daily Consumption with 7-Day Rolling Average",
                    labels={"daily_kWh": "Daily Total (kWh)", "rolling_7d": "7-Day Rolling Avg"}),
            width="stretch",
        )
    with col2:
        sorted_data = (
            df_clean[[consumption_col]]
            .sort_values(consumption_col, ascending=False)
            .reset_index(drop=True)
        )
        sorted_data["percentile"] = (sorted_data.index + 1) / len(sorted_data) * 100
        st.plotly_chart(
            px.line(sorted_data, x="percentile", y=consumption_col, title="Load Duration Curve"),
            width="stretch",
        )

    st.caption(
        "**Left — 7-day rolling average:** instead of showing every noisy day individually, "
        "this line averages the last 7 days together as it moves forward in time, smoothing out "
        "one-off spikes so you can see the real long-term trend. "
        "**Right — Load Duration Curve:** all hourly readings sorted from highest to lowest. "
        "The left side shows your peak usage moments (e.g. top 5% of hours), the right your quietest. "
        "A steep drop means a few hours dominate your bill; a flat curve means usage is spread evenly."
    )

    st.subheader("Unusual Readings — IQR Method")

    kwh = pd.to_numeric(df_clean[consumption_col], errors="coerce").dropna()
    q1, q3 = kwh.quantile(0.25), kwh.quantile(0.75)
    iqr = q3 - q1
    upper_fence = q3 + 1.5 * iqr
    lower_fence = q1 - 1.5 * iqr

    outlier_mask = (kwh < lower_fence) | (kwh > upper_fence)
    df_out = df_clean.loc[outlier_mask.index[outlier_mask]].copy()
    total_readings = len(kwh)

    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        st.metric("Q1 — lower typical", f"{q1:.3f} kWh", help="25% of your readings are below this value")
    with m2:
        st.metric("Q3 — upper typical", f"{q3:.3f} kWh", help="75% of your readings are below this value")
    with m3:
        st.metric("Typical range (IQR)", f"{iqr:.3f} kWh", help="The spread of your middle 50% of readings")
    with m4:
        st.metric("Unusual if above", f"{upper_fence:.3f} kWh", help="Readings above this are flagged as unusually high")
    with m5:
        st.metric(
            "Unusual readings",
            f"{len(df_out)} of {total_readings:,}",
            help=f"{len(df_out) / total_readings * 100:.1f}% of all readings fall outside the IQR fences",
        )

    st.caption(
        "The **IQR (interquartile range)** method looks at the middle 50% of your readings "
        "and flags anything that falls far outside that range — "
        "think of it as: *'this hour looks nothing like a normal hour for you.'*"
    )

    if "weekday" in df_out.columns:
        by_day = (
            df_out.groupby("weekday")[consumption_col]
            .agg(count="count", mean_kWh="mean", max_kWh="max")
            .reset_index()
        )
        st.plotly_chart(
            px.bar(by_day, x="weekday", y="count", color="mean_kWh",
                   color_continuous_scale="Reds", title="IQR Outlier Count by Weekday",
                   labels={"count": "# Outliers", "mean_kWh": "Avg kWh"}),
            width="stretch",
        )
        st.caption(
            "Each bar shows how many flagged readings fall on that weekday. "
            "Color encodes the average kWh of those outliers — darker red means the unusual readings "
            "on that day tend to be higher-consumption events."
        )


# CLUSTERING
def render_clustering(load_clustering_data, WEEKDAY_ORDER):
    st.header("Consumption Clustering")
    st.markdown(
        "This analysis automatically groups all your hourly readings into 4 usage profiles, "
        "without being told what to look for. Each reading gets assigned to the profile it resembles most. "
        "The profiles are then ranked by how much electricity they use: "
        "**0 = your quietest hours** through to **3 = your heaviest-use hours**. "
        "The heatmap below shows which profile dominates at each hour of each day of the week."
    )

    df_clustered, cluster_summary = load_clustering_data()

    if df_clustered is None:
        st.warning(
            "Usage profiling hasn't been run yet. "
            "To unlock this tab, open a terminal in the project folder and run: "
            "`python clustering_with_visuals.py` — then come back and refresh the page."
        )
        with st.expander("Show me the exact command"):
            st.code("python clustering_with_visuals.py", language="bash")
        return

    c_col = next(
        (col for col in df_clustered.columns if any(w in col.lower() for w in ["kwh", "kwatt", "consumption"])),
        df_clustered.columns[0],
    )
    rank_col = "cluster_rank" if "cluster_rank" in df_clustered.columns else "cluster"
    df_clustered["_label"] = df_clustered[rank_col].map(CLUSTER_LABELS)

    col1, col2 = st.columns(2)
    with col1:
        fig_box = px.box(df_clustered, x=rank_col, y=c_col, color=rank_col,
                         color_discrete_map=CLUSTER_PALETTE,
                         title="Consumption Distribution by Cluster",
                         labels={rank_col: "Cluster (0=low to 3=high)", c_col: "kWh"},
                         category_orders={rank_col: [0, 1, 2, 3]})
        fig_box.update_layout(showlegend=False)
        st.plotly_chart(fig_box, width="stretch")

    with col2:
        size_df = df_clustered[rank_col].value_counts().sort_index().reset_index()
        size_df.columns = ["cluster_rank", "count"]
        fig_size = px.bar(size_df, x="cluster_rank", y="count", color="cluster_rank",
                          color_discrete_map=CLUSTER_PALETTE, title="Records per Cluster",
                          text="count")
        fig_size.update_layout(showlegend=False)
        st.plotly_chart(fig_size, width="stretch")

    if {"weekday", "hour"}.issubset(df_clustered.columns):
        st.subheader("Dominant Cluster by Weekday & Hour")
        dominant = (
            df_clustered.groupby(["weekday", "hour"])[rank_col]
            .agg(lambda x: x.mode().iloc[0])
            .unstack()
        )
        dominant = dominant.reindex([d for d in WEEKDAY_ORDER if d in dominant.index])
        colorscale = [
            [0.0, CLUSTER_PALETTE[0]], [0.33, CLUSTER_PALETTE[1]],
            [0.66, CLUSTER_PALETTE[2]], [1.0,  CLUSTER_PALETTE[3]],
        ]
        fig_heat = go.Figure(go.Heatmap(
            z=dominant.values, x=dominant.columns.tolist(), y=dominant.index.tolist(),
            colorscale=colorscale, zmin=0, zmax=3,
            colorbar=dict(title="Cluster rank", tickvals=[0, 1, 2, 3],
                          ticktext=["0 Low", "1 Med-low", "2 Med-high", "3 High"]),
        ))
        fig_heat.update_layout(title="Dominant Cluster — Weekday x Hour",
                               xaxis_title="Hour", yaxis_title="Weekday", height=380)
        st.plotly_chart(fig_heat, width="stretch")

    st.subheader("Cluster Summary Statistics")
    if cluster_summary is not None:
        cols = [col for col in cluster_summary.columns if col != "cluster"]
        st.dataframe(cluster_summary[cols].round(3), width="stretch")
    else:
        on_the_fly = (
            df_clustered.groupby(rank_col)[c_col]
            .agg(avg_kWh="mean", median_kWh="median", min_kWh="min",
                 max_kWh="max", std_kWh="std", count="count")
            .reset_index().round(3)
        )
        on_the_fly["label"] = on_the_fly[rank_col].map(CLUSTER_LABELS)
        st.dataframe(on_the_fly, width="stretch")


# DISCOUNTS
def render_discounts(scenarios, PROCESSED_DIR, WEEKDAY_ORDER, TARIFF,
                     add_offer_eligibility, extract_weekdays, _hours_from_restriction):
    st.header("Discount Recommendations")

    if scenarios is None or scenarios.empty:
        st.warning("No discount scenarios available yet.")
        return

    rank_cols = ["supplier_name", "plan_name", "discount_max_pct", "time_restriction",
                 "requires_smart_meter", "eligibility", "matching_usage_share_pct",
                 "weighted_discount_score"]
    display_cols = [c for c in rank_cols if c in scenarios.columns]
    st.dataframe(scenarios[display_cols].round(2), width="stretch")

    st.divider()
    st.subheader("Consumption vs. Plan Tariff Window")
    st.markdown(
        "Select a plan to see your consumption profile next to the discount window. "
        "Green = discounted rate, red = full rate."
    )

    stats_path = PROCESSED_DIR / "weekly_hourly_stats.csv"
    if not stats_path.exists():
        st.warning("Run the pipeline first to generate `weekly_hourly_stats.csv`.")
        return

    stats_df = pd.read_csv(stats_path)
    consumption_matrix = (
        stats_df.pivot(index="weekday", columns="hour", values="avg_kWh")
        .reindex([d for d in WEEKDAY_ORDER if d in stats_df["weekday"].unique()])
    )

    eligible = add_offer_eligibility(scenarios.copy(), has_smart_meter=None)
    plan_options = (
        eligible[["supplier_name", "plan_name", "discount_min_pct", "time_restriction"]]
        .drop_duplicates().reset_index(drop=True)
    )
    plan_labels = [f"{r.supplier_name} — {r.plan_name}" for r in plan_options.itertuples()]

    selected_label = st.selectbox("Choose a plan", plan_labels)
    sel = plan_options.iloc[plan_labels.index(selected_label)]
    st.info(
        "💡 **What to look for:** the left chart shows when you use the most electricity (darker orange = more usage). "
        "The right chart shows when this plan's discount applies (green cells = cheaper rate, red = full price). "
        "The more your heavy-use hours overlap with the green zone, the more you'll actually save."
    )

    discount_pct = float(sel["discount_min_pct"])
    discounted_rate = round(TARIFF * (1 - discount_pct / 100), 4)
    day_to_idx = {day: i for i, day in enumerate(WEEKDAY_ORDER)}
    price_matrix = np.full((len(WEEKDAY_ORDER), 24), TARIFF)

    for d in extract_weekdays(sel["time_restriction"]):
        for h in _hours_from_restriction(sel["time_restriction"]):
            if d in day_to_idx:
                price_matrix[day_to_idx[d], h] = discounted_rate

    present_weekdays = [d for d in WEEKDAY_ORDER if d in consumption_matrix.index]
    price_df = pd.DataFrame(
        [price_matrix[day_to_idx[d]] for d in present_weekdays],
        index=present_weekdays, columns=list(range(24)),
    )

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(
            px.imshow(consumption_matrix, title="Your Consumption Profile (avg kWh)",
                      color_continuous_scale="Oranges", aspect="auto",
                      labels={"color": "avg kWh", "x": "Hour", "y": "Weekday"}),
            width="stretch",
        )
    with col2:
        st.plotly_chart(
            px.imshow(price_df, title=f"{sel.supplier_name} — {sel.plan_name} (NIS/kWh)",
                      color_continuous_scale="RdYlGn_r", aspect="auto",
                      range_color=[TARIFF * 0.7, TARIFF],
                      labels={"color": "NIS/kWh", "x": "Hour", "y": "Weekday"}),
            width="stretch",
        )


#bCALCULATOR

# These two helpers live at module level so @st.cache_data can manage them across reruns.
# Streamlit hashes DataFrame arguments by content, so the cache is invalidated automatically
# if the underlying data changes (e.g. after re-running the pipeline).

@st.cache_data
def _cached_build_pattern(monthly_kwh, pct_wd_day, pct_wd_evening, pct_wd_night, pct_weekend):
    """Builds the synthetic hourly DataFrame from slider values. Cached because it's called
    on every slider interaction and the result is deterministic."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from discount_calculator import build_custom_pattern_df
    return build_custom_pattern_df(monthly_kwh, pct_wd_day, pct_wd_evening, pct_wd_night, pct_weekend)


@st.cache_data
def _cached_compare_plans(calc_df, offers_df, tariff, has_sm, observation_days):
    """Runs the full plan comparison and annual extrapolation. The cache key includes
    all inputs, so changing the smart-meter toggle or sliders correctly triggers a recalc."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from discount_calculator import compare_all_plans, extrapolate_annual
    results = compare_all_plans(calc_df, offers_df, tariff=tariff, has_smart_meter=has_sm)
    return extrapolate_annual(results, observation_days=observation_days)


def render_calculator(df_clean, offers_df, tariff, has_smart_meter=None):
    """
    Interactive savings calculator with two modes:
      - "My data": uses the actual loaded consumption CSV
      - "Custom pattern": builds a synthetic profile from user sliders
    """
    st.header("Savings Calculator")
    st.markdown(
        "Compare every available plan against your usage — showing real NIS saved, "
        "not just a score. Switch between your actual meter data and a hypothetical pattern."
    )

    mode = st.radio("Usage source", ["My meter data", "Custom pattern"], horizontal=True)

    if mode == "My meter data":
        if df_clean is None or df_clean.empty:
            st.warning("No consumption data loaded. Run the pipeline first.")
            return
        calc_df = df_clean.rename(columns={
            c: "kWh" for c in df_clean.columns if any(w in c.lower() for w in ["kwh", "consumption"])
        })
        observation_days = pd.to_datetime(calc_df["date"]).dt.date.nunique() if "date" in calc_df.columns else 30
        st.caption(f"Using {len(calc_df):,} readings over {observation_days} days.")

    else:
        st.markdown("**Set your typical monthly usage pattern**")
        col1, col2 = st.columns(2)
        with col1:
            monthly_kwh = st.number_input("Monthly consumption (kWh)", min_value=10.0,
                                           max_value=2000.0, value=300.0, step=10.0)
            pct_wd_day     = st.slider("Weekday daytime (07-17)", 0, 100, 30, help="Sun–Thu 07:00–17:00")
            pct_wd_evening = st.slider("Weekday evening (17-23)", 0, 100, 35, help="Sun–Thu 17:00–23:00")
        with col2:
            pct_wd_night   = st.slider("Weekday night (23-07)",   0, 100, 15, help="Sun–Thu 23:00–07:00")
            pct_weekend    = st.slider("Weekend (Fri–Sat)",        0, 100, 20, help="All hours Fri–Sat")

        total_pct = pct_wd_day + pct_wd_evening + pct_wd_night + pct_weekend
        if total_pct == 0:
            st.error("At least one slider must be above 0.")
            return
        if total_pct != 100:
            st.warning(
                f"Your sliders add up to **{total_pct}%**. "
                "They don't need to sum to exactly 100 — the calculator will scale them automatically — "
                "but make sure the proportions reflect how you actually use electricity."
            )
        else:
            st.success("✓ Sliders sum to 100%.")

        calc_df = _cached_build_pattern(monthly_kwh, pct_wd_day, pct_wd_evening,
                                        pct_wd_night, pct_weekend)
        observation_days = 30  # one synthetic month

    # Default to sidebar setting; user can still override here
    sm_default_map = {True: "Yes", False: "No", None: "Unknown"}
    sm_default = sm_default_map.get(has_smart_meter, "Unknown")
    smart_choice = st.selectbox(
        "Do you have a smart meter?",
        ["Unknown", "Yes", "No"],
        index=["Unknown", "Yes", "No"].index(sm_default),
        help="Smart meter required for time-of-use plans. Inherits from sidebar setting.",
    )
    has_sm = {"Yes": True, "No": False, "Unknown": None}[smart_choice]

    if st.button("Calculate savings", type="primary"):
        with st.spinner("Calculating..."):
            results = _cached_compare_plans(calc_df, offers_df, tariff, has_sm, observation_days)

        if results.empty:
            st.warning("No plans to compare.")
            return

        # summary metrics
        best = results.iloc[0]
        eligible = results[results["eligibility"] != "not_eligible_requires_smart_meter"]
        best_eligible = eligible.iloc[0] if not eligible.empty else best

        m1, m2, m3 = st.columns(3)
        m1.metric("Best plan (eligible)", f"{best_eligible['supplier_name']} — {best_eligible['plan_name']}")
        m2.metric("Annual saving", f"₪{best_eligible['annual_nis_saved']:,.0f}")
        m3.metric("Effective discount", f"{best_eligible['effective_discount_pct']:.1f}%")

        st.divider()

        # bar chart
        color_map = {
            "eligible": "#2ecc71",
            "eligible_or_unknown": "#f39c12",
            "unknown_smart_meter_required": "#95a5a6",
            "not_eligible_requires_smart_meter": "#e74c3c",
        }
        results["color"] = results["eligibility"].map(
            lambda e: color_map.get(e, "#aaaaaa")
        )
        results["label"] = results["supplier_name"] + " — " + results["plan_name"]

        fig = go.Figure(go.Bar(
            x=results["annual_nis_saved"],
            y=results["label"],
            orientation="h",
            marker_color=results["color"],
            text=results["annual_nis_saved"].apply(lambda v: f"₪{v:,.0f}"),
            textposition="outside",
            customdata=results[["effective_discount_pct", "matching_usage_share_pct",
                                  "eligibility", "discount_pct"]].values,
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Annual saving: ₪%{x:,.0f}<br>"
                "Advertised discount: %{customdata[3]}%<br>"
                "Effective discount on your usage: %{customdata[0]:.1f}%<br>"
                "Usage in discount window: %{customdata[1]:.1f}%<br>"
                "Eligibility: %{customdata[2]}<extra></extra>"
            ),
        ))
        fig.update_layout(
            title="Projected Annual Savings per Plan",
            xaxis_title="Estimated annual saving (₪)",
            yaxis={"autorange": "reversed", "tickfont": {"size": 11}},
            height=max(400, len(results) * 32),
            margin={"l": 260, "r": 80},
        )
        st.plotly_chart(fig, width="stretch")

        # ---- colour legend ----
        st.caption(
            "🟢 Eligible &nbsp;|&nbsp; 🟡 Eligibility unknown (smart meter unspecified) "
            "&nbsp;|&nbsp; 🔴 Not eligible (requires smart meter you don't have)"
        )

        st.divider()

        # ---- detailed table ----
        st.subheader("Detailed comparison")
        display_cols = [
            "supplier_name", "plan_name", "discount_pct",
            "weekdays_applicable", "hours_applicable",
            "matching_usage_share_pct", "effective_discount_pct",
            "annual_nis_saved", "eligibility",
        ]
        display_cols = [c for c in display_cols if c in results.columns]
        st.dataframe(
            results[display_cols].rename(columns={
                "supplier_name": "Supplier",
                "plan_name": "Plan",
                "discount_pct": "Advertised %",
                "weekdays_applicable": "Days",
                "hours_applicable": "Hours",
                "matching_usage_share_pct": "Usage in window %",
                "effective_discount_pct": "Effective discount %",
                "annual_nis_saved": "Annual saving (₪)",
                "eligibility": "Eligibility",
            }).round(2),
            width="stretch",
        )

        st.caption(
            f"⚠️Savings are extrapolated from {observation_days} days of data to a full year. "
            "Actual savings depend on your annual consumption pattern and any plan fees not "
            "reflected in the discount percentage."
        )


# WEATHER 
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

    corr_temp = df_weather["kWh"].corr(df_weather["temperature_c"])
    corr_hum = df_weather["kWh"].corr(df_weather["humidity_pct"])
    corr_prec = df_weather["kWh"].corr(df_weather["precipitation_mm"])

    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("Corr: kWh vs Temperature", f"{corr_temp:.3f}")
    with m2:
        st.metric("Corr: kWh vs Humidity", f"{corr_hum:.3f}")
    with m3:
        st.metric("Corr: kWh vs Precipitation", f"{corr_prec:.3f}")
    st.caption(
        "**Correlation** measures how closely two things move together, on a scale from -1 to +1. "
        "+1 means they rise and fall perfectly in sync (hotter → always more electricity). "
        "-1 means the opposite (hotter → always less electricity). "
        "0 means no relationship at all. Values around ±0.3 are considered a moderate link; ±0.6 is strong."
    )
    st.divider()

    st.subheader("Daily Trend: Consumption & Temperature")
    daily_weather = (
        df_weather.groupby("date")
        .agg(daily_kWh=("kWh", "sum"), avg_temp=("temperature_c", "mean"))
        .reset_index()
    )
    fig_dual = go.Figure()
    fig_dual.add_trace(go.Scatter(
        x=daily_weather["date"], y=daily_weather["daily_kWh"],
        name="Daily kWh", line=dict(color="#e55c30"), yaxis="y1",
    ))
    fig_dual.add_trace(go.Scatter(
        x=daily_weather["date"], y=daily_weather["avg_temp"],
        name="Avg Temp (C)", line=dict(color="#4a90d9", dash="dot"), yaxis="y2",
    ))
    fig_dual.update_layout(
        title="Daily Consumption vs Average Temperature",
        xaxis_title="Date",
        yaxis=dict(title="Daily kWh", side="left"),
        yaxis2=dict(title="Avg Temperature (C)", side="right", overlaying="y", showgrid=False),
        legend=dict(x=0.01, y=0.99),
    )
    st.plotly_chart(fig_dual, width="stretch")

    st.subheader("Average Consumption by Temperature Band")
    df_weather["temp_band"] = pd.cut(
        df_weather["temperature_c"],
        bins=[-10, 5, 10, 15, 20, 25, 30, 50],
        labels=["<5C", "5-10C", "10-15C", "15-20C", "20-25C", "25-30C", ">30C"],
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


# REPORT 
def render_report(load_report, ROOT):
    st.header("Analysis Report")
    report_text = load_report()

    if report_text is None:
        st.warning("Report not found at `reports/summary_report.md`. Run the pipeline to generate it.")
        return

    clean_report = re.sub(r"^#{1,4} .+\n!\[.*?\]\(figures/.*?\)\n?", "", report_text, flags=re.MULTILINE)
    clean_report = re.sub(r"^!\[.*?\]\(.*?\)\n?", "", clean_report, flags=re.MULTILINE)

    st.markdown("""
        <style>
        .report-box {
            background: #f8f9fa; border-left: 4px solid #4a90d9; border-radius: 6px;
            padding: 2rem 2.5rem; font-size: 0.97rem; line-height: 1.8; color: #1a1a2e;
        }
        .report-box h1 { color: #1a1a2e; font-size: 1.6rem; }
        .report-box h2 { color: #2c3e50; font-size: 1.15rem; margin-top: 1.5rem;
            border-bottom: 1px solid #dee2e6; padding-bottom: 0.25rem; }
        .report-box h3 { color: #4a90d9; font-size: 1rem; }
        .report-box table { width: 100%; border-collapse: collapse; margin: 1rem 0; font-size: 0.87rem; }
        .report-box th { background: #e9ecef; padding: 0.45rem 0.75rem; text-align: left; }
        .report-box td { padding: 0.35rem 0.75rem; border-bottom: 1px solid #e9ecef; }
        .report-box tr:hover td { background: #f1f3f5; }
        </style>""", unsafe_allow_html=True)

    try:
        import markdown as md_lib
        html_content = md_lib.markdown(clean_report, extensions=["tables"])
        st.markdown(f'<div class="report-box">{html_content}</div>', unsafe_allow_html=True)
    except ImportError:
        st.markdown(clean_report)

    html_dir = ROOT / "outputs" / "html"
    existing_htmls = sorted(html_dir.glob("*.html")) if html_dir.exists() else []
    if existing_htmls:
        st.subheader("Generated Visuals")
        st.caption("Only charts that exist on your machine are shown below.")
        for html_file in existing_htmls:
            with st.expander(html_file.stem.replace("_", " ").title()):
                components.html(html_file.read_text(encoding="utf-8"), height=500, scrolling=True)

    with st.expander("View raw markdown"):
        st.code(report_text, language="markdown")

    # Outlier tables referenced in the report
    st.divider()
    st.subheader("Outlier Data")

    outlier_files = {
        "IQR method": ROOT / "data" / "processed" / "outliers_iqr.csv",
        "3-sigma method": ROOT / "data" / "processed" / "outliers_3sigma.csv",
    }

    for label, path in outlier_files.items():
        if not path.exists():
            st.caption(f"{label}: file not found (`{path.name}`).")
            continue

        df_out = pd.read_csv(path)
        # Drop redundant/internal columns for display
        drop_cols = [c for c in ["datetime", "method", "lower_limit", "upper_limit"] if c in df_out.columns]
        display_df = df_out.drop(columns=drop_cols).round(4)

        with st.expander(f"{label} — {len(df_out)} flagged readings", expanded=True):
            st.dataframe(display_df, width="stretch", height=min(400, 40 + len(display_df) * 35))
