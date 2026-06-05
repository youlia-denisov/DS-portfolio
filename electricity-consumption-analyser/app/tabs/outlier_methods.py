"""
outlier_methods.py
------------------
Showcase tab: presents three outlier/anomaly detection approaches side by side.

The goal here is to show what each method finds on this dataset, not to declare
a winner. With ~13 weeks of data we can describe patterns, but we don't have
enough observations to confidently say one method is more "correct" than another.

Methods shown:
  1. 3-Sigma  — flags readings more than 3 standard deviations from the mean.
               Works best when data is roughly normally distributed.
  2. IQR      — flags readings outside Q1 - 1.5*IQR / Q3 + 1.5*IQR fences.
               More robust to skewed distributions.
  3. DBSCAN   — density-based clustering; points that don't belong to any dense
               region are labelled as noise. Captures spatial outliers that the
               other two methods may miss or over-flag.
  4. Silhouette / Elbow — not an outlier detector, but shows how well the
               K-Means clusters are separated, which helps interpret whether
               DBSCAN noise points overlap with poorly-clustered K-Means readings.

Pre-generated figures (from kmeans_silhouette.py and dbscan_analysis.py) are
loaded from FIGURE_DIR if they exist; a friendly message is shown if they don't.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path


def render_outlier_methods(df_clean, consumption_col, figure_dir: Path):
    st.header("Outlier Detection — Method Comparison")

    st.markdown(
        "Three different approaches were applied to flag unusual electricity readings. "
        "Each method uses a different definition of 'unusual', so they don't always agree — "
        "and that's expected. "
        "The table at the bottom summarises what each one found; the sections below explain how to read the results."
    )
    st.info(
        "**Note on dataset size:** with roughly 13 weeks of data, these results describe "
        "patterns in this specific period. They're a useful starting point, "
        "but conclusions about which method is 'better' would need more data to support."
    )

    kwh = pd.to_numeric(df_clean[consumption_col], errors="coerce").dropna()
    total = len(kwh)

    # 3-Sigma bounds
    mean_val = kwh.mean()
    std_val = kwh.std()
    sigma_lower = mean_val - 3 * std_val
    sigma_upper = mean_val + 3 * std_val
    sigma_mask = (kwh < sigma_lower) | (kwh > sigma_upper)
    n_sigma = sigma_mask.sum()

    # IQR bounds
    q1 = kwh.quantile(0.25)
    q3 = kwh.quantile(0.75)
    iqr = q3 - q1
    iqr_lower = q1 - 1.5 * iqr
    iqr_upper = q3 + 1.5 * iqr
    iqr_mask = (kwh < iqr_lower) | (kwh > iqr_upper)
    n_iqr = iqr_mask.sum()

    skewness = kwh.skew()

    # Summary comparison table at the top so the reader knows what's coming
    summary = pd.DataFrame({
        "Method": ["3-Sigma", "IQR", "DBSCAN (noise pts)"],
        "Flagged readings": [n_sigma, n_iqr, "— run pipeline to see"],
        "% of total": [
            f"{100 * n_sigma / total:.1f}%",
            f"{100 * n_iqr / total:.1f}%",
            "—",
        ],
        "Assumption": [
            "Data is roughly normal",
            "Works on skewed data too",
            "Density-based; no distribution assumed",
        ],
    })
    st.dataframe(summary, hide_index=True, width="stretch")

    st.divider()

    # ── METHOD 1: 3-SIGMA ──────────────────────────────────────────────────
    st.subheader("Method 1 — 3-Sigma (Z-score threshold)")
    st.markdown(
        "This method marks a reading as unusual if it falls more than **3 standard deviations** "
        "from the mean. In a perfectly normal distribution, about 0.3% of values would be flagged. "
        f"This dataset has a skewness of **{skewness:.2f}**, "
        + (
            "which is close to symmetric — 3-sigma should be reasonably reliable here."
            if abs(skewness) < 0.5
            else "which is moderately skewed — the mean and standard deviation are pulled toward "
            "the extreme values, so 3-sigma may flag more readings than expected."
            if abs(skewness) < 1.0
            else "which is strongly skewed — the 3-sigma bounds can be distorted by the long tail, "
            "so IQR is likely to give a more stable result on this data."
        )
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("Mean", f"{mean_val:.3f} kWh")
    col2.metric("Upper fence (mean + 3σ)", f"{sigma_upper:.3f} kWh")
    col3.metric("Flagged", f"{n_sigma} readings ({100 * n_sigma / total:.1f}%)")

    # Distribution with sigma fences
    fig_sigma = go.Figure()
    fig_sigma.add_trace(go.Histogram(
        x=kwh, nbinsx=60, name="All readings",
        marker_color="#4a90d9", opacity=0.7,
    ))
    for x, label, color in [
        (sigma_lower, "−3σ", "orange"),
        (sigma_upper, "+3σ", "red"),
        (mean_val, "Mean", "white"),
    ]:
        fig_sigma.add_vline(x=x, line_color=color, line_dash="dash",
                            annotation_text=label, annotation_font_color=color)
    fig_sigma.update_layout(
        title="Distribution of readings with 3-sigma fences",
        xaxis_title="kWh per 15 min",
        yaxis_title="Count",
        showlegend=False,
    )
    st.plotly_chart(fig_sigma, width="stretch")
    st.caption(
        "Readings to the right of the red dashed line (or left of the orange line) "
        "are flagged as 3-sigma outliers. The long right tail visible here is typical of "
        "electricity data — occasional high-consumption events stretch the distribution."
    )

    st.divider()

    # ── METHOD 2: IQR ──────────────────────────────────────────────────────
    st.subheader("Method 2 — IQR (Interquartile Range)")
    st.markdown(
        "Instead of using the mean, IQR looks at the **middle 50% of values** (Q1 to Q3) "
        "and flags anything that sits far outside that range. "
        "Because the median and IQR are not influenced by extreme values, "
        "this method is generally more robust when data is skewed."
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Q1", f"{q1:.3f} kWh")
    col2.metric("Q3", f"{q3:.3f} kWh")
    col3.metric("Upper fence (Q3 + 1.5×IQR)", f"{iqr_upper:.3f} kWh")
    col4.metric("Flagged", f"{n_iqr} readings ({100 * n_iqr / total:.1f}%)")

    # Box plot to visualise the fences intuitively
    fig_box = go.Figure()
    fig_box.add_trace(go.Box(
        y=kwh, name="kWh readings",
        marker_color="#6bda9a",
        boxpoints="outliers",
        jitter=0.3,
    ))
    fig_box.update_layout(
        title="Box plot — IQR method (dots above/below the whiskers are flagged outliers)",
        yaxis_title="kWh per 15 min",
    )
    st.plotly_chart(fig_box, width="stretch")
    st.caption(
        "The box spans Q1 to Q3. The whiskers extend to Q1 − 1.5×IQR and Q3 + 1.5×IQR. "
        "Individual dots beyond the whiskers are the flagged readings."
    )

    # Where do the two methods disagree?
    agree_both = (sigma_mask & iqr_mask).sum()
    only_sigma = (sigma_mask & ~iqr_mask).sum()
    only_iqr = (~sigma_mask & iqr_mask).sum()

    st.markdown(
        f"**Where the two methods agree and disagree:** "
        f"{agree_both} readings are flagged by both methods. "
        f"{only_sigma} readings are flagged only by 3-sigma, "
        f"and {only_iqr} only by IQR. "
        "The disagreements are worth examining — they often reveal readings that sit "
        "in the moderate tail of the distribution, where the 'correct' label is genuinely ambiguous."
    )

    st.divider()

    # ── METHOD 3: DBSCAN ──────────────────────────────────────────────────
    st.subheader("Method 3 — DBSCAN (Density-Based Clustering)")
    st.markdown(
        "DBSCAN doesn't use statistical thresholds. Instead it finds **dense regions** in the "
        "feature space (time of day, day of week, daily totals, etc.) and marks points that "
        "don't belong to any dense region as **noise**. "
        "These noise points are a DBSCAN's equivalent of outliers, but they're identified "
        "by *context* (what other nearby readings look like) rather than by distance from the mean."
    )

    # Load saved DBSCAN figures if the pipeline has been run
    dbscan_eps_fig = figure_dir / "dbscan_1_eps_selection.png"
    dbscan_diversity_fig = figure_dir / "dbscan_2_diversity_map.png"
    dbscan_noise_fig = figure_dir / "dbscan_3_noise_heatmap.png"
    dbscan_dominant_fig = figure_dir / "dbscan_4_dominant_map.png"

    if dbscan_eps_fig.exists():
        col1, col2 = st.columns(2)
        with col1:
            st.image(str(dbscan_eps_fig),
                     caption="k-distance plot used to choose the eps parameter. "
                              "The 'elbow' of the curve suggests a good eps value — "
                              "below that point most readings are in dense clusters; "
                              "above it the algorithm starts treating normal readings as noise.")
        with col2:
            if dbscan_diversity_fig.exists():
                st.image(str(dbscan_diversity_fig),
                         caption="Cluster diversity per hour × day. "
                                  "Each cell shows how many distinct DBSCAN labels appeared "
                                  "at that slot across all weeks. "
                                  "1 = very stable behaviour; higher = that slot is ambiguous "
                                  "or changes character from week to week.")
    else:
        st.info(
            "DBSCAN figures not found. Run `python src/dbscan_analysis.py` from the project root "
            "to generate them, then refresh this page."
        )

    if dbscan_noise_fig.exists():
        st.image(str(dbscan_noise_fig),
                 caption="Noise point frequency by hour × day. "
                          "Darker red = more readings at that slot were labelled noise. "
                          "Concentration in specific hours may indicate behavioural anomalies "
                          "or weather-driven events — but with 13 weeks of data "
                          "we can describe where noise concentrates, not why.")

    if dbscan_dominant_fig.exists():
        with st.expander("Reference: dominant cluster map"):
            st.image(str(dbscan_dominant_fig),
                     caption="Dominant label per slot (most common cluster across weeks). "
                              "Useful only when no single cluster dominates the whole dataset. "
                              "The title shows the dominance % of the largest cluster — "
                              "if that's above ~80%, this map adds little information.")
    elif not dbscan_eps_fig.exists():
        pass  # already shown the info box above

    st.divider()

    # ── SILHOUETTE / ELBOW (K-Means quality context) ──────────────────────
    st.subheader("Context: K-Means Cluster Quality (Silhouette & Elbow)")
    st.markdown(
        "The silhouette score and elbow plot don't detect outliers directly, but they tell you "
        "how well-separated the K-Means clusters are. "
        "A low silhouette score for a cluster means its readings sit close to the boundary "
        "with another cluster — those boundary readings are the ones most likely to be "
        "flagged differently by DBSCAN versus the statistical methods."
    )

    sil_fig = figure_dir / "2_knife_plot.png"
    elbow_fig = figure_dir / "1_choosing_k.png"
    centroid_fig = figure_dir / "3_centroid_heatmap.png"

    if elbow_fig.exists():
        st.image(str(elbow_fig),
                 caption="Left: elbow curve — inertia drops steeply then levels off; "
                          "the 'elbow' suggests a good k. "
                          "Right: silhouette score per k — higher is better separated. "
                          "These two plots together guided the choice of k=4.")
    else:
        st.info(
            "Silhouette/elbow figures not found. "
            "Run `python src/kmeans_silhouette.py` to generate them."
        )

    if sil_fig.exists():
        st.image(str(sil_fig),
                 caption="Per-sample silhouette ('knife') plot for the chosen k. "
                          "Each horizontal bar is one reading; width = its silhouette score. "
                          "Bars to the left of 0 (negative scores) are likely in the wrong cluster. "
                          "Wide, uniform knives = well-separated clusters; "
                          "thin or ragged knives = clusters overlap.")

    if centroid_fig.exists():
        st.image(str(centroid_fig),
                 caption="Centroid heatmap — what distinguishes each cluster in feature space. "
                          "Red = that cluster scores high on this feature relative to others; "
                          "blue = low. Use this to interpret what 'Cluster 2' actually means "
                          "in real electricity terms.")
