"""
outlier_methods.py
------------------
Streamlit tab: runs the comprehensive outlier pipeline live and presents
all four detection methods with a clear auto-selected recommendation.

Methods compared:
  1. 3-Sigma       — mean ± 3σ; works on near-normal distributions
  2. IQR           — quartile fences; robust to skewed data
  3. DBSCAN        — density-based; uses time + consumption features
  4. Isolation Forest — tree-based ML; no distribution assumed

The auto-selector in outlier_pipeline.py picks the most appropriate method
based on skewness and dataset size, and explains why in plain language.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from collections import Counter
import sys

# Make sure src/ is importable when running from the app/ directory
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.outlier_pipeline import run_outlier_pipeline


# ── helpers ────────────────────────────────────────────────────────────────────

def _agreement_venn(results: dict, total: int) -> pd.DataFrame:
    """Build a table showing how many readings each pair of methods both flag."""
    methods = list(results.keys())
    rows = []
    for i, m1 in enumerate(methods):
        for m2 in methods[i + 1:]:
            idx1 = set(results[m1].index)
            idx2 = set(results[m2].index)
            both = len(idx1 & idx2)
            only1 = len(idx1 - idx2)
            only2 = len(idx2 - idx1)
            rows.append({
                "Method A": m1, "Method B": m2,
                "Both flag": both,
                f"Only {m1}": only1,
                f"Only {m2}": only2,
                "Agreement %": round(100 * both / max(len(idx1 | idx2), 1), 1),
            })
    return pd.DataFrame(rows)


def _distribution_fig(kwh: pd.Series, lower: float, upper: float,
                      method_name: str, color: str) -> go.Figure:
    """Histogram with fence lines for a statistical method."""
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=kwh, nbinsx=60, name="All readings",
        marker_color="#4a90d9", opacity=0.7,
    ))
    for x, label, c in [(lower, "Lower fence", "orange"), (upper, "Upper fence", color)]:
        if x > kwh.min():
            fig.add_vline(x=x, line_color=c, line_dash="dash",
                          annotation_text=label, annotation_font_color=c)
    fig.update_layout(
        title=f"{method_name} — distribution with fences",
        xaxis_title="kWh per reading", yaxis_title="Count",
        showlegend=False, margin=dict(t=40, b=30),
    )
    return fig


def _timeline_fig(df: pd.DataFrame, flagged_idx, method_name: str, color: str):
    """Scatter of all readings; flagged ones coloured differently."""
    if "datetime" not in df.columns:
        return None
    df_plot = df[["datetime", "kWh"]].copy()
    df_plot["datetime"] = pd.to_datetime(df_plot["datetime"])
    df_plot["flagged"] = df_plot.index.isin(flagged_idx)
    fig = px.scatter(
        df_plot, x="datetime", y="kWh",
        color="flagged",
        color_discrete_map={False: "#aac4e8", True: color},
        labels={"flagged": "Outlier", "kWh": "kWh", "datetime": ""},
        title=f"{method_name} — flagged readings over time",
        opacity=0.6,
    )
    fig.update_layout(margin=dict(t=40, b=30), showlegend=True)
    return fig


# ── main render function ───────────────────────────────────────────────────────

def render_outlier_methods(df_clean: pd.DataFrame, consumption_col: str,
                           figure_dir: Path):
    st.header("Outlier Detection — Comprehensive Pipeline")
    st.markdown(
        "Four detection methods run automatically on your data. "
        "Each uses a different definition of 'unusual', so they don't always agree — "
        "and that's expected. The pipeline picks the most appropriate method for your "
        "data and explains why."
    )
    st.info(
        "**Dataset note:** with ~13 weeks of data these results describe patterns "
        "in this specific period. They're a useful starting point, but a full year "
        "of data would let us distinguish true anomalies from seasonal variation."
    )

    # ── run the pipeline ───────────────────────────────────────────────────────
    with st.spinner("Running outlier detection…"):
        try:
            res = run_outlier_pipeline(df_clean)
        except Exception as e:
            st.error(f"Pipeline error: {e}")
            return

    # ── recommendation banner ──────────────────────────────────────────────────
    st.subheader("Auto-Selected Method")
    rec_col, why_col = st.columns([1, 3])
    with rec_col:
        st.metric("Recommended", res.recommended)
        st.metric("Skewness", f"{res.stats['skewness']:.2f}")
        st.metric("Total readings", f"{res.stats['n_samples']:,}")
    with why_col:
        st.markdown(f"**Why {res.recommended}?**")
        st.markdown(res.reason)
        st.markdown(
            "_Skewness guide: < 0.5 = near-normal · 0.5–1.5 = moderate skew · > 1.5 = strongly skewed_"
        )

    st.divider()

    # ── summary comparison table ───────────────────────────────────────────────
    st.subheader("All Methods — Summary")
    st.dataframe(res.summary, hide_index=True, use_container_width=True)

    fig_bar = px.bar(
        res.summary, x="Method", y="% of total",
        color="Method", text="Flagged",
        title="Readings flagged as outliers by each method",
        labels={"% of total": "% of readings flagged"},
    )
    fig_bar.update_traces(textposition="outside")
    fig_bar.update_layout(showlegend=False, margin=dict(t=40, b=10))
    st.plotly_chart(fig_bar, use_container_width=True)
    st.caption(
        "Methods flagging very different percentages are worth comparing carefully. "
        "A method flagging >10% of readings on 13 weeks of data is likely over-sensitive."
    )

    st.divider()

    # ── per-method detail ──────────────────────────────────────────────────────
    kwh = pd.to_numeric(df_clean[consumption_col], errors="coerce").dropna()

    # METHOD 1: 3-Sigma ────────────────────────────────────────────────────────
    st.subheader("Method 1 — 3-Sigma (Z-Score Threshold)")
    skew = res.stats["skewness"]
    if abs(skew) < 0.5:
        skew_note = "close to symmetric — 3-sigma should be reasonably reliable here."
    elif abs(skew) < 1.5:
        skew_note = "moderately skewed — IQR may be more reliable."
    else:
        skew_note = "strongly skewed — 3-sigma bounds are distorted by the long tail; IQR is better."

    st.markdown(
        "Flags readings **more than 3 standard deviations** from the mean. "
        "In a perfectly normal distribution, only ~0.3% of values would be flagged. "
        f"Your data has skewness **{skew:.2f}** — it is {skew_note}"
    )
    p = res.stats["3sigma_params"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Mean", f"{p['mean']:.3f} kWh")
    c2.metric("Upper fence (mean + 3σ)", f"{p['upper']:.3f} kWh")
    n_sigma = len(res.results_by_method["3-Sigma"])
    pct_sigma = res.summary.loc[res.summary["Method"] == "3-Sigma", "% of total"].values[0]
    c3.metric("Flagged", f"{n_sigma} ({pct_sigma:.1f}%)")

    st.plotly_chart(_distribution_fig(kwh, p["lower"], p["upper"], "3-Sigma", "red"),
                    use_container_width=True)
    tl = _timeline_fig(df_clean, res.results_by_method["3-Sigma"].index, "3-Sigma", "#e05050")
    if tl:
        st.plotly_chart(tl, use_container_width=True)
        st.caption("Red dots are flagged readings. Clusters in time suggest a seasonal event "
                   "or meter issue rather than random measurement errors.")

    st.divider()

    # METHOD 2: IQR ────────────────────────────────────────────────────────────
    st.subheader("Method 2 — IQR (Interquartile Range)")
    st.markdown(
        "Flags readings outside **Q1 − 1.5×IQR** and **Q3 + 1.5×IQR**. "
        "Uses the median and quartiles instead of the mean, so extreme values "
        "don't distort the fences — more robust when consumption data has a long right tail."
    )
    p = res.stats["iqr_params"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Q1", f"{p['q1']:.3f} kWh")
    c2.metric("Q3", f"{p['q3']:.3f} kWh")
    c3.metric("Upper fence", f"{p['upper']:.3f} kWh")
    n_iqr = len(res.results_by_method["IQR"])
    pct_iqr = res.summary.loc[res.summary["Method"] == "IQR", "% of total"].values[0]
    c4.metric("Flagged", f"{n_iqr} ({pct_iqr:.1f}%)")

    fig_box = go.Figure()
    fig_box.add_trace(go.Box(y=kwh, name="kWh", marker_color="#6bda9a",
                             boxpoints="outliers", jitter=0.3))
    fig_box.update_layout(title="Box plot — dots beyond whiskers are IQR outliers",
                          yaxis_title="kWh per reading", margin=dict(t=40, b=10))
    st.plotly_chart(fig_box, use_container_width=True)
    st.caption("The box spans Q1 to Q3. Whiskers reach the IQR fences. "
               "Dots beyond the whiskers are the flagged readings.")

    tl = _timeline_fig(df_clean, res.results_by_method["IQR"].index, "IQR", "#e8a020")
    if tl:
        st.plotly_chart(tl, use_container_width=True)

    # 3-sigma vs IQR agreement
    idx_s = set(res.results_by_method["3-Sigma"].index)
    idx_i = set(res.results_by_method["IQR"].index)
    st.markdown(
        f"**3-sigma vs IQR:** {len(idx_s & idx_i)} readings flagged by both · "
        f"{len(idx_s - idx_i)} only by 3-sigma · {len(idx_i - idx_s)} only by IQR. "
        "Disagreements sit in the moderate tail — high but not extreme enough for both to agree."
    )

    st.divider()

    # METHOD 3: DBSCAN ─────────────────────────────────────────────────────────
    st.subheader("Method 3 — DBSCAN (Density-Based Clustering)")
    st.markdown(
        "Finds **dense regions** in a multi-feature space (consumption, time of day, "
        "day of week, daily totals, etc.) and marks points outside all dense regions as **noise**. "
        "Unlike the statistical methods, DBSCAN considers *context* — "
        "a high reading at 6 pm on a weekday may be normal, "
        "while the same value at 3 am on a Sunday gets flagged."
    )
    dp = res.stats["dbscan_params"]
    c1, c2, c3 = st.columns(3)
    c1.metric("eps (neighbourhood radius)", dp.get("eps", "—"))
    c2.metric("Clusters found", dp.get("n_clusters_found", "—"))
    c3.metric("Noise points", f"{dp.get('n_noise', '—')} ({dp.get('noise_pct', '—')}%)")

    st.caption(
        f"**eps selection:** {dp.get('eps_selection_note', '—')}  \n"
        f"**Features used:** {', '.join(dp.get('features_used', []))}  \n"
        f"**Resampled to hourly:** {dp.get('resampled_to_hourly', False)} "
        "(15-min rows → hourly avoids micro-clusters with no behavioural meaning)"
    )

    tl = _timeline_fig(df_clean, res.results_by_method["DBSCAN"].index, "DBSCAN", "#9b59b6")
    if tl:
        st.plotly_chart(tl, use_container_width=True)
        st.caption("Purple dots are DBSCAN noise points. These can be low-value readings "
                   "that happen at an unusual time — not just high spikes.")
    elif len(res.results_by_method["DBSCAN"]) > 0:
        st.dataframe(res.results_by_method["DBSCAN"][["kWh"]].sort_values(
            "kWh", ascending=False).head(20), use_container_width=True)

    st.divider()

    # METHOD 4: Isolation Forest ───────────────────────────────────────────────
    st.subheader("Method 4 — Isolation Forest (Machine Learning)")
    st.markdown(
        "Builds many random decision trees and measures how quickly each point gets "
        "**isolated** from the rest. Anomalies are isolated in fewer tree splits — "
        "they receive a higher anomaly score. No distribution assumed; works well "
        "on small datasets and non-linear patterns."
    )
    ip = res.stats["if_params"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Trees built", ip.get("n_estimators", "—"))
    c2.metric("Contamination", str(ip.get("contamination", "—")))
    n_if = len(res.results_by_method["Isolation Forest"])
    if_mask = res.summary["Method"] == "Isolation Forest"
    pct_if = res.summary.loc[if_mask, "% of total"].values[0]