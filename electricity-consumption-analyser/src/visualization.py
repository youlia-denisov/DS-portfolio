"""
This module generates the charts and graphs to help understand the electricity consumption patterns.
The main function is `save_all_visuals` which takes the cleaned data, aggregated stats, and detected outliers to produce a comprehensive set of visualizations.
The visualizations include:
- Heatmap of average consumption by weekday and hour.
- Heatmap of consumption variability (std) by weekday and hour.
- Bar chart of average hourly consumption by weekday with error bars.
- Bar chart of average daily consumption by weekday with error bars.
- Line chart of daily consumption trend with a 7-day rolling average.
- Load-duration curve to show the distribution of consumption values.
- Clustering (K-means) visualizations, including box plot and heatmap, with ranked electricity consumption. 

All charts are saved as interactive HTML files (by Plotly) in the specified output directory, and also displayed immediately for quick analysis.
The clustering visualizations are saved as PNG files for easy sharing and reporting.
"""

from pathlib import Path
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns

from config import WEEKDAY_ORDER

def save_all_visuals(df: pd.DataFrame, hourly: pd.DataFrame, daily: pd.DataFrame, outliers: pd.DataFrame, html_dir: Path) -> None:
    """Original Plotly visualizations - saved as interactive HTML."""
    html_dir.mkdir(parents=True, exist_ok=True)

    # Heatmap of average consumption by weekday and hour
    pivot = df.pivot_table(index="weekday", columns="hour", values="kWh", aggfunc="mean").reindex(WEEKDAY_ORDER)
    fig = px.imshow(pivot, aspect="auto", color_continuous_scale="RdYlGn_r", 
                    title="Average Consumption Heatmap — Weekday vs Hour")
    fig.update_layout(width=1150, height=620, xaxis_title="Hour", yaxis_title="Weekday")
    fig.update_traces(xgap=1, ygap=1)
    fig.write_html(html_dir / "heatmap_weekday_hour.html")

    # Heatmap of consumption variability (std) by weekday and hour
    std_pivot = hourly.pivot_table(index="weekday", columns="hour", values="std_kWh", aggfunc="mean").reindex(WEEKDAY_ORDER)
    fig = px.imshow(std_pivot, aspect="auto", color_continuous_scale="Oranges",
                    title="Consumption Variability Heatmap — Std by Weekday and Hour")
    fig.write_html(html_dir / "heatmap_variability_weekday_hour.html")

    # Bar chart of average hourly consumption by weekday with error bars
    fig = px.bar(hourly, x="hour", y="avg_kWh", color="weekday", error_y="std_kWh", barmode="group",
                 title="Average Hourly Consumption by Weekday", template="plotly_white")
    fig.update_layout(width=1250, height=680, xaxis=dict(dtick=1))
    fig.write_html(html_dir / "hourly_consumption_by_weekday.html")

    # Bar chart of average daily consumption by weekday with error bars
    fig = px.bar(daily, x="weekday", y="avg_daily_kWh", error_y="std_daily_kWh",
                 title="Average Daily Consumption by Weekday", template="plotly_white")
    fig.write_html(html_dir / "daily_consumption_distribution.html")

    # Line chart of daily consumption trend with a 7-day rolling average
    daily_totals = (df.groupby("date", as_index=False).agg(daily_kWh=("kWh", "sum")))
    daily_totals["rolling_7d"] = daily_totals["daily_kWh"].rolling(7, min_periods=1).mean()
    fig = px.line(daily_totals, x="date", y=["daily_kWh", "rolling_7d"],
                  title="Daily Consumption Trend with 7-Day Rolling Average", template="plotly_white")
    fig.write_html(html_dir / "daily_consumption_trend.html")

    # Load-duration curve
    load_curve = df[["kWh"]].sort_values("kWh", ascending=False).reset_index(drop=True)
    load_curve["time_percentile"] = (load_curve.index + 1) / len(load_curve) * 100
    fig = px.line(load_curve, x="time_percentile", y="kWh",
                  title="Load-Duration Curve", template="plotly_white")
    fig.write_html(html_dir / "load_duration_curve.html")

    # Outlier frequency heatmap
    if not outliers.empty:
        outliers = outliers.copy()
        outliers["weekday"] = outliers["datetime"].dt.day_name()
        outliers["hour"] = outliers["datetime"].dt.hour
        outlier_heatmap = (outliers.pivot_table(
            index="weekday", columns="hour", values="kWh", 
            aggfunc="count", fill_value=0)
            .reindex(WEEKDAY_ORDER)
        )
        
        fig = px.imshow(outlier_heatmap, aspect="auto", color_continuous_scale="Reds", 
                        text_auto=False, title="Outlier Frequency Heatmap — Weekday vs Hour")
        fig.update_layout(width=1150, height=620, xaxis_title="Hour", yaxis_title="Weekday")
        fig.update_traces(xgap=1, ygap=1)
        fig.write_html(html_dir / "outlier_frequency_heatmap.html")

# CLUSTERING VISUALIZATIONS

CLUSTER_PALETTE = {
    0: "#2ca25f",   # low use - green
    1: "#fee08b",   # medium-low - yellow
    2: "#fdae61",   # medium-high - orange
    3: "#d73027",   # high use - red
}

def _dominant_value(values: pd.Series):
    """
    Return the most frequent value in a group.
    Used for the dominant-cluster heatmap.
    """

    return values.mode().iloc[0]

def _ensure_cluster_rank(df: pd.DataFrame) -> pd.DataFrame:
    """
    Safety helper.

    If df already has cluster_rank, use it.
    If not, create cluster_rank from average kWh per raw cluster.
    """

    df = df.copy()

    if "cluster_rank" in df.columns:
        return df

    cluster_order = (
        df.groupby("cluster")["kWh"]
        .mean()
        .sort_values()
        .index
    )

    rank_map = {
        cluster: rank
        for rank, cluster in enumerate(cluster_order)
    }

    df["cluster_rank"] = df["cluster"].map(rank_map)

    return df

def plot_elbow_curve(
    inertia_values: dict,
    chosen_k: int,
    output_path: Path,
) -> None:
    """
    Plot inertia vs. k (the elbow curve) and save as PNG.

    How to read this plot:
    - X-axis: number of clusters (k)
    - Y-axis: inertia — lower means more compact clusters
    - The curve drops steeply, then flattens
    - The "elbow" is the bend where improvement slows down
    - The red dashed line marks the chosen k (N_CLUSTERS)

    If the line sits at the elbow, the choice is well justified.
    If the curve hasn't bent yet at that k, consider a larger k.
    If the bend was earlier, consider a smaller k.
    """

    k_values = list(inertia_values.keys())
    inertia_list = list(inertia_values.values())

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(
        k_values,
        inertia_list,
        marker="o",
        color="#2171b5",
        linewidth=2,
        markersize=7,
    )

    # Annotate each point with its inertia value for easy reading.
    for k, inertia in zip(k_values, inertia_list):
        ax.annotate(
            f"{inertia:,.0f}",
            xy=(k, inertia),
            xytext=(0, 10),
            textcoords="offset points",
            ha="center",
            fontsize=8,
            color="#444444",
        )

    # Red dashed line marks the k used in the final model.
    ax.axvline(
        x=chosen_k,
        color="#d73027",
        linestyle="--",
        linewidth=1.5,
        label=f"Chosen k = {chosen_k}",
    )

    ax.set_title("Elbow Method — KMeans Inertia vs. Number of Clusters")
    ax.set_xlabel("Number of clusters (k)")
    ax.set_ylabel("Inertia (within-cluster sum of squares)")
    ax.set_xticks(k_values)
    ax.legend()

    sns.despine()
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved elbow curve to: {output_path}")


def save_clustering_visuals(
    df_clustered: pd.DataFrame,
    figure_dir: Path,
) -> list[Path]: # type: ignore
    """
    Save clustering visualizations.

    Parameters:
    - df_clustered:
        DataFrame returned by run_clustering().
        Expected columns:
        - kWh
        - hour
        - weekday
        - cluster
        - cluster_rank

    - figure_dir:
        Folder where PNG files should be saved.

    Returns:
    - list of generated plot paths
    """

    if df_clustered is None or df_clustered.empty:
        print("No clustered data received. Skipping clustering visuals.")
        return []

    required_cols = ["kWh", "hour", "weekday", "cluster"]
    missing = [col for col in required_cols if col not in df_clustered.columns]

    if missing:
        print(f"Missing clustering visualization columns: {missing}")
        return []

    figure_dir.mkdir(parents=True, exist_ok=True)

    df = _ensure_cluster_rank(df_clustered)

    generated = []

    # 1. Cluster summary dashboard

    sns.set_style("whitegrid")

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    sns.boxplot(
        data=df,
        x="cluster_rank",
        y="kWh",
        hue="cluster_rank",
        palette=CLUSTER_PALETTE,
        legend=False,
        ax=axes[0, 0],
    )
    axes[0, 0].set_title("kWh Distribution by Ranked Cluster")
    axes[0, 0].set_xlabel("Cluster rank: low → high usage")
    axes[0, 0].set_ylabel("kWh")

    hourly_profile = (
        df.groupby(["hour", "cluster_rank"], as_index=False)["kWh"]
        .mean()
    )

    sns.lineplot(
        data=hourly_profile, # type: ignore
        x="hour",
        y="kWh",
        hue="cluster_rank",
        palette=CLUSTER_PALETTE,
        marker="o",
        ax=axes[0, 1],
    )
    axes[0, 1].set_title("Average Hourly Profile by Ranked Cluster")
    axes[0, 1].set_xlabel("Hour")
    axes[0, 1].set_ylabel("Average kWh")
    axes[0, 1].set_xticks(range(0, 24, 2))

    weekday_profile = (
        df.groupby(["weekday", "cluster_rank"], as_index=False)
        .agg(kWh=("kWh", "mean"))
    )

    weekday_profile["weekday"] = pd.Categorical(
        weekday_profile["weekday"],
        categories=WEEKDAY_ORDER,
        ordered=True,
    )

    weekday_profile = weekday_profile.sort_values(["weekday", "cluster_rank"])

    sns.lineplot(
        data=weekday_profile,
        x="weekday",
        y="kWh",
        hue="cluster_rank",
        palette=CLUSTER_PALETTE,
        marker="o",
        ax=axes[1, 0],
    )
    axes[1, 0].set_title("Average Weekday Profile by Ranked Cluster")
    axes[1, 0].set_xlabel("Weekday")
    axes[1, 0].set_ylabel("Average kWh")
    axes[1, 0].tick_params(axis="x", rotation=35)

    cluster_sizes = df["cluster_rank"].value_counts().sort_index()

    bars = axes[1, 1].bar(
        cluster_sizes.index,
        cluster_sizes.values,
        color=[CLUSTER_PALETTE.get(k, "#999999") for k in cluster_sizes.index],
    )
    for bar, count in zip(bars, cluster_sizes.values):
        axes[1, 1].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 20,
            str(count),
            ha="center",
            va="bottom",
            fontsize=9,
        )
    axes[1, 1].set_title("Records per Cluster")
    axes[1, 1].set_xlabel("cluster_rank")
    axes[1, 1].set_ylabel("count")

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=CLUSTER_PALETTE[r], label=f"Rank {r}")
        for r in sorted(CLUSTER_PALETTE)
        if r in cluster_sizes.index
    ]
    axes[1, 1].legend(handles=legend_elements, title="cluster_rank", fontsize=8)

    plt.suptitle("Clustering Summary Dashboard", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()

    dashboard_path = figure_dir / "cluster_dashboard.png"
    plt.savefig(dashboard_path, dpi=300, bbox_inches="tight")
    plt.close()
    generated.append(dashboard_path)
    print(f"Saved cluster dashboard to: {dashboard_path}")

    # 2. Time-grid heatmap
    grid_path = plot_time_grid_heatmap(df, figure_dir)
    if grid_path:
        generated.append(grid_path)

    return generated


def plot_time_grid_heatmap(df: pd.DataFrame, figure_dir: Path) -> "Path | None":
    """
    Time-grid heatmap: hour of day (rows) x day of week (columns),
    coloured by the dominant cluster_rank at each slot.

    Why this matters for analysts
    ------------------------------
    The knife plot and silhouette scores tell a data scientist how well
    the model separated data. This chart tells anyone what the clusters mean
    in real life: you can read off "high-use mode happens every weekday
    evening" at a glance, without understanding a single algorithm.

    How it's built
    --------------
    For every (hour, weekday) combination we find the most common cluster_rank
    across all weeks in the dataset -- that's the "dominant" cluster for that
    slot. The result is a 24x7 grid coloured by cluster rank.

    How to read it
    --------------
    - Rows    = hours 0-23 (midnight at top)
    - Columns = Mon-Sun
    - Colour  = dominant cluster rank (green = low use, red = high use)
    - A solid band of red across weekday evenings = concentrated peak
    - A mixed/patchy column = that day has variable, unpredictable usage
    - A fully green row (e.g. 3am) = always low regardless of day
    """
    required = {"hour", "weekday", "cluster_rank"}
    if not required.issubset(df.columns):
        print(f"plot_time_grid_heatmap: missing columns {required - set(df.columns)}")
        return None

    from config import WEEKDAY_ORDER
    import matplotlib.colors as mcolors

    # Find the most frequent cluster_rank for each (hour, weekday) slot.
    dominant = (
        df.groupby(["hour", "weekday"])["cluster_rank"]
        .agg(lambda s: s.mode().iloc[0])
        .reset_index()
    )

    pivot = (
        dominant.pivot(index="hour", columns="weekday", values="cluster_rank")
        .reindex(columns=WEEKDAY_ORDER)
        .sort_index()
    )

    # Discrete colour map that matches CLUSTER_PALETTE so colours are
    # consistent with the dashboard.
    n_ranks = len(CLUSTER_PALETTE)
    cmap = mcolors.ListedColormap([CLUSTER_PALETTE[r] for r in sorted(CLUSTER_PALETTE)])
    bounds = list(range(n_ranks + 1))
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    fig, ax = plt.subplots(figsize=(10, 8))

    im = ax.imshow(
        pivot.values,
        aspect="auto",
        cmap=cmap,
        norm=norm,
        interpolation="nearest",
    )

    # Annotate each cell with the rank number for precision.
    for row_idx in range(pivot.shape[0]):
        for col_idx in range(pivot.shape[1]):
            val = pivot.values[row_idx, col_idx]
            if not pd.isna(val):
                hex_color = CLUSTER_PALETTE.get(int(val), "#888888").lstrip("#")
                brightness = sum(
                    int(hex_color[i:i+2], 16) for i in (0, 2, 4)
                ) / (3 * 255)
                text_color = "black" if brightness > 0.5 else "white"
                ax.text(col_idx, row_idx, str(int(val)),
                        ha="center", va="center", fontsize=7, color=text_color)

    ax.set_xticks(range(len(WEEKDAY_ORDER)))
    ax.set_xticklabels(WEEKDAY_ORDER, fontsize=9)
    ax.set_yticks(range(24))
    ax.set_yticklabels([f"{h:02d}:00" for h in range(24)], fontsize=7.5)
    ax.set_xlabel("Day of week", fontsize=10)
    ax.set_ylabel("Hour of day", fontsize=10)
    ax.set_title(
        "Dominant Cluster per Hour x Day of Week\n"
        "(colour = most common cluster rank across all weeks; 0 = low use, 3 = high use)",
        fontsize=11,
        pad=12,
    )

    cbar = fig.colorbar(im, ax=ax, ticks=[r + 0.5 for r in range(n_ranks)])
    cbar.ax.set_yticklabels([f"Rank {r}" for r in range(n_ranks)], fontsize=8)
    cbar.set_label("Cluster rank  (0 = low, 3 = high use)", fontsize=8)

    plt.tight_layout()
    grid_path = figure_dir / "cluster_time_grid.png"
    plt.savefig(grid_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved time-grid heatmap to: {grid_path}")
    return grid_path
