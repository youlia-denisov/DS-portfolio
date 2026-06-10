"""
dbscan_analysis.py
------------------
DBSCAN clustering as an alternative to K-Means, run on the same feature
matrix used in clustering.py.

Why DBSCAN?
-----------
K-Means must assign every point to a cluster, so unusual readings (extreme
weather, broken appliance, guests) end up forming their own tiny cluster
rather than being flagged as the anomalies they are.

DBSCAN takes a different approach:
  - It finds regions of high density and calls those clusters.
  - Points that don't belong to any dense region are labelled -1 (noise).
  - You keep all the data; the algorithm just marks unusual readings honestly
    instead of forcing them somewhere they don't belong.

Two parameters control everything:
  eps         -- the radius around a point to search for neighbours.
                 Think of it as "how close is close enough to be a neighbour?"
  min_samples -- how many neighbours must be within eps for a point to be
                 considered part of a dense region (a "core point").

The main challenge: choosing eps is not obvious. This script uses a
k-distance plot to help you pick a good value before running the final model.

A common failure mode: one giant cluster.
  If eps is too large, nearly all points fall within reach of each other and
  DBSCAN merges them into a single mega-cluster. The algorithm technically
  reports multiple clusters (small satellite groups at the fringes), but one
  cluster can hold 95–99 % of all points. Any visualisation that shows only
  the *dominant* label per time slot will then paint the entire grid one colour
  -- which looks like the plot is broken, but actually means eps is too large.

  The selection logic below guards against this with a dominance check:
  after finding candidate eps values, it rejects any where the largest cluster
  accounts for more than MAX_DOMINANCE of all non-noise points.

Input:  data/processed/cleaned_consumption_clustered.csv
        (same file used by kmeans_silhouette.py -- already has feature columns)
Output: figures/dbscan_1_eps_selection.png  (k-distance plot for eps selection)
        figures/dbscan_2_diversity_map.png  (NEW: distinct cluster count per hour×day)
        figures/dbscan_3_noise_heatmap.png  (heatmap of noise point frequency by hour×day)
        figures/dbscan_4_dominant_map.png   (dominant label map, kept for reference)

Notes:
The honest takeaway: there is no clean eps for this dataset (13 weeks). 
The jump from 0.50 (112 clusters, chaotic) to 0.80 (47 clusters, reasonable noise) to 1.20 (mega-cluster) happens very fast. 
That's a sign the data doesn't have well-separated density regions — 
which is exactly what you'd expect from 13 weeks of fairly regular household consumption. 
Most hours are behaviorally similar, so DBSCAN has nothing clear to separate.
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import RobustScaler

# project root on sys.path so config imports work
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from config import PROCESSED_DIR, FIGURE_DIR, WEEKDAY_ORDER

# ── configuration ─────────────────────────────────────────────────────────────

INPUT_FILE = PROCESSED_DIR / "cleaned_consumption_clustered.csv"

# These are the same features used in clustering.py and kmeans_silhouette.py,
# so the two methods are directly comparable.
FEATURE_COLS = [
    "kWh",
    "hour_sin",    "hour_cos",
    "weekday_sin", "weekday_cos",
    "daily_total_kwh",
    "evening_peak",
    "night_baseline",
    "peak_to_baseline",
]

# min_samples: a common rule of thumb is 2 × number of features.
# With 9 features that gives 18, but for 15-min electricity data a smaller
# value (5-10) works better because genuine clusters are dense and numerous.
# Adjust if DBSCAN returns too few or too many clusters.
MIN_SAMPLES = 5

# EPS_CANDIDATES: a wider sweep so we can see the full picture from tight
# (many small clusters, lots of noise) to loose (few large clusters, little noise).
# Values above 1.2 were added based on the k-distance plot elbow, which suggested
# the curve flattens somewhere in the 1.5–2.0 range for this dataset.
EPS_CANDIDATES = [0.3, 0.5, 0.8, 1.2, 1.5, 1.8, 2.0, 2.5]

# When one cluster holds more than this fraction of all non-noise points, the
# dominant-cluster visualisation becomes a single-colour grid and is meaningless.
# We use this threshold to skip such eps values when selecting the "best" one.
MAX_DOMINANCE = 0.80

# ── load and resample to hourly ───────────────────────────────────────────────
#
# The source file has 15-minute readings. Four rows per hour means each dense
# cluster of "normal 15-min windows" becomes its own micro-cluster, giving
# DBSCAN hundreds of groups with no behavioural meaning.
#
# Resampling to hourly before clustering fixes this: each point now represents
# a full hour of consumption, which is the natural unit for behavioural analysis
# (morning peak, midday lull, evening peak, overnight baseline).
#
# Aggregation choices:
#   kWh              → sum (four 15-min readings → one hourly total)
#   daily_total_kwh  → mean (it's the same value for all rows in a day,
#                       so mean and first() are equivalent; mean is safer)
#   evening_peak     → mean (same reasoning)
#   night_baseline   → mean
#   peak_to_baseline → mean
#   hour_sin/cos     → mean (all four rows in the same hour have the same
#                       sin/cos value, so mean = first())
#   weekday_sin/cos  → mean (same)

df_raw = pd.read_csv(INPUT_FILE)
df_raw["datetime"] = pd.to_datetime(df_raw["datetime"])

# check that the required feature columns exist in the source file
missing = [c for c in FEATURE_COLS if c not in df_raw.columns]
if missing:
    raise ValueError(
        f"Missing feature columns: {missing}\n"
        "Run the main pipeline first to generate cleaned_consumption_clustered.csv."
    )

# resample: floor datetime to the hour, then aggregate
df_raw["hour_dt"] = df_raw["datetime"].dt.floor("h")

df = (
    df_raw.groupby("hour_dt")
    .agg(
        kWh=("kWh", "sum"),
        hour_sin=("hour_sin", "mean"),
        hour_cos=("hour_cos", "mean"),
        weekday_sin=("weekday_sin", "mean"),
        weekday_cos=("weekday_cos", "mean"),
        daily_total_kwh=("daily_total_kwh", "mean"),
        evening_peak=("evening_peak", "mean"),
        night_baseline=("night_baseline", "mean"),
        peak_to_baseline=("peak_to_baseline", "mean"),
    )
    .reset_index()
    .rename(columns={"hour_dt": "datetime"})
)

# re-derive hour and weekday labels for the time-grid plots
df["hour"] = df["datetime"].dt.hour
df["weekday"] = df["datetime"].dt.day_name()

print(f"Resampled: {len(df_raw)} 15-min rows → {len(df)} hourly rows")

# RobustScaler: same scaler as the main pipeline, centres on median and
# scales by IQR so outliers don't dominate the distance calculations.
X = RobustScaler().fit_transform(df[FEATURE_COLS])

FIGURE_DIR.mkdir(parents=True, exist_ok=True)

# ── PART 1: k-distance plot for eps selection ─────────────────────────────────
#
# How it works:
#   For each point, compute the distance to its MIN_SAMPLES-th nearest neighbour.
#   Sort all those distances. Plot them.
#
# How to read it:
#   The curve rises slowly at first (dense region), then bends sharply upward.
#   The "elbow" of that bend is a good eps value -- it separates dense points
#   (which will form clusters) from sparse points (which will become noise).
#   If you pick eps too small: almost everything becomes noise.
#   If you pick eps too large: everything merges into one giant cluster.

print("Computing k-distance plot...")
nbrs = NearestNeighbors(n_neighbors=MIN_SAMPLES).fit(X)
distances, _ = nbrs.kneighbors(X)

# distance to the MIN_SAMPLES-th neighbour (last column), sorted ascending
kth_distances = np.sort(distances[:, -1])

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(kth_distances, color="#4a90d9", lw=1.5)

# mark the eps candidates as vertical reference lines so you can see where
# each candidate would cut the sorted distances
for eps_val in EPS_CANDIDATES:
    # find the index where kth_distances first exceeds eps_val
    crossings = np.where(kth_distances >= eps_val)[0]
    if len(crossings):
        ax.axhline(eps_val, ls="--", lw=1, alpha=0.7,
                   label=f"eps = {eps_val}")

ax.set_title(
    f"k-Distance Plot  (k = MIN_SAMPLES = {MIN_SAMPLES})\n"
    "Look for the elbow — that is a good starting value for eps",
    fontsize=11,
)
ax.set_xlabel("Points sorted by distance to k-th neighbour", fontsize=9)
ax.set_ylabel(f"Distance to {MIN_SAMPLES}-th neighbour", fontsize=9)
ax.legend(fontsize=8)
plt.tight_layout()

eps_plot_path = FIGURE_DIR / "dbscan_1_eps_selection.png"
plt.savefig(eps_plot_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {eps_plot_path}")

# ── PART 2: run DBSCAN for each eps candidate and compare results ─────────────
#
# For each eps we report:
#   n_clusters : how many clusters were found (not counting noise)
#   n_noise    : how many points were labelled -1 (noise)
#   noise_%    : percentage of all readings flagged as noise
#
# A good eps gives a meaningful number of clusters (2-5 for this dataset)
# and a noise% that feels plausible -- too low means outliers are being
# absorbed into clusters, too high means eps is too tight.

print("\nDBSCAN results across eps candidates:")
print(f"{'eps':>6}  {'clusters':>10}  {'noise pts':>10}  {'noise %':>8}  {'dominance %':>13}")
print("-" * 56)

results = {}
for eps_val in EPS_CANDIDATES:
    db = DBSCAN(eps=eps_val, min_samples=MIN_SAMPLES).fit(X)
    labels = db.labels_
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = (labels == -1).sum()
    noise_pct = 100 * n_noise / len(labels)

    # dominance: fraction of non-noise points that belong to the largest cluster.
    # 1.0 means one cluster holds all assigned points (mega-cluster problem).
    non_noise = labels[labels >= 0]
    if len(non_noise) > 0:
        largest_cluster_size = pd.Series(non_noise).value_counts().iloc[0]
        dominance = largest_cluster_size / len(non_noise)
    else:
        dominance = 1.0

    results[eps_val] = {
        "labels": labels,
        "n_clusters": n_clusters,
        "n_noise": n_noise,
        "noise_pct": noise_pct,
        "dominance": dominance,
    }
    flag = "  ← mega-cluster" if dominance > MAX_DOMINANCE else ""
    print(f"{eps_val:>6.2f}  {n_clusters:>10}  {n_noise:>10}  {noise_pct:>7.1f}%  {dominance * 100:>12.1f}%{flag}")

# ── PART 3: detailed plots for the eps that gives the most useful result ───────
#
# "Most useful" = more than 1 cluster and noise% below 20%.
# If none qualify, we fall back to the middle eps candidate.

# Select the best eps using three criteria (all must be satisfied):
#   1. Between 2 and 20 clusters   — more is not interpretable; fewer is uninformative
#   2. Noise% below 20%            — if most points are noise, eps is too tight
#   3. Dominance below MAX_DOMINANCE — rejects the "mega-cluster" case where one
#      cluster holds > 80% of assigned points and the grid plot becomes one colour
#
# Candidates are then sorted by number of clusters (ascending) so we prefer the
# simplest structure that still passes all three checks.
good_eps = sorted(
    [
        eps for eps, r in results.items()
        if 2 <= r["n_clusters"] <= 20
        and r["noise_pct"] < 20
        and r["dominance"] < MAX_DOMINANCE
    ],
    key=lambda e: results[e]["n_clusters"],
)

# Fallback 1: relax the dominance constraint and accept modest mega-clusters.
if not good_eps:
    good_eps = sorted(
        [eps for eps, r in results.items() if 2 <= r["n_clusters"] <= 20],
        key=lambda e: results[e]["dominance"],  # prefer least-dominated
    )

# Fallback 2: anything with more than one cluster.
if not good_eps:
    good_eps = sorted(
        [eps for eps, r in results.items() if r["n_clusters"] > 1],
        key=lambda e: results[e]["n_clusters"],
    )

chosen_eps = good_eps[0] if good_eps else EPS_CANDIDATES[len(EPS_CANDIDATES) // 2]
print(f"\nSelected eps={chosen_eps} (dominance={results[chosen_eps]['dominance']*100:.1f}%)")
chosen = results[chosen_eps]
df["dbscan_label"] = chosen["labels"]

print(f"\nUsing eps={chosen_eps} for detailed plots "
      f"({chosen['n_clusters']} clusters, {chosen['noise_pct']:.1f}% noise)")

n_clusters = chosen["n_clusters"]
all_labels = sorted(df["dbscan_label"].unique())  # includes -1 for noise

# build a colour map: grey for noise (-1), then a distinct colour per cluster
from matplotlib.patches import Patch
palette = {-1: "#cccccc"}
cluster_colors = plt.cm.tab10.colors
for i, lbl in enumerate([l for l in all_labels if l >= 0]):
    palette[lbl] = mcolors.to_hex(cluster_colors[i % len(cluster_colors)])

# ── PART 3: cluster diversity heatmap ─────────────────────────────────────────
#
# The dominant-cluster map (one colour per cell = the most common label) becomes
# useless when one cluster dominates: every cell shows the same colour.
# Instead, show CLUSTER DIVERSITY: how many distinct cluster labels appear in
# each hour × day slot across all weeks in the dataset.
#
# How to read it:
#   White / pale = only one cluster ever appears at this slot → very stable behaviour.
#   Dark blue    = several different clusters appear here across weeks → this slot
#                  is where behaviour is most variable or ambiguous.
#
# This plot is informative regardless of how many clusters DBSCAN finds or how
# unbalanced their sizes are.

diversity = (
    df.groupby(["hour", "weekday"])["dbscan_label"]
    .nunique()
    .reset_index(name="n_distinct")
)

diversity_pivot = (
    diversity.pivot(index="hour", columns="weekday", values="n_distinct")
    .reindex(columns=WEEKDAY_ORDER)
    .sort_index()
    .fillna(0)
)

fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(diversity_pivot.values, aspect="auto",
               cmap="Blues", interpolation="nearest", vmin=0)

for ri in range(diversity_pivot.shape[0]):
    for ci in range(diversity_pivot.shape[1]):
        val = diversity_pivot.values[ri, ci]
        if not np.isnan(val) and val > 0:
            ax.text(ci, ri, str(int(val)), ha="center", va="center",
                    fontsize=6.5, color="black" if val < diversity_pivot.values.max() * 0.6 else "white")

ax.set_xticks(range(len(WEEKDAY_ORDER)))
ax.set_xticklabels(WEEKDAY_ORDER, fontsize=9)
ax.set_yticks(range(24))
ax.set_yticklabels([f"{h:02d}:00" for h in range(24)], fontsize=7.5)
ax.set_xlabel("Day of week", fontsize=10)
ax.set_ylabel("Hour of day", fontsize=10)
ax.set_title(
    f"DBSCAN — Cluster Diversity per Hour × Day of Week\n"
    f"eps={chosen_eps}, min_samples={MIN_SAMPLES}  |  "
    f"Number = distinct labels seen at that slot across all weeks",
    fontsize=10, pad=12,
)

cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
cbar.set_label("Number of distinct cluster labels", fontsize=8)
cbar.ax.tick_params(labelsize=8)

plt.tight_layout()
diversity_path = FIGURE_DIR / "dbscan_2_diversity_map.png"
plt.savefig(diversity_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {diversity_path}")

# ── PART 4: noise point heatmap ───────────────────────────────────────────────
#
# Where in the week do the noise points concentrate?
# High noise in a particular hour/day slot means readings there are
# consistently unusual -- evidence for weather-driven or event-driven
# anomalies rather than random noise.

noise_df = df[df["dbscan_label"] == -1].copy()

if noise_df.empty:
    print("  No noise points to plot -- try a smaller eps.")
else:
    noise_pivot = (
        noise_df.groupby(["hour", "weekday"])
        .size()
        .reset_index(name="count")
        .pivot(index="hour", columns="weekday", values="count")
        .reindex(columns=WEEKDAY_ORDER)
        .sort_index()
        .fillna(0)
    )

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(noise_pivot.values, aspect="auto",
                   cmap="Reds", interpolation="nearest")

    ax.set_xticks(range(len(WEEKDAY_ORDER)))
    ax.set_xticklabels(WEEKDAY_ORDER, fontsize=9)
    ax.set_yticks(range(24))
    ax.set_yticklabels([f"{h:02d}:00" for h in range(24)], fontsize=7.5)
    ax.set_xlabel("Day of week", fontsize=10)
    ax.set_ylabel("Hour of day", fontsize=10)
    ax.set_title(
        "DBSCAN Noise Points — Frequency by Hour x Day of Week\n"
        "Dark red = many noise readings at that slot  |  White = no noise",
        fontsize=10, pad=12,
    )

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Number of noise-labelled hourly readings", fontsize=8)
    cbar.ax.tick_params(labelsize=8)

    plt.tight_layout()
    noise_path = FIGURE_DIR / "dbscan_3_noise_heatmap.png"
    plt.savefig(noise_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {noise_path}")

    print("\nTop 10 hour x day slots with most noise points:")
    top_noise = (
        noise_df.groupby(["weekday", "hour"])
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
        .head(10)
    )
    print(top_noise.to_string(index=False))

# ── PART 5: dominant cluster map (kept for reference) ─────────────────────────
#
# This is the original visualisation. It can still be useful when DBSCAN finds
# a well-balanced set of clusters (no mega-cluster), but read it alongside the
# diversity map above -- if diversity is uniformly 1, this plot adds nothing.

dominant = (
    df.groupby(["hour", "weekday"])["dbscan_label"]
    .agg(lambda s: s.mode().iloc[0])
    .reset_index()
)
pivot = (
    dominant.pivot(index="hour", columns="weekday", values="dbscan_label")
    .reindex(columns=WEEKDAY_ORDER)
    .sort_index()
)

label_to_idx = {lbl: i for i, lbl in enumerate(all_labels)}
idx_grid = pivot.map(lambda v: label_to_idx.get(v, 0) if not pd.isna(v) else np.nan)

cmap_dom = mcolors.ListedColormap([palette[lbl] for lbl in all_labels])
norm_dom = mcolors.BoundaryNorm(range(len(all_labels) + 1), cmap_dom.N)

fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(idx_grid.values, aspect="auto", cmap=cmap_dom, norm=norm_dom,
               interpolation="nearest")

for ri in range(pivot.shape[0]):
    for ci in range(pivot.shape[1]):
        val = pivot.values[ri, ci]
        if not pd.isna(val):
            label_str = "N" if int(val) == -1 else str(int(val))
            ax.text(ci, ri, label_str, ha="center", va="center",
                    fontsize=6.5, color="black")

ax.set_xticks(range(len(WEEKDAY_ORDER)))
ax.set_xticklabels(WEEKDAY_ORDER, fontsize=9)
ax.set_yticks(range(24))
ax.set_yticklabels([f"{h:02d}:00" for h in range(24)], fontsize=7.5)
ax.set_xlabel("Day of week", fontsize=10)
ax.set_ylabel("Hour of day", fontsize=10)
dominance_pct = chosen["dominance"] * 100
ax.set_title(
    f"DBSCAN — Dominant Cluster per Hour × Day of Week  "
    f"[largest cluster = {dominance_pct:.0f}% of assigned pts]\n"
    f"eps={chosen_eps}, min_samples={MIN_SAMPLES}  |  "
    f"{n_clusters} clusters  +  {chosen['n_noise']} noise points (N)",
    fontsize=9, pad=12,
)

legend_handles = [
    Patch(facecolor=palette[lbl],
          label="Noise (-1)" if lbl == -1 else f"Cluster {lbl}")
    for lbl in all_labels
]
ax.legend(handles=legend_handles, loc="upper right", fontsize=8, framealpha=0.8)

plt.tight_layout()
dominant_map_path = FIGURE_DIR / "dbscan_4_dominant_map.png"
plt.savefig(dominant_map_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {dominant_map_path}")

print("\nDone.")
print("  dbscan_2_diversity_map.png  — how variable each hour×day slot is")
print("  dbscan_3_noise_heatmap.png  — where noise points concentrate")
print("  dbscan_4_dominant_map.png   — dominant label (useful if dominance is low)")
print("Compare the noise heatmap with cluster_time_grid.png to see whether")
print("DBSCAN noise points coincide with K-Means' smallest cluster.")
