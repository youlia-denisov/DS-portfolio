"""
kmeans_silhouette.py
--------------------
Answers two questions:
  1. What is the best k?          → elbow + silhouette across k values
  2. How good is the chosen k=4?  → per-sample knife plot

Simple silhouette interpretation:
  -1.0 to 0.0 : sample is probably in the wrong cluster
  | Silhouette score | Meaning                                 |
| ---------------: | --------------------------------------- |
|       close to 1 | strong, well-separated clusters         |
|       around 0.5 | reasonable clustering                   |
|       around 0.2 | weak / overlapping clusters             |
|         around 0 | clusters are not clearly separated      |
|         negative | many points may be in the wrong cluster |


Input : PROCESSED_DIR/cleaned_consumption_clustered.csv  (already has cluster labels)
Output: 1_choosing_k.png, 2_knife_plot.png (saved to FIGURE_DIR)

── Feature engineering note ──────────────────────────────────────────────────
The original script clustered on hourly-level features only:
  kWh, hour_sin, hour_cos, weekday_sin, weekday_cos, month_sin, month_cos

The problem: a single hourly reading doesn't describe a *day's behaviour*.
Two days with identical total consumption can have completely different shapes
(flat all day vs. sharp evening spike), and KMeans can't distinguish them
from hourly values alone.

New approach: enrich each hourly row with *daily-aggregate* features so the
algorithm can "see" the full-day context of each measurement.

New features added:
  daily_total_kwh     — total consumption for that calendar day
                        → separates high vs. low demand days
  evening_peak        — mean kWh between 18:00–22:00 for that day
                        → captures typical residential dinner/TV peak
  night_baseline      — mean kWh between 01:00–05:00 for that day
                        → always-on devices (fridge, router) vs. zero-use nights
  is_weekend          — 1 if Saturday or Sunday, 0 otherwise
                        → direct behavioural split; more interpretable than
                           weekday_sin/cos for a binary distinction
  peak_to_baseline    — evening_peak / (night_baseline + 1e-6)
                        → shape descriptor: high ratio = sharp evening spike,
                           ratio ≈ 1 = flat profile all day
                        → the tiny epsilon (1e-6) prevents division by zero
                           on nights where consumption is exactly 0

Why NOT season / rolling averages with only 13 weeks of data?
  • A season window would span nearly the whole dataset → zero variance.
  • A 7-day rolling average would average away the short-term patterns we
    actually want to find.
──────────────────────────────────────────────────────────────────────────────
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import silhouette_score, silhouette_samples


# some manual plumbing to make sure we can import from the project root....

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from config import PROCESSED_DIR, FIGURE_DIR


# ── Load ───────────────────────────────────────────────────────────────────
df = pd.read_csv(f'{PROCESSED_DIR}/cleaned_consumption_clustered.csv')

# The pipeline (clustering.py) now computes all daily aggregate features
# (daily_total_kwh, evening_peak, night_baseline, peak_to_baseline) and
# saves them directly into the clustered CSV. No need to recompute them here.
# We only parse datetime so Part 4 can access it for the centroid plot.
df['datetime'] = pd.to_datetime(df['datetime'])


# ══════════════════════════════════════════════════════════════════════════
# FEATURE MATRIX
# We now use a richer set of features.  Notes on what was kept / changed:
#
#   kWh              — the raw hourly reading (still useful at hourly level)
#   hour_sin/cos     — encode time-of-day cyclically (midnight ≈ 23:00)
#   weekday_sin/cos  — kept for gradual within-week trends
#   daily_total_kwh  — NEW: full-day consumption context
#   evening_peak     — NEW: typical evening demand for that day
#   night_baseline   — NEW: overnight floor for that day
#   is_weekend       — NEW: binary work/rest flag
#   peak_to_baseline — NEW: shape of the day's consumption curve
#
# month_sin / month_cos were REMOVED:
#   With only 13 weeks (~3 months) there is almost no seasonal variance.
#   Including them would add noise without adding signal.
# ══════════════════════════════════════════════════════════════════════════

FEATURE_COLS = [
    'kWh',
    'hour_sin',  'hour_cos',
    'weekday_sin', 'weekday_cos',
    'daily_total_kwh',
    'evening_peak',
    'night_baseline',
    'peak_to_baseline',
]

# RobustScaler centres on the median and scales by the interquartile range.
# This is more appropriate than StandardScaler when your data has outliers
# (e.g. a single unusually high-consumption day).
X = RobustScaler().fit_transform(df[FEATURE_COLS])

COLORS = ['#4FC3F7', '#81C784', '#FFB74D', '#E57373', '#CE93D8', '#4DB6AC']

# ══════════════════════════════════════════════════════════════════════════
# PART 1 — Try k=2..7, record inertia and silhouette score for each
# ══════════════════════════════════════════════════════════════════════════
ks, inertias, sil_scores = [], [], []

for k in range(2, 8):
    km = KMeans(n_clusters=k, random_state=42, n_init='auto')
    labels = km.fit_predict(X)
    ks.append(k)
    inertias.append(km.inertia_)
    sil_scores.append(silhouette_score(X, labels))
    print(f"k={k}  inertia={km.inertia_:.0f}  silhouette={sil_scores[-1]:.3f}")

# Plot: elbow on the left, silhouette on the right
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4), facecolor='#0f1117')
fig.suptitle('Step 1 — Choosing the right number of clusters (k)',
             color='#e2e8f0', fontsize=13, fontweight='bold', y=1.02)

for ax in (ax1, ax2):
    ax.set_facecolor('#161b27')
    ax.tick_params(colors='#6b7a99', labelsize=9)
    for sp in ax.spines.values():
        sp.set_edgecolor('#2a3352')

ax1.plot(ks, inertias, color='#70b8ff', lw=2, marker='o', markersize=5)
ax1.set_title('Elbow Curve (Inertia)', color='#c9d1e0', fontsize=11, pad=10)
ax1.set_xlabel('k', color='#6b7a99')
ax1.set_ylabel('Inertia', color='#6b7a99')
ax1.set_xticks(ks)

ax2.plot(ks, sil_scores, color='#6bda9a', lw=2, marker='o', markersize=5)
best_k_idx = sil_scores.index(max(sil_scores))
ax2.scatter([ks[best_k_idx]], [sil_scores[best_k_idx]], s=100, color='#6bda9a', zorder=5)
ax2.annotate(f'  best k={ks[best_k_idx]}', (ks[best_k_idx], sil_scores[best_k_idx]),
             color='#6bda9a', fontsize=9, va='center')
ax2.set_title('Silhouette Score per k', color='#c9d1e0', fontsize=11, pad=10)
ax2.set_xlabel('k', color='#6b7a99')
ax2.set_ylabel('Silhouette score', color='#6b7a99')
ax2.set_xticks(ks)

plt.tight_layout()
plt.savefig(f'{FIGURE_DIR}/1_choosing_k.png', dpi=150, bbox_inches='tight',
            facecolor=fig.get_facecolor())
plt.close()

# ══════════════════════════════════════════════════════════════════════════
# PART 2 — Knife plot for the saved clustering (k=4)
# Shows each individual point's silhouette score, grouped by cluster
# ══════════════════════════════════════════════════════════════════════════
chosen_labels   = df['cluster'].values
sample_sil      = silhouette_samples(X, chosen_labels)   # one score per row
global_sil      = sample_sil.mean()
n_clusters      = df['cluster'].nunique()
unique_clusters = sorted(df['cluster'].unique())

print(f"\nGlobal silhouette (k={n_clusters}): {global_sil:.3f}")

fig, ax = plt.subplots(figsize=(9, 5), facecolor='#0f1117')
ax.set_facecolor('#161b27')
ax.tick_params(colors='#6b7a99', labelsize=9)
for sp in ax.spines.values():
    sp.set_edgecolor('#2a3352')

y_lower = 0
ytick_positions, ytick_labels = [], []

for i, cluster_id in enumerate(unique_clusters):
    mask    = chosen_labels == cluster_id
    vals    = np.sort(sample_sil[mask])        # sort so the shape looks like a knife
    y_upper = y_lower + len(vals)

    ax.fill_betweenx(np.arange(y_lower, y_upper), 0, vals,
                     facecolor=COLORS[i], alpha=0.75)

    ytick_positions.append((y_lower + y_upper) / 2)
    mean_kwh = df.loc[mask, 'kWh'].mean()
    ytick_labels.append(f'Cluster {cluster_id}\n(avg {mean_kwh:.3f} kWh)')
    y_lower = y_upper + 40   # visual gap between clusters

ax.axvline(global_sil, color='white', lw=1.2, ls='--', alpha=0.6,
           label=f'Global mean = {global_sil:.3f}')
ax.set_yticks(ytick_positions)
ax.set_yticklabels(ytick_labels, fontsize=8.5, color='#c9d1e0')
ax.set_xlabel('Silhouette coefficient  (negative = probably in wrong cluster)',
              color='#6b7a99', fontsize=9)
ax.set_title(f'Step 2 — Per-sample silhouette for k={n_clusters}  '
             f'(global score = {global_sil:.3f})',
             color='#e2e8f0', fontsize=12, fontweight='bold', pad=12)
ax.legend(fontsize=9, framealpha=0, labelcolor='#c9d1e0')
ax.set_xlim(-0.3, 1.0)

plt.tight_layout()
plt.savefig(f'{FIGURE_DIR}/2_knife_plot.png', dpi=150, bbox_inches='tight',
            facecolor=fig.get_facecolor())
plt.close()

# ══════════════════════════════════════════════════════════════════════════
# PART 3 — Cluster interpretation tables
# Shows what each cluster means in real electricity terms
# ══════════════════════════════════════════════════════════════════════════

cluster_summary = (
    df.groupby("cluster")["kWh"]
    .agg(
        count="count",
        mean="mean",
        median="median",
        std="std",
        min="min",
        max="max",
    )
    .round(3)
)

print("\nCluster summary by electricity consumption:")
print(cluster_summary)


cluster_time_summary = (
    df.groupby("cluster")[
        ["kWh", "hour", "weekday", "month"]
    ]
    .mean(numeric_only=True)
    .round(3)
)

print("\nCluster average time profile:")
print(cluster_time_summary)

# ── NEW: summary of the engineered features per cluster ───────────────────
# This table tells you *why* the clusters differ, not just *when* they occur.
# Look for large differences between clusters in daily_total_kwh (volume),
# evening_peak (evening behaviour), and peak_to_baseline (day shape).
cluster_feature_summary = (
    df.groupby("cluster")[
        ['daily_total_kwh', 'evening_peak', 'night_baseline',
         'peak_to_baseline']
    ]
    .mean()
    .round(3)
)

print("\nCluster average engineered features:")
print(cluster_feature_summary)

# ══════════════════════════════════════════════════════════════════════════
# PART 4 — Centroid heatmap
#
# What is a centroid?
#   KMeans represents each cluster by its "centroid" — the average position
#   of all points in that cluster in feature space. After scaling, centroid
#   values tell you how far above or below the median each cluster sits for
#   each feature.
#
# How to read this heatmap:
#   • Each ROW is one cluster.
#   • Each COLUMN is one feature (after RobustScaler).
#   • A WARM (red/orange) cell means that cluster scores HIGH on that feature
#     relative to all other clusters.
#   • A COOL (blue) cell means that cluster scores LOW.
#   • A near-WHITE cell means that cluster is close to the median — unremarkable.
#
# What to look for:
#   • Features where ALL clusters have similar colour → weak separator, probably
#     not contributing much to the clustering.
#   • Features where ONE cluster stands out (very warm or very cool) → that
#     feature is a strong driver for that cluster's identity.
#   • Features where there is a clear gradient across clusters → ordinal
#     separation (e.g. cluster 0 = low, cluster 3 = high).
#
# Why use RobustScaler values and not raw values?
#   Raw kWh and daily_total_kwh are on completely different scales (0.2 vs 35).
#   Plotting raw centroids would make the small features invisible.
#   The scaled values put everything on the same ±N scale so you can compare
#   features side by side fairly.
# ══════════════════════════════════════════════════════════════════════════

# Re-fit KMeans at the chosen k so we have clean centroid coordinates.
# We use the same X (already scaled) and same random_state for reproducibility.
# Note: chosen_k is set to 4 here to match the existing pipeline clustering.
#       Once you update clustering.py to use k=3, change this to 3.
chosen_k = n_clusters   # reuses the cluster count from Part 2

km_final = KMeans(n_clusters=chosen_k, random_state=42, n_init='auto')
km_final.fit(X)

# cluster_centers_ shape: (n_clusters, n_features)
# Each row is one centroid; each column corresponds to a feature in FEATURE_COLS.
centroids = km_final.cluster_centers_

# Sort clusters by their mean kWh centroid value (column 0 = kWh after scaling).
# This puts the "low consumption" cluster at the top and "high" at the bottom,
# making the heatmap easier to interpret visually.
order = centroids[:, 0].argsort()
centroids_sorted = centroids[order]

# Human-readable feature labels for the x-axis.
# Shorter than the column names so they don't overlap.
feature_labels = [
    'kWh', 'hour\nsin', 'hour\ncos',
    'wday\nsin', 'wday\ncos',
    'daily\ntotal', 'eve\npeak', 'night\nbase', 'peak/\nbase',
]

fig, ax = plt.subplots(figsize=(11, 4), facecolor='#0f1117')
ax.set_facecolor('#161b27')
fig.suptitle(
    'Step 4 — Centroid heatmap: which features define each cluster?',
    color='#e2e8f0', fontsize=13, fontweight='bold', y=1.02
)

# imshow renders the matrix as a colour grid.
# vmin/vmax are set symmetrically around 0 so white = median,
# red = above median, blue = below median.
im = ax.imshow(
    centroids_sorted,
    aspect='auto',
    cmap='RdYlBu_r',   # red = high, blue = low, white = median
    vmin=-2, vmax=2
)

# Axis labels
ax.set_xticks(range(len(FEATURE_COLS)))
ax.set_xticklabels(feature_labels, color='#c9d1e0', fontsize=8.5)
ax.set_yticks(range(chosen_k))
ax.set_yticklabels(
    [f'Cluster {order[i]}' for i in range(chosen_k)],
    color='#c9d1e0', fontsize=9
)
ax.tick_params(colors='#6b7a99')
for sp in ax.spines.values():
    sp.set_edgecolor('#2a3352')

# Annotate each cell with its numeric value so you can read exact magnitudes.
for i in range(chosen_k):
    for j in range(len(FEATURE_COLS)):
        val = centroids_sorted[i, j]
        # Choose black or white text depending on cell brightness
        text_color = 'black' if abs(val) < 1.2 else 'white'
        ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                fontsize=7.5, color=text_color)

# Colour bar: explains what the colour scale means
cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
cbar.ax.tick_params(colors='#6b7a99', labelsize=8)
cbar.set_label('Scaled centroid value\n(0 = median, +2 = well above, −2 = well below)',
               color='#6b7a99', fontsize=8)

plt.tight_layout()
plt.savefig(f'{FIGURE_DIR}/3_centroid_heatmap.png', dpi=150, bbox_inches='tight',
            facecolor=fig.get_facecolor())
plt.close()

print("\nCentroid heatmap saved → 3_centroid_heatmap.png")
