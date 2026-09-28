"""Generate Figure 4 replacement: per-neuron median box plots.

The original histogram figure pooled all step-function values from all nodes
of all neurons, treating each node-level value as an independent observation.
PMLS neurons have ~1.7x more nodes than V1/V2 neurons, so despite having fewer
neurons they contributed 29% more histogram observations, skewing the figure.

This script instead computes one summary per neuron (the median of its
step-function values) and plots box plots. Each box therefore represents
exactly one observation per neuron: 67 for V1/V2 and 50 for PMLS/PLLS/21a.
"""
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests

REPO = Path(__file__).resolve().parents[2]
RUN  = REPO / "figures" / "Briggs_7desc_20260820"
OUT  = RUN / "plots" / "figure4_replacement"
OUT.mkdir(parents=True, exist_ok=True)

# ── load data ────────────────────────────────────────────────────────────────
with open(RUN / "all_descriptors.pkl", "rb") as f:
    all_descriptors = pickle.load(f)

cohort = pd.read_csv(RUN / "neuron_class_df.csv")
name_to_class = dict(zip(cohort["neuron_name"], cohort["class"]))

# The seven descriptors used in the paper (Sholl-TMD and Taper excluded)
DESCRIPTORS = ["Tortuosity", "Branching_Pattern", "Wiring", "Flux",
               "Leaf", "Spread", "Energy"]
LABELS = ["Tortuosity", "Branching\nPattern", "Wiring", "Flux",
          "Leaf Index", "Spread", "Energy"]

# ── compute per-neuron medians ────────────────────────────────────────────────
records = []
for desc in DESCRIPTORS:
    if desc not in all_descriptors:
        print(f"WARNING: {desc} not found in all_descriptors; skipping")
        continue
    for neuron_name, step_values in all_descriptors[desc].items():
        if neuron_name not in name_to_class:
            continue
        # step_values is a list of (radius, value) tuples
        vals = [v[1] for v in step_values if v[1] != 0]
        if not vals:
            continue
        records.append({
            "descriptor": desc,
            "neuron":     neuron_name,
            "class":      name_to_class[neuron_name],
            "median":     np.median(vals),
        })

df = pd.DataFrame(records)
print(f"Records: {len(df)}")
print(df.groupby(["descriptor","class"]).size().unstack())

# ── run stats (Mann-Whitney + BH) per descriptor ─────────────────────────────
stat_rows = []
for desc in DESCRIPTORS:
    sub = df[df["descriptor"] == desc]
    g1 = sub[sub["class"] == "V1-V2"]["median"].values
    g2 = sub[sub["class"] == "PMLS-PLLS"]["median"].values
    if len(g1) == 0 or len(g2) == 0:
        continue
    stat, p = mannwhitneyu(g1, g2, alternative="two-sided")
    stat_rows.append({"descriptor": desc, "p_raw": p})

stat_df = pd.DataFrame(stat_rows)
_, p_adj, _, _ = multipletests(stat_df["p_raw"], method="fdr_bh")
stat_df["p_adj"] = p_adj
stat_df.to_csv(OUT / "per_neuron_median_stats.csv", index=False)
print("\nPer-neuron median Mann-Whitney (BH corrected):")
print(stat_df.to_string(index=False))

# ── plotting ──────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.size":        16,
    "axes.titlesize":   18,
    "axes.labelsize":   16,
    "xtick.labelsize":  14,
    "ytick.labelsize":  14,
    "legend.fontsize":  15,
})

CLASS_ORDER  = ["V1-V2", "PMLS-PLLS"]
CLASS_LABEL  = {"V1-V2": "V1/V2", "PMLS-PLLS": "PMLS/PLLS/21a"}
CLASS_COLOR  = {"V1-V2": "#2c7fb8", "PMLS-PLLS": "#f77189"}
CLASS_HATCH  = {"V1-V2": "//", "PMLS-PLLS": "."}

n_desc = len(DESCRIPTORS)
ncols  = 4
nrows  = int(np.ceil(n_desc / ncols))

fig, axes = plt.subplots(nrows=nrows, ncols=ncols,
                         figsize=(4.2 * ncols, 5.0 * nrows))
axes = np.atleast_1d(axes).flatten()

p_lookup = dict(zip(stat_df["descriptor"], stat_df["p_adj"]))

for idx, (desc, label) in enumerate(zip(DESCRIPTORS, LABELS)):
    ax = axes[idx]
    sub = df[df["descriptor"] == desc]

    data_groups = [sub[sub["class"] == cls]["median"].values
                   for cls in CLASS_ORDER]
    positions   = [1, 2]

    bp = ax.boxplot(
        data_groups,
        positions=positions,
        widths=0.5,
        patch_artist=True,
        notch=False,
        showfliers=True,
        flierprops=dict(marker="o", markersize=3, alpha=0.5,
                        markeredgewidth=0.5),
        medianprops=dict(color="black", linewidth=2),
        whiskerprops=dict(linewidth=1.2),
        capprops=dict(linewidth=1.2),
    )

    for patch, cls in zip(bp["boxes"], CLASS_ORDER):
        patch.set_facecolor(CLASS_COLOR[cls])
        patch.set_hatch(CLASS_HATCH[cls])
        patch.set_edgecolor("black")
        patch.set_alpha(0.8)

    # p-value annotation
    p = p_lookup.get(desc, np.nan)
    if p < 0.001:
        p_str = f"p = {p:.2e}"
    elif p < 0.05:
        p_str = f"p = {p:.3f}"
    else:
        p_str = f"p = {p:.2f} (n.s.)"

    y_max = max(np.percentile(g, 95) for g in data_groups if len(g))
    y_min = min(np.percentile(g, 5)  for g in data_groups if len(g))
    y_range = y_max - y_min
    ax.text(1.5, y_max + 0.05 * y_range, p_str,
            ha="center", va="bottom", fontsize=15)

    ax.set_title(label, fontsize=18, pad=8)
    ax.set_xticks([])
    ax.set_ylabel("Median step-function value", fontsize=15)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(False)

# legend in first panel
handles = [
    mpatches.Patch(facecolor=CLASS_COLOR[cls], hatch=CLASS_HATCH[cls],
                   edgecolor="black", alpha=0.8, label=CLASS_LABEL[cls])
    for cls in CLASS_ORDER
]
axes[0].legend(handles=handles, frameon=False, fontsize=15, loc="upper right")

# hide unused panels
for i in range(n_desc, len(axes)):
    axes[i].set_visible(False)

fig.suptitle(
    "Per-neuron median Sholl descriptor values by cortical tier\n"
    "(one observation per neuron; n = 67 V1/V2, 50 PMLS/PLLS/21a)",
    fontsize=18, y=1.02
)
plt.tight_layout()

out_path = OUT / "rev_Figure4_descriptor_boxplots.png"
fig.savefig(out_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"\nFigure saved to {out_path}")
