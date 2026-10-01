"""Generate a publication-quality flowchart summarising the two PERMANOVA tests.

The chart shows the two-question structure:
  1. Does tier (V1/V2 vs PMLS/PLLS/21a) explain morphological variation?
  2. Within mid-level areas, does specific area explain variation?

A footnote notes that V1 and V2 are pooled because the source archive did not
distinguish them individually.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

from _common import PLOTS, RUN

REPO = Path(__file__).resolve().parents[2]
OUT  = PLOTS / "permanova_flowchart"
OUT.mkdir(parents=True, exist_ok=True)

# Read the figures from run_variance_partition.py rather than restating them.
# They were previously hardcoded here, which let the chart keep reporting an
# earlier run's numbers after the analysis was rerun.
_stats = pd.read_csv(PLOTS / "variance_partition" / "permanova_variance_partition.csv")
_tier = _stats[_stats["contrast"].str.startswith("between tier")].iloc[0]
_area = _stats[_stats["contrast"].str.startswith("among areas")].iloc[0]


def _fmt(row):
    # Match the precision used in the manuscript: one decimal for the
    # large between-tier F, two for the small among-area F.
    f_txt = f"{row.pseudo_F:.1f}" if row.pseudo_F >= 10 else f"{row.pseudo_F:.2f}"
    return (f"Variance explained: {row.R2 * 100:.1f} %\n"
            f"pseudo- F = {f_txt},   p = {row.p_perm:.4g}")


TIER_TEXT = _fmt(_tier)
AREA_TEXT = _fmt(_area)
RATIO = _tier.R2 / _area.R2

# ── colour palette (matches paper) ──────────────────────────────────────────
C_NEUTRAL = "#f0f0f0"   # light grey  – top / conclusion boxes
C_V1V2    = "#d6e8f7"   # light blue  – V1/V2 side
C_PMLS    = "#fde8d8"   # light orange – mid-level side
C_BORDER  = "#333333"
C_ARROW   = "#555555"
C_SIG     = "#1a6e1a"   # dark green  – significant result
C_NS      = "#8b0000"   # dark red    – non-significant result

FS_TITLE  = 9.5
FS_BODY   = 8.5
FS_STAT   = 8.0
FS_FOOT   = 7.0

def rounded_box(ax, x, y, w, h, text, facecolor, fontsize=FS_BODY,
                textcolor="black", edgecolor=C_BORDER, lw=0.8, bold=False,
                valign="center"):
    box = FancyBboxPatch((x - w/2, y - h/2), w, h,
                         boxstyle="round,pad=0.02",
                         facecolor=facecolor, edgecolor=edgecolor,
                         linewidth=lw, zorder=3)
    ax.add_patch(box)
    weight = "bold" if bold else "normal"
    ax.text(x, y, text, ha="center", va=valign, fontsize=fontsize,
            color=textcolor, weight=weight, zorder=4,
            multialignment="center",
            wrap=True)

def arrow(ax, x0, y0, x1, y1):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color=C_ARROW,
                                lw=1.0, mutation_scale=10),
                zorder=2)

# ── canvas ───────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7.5, 8.5))
ax.set_xlim(0, 10)
ax.set_ylim(0, 12)
ax.axis("off")

# ── TOP BOX – input ──────────────────────────────────────────────────────────
rounded_box(ax, x=5, y=11.1, w=5.5, h=0.9,
            text="117 reconstructed CG neurons  ·  7 Sholl descriptors combined",
            facecolor=C_NEUTRAL, bold=True, fontsize=FS_TITLE)

# ── two question boxes ────────────────────────────────────────────────────────
arrow(ax, 5, 10.65, 2.6, 9.85)   # left branch
arrow(ax, 5, 10.65, 7.4, 9.85)   # right branch

rounded_box(ax, x=2.6, y=9.35, w=4.6, h=0.95,
            text="Q1  Does dendritic morphology differ\nbetween cortical tiers?",
            facecolor=C_V1V2, bold=True, fontsize=FS_BODY)

rounded_box(ax, x=7.4, y=9.35, w=4.6, h=0.95,
            text="Q2  Within the mid-level tier, does\nmorphology differ by area?",
            facecolor=C_PMLS, bold=True, fontsize=FS_BODY)

# ── group breakdown boxes ─────────────────────────────────────────────────────
arrow(ax, 2.6, 8.87, 2.6, 8.30)
arrow(ax, 7.4, 8.87, 7.4, 8.25)

rounded_box(ax, x=2.6, y=7.7, w=4.2, h=1.1,
            text="V1/V2†:  67 neurons\nvs  PMLS/PLLS/21a:  50 neurons\n"
                 "─────────────────────────────\n"
                 "Note: V1 and V2 could not be separated —\n"
                 "the original dataset labels them together",
            facecolor=C_V1V2, fontsize=FS_FOOT)

rounded_box(ax, x=7.4, y=7.8, w=4.2, h=0.85,
            text="PMLS: 20  ·  PLLS: 20  ·  Area 21a: 10",
            facecolor=C_PMLS, fontsize=FS_BODY)

# ── PERMANOVA result boxes ────────────────────────────────────────────────────
arrow(ax, 2.6, 7.14, 2.6, 6.55)
arrow(ax, 7.4, 7.37, 7.4, 6.55)

rounded_box(ax, x=2.6, y=6.1, w=4.2, h=0.85,
            text=TIER_TEXT,
            facecolor=C_V1V2, fontsize=FS_STAT)

rounded_box(ax, x=7.4, y=6.1, w=4.2, h=0.85,
            text=AREA_TEXT,
            facecolor=C_PMLS, fontsize=FS_STAT)

# ── conclusion boxes ──────────────────────────────────────────────────────────
arrow(ax, 2.6, 5.67, 2.6, 4.9)
arrow(ax, 7.4, 5.67, 7.4, 4.9)

rounded_box(ax, x=2.6, y=4.45, w=4.2, h=0.85,
            text="Tier structure is a reliable organising\nprinciple for CG dendritic morphology",
            facecolor=C_SIG, textcolor="white", bold=True, fontsize=FS_BODY)

rounded_box(ax, x=7.4, y=4.45, w=4.2, h=0.85,
            text="No detectable morphological difference\namong PMLS, PLLS and area 21a",
            facecolor=C_NS, textcolor="white", bold=True, fontsize=FS_BODY)

# ── between-tier ratio note ───────────────────────────────────────────────────
ax.annotate("", xy=(6.9, 6.1), xytext=(3.1, 6.1),
            arrowprops=dict(arrowstyle="<->", color="#666666",
                            lw=0.8, mutation_scale=8), zorder=2)
ax.text(5.0, 6.38, f"Between-tier variation exceeds\nwithin-tier area variation  ×{RATIO:.1f}",
        ha="center", va="bottom", fontsize=FS_FOOT, color="#444444",
        style="italic")

# ── footnote ─────────────────────────────────────────────────────────────────
ax.text(0.15, 0.18,
        "† V1 and V2 are pooled as a single group throughout this study. "
        "The source archive assigns these\n"
        "  neurons to layer 6 visual cortex (Brodmann areas 17 and 18) "
        "without distinguishing the two areas\n"
        "  individually; the original depositors did not provide area labels "
        "at that resolution.",
        ha="left", va="bottom", fontsize=FS_FOOT, color="#555555",
        transform=ax.transData)

plt.tight_layout(pad=0.3)
out_path = OUT / "permanova_flowchart.png"
fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
plt.close()
print(f"Saved to {out_path}")
