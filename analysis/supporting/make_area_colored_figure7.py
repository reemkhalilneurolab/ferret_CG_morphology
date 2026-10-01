"""Standalone, non-manuscript variant of Figure 7: colored by individual
mid-level area (PMLS / PLLS / 21a) instead of the pooled mid-level class.

Produces both representations (topological descriptors and standard
morphometrics), each as a PCA/PCoA scatter and a dendrogram with an
area-colored bar.

V1/V2 remains a single color because the archive does not separate V1 from V2.
Area is inferred from the SWC filename (neurotopo.area_labels.infer_area_label),
the same method already used elsewhere in this repo for area-level analyses
(e.g. run_variance_partition.py's area PERMANOVA).

Output goes to figures/results/plots/area_colored_review/ and is NOT referenced
by any manuscript figure or copied into the manuscript repo.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from analysis.lmeasure_analysis import to_per_object
from analysis.supporting._common import PICKLES, REPO, RUN, load_classes
from neurotopo.area_labels import infer_area_label
import neurotopo.utils as utl

OUT = RUN / "plots" / "area_colored_review"
OUT.mkdir(parents=True, exist_ok=True)

AREA_COLOUR = {
    "V1/V2": "#2c7fb8",
    "PMLS": "#d95f0e",
    "PLLS": "#31a354",
    "21A": "#984ea3",
}
AREA_ORDER = ("V1/V2", "PMLS", "PLLS", "21A")


def area_of(name, cls):
    if cls == "V1-V2":
        return "V1/V2"
    label = infer_area_label(name)
    return label if label in ("PMLS", "PLLS", "21A") else "Unknown"


def pca_scatter(coords, areas, binary_labels, variance, title, filename):
    fig, ax = plt.subplots(figsize=(6.5, 6.0))
    for area in AREA_ORDER:
        sel = (areas == area).to_numpy()
        if sel.sum() == 0:
            continue
        ax.scatter(coords[sel, 0], coords[sel, 1], s=30, alpha=0.85,
                   c=AREA_COLOUR[area], edgecolors="white", linewidths=0.4,
                   label=f"{area} (n={sel.sum()})")

    model = LogisticRegression(max_iter=5000).fit(coords, binary_labels)
    pad_x = 0.05 * np.ptp(coords[:, 0])
    pad_y = 0.05 * np.ptp(coords[:, 1])
    xs = np.linspace(coords[:, 0].min() - pad_x, coords[:, 0].max() + pad_x, 400)
    ys = np.linspace(coords[:, 1].min() - pad_y, coords[:, 1].max() + pad_y, 400)
    gx, gy = np.meshgrid(xs, ys)
    zz = model.decision_function(np.c_[gx.ravel(), gy.ravel()]).reshape(gx.shape)
    ax.contour(gx, gy, zz, levels=[0], colors="k", linestyles="--", linewidths=1.2)

    ax.set_xlabel(f"PC1 ({variance[0] * 100:.1f}%)")
    ax.set_ylabel(f"PC2 ({variance[1] * 100:.1f}%)")
    ax.set_title(title, fontsize=10)
    ax.legend(frameon=False, fontsize=9, loc="best")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / filename, dpi=300)
    plt.close(fig)


def dendrogram_with_area_bar(condensed, index, areas, title, filename, swap_root=False):
    Z = linkage(condensed, method="ward")

    if swap_root:
        # Swap only the two children of the root merge (Z's last row), so the
        # two top-level branches trade places on the page. This does NOT
        # mirror the internal leaf order within either branch -- each
        # subtree's own internal arrangement is untouched, unlike a full
        # left-right flip of the whole axis.
        Z = Z.copy()
        Z[-1, 0], Z[-1, 1] = Z[-1, 1], Z[-1, 0]

    fig, (ax_dendro, ax_bar) = plt.subplots(
        2, 1, figsize=(16, 6.5), gridspec_kw={"height_ratios": [10, 1]})
    dd = dendrogram(Z, no_labels=True, ax=ax_dendro,
                    color_threshold=0, above_threshold_color="#4477aa")
    ax_dendro.set_xticks([])
    ax_dendro.set_ylabel("Distance")
    ax_dendro.spines[["top", "right", "bottom"]].set_visible(False)
    dendro_xlim = ax_dendro.get_xlim()

    leaf_order = [index[i] for i in dd["leaves"]]
    bar_colours = [AREA_COLOUR[areas.loc[n]] for n in leaf_order]
    ax_bar.imshow([list(range(len(bar_colours)))], aspect="auto",
                 cmap=matplotlib.colors.ListedColormap(bar_colours),
                 extent=[dendro_xlim[0], dendro_xlim[1], 0, 1])
    ax_bar.set_xlim(dendro_xlim)
    ax_bar.set_yticks([])
    ax_bar.set_xticks([])
    for spine in ax_bar.spines.values():
        spine.set_visible(True)

    handles = [plt.Rectangle((0, 0), 1, 1, color=AREA_COLOUR[a]) for a in AREA_ORDER]
    fig.legend(handles, ["V1/V2", "PMLS", "PLLS", "21a"], loc="upper right",
              ncol=4, frameon=False, fontsize=10, bbox_to_anchor=(0.98, 0.98))
    fig.suptitle(title, fontsize=11)
    fig.subplots_adjust(top=0.88, hspace=0.05)
    fig.savefig(OUT / filename, dpi=300)
    plt.close(fig)


def main():
    classes = load_classes()
    index = classes.index
    binary_labels = classes["class"].to_numpy()
    areas = pd.Series([area_of(n, c) for n, c in zip(index, classes["class"])], index=index)

    print("Area counts:")
    print(areas.value_counts())

    # ==================== descriptor representation ==========================
    combined = pd.read_pickle(PICKLES / "combined_detection_weighted_matrix.pkl")
    combined = combined.loc[index, index]

    squared = np.asarray(combined, dtype=float) ** 2
    n = len(squared)
    centring = np.eye(n) - np.ones((n, n)) / n
    gram = -0.5 * centring @ squared @ centring
    values, vectors = np.linalg.eigh(gram)
    order = np.argsort(values)[::-1]
    values, vectors = values[order], vectors[:, order]
    positive = values > 0
    desc_coords = vectors[:, :2] * np.sqrt(values[:2])
    desc_variance = values[:2] / values[positive].sum()

    pca_scatter(desc_coords, areas, binary_labels, desc_variance,
               "Topological descriptors, colored by individual area",
               "descriptor_pca_by_area.png")

    desc_condensed = squareform(np.asarray(combined, dtype=float), checks=False)
    dendrogram_with_area_bar(desc_condensed, index, areas,
                             "Descriptor dendrogram, colored bar by individual area",
                             "descriptor_dendrogram_by_area.png")

    # ==================== morphometric representation ========================
    raw = pd.read_csv(REPO / "data" / "Briggs" / "lmeasure_data.csv")
    raw.columns = [c.strip() for c in raw.columns]
    raw["neuron_name"] = (raw["Filename"].astype(str)
                          .str.replace(".swc", "", regex=False)
                          .str.replace(".CNG", "", regex=False)
                          .str.replace(" ", "_", regex=False)
                          .str.replace(".", "_", regex=False)
                          .str.replace("-", "_", regex=False))
    raw = raw[raw["neuron_name"].isin(index)].drop_duplicates("neuron_name")
    raw = raw.set_index("neuron_name").loc[index]
    features = to_per_object(raw.select_dtypes(include=[np.number]))

    # PCA panel (D): matches analysis/supporting/make_manuscript_figures.py
    # exactly -- to_per_object then StandardScaler, no variance filter.
    scaled_pca = StandardScaler().fit_transform(features)
    pca = PCA(n_components=2).fit(scaled_pca)
    morph_coords = pca.transform(scaled_pca)
    pca_scatter(morph_coords, areas, binary_labels, pca.explained_variance_ratio_,
               "Standard morphometrics, colored by individual area",
               "morphometric_pca_by_area.png")

    # Dendrogram panel (C): matches the notebook's morphometric-dendrogram
    # cell exactly -- to_per_object, THEN drop_constant_and_low_variance_columns
    # (40 -> 38 variables), THEN StandardScaler. This is a different feature
    # set from the PCA panel above; that inconsistency already exists in the
    # manuscript-generating pipeline (PCA and dendrogram were written by
    # different scripts) and is reproduced here deliberately, not introduced.
    features_filtered = utl.drop_constant_and_low_variance_columns(features)
    scaled_dendro = StandardScaler().fit_transform(features_filtered)
    morph_condensed = pdist(scaled_dendro, metric="euclidean")
    dendrogram_with_area_bar(morph_condensed, index, areas,
                             "Morphometric dendrogram, colored bar by individual area",
                             "morphometric_dendrogram_by_area.png", swap_root=True)

    print(f"\nWritten to {OUT}")


if __name__ == "__main__":
    main()
