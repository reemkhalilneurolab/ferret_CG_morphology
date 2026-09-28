"""Generate the figure panels used in the manuscript.

Produces two PCA panels drawn in the same style, one for each representation,
each showing only the logistic-regression separating boundary, and selects
representative neurons deterministically for the representative-morphology
figure.

Representatives are the neurons closest to the centroid of each cluster and of
each anatomical class in the principal-component space of the combined distance
matrix, together with the neurons closest to each centroid among the cells whose
cluster assignment differs from their anatomical class.
"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))
from analysis.lmeasure_analysis import to_per_object
from _common import PICKLES, REPO, RUN, load_classes, ward_clusters

OUT = RUN / "plots" / "manuscript_figures"
OUT.mkdir(parents=True, exist_ok=True)

CLASS_COLOUR = {"V1-V2": "#2c7fb8", "PMLS-PLLS": "#d95f0e"}
CLASS_LABEL = {"V1-V2": "V1/V2", "PMLS-PLLS": "PMLS/PLLS/21a"}


def pca_panel(coords, labels, filename, title, variance):
    """Scatter of the first two components with a logistic-regression boundary."""
    fig, ax = plt.subplots(figsize=(5.5, 5.0))
    for cls in ("V1-V2", "PMLS-PLLS"):
        sel = labels == cls
        ax.scatter(coords[sel, 0], coords[sel, 1], s=26, alpha=0.85,
                   c=CLASS_COLOUR[cls], edgecolors="white", linewidths=0.4,
                   label=CLASS_LABEL[cls])

    model = LogisticRegression(max_iter=5000).fit(coords, labels)
    pad_x = 0.05 * np.ptp(coords[:, 0])
    pad_y = 0.05 * np.ptp(coords[:, 1])
    xs = np.linspace(coords[:, 0].min() - pad_x, coords[:, 0].max() + pad_x, 400)
    ys = np.linspace(coords[:, 1].min() - pad_y, coords[:, 1].max() + pad_y, 400)
    gx, gy = np.meshgrid(xs, ys)
    zz = model.decision_function(np.c_[gx.ravel(), gy.ravel()]).reshape(gx.shape)
    ax.contour(gx, gy, zz, levels=[0], colors="k", linestyles="--", linewidths=1.2)

    accuracy = model.score(coords, labels)
    ax.set_xlabel(f"PC1 ({variance[0] * 100:.1f}%)")
    ax.set_ylabel(f"PC2 ({variance[1] * 100:.1f}%)")
    ax.set_title(f"{title}\nlinear separation accuracy {accuracy:.1%}", fontsize=10)
    ax.legend(frameon=False, fontsize=9, loc="best")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / filename, dpi=300)
    plt.close(fig)
    return accuracy


def main():
    classes = load_classes()
    index = classes.index
    labels = classes["class"].to_numpy()

    # --- descriptor representation: PCoA of the combined distance matrix ------
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
    descriptor_coords = vectors[:, :2] * np.sqrt(values[:2])
    descriptor_variance = values[:2] / values[positive].sum()
    acc_d = pca_panel(descriptor_coords, labels, "fig5b_descriptor_pca.png",
                      "Topological descriptors", descriptor_variance)

    # --- morphometric representation ----------------------------------------
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
    scaled = StandardScaler().fit_transform(features)
    pca = PCA(n_components=2).fit(scaled)
    acc_m = pca_panel(pca.transform(scaled), labels, "fig5d_morphometric_pca.png",
                      "Standard morphometric variables",
                      pca.explained_variance_ratio_)

    print(f"descriptor PCA   PC1 {descriptor_variance[0]:.1%} PC2 "
          f"{descriptor_variance[1]:.1%}  separation {acc_d:.1%}")
    print(f"morphometric PCA PC1 {pca.explained_variance_ratio_[0]:.1%} PC2 "
          f"{pca.explained_variance_ratio_[1]:.1%}  separation {acc_m:.1%}")

    # --- deterministic representative neurons --------------------------------
    clusters = ward_clusters(combined, 2)
    frame = pd.DataFrame({"neuron": index, "cls": labels, "cluster": clusters,
                          "PC1": descriptor_coords[:, 0],
                          "PC2": descriptor_coords[:, 1]})

    def nearest(subset, label):
        centroid = subset[["PC1", "PC2"]].mean().to_numpy()
        distance = np.linalg.norm(subset[["PC1", "PC2"]].to_numpy() - centroid, axis=1)
        row = subset.iloc[int(np.argmin(distance))]
        return {"selection": label, "neuron": row["neuron"], "class": row["cls"],
                "cluster": int(row["cluster"])}

    picks = []
    for cluster in sorted(frame["cluster"].unique()):
        picks.append(nearest(frame[frame["cluster"] == cluster],
                             f"cluster {cluster} centroid"))
    for cls in ("V1-V2", "PMLS-PLLS"):
        picks.append(nearest(frame[frame["cls"] == cls],
                             f"{CLASS_LABEL[cls]} class centroid"))

    # Cells whose cluster assignment disagrees with their anatomical class.
    majority = (frame.groupby("cluster")["cls"]
                .agg(lambda s: s.value_counts().idxmax()).to_dict())
    frame["crosses"] = frame.apply(
        lambda r: majority[r["cluster"]] != r["cls"], axis=1)
    for cls in ("V1-V2", "PMLS-PLLS"):
        subset = frame[frame["crosses"] & (frame["cls"] == cls)]
        if len(subset):
            picks.append(nearest(subset, f"{CLASS_LABEL[cls]} in opposite cluster"))

    selection = pd.DataFrame(picks)
    selection.to_csv(OUT / "representative_neurons.csv", index=False)
    print("\nRepresentative neurons")
    print(selection.to_string(index=False))
    print(f"\ncells assigned against their anatomical class: "
          f"{int(frame['crosses'].sum())} of {len(frame)}")


if __name__ == "__main__":
    main()
