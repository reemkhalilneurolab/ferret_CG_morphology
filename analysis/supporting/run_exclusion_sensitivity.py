"""Descriptor-omission sensitivity, recomputed with the correct linkage.

Supersedes plots/descriptor_sensitivity/descriptor_sensitivity_results.csv, which
called scipy's ``linkage`` on the square 117x117 distance matrix. Without
``squareform`` scipy treats each row as an observation vector in 117-dimensional
Euclidean space, producing a partition that does not match the dendrogram
reported in the manuscript.
"""
import pandas as pd
from sklearn.metrics import adjusted_rand_score, silhouette_score

from _common import (PICKLES, PLOTS, RUN, combine, load_classes,
                     load_detection_weights, load_matrices, ward_clusters)

OUT = PLOTS / "descriptor_sensitivity_corrected"
OUT.mkdir(parents=True, exist_ok=True)

matrices = load_matrices()
classes = load_classes()
weights = load_detection_weights()
names = list(matrices)
index = matrices[names[0]].index
labels = classes.loc[index, "class"].to_numpy()

# Verify the corrected pipeline reproduces the manuscript dendrogram.
published = (pd.read_csv(PICKLES / "combined_detection_cluster_assignments.csv")
             .set_index("neuron_name").loc[index, "dendrogram_cluster"].to_numpy())
primary = combine(matrices, names, "none", weights)
check = adjusted_rand_score(published, ward_clusters(primary, 2))
print(f"Corrected linkage vs published dendrogram: ARI = {check:.4f}")
assert check == 1.0, "corrected pipeline must reproduce the published clusters"

scenarios = {
    "all_six_primary": names,
    "drop_tortuosity": [n for n in names if n != "Tortuosity"],
    "drop_branching_pattern": [n for n in names if n != "Branching_Pattern"],
    "drop_wiring": [n for n in names if n != "Wiring"],
    "drop_flux": [n for n in names if n != "Flux"],
    "drop_leaf": [n for n in names if n != "Leaf"],
    "drop_spread": [n for n in names if n != "Spread"],
    "spread_only": ["Spread"],
    "representative_correlated_group_leaf": ["Tortuosity", "Wiring", "Leaf", "Spread"],
}

rows = []
for label, subset in scenarios.items():
    combined = combine(matrices, subset, "none", weights)
    predicted = ward_clusters(combined, 2)
    combined.to_pickle(OUT / f"{label}_combined_matrix.pkl")
    rows.append({
        "scenario": label,
        "descriptors": ";".join(subset),
        "n_descriptors": len(subset),
        "ARI_vs_class": adjusted_rand_score(labels, predicted),
        "silhouette_vs_class": silhouette_score(combined.values, labels,
                                                metric="precomputed"),
        "agreement_with_primary_ARI": adjusted_rand_score(published, predicted),
    })

results = pd.DataFrame(rows)
results.to_csv(OUT / "descriptor_sensitivity_results_corrected.csv", index=False)
print()
print(results.drop(columns="descriptors").to_string(
    index=False, float_format=lambda v: f"{v:.4f}"))

(OUT / "README.txt").write_text(
    "Descriptor-omission sensitivity recomputed with squareform-condensed Ward "
    "linkage. The all_six_primary row reproduces the manuscript dendrogram "
    "exactly (ARI = 1.0). Supersedes ../descriptor_sensitivity/, whose ARI "
    "values were computed on an uncondensed distance matrix and correspond to a "
    "different partition (ARI 0.867 against the published clusters).\n\n"
    "Note that spread_only attains a higher ARI than the six-descriptor "
    "combination. This follows from the descriptor scale contributions reported "
    "in ../descriptor_normalisation/: Spread supplies 80.9 percent of the "
    "unnormalised combined distance.\n",
    encoding="utf-8")
