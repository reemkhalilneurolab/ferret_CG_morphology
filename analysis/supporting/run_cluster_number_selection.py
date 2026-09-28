"""Cluster-number selection and candidate-representation cross-tabs.

Asks whether reducing the six subtypes to two is simply a
difference in the threshold for determining what constitutes a separate cluster?"
The submitted analysis fixed k=2 by configuration, so the question was never
tested. This sweeps k=2..8 and reports silhouette on the combined distance.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, silhouette_score

from _common import (PLOTS, combine, load_classes, load_detection_weights,
                     load_matrices, ward_clusters)

OUT = PLOTS / "cluster_number_selection"
OUT.mkdir(parents=True, exist_ok=True)

matrices = load_matrices()
classes = load_classes()
weights = load_detection_weights()
names = list(matrices)
index = matrices[names[0]].index
labels = classes.loc[index, "class"].to_numpy()
areas = classes.loc[index, "area"].to_numpy()

# Candidate representations. "reduced" drops Branching Pattern by the
# pre-specified rule: of the most correlated descriptor pair identified in
# plots/descriptor_dependence (Branching-Leaf, Spearman rho = 0.887), retain the
# member with the higher detection score (Leaf 77.91 vs Branching 76.44).
candidates = {
    "submitted_unnormalised_six": (names, "none"),
    "normalised_six": (names, "mean"),
    "normalised_reduced_five": ([n for n in names if n != "Branching_Pattern"], "mean"),
    "spread_only": (["Spread"], "mean"),
}

rows = []
for label, (subset, scheme) in candidates.items():
    combined = combine(matrices, subset, scheme, weights)
    for k in range(2, 9):
        predicted = ward_clusters(combined, k)
        rows.append({
            "representation": label, "k": k,
            "silhouette": silhouette_score(combined.values, predicted,
                                           metric="precomputed"),
            "ARI_vs_tier_labels": adjusted_rand_score(labels, predicted),
        })
sweep = pd.DataFrame(rows)
sweep.to_csv(OUT / "cluster_number_sweep.csv", index=False)

print("Silhouette by number of clusters (Ward on the combined distance)")
print(sweep.pivot(index="k", columns="representation", values="silhouette")
      .to_string(float_format=lambda v: f"{v:.4f}"))
print("\nARI against the two tier labels")
print(sweep.pivot(index="k", columns="representation", values="ARI_vs_tier_labels")
      .to_string(float_format=lambda v: f"{v:.4f}"))

print("\nBest k by silhouette:")
for label in candidates:
    sub = sweep[sweep["representation"] == label]
    best = sub.loc[sub["silhouette"].idxmax()]
    print(f"  {label:28s} k={int(best['k'])}  silhouette={best['silhouette']:.4f}")

# ------------------------------------------------------- k=2 cross-tabs
print("\n" + "=" * 62)
for label, (subset, scheme) in candidates.items():
    combined = combine(matrices, subset, scheme, weights)
    predicted = ward_clusters(combined, 2)
    table = pd.crosstab(pd.Series(predicted, name="cluster"),
                        pd.Series(areas, name="area"))
    tier = pd.crosstab(pd.Series(predicted, name="cluster"),
                       pd.Series(labels, name="tier"))
    table.to_csv(OUT / f"cluster_by_area_k2_{label}.csv")
    ari = adjusted_rand_score(labels, predicted)
    print(f"\n{label}  (k=2, ARI={ari:.4f})")
    print(table.to_string())
    purity = (tier.max(axis=1) / tier.sum(axis=1) * 100).round(1)
    print("  cluster purity %:", purity.to_dict())
