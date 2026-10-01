"""Compare descriptor-scale normalisation schemes for the combined distance matrix.

Two questions: how dependence among descriptors affects the combination
weights, and what the single combined distance matrix actually buys.

The six descriptors carry different physical units -- a dimensionless ratio
(Tortuosity), lengths (Wiring), counts (Branching, Flux, Leaf) and an area
(Spread, volume/path). The L1 distance between two step functions inherits those
units, so an unnormalised sum is dominated by whichever descriptor happens to be
expressed in the largest units.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, silhouette_score

from _common import (PLOTS, combine, load_classes, load_detection_weights,
                     load_matrices, normalise, ward_clusters)

OUT = PLOTS / "descriptor_normalisation"
OUT.mkdir(parents=True, exist_ok=True)

matrices = load_matrices()
classes = load_classes()
weights = load_detection_weights()
names = list(matrices)
index = matrices[names[0]].index
labels = classes.loc[index, "class"].to_numpy()
areas = classes.loc[index, "area"].to_numpy()

# ---------------------------------------------------------------- scale table
rows = []
for name in names:
    values = np.asarray(matrices[name], dtype=float)
    off = values[~np.eye(len(values), dtype=bool)]
    rows.append({"descriptor": name, "mean_distance": off.mean(),
                 "max_distance": off.max(), "detection_weight": weights[name]})
scale = pd.DataFrame(rows)
scale["share_unnormalised_%"] = 100 * scale["mean_distance"] / scale["mean_distance"].sum()
scale = scale.sort_values("share_unnormalised_%", ascending=False)
scale.to_csv(OUT / "descriptor_scale_contributions.csv", index=False)
print("Descriptor contribution to the unnormalised combined distance")
print(scale.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

# --------------------------------------------------- scheme x scenario sweep
scenarios = {
    "all_six": names,
    "drop_tortuosity": [n for n in names if n != "Tortuosity"],
    "drop_branching_pattern": [n for n in names if n != "Branching_Pattern"],
    "drop_wiring": [n for n in names if n != "Wiring"],
    "drop_flux": [n for n in names if n != "Flux"],
    "drop_leaf": [n for n in names if n != "Leaf"],
    "drop_spread": [n for n in names if n != "Spread"],
    "spread_only": ["Spread"],
}

records = []
for scheme in ("none", "mean", "max", "median"):
    for label, subset in scenarios.items():
        for weighted in (True, False):
            combined = combine(matrices, subset, scheme,
                               weights if weighted else None)
            predicted = ward_clusters(combined, 2)
            records.append({
                "normalisation": scheme,
                "scenario": label,
                "detection_weighted": weighted,
                "n_descriptors": len(subset),
                "ARI": adjusted_rand_score(labels, predicted),
                "silhouette": silhouette_score(combined.values, labels,
                                               metric="precomputed"),
            })

results = pd.DataFrame(records)
results.to_csv(OUT / "normalisation_comparison.csv", index=False)

print("\nARI against V1/V2 vs extrastriate labels (Ward, k=2, detection-weighted)")
weighted_only = results[results["detection_weighted"]]
print(weighted_only.pivot(index="scenario", columns="normalisation", values="ARI")
      .reindex(list(scenarios))[["none", "mean", "median", "max"]]
      .to_string(float_format=lambda v: f"{v:.4f}"))

print("\nDoes detection weighting change the partition?")
for scheme in ("none", "mean"):
    on = combine(matrices, names, scheme, weights)
    off = combine(matrices, names, scheme, None)
    agreement = adjusted_rand_score(ward_clusters(on, 2), ward_clusters(off, 2))
    print(f"  {scheme:7s}: ARI(weighted, unweighted) = {agreement:.4f}")

# ------------------------------------------------- cross-tabs for the chosen scheme
for scheme in ("none", "mean"):
    combined = combine(matrices, names, scheme, weights)
    predicted = ward_clusters(combined, 2)
    table = pd.crosstab(pd.Series(predicted, name="cluster"),
                        pd.Series(areas, name="area"))
    table.to_csv(OUT / f"cluster_by_area_{scheme}.csv")
    print(f"\nCluster x area, normalisation={scheme}")
    print(table.to_string())
