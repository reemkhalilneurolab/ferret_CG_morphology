"""Bootstrap stability of the candidate combined representations.

The reduced five-descriptor representation is selected by a rule fixed in
advance (normalise for unit commensurability; of the most correlated descriptor
pair, Branching-Leaf at Spearman rho = 0.887, retain the member with the higher
detection score). This script checks that its advantage is not an artefact of
the particular 117-neuron sample.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

from _common import (PLOTS, combine, load_classes, load_detection_weights,
                     load_matrices, ward_clusters)

OUT = PLOTS / "descriptor_normalisation"
OUT.mkdir(parents=True, exist_ok=True)
RNG = np.random.default_rng(20260819)
N_BOOT = 500

matrices = load_matrices()
classes = load_classes()
weights = load_detection_weights()
names = list(matrices)
index = matrices[names[0]].index
labels = pd.Series(classes.loc[index, "class"].to_numpy(), index=index)

candidates = {
    "submitted_unnormalised_six": (names, "none"),
    "normalised_six": (names, "mean"),
    "normalised_reduced_five_drop_branching": (
        [n for n in names if n != "Branching_Pattern"], "mean"),
    "normalised_reduced_five_drop_leaf": (
        [n for n in names if n != "Leaf"], "mean"),
    "spread_only": (["Spread"], "mean"),
}

combined_full = {k: combine(matrices, s, sc, weights)
                 for k, (s, sc) in candidates.items()}

records = []
positions = np.arange(len(index))
for _ in range(N_BOOT):
    # Subsample without replacement so the distance submatrix stays a valid
    # metric (a bootstrap with replacement creates zero-distance duplicates).
    take = RNG.choice(positions, size=int(0.8 * len(positions)), replace=False)
    sub_index = index[take]
    truth = labels.loc[sub_index].to_numpy()
    row = {}
    for label, matrix in combined_full.items():
        sub = matrix.loc[sub_index, sub_index]
        row[label] = adjusted_rand_score(truth, ward_clusters(sub, 2))
    records.append(row)

boot = pd.DataFrame(records)
summary = pd.DataFrame({
    "ARI_full_sample": {k: adjusted_rand_score(labels.to_numpy(),
                                               ward_clusters(v, 2))
                        for k, v in combined_full.items()},
    "ARI_boot_mean": boot.mean(),
    "ARI_boot_sd": boot.std(),
    "ARI_boot_p05": boot.quantile(0.05),
    "ARI_boot_p95": boot.quantile(0.95),
})
summary.to_csv(OUT / "representation_stability.csv")
print(f"80% subsampling, {N_BOOT} replicates, Ward k=2")
print(summary.to_string(float_format=lambda v: f"{v:.4f}"))

best = "normalised_reduced_five_drop_branching"
print(f"\nProportion of replicates where {best} beats each alternative:")
for other in boot.columns:
    if other == best:
        continue
    print(f"  vs {other:42s} {(boot[best] > boot[other]).mean():.3f}")
