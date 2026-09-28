"""Compare the three descriptor-combination methods on the Briggs cohort.

Method A  detection-weighted sum   -- what the manuscript currently reports.
                                      Not one of the three methods published in
                                      Khalil et al.; detection is defined there
                                      as a diagnostic and as a feature-selection
                                      filter (S1.4, S4.3), not as a multiplier.
Method B  grid search (S4.2)       -- the published method, re-run correctly.
Method C  equal weights            -- what the submitted odometer effectively
                                      produced, since it never moved Leaf or
                                      Spread off their initial value.
Method D  detection feature selection (S4.3) -- drop descriptors below the
                                      published 80% detection threshold.
"""
import json
from itertools import combinations_with_replacement

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, silhouette_score

from _common import (PLOTS, combine, load_classes, load_detection_weights,
                     load_matrices, normalise, ward_clusters)

OUT = PLOTS / "method_comparison"
OUT.mkdir(parents=True, exist_ok=True)
RNG = np.random.default_rng(20260819)

matrices = load_matrices()
classes = load_classes()
detection = load_detection_weights()
names = list(matrices)
index = matrices[names[0]].index
labels = classes.loc[index, "class"].to_numpy()
areas = classes.loc[index, "area"].to_numpy()

# The grid-search arms are optional. The manuscript combines descriptors by
# detection weighting and reports no grid-search result, so requiring the audit
# output here made the whole comparison unrunnable for no benefit. Run
# run_gridsearch_audit.py first if those arms are wanted.
_grid_path = PLOTS / "gridsearch_audit" / "gridsearch_results.json"
if _grid_path.exists():
    with _grid_path.open() as fh:
        grid = json.load(fh)
else:
    grid = None
    print(f"No grid-search audit at {_grid_path}; skipping those two arms.")

# Published S4.3 rule: exclude a descriptor whose detection is below 80% on all
# classes. Per-class detection rates are in plots/Detection/detection_rates.csv.
rates = pd.read_csv(PLOTS / "Detection" / "detection_rates.csv")
per_class = (rates[rates["score_type"] == "mean"]
             .pivot(index="descriptor", columns="class", values="value"))
selected = [n for n in names if (per_class.loc[n] >= 80).any()]
print(f"S4.3 feature selection at 80%: retained {selected}")

methods = {
    "A_detection_weighted_submitted": (
        combine(matrices, names, "none", detection)),
    "C_equal_weights": (
        combine(matrices, names, "none", {n: 1.0 for n in names})),
    "D_detection_feature_selection": (
        combine(matrices, selected, "none", {n: 1.0 for n in selected})),
}
if grid is not None:
    methods["B_gridsearch_published_raw"] = combine(
        matrices, names, "none", grid["none"]["best_weights"])
    methods["B_gridsearch_published_normalised"] = combine(
        matrices, names, "mean", grid["mean"]["best_weights"])

rows = []
for label, combined in methods.items():
    predicted = ward_clusters(combined, 2)
    table = pd.crosstab(pd.Series(predicted, name="cluster"),
                        pd.Series(areas, name="area"))
    table.to_csv(OUT / f"cluster_by_area_{label}.csv")
    tier = pd.crosstab(pd.Series(predicted, name="cluster"),
                       pd.Series(labels, name="tier"))
    rows.append({
        "method": label,
        "ARI": adjusted_rand_score(labels, predicted),
        "silhouette": silhouette_score(combined.values, labels,
                                       metric="precomputed"),
        "min_cluster_purity_%": float((tier.max(axis=1) / tier.sum(axis=1) * 100).min()),
        "n_misassigned": int((tier.sum(axis=1) - tier.max(axis=1)).sum()),
    })

summary = pd.DataFrame(rows)
summary.to_csv(OUT / "method_comparison.csv", index=False)
print()
print(summary.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

# --------------------------------------------------------------- stability
print("\n80% subsampling, 300 replicates")
positions = np.arange(len(index))
boot = {k: [] for k in methods}
for _ in range(300):
    take = RNG.choice(positions, size=int(0.8 * len(positions)), replace=False)
    sub_index = index[take]
    truth = classes.loc[sub_index, "class"].to_numpy()
    for label, combined in methods.items():
        sub = combined.loc[sub_index, sub_index]
        boot[label].append(adjusted_rand_score(truth, ward_clusters(sub, 2)))

stability = pd.DataFrame({
    "ARI_full": summary.set_index("method")["ARI"],
    "ARI_boot_mean": {k: np.mean(v) for k, v in boot.items()},
    "ARI_boot_sd": {k: np.std(v) for k, v in boot.items()},
    "ARI_boot_p05": {k: np.quantile(v, 0.05) for k, v in boot.items()},
})
stability.to_csv(OUT / "method_stability.csv")
print(stability.to_string(float_format=lambda v: f"{v:.4f}"))

if "B_gridsearch_published_raw" in methods:
    print("\nCluster x area for the published grid-search method (raw)")
    predicted = ward_clusters(methods["B_gridsearch_published_raw"], 2)
    print(pd.crosstab(pd.Series(predicted, name="cluster"),
                      pd.Series(areas, name="area")).to_string())
