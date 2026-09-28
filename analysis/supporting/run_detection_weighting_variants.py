"""Can detection-based weighting be made to do anything on this dataset?

The method as implemented is w_d = mean_C det_d(C), then D = sum_d w_d D_d.
Detection rates are bounded percentages, so on this cohort they span only
65.98-83.38, a ratio of 1.26:1, while the raw descriptor distance matrices span
0.052-322.9, a ratio of 6210:1. The weights therefore cannot influence the
result, and the weighted and unweighted partitions are identical.

This script asks whether any faithful variant of the idea recovers an effect:
normalising the matrices first, and sharpening the weight contrast. The binary
threshold variant is exactly the detection-based feature selection published in
Khalil et al. S4.3, i.e. the limiting case of detection weighting.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, silhouette_score

from _common import (PLOTS, load_classes, load_matrices, normalise,
                     ward_clusters)

OUT = PLOTS / "detection_weighting_variants"
OUT.mkdir(parents=True, exist_ok=True)

matrices = load_matrices()
classes = load_classes()
names = list(matrices)
index = matrices[names[0]].index
labels = classes.loc[index, "class"].to_numpy()

rates = pd.read_csv(PLOTS / "Detection" / "detection_rates.csv")
per_class = (rates[rates["score_type"] == "mean"]
             .pivot(index="descriptor", columns="class", values="value"))
det = {n: float(per_class.loc[n].mean()) for n in names}
det_max = {n: float(per_class.loc[n].max()) for n in names}


def weight_schemes():
    lo, hi = min(det.values()), max(det.values())
    yield "as_implemented", {n: det[n] for n in names}
    for k in (2, 4, 8, 16, 32):
        yield f"power_{k}", {n: (det[n] / hi) ** k for n in names}
    # Rescale the observed range onto [0, 1]: maximum contrast without a threshold.
    yield "minmax_rescaled", {n: (det[n] - lo) / (hi - lo) for n in names}
    # Published S4.3 rule: keep a descriptor if it reaches 80% on any class.
    yield "threshold_80_published_S4.3", {n: float(det_max[n] >= 80) for n in names}
    yield "unweighted", {n: 1.0 for n in names}


def combine(weights, scheme):
    total = None
    for n in names:
        term = normalise(matrices[n], scheme) * weights[n]
        total = term if total is None else total + term
    return pd.DataFrame(total, index=index, columns=index)


baseline = {}
rows = []
for norm in ("none", "mean"):
    baseline[norm] = ward_clusters(combine({n: 1.0 for n in names}, norm), 2)
    for label, weights in weight_schemes():
        if sum(weights.values()) == 0:
            continue
        combined = combine(weights, norm)
        predicted = ward_clusters(combined, 2)
        spread = max(weights.values()) / min(v for v in weights.values() if v > 0)
        rows.append({
            "normalisation": norm,
            "scheme": label,
            "weight_ratio": spread,
            "n_dropped": sum(1 for v in weights.values() if v == 0),
            "ARI_vs_class": adjusted_rand_score(labels, predicted),
            "silhouette": silhouette_score(combined.values, labels,
                                           metric="precomputed"),
            "differs_from_unweighted": adjusted_rand_score(baseline[norm],
                                                           predicted) < 1.0,
        })

results = pd.DataFrame(rows)
results.to_csv(OUT / "detection_weighting_variants.csv", index=False)

for norm in ("none", "mean"):
    print(f"\n=== normalisation = {norm} ===")
    sub = results[results["normalisation"] == norm].drop(columns="normalisation")
    print(sub.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

changed = results[results["differs_from_unweighted"]]
print(f"\nSchemes that change the partition at all: {len(changed)} of {len(results)}")
if len(changed):
    print(changed[["normalisation", "scheme", "ARI_vs_class"]]
          .to_string(index=False, float_format=lambda v: f"{v:.4f}"))
