"""Resampling stability of the descriptor representation against morphometrics.

Clustering indices computed on the full 117-neuron sample do not indicate how
stable a representation is under a different sample composition. This script
implements the method Table 1 describes: across 400 replicates, 80% of the
neurons (94 of 117) are sampled without replacement, clustering is recomputed
on each subset, and the Adjusted Rand Index against the anatomical labels of
the sampled neurons is recorded, for both the combined descriptor
representation and the standard morphometric (L-Measure) representation.

Sampling is without replacement rather than bootstrap resampling, because a
duplicated neuron would have zero distance to itself and would be joined
automatically by the linkage algorithm.

No script producing Table 1's exact reported values (mean ARI 0.495 (SD 0.126)
for descriptors, 0.378 (SD 0.127) for morphometric, descriptors ahead in 67.8%
of replicates, 5th-percentile ARI 0.339 vs 0.153) could be located anywhere in
this repository, including its full commit history, or elsewhere on the
machine this analysis was prepared on. This script is a fresh implementation
of the method as stated, not a recovery of that original run: a stochastic
resampling procedure will not reproduce the same decimals from a different
random draw even given identical code.
"""
import numpy as np
import pandas as pd
from scipy.spatial.distance import squareform, pdist
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler

from _common import (PLOTS, REPO, combine, load_classes, load_detection_weights,
                     load_matrices, ward_clusters)
from analysis.lmeasure_analysis import to_per_object
import neurotopo.utils as utl

OUT = PLOTS / "representation_stability_vs_morphometric"
OUT.mkdir(parents=True, exist_ok=True)

N_REPLICATES = 400
SAMPLE_FRACTION = 0.8
RNG = np.random.default_rng(20260819)

# ── descriptor representation ────────────────────────────────────────────────
matrices = load_matrices()
classes = load_classes()
weights = load_detection_weights()
combined = combine(matrices, list(matrices), "mean", weights)

# ── morphometric representation ──────────────────────────────────────────────
lm = pd.read_csv(REPO / "data" / "Briggs" / "lmeasure_data.csv")
lm["Filename"] = (lm["Filename"].str.replace(r"\.swc$", "", regex=True)
                                .str.replace(".CNG", "", regex=False)
                                .str.replace(r"[ .\-]", "_", regex=True))
lm = lm.set_index("Filename")
numeric = lm.select_dtypes(include=["number"])
numeric = to_per_object(numeric)
numeric = utl.drop_constant_and_low_variance_columns(numeric)
numeric = numeric.loc[[n for n in combined.index if n in numeric.index]]

scaled = pd.DataFrame(StandardScaler().fit_transform(numeric),
                      index=numeric.index, columns=numeric.columns)
lm_distance = pd.DataFrame(squareform(pdist(scaled.values, metric="euclidean")),
                           index=numeric.index, columns=numeric.index)

# ── shared cohort and resampling ─────────────────────────────────────────────
shared_index = [n for n in combined.index if n in lm_distance.index]
combined_shared = combined.loc[shared_index, shared_index]
lm_shared = lm_distance.loc[shared_index, shared_index]
labels_shared = classes.loc[shared_index, "class"]
positions = np.arange(len(shared_index))
sample_size = int(SAMPLE_FRACTION * len(positions))

records = {"descriptors": [], "morphometric": []}
for _ in range(N_REPLICATES):
    take = RNG.choice(positions, size=sample_size, replace=False)
    sub_index = [shared_index[i] for i in take]
    truth = labels_shared.loc[sub_index].to_numpy()
    records["descriptors"].append(adjusted_rand_score(
        truth, ward_clusters(combined_shared.loc[sub_index, sub_index], 2)))
    records["morphometric"].append(adjusted_rand_score(
        truth, ward_clusters(lm_shared.loc[sub_index, sub_index], 2)))

boot = pd.DataFrame(records)
summary = pd.DataFrame({
    "ARI_boot_mean": boot.mean(),
    "ARI_boot_sd": boot.std(),
    "ARI_boot_p05": boot.quantile(0.05),
})
summary["beats_the_other"] = [
    (boot["descriptors"] > boot["morphometric"]).mean(),
    (boot["morphometric"] > boot["descriptors"]).mean(),
]
summary.to_csv(OUT / "representation_stability_vs_morphometric.csv")

print(f"{N_REPLICATES} replicates, {SAMPLE_FRACTION:.0%} of {len(shared_index)} "
      f"neurons ({sample_size}), sampled without replacement, Ward k=2")
print(summary.to_string(float_format=lambda v: f"{v:.4f}"))
print()
print("Table 1 in the paper reports: descriptors 0.495 (SD 0.126), "
      "morphometric 0.378 (SD 0.127),")
print("descriptors ahead in 67.8% of replicates, 5th percentile 0.339 vs 0.153.")
print("The values above are a fresh run of the stated method, not that original "
      "run; see this file's docstring.")
