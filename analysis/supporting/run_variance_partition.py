"""Between-tier versus within-tier morphological variance.

The question this analysis answers is whether
intra-regional or inter-regional variation in cortico-geniculate neuronal
morphology is greater." He then asks why the six subtypes of Adusei et al. were
not recovered.

Those are different questions, and the second rests on a category error. Adusei
et al. partitioned variance among 50 extrastriate neurons only, so every
component of that variance was within-tier. The present study pools 117 neurons
across two tiers, which adds a between-tier axis to the variance being
partitioned. If the between-tier component is large, the first split of the
pooled dendrogram must capture it, and within-tier substructure necessarily
appears at deeper cuts. Failing to see six clusters at the top level of a pooled
dendrogram is therefore not a failure to replicate.

This script quantifies the two components by distance-based PERMANOVA on the
combined descriptor distance matrix, and shows that within-extrastriate
substructure is still present in the present data once the between-tier axis is
removed.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

from _common import (PICKLES, PLOTS, load_classes, load_matrices, normalise,
                     ward_clusters)

OUT = PLOTS / "variance_partition"
OUT.mkdir(parents=True, exist_ok=True)
RNG = np.random.default_rng(20260819)


def permanova(distance, labels, n_perm=4999):
    """Distance-based partition of sums of squares (Anderson 2001).

    Returns R2 = SS_between / SS_total and a permutation p-value for pseudo-F.
    """
    D = np.asarray(distance, dtype=float)
    n = len(D)
    sq = D ** 2
    ss_total = sq[np.triu_indices(n, 1)].sum() / n

    def ss_within(lab):
        total = 0.0
        for level in np.unique(lab):
            idx = np.flatnonzero(lab == level)
            if len(idx) < 2:
                continue
            block = sq[np.ix_(idx, idx)]
            total += block[np.triu_indices(len(idx), 1)].sum() / len(idx)
        return total

    labels = np.asarray(labels)
    groups = len(np.unique(labels))
    within = ss_within(labels)
    between = ss_total - within
    f_obs = (between / (groups - 1)) / (within / (n - groups))

    count = 1
    for _ in range(n_perm):
        permuted = RNG.permutation(labels)
        w = ss_within(permuted)
        f = ((ss_total - w) / (groups - 1)) / (w / (n - groups))
        if f >= f_obs:
            count += 1
    return between / ss_total, f_obs, count / (n_perm + 1)


matrices = load_matrices()
classes = load_classes()
names = list(matrices)
index = matrices[names[0]].index

combined = None
for name in names:
    term = normalise(matrices[name], "none")
    combined = term if combined is None else combined + term
combined = pd.DataFrame(combined, index=index, columns=index)

tier = classes.loc[index, "class"].to_numpy()
area = classes.loc[index, "area"].to_numpy()

rows = []

# 1. Between-tier variance, all 117 neurons.
r2, f, p = permanova(combined, tier)
rows.append({"contrast": "between tier (V1/V2 vs extrastriate)", "n": len(index),
             "groups": 2, "R2": r2, "pseudo_F": f, "p_perm": p})

# 2. Among-area variance within the extrastriate tier only, exactly the
#    population Adusei et al. analysed.
extra = index[tier == "PMLS-PLLS"]
r2e, fe, pe = permanova(combined.loc[extra, extra], area[tier == "PMLS-PLLS"])
rows.append({"contrast": "among areas within extrastriate (PMLS/PLLS/21A)",
             "n": len(extra), "groups": 3, "R2": r2e, "pseudo_F": fe, "p_perm": pe})

# 3. Among-area variance within V1/V2 is not testable: all 67 filenames carry
#    the same area label, so V1 and V2 cannot be separated from the archive.

summary = pd.DataFrame(rows)
summary.to_csv(OUT / "permanova_variance_partition.csv", index=False)
print("Distance-based variance partition on the combined descriptor distance")
print(summary.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
print(f"\nBetween-tier variance is {r2 / r2e:.1f}x the among-area variance "
      f"within the extrastriate tier.")

# ---------------------------------------------------------------- substructure
# Does within-extrastriate substructure survive in the present representation
# once the between-tier axis is removed?
print("\nWithin-extrastriate substructure, 50 neurons clustered on their own")
sub = combined.loc[extra, extra]
area_extra = pd.Series(area[tier == "PMLS-PLLS"], index=extra)
for k in (2, 3, 4, 5, 6):
    labels_k = ward_clusters(sub, k)
    table = pd.crosstab(pd.Series(labels_k, index=extra, name=f"cluster_k{k}"),
                        area_extra.rename("area"))
    if k == 6:
        table.to_csv(OUT / "within_extrastriate_six_cluster_by_area.csv")
        print(f"\nk=6 (the Adusei cut), present descriptors:")
        print(table.to_string())

# Where do the 50 extrastriate neurons sit in the pooled dendrogram?
pooled = pd.Series(ward_clusters(combined, 2), index=index)
print("\nPooled k=2 assignment of the same 50 extrastriate neurons:")
print(pd.crosstab(pooled.loc[extra].rename("pooled_cluster"),
                  area_extra.rename("area")).to_string())

agreement = adjusted_rand_score(ward_clusters(sub, 6), pooled.loc[extra])
print(f"\nARI between the within-tier six-cluster solution and the pooled "
      f"two-cluster assignment: {agreement:.4f}")
print("A low value is the expected result: the two analyses partition different "
      "variance and answer different questions.")
