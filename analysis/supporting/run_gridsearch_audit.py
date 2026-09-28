"""Audit and re-run the published grid-search combination method (Supplementary S4.2).

Khalil et al., Supplementary S4.2 defines the combination

    d = a_1 d_1 + ... + a_n d_n,   a_i non-negative, sampled from a uniform grid

scored, for two classes, by

    sc = E / (I_1 + I_2)

where I_k is the maximal within-class distance in class k and E the minimal
between-class distance, maximised over the grid. S4.4 requires a permutation
test to show the separation is not overfitting.

Two implementation defects are quantified here:

1. The grid runs over [1.0, 3.0] per descriptor, so no descriptor can be
   excluded (a_i = 0 is unreachable) and the achievable weight ratio is capped
   at 3:1. The paper specifies non-negative constants.
2. With dx=0.1 the grid has 21**6 = 85,766,121 points but MAX_ITERATIONS is
   1,000,000, and increment_counter is a little-endian odometer. Leaf and Spread
   therefore never move off their initial value of 1.0.

Because the score is homogeneous of degree zero in a, only the direction of a
matters, so the search space is the simplex. This script searches a regular
simplex grid, which is exhaustive over the meaningful space.
"""
import json
from itertools import combinations_with_replacement

import numpy as np
import pandas as pd

from _common import PLOTS, load_classes, load_matrices, normalise

OUT = PLOTS / "gridsearch_audit"
OUT.mkdir(parents=True, exist_ok=True)
RNG = np.random.default_rng(20260819)

matrices = load_matrices()
classes = load_classes()
names = list(matrices)
index = matrices[names[0]].index
labels = classes.loc[index, "class"].to_numpy()

# ------------------------------------------------------------------ pair table
iu, ju = np.triu_indices(len(index), k=1)
pair_tables = {}
for scheme in ("none", "mean"):
    pair_tables[scheme] = np.column_stack(
        [normalise(matrices[n], scheme)[iu, ju] for n in names])
same = labels[iu] == labels[ju]
class_of_pair = np.where(same, labels[iu], "between")


def simplex_grid(n, m):
    """Regular grid on the (n-1)-simplex with resolution m."""
    points = []
    for cut in combinations_with_replacement(range(m + 1), n - 1):
        cuts = (0,) + cut + (m,)
        points.append([cuts[i + 1] - cuts[i] for i in range(n)])
    return np.asarray(points, dtype=float) / m


def score_batch(pairs, alphas, group_labels, batch=4096):
    """sc = E / sum_k I_k for each alpha row."""
    masks = {c: group_labels == c for c in np.unique(group_labels)}
    within = [m for c, m in masks.items() if c != "between"]
    between = masks["between"]
    out = np.empty(len(alphas))
    for start in range(0, len(alphas), batch):
        chunk = alphas[start:start + batch]
        combined = pairs @ chunk.T                      # (n_pairs, chunk)
        internal = sum(combined[m].max(axis=0) for m in within)
        external = combined[between].min(axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            out[start:start + batch] = np.where(internal > 0, external / internal, 0.0)
    return out


RESOLUTION = 20
alphas = simplex_grid(len(names), RESOLUTION)
print(f"simplex grid resolution {RESOLUTION}: {len(alphas):,} weight vectors")

results = {}
for scheme, pairs in pair_tables.items():
    scores = score_batch(pairs, alphas, class_of_pair)
    best = int(np.argmax(scores))
    weights = dict(zip(names, alphas[best].round(4)))

    # The submitted configuration: all weights equal, reachable only because the
    # odometer never moved Leaf or Spread off 1.0.
    equal = np.full((1, len(names)), 1.0 / len(names))
    equal_score = float(score_batch(pairs, equal, class_of_pair)[0])

    results[scheme] = {
        "best_score": float(scores[best]),
        "best_weights": {k: float(v) for k, v in weights.items()},
        "equal_weight_score": equal_score,
        "improvement_over_equal": float(scores[best]) / equal_score if equal_score else np.inf,
        "n_zero_weights_in_best": int((alphas[best] == 0).sum()),
    }
    print(f"\n--- normalisation = {scheme} ---")
    print(f"  best sc = {scores[best]:.5f}   equal-weight sc = {equal_score:.5f}")
    print(f"  best weights: {weights}")

# ------------------------------------------------------- permutation test S4.4
PERMUTATIONS = 200
PERM_RESOLUTION = 12
perm_alphas = simplex_grid(len(names), PERM_RESOLUTION)
print(f"\npermutation test: {PERMUTATIONS} permutations on a "
      f"{len(perm_alphas):,}-point grid")

for scheme, pairs in pair_tables.items():
    observed = score_batch(pairs, perm_alphas, class_of_pair).max()
    null = np.empty(PERMUTATIONS)
    for r in range(PERMUTATIONS):
        shuffled = RNG.permutation(labels)
        pair_groups = np.where(shuffled[iu] == shuffled[ju], shuffled[iu], "between")
        null[r] = score_batch(pairs, perm_alphas, pair_groups).max()
    p_value = float((null >= observed).sum() + 1) / (PERMUTATIONS + 1)
    results[scheme].update({
        "permutation_observed": float(observed),
        "permutation_null_mean": float(null.mean()),
        "permutation_null_p95": float(np.quantile(null, 0.95)),
        "permutation_p_value": p_value,
    })
    print(f"  {scheme:5s}: observed={observed:.5f}  null mean={null.mean():.5f}  "
          f"null p95={np.quantile(null, 0.95):.5f}  p={p_value:.4f}")

(OUT / "gridsearch_results.json").write_text(json.dumps(results, indent=2),
                                             encoding="utf-8")
pd.DataFrame(results).T.to_csv(OUT / "gridsearch_summary.csv")
