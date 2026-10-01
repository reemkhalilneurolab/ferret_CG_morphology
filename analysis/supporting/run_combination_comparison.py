"""Compare the two published combination methods on the corrected descriptors.

Method A: detection-based weighting, introduced in the present manuscript.
    w_d = mean over classes of det_d(C), then D = sum_d w_d * D_d.
    Detection follows [25] section 1.4: det(C) is the largest value of
    min(recall, purity) over balls, where the ball centre counts as a member.

Method B: exhaustive grid search over the simplex, from [25] section 4.2 and
    described in the manuscript Methods. Maximises the ratio of external to
    internal distance, where internal is the sum over classes of the largest
    within-class distance and external is the smallest between-class distance.

Both are run on the corrected descriptor matrices (apical-only Spread, Leaf as
R/L, unrounded L1) over the 115-neuron pyramidal cohort. Results are reported
with and without descriptor-scale normalisation, since the six descriptors carry
different physical units.

Usage:  python run_combination_comparison.py [grid_resolution]
"""
import sys
from itertools import combinations_with_replacement

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import cophenet, linkage
from scipy.spatial.distance import squareform
from sklearn.metrics import (adjusted_rand_score, normalized_mutual_info_score,
                             silhouette_score)

from _common import PLOTS, load_classes, load_matrices, normalise, ward_clusters

RESOLUTION = int(sys.argv[1]) if len(sys.argv) > 1 else 50
N_PERM = 999
RNG = np.random.default_rng(20260820)

OUT = PLOTS / "combination_comparison"
OUT.mkdir(parents=True, exist_ok=True)


def simplex_grid(n_axes, resolution):
    """All weight vectors on the simplex with the given denominator."""
    points = []
    for cut in combinations_with_replacement(range(resolution + 1), n_axes - 1):
        edges = (0,) + cut + (resolution,)
        points.append([edges[i + 1] - edges[i] for i in range(n_axes)])
    return np.asarray(points, dtype=float) / resolution


def separation_score(pair_matrix, weights, within_masks, between_mask):
    """External/internal ratio of [25] section 4.2, vectorised over weights."""
    combined = pair_matrix @ weights.T
    internal = sum(combined[mask].max(axis=0) for mask in within_masks)
    external = combined[between_mask].min(axis=0)
    return np.where(internal > 0, external / internal, 0.0)


def evaluate(matrix, labels):
    predicted = ward_clusters(matrix, 2)
    condensed = squareform(np.asarray(matrix, dtype=float), checks=False)
    table = pd.crosstab(pd.Series(predicted), pd.Series(labels))
    return {
        "ARI": adjusted_rand_score(labels, predicted),
        "NMI": normalized_mutual_info_score(labels, predicted),
        "Silhouette": silhouette_score(np.asarray(matrix), labels,
                                       metric="precomputed"),
        "Purity": table.max(axis=1).sum() / len(labels),
        "Cophenetic": cophenet(linkage(condensed, "ward"), condensed)[0],
    }, predicted


def main():
    matrices = load_matrices()
    classes = load_classes()
    names = list(matrices)
    index = matrices[names[0]].index
    labels = classes.loc[index, "class"].to_numpy()
    areas = classes.loc[index, "area"].to_numpy()
    print(f"cohort: {len(index)} neurons, descriptors: {names}")

    # ------------------------------------------------ descriptor scale table
    rows = []
    for name in names:
        values = np.asarray(matrices[name], dtype=float)
        off = values[~np.eye(len(values), dtype=bool)]
        rows.append({"descriptor": name, "mean_distance": off.mean(),
                     "max_distance": off.max(),
                     "unique_values": len(np.unique(off))})
    scale = pd.DataFrame(rows)
    scale["share_unnormalised_%"] = (100 * scale["mean_distance"]
                                     / scale["mean_distance"].sum())
    scale = scale.sort_values("share_unnormalised_%", ascending=False)
    scale.to_csv(OUT / "descriptor_scale_contributions.csv", index=False)
    print("\nDescriptor scales on the corrected matrices")
    print(scale.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    # -------------------------------------------------- detection weighting
    rates = pd.read_csv(PLOTS / "Detection" / "detection_rates.csv")
    per_class = (rates[rates["score_type"] == "mean"]
                 .pivot(index="descriptor", columns="class", values="value"))
    detection = {n: float(per_class.loc[n].mean()) for n in names}
    print("\nDetection weights (corrected detection):")
    for n in names:
        print(f"  {n:20s} {detection[n]:7.3f}")

    # ------------------------------------------------------- grid search
    iu, ju = np.triu_indices(len(index), k=1)
    same = labels[iu] == labels[ju]
    within_masks = [same & (labels[iu] == c) for c in np.unique(labels)]
    between_mask = ~same

    results, partitions = {}, {}
    for scheme in ("none", "mean"):
        normed = {n: normalise(matrices[n], scheme) for n in names}
        pair_matrix = np.column_stack([normed[n][iu, ju] for n in names])

        grid = simplex_grid(len(names), RESOLUTION)
        print(f"\n[{scheme}] searching {len(grid):,} weight vectors "
              f"(resolution 1/{RESOLUTION})")
        best_score, best_weights = -np.inf, None
        for start in range(0, len(grid), 8192):
            block = grid[start:start + 8192]
            scores = separation_score(pair_matrix, block, within_masks,
                                      between_mask)
            k = int(np.argmax(scores))
            if scores[k] > best_score:
                best_score, best_weights = scores[k], block[k]

        def build(weights):
            total = sum(weights[i] * normed[n] for i, n in enumerate(names))
            return pd.DataFrame(total, index=index, columns=index)

        candidates = {
            "detection_weighted": np.array([detection[n] for n in names]),
            "grid_search": best_weights,
            "equal_weights": np.full(len(names), 1.0 / len(names)),
        }
        for label, weights in candidates.items():
            matrix = build(weights)
            metrics, predicted = evaluate(matrix, labels)
            metrics["separation_score"] = float(separation_score(
                pair_matrix, weights[None, :] / weights.sum(),
                within_masks, between_mask)[0])
            results[(scheme, label)] = metrics
            partitions[(scheme, label)] = predicted
            if label == "grid_search":
                pd.Series(dict(zip(names, best_weights))).to_csv(
                    OUT / f"grid_search_weights_{scheme}.csv")

        # Permutation test for the grid-search optimum.
        null = np.empty(N_PERM)
        for i in range(N_PERM):
            shuffled = RNG.permutation(labels)
            s = shuffled[iu] == shuffled[ju]
            wm = [s & (shuffled[iu] == c) for c in np.unique(shuffled)]
            best = -np.inf
            for start in range(0, len(grid), 8192):
                v = separation_score(pair_matrix, grid[start:start + 8192],
                                     wm, ~s)
                best = max(best, v.max())
            null[i] = best
        p_value = (1 + (null >= best_score).sum()) / (1 + N_PERM)
        print(f"[{scheme}] grid optimum {best_score:.5f}, "
              f"null median {np.median(null):.5f}, p = {p_value:.4f}")
        pd.DataFrame({"null_score": null}).to_csv(
            OUT / f"permutation_null_{scheme}.csv", index=False)

    summary = pd.DataFrame(results).T
    summary.index.names = ["normalisation", "method"]
    summary.to_csv(OUT / "combination_comparison.csv")
    print("\n" + "=" * 70)
    print(summary.to_string(float_format=lambda v: f"{v:.4f}"))

    print("\nCluster x area for each candidate")
    for key, predicted in partitions.items():
        print(f"\n{key}")
        print(pd.crosstab(pd.Series(predicted, name="cluster"),
                          pd.Series(areas, name="area")).to_string())


if __name__ == "__main__":
    main()
