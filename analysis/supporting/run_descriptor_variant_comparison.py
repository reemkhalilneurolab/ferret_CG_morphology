"""Compare the published variants of the Leaf and Spread descriptors.

Correcting Leaf and Spread to the definitions written in the manuscript lowered
their individual class separation sharply (Leaf ARI 0.450 -> 0.093, Spread
0.499 -> 0.239). Before deciding which definition the paper should use, this
compares every variant that has a published basis, so the choice rests on an
argument rather than on whichever scores best.

Leaf variants
  R      raw reachable-leaf count, the construction in Khalil et al. [25]
  R_over_L  reachable leaves / total leaves, the manuscript's stated definition
  L_over_R  the reciprocal, an earlier implementation

Spread variants
  apical  apical tree only, the manuscript's stated definition
  whole   whole dendritic tree, the current definition

A key question for interpretation: how much of each variant's class separation
comes from neuron size rather than shape? L/R and R both scale with the total
leaf count, and extrastriate cells have more leaves than V1/V2 cells, so those
variants can separate the classes partly by encoding size.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, silhouette_score

import config
import neurotopo.descriptors as D
import neurotopo.utils as utl
from _common import RUN, load_classes, ward_clusters

OUT = RUN / "plots" / "descriptor_variants"
OUT.mkdir(parents=True, exist_ok=True)


def leaf_profile(neuron, mode):
    """Leaf step function under a given convention."""
    import networkx as nx
    graph = neuron["neuron_as_graph"]
    frame = neuron["swc_df"]
    radius_col = "euclidean_distance_to_soma"
    bifurcations = frame[frame["node_type"] == "bifurcation"][["ID", radius_col]]
    terminations = frame[frame["node_type"] == "termination"][["ID", radius_col]]
    leaf_ids = set(terminations["ID"])
    paths = [nx.shortest_path(graph, source=1, target=t) for t in leaf_ids]
    total = len(leaf_ids)

    values = [(0.0, {"R": float(total), "R_over_L": 1.0, "L_over_R": 1.0}[mode])]
    for _, row in pd.concat([bifurcations, terminations]).iterrows():
        node = row["ID"]
        reachable = 1 if node in leaf_ids else sum(1 for p in paths if node in p)
        if mode == "R":
            value = float(reachable)
        elif mode == "R_over_L":
            value = reachable / total
        else:
            value = total / reachable
        values.append((row[radius_col], value))
    values.sort(key=lambda t: t[0])
    return D.normalize_descriptor_radii(values)


def spread_profile(neuron, scope):
    """Spread step function over the apical tree or the whole dendritic tree."""
    original = neuron["swc_df"]
    if scope == "whole":
        neuron = dict(neuron)
        neuron["swc_df"] = original
        # Temporarily accept every dendritic type by relabelling basal as apical.
        frame = original.copy()
        frame.loc[frame["Type"] == 3, "Type"] = D.APICAL_TYPE
        neuron["swc_df"] = frame
    try:
        # Returns (spread_index, volume, path); only the index is the descriptor.
        return D.calculate_spread_descriptor(neuron)[0]
    except D.NoApicalDendriteError:
        return None


def distance_matrix(profiles, names):
    n = len(names)
    out = np.zeros((n, n))
    for i in range(n):
        xi = [r for r, _ in profiles[names[i]]]
        yi = [v for _, v in profiles[names[i]]]
        for j in range(i + 1, n):
            xj = [r for r, _ in profiles[names[j]]]
            yj = [v for _, v in profiles[names[j]]]
            d = utl.calculate_l1_difference((xi, yi), (xj, yj))
            out[i, j] = out[j, i] = d
    return pd.DataFrame(out, index=names, columns=names)


def main():
    config.PLOT_STEP_FUNCTION = False
    config.PLOT_NEURON = False
    from neurotopo.neuron_processor import process_swc_directory

    neurons, class_frame, _ = process_swc_directory()
    names = sorted(neurons)
    classes = class_frame.set_index("neuron_name").loc[names, "class"].to_numpy()
    leaf_totals = np.array([
        (neurons[n]["swc_df"]["node_type"] == "termination").sum() for n in names
    ], dtype=float)

    variants = {}
    for mode in ("R", "R_over_L", "L_over_R"):
        variants[f"Leaf[{mode}]"] = {n: leaf_profile(neurons[n], mode) for n in names}
    for scope in ("apical", "whole"):
        profiles = {n: spread_profile(neurons[n], scope) for n in names}
        if all(p is not None for p in profiles.values()):
            variants[f"Spread[{scope}]"] = profiles

    rows = []
    for label, profiles in variants.items():
        matrix = distance_matrix(profiles, names)
        predicted = ward_clusters(matrix, 2)
        values = np.asarray(matrix, dtype=float)
        off = values[np.triu_indices(len(names), 1)]
        # How much of the geometry simply tracks neuron size?
        size_gap = np.abs(leaf_totals[:, None] - leaf_totals[None, :])
        size_corr = pd.Series(off).corr(
            pd.Series(size_gap[np.triu_indices(len(names), 1)]), method="spearman")
        rows.append({
            "variant": label,
            "ARI": adjusted_rand_score(classes, predicted),
            "silhouette": silhouette_score(values, classes, metric="precomputed"),
            "mean_distance": off.mean(),
            "rho_with_leaf_count_difference": abs(size_corr),
        })
        matrix.to_pickle(OUT / f"{label.replace('[', '_').replace(']', '')}_matrix.pkl")

    result = pd.DataFrame(rows).sort_values("ARI", ascending=False)
    result.to_csv(OUT / "descriptor_variant_comparison.csv", index=False)
    print(f"cohort: {len(names)} neurons\n")
    print(result.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("\nrho_with_leaf_count_difference is the Spearman correlation between a")
    print("variant's pairwise distances and the absolute difference in leaf count.")
    print("A high value means the variant separates cells largely by size.")


if __name__ == "__main__":
    main()
