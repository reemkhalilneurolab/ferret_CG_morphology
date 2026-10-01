"""Build and evaluate a size-corrected, non-redundant morphometric feature set.

The exported L-Measure table stores Total values. For any quantity measured per
compartment, per branch or per bifurcation, the total is that quantity's mean
multiplied by the number of objects, so the column becomes a proxy for neuron
size. Summing a per-branch ratio such as Contraction, which averages about 0.92,
over 166 branches simply recovers the branch count.

Three feature sets are compared:

    totals     the exported table as-is
    per_object each metric divided by the number of objects it was measured over
    reduced    per_object with redundant variables removed, keeping one
               representative from each group of mutually correlated features

Denominators are validated against each metric's admissible range where one
exists, so the conversion can be checked rather than taken on trust.
"""
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.metrics import (adjusted_rand_score, normalized_mutual_info_score,
                             silhouette_score)
from sklearn.preprocessing import StandardScaler

from _common import REPO, RUN

OUT = RUN / "plots" / "lmeasure_corrected"
OUT.mkdir(parents=True, exist_ok=True)

# Metrics whose Total is already the quantity of interest.
PER_NEURON = ["Soma_Surface", "N_stems", "N_bifs", "N_branch", "N_tips",
              "Width", "Height", "Depth", "Length", "Surface", "Volume"]
PER_COMPARTMENT = ["Diameter", "Diameter_pow", "EucDistance", "PathDistance",
                   "Helix", "SectionArea", "Branch_Order", "Terminal_degree"]
PER_BRANCH = ["Branch_pathlength", "Contraction", "Fragmentation",
              "Taper_1", "Taper_2"]
PER_BIFURCATION = ["Bif_ampl_local", "Bif_ampl_remote", "Bif_tilt_local",
                   "Bif_tilt_remote", "Bif_torque_local", "Bif_torque_remote",
                   "Partition_asymmetry", "Pk", "Pk_2", "Pk_classic",
                   "Daughter_Ratio", "Parent_Daughter_Ratio",
                   "HillmanThreshold", "Diam_threshold", "Last_parent_diam"]
PER_TIP = ["TerminalSegment"]

DROP = {
    "Type": "sum of SWC structure codes (soma 1, basal 3, apical 4); a node "
            "count and composition proxy, not a morphological measurement",
    "Rall_Power": "zero for 142 of 145 reconstructions; carries no information",
    "Fractal_Dim": "no candidate denominator yields values >= 1, so the total "
                   "cannot be resolved into a valid fractal dimension",
}

BOUNDS = {
    "Contraction": (0.0, 1.0), "Partition_asymmetry": (0.0, 1.0),
    "Daughter_Ratio": (1.0, np.inf), "Fragmentation": (1.0, np.inf),
    "Bif_ampl_local": (0.0, 180.0), "Bif_ampl_remote": (0.0, 180.0),
    "Bif_tilt_local": (0.0, 180.0), "Bif_tilt_remote": (0.0, 180.0),
    "Bif_torque_local": (0.0, 180.0), "Bif_torque_remote": (0.0, 180.0),
}

SCOPE = {}
for group, label in ((PER_NEURON, "per_neuron"), (PER_COMPARTMENT, "per_compartment"),
                     (PER_BRANCH, "per_branch"), (PER_BIFURCATION, "per_bifurcation"),
                     (PER_TIP, "per_tip")):
    for metric in group:
        SCOPE[metric] = label


def canonical(series):
    return (series.astype(str)
            .str.replace(".swc", "", regex=False).str.replace(".CNG", "", regex=False)
            .str.replace(" ", "_", regex=False).str.replace(".", "_", regex=False)
            .str.replace("-", "_", regex=False))


def build_per_object(raw):
    """Divide each metric by the number of objects it was measured over."""
    counts = {
        "per_neuron": pd.Series(1.0, index=raw.index),
        # Total Fragmentation is compartments-per-branch summed over branches,
        # i.e. the total compartment count.
        "per_compartment": raw["Fragmentation"].astype(float),
        "per_branch": raw["N_branch"].astype(float),
        "per_bifurcation": raw["N_bifs"].astype(float),
        "per_tip": raw["N_tips"].astype(float),
    }
    converted, records = pd.DataFrame(index=raw.index), []
    for metric in raw.columns:
        if metric in DROP:
            records.append({"metric": metric, "scope": "dropped", "denominator": "",
                            "validated": "", "note": DROP[metric]})
            continue
        scope = SCOPE[metric]
        values = raw[metric].astype(float) / counts[scope]
        converted[metric] = values
        low, high = BOUNDS.get(metric, (-np.inf, np.inf))
        ok = values.min() >= low - 1e-9 and values.max() <= high + 1e-9
        records.append({
            "metric": metric, "scope": scope,
            "denominator": {"per_neuron": "1", "per_compartment": "total compartments",
                            "per_branch": "N_branch", "per_bifurcation": "N_bifs",
                            "per_tip": "N_tips"}[scope],
            "min": values.min(), "max": values.max(),
            "validated": "" if metric not in BOUNDS else ("pass" if ok else "FAIL"),
            "note": "" if metric not in BOUNDS else f"admissible range [{low}, {high}]",
        })
    return converted, pd.DataFrame(records)


def reduce_redundancy(frame, threshold=0.9):
    """Keep one representative from each group of mutually correlated features.

    Groups are the connected components of the graph joining features whose
    absolute Spearman correlation exceeds the threshold. The representative is
    the member with the highest mean absolute correlation to the rest of its
    group, i.e. the one that best stands in for it.
    """
    corr = frame.corr(method="spearman").abs()
    remaining, groups = list(frame.columns), []
    while remaining:
        seed = remaining[0]
        component, frontier = {seed}, [seed]
        while frontier:
            current = frontier.pop()
            for other in remaining:
                if other not in component and corr.loc[current, other] > threshold:
                    component.add(other)
                    frontier.append(other)
        groups.append(sorted(component))
        remaining = [c for c in remaining if c not in component]

    keep, mapping = [], []
    for group in groups:
        if len(group) == 1:
            representative = group[0]
        else:
            within = corr.loc[group, group]
            representative = within.mean(axis=1).idxmax()
        keep.append(representative)
        mapping.append({"representative": representative, "group_size": len(group),
                        "members": ";".join(group)})
    return frame[keep], pd.DataFrame(mapping)


def evaluate(frame, labels):
    scaled = StandardScaler().fit_transform(frame)
    Z = linkage(scaled, method="ward")
    predicted = fcluster(Z, t=2, criterion="maxclust")
    table = pd.crosstab(pd.Series(predicted), pd.Series(labels))
    size = frame["N_branch"] if "N_branch" in frame else None
    rho = np.nan
    if size is not None:
        from sklearn.decomposition import PCA
        pc1 = PCA(n_components=1).fit_transform(scaled)[:, 0]
        rho = abs(pd.Series(pc1).corr(pd.Series(size.to_numpy()), method="spearman"))
    return {
        "n_features": frame.shape[1],
        "ARI": adjusted_rand_score(labels, predicted),
        "NMI": normalized_mutual_info_score(labels, predicted),
        "Silhouette": silhouette_score(scaled, labels),
        "Purity": table.max(axis=1).sum() / len(labels),
        "abs_rho_PC1_vs_N_branch": rho,
    }, predicted


def main():
    raw = pd.read_csv(REPO / "data" / "Briggs" / "lmeasure_data.csv")
    raw.columns = [c.strip() for c in raw.columns]
    raw["neuron_name"] = canonical(raw["Filename"])

    cohort = pd.read_csv(RUN / "neuron_class_df.csv").set_index("neuron_name")
    raw = raw[raw["neuron_name"].isin(cohort.index)].drop_duplicates("neuron_name")
    raw = raw.set_index("neuron_name").loc[cohort.index]
    labels = cohort["class"].to_numpy()
    numeric = raw.select_dtypes(include=[np.number])
    print(f"cohort {len(raw)} neurons, {numeric.shape[1]} numeric metrics")

    per_object, mapping = build_per_object(numeric)
    mapping.to_csv(OUT / "denominator_map.csv", index=False)
    failures = mapping[mapping["validated"] == "FAIL"]
    print(f"denominator range checks: {len(failures)} failures")

    reduced, groups = reduce_redundancy(per_object)
    groups.to_csv(OUT / "redundancy_groups.csv", index=False)
    per_object.to_csv(OUT / "lmeasure_per_object.csv")
    reduced.to_csv(OUT / "lmeasure_reduced.csv")

    print(f"\nredundancy grouping at |Spearman rho| > 0.9: "
          f"{per_object.shape[1]} features -> {reduced.shape[1]} representatives")
    for _, row in groups[groups["group_size"] > 1].iterrows():
        print(f"  {row['representative']:22s} stands for {row['group_size']:2d}: "
              f"{row['members']}")

    totals = numeric.drop(columns=[c for c in DROP if c in numeric], errors="ignore")
    sets = {"totals_as_exported": numeric, "totals_minus_dropped": totals,
            "per_object": per_object, "per_object_reduced": reduced}

    print("\nredundancy with neuron size, by feature set")
    rows = []
    for name, frame in sets.items():
        corr = frame.corrwith(frame["N_branch"], method="spearman").abs().drop("N_branch")
        rows.append({"feature_set": name, "n_features": frame.shape[1],
                     "n_rho_gt_0.9": int((corr > 0.9).sum()),
                     "n_rho_gt_0.7": int((corr > 0.7).sum()),
                     "median_rho": corr.median()})
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    print("\nclustering performance, Ward on z-scored features, k=2")
    results = {}
    for name, frame in sets.items():
        metrics, _ = evaluate(frame, labels)
        results[name] = metrics
    summary = pd.DataFrame(results).T
    summary.to_csv(OUT / "lmeasure_feature_set_comparison.csv")
    print(summary.to_string(float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    main()
