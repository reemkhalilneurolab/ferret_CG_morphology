"""Adusei-inspired six-cluster analysis on size-corrected morphometric variables.

Supersedes plots/adusei_six_cluster_available_metrics/, which clustered
z-scored L-Measure *Total* values. Those totals made branch count the dominant
axis of the feature space (15 of 39 variables at |Spearman rho| > 0.9 with
N_branch), so the resulting clusters largely ordered cells by size.

This rerun uses the per-object averages built by build_lmeasure_percell.py and
reports both solutions side by side.
"""
import json

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from _common import REPO, RUN

OUT = RUN / "plots" / "adusei_six_cluster_corrected"
OUT.mkdir(parents=True, exist_ok=True)
PERCELL = RUN / "plots" / "lmeasure_percell" / "lmeasure_data_percell.csv"


def canonical(series):
    return (series.astype(str)
            .str.replace(".swc", "", regex=False)
            .str.replace(".CNG", "", regex=False)
            .str.replace(" ", "_", regex=False)
            .str.replace(".", "_", regex=False)
            .str.replace("-", "_", regex=False))


meta = pd.read_csv(RUN / "neuron_class_df.csv")
meta["neuron_name"] = meta["neuron_name"].astype(str).str.replace(".CNG", "", regex=False)
extrastriate = meta[meta["area"].isin(["PMLS", "PLLS", "21A"])]
wanted = set(extrastriate["neuron_name"])


def prepare(path):
    table = pd.read_csv(path)
    table.columns = [c.strip() for c in table.columns]
    table["neuron_name"] = canonical(table["Filename"])
    table = table[table["neuron_name"].isin(wanted)].drop_duplicates("neuron_name")
    table = table.set_index("neuron_name")
    numeric = (table.select_dtypes(include=[np.number])
               .dropna(axis=1, how="all").dropna(axis=0, how="any"))
    return numeric


solutions = {
    "submitted_totals": prepare(REPO / "data" / "Briggs" / "lmeasure_data.csv"),
    "corrected_per_object": prepare(PERCELL),
}

results, assignments = {}, {}
for label, numeric in solutions.items():
    areas = extrastriate.set_index("neuron_name").loc[numeric.index, "area"]
    scaled = StandardScaler().fit_transform(numeric)
    clusters = fcluster(linkage(scaled, method="ward"), t=6, criterion="maxclust")
    pca = PCA(n_components=2).fit(scaled)
    components = pca.transform(scaled)

    assign = pd.DataFrame({
        "neuron_name": numeric.index, "area": areas.to_numpy(),
        "cluster": clusters, "PC1": components[:, 0], "PC2": components[:, 1],
    })
    assignments[label] = assign
    assign.to_csv(OUT / f"cluster_assignments_{label}.csv", index=False)
    pd.crosstab(assign["cluster"], assign["area"]).to_csv(
        OUT / f"cluster_by_area_{label}.csv")

    # How strongly does the solution simply order cells by size?
    size = numeric["N_branch"]
    size_corr = abs(pd.Series(components[:, 0]).corr(
        pd.Series(size.to_numpy()), method="spearman"))

    results[label] = {
        "n_neurons": int(len(numeric)),
        "n_features": int(numeric.shape[1]),
        "silhouette": float(silhouette_score(scaled, clusters)),
        "PC1_variance": float(pca.explained_variance_ratio_[0]),
        "PC2_variance": float(pca.explained_variance_ratio_[1]),
        "abs_spearman_PC1_vs_N_branch": float(size_corr),
        "area_counts": assign["area"].value_counts().to_dict(),
    }

agreement = adjusted_rand_score(
    assignments["submitted_totals"].set_index("neuron_name").loc[
        assignments["corrected_per_object"]["neuron_name"], "cluster"],
    assignments["corrected_per_object"]["cluster"])
results["ARI_between_solutions"] = float(agreement)

(OUT / "analysis_summary.json").write_text(json.dumps(results, indent=2),
                                           encoding="utf-8")
print(json.dumps(results, indent=2))

print("\nCluster x area, corrected per-object features")
print(pd.crosstab(assignments["corrected_per_object"]["cluster"],
                  assignments["corrected_per_object"]["area"]).to_string())
