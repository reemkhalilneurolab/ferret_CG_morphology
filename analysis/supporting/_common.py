"""Shared helpers for the supporting analyses.

All paths resolve from the repository root so a clean checkout reproduces the
outputs without editing absolute paths.
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

REPO = Path(__file__).resolve().parents[2]
# These scripts are run directly, which puts this directory on sys.path but not
# the repository root, so neurotopo/ and analysis/ would otherwise be
# unimportable.
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
# Which run to analyse. Set NEUROTOPO_RUN to point these scripts at a
# different run for side-by-side comparison.
RUN = REPO / "figures" / os.environ.get("NEUROTOPO_RUN", "results")
PICKLES = RUN / "pickles"
PLOTS = RUN / "plots"


def load_matrices():
    """Return the per-descriptor distance matrices."""
    return pd.read_pickle(PICKLES / "individual_distance_matrices.pkl")


def load_classes():
    """Neuron metadata, indexed by neuron name.

    The area label is derived from the filename rather than read from the CSV.
    Only some runs wrote an ``area`` column, so deriving it keeps these
    analyses runnable against any run.
    """
    classes = pd.read_csv(RUN / "neuron_class_df.csv").set_index("neuron_name")
    if "area" not in classes.columns:
        from neurotopo.area_labels import infer_area_label
        classes["area"] = [infer_area_label(name) for name in classes.index]
    return classes


def load_detection_weights():
    """Mean detection score per descriptor.

    Detection beta is computed from neighbour rank order, so these scores are
    invariant to positive rescaling of a distance matrix and remain valid after
    normalisation.
    """
    rates = pd.read_csv(PLOTS / "Detection" / "detection_rates.csv")
    mean_rates = rates[rates["score_type"] == "mean"]
    return mean_rates.groupby("descriptor")["value"].mean().to_dict()


def normalise(matrix, scheme):
    """Rescale a distance matrix so descriptors are unit-commensurable.

    ``none`` reproduces the originally submitted behaviour, in which the
    combination is dominated by whichever descriptor carries the largest
    physical units.
    """
    values = np.asarray(matrix, dtype=float)
    off_diagonal = values[~np.eye(len(values), dtype=bool)]
    if scheme == "none":
        return values
    if scheme == "mean":
        return values / off_diagonal.mean()
    if scheme == "max":
        return values / off_diagonal.max()
    if scheme == "median":
        return values / np.median(off_diagonal)
    raise ValueError(f"unknown normalisation scheme {scheme!r}")


def combine(matrices, names, scheme="mean", weights=None):
    """Weighted sum of normalised descriptor distance matrices."""
    combined = None
    for name in names:
        term = normalise(matrices[name], scheme)
        if weights is not None:
            term = term * float(weights[name])
        combined = term if combined is None else combined + term
    index = matrices[names[0]].index
    return pd.DataFrame(combined, index=index, columns=index)


def ward_clusters(combined, n_clusters):
    """Ward clustering on a *condensed* distance matrix.

    ``squareform`` is required: passing the square matrix directly makes scipy
    treat each row as an observation vector in n-dimensional Euclidean space,
    which is not Ward linkage on the supplied distances.
    """
    condensed = squareform(np.asarray(combined, dtype=float), checks=False)
    return fcluster(linkage(condensed, method="ward"), t=n_clusters, criterion="maxclust")
