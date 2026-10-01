"""Convert the L-Measure Total table into per-object averages.

Checks whether the use of
these variables does not overweight/underweight certain aspects of the
morphology."

data/Briggs/lmeasure_data.csv stores L-Measure *Total* (sum) values. For any
quantity measured per compartment, per branch or per bifurcation, the total is
that quantity's mean multiplied by the number of objects, so the column becomes a
proxy for neuron size. In the submitted table 16 of 43 variables correlate with
N_branch at |Spearman rho| > 0.9, which means branch count is effectively
weighted sixteen times over in the z-scored Euclidean clustering.

Each denominator below is validated against the metric's admissible range where
one exists (Contraction and Partition_asymmetry lie in [0, 1]; Daughter_Ratio is
at least 1; bifurcation angles lie in [0, 180]). The mapping is written to
lmeasure_normalisation_map.csv so it can be audited.
"""
import numpy as np
import pandas as pd

from _common import REPO, RUN

OUT = RUN / "plots" / "lmeasure_percell"
OUT.mkdir(parents=True, exist_ok=True)

raw = pd.read_csv(REPO / "data" / "Briggs" / "lmeasure_data.csv")
raw.columns = [c.strip() for c in raw.columns]

# Total Fragmentation is the number of compartments per branch summed over
# branches, i.e. the total compartment count.
counts = {
    "per_neuron": pd.Series(1.0, index=raw.index),
    "per_compartment": raw["Fragmentation"].astype(float),
    "per_branch": raw["N_branch"].astype(float),
    "per_bifurcation": raw["N_bifs"].astype(float),
    "per_tip": raw["N_tips"].astype(float),
}

# Metrics whose L-Measure "Total" is already the quantity of interest.
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

# Excluded, with the reason recorded for the supplement.
DROP = {
    "Type": "sum of SWC structure codes (1 soma / 3 basal / 4 apical); a node-count "
            "and composition proxy, not a morphological measurement",
    "Rall_Power": "returned 0 for 142 of 145 reconstructions; uninformative",
    "Fractal_Dim": "no candidate denominator yields values >= 1, so the L-Measure "
                   "total cannot be resolved into a valid fractal dimension",
}

SCOPE = {}
for name in PER_NEURON:
    SCOPE[name] = "per_neuron"
for name in PER_COMPARTMENT:
    SCOPE[name] = "per_compartment"
for name in PER_BRANCH:
    SCOPE[name] = "per_branch"
for name in PER_BIFURCATION:
    SCOPE[name] = "per_bifurcation"
for name in PER_TIP:
    SCOPE[name] = "per_tip"

# Admissible ranges used to validate the denominator choice.
BOUNDS = {
    "Contraction": (0.0, 1.0), "Partition_asymmetry": (0.0, 1.0),
    "Daughter_Ratio": (1.0, np.inf), "Parent_Daughter_Ratio": (0.0, np.inf),
    "Bif_ampl_local": (0.0, 180.0), "Bif_ampl_remote": (0.0, 180.0),
    "Bif_tilt_local": (0.0, 180.0), "Bif_tilt_remote": (0.0, 180.0),
    "Bif_torque_local": (0.0, 180.0), "Bif_torque_remote": (0.0, 180.0),
    "Fragmentation": (1.0, np.inf),
}

metrics = [c for c in raw.columns if c not in ("group", "Filename")]
converted = pd.DataFrame({"group": raw["group"], "Filename": raw["Filename"]})
rows = []
for name in metrics:
    if name in DROP:
        rows.append({"metric": name, "scope": "dropped", "denominator": "",
                     "min": np.nan, "max": np.nan, "validated": "",
                     "note": DROP[name]})
        continue
    scope = SCOPE[name]
    values = raw[name].astype(float) / counts[scope]
    converted[name] = values
    low, high = BOUNDS.get(name, (-np.inf, np.inf))
    ok = bool(values.min() >= low - 1e-9 and values.max() <= high + 1e-9)
    rows.append({
        "metric": name, "scope": scope,
        "denominator": {"per_neuron": "1", "per_compartment": "total compartments",
                        "per_branch": "N_branch", "per_bifurcation": "N_bifs",
                        "per_tip": "N_tips"}[scope],
        "min": values.min(), "max": values.max(),
        "validated": "" if name not in BOUNDS else ("pass" if ok else "FAIL"),
        "note": "" if name not in BOUNDS else f"admissible range [{low}, {high}]",
    })

mapping = pd.DataFrame(rows)
mapping.to_csv(OUT / "lmeasure_normalisation_map.csv", index=False)
converted.to_csv(OUT / "lmeasure_data_percell.csv", index=False)

failures = mapping[mapping["validated"] == "FAIL"]
print(f"Converted {len(metrics) - len(DROP)} metrics, dropped {len(DROP)}.")
print(f"Range validation failures: {len(failures)}")
if len(failures):
    print(failures.to_string(index=False))

# --------------------------------------------------- size-dominance comparison
def size_dominance(frame, label):
    numeric = frame.drop(columns=[c for c in ("group", "Filename") if c in frame])
    rho = numeric.corrwith(numeric["N_branch"], method="spearman").abs()
    rho = rho.drop(labels=["N_branch"])
    return pd.Series({
        "representation": label,
        "n_variables": len(rho),
        "n_rho_gt_0.9": int((rho > 0.9).sum()),
        "n_rho_gt_0.7": int((rho > 0.7).sum()),
        "median_rho": rho.median(),
    })

before = raw.drop(columns=list(DROP), errors="ignore")
comparison = pd.DataFrame([
    size_dominance(before, "submitted (L-Measure totals)"),
    size_dominance(converted, "corrected (per-object averages)"),
])
comparison.to_csv(OUT / "size_dominance_comparison.csv", index=False)
print("\nCorrelation with N_branch, before and after conversion")
print(comparison.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
