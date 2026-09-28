# Morphology of ferret corticogeniculate neurons

Code, data and reconstructions for a comparison of corticogeniculate (CG) neuron
morphology between lower-order visual cortex (V1, V2) and mid-level extrastriate
cortex (PMLS, PLLS, area 21a) in the ferret, using topological Sholl descriptors.

Everything needed to reproduce the analysis is in this repository: the 117 SWC
reconstructions, the L-Measure table, the descriptor code, and the precomputed
outputs behind the published figures.

Descriptor method: Khalil R, Kallel S, Farhat A, Dlotko P. Topological Sholl
descriptors for neuronal clustering and classification. *PLoS Comput Biol.*
2022 Jun 22;18(6):e1010229.

## What is here

```
config.py                    all analysis settings
reproduce_paper.py           the pipeline, in three stages
notebooks/
    reproduce_figures.ipynb  walkthrough: data -> descriptors -> figures
neurotopo/
    neuron_processor.py      reads and standardises SWC files
    descriptors.py           the seven descriptors
    distances.py             pairwise L1 distances between step functions
    utils.py                 geometry and the step-function L1 distance
analysis/
    detection.py             detection rates and the weighted combination
    data_analysis.py         dendrograms, clustering, PCA
    lmeasure_analysis.py     the conventional morphometric comparison
    supporting/              the supporting analyses (see below)
data/Briggs/
    V1-V2/                   67 reconstructions
    PMLS-PLLS/               50 reconstructions
    lmeasure_data.csv        40 L-Measure variables per neuron
figures/results/             step functions, distance matrices and result
                             tables behind the published figures
```

Every figure is computed from the data, by `reproduce_paper.py` or by the
notebook. None is stored as an image.

## Requirements

Python 3.9 or newer.

```bash
pip install -r requirements.txt
```

This runs on a normal laptop. There is no cluster or cloud dependency: the
whole cohort is 117 neurons, and the distance matrices are 117 x 117.

## Running it

From the repository root:

```bash
python reproduce_paper.py descriptors    # SWC -> step functions
python reproduce_paper.py figures        # histograms, boxplots, L-Measure
python reproduce_paper.py dendrograms    # distances, combination, clustering
```

or all three, reusing whatever is already computed:

```bash
python reproduce_paper.py all
```

On Windows, set UTF-8 output first, otherwise a diagnostic message with a
non-ASCII character raises `UnicodeEncodeError` under `cp1252`:

```powershell
$env:PYTHONIOENCODING = 'utf-8'
```

Each stage caches its result and reuses it, so a rerun resumes rather than
recomputing. Results are written to `figures/Briggs_paper_reproduction/`.

### Reproducing from scratch

This repository ships the descriptor outputs, so by default the descriptor
stage loads them rather than recomputing. To force a genuine end-to-end
reproduction from the SWC files, run with a fresh run name:

```bash
NEUROTOPO_RUN=my_run python reproduce_paper.py all
```

```powershell
$env:NEUROTOPO_RUN = 'my_run'; python reproduce_paper.py all
```

That writes to `figures/my_run/` and recomputes everything, which takes roughly
an hour, dominated by the descriptor stage. Setting `PLOT_STEP_FUNCTION = False`
in `config.py` cuts that substantially: it otherwise writes one PNG per neuron
per descriptor.

## The notebook

`notebooks/reproduce_figures.ipynb` walks through the analysis end to end,
showing the code that produces each published figure alongside the figure
itself. It reads the shipped precomputed outputs, so it runs in under a minute
without recomputing descriptors.

```bash
pip install jupyter
jupyter notebook notebooks/reproduce_figures.ipynb
```

## Supporting analyses

`analysis/supporting/` holds the analyses reported alongside the main result.
Each is a standalone script run from the repository root, for example:

```bash
python analysis/supporting/run_variance_partition.py
```

| Script | What it reports |
|---|---|
| `run_variance_partition.py` | PERMANOVA: variance explained by cortical tier vs by area |
| `run_representation_stability.py` | subsampling stability of the two representations |
| `run_cluster_number_selection.py` | silhouette across cluster numbers |
| `run_normalisation_comparison.py` | effect of rescaling descriptor distance matrices |
| `run_combination_comparison.py` | detection weighting vs unweighted |
| `run_gridsearch_audit.py` | exhaustive grid search over descriptor weights |
| `run_method_comparison.py` | descriptors vs conventional morphometrics |
| `run_lmeasure_correction.py` | per-object vs total L-Measure variables |
| `run_descriptor_variant_comparison.py` | sensitivity to descriptor definitions |
| `run_exclusion_sensitivity.py` | sensitivity to the excluded reconstruction |
| `make_manuscript_figures.py` | regenerates the published figure panels |

These read from `figures/results/`, which is included. To
point them at your own run instead, set `NEUROTOPO_RUN`.

## Data

117 reconstructions, from two published studies, retrieved from NeuroMorpho.Org:

| Group | n | Areas | Source |
|---|---|---|---|
| Lower-order | 67 | V1, V2 | Hasse et al. 2018 |
| Mid-level | 50 | PMLS, PLLS, area 21a | Adusei et al. 2021 |

The archive does not separate V1 from V2; both are labelled as visual cortex
(Brodmann areas 17 and 18), so they are analysed as one lower-order group.

Class labels come from the folder layout: each immediate subfolder of
`data/Briggs/` is one class. `lmeasure_data.csv` holds 40 L-Measure variables
per neuron, used for the comparison against conventional morphometrics.

One reconstruction is excluded, listed in `EXCLUDED_NEURONS` in `config.py`
with the reason.

## Settings that change the numbers

In `config.py`:

- `REMOVE_TYPES = ['axon']` — the descriptors characterise the dendritic tree.
- `NORMALIZE_DESCRIPTOR_RADII = True` — each neuron's radial coordinate is
  divided by its own maximum distance from the soma, so descriptors compare
  shape rather than size.
- `DISTANCE_METRIC = 'l1'` — the L1 distance between step functions.
- `NORMALIZE_DESCRIPTOR_DISTANCES = True` — rescales each descriptor distance
  matrix to unit mean before combining. The descriptors carry different
  physical units and their mean pairwise distances span about four orders of
  magnitude, so an unnormalised sum is governed by whichever descriptor has the
  largest units rather than by what any of them measure. The comment in
  `config.py` sets this out in full.
- `LINKAGE_METHOD = 'ward'`, `NUM_OF_CLUSTERS = 2`.

## Descriptors

Tortuosity, Branching Pattern, Wiring, Flux, Leaf, Spread and Energy. Each maps
a neuron to a step function on [0, 1] against normalised radial distance from
the soma, so one L1 distance applies to all of them. The seven per-descriptor
distance matrices are combined into one by weighting each by its detection
rate.

Two descriptors from the reference method are not used here. Taper rate needs
per-node dendritic thickness, and the SWC radius column in this dataset holds
tracing presets rather than measurements: 52 of the 117 reconstructions carry a
single radius for every dendritic node, and the rest use between two and seven
discrete levels that are integer multiples of a per-file base value. Sholl-TMD
produces persistence diagrams rather than real-valued step functions, so it
admits no L1 distance between functions and cannot enter a weighted sum of
step-function distance matrices.

## Licence

See `LICENSE`.
