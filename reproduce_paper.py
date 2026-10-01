from __future__ import annotations

import argparse
import copy
import json
import os
import pickle
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import config
import neurotopo.descriptors as descriptors
import neurotopo.distances as distances
from analysis.data_analysis import DendrogramAnalysis, Plotter
from analysis.detection import Detection
from analysis.lmeasure_analysis import LMeasureAnalysis
from neurotopo.neuron_processor import process_swc_directory


OUTPUT = Path(config.SAVE_DIRECTORY)
PLOTS = Path(config.PLOTS_DIRECTORY)
PICKLES = Path(config.PICKLE_DIRECTORY)

DESCRIPTOR_FUNCTIONS = {
    'Tortuosity': descriptors.calculate_tortuosity,
    'Branching_Pattern': descriptors.calculate_branching_pattern,
    'Wiring': descriptors.calculate_wiring_descriptor,
    'Flux': descriptors.calculate_flux,
    'Leaf': descriptors.calculate_leaf,
    'Spread': descriptors.calculate_spread_descriptor,
    'Energy': descriptors.calculate_energy,
}

# Taper rate and Sholl-TMD are not computable for this dataset.
#
# Taper rate requires per-node dendritic thickness. The SWC radius column here
# holds tracing presets rather than measurements: 52 of the 117 reconstructions
# carry a single radius for every dendritic node, and the rest use between two
# and seven discrete levels that are integer multiples of a per-file base value.
# A rate of change of diameter cannot be recovered from that.
#
# Sholl-TMD produces persistence diagrams rather than real-valued step
# functions, so it admits no L1 distance between functions and cannot enter a
# weighted sum of the step-function distance matrices.


def ensure_directories():
    for directory in (
        OUTPUT,
        PLOTS,
        PICKLES,
        Path(config.LMEASURE_FIGURES),
        Path(config.DETECTION_DIRECTORY),
        Path(config.METRIC_LEARNING_DIRECTORY),
        Path(config.MORPHOMETRICS_DIRECTORY),
        Path(config.DESCRIPTORS_DIRECTORY),
    ):
        directory.mkdir(parents=True, exist_ok=True)


def write_manifest():
    manifest = {
        'python': sys.version,
        'platform': platform.platform(),
        'base_directory': config.BASE_DIRECTORY,
        'save_directory': config.SAVE_DIRECTORY,
        'project_name': config.PROJECT_NAME,
        'distance_metric': config.DISTANCE_METRIC,
        'linkage_method': config.LINKAGE_METHOD,
        'number_of_clusters': config.NUM_OF_CLUSTERS,
        'normalize_descriptor_radii': config.NORMALIZE_DESCRIPTOR_RADII,
        'lmeasure_standardize': config.LMEASURE_DATA_STANDARIZE,
        'lmeasure_normalization': config.LMEASURE_NORMALIZE_METHOD,
        'remove_types': config.REMOVE_TYPES,
        'excluded_descriptors': config.EXCLUDED_DESCRIPTORS,
        'excluded_neurons': config.EXCLUDED_NEURONS,
        'plot_neuron': config.PLOT_NEURON,
        'plot_step_function': config.PLOT_STEP_FUNCTION,
        'descriptors_retained_for_paper': list(DESCRIPTOR_FUNCTIONS),
    }
    (OUTPUT / 'run_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')


def load_or_process_neurons():
    data_path = OUTPUT / 'neuron_data.pkl'
    classes_path = OUTPUT / 'neuron_class_df.csv'
    if data_path.exists() and classes_path.exists():
        print('Loading cached neuron data from the clean reproduction folder.')
        with data_path.open('rb') as handle:
            neuron_data = pickle.load(handle)
        neuron_class_df = pd.read_csv(classes_path)
    else:
        print('Processing raw SWC files.')
        neuron_data, neuron_class_df, report = process_swc_directory()
        with data_path.open('wb') as handle:
            pickle.dump(neuron_data, handle)
        neuron_class_df.to_csv(classes_path, index=False)
        report.to_csv(OUTPUT / 'swc_standardization_report.csv', index=False)
    counts = neuron_class_df['class'].value_counts().to_dict()
    # 117 reconstructions are available; 31_1_3_PMLS and 31_1_6_PMLS are traced
    # with basal dendrites only and so are not pyramidal cells, leaving 115.
    # See config.EXCLUDED_NEURONS. Derived from config rather than hardcoded so
    # the guard stays honest if the exclusion list changes.
    cohort_exclusions = sum(
        1 for name in config.EXCLUDED_NEURONS
        if name.endswith(('_PMLS', '_PLLS', '_21a'))
    )
    expected_total = 117 - cohort_exclusions
    if len(neuron_data) != expected_total:
        raise RuntimeError(
            f'Unexpected paper cohort: {len(neuron_data)} neurons '
            f'(expected {expected_total}), classes={counts}'
        )
    return neuron_data, neuron_class_df


def stage_descriptors():
    neuron_data, neuron_class_df = load_or_process_neurons()
    descriptor_path = OUTPUT / 'all_descriptors.pkl'
    polarity_path = OUTPUT / 'polarity_matrices.pkl'
    if descriptor_path.exists() and polarity_path.exists():
        print('Loading cached paper descriptors.')
        with descriptor_path.open('rb') as handle:
            all_descriptors = pickle.load(handle)
        return neuron_data, neuron_class_df, all_descriptors

    all_descriptors = {name: {} for name in DESCRIPTOR_FUNCTIONS}
    polarity_matrices = {}
    total = len(neuron_data)
    for index, (neuron_name, neuron) in enumerate(neuron_data.items(), 1):
        print(f'Computing descriptors {index}/{total}: {neuron_name}')
        for descriptor_name, function in DESCRIPTOR_FUNCTIONS.items():
            result = function(neuron)
            # Spread returns (index, volume, path) and Energy returns
            # (values, vectors); in both cases the first element is the step
            # function that defines the descriptor.
            if isinstance(result, tuple):
                result = result[0]
            all_descriptors[descriptor_name][neuron_name] = result
        if 'Polarity' not in config.EXCLUDED_DESCRIPTORS:
            polarity_matrices[neuron_name] = descriptors.calculate_polarity(neuron)

    with descriptor_path.open('wb') as handle:
        pickle.dump(all_descriptors, handle)
    with polarity_path.open('wb') as handle:
        pickle.dump(polarity_matrices, handle)
    descriptors.plot_median_leaf_jump_data(neuron_data, neuron_class_df, all_descriptors)
    print('Descriptor stage complete:', list(all_descriptors))
    return neuron_data, neuron_class_df, all_descriptors


def stage_histograms_and_lmeasure():
    _, neuron_class_df, all_descriptors = stage_descriptors()

    lmeasure_data = pd.read_csv(config.LMEASURE_CSV_PATH)
    lmeasure = LMeasureAnalysis(
        neuron_class_df=neuron_class_df.copy(deep=True),
        lmeasure_data=lmeasure_data,
    )
    lmeasure.run_analysis()

    plotter = Plotter(n_components=2, neuron_class_df=neuron_class_df)
    panels = {}
    for descriptor_name, descriptor_data in all_descriptors.items():
        descriptor_values = descriptors.extract_descriptor_values(descriptor_data)
        plotter.plot_descriptor_histograms_by_class(
            descriptor_values,
            descriptor_name=descriptor_name,
        )
        plotter.plot_descriptor_boxplots_by_class(
            descriptor_values,
            descriptor_name=descriptor_name,
        )
        panels[descriptor_name] = descriptor_values
    plotter.plot_descriptor_histograms_by_class_one_panel(panels)
    plotter.plot_descriptor_histograms_by_class_one_panel_with_texture(panels)
    print('Histogram and L-Measure figure stage complete.')


def stage_distance_dendrograms():
    _, neuron_class_df, all_descriptors = stage_descriptors()
    distance_path = PICKLES / 'individual_distance_matrices.pkl'
    if distance_path.exists():
        print('Loading cached individual distance matrices.')
        with distance_path.open('rb') as handle:
            distance_matrices = pickle.load(handle)
    else:
        distance_matrices = distances.compute_pairwise_differences(all_descriptors)
        with distance_path.open('wb') as handle:
            pickle.dump(distance_matrices, handle)

    # Quantify dependence among descriptor distance representations while
    # preserving the existing detection-weighted combination and optimizer.
    distances.write_descriptor_distance_dependence_report(distance_matrices, PLOTS)

    detection = Detection(
        distance_matrices=distance_matrices,
        neuron_class_df=neuron_class_df,
    )
    _, _, _, _, combined_matrix, _ = detection.evaluate_all_detection_rates()
    combined_matrix.to_pickle(PICKLES / 'combined_detection_weighted_matrix.pkl')

    dendrogram = DendrogramAnalysis(
        data=combined_matrix,
        neuron_class_df=copy.deepcopy(neuron_class_df),
        save_directory=config.PLOTS_DIRECTORY,
        descriptor_name='combined_matrices_detection_weighted',
        highlight_neurons=config.HIGHLIGHT_NEURONS,
        is_distance_matrix=True,
        cutoff_dist=0,
    )
    clustered_df = dendrogram.run_analysis()
    clustered_df.to_csv(PICKLES / 'combined_detection_cluster_assignments.csv', index=False)

    plotter = Plotter(n_components=2, neuron_class_df=clustered_df)
    plotter.plot_pca_from_distance_matrix(
        combined_matrix,
        'combined_distance_matrix_with_weights',
    )
    print('Distance-matrix and combined-dendrogram stage complete.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'stage',
        choices=('descriptors', 'figures', 'dendrograms', 'all'),
        default='all',
        nargs='?',
    )
    args = parser.parse_args()
    ensure_directories()
    write_manifest()
    print('Input:', config.BASE_DIRECTORY)
    print('Output:', config.SAVE_DIRECTORY)
    if args.stage == 'descriptors':
        stage_descriptors()
    elif args.stage == 'figures':
        stage_histograms_and_lmeasure()
    elif args.stage == 'dendrograms':
        stage_distance_dendrograms()
    else:
        stage_histograms_and_lmeasure()
        stage_distance_dendrograms()


if __name__ == '__main__':
    main()
