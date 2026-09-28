import os
# Cause: This warning is about the change in default value for the n_init parameter of KMeans. Currently, the default is 10, but it will change to 'auto' in future versions.
# Solution: To avoid this warning and ensure consistent behavior in future versions of scikit-learn, you should explicitly set the n_init parameter when instantiating KMeans.
os.environ["OMP_NUM_THREADS"] = "1" 
import config

import numpy as np
import neurotopo.utils as utl
import config
import pandas as pd
from scipy.stats import skew, kurtosis, mode, entropy, hmean
from numpy import trapz
from scipy.stats import pearsonr
from scipy.signal import find_peaks
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis, entropy, mode, pearsonr, hmean
from scipy.integrate import trapezoid
import numpy as np
from scipy.optimize import differential_evolution
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
from itertools import product
import pickle
import numpy as np
import pandas as pd
from itertools import product
import time
import os
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import SpectralClustering
import community.community_louvain as community
import hdbscan
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis, entropy, mode, hmean
from scipy.signal import find_peaks
from scipy.fftpack import fft
from scipy.stats import mode

import warnings
warnings.filterwarnings('ignore')




exclude_descriptors = config.EXCLUDED_DESCRIPTORS
os.makedirs(config.PICKLE_DIRECTORY, exist_ok=True)
combination_mode = config.DISTANCE_MATRICES_COMBINATION_MODE


import numpy as np
import pandas as pd
from scipy.spatial.distance import squareform
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import SpectralClustering

def compute_similarity_matrices(distance_matrices, gamma=None):
    """
    Converts distance matrices into similarity matrices using the Gaussian (RBF) kernel.

    Parameters:
    - distance_matrices: Dictionary of distance matrices for each descriptor.
    - gamma (float, optional): Kernel width parameter. If None, it is set to 1/median(distance).

    Returns:
    - similarity_matrices: Dictionary where each descriptor has a similarity matrix.
    """
    similarity_matrices = {}

    for descriptor, dist_matrix in distance_matrices.items():
        print(f"Computing similarity matrix for {descriptor}...")

        # Convert to numpy array
        distance_array = dist_matrix.to_numpy()

        # Compute gamma if not provided
        if gamma is None:
            median_distance = np.median(distance_array[distance_array > 0])  # Ignore zero distances
            gamma = 1 / median_distance if median_distance > 0 else 1  # Prevent division by zero

        # Apply Gaussian (RBF) kernel
        similarity_array = np.exp(-gamma * np.square(distance_array))

        # Convert back to DataFrame
        similarity_matrices[descriptor] = pd.DataFrame(similarity_array, 
                                                        index=dist_matrix.index, 
                                                        columns=dist_matrix.columns)

    return similarity_matrices




def process_similarity_matrices(similarity_matrices, clustering_method="louvain", threshold=0.5, use_knn=False, k_neighbors=5):
    """
    Processes all descriptors, constructs neuron graphs, applies clustering, and visualizes results.

    Parameters:
    - similarity_matrices (dict): Dictionary where each descriptor has a similarity matrix.
    - clustering_method (str): "louvain", "spectral", or "hdbscan".
    - threshold (float): Minimum similarity required to connect neurons (if not using KNN).
    - use_knn (bool): If True, construct a K-Nearest Neighbors graph instead of using a threshold.
    - k_neighbors (int): Number of neighbors in KNN graph construction.
    - save_dir (str): Directory to save plots.

    Returns:
    - all_graphs (dict): Dictionary where each descriptor has a NetworkX neuron similarity graph.
    - all_cluster_labels (dict): Dictionary where each descriptor has neuron cluster labels.
    """

    all_graphs = {}
    all_cluster_labels = {}

    for descriptor, sim_matrix in similarity_matrices.items():
        print(f"Processing descriptor: {descriptor}")

        # Step 1: Construct Graph from Similarity Matrix
        print(f"  - Constructing neuron graph for {descriptor}")
        neurons = sim_matrix.index
        graph = nx.Graph()

        # Add neurons as nodes
        for neuron in neurons:
            graph.add_node(neuron)

        similarity_array = sim_matrix.to_numpy()

        if use_knn:
            # Use KNN-based method to construct edges
            from sklearn.neighbors import NearestNeighbors
            knn = NearestNeighbors(n_neighbors=k_neighbors, metric="precomputed")
            knn.fit(1 - similarity_array)  # Convert similarity to distance
            _, indices = knn.kneighbors(1 - similarity_array)

            for i, neighbors in enumerate(indices):
                for j in neighbors:
                    if i != j:
                        graph.add_edge(neurons[i], neurons[j], weight=similarity_array[i, j])

        else:
            # Use threshold-based method to construct edges
            for i in range(len(neurons)):
                for j in range(i + 1, len(neurons)):
                    if similarity_array[i, j] > threshold:  # Only add edges above the threshold
                        graph.add_edge(neurons[i], neurons[j], weight=similarity_array[i, j])

        # Step 2: Apply Clustering
        print(f"  - Applying {clustering_method} clustering for {descriptor}")
        neuron_cluster_labels = apply_neuron_clustering(graph, method=clustering_method)

        # Step 3: Plot and Save Clustered Graph
        print(f"  - Plotting and saving neuron graph clustering results for {descriptor}")
        plot_neuron_clusters(graph, neuron_cluster_labels, descriptor_name=f"louvain_{descriptor}")

        # Step 2: Apply Clustering
        print(f"  - Applying {clustering_method} clustering for {descriptor}")
        neuron_cluster_labels = apply_neuron_clustering(graph, method="spectral")

        # Step 3: Plot and Save Clustered Graph
        print(f"  - Plotting and saving neuron graph clustering results for {descriptor}")
        plot_neuron_clusters(graph, neuron_cluster_labels, descriptor_name=f"Spectral_{descriptor}")

        # Store results
        all_graphs[descriptor] = graph
        all_cluster_labels[descriptor] = neuron_cluster_labels

    return all_graphs, all_cluster_labels




import community.community_louvain as community
from sklearn.cluster import SpectralClustering
import hdbscan

def apply_neuron_clustering(graph, method="louvain"):
    """
    Clusters neurons using a graph-based clustering method.

    Parameters:
    - graph (networkx.Graph): The neuron-level similarity graph.
    - method (str): Clustering method: "louvain", "spectral", or "hdbscan".

    Returns:
    - cluster_labels (dict): Dictionary where each neuron has a cluster label.
    """

    neurons = list(graph.nodes)
    adjacency_matrix = nx.to_numpy_array(graph)

    if method == "spectral":
        clustering = SpectralClustering(n_clusters=3, affinity='precomputed', random_state=42)
        labels = clustering.fit_predict(adjacency_matrix)

    elif method == "louvain":
        partition = community.best_partition(nx.Graph(graph))  # Convert to undirected graph
        labels = [partition[neuron] for neuron in neurons]

    elif method == "hdbscan":
        clustering = hdbscan.HDBSCAN(metric="precomputed", min_cluster_size=3)
        labels = clustering.fit_predict(1 - adjacency_matrix)  # Convert similarity to distance

    else:
        raise ValueError("Invalid clustering method. Choose 'spectral', 'louvain', or 'hdbscan'.")

    return dict(zip(neurons, labels))


import os
import matplotlib.pyplot as plt

def plot_neuron_clusters(graph, cluster_labels, descriptor_name="neuron_clusters"):
    """
    Plots the clustered neuron graph.

    Parameters:
    - graph (networkx.Graph): The neuron similarity graph.
    - cluster_labels (dict): Cluster labels for each neuron.
    - save_dir (str): Directory to save plots.
    - descriptor_name (str): Descriptor name for saving plots.
    """


    unique_clusters = set(cluster_labels.values())
    color_map = {cluster: plt.cm.viridis(i / len(unique_clusters)) for i, cluster in enumerate(unique_clusters)}

    plt.figure(figsize=(8, 6))
    pos = nx.spring_layout(graph, seed=42)  # Positioning of nodes
    node_colors = [color_map[cluster_labels[node]] for node in graph.nodes]

    nx.draw(
        graph, pos, node_color=node_colors, with_labels=False,
        edge_color='gray', node_size=500, font_size=10
    )

    plt.title(f"Neuron-Level Graph Clustering - {descriptor_name}")
    
    similarity_dir = os.path.join(config.PLOTS_DIRECTORY, "similarity_matrix")

    if not os.path.exists(similarity_dir):
        os.makedirs(similarity_dir)


    file_name = f"Similarity_Matrix_Neuron_Clustering_{descriptor_name}.png"
    plt.savefig(os.path.join(similarity_dir, file_name), dpi=300, bbox_inches='tight', pad_inches=0.5)
    plt.close()





def compute_pairwise_differences(all_descriptors):
    """
    Compute the pairwise L1 distances between the step functions of neurons for each descriptor.
    
    Parameters:
    - all_descriptors: A dictionary where each key is a descriptor and each value is a dictionary
                       with neuron names as keys and their respective step function lists of tuples as values.
                       
    Returns:
    - distance_matrices: A dictionary where each key is a descriptor and each value is a DataFrame
                         representing the pairwise L1 distances between neurons for that descriptor.
    """
    import pandas as pd

    # Dictionary to store the distance matrices for each descriptor
    distance_matrices = {}
    
    

    # Iterate over each descriptor
    for descriptor_name, descriptor_data in all_descriptors.items():
        
        # Skip the descriptor if it is in the exclusion list
        if descriptor_name in exclude_descriptors :
           print(f'Skipping descriptor {descriptor_name} as it is in the exclusion list.')
           continue
        
        print(f'Calculating pairwise {config.DISTANCE_METRIC} differences for {descriptor_name}')
        
        # Get the list of neuron names
        neuron_names = list(descriptor_data.keys())
        num_neurons = len(neuron_names)
        
        # Create a DataFrame to store pairwise distances, initialized with zeros
        distance_matrix = pd.DataFrame(
            data=0.0,
            index=neuron_names,
            columns=neuron_names
        )
        
        # Compute pairwise L1 differences for each pair of neurons
        for i in range(num_neurons):
            for j in range(i, num_neurons):
                
                if i == j:
                    continue
                
                neuron1_name = neuron_names[i]
                neuron2_name = neuron_names[j]
                
                # Extract the step function data for both neurons.
                step_func1 = descriptor_data[neuron1_name]
                step_func2 = descriptor_data[neuron2_name]
                
                # Convert the step function tuples into separate x and y lists
                x1, y1 = zip(*step_func1)
                x2, y2 = zip(*step_func2)

                # Calculate the L1 difference between the two step functions.
                # diff = utl.calculate_l1_difference((x1, y1), (x2, y2))
                
                                # Compute the selected distance metric
                if config.DISTANCE_METRIC == "l1":
                    diff = utl.calculate_l1_difference((x1, y1), (x2, y2))
                elif config.DISTANCE_METRIC == "wasserstein":
                    diff = utl.calculate_wasserstein_distance((x1, y1), (x2, y2))
                else:
                    raise ValueError("Invalid distance metric. Use 'l1' or 'wasserstein'.")
                
                # Store the calculated difference in the distance matrix
                distance_matrix.at[neuron1_name, neuron2_name] = diff
                distance_matrix.at[neuron2_name, neuron1_name] = diff

        # Store the distance matrix for the current descriptor
        distance_matrices[descriptor_name] = distance_matrix

    return distance_matrices

# give me anither soluton based on wighted euclidean sum 
# ====================================================================

# import numpy as np
# import pandas as pd
# from itertools import combinations
# import os
# from scipy.cluster.hierarchy import dendrogram, linkage
# import matplotlib.pyplot as plt

# def initialize_weights(distance_matrices, initial_weight=1.0):
#     """Initialize weights for each matrix descriptor."""
#     return {descriptor: initial_weight for descriptor in distance_matrices}

# def combine_matrices_with_weights(distance_matrices, weights):
#     """Combine matrices using the provided weights dictionary."""
#     combined_dist_mat = np.zeros_like(next(iter(distance_matrices.values())))
#     for descriptor in distance_matrices:
#         combined_dist_mat += weights[descriptor] * distance_matrices[descriptor]
#     return combined_dist_mat

# def increment_counter(weights, start, stop, dx):
#     """Increment weights considering stop conditions with full combination exploration."""
#     carry = True  # Indicates if we need to carry the increment to the next weight
#     for descriptor in weights:
#         if carry:
#             if weights[descriptor] + dx <= stop[descriptor]:
#                 weights[descriptor] += dx
#                 carry = False  # Increment successful, no carry needed
#             else:
#                 weights[descriptor] = start[descriptor]  # Reset this weight and carry the increment
#     return weights


# def combine_and_optimize_distance_matrices(distance_matrices, neuron_class_df, dx=0.1, max_iterations=10000):
#     weights = initialize_weights(distance_matrices, 1.0)
#     start_weights = initialize_weights(distance_matrices, 1.0)
#     stop_weights = initialize_weights(distance_matrices, 3.0)
    
#     current_best_score = 0
#     current_best_weights = weights.copy()
#     iteration_count = 0
#     classes = neuron_class_df['class'].unique()
    
#     # Dictionary to store score and weights at each iteration
#     score_tracking = {}
    
#     while iteration_count < max_iterations:
#         print(f'Iteration {iteration_count} out of {max_iterations} max iterations')
#         combined_matrix = combine_matrices_with_weights(distance_matrices, weights)
        
#         max_distance_within_class = []
#         for current_class in classes:
#             neuron_names = neuron_class_df[neuron_class_df['class'] == current_class]['neuron_name']
#             internal_distances = combined_matrix.loc[neuron_names, neuron_names]
#             max_distance = np.nanmax(internal_distances.values)
#             max_distance_within_class.append(max_distance)
#             print(f"Maximum internal distance for class {current_class}: {max_distance}")
        
#         internal_dist = sum(max_distance_within_class)
        
#         min_external_distances = []
#         for class_1, class_2 in combinations(classes, 2):
#             indices_1 = neuron_class_df[neuron_class_df['class'] == class_1]['neuron_name'].tolist()
#             indices_2 = neuron_class_df[neuron_class_df['class'] == class_2]['neuron_name'].tolist()
#             external_distances = combined_matrix.loc[indices_1, indices_2]
#             min_external_distances.append(external_distances.min().min())
        
#         external_dist = min(min_external_distances) if min_external_distances else float('inf')
#         score = external_dist / internal_dist if internal_dist > 0 else 0
       
       
        
#         if score > current_best_score:
#             current_best_score = score
#             current_best_weights = weights.copy()
#             print("New best score:", current_best_score, "with weights:", current_best_weights)
        
#         # Update score and weights tracking
#         score_tracking[iteration_count] = {'score': score, 
#                                            'weights': weights.copy(), 
#                                            'current_best_weights': current_best_weights.copy()}
        
#         weights = increment_counter(weights, start_weights, stop_weights, dx)
#         if weights == current_best_weights:
#             break
#         iteration_count += 1
        
#     # Convert score tracking dictionary to DataFrame
#     score_data = []
#     for iteration, data in score_tracking.items():
#         row = {
#             'iteration': iteration,
#             'score': data['score'],
#             **{f'weight_{k}': v for k, v in data['weights'].items()},
#             **{f'best_weight_{k}': v for k, v in data['current_best_weights'].items()}
#         }
#         score_data.append(row)
#     score_df = pd.DataFrame(score_data)
    
#     combined_dist_matrix = combine_matrices_with_weights(distance_matrices, current_best_weights)
    
#     # Save the combined distance matrix and the best weights
#     # Update these paths according to your configuration
#     np.save(os.path.join(config.PICKLE_DIRECTORY, 'combined_dist_matrix.npy'), combined_dist_matrix)
#     np.save(os.path.join(config.PICKLE_DIRECTORY, 'best_weights.npy'), current_best_weights)
#     np.save(os.path.join(config.PICKLE_DIRECTORY, 'score_df.npy'), score_df)
    
#     return combined_dist_matrix, current_best_weights, score_df

# ====================================================================================


import numpy as np
import pandas as pd
from itertools import combinations
from sklearn.decomposition import PCA
import os



# def combine_matrices_with_weights(distance_matrices, weights):
#     """Combine matrices using the provided weights dictionary."""
#     # Assuming `distance_matrices` are pandas DataFrames
#     combined_dist_mat = pd.DataFrame(
#         np.zeros_like(next(iter(distance_matrices.values()))), 
#         index=next(iter(distance_matrices.values())).index, 
#         columns=next(iter(distance_matrices.values())).columns
#     )
    
#     for descriptor in distance_matrices:
#         combined_dist_mat += weights[descriptor] * distance_matrices[descriptor]
#     return combined_dist_mat

def simplex_weight_grid(n_descriptors, resolution):
    """Regular grid on the (n-1)-simplex, for the S4.2 grid search.

    The S4.2 score E / sum_k I_k is homogeneous of degree zero in the weight
    vector, so only its direction matters and the simplex is the whole search
    space. A regular simplex grid is therefore exhaustive, unlike the
    box grid used by ``increment_counter`` below, which both excludes zero
    weights and is truncated by MAX_ITERATIONS long before it finishes.
    """
    from itertools import combinations_with_replacement

    points = []
    for cut in combinations_with_replacement(range(resolution + 1), n_descriptors - 1):
        cuts = (0,) + cut + (resolution,)
        points.append([cuts[i + 1] - cuts[i] for i in range(n_descriptors)])
    return np.asarray(points, dtype=float) / resolution


def grid_search_optimal_weights(distance_matrices, neuron_class_df, resolution=20):
    """Exhaustive grid search over descriptor weights.

    Implements supplementary section 4.2 of Khalil R, Kallel S, Farhat A,
    Dlotko P, Topological Sholl descriptors for neuronal clustering and
    classification, PLoS Comput Biol 2022;18(6):e1010229.

    Maximises ``sc = E / sum_k I_k`` where ``I_k`` is the maximal within-class
    distance in class k and ``E`` the minimal between-class distance, over
    non-negative weights on the simplex.

    Returns ``(best_weights, best_score, combined_matrix)``.

    Prefer this over :func:`combine_and_optimize_distance_matrices`, whose grid
    spans only [1.0, 3.0] per descriptor -- so a descriptor can never be
    dropped -- and which explores about 1% of even that box before hitting
    MAX_ITERATIONS.
    """
    names = list(distance_matrices)
    index = distance_matrices[names[0]].index
    labels = neuron_class_df.set_index("neuron_name").loc[index, "class"].to_numpy()

    iu, ju = np.triu_indices(len(index), k=1)
    pairs = np.column_stack(
        [np.asarray(distance_matrices[n], dtype=float)[iu, ju] for n in names])
    same = labels[iu] == labels[ju]
    within = [same & (labels[iu] == c) for c in np.unique(labels)]
    between = ~same

    alphas = simplex_weight_grid(len(names), resolution)
    best_score, best_alpha = -np.inf, None
    for start in range(0, len(alphas), 4096):
        chunk = alphas[start:start + 4096]
        combined = pairs @ chunk.T
        internal = sum(combined[m].max(axis=0) for m in within)
        external = combined[between].min(axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            scores = np.where(internal > 0, external / internal, 0.0)
        top = int(np.argmax(scores))
        if scores[top] > best_score:
            best_score, best_alpha = float(scores[top]), chunk[top]

    weights = {n: float(w) for n, w in zip(names, best_alpha)}
    combined_matrix = combine_matrices_with_weights(distance_matrices, weights)
    return weights, best_score, combined_matrix


def initialize_weights(distance_matrices, initial_weight=1.0):
    """Initialize weights for each matrix descriptor."""
    return {descriptor: initial_weight for descriptor in distance_matrices}

def increment_counter(weights, start, stop, dx):
    """Increment weights considering stop conditions with full combination exploration.

    Known limitation: this is a little-endian odometer over the box
    [start, stop]^n. With the configured dx=0.1 over [1.0, 3.0] and six
    descriptors the box holds 21**6 = 85,766,121 points, so MAX_ITERATIONS =
    1,000,000 stops it after ~1.2%, leaving the last two descriptors pinned at
    their initial weight. Use :func:`grid_search_optimal_weights` instead.
    """
    carry = True  # Indicates if we need to carry the increment to the next weight
    all_reset = True  # Flag to check if all descriptors have been reset

    for descriptor in weights:
        if carry:
            if weights[descriptor] + dx <= stop[descriptor]:
                weights[descriptor] += dx
                carry = False  # Increment successful, no carry needed
                all_reset = False  # At least one descriptor was incremented
            else:
                weights[descriptor] = start[descriptor]  # Reset this weight and carry the increment

    # If all weights have been reset, it means we've fully cycled through
    if all_reset:
        return None  # Indicate that we are done with all combinations

    return weights

def combine_and_optimize_distance_matrices(distance_matrices, neuron_class_df, dx, max_iterations,combination_type):
    
    weights = initialize_weights(distance_matrices, 1.0)
    start_weights = initialize_weights(distance_matrices, 1.0)
    stop_weights = initialize_weights(distance_matrices, 3.0)
    
    current_best_score = 0
    current_best_weights = weights.copy()
    iteration_count = 0
    classes = neuron_class_df['class'].unique()
    
    # Dictionary to store score and weights at each iteration
    score_tracking = {}
    
    while iteration_count < max_iterations:
        
        # Normalize a copy for matrix combination, but keep the raw grid
        # counter in the 1.0-to-3.0 range. Replacing the counter itself with
        # normalized values prevents increment_counter() from ever carrying
        # to the remaining descriptors.
        norm_factor = sum(weights.values())
        normalized_weights = {k: v / norm_factor for k, v in weights.items()}

        if iteration_count == 0 or iteration_count % 10000 == 0:
            print(f'{combination_type} Iteration {iteration_count} out of {max_iterations} max iterations')
        if combination_mode == 'max_component':
            combined_matrix = combine_matrices_max_component(distance_matrices, normalized_weights)
        elif combination_mode == 'min_component':
            combined_matrix = combine_matrices_min_component(distance_matrices, normalized_weights)
        elif combination_mode == 'nonlinear_power_mean':
            combined_matrix = combine_matrices_nonlinear_power_mean(distance_matrices, normalized_weights, p=0.5)
        else:
            combined_matrix = combine_matrices_with_weights(distance_matrices, normalized_weights)

        # combined_matrix = combine_matrices_with_weights(distance_matrices, weights)
     
       
        max_distance_within_class = []
        for current_class in classes:
            neuron_names = neuron_class_df[neuron_class_df['class'] == current_class]['neuron_name']
            internal_distances = combined_matrix.loc[neuron_names, neuron_names]
            max_distance = np.nanmax(internal_distances.values)
            max_distance_within_class.append(max_distance)
            # print(f"Maximum internal distance for class {current_class}: {max_distance}")
        
        internal_dist = sum(max_distance_within_class)
        
        min_external_distances = []
        for class_1, class_2 in combinations(classes, 2):
            indices_1 = neuron_class_df[neuron_class_df['class'] == class_1]['neuron_name'].tolist()
            indices_2 = neuron_class_df[neuron_class_df['class'] == class_2]['neuron_name'].tolist()
            external_distances = combined_matrix.loc[indices_1, indices_2]
            
            # Handle the case where external_distances might be empty
            if not external_distances.empty:
                min_external_distances.append(external_distances.min().min())
        
        # external_dist = min(min_external_distances) if min_external_distances else float('inf')
        external_dist = max(min_external_distances) if min_external_distances else float('inf')

        score = external_dist / internal_dist if internal_dist > 0 else 0
        
        if score > current_best_score:
            current_best_score = score
            current_best_weights = normalized_weights.copy()
            # print("New best score:", current_best_score, "with weights:", current_best_weights)
        
        # Update score and weights tracking
        score_tracking[iteration_count] = {
            'score': score,
            'weights': normalized_weights.copy(),
            'current_best_weights': current_best_weights.copy()
        }
        
        # Increment weights, check if all combinations are done
        new_weights = increment_counter(weights, start_weights, stop_weights, dx)
        if new_weights is None:  # All combinations have been explored
            break
        else:
            weights = new_weights

        iteration_count += 1

    # Convert score tracking dictionary to DataFrame
    score_data = []
    for iteration, data in score_tracking.items():
        row = {
            'iteration': iteration,
            'score': data['score'],
            **{f'weight_{k}': v for k, v in data['weights'].items()},
            **{f'best_weight_{k}': v for k, v in data['current_best_weights'].items()}
        }
        score_data.append(row)
    score_df = pd.DataFrame(score_data)
    
    combined_dist_matrix = combine_matrices_with_weights(distance_matrices, current_best_weights)
    
    

    # Save the combined distance matrix and the best weights
    # Update these paths according to your configuration
    np.save(os.path.join(config.PICKLE_DIRECTORY, 'combined_dist_matrix.npy'), combined_dist_matrix)
    # np.save(os.path.join(config.PICKLE_DIRECTORY, 'best_weights.npy'), current_best_weights)
    # np.save(os.path.join(config.PICKLE_DIRECTORY, 'score_df.npy'), score_df)
    
    # Using to_pickle to save DataFrame
    score_df.to_pickle(os.path.join(config.PICKLE_DIRECTORY, 'score_df.pkl'))
    combined_dist_matrix.to_pickle(os.path.join(config.PICKLE_DIRECTORY, 'combined_dist_matrix.pkl'))
    pickle_file_path = os.path.join(config.PICKLE_DIRECTORY, 'best_weights.pkl')

    # Saving the dictionary
    with open(pickle_file_path, 'wb') as file:
        pickle.dump(current_best_weights, file)
        
    score_df.to_csv(os.path.join(config.PLOTS_DIRECTORY, 'score_df.csv'), index=False)
    # Convert dict to single-row DataFrame
    best_weights_df = pd.DataFrame([current_best_weights])
    
    # Save to CSV
    best_weights_df.to_csv(os.path.join(config.PLOTS_DIRECTORY, 'best_weights.csv'), index=False)
    
    print("Running permutation test...")
    permutation_scores, p_value = run_permutation_test(
        distance_matrices=distance_matrices,
        neuron_class_df=neuron_class_df,
        best_weights=current_best_weights,
        true_score=current_best_score,
        n_permutations=100
    )
    
    # Determine interpretation
    if p_value <= 0.01:
        interpretation = "Very strong evidence against randomness"
    elif p_value <= 0.05:
        interpretation = "Strong evidence"
    elif p_value <= 0.1:
        interpretation = "Weak evidence"
    else:
        interpretation = "No evidence (likely random)"
    
    # Print results
    print(f"Permutation Test Completed:")
    print(f"P-Value: {p_value:.6f}")
    print(f"Interpretation: {interpretation}")
    
    # Save summary to CSV
    summary_df = pd.DataFrame([{
        'p_value': round(p_value, 6),
        'interpretation': interpretation
    }])
    summary_path = os.path.join(config.PLOTS_DIRECTORY, 'permutation_summary_for_the_combination_of_distances.csv')
    summary_df.to_csv(summary_path, index=False)
    
    return combined_dist_matrix, current_best_weights, score_df

import matplotlib.pyplot as plt

def run_permutation_test(distance_matrices, neuron_class_df, best_weights, true_score, n_permutations=100, random_state=42):
    np.random.seed(random_state)
    permutation_scores = []

    for _ in range(n_permutations):
        shuffled_df = neuron_class_df.copy()
        shuffled_df['class'] = np.random.permutation(neuron_class_df['class'])

        combined_matrix = combine_matrices_with_weights(distance_matrices, best_weights)

        # Compute internal distances
        internal_distances = []
        for cls in shuffled_df['class'].unique():
            neuron_names = shuffled_df[shuffled_df['class'] == cls]['neuron_name']
            internal = combined_matrix.loc[neuron_names, neuron_names]
            internal_distances.append(np.nanmax(internal.values))
        internal_sum = sum(internal_distances)

        # Compute external distances
        external_distances = []
        for class_1, class_2 in combinations(shuffled_df['class'].unique(), 2):
            group_1 = shuffled_df[shuffled_df['class'] == class_1]['neuron_name']
            group_2 = shuffled_df[shuffled_df['class'] == class_2]['neuron_name']
            between = combined_matrix.loc[group_1, group_2]
            if not between.empty:
                external_distances.append(between.min().min())
        external = max(external_distances) if external_distances else float('inf')

        permuted_score = external / internal_sum if internal_sum > 0 else 0
        permutation_scores.append(permuted_score)

    # Compute p-value
    count = sum(s >= true_score for s in permutation_scores)
    p_value = (count + 1) / (n_permutations + 1)

    # Save results
    df = pd.DataFrame({'permuted_score': permutation_scores})
    df.to_csv(os.path.join(config.PICKLE_DIRECTORY, 'permutation_scores.csv'), index=False)
    with open(os.path.join(config.PICKLE_DIRECTORY, 'permutation_p_value.txt'), 'w') as f:
        f.write(f"Original Score: {true_score:.6f}\n")
        f.write(f"P-Value: {p_value:.6f}\n")

    # Plot
    plt.figure(figsize=(8, 6))
    plt.hist(permutation_scores, bins=30, alpha=0.75, label='Permutation Scores')
    plt.axvline(true_score, color='red', linestyle='dashed', linewidth=2, label=f'True Score = {true_score:.4f}')
    plt.xlabel('Score')
    plt.ylabel('Frequency')
    plt.title(f'Permutation Test (p = {p_value:.4f})')
    plt.legend()
    plot_path = os.path.join(config.PLOTS_DIRECTORY, 'permutation_test_plot_for_the_combination_of_distances.png')
    plt.savefig(plot_path, dpi=300)
    plt.close()
    
    
    return permutation_scores, p_value

# ==================================================================

def combine_matrices_with_weights(distance_matrices, weights):
    template_matrix = next(iter(distance_matrices.values()))
    combined_dist_mat = pd.DataFrame(
        np.zeros_like(template_matrix), index=template_matrix.index, columns=template_matrix.columns
    )
    for descriptor in distance_matrices:
        combined_dist_mat += weights[descriptor] * distance_matrices[descriptor]
    return combined_dist_mat

# def combine_matrices_with_weights(distance_matrices, weights):
#     """Combine distance matrices using weighted sum, with internal normalization.
    
#     Args:
#         distance_matrices (dict): Dictionary of {descriptor: distance_matrix}.
#         weights (dict): Dictionary of {descriptor: weight}. Automatically normalized.
    
#     Returns:
#         pd.DataFrame: Combined matrix with same indices/columns as input.
#     """
#     # Normalize weights to sum to 1.0
#     norm_factor = sum(weights.values())
#     normalized_weights = {k: v / norm_factor for k, v in weights.items()}
    
#     # Initialize output matrix with correct shape/index/columns
#     template_matrix = next(iter(distance_matrices.values()))
#     combined_dist_mat = pd.DataFrame(
#         np.zeros_like(template_matrix),
#         index=template_matrix.index,
#         columns=template_matrix.columns
#     )
    
#     # Apply weighted sum
#     for descriptor in distance_matrices:
#         combined_dist_mat += normalized_weights[descriptor] * distance_matrices[descriptor]
    
#     return combined_dist_mat


def combine_matrices_max_component(distance_matrices, weights):
    matrices = []
    for descriptor in distance_matrices:
        matrices.append(weights[descriptor] * distance_matrices[descriptor])
    stacked = np.stack([m.to_numpy() for m in matrices])
    combined_array = np.max(stacked, axis=0)
    template = next(iter(distance_matrices.values()))
    return pd.DataFrame(combined_array, index=template.index, columns=template.columns)


def combine_matrices_min_component(distance_matrices, weights):
    matrices = []
    for descriptor in distance_matrices:
        matrices.append(weights[descriptor] * distance_matrices[descriptor])
    stacked = np.stack([m.to_numpy() for m in matrices])
    combined_array = np.min(stacked, axis=0)
    template = next(iter(distance_matrices.values()))
    return pd.DataFrame(combined_array, index=template.index, columns=template.columns)


def combine_matrices_nonlinear_power_mean(distance_matrices, weights, p=0.5):
    matrices = []
    for descriptor in distance_matrices:
        weighted = np.power(distance_matrices[descriptor].to_numpy(), p) * weights[descriptor]
        matrices.append(weighted)
    summed = np.sum(matrices, axis=0)
    combined_array = np.power(summed, 1 / p)
    template = next(iter(distance_matrices.values()))
    return pd.DataFrame(combined_array, index=template.index, columns=template.columns)
# 


# ===========================================================================
# Example call to your main function goes here, assuming `distance_matrices` and `neuron_class_df` are defined.


# import numpy as np
# import pandas as pd
# from itertools import combinations
# import os
# from scipy.cluster.hierarchy import dendrogram, linkage
# import matplotlib.pyplot as plt

# def combine_matrices_with_weights(weights, dist_matrices):
#     combined_dist_mat_with_weights = np.zeros_like(next(iter(dist_matrices.values())))
#     for descriptor, weight in zip(dist_matrices, weights):
#         combined_dist_mat_with_weights += weight * dist_matrices[descriptor]
        
#     return combined_dist_mat_with_weights


# def combine_and_optimize_distance_matrices(distance_matrices, neuron_class_df, dx=0.3, max_iterations=1000):
#     num_matrices = len(distance_matrices)
#     start_weights = np.full(num_matrices, 1.0)  # Initialize start weights as floats
#     stop_weights = np.full(num_matrices, 3.0)  # Initialize stop weights as floats
    
#     current = np.array(start_weights, dtype=float)  # Ensure current is a float array
#     current_best_score = 0  # Initialize best score
#     current_best_weights = np.array(start_weights, dtype=float)  # Copy as a float array
#     iteration_count = 0
#     classes = neuron_class_df['class'].unique()
    
#     while iteration_count < max_iterations:
#         print(f'Iteration {iteration_count}  out of {max_iterations} max iterations')
#         combined_matrix = np.zeros_like(next(iter(distance_matrices.values())))
#         for i, descriptor in enumerate(distance_matrices):
#             combined_matrix += current[i] * distance_matrices[descriptor]
        
#         max_distance_within_class = []
        
#         for current_class in classes:
#             # Get indices for the current class
#             neuron_names = neuron_class_df[neuron_class_df['class'] == current_class]['neuron_name']
#             # Extract the submatrix for these indices from the combined_matrix
#             internal_distances = combined_matrix.loc[neuron_names, neuron_names]
#             # Calculate the maximum internal distance within this class
#             max_distance = np.nanmax(internal_distances.values)
#             max_distance_within_class.append(max_distance)
#             print(f"Maximum internal distance for class {current_class}: {max_distance}")
        
#         # Sum all maximum distances
#         internal_dist = sum(max_distance_within_class)
           
        
#         min_external_distances = []
        
#         for class_1, class_2 in combinations(classes, 2):
#             # Get neuron names for each class
#             indices_1 = neuron_class_df[neuron_class_df['class'] == class_1]['neuron_name'].tolist()
#             indices_2 = neuron_class_df[neuron_class_df['class'] == class_2]['neuron_name'].tolist()
        
#             # Extract the distances between the two sets of indices from the combined matrix
#             if indices_1 and indices_2:  # Ensure there are indices to prevent errors
#                 external_distances = combined_matrix.loc[indices_1, indices_2]
#                 # Append the minimum of the minimums of the distances matrix
#                 min_external_distances.append(external_distances.min().min())
        
#         # Calculate the minimum external distance, default to float('inf') if no distances were calculated
#         external_dist = min(min_external_distances) if min_external_distances else float('inf')
        
        
#         score = external_dist / internal_dist if internal_dist > 0 else 0
        
#         if score > current_best_score:
#             current_best_score = score
#             current_best_weights = current.copy()
#             print("New best score:", current_best_score, "with weights:", current_best_weights)
        
#         new_counter = increment_counter(current, start_weights, stop_weights, dx)
#         if np.array_equal(current, new_counter):
#             break
#         current = new_counter
#         iteration_count += 1
    
#     combined_dist_matrix = combine_matrices_with_weights(current_best_weights, distance_matrices)

#     np.save(os.path.join(config.PICKLE_DIRECTORY, 'combined_dist_matrix.npy'), combined_matrix)
#     np.save(os.path.join(config.PICKLE_DIRECTORY, 'best_weights.npy'), current_best_weights)
    
#     return combined_dist_matrix,  current_best_weights


# def increment_counter(current, start, stop, dx):
#     for i in range(len(current)):
#         if current[i] + dx <= stop[i]:
#             current[i] += dx
#             return current  # Return immediately once the first incrementable weight is incremented
#         else:
#             current[i] = start[i]  # Reset this weight and try to increment the next weight
#     return current  # Return the reset to start if all weights hit the maximum

# =======================

# def increment_counter(current, start, stop, dx):
#     position_to_increment = 0  # Start checking from the first element
#     new_counter = current.copy()  # Make a copy to avoid modifying the original array

#     # Find the first position where the current value is less than its corresponding stop value
#     while position_to_increment != len(start) and new_counter[position_to_increment] >= stop[position_to_increment]:
#         position_to_increment += 1

#     # If a valid position is found that hasn't reached the stop value
#     if position_to_increment != len(start):
#         new_counter[position_to_increment] += dx
#         # Reset all positions before this incremented position to their start values
#         for kk in range(position_to_increment):
#             new_counter[kk] = start[kk]

#     return new_counter





# ====================================================================
def combine_distance_matrices_with_optimal_weights(distance_matrices, n_clusters):
    """
    Combines multiple precomputed distance matrices with optimized weights to achieve the best clustering result.
    
    Parameters:
    - distance_matrices (dict): Dictionary where keys are descriptor names and values are square distance matrices.
    - n_clusters (int): Number of clusters for clustering and silhouette score calculation.
    
    Returns:
    - combined_matrix (np.ndarray): Combined distance matrix with optimal weights.
    - optimal_weights (np.ndarray): Array of optimal weights for each distance matrix.
    """
    
    # Define the objective function to minimize
    def objective(weights):
        weights = np.array(weights) / np.sum(weights)  # Normalize weights to sum to 1
        
        # Combine the distance matrices with the current weights
        combined_matrix = np.zeros_like(next(iter(distance_matrices.values())))
        for (descriptor_name, distance_matrix), weight in zip(distance_matrices.items(), weights):
            combined_matrix += weight * distance_matrix
        
        # Perform clustering on the combined matrix
        kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
        labels = kmeans.fit_predict(combined_matrix)
        
        # Calculate silhouette score based on the combined matrix
        silhouette_avg = silhouette_score(combined_matrix, labels, metric="precomputed")
        
        # Return negative silhouette score for minimization
        return -silhouette_avg

    # Define bounds for weights (0 to 1 for each distance matrix)
    bounds = [(0, 2) for _ in range(len(distance_matrices))]

    # Run differential evolution to find optimal weights
    result = differential_evolution(objective, bounds, strategy='best1bin', tol=0.01)
    optimal_weights = result.x / np.sum(result.x)  # Normalize weights to sum to 1

    # Combine matrices using the optimal weights
    template_matrix = next(iter(distance_matrices.values()))
    combined_matrix = pd.DataFrame(
        np.zeros_like(template_matrix),
        index=template_matrix.index,
        columns=template_matrix.columns,
    )
    for (descriptor_name, distance_matrix), weight in zip(distance_matrices.items(), optimal_weights):
        combined_matrix += weight * distance_matrix
        
    combined_matrix.to_pickle(os.path.join(config.PICKLE_DIRECTORY, 'combined_dist_matrix_with_optimal_weights.pkl'))
    pickle_file_path = os.path.join(config.PICKLE_DIRECTORY, 'optimal_weights.pkl')

    # Saving the dictionary
    with open(pickle_file_path, 'wb') as file:
        pickle.dump(optimal_weights, file)
        
    # Convert weights to a DataFrame with descriptor names
    optimal_weights_df = pd.DataFrame({
        'descriptor': list(distance_matrices.keys()),
        'weight': optimal_weights
    })
    
    # Save to CSV
    optimal_weights_df.to_csv(os.path.join(config.PICKLE_DIRECTORY, 'optimal_weights.csv'), index=False)


    return combined_matrix, optimal_weights


def combine_distance_matrices_entropy(distance_matrices):
    entropies = []
    for matrix in distance_matrices.values():
        # Calculate entropy for each row of the distance matrix and average
        entropy = -np.sum(matrix * np.log2(matrix + 1e-8), axis=1).mean()
        entropies.append(entropy)

    entropies = np.array(entropies)
    weights = entropies / np.sum(entropies)  # Normalize weights

    # Combine matrices using entropy-based weights
    template_matrix = next(iter(distance_matrices.values()))
    combined_matrix = pd.DataFrame(
        np.zeros_like(template_matrix),
        index=template_matrix.index,
        columns=template_matrix.columns,
    )
    for (descriptor_name, distance_matrix), weight in zip(distance_matrices.items(), weights):
        combined_matrix += weight * distance_matrix
        
    combined_matrix.to_pickle(os.path.join(config.PICKLE_DIRECTORY, 'combined_dist_matrix_entropy.pkl'))
    pickle_file_path = os.path.join(config.PICKLE_DIRECTORY, 'optimal_weights_entropy.pkl')

    # Saving the dictionary
    with open(pickle_file_path, 'wb') as file:
        pickle.dump(weights, file)


    return combined_matrix, weights
# =============================================================================================



def extract_statistical_features(all_descriptors):
    """
    Extract statistical features from each neuron's step function for each descriptor.

    Parameters:
    - all_descriptors: Dictionary where each key is a descriptor and each value is a dictionary
                       with neuron names as keys and their respective step function lists of tuples as values.

    Returns:
    - features_df: A DataFrame where each row represents a neuron, and columns represent extracted features
                   for each descriptor.
    """

    data = []
    neuron_names = []

    # Remove excluded descriptors
    all_descriptors = {k: v for k, v in all_descriptors.items() if k not in exclude_descriptors}

    for neuron_name in set(neuron for descriptor_data in all_descriptors.values() for neuron in descriptor_data):
        neuron_features = []

        for descriptor_name, descriptor_data in all_descriptors.items():
            if descriptor_name in exclude_descriptors:
                continue
            
            step_function = descriptor_data[neuron_name]
            _, vector = zip(*step_function)
            vector = np.array(vector)

            # Basic Statistics
            mean_val = np.mean(vector)
            std_val = np.std(vector)
            median_val = np.median(vector)
            min_val = np.min(vector)
            max_val = np.max(vector)
            var_val = np.var(vector)
            iqr_val = np.percentile(vector, 75) - np.percentile(vector, 25)

            # Step Function Features
            num_steps = len(np.unique(vector))
            diff_values = np.diff(vector)
            max_step_size = np.max(diff_values) if len(diff_values) > 0 else 0
            max_abs_step_size = np.max(np.abs(diff_values)) if len(diff_values) > 0 else 0

            # Integral & Cumulative Sum
            area_under_curve = trapz(vector)
            cumulative_sum = np.sum(vector)

            # Distribution Metrics
            skewness_val = skew(vector)
            kurtosis_val = kurtosis(vector, fisher=True)
            entropy_val = entropy(vector / np.sum(vector)) if np.sum(vector) > 0 else 0

            # Mode
            # mode_result = mode(vector)
            # mode_val = mode_result.mode[0] if mode_result.count.size > 0 else 0
            # Pass keepdims=True to get the old behavior
            mode_result = mode(vector, keepdims=True)
            mode_val = mode_result.mode[0] if mode_result.count.size > 0 else 0

            # Correlation
            try:
                auto_corr = np.correlate(vector, vector, mode='full')
                correlation_coefficient = np.corrcoef(auto_corr[:-1], auto_corr[1:])[0, 1]
            except:
                correlation_coefficient = 0

            # Fourier Transform Features
            frequencies = np.fft.fftfreq(len(vector))
            fft_values = fft(vector)
            amplitude_spectrum = np.abs(fft_values)
            max_fft_amp = np.max(amplitude_spectrum)
            dominant_freq = frequencies[np.argmax(amplitude_spectrum)]
            weighted_freq = np.sum(frequencies * amplitude_spectrum) / np.sum(amplitude_spectrum)
            fft_entropy = entropy(amplitude_spectrum / np.sum(amplitude_spectrum))

            # Harmonic Mean (Only for positive values)
            try:
                harmonic_mean = hmean(vector[vector > 0])
            except:
                harmonic_mean = 0

            # Feature Vector
            neuron_features.extend([
                mean_val, median_val, std_val, var_val, min_val, max_val, iqr_val,
                num_steps, max_step_size, max_abs_step_size, area_under_curve, cumulative_sum,
                skewness_val, kurtosis_val, entropy_val, mode_val, correlation_coefficient,
                max_fft_amp, dominant_freq, weighted_freq, fft_entropy, harmonic_mean
            ])

        data.append(neuron_features)
        neuron_names.append(neuron_name)

    # Feature Names
    feature_names = [
        "mean", "median", "std_dev", "variance", "min", "max", "iqr",
        "num_steps", "max_step_size", "max_abs_step_size", "area_under_curve", "cumulative_sum",
        "skewness", "kurtosis", "entropy", "mode", "correlation",
        "max_fft_amp", "dominant_freq", "weighted_freq", "fft_entropy", "harmonic_mean"
    ]

    descriptors = list(all_descriptors.keys())
    feature_columns = [f"{feat}_{descriptor}" for descriptor in descriptors for feat in feature_names]

    # Create DataFrame
    features_df = pd.DataFrame(data, index=neuron_names, columns=feature_columns)

    return features_df




# def extract_statistical_features(all_descriptors):
#     """
#     Extract statistical features from each neuron's step function for each descriptor.
    
#     Parameters:
#     - all_descriptors: Dictionary where each key is a descriptor and each value is a dictionary
#                        with neuron names as keys and their respective step function lists of tuples as values.
    
#     Returns:
#     - features_df: A DataFrame where each row represents a neuron, and columns represent extracted features
#                    for each descriptor.
#     """
#     # List to store the final data for DataFrame construction
#     data = []
#     neuron_names = []
    
#     # # Make a copy of all_descriptors and remove the excluded descriptors
#     # all_descriptors = all_descriptors.copy()
#     # for descriptor in all_descriptors:
#     #     all_descriptors.pop(descriptor, None)
        
#         # Create a filtered dictionary by excluding the descriptors in `exclude_descriptors`
#     all_descriptors = {k: v for k, v in all_descriptors.items() if k not in exclude_descriptors}

    
#     # Iterate over each neuron
#     for neuron_name in set(neuron for descriptor_data in all_descriptors.values() for neuron in descriptor_data):
#         neuron_features = []
        
#         # Process each descriptor for the current neuron
#         for descriptor_name, descriptor_data in all_descriptors.items():
#             # Check if the descriptor is in the exclusion list
#             if descriptor_name in exclude_descriptors:
#                 print(f'Skipping descriptor {descriptor_name} as it is in the exclusion list.')
#                 continue
            
#             step_function = descriptor_data[neuron_name]
#             _, vector = zip(*step_function)
#             vector = np.array(vector)

#             # Feature extraction
#             f1 = np.mean(vector)
#             f2 = np.median(vector)
#             f3 = np.std(vector)
#             f4 = np.var(vector)
#             f5 = np.max(vector)
#             f6 = np.min(vector)
#             f7 = trapz(vector)
#             f8 = np.cumsum(vector).sum()
#             f9 = skew(vector)
#             f10 = kurtosis(vector, fisher=True)
#             auto_corr = np.correlate(vector, vector, mode='full')
#             try:
#                 f11 = pearsonr(auto_corr[:-1], auto_corr[1:])[0]
#             except:
#                 f11 = -2
#             f12 = f5 - f6
#             q1 = np.percentile(vector, 25)
#             q3 = np.percentile(vector, 75)
#             f13 = q3 - q1
#             f14 = (f3 / f1) * 100 if f1 != 0 else 0
#             correlation_matrix = np.corrcoef(vector, vector)
#             f15 = correlation_matrix[0, 1]

            
#             # 16. Mode (most frequent value in the vector)
#             vector_rounded = np.round(vector, decimals=3)

#             # Calculate the mode
#             mode_result = mode(vector_rounded)
#             # Check if mode_result.mode is a scalar or array
#             if isinstance(mode_result.mode, np.ndarray):
#                 f16 = mode_result.mode[0] if mode_result.mode.size > 0 else None
#             else:
#                 f16 = mode_result.mode  # If it's a scalar, use it directly
            
            
#             try:
#                 hist, _ = np.histogram(vector_rounded, bins='auto')
#                 f17 = np.max(hist)
#             except:
#                 f17 = -1
#             f18 = np.max(np.diff(vector))
#             f19 = np.max(np.abs(np.diff(vector)))
#             f20 = entropy(vector)
#             frequencies = np.fft.fftfreq(len(vector))
#             fft_values = np.fft.fft(vector)
#             amplitude_spectrum = np.abs(fft_values)
#             f21 = np.max(amplitude_spectrum)
#             dominant_frequency_index = np.argmax(amplitude_spectrum)
#             f22 = frequencies[dominant_frequency_index]
#             f23 = np.sum(vector ** 2)
#             f24 = np.sqrt(np.mean(vector ** 2))
#             try:
#                 f25 = (vector[-1] - vector[0]) / len(vector)
#             except:
#                 f25 = 0
#             f26 = np.ptp(vector)
#             f27 = entropy(np.histogram(vector, bins='auto')[0])
#             try:
#                 f28 = hmean(vector[vector > 0])
#             except:
#                 f28 = 0
#             weighted_freq = np.sum(frequencies * amplitude_spectrum) / np.sum(amplitude_spectrum)
#             f29 = weighted_freq
#             f30 = entropy(amplitude_spectrum / np.sum(amplitude_spectrum))
            
#             # Add all features for this descriptor, prefixing with "f{i}_{descriptor_name}"
#             neuron_features.extend([
#                 f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15, f16, f17, f18,
#                 f19, f20, f21, f22, f23, f24, f25, f26, f27, f28, f29, f30
#             ])

#         # Store the features for the neuron
#         data.append(neuron_features)
#         neuron_names.append(neuron_name)

#     # Generate column names
#     descriptors = list(all_descriptors.keys())
#     feature_columns = [f"f{i+1}_{descriptor}" for descriptor in descriptors for i in range(30)]

#     # Create DataFrame
#     features_df = pd.DataFrame(data, index=neuron_names, columns=feature_columns)
#     # features_df.index.name = "Neuron"

#     # # Reset index to add neuron names as a column
#     # features_df.reset_index(inplace=True)

#     return features_df






# def extract_statistical_features(all_descriptors):
#     """
#     Extract statistical features from each neuron's step function for each descriptor.
    
#     Parameters:
#     - all_descriptors: Dictionary where each key is a descriptor and each value is a dictionary
#                        with neuron names as keys and their respective step function lists of tuples as values.
#     - normalization_method: Method to normalize the vector values, options are "min-max", "z-score", or "none".
    
#     Returns:
#     - features_df: A DataFrame where each row represents a neuron, and columns represent extracted features
#                    for each descriptor.
#     """
#     # Dictionary to store features for each neuron
#     feature_data = {}
    
#     feature_normalization_method = config.FEATURE_NORMALIZATION_METHOD
    
#     # Iterate over each descriptor
#     for descriptor_name, descriptor_data in all_descriptors.items():
        
#         print(f'Extracting features for descriptor: {descriptor_name}')
        
#         # Skip the descriptor if it is in the exclusion list
#         if descriptor_name in exclude_descriptors:
#            print(f'Skipping descriptor {descriptor_name} as it is in the exclusion list.')
#            continue
        
#         # Iterate over each neuron and its step function data
#         for neuron_name, step_function in descriptor_data.items():
            
#             # Extract the y-values from the step function
#             _, vector = zip(*step_function)
#             vector = np.array(vector)

#             # # Apply normalization based on the specified method
#             # if feature_normalization_method == "min-max":
#             #     vector_normalized = (vector - np.min(vector)) / (np.max(vector) - np.min(vector)) if np.max(vector) != np.min(vector) else vector
#             # elif feature_normalization_method == "z-score":
#             #     vector_normalized = (vector - np.mean(vector)) / np.std(vector) if np.std(vector) != 0 else vector
#             # else:
#             #     vector_normalized = vector  # No normalization

#             # Original Features
#             # 1. Mean
#             f1 = np.mean(vector)
#             # 2. Median
#             f2 = np.median(vector)
#             # 3. Standard Deviation
#             f3 = np.std(vector)
#             # 4. Variance
#             f4 = np.var(vector)
#             # 5. Maximum value
#             f5 = np.max(vector)
#             # 6. Minimum value
#             f6 = np.min(vector)
#             # 7. Area under the curve using trapezoidal integration
#             f7 = trapz(vector)
#             # 8. Sum of the cumulative sum of values
#             f8 = np.cumsum(vector).sum()
#             # 9. Skewness (asymmetry of the distribution)
#             f9 = skew(vector)
#             # 10. Kurtosis (tailedness of the distribution)
#             f10 = kurtosis(vector, fisher=True)
            
#             # Autocorrelation
#             # 11. Lag-1 Autocorrelation
#             auto_corr = np.correlate(vector, vector, mode='full')
#             try:
#                 f11 = pearsonr(auto_corr[:-1], auto_corr[1:])[0]
#             except:
#                 f11 = -2
            
#             # 12. Range (difference between max and min)
#             f12 = f5 - f6
#             # 13. Interquartile Range (spread between the 25th and 75th percentiles)
#             q1 = np.percentile(vector, 25)
#             q3 = np.percentile(vector, 75)
#             f13 = q3 - q1
#             # 14. Coefficient of Variation (CV) as a percentage
#             f14 = (f3 / f1) * 100 if f1 != 0 else 0
            
#             # 15. Self-Correlation Coefficient
#             correlation_matrix = np.corrcoef(vector, vector)
#             f15 = correlation_matrix[0, 1]
            
#             # 16. Mode (most frequent value in the vector)
#             vector_rounded = np.round(vector, decimals=3)

#             # Calculate the mode
#             mode_result = mode(vector_rounded)
#             # Check if mode_result.mode is a scalar or array
#             if isinstance(mode_result.mode, np.ndarray):
#                 f16 = mode_result.mode[0] if mode_result.mode.size > 0 else None
#             else:
#                 f16 = mode_result.mode  # If it's a scalar, use it directly
            
#             # 17. Maximum Histogram Bin Count
#             try:
#                 hist, _ = np.histogram(vector_rounded, bins='auto')
#                 f17 = np.max(hist)
#             except:
#                 f17 = -1
            
#             # 18. Maximum Difference Between Consecutive Values
#             f18 = np.max(np.diff(vector))
#             # 19. Maximum Absolute Difference Between Consecutive Values
#             f19 = np.max(np.abs(np.diff(vector)))
#             # 20. Entropy (distribution complexity)
#             f20 = entropy(vector)
            
#             # Fourier transform analysis
#             # 21. Maximum Amplitude in the Frequency Spectrum
#             frequencies = np.fft.fftfreq(len(vector))
#             fft_values = np.fft.fft(vector)
#             amplitude_spectrum = np.abs(fft_values)
#             f21 = np.max(amplitude_spectrum)
#             # 22. Dominant Frequency
#             dominant_frequency_index = np.argmax(amplitude_spectrum)
#             f22 = frequencies[dominant_frequency_index]
            
#             # Additional Suggested Features
#             # 23. Energy (sum of squares of values)
#             f23 = np.sum(vector ** 2)
#             # 24. Root Mean Square (RMS) - average amplitude
#             f24 = np.sqrt(np.mean(vector ** 2))
#             # 25. Slope over the function (linear trend)
#             try:
#                 f25 = (vector[-1] - vector[0]) / len(vector)
#             except:
#                 f25 = 0
#             # 26. Peak-to-Peak Distance (difference between highest and lowest points)
#             f26 = np.ptp(vector)
#             # 27. Approximate Entropy (histogram-based entropy measure of complexity)
#             f27 = entropy(np.histogram(vector, bins='auto')[0])
#             # 28. Harmonic Mean (for positive values only)
#             try:
#                 f28 = hmean(vector[vector > 0])
#             except:
#                 f28 = 0
#             # 29. Frequency Centroid (weighted average frequency)
#             weighted_freq = np.sum(frequencies * amplitude_spectrum) / np.sum(amplitude_spectrum)
#             f29 = weighted_freq
#             # 30. Spectral Entropy (entropy of frequency spectrum distribution)
#             f30 = entropy(amplitude_spectrum / np.sum(amplitude_spectrum))
            
#             # Store all features for the current neuron and descriptor
#             feature_data[(neuron_name, descriptor_name)] = [
#                 f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15, 
#                 f16, f17, f18, f19, f20, f21, f22, f23, f24, f25, f26, f27, f28, f29, f30
#             ]
    
#     # Convert the feature data dictionary to a DataFrame
#     feature_columns = [f"{descriptor_name}_f{i+1}" for descriptor_name, _ in feature_data.keys() for i in range(30)]
#     features_df = pd.DataFrame.from_dict(feature_data, orient='index', columns=feature_columns)
    
#     # Convert multi-index to single index for neuron identification
#     features_df.index = pd.MultiIndex.from_tuples(features_df.index, names=["Neuron", "Descriptor"])
    
#     return features_df




def write_descriptor_distance_dependence_report(distance_matrices, output_directory=None):
    """Quantify redundancy among descriptor pairwise-distance matrices.

    Each descriptor matrix is represented by its strict upper triangle, so
    every descriptor is compared over the same unordered neuron pairs. This
    is a diagnostic for descriptor dependence; it does not alter the primary
    distance combination or optimization procedure.
    """
    import json
    from pathlib import Path
    from scipy.stats import pearsonr, spearmanr

    if not distance_matrices:
        raise ValueError("At least one distance matrix is required.")

    descriptors = list(distance_matrices)
    template = distance_matrices[descriptors[0]]
    n = len(template.index)
    upper_indices = np.triu_indices(n, k=1)
    vectors = {}
    for descriptor in descriptors:
        matrix = distance_matrices[descriptor]
        if not matrix.index.equals(template.index) or not matrix.columns.equals(template.columns):
            raise ValueError(f"Distance-matrix labels do not match for {descriptor}.")
        vector = matrix.to_numpy(dtype=float)[upper_indices]
        if not np.isfinite(vector).all():
            raise ValueError(f"Distance matrix contains non-finite values: {descriptor}.")
        vectors[descriptor] = vector

    pearson = pd.DataFrame(np.eye(len(descriptors)), index=descriptors, columns=descriptors)
    pearson_p = pd.DataFrame(np.zeros((len(descriptors), len(descriptors))), index=descriptors, columns=descriptors)
    spearman = pearson.copy()
    spearman_p = pearson_p.copy()
    pair_rows = []

    for i, descriptor_a in enumerate(descriptors):
        for j in range(i + 1, len(descriptors)):
            descriptor_b = descriptors[j]
            pearson_r, pearson_p_value = pearsonr(vectors[descriptor_a], vectors[descriptor_b])
            spearman_r, spearman_p_value = spearmanr(vectors[descriptor_a], vectors[descriptor_b])
            pearson.loc[descriptor_a, descriptor_b] = pearson.loc[descriptor_b, descriptor_a] = pearson_r
            pearson_p.loc[descriptor_a, descriptor_b] = pearson_p.loc[descriptor_b, descriptor_a] = pearson_p_value
            spearman.loc[descriptor_a, descriptor_b] = spearman.loc[descriptor_b, descriptor_a] = spearman_r
            spearman_p.loc[descriptor_a, descriptor_b] = spearman_p.loc[descriptor_b, descriptor_a] = spearman_p_value
            pair_rows.append({
                "descriptor_a": descriptor_a,
                "descriptor_b": descriptor_b,
                "n_neuron_pairs": int(len(vectors[descriptor_a])),
                "pearson_r": float(pearson_r),
                "pearson_p": float(pearson_p_value),
                "spearman_rho": float(spearman_r),
                "spearman_p": float(spearman_p_value),
            })

    eigenvalues = np.linalg.eigvalsh(pearson.to_numpy(dtype=float))
    eigenvalues = np.clip(eigenvalues, 0.0, None)
    effective_count = float(eigenvalues.sum() ** 2 / np.square(eigenvalues).sum())
    output = Path(output_directory or config.PLOTS_DIRECTORY) / "descriptor_dependence"
    output.mkdir(parents=True, exist_ok=True)
    pearson.to_csv(output / "pearson_distance_correlation.csv")
    pearson_p.to_csv(output / "pearson_distance_correlation_pvalues.csv")
    spearman.to_csv(output / "spearman_distance_correlation.csv")
    spearman_p.to_csv(output / "spearman_distance_correlation_pvalues.csv")
    pd.DataFrame(pair_rows).to_csv(output / "descriptor_distance_correlation_pairs.csv", index=False)
    summary = {
        "descriptors": descriptors,
        "n_neurons": n,
        "n_unordered_neuron_pairs": int(len(upper_indices[0])),
        "effective_descriptor_count": effective_count,
        "interpretation": "Diagnostic only; primary joint distance weighting is unchanged.",
    }
    (output / "descriptor_dependence_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return {"pearson": pearson, "spearman": spearman, "pairs": pd.DataFrame(pair_rows), "summary": summary}
