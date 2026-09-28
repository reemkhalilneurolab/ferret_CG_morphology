"""Topological Sholl descriptors.

Each descriptor maps a neuron to a real-valued step function on [0, 1], the
normalised radial distance from the soma, so that neurons can be compared by the
L1 distance between their descriptor functions.

Reference implementation follows:

    Khalil R, Kallel S, Farhat A, Dlotko P. Topological Sholl descriptors for
    neuronal clustering and classification. PLoS Comput Biol. 2022 Jun
    22;18(6):e1010229. doi:10.1371/journal.pcbi.1010229

Section numbers cited below refer to that paper's supplementary material.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import config
import networkx as nx
from numpy.linalg import norm
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.manifold import MDS
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
import seaborn as sns
from scipy.spatial import ConvexHull, QhullError

import warnings
warnings.filterwarnings('ignore')

exclude_descriptors = config.EXCLUDED_DESCRIPTORS

# SWC structure codes (soma = 1, axon = 2, basal dendrite = 3, apical = 4).
SOMA_TYPE = 1
BASAL_TYPE = 3
APICAL_TYPE = 4

def normalize_descriptor_radii(descriptor_values):
    """
    Normalize radius (first value of each tuple) to [0, 1] range
    if CONFIG.NORMALIZE_DESCRIPTOR_RADII is True.

    Parameters:
    - descriptor_values: List of tuples (radius, value)

    Returns:
    - List of tuples with normalized radii
    """


    max_radius = max((r for r, _ in descriptor_values), default=0)
    if max_radius == 0:
        normalized = [(0.0, v) for r, v in descriptor_values]
    else:
        normalized = [(r / max_radius, v) for r, v in descriptor_values]
    
    normalized.sort(key=lambda x: x[0])
    return normalized


def plot_step_function(descriptor_name, neuron_name, neuron_class, step_data):
    """
    Generalized plotting function for a single step function of different descriptors.
    Creates subdirectories matching 'descriptor_name + "_step_functions"'.

    Parameters:
    - descriptor_name: Name of the descriptor being plotted.
    - neuron_name: Name of the neuron.
    - step_data: List of tuples (distance, value) to plot, where distance is from the soma.
    """
    # Unpack the distances (x) and values (y)
    x, y = zip(*step_data)

    main_directory = os.path.join(config.DESCRIPTORS_DIRECTORY, "step_functions")
    plot_directory = os.path.join(main_directory, f"{descriptor_name}_step_functions")
    os.makedirs(plot_directory, exist_ok=True)

    # Create the plot
    plt.figure(figsize=(10, 8))
    plt.step(x, y, where='post', label=descriptor_name.capitalize(), color='blue', linestyle='-')
    plt.axhline(y=0, color='red', linestyle=':', linewidth=1)
    plt.xlabel('Distance from Soma (micrometers)')
    plt.ylabel(descriptor_name.capitalize())
    plt.title(f'{descriptor_name.capitalize()} for {neuron_name}')
    plt.legend()

    # Save the plot to the designated subdirectory
    save_path = os.path.join(plot_directory, f'{neuron_class}_{neuron_name}_{descriptor_name}.png')
    plt.savefig(save_path)
    
    plt.close()
    
def calculate_energy(neuron_data):
    """Energy descriptor, nodal distribution about the soma.

    Supplementary section 1.6.7 treats each node as a charged particle and takes
    the field intensity at the soma:

        phi(r) = |E_{N_r}(0, 0, 0)|

    Each node contributes a vector of length q_i directed along (P - P_i)/|PP_i|.
    Evaluated at the soma this reduces to the sum of unit vectors pointing to
    each node, and the bracket denotes the square intensity, i.e. the sum of the
    squared components without a square root. Nodes evenly distributed about the
    soma cancel and give a small value; nodes lying to one side reinforce and
    give a large one.

    The charge q_i is the dendritic thickness where available and 1 otherwise.
    The SWC radius column here holds tracing presets rather than measured
    thicknesses, so q_i = 1 throughout.

    The soma itself is excluded because its unit vector is undefined at zero
    distance.
    """
    # G_df = neuron_data['swc_df']
    G_df = neuron_data['swc_df'].copy()
    neuron_name = neuron_data['neuron_name']
    neuron_class = neuron_data['class']
    soma_position = np.array([0.0, 0.0, 0.0])  # Position of the soma at origin

    # Filter the DataFrame for bifurcation and termination nodes only, excluding the soma point
    # filtered_df = G_df[(G_df['node_type'].isin(['bifurcation', 'termination'])) & (G_df['euclidean_distance_to_soma'] > 0)]
    
    # Filter the DataFrame for bifurcation and termination nodes only, excluding the soma point
    filtered_df = G_df[(G_df['node_type'].isin(['bifurcation', 'termination'])) & (G_df['euclidean_distance_to_soma'] > 0)].reset_index(drop=True)

    # Extract positions and distances for filtered nodes
    positions = filtered_df[['X', 'Y', 'Z']].values
    distances_to_soma = filtered_df['euclidean_distance_to_soma'].values

    # Compute electric field vector components for each node, using distances
    E_components = np.divide(positions - soma_position, distances_to_soma[:, np.newaxis])

    # Define radii based on the euclidean distances of filtered nodes, sorted
    radii = np.sort(distances_to_soma)  # Correct use of np.unique()

    # Calculate energy values and store vectors for each radius
    energy_values = []
    energy_vectors = []

    for radius in radii:
        within_radius = distances_to_soma <= radius
        E_vector = np.sum(E_components[within_radius], axis=0)
        energy_value = np.sum(E_vector ** 2)
        
        energy_values.append((radius, energy_value))
        energy_vectors.append(E_vector)
    
    energy_values.insert(0, (0, 0))  # T(0) = 0 for the soma
    energy_vectors = np.array(energy_vectors)  
    
    if config.NORMALIZE_DESCRIPTOR_RADII:
        energy_values = normalize_descriptor_radii(energy_values)
        
        max_radius = max(r for r, _ in energy_values)
        if max_radius > 0:
            energy_vectors = energy_vectors / max_radius
        else:
            # fallback: set to zeros or leave unchanged depending on your needs
            energy_vectors = energy_vectors  # or: np.zeros_like(energy_vectors)


    
    # Plotting and additional outputs (assuming plotting functions and config are defined)
    if config.PLOT_STEP_FUNCTION:
        plot_step_function("energy", neuron_name, neuron_class, energy_values)

    return energy_values, energy_vectors


def calculate_polarity(neuron_data):
    
    _ , energy_vectors = calculate_energy(neuron_data)
    
    # G_df = neuron_data['swc_df']
    G_df = neuron_data['swc_df'].copy()
    neuron_name = neuron_data['neuron_name']
    soma_position = np.array([0.0, 0.0, 0.0])  # Position of the soma at origin

    # Filter the DataFrame for bifurcation and termination nodes only, excluding the soma point
    # filtered_df = G_df[(G_df['node_type'].isin(['bifurcation', 'termination'])) & (G_df['euclidean_distance_to_soma'] > 0)]
    
    # Filter the DataFrame for bifurcation and termination nodes only, excluding the soma point
    filtered_df = G_df[(G_df['node_type'].isin(['bifurcation', 'termination'])) & (G_df['euclidean_distance_to_soma'] > 0)].reset_index(drop=True)

    norms = np.linalg.norm(energy_vectors, axis=1)
    valid_norms = norms[:, np.newaxis] * norms
    cos_similarity = np.dot(energy_vectors, energy_vectors.T) / np.where(valid_norms == 0, 1e-10, valid_norms)
    cos_similarity = np.clip(cos_similarity, -1.0, 1.0)
    polarity_matrix = np.arccos(cos_similarity)
    polarity_matrix[np.isnan(polarity_matrix)] = 0
    
    # Access node IDs and create index map
    # node_ids = filtered_df.index.values
    node_ids = filtered_df["ID"].values
    node_index_map = pd.Series(data=range(len(node_ids)), index=node_ids).to_dict()   
    
    # # Create plane mapping using node indices
    # plane_mapping = {node_index_map[node_id]: 'upper plane' if z > 0 else 'lower plane' 
    #                  for node_id, z in zip(node_ids, filtered_df['Z'])}
    
    # Combine plane mapping and node index map for easy access to both label and color
    plane_mapping = {node_index_map[node_id]: ('upper plane' if z > 0 else 'lower plane', node_id)
                      for node_id, z in zip(node_ids, filtered_df['Z'])}
    
    # Plot the dendrogram if configured
    if config.PLOT_POLARITY:
        plot_dendrogram_from_polarity_matrix(polarity_matrix, plane_mapping, neuron_name)
        plot_pca_from_polarity_matrix(polarity_matrix, plane_mapping, neuron_name)
        
    return polarity_matrix
    


# def calculate_energy_and_polarity_matrix(neuron_data):
    
#     # G_df = neuron_data['swc_df']
#     G_df = neuron_data['swc_df'].copy()
#     neuron_name = neuron_data['neuron_name']
#     soma_position = np.array([0.0, 0.0, 0.0])  # Position of the soma at origin

#     # Filter the DataFrame for bifurcation and termination nodes only, excluding the soma point
#     # filtered_df = G_df[(G_df['node_type'].isin(['bifurcation', 'termination'])) & (G_df['euclidean_distance_to_soma'] > 0)]
    
#     # Filter the DataFrame for bifurcation and termination nodes only, excluding the soma point
#     filtered_df = G_df[(G_df['node_type'].isin(['bifurcation', 'termination'])) & (G_df['euclidean_distance_to_soma'] > 0)].reset_index(drop=True)

#     # Extract positions and distances for filtered nodes
#     positions = filtered_df[['X', 'Y', 'Z']].values
#     distances_to_soma = filtered_df['euclidean_distance_to_soma'].values

#     # Compute electric field vector components for each node, using distances
#     E_components = np.divide(positions - soma_position, distances_to_soma[:, np.newaxis])

#     # Define radii based on the euclidean distances of filtered nodes, sorted
#     radii = np.sort(distances_to_soma)  # Correct use of np.unique()

#     # Calculate energy values and store vectors for each radius
#     energy_values = []
#     energy_vectors = []

#     for radius in radii:
#         within_radius = distances_to_soma <= radius
#         E_vector = np.sum(E_components[within_radius], axis=0)
#         energy_value = np.sum(E_vector ** 2)
        
#         energy_values.append((radius, energy_value))
#         energy_vectors.append(E_vector)
    
#     energy_values.insert(0, (0, 0))  # T(0) = 0 for the soma
#     energy_vectors = np.array(energy_vectors)

#     # Calculate polarity matrix
#     norms = np.linalg.norm(energy_vectors, axis=1)
#     valid_norms = norms[:, np.newaxis] * norms
#     cos_similarity = np.dot(energy_vectors, energy_vectors.T) / np.where(valid_norms == 0, 1e-10, valid_norms)
#     cos_similarity = np.clip(cos_similarity, -1.0, 1.0)
#     polarity_matrix = np.arccos(cos_similarity)
#     polarity_matrix[np.isnan(polarity_matrix)] = 0

    
#     # Access node IDs and create index map
#     # node_ids = filtered_df.index.values
#     node_ids = filtered_df["ID"].values
#     node_index_map = pd.Series(data=range(len(node_ids)), index=node_ids).to_dict()   
    
#     # # Create plane mapping using node indices
#     # plane_mapping = {node_index_map[node_id]: 'upper plane' if z > 0 else 'lower plane' 
#     #                  for node_id, z in zip(node_ids, filtered_df['Z'])}
    
#     # Combine plane mapping and node index map for easy access to both label and color
#     plane_mapping = {node_index_map[node_id]: ('upper plane' if z > 0 else 'lower plane', node_id)
#                       for node_id, z in zip(node_ids, filtered_df['Z'])}
    
#     # Plotting and additional outputs (assuming plotting functions and config are defined)
#     if config.PLOT_STEP_FUNCTION:
#         plot_step_function("energy", neuron_name, energy_values)


#     # Plot the dendrogram if configured
#     if config.PLOT_POLARITY:
#         plot_dendrogram_from_polarity_matrix(polarity_matrix, plane_mapping, neuron_name)
#         plot_pca_from_polarity_matrix(polarity_matrix, plane_mapping, neuron_name)
        
#     return energy_values, polarity_matrix


def plot_dendrogram_from_polarity_matrix(polarity_matrix, plane_mapping, neuron_name):
    """
    Plot a dendrogram based on the polarity matrix with labels colored by their spatial plane.

    Parameters:
    - polarity_matrix: A symmetric matrix of angles between node vectors.
    - plane_mapping: Dictionary mapping matrix index to a tuple (plane, original node_id).
    - neuron_name: Name of the neuron for labeling the plot.
    """
    # Convert the polarity matrix to a condensed form for hierarchical clustering
    condensed_matrix = polarity_matrix[np.triu_indices(len(polarity_matrix), k=1)]
    
    # Perform hierarchical clustering
    linked = linkage(condensed_matrix, config.LINKAGE_METHOD)

    # Create figure for plotting
    plt.figure(figsize=(10, 6))

    # Map labels to original node IDs for the dendrogram
    real_labels = [plane_mapping[i][1] for i in range(len(plane_mapping))]  # Extract node_id from plane_mapping

    # Create a dendrogram
    dendro = dendrogram(linked, labels=real_labels, color_threshold=0)

    # Color labels based on plane mapping
    ax = plt.gca()
    xlbls = ax.get_xmajorticklabels()
    
    for lbl in xlbls:
        # Get the real node ID from the label text
        node_id = int(lbl.get_text())
    
        # Find the index of this node_id in real_labels
        label_index = real_labels.index(node_id)
    
        # Use label_index to access the plane from plane_mapping
        plane = plane_mapping[label_index][0]  # Get plane info from plane_mapping
    
        # Set color based on the plane
        lbl.set_color('red' if plane == 'upper plane' else 'blue')

    # Create the appropriate subdirectory for the descriptor using the neuron-specific path
    plot_directory = os.path.join(config.PLOTS_DIRECTORY, f"Polarity_Dendrograms")
    os.makedirs(plot_directory, exist_ok=True)

    save_path = os.path.join(config.DESCRIPTORS_DIRECTORY, f'{neuron_name}_Dendrogram_Polarity.png')
    plt.savefig(save_path)
    plt.close()




def plot_pca_from_polarity_matrix(distance_matrix, plane_mapping, neuron_name, n_components=2):
    # Perform Multidimensional Scaling (MDS)
    mds = MDS(n_components=n_components, dissimilarity='precomputed', random_state=42, normalized_stress=False)
    mds_transformed = mds.fit_transform(distance_matrix)

    # 
    # Perform PCA
    pca = PCA(n_components=2)
    pca_transformed = pca.fit_transform(mds_transformed)

    # Use real_labels based on the polarity matrix indices
    real_labels = [plane_mapping[idx][1] for idx in range(len(plane_mapping))]

    # Precompute colors based on the plane mapping
    colors = ['red' if plane_mapping[idx][0] == 'upper plane' else 'blue' for idx in range(len(plane_mapping))]

    # Plotting the PCA-transformed data
    plt.figure(figsize=(8, 6))
    for idx, (x, y) in enumerate(pca_transformed):
        plt.scatter(x, y, color=colors[idx], label=real_labels[idx] if idx == 0 or real_labels[idx] not in real_labels[:idx] else "")
        plt.text(x, y, str(real_labels[idx]), fontsize=9, ha='right')

    plt.title(f'PCA of MDS Transformed Data - {neuron_name}')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.grid(True)
    plt.legend(loc="best", title="Node ID", markerscale=0.6)
    # Create the appropriate subdirectory for the descriptor using the neuron-specific path
    plot_directory = os.path.join(config.SAVE_DIRECTORY, f"Polarity_Dendrograms")
    os.makedirs(plot_directory, exist_ok=True)

    save_path = os.path.join(config.DESCRIPTORS_DIRECTORY, f'{neuron_name}_PCA_Polarity.png')
    plt.savefig(save_path)
    plt.close()




# def calculate_flux(neuron_data):
#     import numpy as np
#     import nibabel as nib
#     import os

#     G_graph = neuron_data['neuron_as_graph']
#     G_df = neuron_data['swc_df'].copy()
#     G_df.set_index('ID', inplace=True)
#     coord = G_df[['X', 'Y', 'Z']].to_numpy()
#     distances = G_df['euclidean_distance_to_soma'].to_numpy()
#     node_ids = G_df.index.to_numpy()
#     id_to_index = {id_val: idx for idx, id_val in enumerate(node_ids)}

#     termination_dict = {tp: tp for tp in neuron_data['termination_ids']}
#     termination_paths = {tp: nx.shortest_path(G_graph, source=1, target=tp) for tp in neuron_data['termination_ids']}
    
#     radii = np.sort(G_df.loc[np.concatenate((neuron_data['bifurcation_ids'], neuron_data['termination_ids'])), 'euclidean_distance_to_soma'].unique()) 
    
#     flux_values = []
#     for radius in radii:
#         cumulative_flux = 0
#         seen_flux = set()
        
#         for termination, path in termination_paths.items():
#             for i in range(len(path) - 1):
#                 parent_id = path[i]
#                 child_id = path[i + 1]
#                 parent_idx = id_to_index[parent_id]
#                 child_idx = id_to_index[child_id]

#                 parent_dist = distances[parent_idx]
#                 child_dist = distances[child_idx]
                
#                 if parent_dist > radius and child_dist > radius:
#                     break
                
#                 if parent_dist <= radius < child_dist or child_id in termination_dict:
#                     if child_id in termination_dict and child_dist == radius:
#                         intersection_point = coord[child_idx]
#                         node_id = str(child_id)
#                         termination_dict.pop(child_id)
#                     else:
#                         t = (radius - parent_dist) / (child_dist - parent_dist)
#                         intersection_point = coord[parent_idx] + t * (coord[child_idx] - coord[parent_idx])
#                         node_id = str(parent_id) + '-' + str(child_id)
                    
#                     radial_vector = intersection_point - coord[id_to_index[1]]
#                     segment_vector = coord[child_idx] - coord[parent_idx]

#                     norm_radial = np.linalg.norm(radial_vector)
#                     norm_segment = np.linalg.norm(segment_vector)
                    
#                     if norm_radial > 0 and norm_segment > 0:
#                         cosine_angle = np.dot(segment_vector, radial_vector) / (norm_radial * norm_segment)
#                         angle_rad = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
#                         flux = angle_rad
                        
#                         if flux not in seen_flux:
#                             seen_flux.add(flux)
#                             cumulative_flux += flux
        
#         flux_values.append((radius, cumulative_flux))

#     return flux_values




# need to test teh abocve as it it a more effient implemetation





def calculate_flux(neuron_data):
    """
    Calculate the flux descriptor for given neuron data by computing the dot product of segment vectors
    with radial vectors at intersection points of branches with spheres at specific radii.
    Now also records detailed information about each node where flux is calculated and the interpolation points.

    Parameters:
    - neuron_data: Dictionary containing neuron information including the graph and spatial data.

    Returns:
    - flux_details: Dictionary with radii as keys, each containing details on the cumulative flux and nodes involved.
    """
    G_graph = neuron_data['neuron_as_graph']
    G_df = neuron_data['swc_df'].copy()

    G_df.set_index('ID', inplace=True)
    coord = G_df[['X', 'Y', 'Z']].to_numpy()
    distances = G_df['euclidean_distance_to_soma'].to_numpy()
    node_ids = G_df.index.to_numpy()
    id_to_index = {id_val: idx for idx, id_val in enumerate(node_ids)}

    bifurcation_points = neuron_data['bifurcation_ids']
    termination_points = neuron_data['termination_ids']
    combined_points = np.concatenate((bifurcation_points, termination_points))
    radii = np.sort(G_df.loc[combined_points, 'euclidean_distance_to_soma'].unique()) 
    
    termination_dict = {tp: tp for tp in neuron_data['termination_ids']}

    flux_details = {}  # Dictionary to store details of flux calculations at each radius
    node_id = 0
    flux_values = [(0, 0.0)]  # Start with the flux at the soma
    # Iterate over each radius
    for radius in radii:
        seen_flux = set()
        cumulative_flux = 0
        node_details = []  # List to store details for each node at this radius

        # Iterate over all paths to termination points
        for termination in termination_points:
            path = nx.shortest_path(G_graph, source=1, target=termination)
            # print("Path from soma to termination ID", termination, ":", path)# Assuming soma is ID 1
            for i in range(len(path) - 1):
                parent_id = path[i]
                child_id = path[i + 1]
                parent_idx = id_to_index[parent_id]
                child_idx = id_to_index[child_id]

                parent_dist = distances[parent_idx]
                child_dist = distances[child_idx]
                
                # print (f'AT Radius {radius} ')
                # print(f'parent id {parent_id}  with distance {parent_dist} ')
                # print(f'child id {child_id}  with distance {child_dist} ')
                
                
                if parent_dist > radius and child_dist > radius :
                    break
                # Check if the segment crosses the sphere boundary
                if parent_dist <= radius < child_dist or child_id in termination_dict:
                    
                    if child_id in termination_dict and child_dist == radius  :
                        intersection_point = coord[child_idx]
                        node_id = str(child_id)
                        termination_dict.pop(child_id)
                       
                    else:
                        # Find intersection point using linear interpolation
                        t = (radius - parent_dist) / (child_dist - parent_dist)
                        intersection_point = coord[parent_idx] + t * (coord[child_idx] - coord[parent_idx])
                        node_id = str(parent_id) + '-' + str(child_id)
                        
                    radial_vector = intersection_point - coord[id_to_index[1]]  # Soma is at ID 1
                    segment_vector = coord[child_idx] - coord[parent_idx]

                    norm_radial = np.linalg.norm(radial_vector)
                    norm_segment = np.linalg.norm(segment_vector)
                    
                    
                    if norm_radial > 0 and norm_segment > 0:
  
                        # # *****************************METHOD 1************************
                        # Calculate cosine of the angle between the segment and radial vectors
                        cosine_angle = np.dot(segment_vector, radial_vector) / (norm_radial * norm_segment)
                        # Calculate the angle in radians
                        angle_rad = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
                        # Use the angle in radians to calculate flux, instead of using dot product directly
                        flux = angle_rad  # This replaces the previous flux calculation
        
                        
                        if flux not in seen_flux:
                            seen_flux.add(flux)  # Add the new unique flux value to the set
                            cumulative_flux += flux
                            # node_details.append({
                            #     'flux_at_node': flux,
                            #     'node_id': node_id,  # No specific node ID since this is an interpolated point
                            #     'interpolated_from': (parent_id, child_id)  # IDs of the points used for interpolation
                            # })
                            
                            
        flux_values.append((radius, cumulative_flux))
        # Store the calculated flux and node details for the current radius
        # flux_details[radius] = {
        #     'total_flux': cumulative_flux,
        #     'details': node_details
        # }
        

    if config.NORMALIZE_DESCRIPTOR_RADII:
        flux_values = normalize_descriptor_radii(flux_values)           

    if config.PLOT_STEP_FUNCTION:
        plot_step_function(
            descriptor_name="flux",
            neuron_name=neuron_data['neuron_name'],
            neuron_class=neuron_data['class'],
            step_data=flux_values
        )

    return flux_values


def calculate_tortuosity(neuron_data):
    """
    Process bifurcation and termination points to calculate path and Euclidean distances
    between these points and their nearest bifurcation or soma parent, including path distance
    through markers if necessary, and update neuron_data['swc_df'] in place.
    """
    
    swc_df = neuron_data['swc_df']
    
    # Sort SWC DataFrame by Euclidean distance to soma
    swc_df = swc_df.sort_values(by='euclidean_distance_to_soma').reset_index(drop=True)
    
    # Identify bifurcation and termination points
    branch_points = swc_df[swc_df['node_type'].isin(['bifurcation', 'termination'])]

    # Precompute dictionaries for easy lookup
    positions_dict = {row['ID']: np.array([row['X'], row['Y'], row['Z']]) for _, row in swc_df.iterrows()}
    euc_to_parent_dict = dict(zip(swc_df['ID'], swc_df['euc_to_parent']))
    soma_id = swc_df[swc_df['Parent'] == -1]['ID'].values[0]

    # Prepare lists to store results
    branch_ids = branch_points['ID'].values
    parent_branch_ids = []
    path_distances = []
    euclidean_distances = []
    path_to_euclidean_ratios = []

    # Vectorized approach for finding nearest bifurcation or soma parent
    for branch_id in branch_ids:
        current_id = branch_id
        path_distance = 0.0
        path_complete = False

        # Traversal to find the nearest bifurcation or soma parent
        while not path_complete:
            matching_rows = swc_df.loc[swc_df['ID'] == current_id, 'Parent']
            if matching_rows.empty:
                neuron_name = neuron_data['neuron_name']
                raise ValueError(f"No matching parent found for {neuron_name} current_id: {current_id}")
            else:
                parent_id = matching_rows.values[0]
                    
            parent_id = swc_df.loc[swc_df['ID'] == current_id, 'Parent'].values[0]
            path_distance += euc_to_parent_dict[current_id]

            # Check if the parent is soma or a bifurcation
            if parent_id == soma_id or parent_id in branch_ids:
                euclidean_distance = np.linalg.norm(positions_dict[branch_id] - positions_dict[parent_id])
                ratio = path_distance / euclidean_distance if euclidean_distance != 0 else np.nan
                # print('id = ' + str(branch_id ))
                # print('parent = ' + str(parent_id) )
                # print('path ' + str(path_distance) )
                # print('euclidean ' + str(euclidean_distance) )
                # print('ratio ' + str(ratio) )
                # print('====================')
                
                # Store results
                parent_branch_ids.append(parent_id)
                path_distances.append(path_distance)
                euclidean_distances.append(euclidean_distance)
                path_to_euclidean_ratios.append(ratio)

                path_complete = True
            current_id = parent_id

    # Update neuron_data['swc_df'] directly
    swc_df.loc[swc_df['ID'].isin(branch_ids), 'parent_branch_id'] = parent_branch_ids
    swc_df.loc[swc_df['ID'].isin(branch_ids), 'path_distance_to_bifurcation_parent'] = path_distances
    swc_df.loc[swc_df['ID'].isin(branch_ids), 'euclidean_distance_to_bifurcation_parent'] = euclidean_distances
    swc_df.loc[swc_df['ID'].isin(branch_ids), 'path_to_euclidean_segment_ratio'] = path_to_euclidean_ratios

    # The swc_df in neuron_data is now updated directly
    neuron_data['swc_df'] = swc_df
    
    # Calculate the average path-to-euclidean segment ratio for tortuosity measure
    tortuosity_values = []
    sorted_branch_points = swc_df[(swc_df['node_type'].isin(['bifurcation', 'termination']))].sort_values(by='euclidean_distance_to_soma')
    cumulative_sum = sorted_branch_points['path_to_euclidean_segment_ratio'].cumsum()
    count = np.arange(1, len(cumulative_sum) + 1)  # Array of counts up to each radius
    average_tortuosity = cumulative_sum / count  # Element-wise division for the average
    # Replace any NaN with 1 (element-wise)
    average_tortuosity = [1 if np.isnan(value) else value for value in average_tortuosity]
    
    radii = sorted_branch_points['euclidean_distance_to_soma'].values

    # Combine radii and average tortuosity into tuples
    tortuosity_values = list(zip(radii, average_tortuosity))
    
    tortuosity_values.insert(0, (0, 1))  # T(0) = 1 for the soma
    
    if config.NORMALIZE_DESCRIPTOR_RADII:
        tortuosity_values = normalize_descriptor_radii(tortuosity_values)  

    # Plot the tortuosity step function if configured
    if config.PLOT_STEP_FUNCTION:
        plot_step_function(
            descriptor_name="tortuosity",
            neuron_name=neuron_data['neuron_name'],
            neuron_class=neuron_data['class'],
            step_data=tortuosity_values
        )

    return tortuosity_values







# add and do not subtract in branching pattern 
# change the color of nodes for bif and terminations in plot neuron




def calculate_branching_pattern(neuron_data):
    """
    Calculate the branching pattern of a neuron based on Euclidean distances from the soma.

    Parameters:
    - neuron_data: Dictionary containing neuron information and distances.

    Returns:
    - branching_euclidean: List of tuples (euclidean_distance_to_soma, number_of_branches).
    """
    # Extract swc_df from the neuron_data dictionary
    # swc_df = neuron_data['swc_df']
    swc_df = neuron_data['swc_df'].copy()
    neuron_name = neuron_data['neuron_name']

    # Filter bifurcation, termination points, and include the soma
    bifurcation_points = swc_df[swc_df['node_type'] == 'bifurcation']
    termination_points = swc_df[swc_df['node_type'] == 'termination']
    soma = swc_df[swc_df['node_type'] == 'soma']

    # Combine soma, bifurcation, and termination points, sorting them by Euclidean distance to the soma
    combined_points = pd.concat([soma, bifurcation_points, termination_points])
    combined_points = combined_points.sort_values(by='euclidean_distance_to_soma').reset_index(drop=True)

    # Identify bifurcation and termination points
    is_bifurcation = combined_points['node_type'] == 'bifurcation'
    is_termination = combined_points['node_type'] == 'termination'

    # Calculate the cumulative number of branches
    number_of_branches = is_bifurcation.cumsum() + is_termination.cumsum()

    # Pair each Euclidean distance with its respective branch count
    branching_euclidean = list(zip(combined_points['euclidean_distance_to_soma'], number_of_branches))
    
    
    if config.NORMALIZE_DESCRIPTOR_RADII:
        branching_euclidean = normalize_descriptor_radii(branching_euclidean)  
        
    # Plot the step function if configured
    if config.PLOT_STEP_FUNCTION:
        plot_step_function(
            descriptor_name="branching_pattern",
            neuron_name=neuron_name,
            neuron_class = neuron_data['class'], 
            step_data=branching_euclidean
        )
    
    return branching_euclidean


def calculate_wiring_descriptor(neuron_data):
    """
    Calculate the wiring descriptor as the cumulative path distance within each bifurcation or termination radius.
    
    Parameters:
    - neuron_data: Dictionary containing neuron information and distances.
    
    Returns:
    - wiring_values: List of tuples (euclidean_distance_to_soma, cumulative_path_distance).
    """
    # Extract the necessary data from the neuron_data dictionary
    # swc_df = neuron_data['swc_df']
    swc_df = neuron_data['swc_df'].copy()
    neuron_name = neuron_data['neuron_name']
    neuron_class = neuron_data['class']
    
    radius_col = 'euclidean_distance_to_soma'
    parent_col = 'euc_to_parent'
    coord_cols = ['X', 'Y', 'Z']
    
    excluded_structure = config.PLOT_SWC_FILTER_TYPE


    
    # Filter to include only bifurcation and termination points for the radii
    bif_term_swc_df = swc_df[
        (swc_df['node_type'].isin(['bifurcation', 'termination', 'soma'])) &
        (swc_df[radius_col].notna())
    ]
    
    # Get unique sorted radii (euclidean distances of bifurcation and termination points to the soma)
    radii = np.sort(bif_term_swc_df[radius_col].unique())
    wiring_values = []

    # Iterate over each radus to calculate the cumulative path length
    for radius in radii:
        # Filter nodes within the current radius
        nodes_within_radius = swc_df[swc_df[radius_col] <= radius]

        # Calculate the sum of euc_to_parent for nodes within the radius
        total_path_length = nodes_within_radius[parent_col].sum()
        
        max_euc_radius = nodes_within_radius[radius_col].max()
      
        wiring_value = total_path_length / max_euc_radius if max_euc_radius > 0 else 0

        # divided by the linear radius in each ball (sadok April 28, 2025)
        wiring_values.append((radius, wiring_value))
                
    
    bifurcation_radii = neuron_data['bifurcation_radii']
    termination_radii = neuron_data['termination_radii']
    combined_radii = neuron_data['combined_radii']
    primary_branch_radii = neuron_data['primary_branch_radii']
    
    # normalize the radii ( x values ) to 0 and 1
    if config.NORMALIZE_DESCRIPTOR_RADII:
        wiring_values = normalize_descriptor_radii(wiring_values)  
        bifurcation_radii = neuron_data['bifurcation_radii_normalized']
        termination_radii = neuron_data['termination_radii_normalized']
        combined_radii = neuron_data['combined_radii_normalized']
        primary_branch_radii = neuron_data['primary_branch_radii_normalized']



    
    if config.PLOT_STEP_FUNCTION:
        plot_step_function_leaf(
            descriptor_name="wiring",
            neuron_name=neuron_data['neuron_name'],
            neuron_class=neuron_data['class'],
            step_data=wiring_values,
            bifurcation_radii=bifurcation_radii,
            termination_radii=termination_radii,
            primary_branch_radii=primary_branch_radii
         )


    return wiring_values


def calculate_spread_descriptor(neuron_data):
    """
    Calculate spread index (volume / path length) over radius values.
    Includes only nodes not matching excluded_structure in all calculations.

    Degenerate geometry is handled from the mathematical definition instead
    of by inserting sentinel measurements:

    * Fewer than four unique points cannot enclose a three-dimensional volume,
      so their convex-hull volume is 0.
    * Collinear or coplanar points also enclose zero three-dimensional volume.
    * A radius with zero cumulative path length has an undefined volume/path
      ratio (normally the soma-only radius), so it is omitted from the Spread
      descriptor rather than assigned an arbitrary numeric value.
    * A Qhull failure for points that do span three dimensions is treated as a
      data/computation error and reported with the neuron and radius.

    The previous implementation is retained as comments below for auditability.
    """
    swc_df = neuron_data['swc_df'].copy()
    radius_col = 'euclidean_distance_to_soma'
    parent_col = 'euc_to_parent'
    coord_cols = ['X', 'Y', 'Z']

    neuron_name = neuron_data['neuron_name']
    neuron_class = neuron_data['class']

    # Spread is evaluated over the whole dendritic tree, consistent with the other
    # five descriptors. The axon has already been removed at load time
    # (neuron_processor applies config.REMOVE_TYPES), so this is soma + basal +
    # apical.
    #
    # Restricting the hull to the apical subtree was considered as a way to limit
    # the influence of outlying basal nodes, but it is not supported: the
    # apical-only variant separates the cortical tiers less well while being no
    # less dependent on neuron size, and it would measure Spread on a different
    # subtree from the one all other descriptors use. It also leaves the
    # descriptor undefined for reconstructions carrying no apical label, of which
    # there are two here; taking the whole tree measures their geometry from the
    # basal nodes instead.
    filtered_df = swc_df[swc_df['Type'].isin([SOMA_TYPE, BASAL_TYPE, APICAL_TYPE])]

    # ✅ Define radii based only on included structures
    bif_term_df = filtered_df[
        (filtered_df['node_type'].isin(['bifurcation', 'termination', 'soma'])) &
        (filtered_df[radius_col].notna())
    ]
    radii = np.sort(bif_term_df[radius_col].unique())

    spread_index_values = []
    volume_values = []
    path_values = []

    for radius in radii:
        # Nodes within radius (only from included structures)
        nodes_within_radius = filtered_df[filtered_df[radius_col] <= radius]

        total_path_length = nodes_within_radius[parent_col].sum()

        # Spread = volume/path is undefined until a positive amount of neurite
        # path has accumulated. Omitting this radius avoids replacing 0/0 with
        # a made-up observation. Volume and path are still retained below.
        has_positive_path = np.isfinite(total_path_length) and total_path_length > 0

        points_array = nodes_within_radius[coord_cols].to_numpy(dtype=float)
        points_array = points_array[np.isfinite(points_array).all(axis=1)]
        unique_points = np.unique(points_array, axis=0)

        if len(unique_points) < 4:
            # Fewer than four unique points have zero 3-D convex-hull volume.
            convex_hull_volume = 0.0
        else:
            # Affine rank < 3 means that the points are collinear or coplanar;
            # either case has exactly zero volume in three dimensions.
            affine_rank = np.linalg.matrix_rank(unique_points - unique_points[0])
            if affine_rank < 3:
                convex_hull_volume = 0.0
            else:
                try:
                    convex_hull_volume = float(ConvexHull(unique_points).volume)
                except QhullError as error:
                    raise RuntimeError(
                        "ConvexHull failed for full-dimensional points in "
                        f"neuron {neuron_name!r} at radius {radius!r}."
                    ) from error

        if has_positive_path:
            spread_index = convex_hull_volume / float(total_path_length)
            spread_index_values.append((radius, spread_index))

        # Historical submitted-analysis fallback retained for audit purposes:
        # if len(nodes_within_radius) >= 4:
        #     points_array = nodes_within_radius[coord_cols].values
        #     try:
        #         hull = ConvexHull(points_array)
        #         convex_hull_volume = hull.volume
        #     except:
        #         convex_hull_volume = 1601
        # else:
        #     convex_hull_volume = 1600
        # spread_index = (
        #     convex_hull_volume / total_path_length
        #     if total_path_length > 0 else 1602
        # )
        # if spread_index == 0:
        #     spread_index = 999

        volume_values.append((radius, convex_hull_volume))
        path_values.append((radius, total_path_length))

    # Normalize radius-based data if enabled
    if config.NORMALIZE_DESCRIPTOR_RADII:
        max_radius = max((r for r, _ in spread_index_values), default=0)
        norm = lambda r: r / max_radius if max_radius > 0 else 0.0

        spread_index_values = [(norm(r), v) for r, v in spread_index_values]
        volume_values = [(norm(r), v) for r, v in volume_values]
        path_values = [(norm(r), v) for r, v in path_values]

        bifurcation_radii = filtered_df.loc[filtered_df['node_type'] == 'bifurcation', radius_col].dropna().map(norm).tolist()
        termination_radii = filtered_df.loc[filtered_df['node_type'] == 'termination', radius_col].dropna().map(norm).tolist()
        primary_branch_radii = filtered_df.loc[filtered_df['node_type'] == 'soma', radius_col].dropna().map(norm).tolist()
    else:
        bifurcation_radii = filtered_df.loc[filtered_df['node_type'] == 'bifurcation', radius_col].dropna().tolist()
        termination_radii = filtered_df.loc[filtered_df['node_type'] == 'termination', radius_col].dropna().tolist()
        primary_branch_radii = filtered_df.loc[filtered_df['node_type'] == 'soma', radius_col].dropna().tolist()

   

    if config.PLOT_STEP_FUNCTION:        
           
        plot_step_function_leaf("spread", neuron_name, neuron_class, spread_index_values,
                                bifurcation_radii, termination_radii, primary_branch_radii)
        
        plot_step_function_leaf("volume", neuron_name, neuron_class, volume_values,
                                bifurcation_radii, termination_radii, primary_branch_radii)

    return spread_index_values, volume_values, path_values





def plot_step_function_leaf(descriptor_name, neuron_name, neuron_class, step_data,
                       bifurcation_radii=None, termination_radii=None, primary_branch_radii=None, jump_locations=None):

    """
    Generalized plotting function for a single step function of different descriptors.
    Includes optional overlay of bifurcation and termination points as dots.
    Saves plot to descriptor-specific directory.

    Parameters:
    - descriptor_name: Name of the descriptor being plotted.
    - neuron_name: Name of the neuron.
    - neuron_class: Class of the neuron.
    - step_data: List of tuples (distance, value).
    - bifurcation_radii: List of radii for bifurcation nodes.
    - termination_radii: List of radii for termination nodes.
    """
    import matplotlib.pyplot as plt
    import os

    x, y = zip(*step_data)

    main_directory = os.path.join(config.DESCRIPTORS_DIRECTORY, "step_functions")
    plot_directory = os.path.join(main_directory, f"{descriptor_name}_step_functions")
    os.makedirs(plot_directory, exist_ok=True)

    plt.figure(figsize=(10, 8))
    plt.step(x, y, where='post', label=descriptor_name.capitalize(), color='blue', linestyle='-')
    plt.axhline(y=0, color='red', linestyle=':', linewidth=1)

    if bifurcation_radii:
        plt.plot(bifurcation_radii, [0]*len(bifurcation_radii), 'ro', label='Bifurcations')
    if termination_radii:
        plt.plot(termination_radii, [0]*len(termination_radii), 'go', label='Terminations')

    if primary_branch_radii:
        plt.plot(primary_branch_radii, [0]*len(primary_branch_radii), 'bo', label='Primary Branch Starts')
        
    if jump_locations:
        plt.plot(jump_locations, [0]*len(jump_locations), 'kx', label='Leaf Jumps', markersize=8)


    

    # Axis label and tick font sizes
    label_fontsize = 16
    tick_fontsize = 14

    plt.xlabel('Distance from Soma (micrometers)', fontsize=label_fontsize)
    plt.ylabel(descriptor_name.capitalize() + " Descriptor Values", fontsize=label_fontsize)
    plt.xticks(fontsize=tick_fontsize)
    plt.yticks(fontsize=tick_fontsize)

    plt.title(f'{descriptor_name.capitalize()} for {neuron_name}', fontsize=label_fontsize)
    # plt.legend(fontsize=14)
    # plt.legend(fontsize=14, loc='lower right')

    save_path = os.path.join(plot_directory, f'{neuron_class}_{neuron_name}_{descriptor_name}.png')
    plt.savefig(save_path, dpi=300)
    plt.close()




def calculate_leaf(neuron_data):
    """
    Calculate the leaf index descriptor for a given neuron.

    Supplementary section 1.6.5 defines the leaf index li(P) of a node P as the
    number of terminal points reachable from it, with li(P) = 1 when P is itself
    a leaf. Here that count is divided by the total number of leaves L:

        zeta_l(t_i) = R(i) / L

    so the descriptor is bounded in (0, 1]. The soma reaches every leaf and takes
    the value 1; a termination reaches only itself and takes the value 1 / L.

    Dividing by L makes the descriptor independent of neuron size. The raw count
    and its reciprocal both correlate strongly with total leaf number, so either
    would measure how large a neuron is at least as much as how its branches are
    arranged.

    Parameters:
    - neuron_data: Dictionary containing neuron information and distances.

    Returns:
    - leaf_values: List of tuples (radius, reachable_leaf_fraction).
    """
    G_graph = neuron_data['neuron_as_graph']
    G_df = neuron_data['swc_df'].copy()

    # # Decide which distance columns to use based on normalization flag
    # if config.NORMALIZE_DESCRIPTOR_RADII:
    #     radius_col = 'euclidean_distance_to_soma_normalized'
    #     parent_col = 'euc_to_parent_normalized'
    #     path_col = 'path_distance_to_soma_normalized'
    #     coord_cols = ['X_normalized', 'Y_normalized', 'Z_normalized']
    # else:
    #     radius_col = 'euclidean_distance_to_soma'
    #     parent_col = 'euc_to_parent'
    #     path_col = 'path_distance_to_soma'
    #     coord_cols = ['X', 'Y', 'Z']
    
    radius_col = 'euclidean_distance_to_soma'
    radius_col_normalized = 'euclidean_distance_to_soma_normalized'
    parent_col = 'euc_to_parent'
    
    # Extract bifurcation and termination points with appropriate distance column
    bifurcation_points = G_df[G_df['node_type'] == 'bifurcation'][['ID', radius_col, radius_col_normalized]]
    termination_points_df = G_df[G_df['node_type'] == 'termination'][['ID', radius_col, radius_col_normalized]]
    termination_points = set(termination_points_df['ID'])

    # Combine bifurcation and termination points
    combined_points = pd.concat([bifurcation_points, termination_points_df])
    combined_count = len(combined_points)

    # Get all unique paths from the soma (ID=1) to termination points
    paths = [
        nx.shortest_path(G_graph, source=1, target=termination)
        for termination in termination_points
    ]

    # Prepare list to store the leaf values
    leaf_values = []
    total_reachable_terminations = len(termination_points)
    leaf_values.append((0.0, 1.0))  # Soma reaches every leaf: R = L, so R/L = 1

    # Process bifurcations and terminations
    for _, row in combined_points.iterrows():
        node_id = row['ID']
        radius = row[radius_col]

        # Leaves whose soma-to-leaf path passes through this node are exactly
        # the leaves reachable from it.
        if node_id in termination_points:
            reachable_terminations = 1  # a leaf reaches only itself
        else:
            reachable_terminations = sum(1 for path in paths if node_id in path)
        value = reachable_terminations / total_reachable_terminations
        leaf_values.append((radius, value))

    leaf_values.sort(key=lambda x: x[0])
    
    median_all_nodes = float(np.median(combined_points[radius_col]))
    bifurcation_radii = neuron_data['bifurcation_radii']
    termination_radii = neuron_data['termination_radii']
    combined_radii = neuron_data['combined_radii']    
    primary_branch_radii = neuron_data['primary_branch_radii']
    
    if config.NORMALIZE_DESCRIPTOR_RADII:
        leaf_values = normalize_descriptor_radii(leaf_values)
        
        median_all_nodes = float(np.median(combined_points[radius_col_normalized]))
        bifurcation_radii = neuron_data['bifurcation_radii_normalized']
        termination_radii = neuron_data['termination_radii_normalized']
        combined_radii = neuron_data['combined_radii_normalized']
        primary_branch_radii = neuron_data['primary_branch_radii_normalized']
        
    
    # ==================Medians====================================
   
    step_function_data = {
    'leaf_jump_count': 0,
    'x_locations': []
    }
    # Nodes from which every leaf is reachable. Under the R/L convention this is
    # 1.0; taking the observed maximum keeps the diagnostic independent of the
    # descriptor's scaling convention.
    max_val = max((v for _, v in leaf_values), default=0)

    for i in range(len(leaf_values) - 1):
        current_x, current_y = leaf_values[i]
        next_y = leaf_values[i + 1][1]
    
        if current_y == max_val and next_y != max_val:
            step_function_data['leaf_jump_count'] += 1
            step_function_data['x_locations'].append(current_x)
            
    median_jumps = float(np.median(step_function_data['x_locations'])) if step_function_data['x_locations'] else None
    
    step_function_data['leaf_jumps_median'] = median_jumps
    step_function_data['leaf_all_nodes_median'] = median_all_nodes
    step_function_data['leaf_jump_index'] = step_function_data['leaf_jump_count'] / combined_count if combined_count > 0 else 0
      
    step_function_data['bifurcation_radii_median'] = float(np.median(bifurcation_radii)) if bifurcation_radii else None
    step_function_data['termination_radii_median'] = float(np.median(termination_radii)) if termination_radii else None
    step_function_data['combined_radii_median'] = float(np.median(combined_radii)) if combined_radii else None
        
    
    if median_all_nodes and median_jumps is not None:
        step_function_data['leaf_nodal_index'] = (median_all_nodes - median_jumps) / median_all_nodes
    else:
        step_function_data['leaf_nodal_index'] = None
    
    # ==================================================


    neuron_name = neuron_data['neuron_name']
    neuron_data['step_function_data'] = step_function_data



    if config.PLOT_STEP_FUNCTION:
        plot_step_function_leaf(
            descriptor_name="leaf",
            neuron_name=neuron_data['neuron_name'],
            neuron_class=neuron_data['class'],
            step_data=leaf_values,
            bifurcation_radii=bifurcation_radii,
            termination_radii=termination_radii,
            primary_branch_radii=primary_branch_radii
            # jump_locations=step_function_data['x_locations']
        )

    return leaf_values


def plot_median_leaf_jump_data(neuron_data, neuron_class_df,all_descriptors ):
    class_lookup = dict(zip(neuron_class_df['neuron_name'], neuron_class_df['class']))
    class_data = {}
    bifurcation_radii_by_class = {}
    termination_radii_by_class = {}
    
    
    extracted_volume_data = {}
    
    extracted_spread_data = {}

    for descriptor_name, descriptor_data in all_descriptors.items():
        # Skip the descriptor if it is in the exclusion list
        if descriptor_name in config.EXCLUDED_DESCRIPTORS:
            print(f'Skipping descriptor {descriptor_name} as it is in the exclusion list.')
            continue
    
        if descriptor_name == "Spread":
            extracted_spread_data = {
                neuron: max((v[1] for v in values), default=0)
                for neuron, values in descriptor_data.items()
            }
            
        if descriptor_name == "Volume":
            extracted_volume_data = {
                neuron: max((v[1] for v in values), default=0)
                for neuron, values in descriptor_data.items()
            }
            
   


    for neuron_name, data in neuron_data.items():
        if 'step_function_data' not in data:
            continue

        step_data = data['step_function_data']
        jumps = step_data.get('x_locations', [])
        if not jumps:
            continue

        neuron_class = class_lookup.get(neuron_name, 'Unknown')
        if neuron_class not in class_data:
            class_data[neuron_class] = []
            bifurcation_radii_by_class[neuron_class] = []
            termination_radii_by_class[neuron_class] = []

        median = float(step_data.get('leaf_jumps_median', np.median(jumps)))
        jump_index = round(step_data.get('leaf_jump_index', 0), 2)
        nodal_index = round(step_data.get('leaf_nodal_index', 0), 2)
     

        bifurcation_radii = data['bifurcation_radii']
        termination_radii = data['termination_radii']
        combined_radii = data['combined_radii']
        primary_branch_radii = data['primary_branch_radii']


        bif_median = step_data.get('bifurcation_radii_median', None)
        term_median = step_data.get('termination_radii_median', None)
        comb_median = step_data.get('combined_radii_median', None)


        class_data[neuron_class].append({
            'name': neuron_name,
            'jumps': jumps,
            'median': median,
            'jump_index': jump_index,
            'nodal_index': nodal_index,
            'bif_median': bif_median,
            'term_median': term_median,
            'comb_median': comb_median,
   
        })
        
        

        bifurcation_radii_by_class[neuron_class].extend(bifurcation_radii)
        termination_radii_by_class[neuron_class].extend(termination_radii)

    main_directory = os.path.join(config.DESCRIPTORS_DIRECTORY, "box_plots_leaf_jumps")
    os.makedirs(main_directory, exist_ok=True)

    for neuron_class, entries in class_data.items():
        sorted_entries = sorted(entries, key=lambda x: x['median'])
        data_to_plot = [x['jumps'] for x in sorted_entries]
        labels = [f"{x['name']} (j={x['jump_index']:.2f}, n={x['nodal_index']:.2f})" for x in sorted_entries]

        # Boxplot of jump locations
        plt.figure(figsize=(12, max(4, len(labels) * 0.5)))
        plt.boxplot(data_to_plot, vert=False, patch_artist=True,
                    boxprops=dict(facecolor='lightblue', color='blue'),
                    medianprops=dict(color='red'),
                    whiskerprops=dict(color='gray'),
                    capprops=dict(color='gray'),
                    flierprops=dict(marker='o', markerfacecolor='gray', markersize=5))
        plt.yticks(ticks=np.arange(1, len(labels) + 1), labels=labels, fontsize=8)
        plt.xlabel("Distance from Soma (micrometers)")
        plt.title(f"Jump X-Locations Boxplot - Class: {neuron_class}")
        plt.tight_layout()
        plt.savefig(os.path.join(main_directory, f"{neuron_class}_jump_boxplots.png"))
        plt.close()

        # Barplot: Jump Index
        jump_sorted = sorted(entries, key=lambda x: x['jump_index'], reverse=True)
        j_labels = [x['name'] for x in jump_sorted]
        j_values = [x['jump_index'] for x in jump_sorted]

        plt.figure(figsize=(12, max(4, len(j_labels) * 0.4)))
        plt.barh(j_labels, j_values, color='cornflowerblue')
        plt.xlabel("Jump Index (j)")
        plt.title(f"Jump Index per Neuron - Class: {neuron_class}")
        plt.tight_layout()
        plt.savefig(os.path.join(main_directory, f"{neuron_class}_jump_index_bars.png"))
        plt.close()   
        
        
        

        # Barplot: Nodal Index
        nodal_sorted = sorted(entries, key=lambda x: x['nodal_index'], reverse=True)
        n_labels = [x['name'] for x in nodal_sorted]
        n_values = [x['nodal_index'] for x in nodal_sorted]

        plt.figure(figsize=(12, max(4, len(n_labels) * 0.4)))
        plt.barh(n_labels, n_values, color='seagreen')
        plt.xlabel("Nodal Index (n)")
        plt.title(f"Nodal Index per Neuron - Class: {neuron_class}")
        plt.tight_layout()
        plt.savefig(os.path.join(main_directory, f"{neuron_class}_nodal_index_bars.png"))
        plt.close()
        
        # Barplot: Volume Index

        


        spread_entries = [(x['name'], extracted_spread_data.get(x['name'], 0)) for x in entries]
        volume_sorted = sorted(spread_entries, key=lambda x: x[1], reverse=True)
        vol_labels = [x[0] for x in volume_sorted]
        vol_values = [x[1] for x in volume_sorted]        
        plt.figure(figsize=(12, max(4, len(vol_labels) * 0.4)))
        plt.barh(vol_labels, vol_values, color='purple')
        plt.xlabel("Spread")
        plt.title(f"spread per Neuron - Class: {neuron_class}")
        plt.tight_layout()
        plt.savefig(os.path.join(main_directory, f"{neuron_class}_Spread_bars.png"))
        plt.close()


        volume_entries = [(x['name'], extracted_volume_data.get(x['name'], 0)) for x in entries]
        volume_sorted = sorted(volume_entries, key=lambda x: x[1], reverse=True)
        vol_labels = [x[0] for x in volume_sorted]
        vol_values = [x[1] for x in volume_sorted]        
        plt.figure(figsize=(12, max(4, len(vol_labels) * 0.4)))
        plt.barh(vol_labels, vol_values, color='purple')
        plt.xlabel("Volume")
        plt.title(f"Volume per Neuron - Class: {neuron_class}")
        plt.tight_layout()
        plt.savefig(os.path.join(main_directory, f"{neuron_class}_volume_bars.png"))
        plt.close()

        

        # --- CLASS-LEVEL RADIUS PLOTS ---
        bif_r_all = bifurcation_radii_by_class.get(neuron_class, [])
        term_r_all = termination_radii_by_class.get(neuron_class, [])
        comb_r_all = bif_r_all + term_r_all

        if bif_r_all:
            plt.figure(figsize=(12, 5))
            plt.bar(range(len(bif_r_all)), sorted(bif_r_all, reverse=True), color='steelblue')
            plt.title(f"All Bifurcation Radii - Class: {neuron_class}")
            plt.xlabel("Bifurcation Node Index")
            plt.ylabel("Normalized Radius")
            plt.tight_layout()
            plt.savefig(os.path.join(main_directory, f"{neuron_class}_all_bifurcation_radii.png"))
            plt.close()

        if term_r_all:
            plt.figure(figsize=(12, 5))
            plt.bar(range(len(term_r_all)), sorted(term_r_all, reverse=True), color='indianred')
            plt.title(f"All Termination Radii - Class: {neuron_class}")
            plt.xlabel("Termination Node Index")
            plt.ylabel("Normalized Radius")
            plt.tight_layout()
            plt.savefig(os.path.join(main_directory, f"{neuron_class}_all_termination_radii.png"))
            plt.close()

        if comb_r_all:
            plt.figure(figsize=(12, 5))
            plt.bar(range(len(comb_r_all)), sorted(comb_r_all, reverse=True), color='slategray')
            plt.title(f"All Combined Radii - Class: {neuron_class}")
            plt.xlabel("Node Index")
            plt.ylabel("Normalized Radius")
            plt.tight_layout()
            plt.savefig(os.path.join(main_directory, f"{neuron_class}_all_combined_radii.png"))
            plt.close()

        # --- PER-NEURON MEDIAN RADIUS PLOTS ---
        def plot_medians(entries, key, color, xlabel, title, filename):
            medians = [(x['name'], x[key]) for x in entries if x[key] is not None]
            if medians:
                medians_sorted = sorted(medians, key=lambda x: x[1], reverse=True)
                labels = [x[0] for x in medians_sorted]
                values = [x[1] for x in medians_sorted]
                plt.figure(figsize=(12, max(4, len(labels) * 0.4)))
                plt.barh(labels, values, color=color)
                plt.xlabel(xlabel)
                plt.title(title)
                plt.tight_layout()
                plt.savefig(os.path.join(main_directory, filename))
                plt.close()

        plot_medians(entries, 'bif_median', 'darkorange',
                     "Median Bifurcation Radius (normalized)",
                     f"Per-Neuron Median Bifurcation Radii - Class: {neuron_class}",
                     f"{neuron_class}_median_bifurcation_radii_per_neuron.png")

        plot_medians(entries, 'term_median', 'firebrick',
                     "Median Termination Radius (normalized)",
                     f"Per-Neuron Median Termination Radii - Class: {neuron_class}",
                     f"{neuron_class}_median_termination_radii_per_neuron.png")

        plot_medians(entries, 'comb_median', 'dimgray',
                     "Median Combined Radius (normalized)",
                     f"Per-Neuron Median Combined Radii - Class: {neuron_class}",
                     f"{neuron_class}_median_combined_radii_per_neuron.png")






def compute_all_descriptors_by_type(neuron_vectors):
    """
    Iterates over all neurons and computes each descriptor across all neurons, except those specified to be excluded.
    Groups the results by descriptor type rather than by neuron.
    
    Parameters:
    - neuron_vectors: Dictionary containing neuron data.
    - exclude_descriptors: List of descriptor names to exclude from computation.
    
    Returns:
    - all_descriptors: Dictionary where each key is a descriptor name and each value is a
      dictionary with neuron names as keys and their computed descriptor values.
    """
    
    polarity_matrices = {}
    
    # Initialize a dictionary for each descriptor.
    all_descriptors = {
        'Tortuosity': {},
        'Branching_Pattern': {},
        'Wiring': {},
        'Flux': {},
        # 'Energy': {}, 
        'Leaf': {},
        'Spread' : {},
        # 'Volume' : {},
        # 'Path' : {},
    }
    
    # Descriptor functions mapped to their names
    descriptor_functions = {
        'Tortuosity': calculate_tortuosity,
        'Branching_Pattern': calculate_branching_pattern,
        'Wiring': calculate_wiring_descriptor,
        'Flux': calculate_flux,
        # # 'Energy_and_Polarity': calculate_energy_and_polarity_matrix,  # Handles both Energy and Polarity
        # 'Energy': calculate_energy,
        'Leaf': calculate_leaf,
        'Spread' : calculate_spread_descriptor,
    }

    # Iterate over each neuron and compute descriptors.
    total_neurons = len(neuron_vectors)
    for idx, (neuron_name, neuron_data) in enumerate(neuron_vectors.items(), start=1):
        print(f'Processing ({idx}/{total_neurons}): {neuron_name}')
        
        # Skip processing if neuron is in the exclusion list
        if neuron_name in config.EXCLUDED_NEURONS:
            print(f"Skipping excluded neuron: {neuron_name}")
            continue
        
        
    
        # Iterate through each descriptor and compute if not in exclude list and not already handled
        for descriptor_name, func in descriptor_functions.items():

            if descriptor_name not in exclude_descriptors:
                
                if descriptor_name == 'Energy':
                    # Specifically capture only the energy values from the energy function
                    energy_values, _ = func(neuron_data)  # Discard energy_vectors
                    all_descriptors[descriptor_name][neuron_name] = energy_values
                
                elif descriptor_name == 'Spread':
                    volume_index_values, volume_values, path = func(neuron_data)
                    all_descriptors[descriptor_name][neuron_name] = volume_index_values
                    all_descriptors['Volume'][neuron_name] = volume_values
                    all_descriptors['Path'][neuron_name] = path
                
                else:
                    result = func(neuron_data)
                    all_descriptors[descriptor_name][neuron_name] = result
            
            # Handle Polarity separately and store it in the dictionary
            if 'Polarity' not in exclude_descriptors:
               polarity_matrix = calculate_polarity(neuron_data)
               polarity_matrices[neuron_name] = polarity_matrix  
    
    

    return all_descriptors, polarity_matrices


def plot_descriptor_values(all_descriptors):
    
    summary_by_descriptor = {}
    
    for descriptor_name, descriptor_data in all_descriptors.items():
        # Skip the descriptor if it is in the exclusion list
        if descriptor_name in config.EXCLUDED_DESCRIPTORS:
            print(f'Skipping descriptor {descriptor_name} as it is in the exclusion list.')
            continue

        # Extract only the y-values for box plot generation
        extracted_data = {neuron: [value[1] for value in values] for neuron, values in descriptor_data.items()}
        
        # Convert to wide-format DataFrame
        df = pd.DataFrame.from_dict(extracted_data, orient='index').T

        # Sort neurons by their median values for plotting
        median_values = df.median().sort_values()
        sorted_columns = median_values.index.tolist()

        # Create a box plot for all neurons
        plt.figure(figsize=(15, 8))  # Adjust the figure size as needed
        
        ax = sns.boxplot(data=df[sorted_columns], 
                         orient='h', 
                         palette="viridis",
                         fliersize=0.5,
                         flierprops={'marker': '.', 
                                     'markersize': 4, 
                                     'markerfacecolor': 'black',
                                     'alpha': 0.3})
        
        ax.set_title(f'Box plots for {descriptor_name}')
        ax.set_xlabel('Values')
        ax.set_ylabel('Neurons')
        plt.xticks(rotation=45)  # Rotate x-axis labels for better visibility if needed
        plt.grid(False)  # Optional: Adds a grid for better readability
        plt.tight_layout()

        # Save the plot
        plot_filename = os.path.join(config.DESCRIPTORS_DIRECTORY, f'{descriptor_name}_boxplot_raw_descriptor_values.png')
        plt.savefig(plot_filename)
        plt.close()
        
        # Analyze: Compute summary stats per neuron
        summary = {}
        
        for neuron in df.columns:
            values = df[neuron].dropna()
            if len(values) == 0:
                continue
        
            q1 = values.quantile(0.25)
            q3 = values.quantile(0.75)
            iqr = q3 - q1
            whisker_max = q3 + 1.5 * iqr
            actual_max = values.max()
        
            summary[neuron] = {
                "count": len(values),
                "min": values.min(),
                "q1": q1,
                "median": values.median(),
                "q3": q3,
                "max": actual_max,
                "iqr": iqr,
                "whisker_max": whisker_max,
                "no_top_whisker": actual_max <= q3
            }
        
        summary_df = pd.DataFrame(summary).T
        summary_by_descriptor[descriptor_name] = summary_df
        
    return summary_by_descriptor



def extract_descriptor_values(descriptor_data):
    """
    Extracts y-values from a descriptor data structure and returns a DataFrame with neuron names and their corresponding y-values.

    Parameters:
    - descriptor_data (dict): Dictionary with neuron names as keys and lists of tuples (data points) as values.

    Returns:
    - DataFrame: A DataFrame where each row corresponds to a neuron and contains a list of y-values.
    """
    # Extract only the y-values for each neuron
    extracted_data = {neuron: [value[1] for value in values] for neuron, values in descriptor_data.items()}

    # Convert the extracted data to a DataFrame
    neuron_df = pd.DataFrame(list(extracted_data.items()), columns=['neuron_name', 'values'])
    
    return neuron_df

# def compute_all_descriptors_by_type(neuron_vectors):
#     """
#     Iterates over all neurons and computes each descriptor across all neurons, except those specified to be excluded.
#     Groups the results by descriptor type rather than by neuron.
    
#     Parameters:
#     - neuron_vectors: Dictionary containing neuron data.
#     - exclude_descriptors: List of descriptor names to exclude from computation.
    
#     Returns:
#     - all_descriptors: Dictionary where each key is a descriptor name and each value is a
#       dictionary with neuron names as keys and their computed descriptor values.
#     """
#     # Initialize a dictionary for each descriptor.
#     all_descriptors = {
#         'Tortuosity': {},
#         'Branching_Pattern': {},
#         'Wiring': {},
#         'Flux': {},
#         'Energy': {},  # Assumes if Energy is out, Polarity is also out
#         'Leaf': {},
#         'Polarity': {}  # Treated the same as Energy
#     }
    
#     # Descriptor functions mapped to their names
#     descriptor_functions = {
#         'Tortuosity': calculate_tortuosity,
#         'Branching_Pattern': calculate_branching_pattern,
#         'Wiring': calculate_wiring_descriptor,
#         'Flux': calculate_flux,
#         'Energy_and_Polarity': calculate_energy_and_polarity_matrix,  # Handles both Energy and Polarity
#         'Leaf': calculate_leaf
#     }

#     # Iterate over each neuron and compute descriptors.
#     for neuron_name, neuron_data in neuron_vectors.items():
#         print(f'Processing descriptors for {neuron_name}')
        
#         # Check grouped descriptors first
#         if 'Energy' in exclude_descriptors or 'Polarity' in exclude_descriptors:
#             print(f'Skipping Energy and Polarity descriptors as they are in the exclusion list.')
#         else:
#             energy, polarity_matrix = calculate_energy_and_polarity_matrix(neuron_data)
#             all_descriptors['Energy'][neuron_name] = energy
#             all_descriptors['Polarity'][neuron_name] = polarity_matrix
        
#         # Iterate through each descriptor and compute if not in exclude list and not already handled
#         for descriptor_name, func in descriptor_functions.items():
#             if descriptor_name in ['Energy_and_Polarity', 'Energy', 'Polarity']:  # Skip already handled or grouped
#                 continue
#             if descriptor_name in exclude_descriptors:
#                 print(f'Skipping descriptor {descriptor_name} as it is in the exclusion list.')
#                 continue
            
#             # Compute and store the result
#             result = func(neuron_data)
#             all_descriptors[descriptor_name][neuron_name] = result

#     return all_descriptors



# def compute_all_descriptors_by_type(neuron_vectors):
#     """
#     Iterates over all neurons and computes each descriptor across all neurons.
#     Groups the results by descriptor type rather than by neuron.
    
#     Parameters:
#     - neuron_vectors: Dictionary containing neuron data.
    
#     Returns:
#     - all_descriptors: Dictionary where each key is a descriptor name and each value is a
#       dictionary with neuron names as keys and their computed descriptor values.
#     """
#     # Initialize a dictionary for each descriptor.
#     all_descriptors = {
#         'Tortuosity': {},
#         'Branching_Pattern': {},
#         'Wiring': {},
#         'Flux': {},
#         'Energy': {},
#         'Leaf': {},
#         'Polarity':{}
#     }
    
#     # Iterate over each neuron and compute descriptors.
#     for neuron_name, neuron_data in neuron_vectors.items():
#         print(f'Processing descriptors for {neuron_name}')

#         # Calculate the tortuosity descriptor.
#         tortuosity_values = calculate_tortuosity(neuron_data)
#         all_descriptors['Tortuosity'][neuron_name] = tortuosity_values

#         # Calculate the branching pattern descriptor.
#         branching_euclidean = calculate_branching_pattern(neuron_data)
#         all_descriptors['Branching_Pattern'][neuron_name] = branching_euclidean

#         # Calculate the wiring descriptor.
#         wiring = calculate_wiring_descriptor(neuron_data)
#         all_descriptors['Wiring'][neuron_name] = wiring

#         # Calculate the flux descriptor.
#         flux = calculate_flux(neuron_data)
#         all_descriptors['Flux'][neuron_name] = flux

        
#         energy, polarity_matrix= calculate_energy_and_polarity_matrix(neuron_data)
#         all_descriptors['Energy'][neuron_name] = energy
#         all_descriptors['Polarity'][neuron_name] = polarity_matrix
        

#         # Calculate the leaf descriptor.
#         leaf = calculate_leaf(neuron_data)
#         all_descriptors['Leaf'][neuron_name] = leaf

#     return all_descriptors







