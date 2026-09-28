import numpy as np
import os
import networkx as nx
import pandas as pd
import config
import neurotopo.utils as utl
import plotly.graph_objects as go
import re
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd
import numpy as np
import os


# Mapping from component names to SWC type numbers
COMPONENT_TYPE_MAP = {
    'axon': 2,
    'basal': 3,
    'apical': 4
}


def process_neuron(filepath):
    """
    Processes the SWC file DataFrame to compute path and Euclidean distances for all nodes,
    including bifurcations, terminations, and marker nodes.
    
    Returns a DataFrame with the following columns:
    - node id (as integer)
    - node type (soma, termination, bifurcation, marker)
    - euclidean distance
    - path distance
    - euc_to_parent
    """
  
    
    original_swc_df = pd.read_csv(filepath, delim_whitespace=True, comment='#', header=None,
                         names=['ID', 'Type', 'X', 'Y', 'Z', 'R', 'Parent'])
    
    
    try:
        swc_df, standardization_description = utl.standardize_swc_full(filepath)
        standardization_status = "pass"
    except Exception as e:
        swc_df = original_swc_df  # fallback to unmodified
        standardization_description = str(e)
        standardization_status = "fail"



    # Convert names in remove_types to their corresponding numerical values
    remove_type_nums = [COMPONENT_TYPE_MAP.get(t, t) for t in config.REMOVE_TYPES]

    # Remove the specified types, but always keep the soma (Type == 1)
    if remove_type_nums:
        swc_df = swc_df[~swc_df['Type'].isin(remove_type_nums) | (swc_df['Type'] == 1)]
    
    # if config.NORMALIZE:
    #     swc_df = utl.normalize_swc(swc_df)
    #     # swc_df = utl.min_max_with_soma_zero(swc_df)
    
    
        
    
    # Ensure node IDs and Parent IDs are integers
    # swc_df[['ID', 'Parent']] = swc_df[['ID', 'Parent']].astype(int)
    swc_df = swc_df.copy()
    swc_df.loc[:, ['ID', 'Parent']] = swc_df[['ID', 'Parent']].astype(int)

    # Soma is the first point where Parent is -1
    soma = swc_df[swc_df['Parent'] == -1].iloc[0]
    soma_id = int(soma['ID'])
    soma_position = np.array([soma['X'], soma['Y'], soma['Z']])

    # Count primary branches: children of the soma
    primary_branch_count = (swc_df['Parent'] == soma_id).sum()

    # Precompute positions as a dictionary {ID: numpy array of (X, Y, Z)}
    positions = np.vstack(swc_df[['X', 'Y', 'Z']].values)
    # positions_dict = {row['ID']: pos for row, pos in zip(swc_df.to_dict('records'), positions)}
    
    positions_dict = {int(row['ID']): pos for row, pos in zip(swc_df.to_dict('records'), positions)}


    # # Generate weighted edges using Euclidean distance between each node and its parent
    # edges = swc_df[swc_df['Parent'] != -1].apply(
    #     lambda row: (
    #         row['ID'],
    #         row['Parent'],
    #         np.linalg.norm(positions_dict[row['ID']] - positions_dict[row['Parent']])
    #     ),
    #     axis=1
    # ).tolist()
    
    edges = swc_df[swc_df['Parent'] != -1].apply(
    lambda row: (
        int(row['ID']),
        int(row['Parent']),
        np.linalg.norm(positions_dict[int(row['ID'])] - positions_dict[int(row['Parent'])])
            ),
            axis=1
        ).tolist()


    # Create a NetworkX graph
    G = nx.Graph()
    

    # G.add_nodes_from(swc_df['ID'])
    # G.add_nodes_from((node_id, {'pos': positions_dict[node_id]}) for node_id in swc_df['ID'])
    G.add_weighted_edges_from(edges)
    
    G.graph['soma_id'] = soma_id
    
    # Identify bifurcation points (nodes with more than one child)
    parent_counts = swc_df['Parent'].value_counts()
    bifurcation_ids = parent_counts[parent_counts > 1].index.values
    bifurcation_ids = np.setdiff1d(bifurcation_ids, [soma_id])  # Exclude soma
    bifurcation_ids = np.sort(bifurcation_ids)

    # Identify termination points (nodes with no children)
    termination_ids = swc_df.loc[~swc_df['ID'].isin(swc_df['Parent']), 'ID'].values
    termination_ids = np.setdiff1d(termination_ids, [soma_id])  # Exclude soma

    # Identify marker points (nodes that are not bifurcations, terminations, or the soma)
    marker_ids = swc_df.loc[
        ~swc_df['ID'].isin([soma_id] + bifurcation_ids.tolist() + termination_ids.tolist()), 'ID'
    ].values

    # Compute Euclidean distances for each node
    swc_df['euclidean_distance_to_soma'] = np.linalg.norm(
        positions - soma_position, axis=1
    )

    # Calculate euc_to_parent for each node (vectorized)
    swc_df['euc_to_parent'] = swc_df.apply(
        lambda row: np.linalg.norm(positions_dict[row['ID']] - positions_dict[row['Parent']])
        if row['Parent'] != -1 else 0,
        axis=1
    )

    # Precompute path distances using NetworkX for all nodes
    path_distances = nx.single_source_dijkstra_path_length(G, source=soma_id, weight='weight')

    # Map the path distances back to the DataFrame
    swc_df['path_distance_to_soma'] = swc_df['ID'].map(path_distances)
    
    # --- Normalize distances ---
    swc_df['euclidean_distance_to_soma_normalized'] = (
        swc_df['euclidean_distance_to_soma'] / swc_df['euclidean_distance_to_soma'].max()
    )
    
    swc_df['euc_to_parent_normalized'] = (
        swc_df['euc_to_parent'] / swc_df['euc_to_parent'].max()
    )
    
    swc_df['path_distance_to_soma_normalized'] = (
        swc_df['path_distance_to_soma'] / swc_df['path_distance_to_soma'].max()
    )
    
    # Normalize X, Y, Z coordinates too
    scaling_factor = swc_df['euclidean_distance_to_soma'].max()
    swc_df['X_normalized'] = swc_df['X'] / scaling_factor
    swc_df['Y_normalized'] = swc_df['Y'] / scaling_factor
    swc_df['Z_normalized'] = swc_df['Z'] / scaling_factor
    

    # Create a DataFrame for the node types
    node_ids = [soma_id] + bifurcation_ids.tolist() + termination_ids.tolist() + marker_ids.tolist()
    node_types = (
        ['soma'] + 
        ['bifurcation'] * len(bifurcation_ids) + 
        ['termination'] * len(termination_ids) + 
        ['marker'] * len(marker_ids)
    )

    neuron_df = pd.DataFrame({
        'ID': node_ids,
        'node_type': node_types
    })

    # Merge neuron_df with swc_df based on the 'ID' column to combine all relevant data
    swc_df_combined = swc_df.merge(neuron_df, on='ID', how='left', suffixes=('', '_type'))
    
    
    
    # # Add nodes with position and node_type attributes
    # for _, row in swc_df_combined.iterrows():
    #     G.add_node(row['ID'], pos=positions_dict[row['ID']], node_type=row['node_type'])
    
    # for _, row in swc_df_combined.iterrows():
    #     G.add_node(int(row['ID']), pos=positions_dict[int(row['ID'])], node_type=row['node_type'])
    
    for _, row in swc_df_combined.iterrows():
        G.add_node(
            int(row['ID']),
            pos=positions_dict[int(row['ID'])],
            node_type=row['node_type'],
            Type=int(row['Type'])  # Ensure it's an int
        )



    # Return the bifurcation IDs, distances, and the combined DataFrame
    combined_path_distances = swc_df_combined['path_distance_to_soma'].values
    combined_euclidean_distances = swc_df_combined['euclidean_distance_to_soma'].values
    
    # # Ensure no division by zero by replacing zero values with NaN temporarily
    # swc_df_combined['path_over_euclidean'] = swc_df_combined['path_distance_to_soma'] / swc_df_combined['euclidean_distance_to_soma'].replace(0, np.nan)
    
    # # Replace NaN values with zero if you want to handle cases where the euclidean distance is zero
    # swc_df_combined['path_over_euclidean'].fillna(0, inplace=True)
    
    # ===========number of branches ==============================================
    # Build child adjacency list
    children_map = swc_df.groupby('Parent')['ID'].apply(list).to_dict()
    
    # Set of all bifurcations and terminations
    branching_points = set(bifurcation_ids).union(set(termination_ids))
    
    # Function to trace a branch starting from a bifurcation
    def trace_branch(start_node):
        branch = []
        current = start_node
        while True:
            children = children_map.get(current, [])
            if len(children) != 1:
                break
            next_node = children[0]
            branch.append(next_node)
            if next_node in branching_points:
                break
            current = next_node
        return branch
    
    # Find all branches starting from bifurcation points
    branches = []
    for bif_id in bifurcation_ids:
        for child in children_map.get(bif_id, []):
            if child == soma_id:
                continue  # Skip if for some reason child is soma (shouldn't happen)
            branch = [bif_id] + trace_branch(child)
            # Exclude if the branch ends at the soma
            if soma_id not in branch:
                branches.append(branch)
    
    # Count valid branches
    num_branches = len(branches)
    
    # Count the number of compartments per structure ID
    structure_counts = swc_df['Type'].value_counts().sort_index()
    structure_id_counts = list(structure_counts.items())
    
    
    # Always calculate and store both normalized and unnormalized radii

    
        
    
    return (
        G, bifurcation_ids, termination_ids, marker_ids,
        combined_path_distances, combined_euclidean_distances,
        swc_df_combined, len(swc_df), len(bifurcation_ids),
        len(termination_ids), len(marker_ids), primary_branch_count, num_branches, structure_id_counts, standardization_status, standardization_description
    )




def process_swc_directory():
    """
    Processes all SWC files in a directory and returns neuron vectors.
    
    Returns:
    - neuron_vectors: A dictionary where each key is a neuron name and the value contains the processed neuron data.
    """
    # class_dirs = [os.path.join(config.BASE_DIRECTORY, class_dir) for class_dir in os.listdir(config.BASE_DIRECTORY)]
    # all_swc_files = [f for class_dir in class_dirs for f in os.listdir(class_dir) if f.endswith('.swc')]
    # total_files = len(all_swc_files)
    
    base_dir = config.BASE_DIRECTORY

    # Only include subdirectories, skip files like .csv
    class_dirs = [
        os.path.join(base_dir, d)
        for d in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, d))
    ]

    all_swc_files = [
        f for class_dir in class_dirs
        for f in os.listdir(class_dir)
        if f.endswith('.swc')
    ]
    
    total_files = len(all_swc_files)

    neuron_vectors = {}
    neuron_counter = 0
    neuron_class_list = []
    
    standardization_report = []


    # Loop through class directories and process each SWC file
    for class_dir in class_dirs:
        class_name = os.path.basename(class_dir)
        swc_files = [f for f in os.listdir(class_dir) if f.endswith('.swc')]

        for swc_file in swc_files:
            
            # Normalize the filename
            normalized_file_name = swc_file[:-4]  # Remove the '.swc' extension
            normalized_file_name = normalized_file_name.replace('.CNG', '')  # Remove '.CNG' if present
            normalized_file_name = normalized_file_name.replace(' ', '_').replace('.', '_').replace('-', '_')


            filepath = os.path.join(class_dir, swc_file)
            # swc_df = utl.read_swc_file(file_path)
            # neuron_name = swc_file.split('.')[0]
            neuron_name = normalized_file_name
            # # Reformat neuron name exactly as in the CSV cleaning procedure
            # neuron_name = swc_file
    
            # # Remove ".swc" at the end
            # neuron_name = re.sub(r"\.swc$", "", neuron_name)
    
            # # Remove ".CNG"
            # neuron_name = neuron_name.replace(".CNG", "")
    
            # # Replace dot between digits with dash (e.g., 1.2 -> 1-2)
            # neuron_name = re.sub(r"(\d)\.(\d)", r"\1-\2", neuron_name)

            # # Remove any other remaining dots
            # neuron_name = neuron_name.replace(".", "")
            
            # Check if the neuron name is in the exclusion list
            if neuron_name in config.EXCLUDED_NEURONS:
                print(f'Skipping excluded neuron: {neuron_name}')
                continue  # Skip this neuron and move to the next one
            
            neuron_counter += 1
            print(f'Processing neuron {neuron_counter} out of {total_files}')
            
            # Function call for process_neuron with updated return values
            G, bifurcation_ids, termination_ids, marker_ids, combined_path_distances, combined_euclidean_distances, swc_df_combined, total_nodes_count, bifurcation_count, termination_count, marker_count, primary_branch_count,num_branches, structure_id_counts, standardization_status, standardization_description= process_neuron(filepath)
            
            
            
            standardization_report.append({
                                            "filename": swc_file,
                                            "status": standardization_status,
                                            "description": standardization_description
                                        })

            
            neuron3d_dir = os.path.join(config.PLOTS_DIRECTORY, f"Neuron_3d_Images")
            if config.PLOT_NEURON_3D:
                os.makedirs(neuron3d_dir, exist_ok=True)
                # Plot and save the full graph
                # plot_graph_3d(G, f"Full Neuron Graph of {neuron_name}", os.path.join(neuron_dir, f"{neuron_name}_full_graph.png"))
                utl.plot_graph_3d_with_plotly(G, f"Full Neuron Graph of {neuron_name}", os.path.join(neuron3d_dir, f"{class_name}_{neuron_name}_full_graph.html"))
            
            neuron_dir = os.path.join(config.PLOTS_DIRECTORY, f"Neuron_Images")
            if config.PLOT_NEURON:                
                os.makedirs(neuron_dir, exist_ok=True)
                utl.plot_swc(swc_df_combined, neuron_name, class_name, neuron_dir)
                
            radius_col = 'euclidean_distance_to_soma_normalized' if config.NORMALIZE_DESCRIPTOR_RADII else 'euclidean_distance_to_soma'

            # Unnormalized
            bifurcation_radii = swc_df_combined[swc_df_combined['node_type'] == 'bifurcation']['euclidean_distance_to_soma'].tolist()
            termination_radii = swc_df_combined[swc_df_combined['node_type'] == 'termination']['euclidean_distance_to_soma'].tolist()
            combined_radii = bifurcation_radii + termination_radii
            
            # Normalized
            bifurcation_radii_normalized = swc_df_combined[swc_df_combined['node_type'] == 'bifurcation']['euclidean_distance_to_soma_normalized'].tolist()
            termination_radii_normalized = swc_df_combined[swc_df_combined['node_type'] == 'termination']['euclidean_distance_to_soma_normalized'].tolist()
            combined_radii_normalized = bifurcation_radii_normalized + termination_radii_normalized
            
            
            primary_branch_radii = swc_df_combined[swc_df_combined['Parent'] == 1]['euclidean_distance_to_soma'].tolist()
            primary_branch_radii_normalized = swc_df_combined[swc_df_combined['Parent'] == 1]['euclidean_distance_to_soma_normalized'].tolist()

            # Store the results in the neuron_vectors dictionary
            neuron_vectors[neuron_name] = {
                'class': class_name,
                'swc_df': swc_df_combined,
                'bifurcation_ids': bifurcation_ids,
                'termination_ids': termination_ids,
                'bifurcation_radii' : bifurcation_radii,
                'termination_radii' :termination_radii,
                'combined_radii'  : combined_radii ,
                'primary_branch_radii' : primary_branch_radii,
                'bifurcation_radii_normalized' : bifurcation_radii_normalized,
                'termination_radii_normalized' : termination_radii_normalized,
                'combined_radii_normalized' :  combined_radii_normalized,    
                'primary_branch_radii_normalized' : primary_branch_radii_normalized,             
                'marker_ids': marker_ids,
                'path_distances': combined_path_distances,
                'euclidean_distances': combined_euclidean_distances,
                'neuron_name': neuron_name,
                'total_nodes': total_nodes_count,
                'bifurcation_count': bifurcation_count,
                'termination_count': termination_count,
                'primary_branch_count': primary_branch_count,
                'num_branches': num_branches,
                'structure_id_counts' : structure_id_counts,
                'neuron_as_graph': G
                
            }
            
            # Append the neuron name and class name to the list for later conversion to DataFrame
            neuron_class_list.append({'neuron_name': neuron_name, 'class': class_name})
            
            

    # Convert the list of dictionaries into a DataFrame
    neuron_class_df = pd.DataFrame(neuron_class_list)
    
    swc_report_df = pd.DataFrame(standardization_report)
    
    unique_classes = neuron_class_df['class'].unique()
    color_mapping = {cls: config.COLORS_PALETTE[i % len(config.COLORS_PALETTE)] for i, cls in enumerate(unique_classes)}
    neuron_class_df['class_color'] = neuron_class_df['class'].map(color_mapping)
    
    plot_swc_file_stats(neuron_vectors)

        
    return neuron_vectors, neuron_class_df, swc_report_df


# def plot_swc_file_stats(neuron_data):

#     rows = []
    
#     for neuron_name, data in neuron_data.items():
#         counts = dict(data["structure_id_counts"])
#         counts["neuron_name"] = neuron_name
#         rows.append(counts)
    
#     df_structure_counts = pd.DataFrame(rows).set_index("neuron_name")
    
#     import matplotlib.pyplot as plt
#     # Define colors for NaN, zero, and non-zero
#     # Example: gray for NaN, red for zero, green for non-zero
#     colors = ['gray', 'red', 'green']
#     custom_cmap = mcolors.ListedColormap(colors)
    
#     # Create a new matrix to encode the three categories:
#     # 0 - NaN
#     # 1 - Zero
#     # 2 - Non-zero
#     category_matrix = np.zeros_like(df_structure_counts.values, dtype=int)
    
#     # Apply encoding
#     for i in range(df_structure_counts.shape[0]):
#         for j in range(df_structure_counts.shape[1]):
#             val = df_structure_counts.iat[i, j]
#             if pd.isna(val):
#                 category_matrix[i, j] = 0  # NaN
#             elif val == 0:
#                 category_matrix[i, j] = 1  # Zero
#             else:
#                 category_matrix[i, j] = 2  # Non-zero
    
#     # Plot with custom colormap
#     plt.figure(figsize=(8, 2 + 0.4 * len(df_structure_counts)))
#     plt.imshow(category_matrix, cmap=custom_cmap, vmin=0, vmax=2)
#     cbar = plt.colorbar(label='Category')
#     cbar.set_ticks([0, 1, 2])
#     cbar.set_ticklabels(['NaN', 'Zero', 'Non-zero'])
    
#     plt.title("Neuron Structure ID Counts with NaN, Zero, and Non-zero Highlighting")
#     plt.xticks(ticks=range(df_structure_counts.shape[1]), labels=df_structure_counts.columns)
#     plt.yticks(ticks=range(df_structure_counts.shape[0]), labels=df_structure_counts.index)
#     plt.tight_layout()
#     filename = os.path.join(config.PLOTS_DIRECTORY, f"swc_file_stats.png")
#     plt.savefig(filename, dpi=300)
#     plt.close()



def plot_swc_file_stats(neuron_data):
    import os
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import pandas as pd
    import numpy as np

    COMPONENT_TYPE_MAP = {
        'undefined' : 0,
        'soma':1,
        'axon': 2,
        'basal': 3,
        'apical': 4,
        'other' : 5
    }
    REVERSE_COMPONENT_MAP = {v: k.capitalize() + f"({v})" for k, v in COMPONENT_TYPE_MAP.items()}

    rows = []
    for neuron_name, data in neuron_data.items():
        counts = dict(data["structure_id_counts"])
        counts["neuron_name"] = neuron_name
        rows.append(counts)

    df_structure_counts = pd.DataFrame(rows).set_index("neuron_name")

    # Map column labels
    mapped_columns = [REVERSE_COMPONENT_MAP.get(col, f"Other({col})") for col in df_structure_counts.columns]
    df_structure_counts.columns = mapped_columns

    # Fill matrix for plotting
    values_matrix = df_structure_counts.fillna(-1).values.astype(float)

    # Color setup
    gradient = plt.cm.Blues
    nan_color = 'gray'
    zero_color = 'red'
    cmap_colors = [nan_color, zero_color] + [gradient(i) for i in np.linspace(0.3, 1, 100)]
    custom_cmap = mcolors.ListedColormap(cmap_colors)
    bounds = [-1.5, -0.1, 0.1] + list(np.linspace(0.1, values_matrix.max() + 1, 100))
    norm = mcolors.BoundaryNorm(bounds, custom_cmap.N)

    # Replace values: NaN -> -1, 0 -> 0, else keep
    # plot_matrix = np.where(df_structure_counts.isna(), -1, df_structure_counts.fillna(0))
    plot_matrix = df_structure_counts.copy().values

    # Plot
    plt.figure(figsize=(14, 2 + 0.6 * len(df_structure_counts)))
    im = plt.imshow(plot_matrix, cmap=custom_cmap, norm=norm)

    # Annotate with count values
    for i in range(plot_matrix.shape[0]):
        for j in range(plot_matrix.shape[1]):
            val = df_structure_counts.iat[i, j]
            if pd.isna(val):
                label = "NaN"
            elif val == 0:
                label = "0"
            else:
                label = f"{int(val)}"
            plt.text(j, i, label, ha='center', va='center', fontsize=9, color='black')

    plt.colorbar(im, label='Count Intensity')
    plt.xticks(ticks=range(df_structure_counts.shape[1]), labels=df_structure_counts.columns, rotation=45)
    plt.yticks(ticks=range(df_structure_counts.shape[0]), labels=df_structure_counts.index)
    plt.title("Neuron Structure ID Counts with Custom Labels and Gradient")
    plt.tight_layout()

    filename = os.path.join(config.PLOTS_DIRECTORY, f"swc_file_stats.png")
    plt.savefig(filename, dpi=300)
    plt.close()
    
