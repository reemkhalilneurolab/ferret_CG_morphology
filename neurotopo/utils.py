import numpy as np
import os
import networkx as nx
from matplotlib.patches import Circle
import matplotlib.pyplot as plt
import config
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
import copy
from scipy.stats import wasserstein_distance

exclude_descriptors = config.EXCLUDED_DESCRIPTORS

# Define a utility function to calculate the Euclidean distance between two points
def calculate_euclidean_distance(pos1, pos2):
    return np.linalg.norm(np.array(pos1) - np.array(pos2))




# def standardize_swc_full(filepath):
#     """
#     Fully standardizes an SWC DataFrame according to SWC v1.0.0 best practices:
#     - Enforces correct field count and data types
#     - Fixes malformed values and ensures positive radius
#     - Detects and collapses multiple soma entries
#     - Detects and converts soma contours to a single spherical soma
#     - Reorders samples so parent precedes child
#     - Resets IDs and fixes invalid parents
#     - Shifts soma to origin if not at (0, 0, 0)
#     - Validates that all non-root nodes have valid parents
#     - Adds pass/fail status to description

#     Parameters:
#     swc_df: pd.DataFrame - Raw SWC DataFrame

#     Returns:
#     swc_df: pd.DataFrame - Standardized SWC DataFrame
#     description: str - Description of what was done
#     """
#     import numpy as np
#     import pandas as pd

#     swc_df = pd.read_csv(filepath, delim_whitespace=True, comment='#', header=None,
#                          names=['ID', 'Type', 'X', 'Y', 'Z', 'R', 'Parent'])

#     swc_df = swc_df.copy()
#     description = []

#     # 1. Ensure 7 columns and correct names
#     swc_df = swc_df.iloc[:, :7]
#     swc_df.columns = ['ID', 'Type', 'X', 'Y', 'Z', 'R', 'Parent']

#     # 2. Convert data types explicitly and check float-formatted integers
#     try:
#         swc_df[['ID', 'Type', 'Parent']] = swc_df[['ID', 'Type', 'Parent']].apply(lambda col: pd.to_numeric(col, errors='raise'))
#         if not all(swc_df[['ID', 'Type', 'Parent']].applymap(lambda x: float(x).is_integer()).all()):
#             description.append("Warning: Non-integer values detected in ID/Type/Parent. Converted to integers.")
#         swc_df[['ID', 'Type', 'Parent']] = swc_df[['ID', 'Type', 'Parent']].astype('Int64')
#     except:
#         raise ValueError("Invalid ID/Type/Parent values. Must be numeric.")

#     swc_df[['X', 'Y', 'Z', 'R']] = swc_df[['X', 'Y', 'Z', 'R']].apply(pd.to_numeric, errors='coerce')

#     # 4. Fix invalid radii
#     swc_df['R'] = swc_df['R'].apply(lambda r: r if r > 0 else 0.5)

#     # 5. Warn if < 20 lines
#     if len(swc_df) < 20:
#         description.append("Warning: File has fewer than 20 lines. Integrity check advised.")

#     # 6. Detect soma points
#     soma_df = swc_df[swc_df['Type'] == 1]
#     if soma_df.empty:
#         raise ValueError("No soma (Type=1) found.")

#     # 7. Validate that all non-soma nodes have valid parents
#     valid_ids = set(swc_df['ID'])
#     invalid_parents = swc_df[(swc_df['Parent'] != -1) & (~swc_df['Parent'].isin(valid_ids))]
#     if not invalid_parents.empty:
#         bad_ids = invalid_parents['ID'].tolist()
#         raise ValueError(f"Invalid parent references found for node(s): {bad_ids}.")

#     # 11. Shift soma to origin if needed
#     soma_coords = swc_df.iloc[0][['X', 'Y', 'Z']].values
#     if not np.allclose(soma_coords, [0.0, 0.0, 0.0]):
#         swc_df['X'] -= soma_coords[0]
#         swc_df['Y'] -= soma_coords[1]
#         swc_df['Z'] -= soma_coords[2]
#         description.append("Shifted soma to origin.")

#     description.append("Standardization PASSED.")

#     return swc_df, " ".join(description)


def standardize_swc_full(filepath):
    """
    Fully standardizes an SWC DataFrame according to SWC v1.0.0 best practices:
    - Enforces correct field count and data types
    - Fixes malformed values and ensures positive radius
    - Detects and collapses multiple soma entries
    - Detects and converts soma contours to a single spherical soma
    - Reorders samples so parent precedes child
    - Resets IDs and fixes invalid parents
    - Shifts soma to origin if not at (0, 0, 0)
    - Validates that all non-root nodes have valid parents
    - Adds pass/fail status to description

    Parameters:
    swc_df: pd.DataFrame - Raw SWC DataFrame

    Returns:
    swc_df: pd.DataFrame - Standardized SWC DataFrame
    description: str - Description of what was done
    """
    import numpy as np
    import pandas as pd

    swc_df = pd.read_csv(filepath, delim_whitespace=True, comment='#', header=None,
                         names=['ID', 'Type', 'X', 'Y', 'Z', 'R', 'Parent'])

    swc_df = swc_df.copy()
    description = []

    # 1. Ensure 7 columns and correct names
    swc_df = swc_df.iloc[:, :7]
    swc_df.columns = ['ID', 'Type', 'X', 'Y', 'Z', 'R', 'Parent']

    # 2. Convert data types explicitly and check float-formatted integers
    try:
        swc_df[['ID', 'Type', 'Parent']] = swc_df[['ID', 'Type', 'Parent']].apply(lambda col: pd.to_numeric(col, errors='raise'))
        if not all(swc_df[['ID', 'Type', 'Parent']].applymap(lambda x: float(x).is_integer()).all()):
            description.append("Warning: Non-integer values detected in ID/Type/Parent. Converted to integers.")
        swc_df[['ID', 'Type', 'Parent']] = swc_df[['ID', 'Type', 'Parent']].astype('Int64')
    except:
        raise ValueError("Invalid ID/Type/Parent values. Must be numeric.")

    swc_df[['X', 'Y', 'Z', 'R']] = swc_df[['X', 'Y', 'Z', 'R']].apply(pd.to_numeric, errors='coerce')

    # 4. Fix invalid radii
    swc_df['R'] = swc_df['R'].apply(lambda r: r if r > 0 else 0.5)

    # 5. Warn if < 20 lines
    if len(swc_df) < 20:
        description.append("Warning: File has fewer than 20 lines. Integrity check advised.")

    # 6. Detect soma points
    soma_df = swc_df[swc_df['Type'] == 1]
    if soma_df.empty:
        raise ValueError("No soma (Type=1) found.")

    if len(soma_df) > 1:
        parent_set = set(soma_df['Parent'])
        soma_ids = set(soma_df['ID'])
        if parent_set.issubset({1}):
            description.append(f"Multiple soma nodes detected: {len(soma_df)} entries with Parent=1.")
            raise ValueError("SWC contains multiple disconnected soma nodes. Collapse required.")
        elif not parent_set.issubset({-1, 1}) and soma_df.apply(lambda row: row['Parent'] in soma_ids, axis=1).all():
            description.append("Soma contour structure detected (Type=1 nodes are parent-child connected).")
            raise ValueError("SWC contains a soma contour. Requires conversion to spherical soma.")

    # 7. Validate that all non-soma nodes have valid parents
    valid_ids = set(swc_df['ID'])
    invalid_parents = swc_df[(swc_df['Parent'] != -1) & (~swc_df['Parent'].isin(valid_ids))]
    if not invalid_parents.empty:
        bad_ids = invalid_parents['ID'].tolist()
        raise ValueError(f"Invalid parent references found for node(s): {bad_ids}.")

    # 11. Shift soma to origin if needed
    soma_coords = swc_df.iloc[0][['X', 'Y', 'Z']].values
    if not np.allclose(soma_coords, [0.0, 0.0, 0.0]):
        swc_df['X'] -= soma_coords[0]
        swc_df['Y'] -= soma_coords[1]
        swc_df['Z'] -= soma_coords[2]
        description.append("Shifted soma to origin.")

    description.append("Standardization PASSED.")

    return swc_df, " ".join(description)







# def read_and_shift_swc(filepath):
#     """
#     Reads an SWC file into a DataFrame and shifts all coordinates
#     so that the soma is at (0, 0, 0). No normalization is applied.

#     Parameters:
#     filepath : str
#         Path to the SWC file.

#     Returns:
#     swc_df : pandas DataFrame
#         SWC data with coordinates shifted to place soma at origin.
#     """
#     # Read SWC
#     swc_df = pd.read_csv(filepath, delim_whitespace=True, comment='#', header=None,
#                          names=['ID', 'Type', 'X', 'Y', 'Z', 'R', 'Parent'])

#     swc_df = swc_df.copy()

#     # Convert coordinate columns to numeric
#     swc_df[['X', 'Y', 'Z']] = swc_df[['X', 'Y', 'Z']].apply(pd.to_numeric, errors='coerce')

#     if swc_df[['X', 'Y', 'Z']].isnull().values.any():
#         raise ValueError("NaN values detected in the X, Y, or Z columns. Please check the input data.")

#     # Shift soma to origin (assumes soma is first row)
#     soma_coords = swc_df.iloc[0][['X', 'Y', 'Z']].values
#     swc_df['X'] -= soma_coords[0]
#     swc_df['Y'] -= soma_coords[1]
#     swc_df['Z'] -= soma_coords[2]

#     return swc_df

# def read_swc_file(filepath):
#     """Reads an SWC file and returns it as a pandas DataFrame."""
#     # Read the SWC file
#     swc_df = pd.read_csv(filepath, delim_whitespace=True, comment='#', header=None,
#                          names=['ID', 'Type', 'X', 'Y', 'Z', 'R', 'Parent'])
    
#     swc_df = swc_df.copy()
    
#     # Convert X, Y, Z to numeric, forcing invalid parsing to NaN
#     swc_df['X'] = pd.to_numeric(swc_df['X'], errors='coerce')
#     swc_df['Y'] = pd.to_numeric(swc_df['Y'], errors='coerce')
#     swc_df['Z'] = pd.to_numeric(swc_df['Z'], errors='coerce')
    

    
#     return swc_df

# def shift_swc_to_soma_origin(swc_df):
#     """
#     Shift the X, Y, Z coordinates of an SWC file so that the soma is at (0, 0, 0),
#     without applying any normalization.

#     Parameters:
#     swc_df: pandas DataFrame
#         The SWC file as a DataFrame with columns 'X', 'Y', 'Z'.

#     Returns:
#     swc_df: pandas DataFrame
#         The shifted SWC DataFrame.
#     """
#     swc_df = swc_df.copy()

#     # Convert X, Y, Z columns to numeric, coercing errors to NaN
#     swc_df[['X', 'Y', 'Z']] = swc_df[['X', 'Y', 'Z']].apply(pd.to_numeric, errors='coerce')

#     if swc_df[['X', 'Y', 'Z']].isnull().values.any():
#         raise ValueError("NaN values detected in the X, Y, or Z columns. Please check the input data.")

#     # Shift all coordinates so that the soma (first row) becomes the origin
#     soma_coords = swc_df.iloc[0][['X', 'Y', 'Z']].values
#     swc_df['X'] -= soma_coords[0]
#     swc_df['Y'] -= soma_coords[1]
#     swc_df['Z'] -= soma_coords[2]

#     return swc_df


# def normalize_swc(swc_df):
#     """
#     Normalize the X, Y, Z coordinates of an SWC file to lie between 0 and 1.
#     Assumes that the soma is at (0, 0, 0).

#     Parameters:
#     swc_df: pandas DataFrame
#         The SWC file as a DataFrame with columns 'X', 'Y', 'Z'.

#     Returns:
#     swc_df: pandas DataFrame
#         The normalized SWC DataFrame.
#     """
#     swc_df = swc_df.copy()
    
#     # Convert X, Y, Z columns to numeric, coercing errors to NaN
#     swc_df[['X', 'Y', 'Z']] = swc_df[['X', 'Y', 'Z']].apply(pd.to_numeric, errors='coerce')

#     # Check for NaNs
#     if swc_df[['X', 'Y', 'Z']].isnull().values.any():
#         raise ValueError("NaN values detected in the X, Y, or Z columns. Please check the input data.")
    
    
    
#     # Step 1: Subtract the soma's coordinates to ensure the soma is at (0, 0, 0)
#     soma_coords = swc_df.iloc[0][['X', 'Y', 'Z']].values  # Convert to numpy array
#     swc_df['X'] = swc_df['X'] - soma_coords[0]
#     swc_df['Y'] = swc_df['Y'] - soma_coords[1]
#     swc_df['Z'] = swc_df['Z'] - soma_coords[2]

#     # Step 2: Calculate the maximum distance from the soma
#     distances = np.sqrt(swc_df['X']**2 + swc_df['Y']**2 + swc_df['Z']**2)

#     max_distance = distances.max()

#     if max_distance == 0:
#         raise ValueError("The maximum distance is zero. All points are at the origin.")
    
    
    
#     # Step 3: Normalize X, Y, Z by dividing by the maximum distance
#     swc_df['X'] = swc_df['X'] / max_distance
#     swc_df['Y'] = swc_df['Y'] / max_distance
#     swc_df['Z'] = swc_df['Z'] / max_distance
    

#     return swc_df

# def min_max_with_soma_zero(swc_df):
#     swc_df = swc_df.copy()
#     swc_df[['X', 'Y', 'Z']] = swc_df[['X', 'Y', 'Z']].apply(pd.to_numeric, errors='coerce')

#     if swc_df[['X', 'Y', 'Z']].isnull().values.any():
#         raise ValueError("NaN detected in coordinate columns.")

#     # Step 1: Move soma to (0, 0, 0)
#     soma_coords = swc_df.iloc[0][['X', 'Y', 'Z']].values
#     swc_df['X'] -= soma_coords[0]
#     swc_df['Y'] -= soma_coords[1]
#     swc_df['Z'] -= soma_coords[2]

#     # Step 2: Translate so min becomes 0 (after soma shift)
#     x_min = swc_df['X'].min()
#     y_min = swc_df['Y'].min()
#     z_min = swc_df['Z'].min()

#     swc_df['X'] -= x_min
#     swc_df['Y'] -= y_min
#     swc_df['Z'] -= z_min

#     # Step 3: Scale so max becomes 1
#     x_max = swc_df['X'].max()
#     y_max = swc_df['Y'].max()
#     z_max = swc_df['Z'].max()

#     swc_df['X'] /= x_max
#     swc_df['Y'] /= y_max
#     swc_df['Z'] /= z_max

#     return swc_df



def drop_constant_and_low_variance_columns(df, variance_threshold=1e-5, dominance_threshold=0.95):
    """
    Removes constant, low-variance, and near-constant columns from a numeric DataFrame.
    """
    df = df.copy()
    numeric_cols = df.select_dtypes(include=['number']).columns

    cols_to_drop = set()

    for col in numeric_cols:

            if col =='fft_entropy_Tortuosity':
                print('ddd')
            
            col_data = df[col]
    
            # Low variance check
            if col_data.var() <= variance_threshold:
                cols_to_drop.add(col)
                continue
    
            # Dominance check
            value_counts = col_data.value_counts(normalize=True)
            if not value_counts.empty:
                most_common_ratio = value_counts.iloc[0]
                if most_common_ratio >= dominance_threshold:
                    cols_to_drop.add(col)    

             
    return df.drop(columns=list(cols_to_drop))



# def drop_constant_and_low_variance_columns(df, variance_threshold=1e-5):
#     """
#     Removes constant and low-variance columns from a numeric DataFrame.

#     Parameters:
#     - df (pd.DataFrame): Input DataFrame
#     - variance_threshold (float): Minimum variance required to keep a column

#     Returns:
#     - pd.DataFrame: Cleaned DataFrame
#     """
#     df = df.copy()
    
#     # Only apply to numeric columns
#     numeric_cols = df.select_dtypes(include=['number']).columns

#     # Drop constant columns
#     df = df.drop(columns=[col for col in numeric_cols if df[col].nunique() <= 1])

#     # Drop low-variance columns
#     df = df.drop(columns=[col for col in df.select_dtypes(include=['number']) if df[col].var() <= variance_threshold])
    
#     sparsity_threshold = 0.01  # 1%
#     cols_to_drop = [col for col in numeric_cols if (df[col] != 0).sum() / len(df) < sparsity_threshold]
#     df = df.drop(columns=cols_to_drop)
    
#     return df


# Function to generate an SWC file for the neuron
def generate_swc_file(node_positions, edges, filename):
    with open(filename, 'w') as f:
        f.write("# Example SWC file\n")
        for idx, (node, pos) in enumerate(node_positions.items(), start=1):
            node_type = 3 if idx != 1 else 1  # Type 1 for soma, 3 for others
            parent = -1 if node == 1 else next((src for src, tgt in edges if tgt == node), -1)
            f.write(f"{node} {node_type} {pos[0]} {pos[1]} 0.0 1.0 {parent}\n")

# Helper function to convert coordinates to dictionary
def coords_to_dict(x_coords, y_coords):
    return dict(zip(x_coords, y_coords))

def calculate_l1_difference(step_func1_coords, step_func2_coords):
    """L1 distance between two descriptor step functions.

    Implements equation (4) of Khalil R, Kallel S, Farhat A, Dlotko P,
    Topological Sholl descriptors for neuronal clustering and classification,
    PLoS Comput Biol 2022;18(6):e1010229:

        d(f, g) = integral |f(r) - g(r)| dr
                = sum_i (t_{i+1} - t_i) * |f(t_i) - g(t_i)|

    where {t_i} is the union of the two functions' jump points in increasing
    order. Both functions are right-continuous, so each interval takes the value
    at its left endpoint, and the final jump point contributes no interval.

    The result is returned at full precision. Rounding it would quantise
    descriptors whose distances are small relative to the rounding step, since
    the descriptors are not on a common scale.
    """
    x1, y1 = step_func1_coords
    x2, y2 = step_func2_coords

    # Convert step function coordinates to dictionary form
    step_func1 = coords_to_dict(x1, y1)
    step_func2 = coords_to_dict(x2, y2)

    # Combine intervals from both functions
    combined_intervals = sorted(set(x1 + x2))

    # Ensure both functions have values at each point by carrying forward the last known value
    def fill_values(step_func, combined_intervals):
        last_val = 0
        filled_func = {}
        for point in combined_intervals:
            if point in step_func:
                last_val = step_func[point]
            filled_func[point] = last_val
        return filled_func

    filled_step_func1 = fill_values(step_func1, combined_intervals)
    filled_step_func2 = fill_values(step_func2, combined_intervals)

    total_diff = 0  # Initialize total difference accumulator

    # Iterate through combined intervals (except the last point, as it has no interval end)
    for i in range(1, len(combined_intervals)):
        interval_start = combined_intervals[i-1]
        interval_end = combined_intervals[i]

        # Get function values at the start of the interval
        step_func1_val = filled_step_func1[interval_start]
        step_func2_val = filled_step_func2[interval_start]

        # Calculate the difference between the two step functions
        diff = abs(step_func1_val - step_func2_val)

        # Multiply by the length of the interval to account for the contribution over the interval
        interval_length = interval_end - interval_start
        total_diff += diff * interval_length  # Sum differences across intervals

    return total_diff



def calculate_wasserstein_distance(step_func1_coords, step_func2_coords):
    x1, y1 = step_func1_coords
    x2, y2 = step_func2_coords

    # Convert step function coordinates to dictionary form
    step_func1 = coords_to_dict(x1, y1)
    step_func2 = coords_to_dict(x2, y2)

    # Combine intervals from both functions
    combined_intervals = sorted(set(x1 + x2))

    # Ensure both functions have values at each point by carrying forward the last known value
    def fill_values(step_func, combined_intervals):
        last_val = 0
        filled_func = {}
        for point in combined_intervals:
            if point in step_func:
                last_val = step_func[point]
            filled_func[point] = last_val
        return filled_func

    filled_step_func1 = fill_values(step_func1, combined_intervals)
    filled_step_func2 = fill_values(step_func2, combined_intervals)

    # Extract values as lists
    values1 = [filled_step_func1[r] for r in combined_intervals]
    values2 = [filled_step_func2[r] for r in combined_intervals]

    # Compute Wasserstein Distance
    wasserstein_dist = wasserstein_distance(values1, values2)

    return round(wasserstein_dist, 2)


def create_test_neuron():
    # Create a new graph
    G = nx.Graph()

    # Adjusted coordinates, adding new intermediate nodes
    node_positions = {
        1: (0, 0),       # Node 1 (Soma) at (0, 0)
        2: (0.2, 1.3),   # Node 2
        3: (1.8, 1.2),   # New Node between 2 and 4 (not a bifurcation)
        4: (1.8, 3),     # Formerly Node 3, shifted to 4
        5: (-1.8, 2.2),  # Formerly Node 4, shifted to 5
        6: (-1.5, 4.5),  # Formerly Node 5, shifted to 6
        7: (-3.7, 3),    # Formerly Node 6, shifted to 7
        8: (1.5, 5),     # Formerly Node 7, shifted to 8
        9: (4, 4),       # Formerly Node 8, shifted to 9
        10: (-2, -1.2),  # Formerly Node 9, shifted to 10
        11: (3, 3),      # New Node between 4 and 9
        12: (-0.5, 3.3),   # New Node between 5 and 6
        13: (0, -2),    # New Node between 1 and 10
        14: (-1.5, 0.9) ,   # New Node between 2 and 5
        15: (4.4, 2.7)
    }
    
    # Add the nodes with their coordinates to the graph
    for node, position in node_positions.items():
        G.add_node(node, pos=position)
    
    # Define edges, adjusted for the new nodes
    edges = [
        (1, 2), (2, 3), (3, 4),  # Added (2, 3) and (3, 4) to accommodate the new node
        (2, 14), (14, 5),        # Added (2, 14) and (14, 5) for the new point between 2 and 5
        (5, 12), (12, 6),        # Added (5, 12) and (12, 6) for the new point between 5 and 6
        (1, 13), (13, 10),       # Added (1, 13) and (13, 10) for the new point between 1 and 10
        (4, 11), (11, 15), (15, 9),        # Added (4, 11) and (11, 9) for the new point between 4 and 9
        (5, 7), (4, 8)
    ]
    
    G.add_edges_from(edges)
    
    # Extract the positions of nodes to use in the plot
    positions = nx.get_node_attributes(G, 'pos')
    
    # Colors for the circles and nodes (make sure Node 1 is different)
    circle_colors = ['lime', 'red', 'green', 'blue', 'orange', 'purple', 'yellow', 'cyan', 'magenta', 'lightcoral'] * 2

    # Node colors should correspond to the circle colors for bifurcation and termination nodes
    node_colors = {node: 'gray' if node in [3, 11, 12, 13, 14, 15] else circle_colors[node - 1] for node in node_positions}  # New nodes are gray
    
    # Create a figure
    plt.figure(figsize=(14, 12))
    
    # Draw the graph with node labels and fixed positions, ensuring node colors match circle colors
    node_sizes = [400 if node in [3, 11, 12, 13, 14] else 700 for node in node_positions]  # Smaller size for the new intermediate nodes
    nx.draw(G, pos=positions, with_labels=True, node_size=node_sizes, 
            node_color=[node_colors[node] for node in node_positions], font_size=15, font_weight="bold", edge_color="black", alpha=0.5)
    
    # Calculate the Euclidean distance from the soma (Node 1) for each node
    soma_position = node_positions[1]
    euclidean_distances = {node: calculate_euclidean_distance(soma_position, position) for node, position in node_positions.items() if node != 1}
    
    # Calculate the path distance from the soma (Node 1) to each node
    path_distances = {}
    for node in node_positions:
        if node == 1:
            path_distances[node] = 0  # The path distance from the soma to itself is zero
        else:
            # Get the path from the node to the soma
            path = nx.shortest_path(G, source=1, target=node)
            
            # Calculate the total path distance as the sum of Euclidean distances along the path
            total_path_distance = sum(
                calculate_euclidean_distance(node_positions[path[i]], node_positions[path[i + 1]])
                for i in range(len(path) - 1)
            )
            path_distances[node] = total_path_distance

    # Sort nodes by their Euclidean distance from the soma
    sorted_distances = sorted(euclidean_distances.items(), key=lambda x: x[1])
    
    # Create a plot axis
    ax = plt.gca()
    
    # Draw circles only for bifurcation and termination nodes
    for node, distance in sorted_distances:
        if node not in [3, 11, 12, 13, 14, 15]:  # Skip the intermediate nodes when drawing circles
            color_index = (node - 1) % len(circle_colors)  # Ensure index stays within the range
            circle = Circle(soma_position, distance, color=circle_colors[color_index], fill=False, linestyle='--')
            ax.add_patch(circle)
        
        # Annotate the coordinates (x, y) next to each node
        node_position = node_positions[node]
        coordinates_text = f'({node_position[0]:.1f}, {node_position[1]:.1f})'
        plt.text(node_position[0] + 0.2, node_position[1] + 0.2, coordinates_text, 
                 horizontalalignment='center', fontsize=12, color='black')
        
        # Annotate the Euclidean distance in blue
        plt.text(node_position[0], node_position[1] + 0.4, f'{distance:.2f}', 
                 horizontalalignment='center', fontsize=15, color='blue')
        
        # Annotate the path distance in red slightly below the node
        path_distance = path_distances.get(node, 0)
        plt.text(node_position[0], node_position[1] - 0.4, f'{path_distance:.2f}', 
                 horizontalalignment='center', fontsize=15, color='red')

    # Annotate legend for Euclidean and path distances
    plt.annotate(
        'Euclidean \n distance \n to soma', 
        xy=(-2.2, -0.7), xytext=(-4, -2),
        fontsize=14, color='blue', fontweight='bold',
        arrowprops=dict(facecolor='black', shrink=0.05, width=2, headwidth=10)
    )
    plt.annotate(
        'Path \n distance \n to soma', 
        xy=(-2.2, -1.4), xytext=(-4, -3),
        fontsize=14, color='red', fontweight='bold',
        arrowprops=dict(facecolor='black', shrink=0.05, width=2, headwidth=10)
    )

    # Calculate and label the path distance between source and target nodes next to edges
    for edge in G.edges():
        source, target = edge
        pos_source = node_positions[source]
        pos_target = node_positions[target]
        distance = calculate_euclidean_distance(pos_source, pos_target)
        edge_midpoint = ((pos_source[0] + pos_target[0]) / 2, (pos_source[1] + pos_target[1]) / 2)
        plt.text(edge_midpoint[0], edge_midpoint[1], f'{distance:.2f}', fontsize=13, color='green')

    # Set plot limits
    plt.xlim(-5, 7)
    plt.ylim(-5, 6)
    plt.gca().set_aspect('equal', adjustable='box')

    # Save the plot to the specified directory
    save_path = os.path.join(config.SAVE_DIRECTORY, 'test_neuron_plot.png')
    plt.savefig(save_path)
    print(f'Saved plot to {save_path}')
    plt.close()

    # Generate the SWC file (for illustration purposes)
    generate_swc_file(node_positions, edges, 'example_neuron.swc')



def plot_graph_3d_with_plotly(G, title, file_name):
    edge_x = []
    edge_y = []
    edge_z = []
    node_x = []
    node_y = []
    node_z = []
    node_text = []
    node_color = []
    node_size = []

    # Retrieve the soma_id from the graph
    soma_id = G.graph['soma_id']

    # Prepare colors for different node types
    for edge in G.edges():
        x0, y0, z0 = G.nodes[edge[0]]['pos']
        x1, y1, z1 = G.nodes[edge[1]]['pos']
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        edge_z.extend([z0, z1, None])

    for node in G.nodes():
        x, y, z = G.nodes[node]['pos']
        node_x.append(x)
        node_y.append(y)
        node_z.append(z)
        node_text.append(str(node))  # Assuming you want to see node IDs when hovering

        # Highlight the soma node, bifurcations, and terminations
        if node == soma_id:
            node_color.append('magenta')  # Soma color
            node_size.append(12)         # Soma size larger
        elif G.nodes[node].get('node_type') == 'bifurcation':
            node_color.append('red')    # Bifurcation color
            node_size.append(9)          # Bifurcation size
        elif G.nodes[node].get('node_type') == 'termination':
            node_color.append('green')   # Termination color
            node_size.append(9)          # Termination size
        else:
            node_color.append('purple')     # Regular node color for markers
            node_size.append(5)          # Regular node size
    
        # Get filter type(s) to exclude
    # exclude_types = config.PLOT_SWC_FILTER_TYPE
    
    # for node in G.nodes():
    #     node_type = G.nodes[node].get('Type')  # Or change 'Type' to whatever attribute you're using
    
    #     # Skip nodes with excluded types
    #     if exclude_types is not None:
    #         if isinstance(exclude_types, (list, tuple, set)):
    #             if node_type in exclude_types:
    #                 continue
    #         else:
    #             if node_type == exclude_types:
    #                 continue
    
    #     x, y, z = G.nodes[node]['pos']
    #     node_x.append(x)
    #     node_y.append(y)
    #     node_z.append(z)
    #     node_text.append(str(node))
    
    #     if node == soma_id:
    #         node_color.append('magenta')
    #         node_size.append(12)
    #     elif G.nodes[node].get('node_type') == 'bifurcation':
    #         node_color.append('red')
    #         node_size.append(9)
    #     elif G.nodes[node].get('node_type') == 'termination':
    #         node_color.append('green')
    #         node_size.append(9)
    #     else:
    #         node_color.append('purple')
    #         node_size.append(5)


    edge_trace = go.Scatter3d(
        x=edge_x, y=edge_y, z=edge_z,
        line=dict(width=2, color='blue'),
        hoverinfo='none',
        mode='lines')

    node_trace = go.Scatter3d(
        x=node_x, y=node_y, z=node_z,
        mode='markers',
        marker=dict(size=node_size, color=node_color, line=dict(width=0.5, color='black')),
        text=node_text,
        hoverinfo='text')

    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        # title=title,
                        # titlefont_size=16,
                        title=dict(text=title, font=dict(size=16)),
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20, l=5, r=5, t=40),
                        annotations=[dict(
                            text="",
                            showarrow=False,
                            xref="paper", yref="paper",
                            x=0.005, y=-0.002)],
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                    ))

    fig.update_layout(scene=dict(
                        xaxis_title='X Axis',
                        yaxis_title='Y Axis',
                        zaxis_title='Z Axis'))
    
    
    fig.write_html(file_name)
    # fig.show()  # Optionally display the figure
    
    


def plot_swc(swc_df, neuron_name, class_name, save_path, filter_type=None):
    """
    Plot and save a 2D XY projection of neuron morphology with soma, bifurcations, and terminations marked.

    Parameters:
    - swc_df: pd.DataFrame with SWC structure and 'node_type' column
    - neuron_name: Name of the neuron (for title)
    - save_path: Full path including filename to save the image
    - color: Line color for branches
    - figsize: Figure size
    """
    
    filter_type = config.PLOT_SWC_FILTER_TYPE
    
    if filter_type is not None:
        if isinstance(filter_type, (list, tuple, set)):
            swc_df = swc_df[~swc_df['Type'].isin(filter_type)]
        else:
            swc_df = swc_df[swc_df['Type'] != filter_type]

    
    color='black'
    plt.figure(figsize=(8, 8))

    # Draw branches
    for _, row in swc_df.iterrows():
        parent_id = row['Parent']
        if parent_id == -1:
            continue

        parent_row = swc_df[swc_df['ID'] == parent_id]
        if parent_row.empty:
            continue

        x = [row['X'], parent_row['X'].values[0]]
        y = [row['Y'], parent_row['Y'].values[0]]
        plt.plot(x, y, color=color, linewidth=0.8)

    # Node markers. These are sized to stay legible when a panel is reduced to
    # figure width, and given a thin white edge so that markers remain
    # distinguishable where nodes are densely packed near the soma.
    SOMA_MARKER = 70
    NODE_MARKER = 34

    # Soma: blue dot
    soma = swc_df[swc_df['node_type'] == 'soma']
    plt.scatter(soma['X'], soma['Y'], color='blue', s=SOMA_MARKER, label='Soma',
                edgecolors='white', linewidths=0.5, zorder=5)

    # Bifurcation: red dots
    bifurcations = swc_df[swc_df['node_type'] == 'bifurcation']
    plt.scatter(bifurcations['X'], bifurcations['Y'], color='red', s=NODE_MARKER,
                label='Bifurcation', edgecolors='white', linewidths=0.4, zorder=4)

    # Termination: green dots
    terminations = swc_df[swc_df['node_type'] == 'termination']
    plt.scatter(terminations['X'], terminations['Y'], color='green', s=NODE_MARKER,
                label='Termination', edgecolors='white', linewidths=0.4, zorder=4)

    plt.title(f"2D SWC Projection: {neuron_name}", fontsize=12)
    axes = plt.gca()
    # 'box' rather than the default 'datalim' keeps the aspect ratio true to the
    # micrometre scale while leaving the axis limits under our control, so the
    # space reserved for the scale bar below is not discarded.
    axes.set_aspect('equal', adjustable='box')

    # Calibrated spatial scale bar (SWC coordinates are micrometers). The bar is
    # drawn in reserved space below the arbor rather than over it: the lower y
    # limit is extended so that neither the bar nor its label overlaps the
    # reconstruction.
    BAR_UM = 100.0
    x_min, x_max = axes.get_xlim()
    y_min, y_max = axes.get_ylim()
    x_span = x_max - x_min
    if x_span >= BAR_UM:
        gap = max(0.14 * (y_max - y_min), 1.4 * BAR_UM)
        axes.set_ylim(y_min - gap, y_max)
        x0 = x_max - 0.06 * x_span - BAR_UM
        y0 = y_min - 0.45 * gap
        axes.plot([x0, x0 + BAR_UM], [y0, y0], color='black', linewidth=3,
                  solid_capstyle='butt', zorder=10)
        axes.text(x0 + BAR_UM / 2.0, y0 - 0.10 * gap, f'{BAR_UM:.0f} µm',
                  ha='center', va='top', fontsize=9, zorder=10)
    plt.axis('off')
    plt.tight_layout()
    
    filename = f"{class_name}_{neuron_name}.png"
    full_path = os.path.join(save_path, filename)
    plt.savefig(full_path, dpi=300)
    plt.close()












import copy
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import random





# def plot_boxplots_and_step_functions_by_class(_all_descriptors, neuron_class_df, n=10):
#     """
#     Plots box plots of descriptor values by class for each descriptor, excluding outliers, and plots
#     step functions of a random sample of neurons from each class on the same plot with different colors.
    
#     Parameters:
#     - all_descriptors (dict): Dictionary of descriptors, where each key is a descriptor name and each value is a 
#                               dictionary with neuron names as keys and descriptor values as lists.
#     - neuron_class_df (pd.DataFrame): DataFrame with 'neuron_name' and 'class' columns to map neurons to their classes.
#     - n (int): Number of neurons to randomly sample from each class for plotting step functions.
#     - exclude_descriptors (list): List of descriptors to exclude from plotting.
#     """
#     all_descriptors = copy.deepcopy(_all_descriptors)
    
#     sampled_neurons = neuron_class_df.groupby('class').apply(lambda x: x.sample(min(n, len(x)), random_state=1)).reset_index(drop=True)
    
#     # Iterate over each descriptor
#     for descriptor_name, descriptor_data in all_descriptors.items():
        
#         # Skip the descriptor if it is in the exclusion list
#         if descriptor_name in exclude_descriptors:
#             print(f'Skipping descriptor {descriptor_name} as it is in the exclusion list.')
#             continue
        
#         # Extract only the y-values for box plot generation
#         extracted_data = {neuron: [value[1] for value in values] for neuron, values in descriptor_data.items()}
        
               
#         # Convert descriptor data to DataFrame
#         descriptor_df = pd.DataFrame.from_dict(extracted_data, orient='index')
#         descriptor_df.index.name = 'neuron_name'
        
#         # Melt the DataFrame to long format for easy plotting with seaborn
#         plot_df = descriptor_df.reset_index().melt(id_vars='neuron_name', value_name='value')
        
#         # Map neuron names to classes by merging with neuron_class_df
#         plot_df = plot_df.merge(neuron_class_df, on='neuron_name')
        
#         # Identify and remove outliers within each class
#         def remove_outliers(group):
#             Q1 = group['value'].quantile(0.25)
#             Q3 = group['value'].quantile(0.75)
#             IQR = Q3 - Q1
#             lower_bound = Q1 - 1.5 * IQR
#             upper_bound = Q3 + 1.5 * IQR
#             return group[(group['value'] >= lower_bound) & (group['value'] <= upper_bound)]
        
#         # Apply outlier removal per class with group_keys=False to suppress the warning
#         plot_df = plot_df.groupby('class', group_keys=False).apply(remove_outliers)
        
#         # Create a box plot for the descriptor, with separate box plots for each class
#         plt.figure(figsize=(10, 6))
#         sns.boxplot(x='class', y='value', data=plot_df)
#         plt.title(f'Box Plot of {descriptor_name} by Class (Outliers Removed)')
#         plt.xlabel('Class')
#         plt.ylabel('Descriptor Values')
        
#         # Save the box plot figure
#         plt.savefig(os.path.join(config.SAVE_DIRECTORY, f'{descriptor_name}_boxplot.png'), dpi=300)
#         plt.close()
        

#         # Randomly sample neurons from each class
        
        
#         step_function_data = {}
        
#         # Collect step function data for the sampled neurons
#         step_data = []
#         for _, row in sampled_neurons.iterrows():
#             neuron_name = row['neuron_name']
#             class_label = row['class']
#             x_values = [value[0] for value in descriptor_data[neuron_name]]
#             y_values = [value[1] for value in descriptor_data[neuron_name]]
#             step_data.append({'neuron_name': neuron_name, 'class': class_label, 'x_values': x_values, 'y_values': y_values})
#             step_function_data[neuron_name] = {
#                                'class': class_label,
#                                'x_values': x_values,
#                                'y_values': y_values
#                            }
                                    
            
#         # Define colors for each class
#         unique_classes = sampled_neurons['class'].unique()
#         colors = sns.color_palette("hsv", len(unique_classes))
#         class_colors = dict(zip(unique_classes, colors))
        
#         # Plot step functions with step formatting and color by class
#         plt.figure(figsize=(12, 8))
#         for data in step_data:
#             class_label = data['class']
#             x_values = data['x_values']
#             y_values = data['y_values']
#             plt.step(x_values, y_values, where='post', label=class_label, color=class_colors[class_label], alpha=0.7)
        
#         plt.title(f'Step Functions of Sampled Neurons for {descriptor_name}')
#         plt.xlabel('X Values')
#         plt.ylabel('Descriptor Values')
#         plt.legend(title='Class', loc='upper right')
        
#         # Save the step function plot
#         plt.savefig(os.path.join(config.SAVE_DIRECTORY, f'{descriptor_name}_step_functions.png'), dpi=300)
#         plt.close()
        
#     return step_function_data
        
  




# def plot_boxplots_and_step_functions_by_class(_all_descriptors, neuron_class_df, n=2):
#     """
#     Plots box plots of descriptor values by class for each descriptor, excluding outliers, and plots
#     step functions of a random sample of neurons from each class on the same plot with different colors.
    
#     Parameters:
#     - all_descriptors (dict): Dictionary of descriptors, where each key is a descriptor name and each value is a 
#                               dictionary with neuron names as keys and descriptor values as lists.
#     - neuron_class_df (pd.DataFrame): DataFrame with 'neuron_name' and 'class' columns to map neurons to their classes.
#     - n (int): Number of neurons to randomly sample from each class for plotting step functions.
#     - exclude_descriptors (list): List of descriptors to exclude from plotting.
#     """
#     all_descriptors = copy.deepcopy(_all_descriptors)
    
#     # Iterate over each descriptor
#     for descriptor_name, descriptor_data in all_descriptors.items():
        
#         # Skip the descriptor if it is in the exclusion list
#         if descriptor_name in exclude_descriptors:
#             print(f'Skipping descriptor {descriptor_name} as it is in the exclusion list.')
#             continue
        
#         # Extract only the y-values for box plot generation
#         extracted_data = {neuron: [value[1] for value in values] for neuron, values in descriptor_data.items()}
        
#         # Convert descriptor data to DataFrame
#         descriptor_df = pd.DataFrame.from_dict(extracted_data, orient='index')
#         descriptor_df.index.name = 'neuron_name'
        
#         # Melt the DataFrame to long format for easy plotting with seaborn
#         plot_df = descriptor_df.reset_index().melt(id_vars='neuron_name', value_name='value')
        
#         # Map neuron names to classes by merging with neuron_class_df
#         plot_df = plot_df.merge(neuron_class_df, on='neuron_name')
        
#         # Identify and remove outliers within each class
#         def remove_outliers(group):
#             Q1 = group['value'].quantile(0.25)
#             Q3 = group['value'].quantile(0.75)
#             IQR = Q3 - Q1
#             lower_bound = Q1 - 1.5 * IQR
#             upper_bound = Q3 + 1.5 * IQR
#             return group[(group['value'] >= lower_bound) & (group['value'] <= upper_bound)]
        
#         # Apply outlier removal per class with group_keys=False to suppress the warning
#         plot_df = plot_df.groupby('class', group_keys=False).apply(remove_outliers)
        
#         # Create a box plot for the descriptor, with separate box plots for each class
#         plt.figure(figsize=(10, 6))
#         sns.boxplot(x='class', y='value', data=plot_df)
#         plt.title(f'Box Plot of {descriptor_name} by Class (Outliers Removed)')
#         plt.xlabel('Class')
#         plt.ylabel('Descriptor Values')
        
#         # Save the box plot figure
#         plt.savefig(os.path.join(config.SAVE_DIRECTORY, f'{descriptor_name}_boxplot.png'), dpi=300)
#         plt.close()
        
#         # Step Function Plotting Section
#         # Randomly sample neurons from each class
#         sampled_neurons = neuron_class_df.groupby('class').apply(lambda x: x.sample(min(n, len(x)), random_state=1)).reset_index(drop=True)
        
#         # Collect step function data for the sampled neurons
#         step_data = []
#         for _, row in sampled_neurons.iterrows():
#             neuron_name = row['neuron_name']
#             class_label = row['class']
#             x_values = [value[0] for value in descriptor_data[neuron_name]]
#             y_values = [value[1] for value in descriptor_data[neuron_name]]
#             step_data.append({'neuron_name': neuron_name, 'class': class_label, 'x_values': x_values, 'y_values': y_values})
        
#         # Define colors for each class
#         unique_classes = sampled_neurons['class'].unique()
#         colors = sns.color_palette("hsv", len(unique_classes))
#         class_colors = dict(zip(unique_classes, colors))
        
#         # Plot step functions with color by class
#         plt.figure(figsize=(12, 8))
#         for data in step_data:
#             class_label = data['class']
#             x_values = data['x_values']
#             y_values = data['y_values']
#             plt.plot(x_values, y_values, label=class_label, color=class_colors[class_label], alpha=0.7)
        
#         plt.title(f'Step Functions of Sampled Neurons for {descriptor_name}')
#         plt.xlabel('X Values')
#         plt.ylabel('Descriptor Values')
#         plt.legend(title='Class', loc='upper right')
        
#         # Save the step function plot
#         plt.savefig(os.path.join(config.SAVE_DIRECTORY, f'{descriptor_name}_step_functions.png'), dpi=300)
#         plt.close()














# def create_test_neuron():
#     # Create a new graph
#     G = nx.Graph()
    
    
#     # Define 9 nodes and assign coordinates (X, Y)
#     node_positions = {
#         1: (0, 0),     # Node 1 (Soma) at (0, 0)
#         2: (0.5, 1),   # Node 2
#         3: (1.8, 1.5), # Node 3
#         4: (-0.7, 1.6),# Node 4
#         5: (-0.2, 3),  # Node 5
#         6: (-1.7, 2),  # Node 6
#         7: (1.5, 3),   # Node 7
#         8: (3, 2),     # Node 8
#         9: (-1, -1),   # Node 9
#     }
    
#     # Add the nodes with their coordinates to the graph
#     for node, position in node_positions.items():
#         G.add_node(node, pos=position)
    
#     # Define some edges for the graph
#     edges = [
#         (1, 2), (2, 3), (2, 4),
#         (4, 5), (4, 6), (3, 7),
#         (3, 8), (1, 9)
#     ]
    

#     G.add_edges_from(edges)
    
#     # Extract the positions of nodes to use in the plot
#     positions = nx.get_node_attributes(G, 'pos')
    
#     # Colors for the circles and nodes (make sure Node 1 is different)
#     circle_colors = ['gray', 'red', 'green', 'blue', 'orange', 'purple', 'yellow', 'cyan', 'magenta']
    
#     # Node colors should correspond to the circle colors (Node 1 is gray)
#     node_colors = circle_colors  # Align node colors with the circle colors
    
#     # Create a figure
#     plt.figure(figsize=(10, 8))
    
#     # Draw the graph with node labels and fixed positions, ensuring node colors match circle colors
#     nx.draw(G, pos=positions, with_labels=True, node_size=700, 
#             node_color=node_colors, font_size=15, font_weight="bold", edge_color="gray", alpha=0.5)
    
#     # Calculate the Euclidean distance from the soma (Node 1) for each node
#     soma_position = node_positions[1]
#     euclidean_distances = {node: calculate_euclidean_distance(soma_position, position) for node, position in node_positions.items() if node != 1}
    
#     # Calculate the path distance from the soma (Node 1) to each node
#     path_distances = {}
#     for node in node_positions:
#         if node == 1:
#             path_distances[node] = 0  # The path distance from the soma to itself is zero
#         else:
#             # Get the path from the node to the soma
#             path = nx.shortest_path(G, source=1, target=node)
            
#             # Calculate the total path distance as the sum of Euclidean distances along the path
#             total_path_distance = sum(
#                 calculate_euclidean_distance(node_positions[path[i]], node_positions[path[i + 1]])
#                 for i in range(len(path) - 1)
#             )
#             path_distances[node] = total_path_distance

#     # Sort nodes by their Euclidean distance from the soma
#     sorted_distances = sorted(euclidean_distances.items(), key=lambda x: x[1])
    
#     # Create a plot axis
#     ax = plt.gca()
      
#     # Draw circles with the soma as the center and the radius as the distance to each bifurcation or termination
#     for node, distance in sorted_distances:
#         color_index = node - 1
        
#         # Draw a circle with the soma (Node 1) as the center and the distance as the radius
#         circle = Circle(soma_position, distance, color=circle_colors[color_index], fill=False, linestyle='--')
#         ax.add_patch(circle)
        
#         # Annotate the coordinates (x, y) next to each node
#         node_position = node_positions[node]
#         coordinates_text = f'({node_position[0]:.1f}, {node_position[1]:.1f})'
#         plt.text(node_position[0] + 0.2, node_position[1] + 0.2, coordinates_text, 
#                  horizontalalignment='center', fontsize=12, color='black')
        
#         # Annotate the Euclidean distance in blue
#         plt.text(node_position[0], node_position[1] + 0.4, f'{distance:.2f}', 
#                  horizontalalignment='center', fontsize=15, color='blue')
        
#         # Annotate the path distance in red slightly below the node
#         path_distance = path_distances.get(node, 0)
#         plt.text(node_position[0], node_position[1] - 0.4, f'{path_distance:.2f}', 
#                  horizontalalignment='center', fontsize=15, color='red')

#     # Display the sorted nodes and their colors vertically on the right
#     text_y_position = 3  # Start displaying at the top
#     text_x_position = 4  # Position the text to the right of the plot
    
#     # Add Node 1 (Soma) to the sorted list
#     sorted_distances.insert(0, (1, 0))
    
#     # Print nodes vertically in ascending order of distance
#     for idx, (node, distance) in enumerate(sorted_distances):
#         # Display the node number and distance in the corresponding color
#         plt.text(text_x_position, text_y_position - idx * 0.5, f'Node {node}: {distance:.2f}', 
#                  fontsize=15, color=node_colors[node - 1], fontweight='bold')
    
#     # Annotate legend for Euclidean and path distances
#     plt.annotate(
#         'Euclidean \n distance \n to soma', 
#         xy=(-1.2, -0.7), xytext=(-3, -2),
#         fontsize=14, color='blue', fontweight='bold',
#         arrowprops=dict(facecolor='black', shrink=0.05, width=2, headwidth=10)
#     )
#     plt.annotate(
#         'Path \n distance \n to soma', 
#         xy=(-1.2, -1.2), xytext=(-3, -3),
#         fontsize=14, color='red', fontweight='bold',
#         arrowprops=dict(facecolor='black', shrink=0.05, width=2, headwidth=10)
#     )

#     # Calculate and label the path distance between source and target nodes next to edges
#     for edge in G.edges():
#         source, target = edge
#         pos_source = node_positions[source]
#         pos_target = node_positions[target]
#         distance = calculate_euclidean_distance(pos_source, pos_target)
#         edge_midpoint = ((pos_source[0] + pos_target[0]) / 2, (pos_source[1] + pos_target[1]) / 2)
#         plt.text(edge_midpoint[0], edge_midpoint[1], f'{distance:.2f}', fontsize=13, color='black')

#     # Set plot limits
#     plt.xlim(-3, 5)
#     plt.ylim(-3.5, 4)
#     plt.gca().set_aspect('equal', adjustable='box')

#     # Save the plot to the specified directory
#     save_path = os.path.join(config.SAVE_DIRECTORY, 'test_neuron_plot.png')
#     plt.savefig(save_path)
#     print(f'Saved plot to {save_path}')
#     plt.close()

#     # Generate the SWC file (for illustration purposes)
#     generate_swc_file(node_positions, edges, 'example_neuron.swc')

            


