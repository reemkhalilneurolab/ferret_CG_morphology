
# Cause: This warning is about the change in default value for the n_init parameter of KMeans. Currently, the default is 10, but it will change to 'auto' in future versions.
# Solution: To avoid this warning and ensure consistent behavior in future versions of scikit-learn, you should explicitly set the n_init parameter when instantiating KMeans.
import os
import sys
os.environ["OMP_NUM_THREADS"] = "1" 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import seaborn as sns
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster, set_link_color_palette
from sklearn.preprocessing import LabelEncoder
from scipy.spatial.distance import squareform, pdist
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
# Absolute path to the directory where config.py is located
# config_path = r"C:\Users\ahmad\Documents\GitHub\neurotopo\config"  # Change to your actual path

# # Add to sys.path
# sys.path.append(config_path)
import config
import math
from sklearn.manifold import MDS
import copy
from scipy.spatial import ConvexHull
from collections import defaultdict
from sklearn.cluster import SpectralClustering
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import umap
import networkx as nx
import matplotlib as mpl
import numpy as np
from scipy.interpolate import interp1d
import copy
import kmapper as km
from sklearn.cluster import DBSCAN
from sklearn.manifold import MDS, Isomap
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage, cophenet, fcluster
from scipy.spatial.distance import pdist
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score


import warnings
warnings.filterwarnings('ignore')

exclude_descriptors = config.EXCLUDED_DESCRIPTORS

os.makedirs(config.PLOTS_DIRECTORY, exist_ok=True)


class Plotter:
    def __init__(self, n_components=2, neuron_class_df=None):
        """
        Initializes the plotting class with a default number of principal components
        and a DataFrame mapping neuron names to their classes.
        
        Parameters:
        - n_components (int): Default number of principal components for dimensionality reduction.
        - neuron_class_df (pd.DataFrame): DataFrame with mapping between neuron names and their classes.
        """
        self.n_components = n_components
        self.neuron_class_df = neuron_class_df
        self.dimentionality_reduction_plot_dir = os.path.join(config.PLOTS_DIRECTORY, "dimentionality_reduction")
        os.makedirs(self.dimentionality_reduction_plot_dir, exist_ok=True)

    def plot_pca_from_distance_matrix(self, distance_matrix, descriptor_name=''):
        """
        Plots PCA from a combined distance matrix using Multidimensional Scaling (MDS),
        coloring points according to the classes defined in neuron_class_df.
        
        Parameters:
        - distance_matrix (pd.DataFrame): A combined distance matrix with neuron names as index and columns.
        - descriptor_name (str): Optional descriptor name for naming the saved file.
        """
        # Step 1: Classical Multidimensional Scaling (MDS)
        mds = MDS(n_components=self.n_components, normalized_stress='auto', dissimilarity="precomputed", random_state=42)
        mds_coords = mds.fit_transform(distance_matrix)

        # Step 2: Apply PCA
        pca = PCA(n_components=self.n_components)
        pca_coords = pca.fit_transform(mds_coords)
        
        # Reindex neuron_class_df to match the index of the distance matrix for consistent color mapping
        class_colors = self.neuron_class_df.set_index('neuron_name')['class_color'].reindex(distance_matrix.index).values


        # # Extract labels for neurons based on the distance matrix's index
        # labels = self.neuron_class_df.set_index('neuron_name')['class'].reindex(distance_matrix.index).values
        
        # # # Convert class labels from strings to integers
        # # label_encoder = LabelEncoder()
        # # numeric_labels = label_encoder.fit_transform(labels)
        
        # # Convert class labels from strings to integers and assign colors
        # unique_classes = np.unique(labels)
        # color_dict = {cls: config.COLORS_PALLETE[i] for i, cls in enumerate(unique_classes)}
        
        # # Map each label to a color
        # class_colors = np.array([color_dict[label] for label in labels])
        
        # Step 3: Plotting
        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(pca_coords[:, 0], pca_coords[:, 1], c=class_colors, cmap='viridis', marker='o', edgecolor='k', s=50, alpha=0.6)
        # plt.colorbar(scatter)
        plt.title(f"PCA Projection of {descriptor_name}")
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        plt.grid(False)
        plt.legend(title='', loc='upper right',frameon=False)

        # Save and close the plot
        file_name = f"PCA_from_{descriptor_name}_distance_matrix.png"
        plt.savefig(os.path.join(self.dimentionality_reduction_plot_dir, file_name), dpi=300, bbox_inches='tight', pad_inches=0.5)
        plt.close()
        
        self.plot_tsne_from_distance_matrix(distance_matrix, descriptor_name)
        # self.plot_umap_from_distance_matrix(distance_matrix, descriptor_name)
        self.plot_cluster_graph_from_distance_matrix(distance_matrix, descriptor_name)
        
        
    def analyze_linearity_from_distance_matrix(self, distance_matrix, descriptor_name=''):

        # Step 1: MDS to get embedding from precomputed distances
        mds = MDS(n_components=20, dissimilarity='precomputed', random_state=42)
        mds_coords = mds.fit_transform(distance_matrix.values)
    
        # Step 2: PCA Scree Plot
        pca = PCA()
        pca.fit(mds_coords)
        explained_variance = np.cumsum(pca.explained_variance_ratio_)
    
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.plot(explained_variance, marker='o')
        plt.title(f'PCA Scree Plot ({descriptor_name})')
        plt.xlabel('Number of Components')
        plt.ylabel('Cumulative Explained Variance')
        plt.grid(True)
        
        # Add interpretation as text box
        total_variance = explained_variance[1]
        message = (
            f"First 2 PCs explain {total_variance*100:.1f}% variance\n"
            + ("→ Likely linear" if total_variance > 0.9 else "→ May be nonlinear")
        )
        
        plt.text(
            0.95, 0.05, message,
            transform=plt.gca().transAxes,
            fontsize=10, color='black',
            ha='right', va='bottom',
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray')
        )

    
        # Step 3: Isomap Residual Variance
        residuals = []
        dims = range(1, min(20, len(distance_matrix.columns)))
        for d in dims:
            iso = Isomap(n_neighbors=5, n_components=d, metric='precomputed')
            iso.fit(distance_matrix.values)
            residuals.append(iso.reconstruction_error())
    
        plt.subplot(1, 2, 2)
        plt.plot(dims, residuals, marker='o', color='orange')
        plt.title(f'Isomap Residual Variance ({descriptor_name})')
        plt.xlabel('Embedding Dimensions')
        plt.ylabel('Residual Variance')
        plt.grid(True)
        
        # Interpretation logic
        low_dim_error = residuals[1] if len(residuals) > 1 else residuals[0]
        message = (
            f"Residual @ 2D: {low_dim_error:.3f}\n"
            + ("→ Low error → likely linear or low-dimensional"
               if low_dim_error < 0.1 else
               "→ High error → likely nonlinear structure")
        )
        
        # Add text box
        plt.text(
            0.95, 0.05,
            message,
            transform=plt.gca().transAxes,
            fontsize=10, color='black',
            ha='right', va='bottom',
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray')
        )
    
        plt.tight_layout()        

        file_name = f"linearity_analysis_{descriptor_name}.png"
        plt.savefig(os.path.join(self.dimentionality_reduction_plot_dir, file_name), dpi=300, bbox_inches='tight', pad_inches=0.5)
        plt.close()



    def plot_tsne_from_distance_matrix(self, distance_matrix, descriptor_name=''):
        """
        Plots t-SNE from a combined distance matrix using the distance matrix directly,
        coloring points according to the classes defined in neuron_class_df.
        
        Parameters:
        - distance_matrix (pd.DataFrame): A combined distance matrix with neuron names as index and columns.
        - descriptor_name (str): Optional descriptor name for naming the saved file.
        """
        # Step 1: Apply t-SNE
        tsne = TSNE(n_components=2, metric='precomputed', init='random', random_state=42)
        tsne_coords = tsne.fit_transform(distance_matrix)
    
        # Retrieve colors for neuron classes
        class_colors = self.neuron_class_df.set_index('neuron_name')['class_color'].reindex(distance_matrix.index).values
    
        # Step 2: Plotting
        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(tsne_coords[:, 0], tsne_coords[:, 1], c=class_colors, cmap='viridis', marker='o', edgecolor='k', s=50, alpha=0.6)
        plt.title(f"t-SNE Projection of {descriptor_name}")
        plt.xlabel("t-SNE Dimension 1")
        plt.ylabel("t-SNE Dimension 2")
        plt.grid(False)
    
        # Save and close the plot
        file_name = f"tSNE_from_{descriptor_name}_distance_matrix.png"
        plt.savefig(os.path.join(self.dimentionality_reduction_plot_dir, file_name), dpi=300, bbox_inches='tight', pad_inches=0.5)
        plt.close()
        
        # Step 3: Run Mapper
        mapper = km.KeplerMapper()
        X = tsne_coords  # use 2D t-SNE coords as both lens and data
        
        graph = mapper.map(
            lens=X,
            X=X,
            clusterer=DBSCAN(eps=0.5, min_samples=3),
            cover=km.Cover(n_cubes=10, perc_overlap=0.3)
        )
        
        # Step 4: Visualize Mapper graph
        #
        # The DBSCAN parameters above are absolute, so whether any cluster forms
        # depends on the scale of the t-SNE embedding. On a combined distance
        # matrix the Mapper graph can come out empty, which is a property of this
        # optional diagram rather than of the clustering it illustrates, so it is
        # reported and skipped instead of stopping the run.
        mapper_output_path = os.path.join(self.dimentionality_reduction_plot_dir, f"mapper_graph_from_{descriptor_name}.html")
        if not graph.get("nodes"):
            print(f"Mapper graph for {descriptor_name} is empty at the current "
                  f"DBSCAN settings; skipping this diagram.")
            return
        mapper.visualize(
            graph,
            path_html=mapper_output_path,
            title=f"KeplerMapper Mapper Graph from t-SNE of {descriptor_name}"
        )
        


    def plot_umap_from_distance_matrix(self, distance_matrix, descriptor_name=''):
        """
        Plots UMAP from a combined distance matrix using the distance matrix directly,
        coloring points according to the classes defined in neuron_class_df.
        
        Parameters:
        - distance_matrix (pd.DataFrame): A combined distance matrix with neuron names as index and columns.
        - descriptor_name (str): Optional descriptor name for naming the saved file.
        """
        # Step 1: Apply UMAP
        umap_model = umap.UMAP(n_components=2, metric='precomputed', random_state=42)
        umap_coords = umap_model.fit_transform(distance_matrix)
    
        # Retrieve colors for neuron classes
        class_colors = self.neuron_class_df.set_index('neuron_name')['class_color'].reindex(distance_matrix.index).values
    
        # Step 2: Plotting
        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(umap_coords[:, 0], umap_coords[:, 1], c=class_colors, cmap='viridis', marker='o', edgecolor='k', s=50, alpha=0.6)
        plt.title(f"UMAP Projection of {descriptor_name}")
        plt.xlabel("UMAP Dimension 1")
        plt.ylabel("UMAP Dimension 2")
        plt.grid(False)
    
        # Save and close the plot
        file_name = f"UMAP_from_{descriptor_name}_distance_matrix.png"
        plt.savefig(os.path.join(self.dimentionality_reduction_plot_dir, file_name), dpi=300, bbox_inches='tight', pad_inches=0.5)
        plt.close()
        

    # def plot_cluster_graph_from_distance_matrix(self, distance_matrix, descriptor_name=''):
    #   """
    #   Plots a cluster graph from a combined distance matrix using NetworkX,
    #   with nodes representing neurons and edges representing the distances between them.
    #   """
    #   neuron_names = distance_matrix.index.tolist()
    #   n = len(neuron_names)
    
    #   # Create a graph
    #   G = nx.Graph()
    #   for i in range(n):
    #       for j in range(i + 1, n):
    #           ni = neuron_names[i]
    #           nj = neuron_names[j]
    #           distance = distance_matrix.loc[ni, nj]
    #           G.add_edge(ni, nj, weight=distance)
    
    #   # Compute layout based on distances
    #   pos = nx.spring_layout(G, weight='weight')
    #   edges = G.edges()
    #   weights = [G[u][v]['weight'] for u, v in edges]
    
    #   fig, ax = plt.subplots(figsize=(10, 8))

    #   nx.draw_networkx_nodes(G, pos, node_color='blue', node_size=50, ax=ax)
    #   nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color=weights, width=2, edge_cmap=plt.cm.Blues, ax=ax)
    #   nx.draw_networkx_labels(G, pos, font_size=8, font_color='white', ax=ax)
    
    #   # Title
    #   ax.set_title('Cluster Graph from Precomputed Distances')
    
    #   # Add colorbar linked to edge weights
    #   norm = mpl.colors.Normalize(vmin=min(weights), vmax=max(weights))
    #   sm = mpl.cm.ScalarMappable(cmap=plt.cm.Blues, norm=norm)
    #   sm.set_array([])  # needed for older matplotlib
    #   fig.colorbar(sm, ax=ax, orientation='vertical', label='Edge Weight (Distance)')
    
    #   ax.axis('off')
    
    #   # Save
    #   similarity_dir = os.path.join(config.PLOTS_DIRECTORY, "similarity_matrix")
    #   if not os.path.exists(similarity_dir):
    #     os.makedirs(similarity_dir)
    
    #   file_name = f"Cluster_Graph_from_{descriptor_name}_distance_matrix.png"
    #   plt.savefig(os.path.join(similarity_dir, file_name), dpi=300, bbox_inches='tight', pad_inches=0.5)
    #   plt.close()
    
    def plot_cluster_graph_from_distance_matrix(self, distance_matrix, descriptor_name='', k=3):
        """
        Plots a cluster graph from a combined distance matrix using NetworkX,
        showing only the k-nearest neighbors per neuron.
        Nodes are colored by original class using self.neuron_class_df.
        
        Parameters:
            distance_matrix (pd.DataFrame): Precomputed distance matrix
            descriptor_name (str): Optional descriptor name for output file
            k (int): Number of nearest neighbors to connect per node
        """
        import matplotlib as mpl
    
        neuron_names = distance_matrix.index.tolist()
        G = nx.Graph()
    
        # Add only k-nearest edges per node
        for neuron in neuron_names:
            distances = distance_matrix.loc[neuron].copy()
            distances = distances.drop(neuron)  # exclude self
            nearest = distances.nsmallest(k)   # get k smallest distances
            for neighbor, dist in nearest.items():
                G.add_edge(neuron, neighbor, weight=dist)
    
        # Compute layout
        pos = nx.spring_layout(G, weight='weight')
    
        # Map neuron to class
        neuron_to_class = dict(zip(self.neuron_class_df['neuron_name'], self.neuron_class_df['class']))
        unique_classes = sorted(set(neuron_to_class[n] for n in G.nodes()))
        
                
               
        # Assign colors per class
        custom_colors = ['#FF5733', '#3366CC']  # orange, blue
        class_to_color = {cls: custom_colors[i] for i, cls in enumerate(unique_classes)}
        node_colors = [class_to_color[neuron_to_class[n]] for n in G.nodes()]
    
        # Edge weights for coloring
        edges = G.edges()
        weights = [G[u][v]['weight'] for u, v in edges]
    
        # Plotting
        fig, ax = plt.subplots(figsize=(10, 8))
    
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=50, ax=ax)
        # nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color=weights, width=2, edge_cmap=plt.cm.Blues, ax=ax)
        # No node labels
    
        # Title
        ax.set_title(f'k-NN Cluster Graph (k={k}) from {descriptor_name} Distance Matrix')
        
        edge_cmap = plt.cm.coolwarm
        nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color=weights,
                       width=2, edge_cmap=edge_cmap, ax=ax)

        # Edge weight colorbar
        norm = mpl.colors.Normalize(vmin=min(weights), vmax=max(weights))
        # sm = mpl.cm.ScalarMappable(cmap=plt.cm.Blues, norm=norm)
        sm = mpl.cm.ScalarMappable(cmap=edge_cmap, norm=norm)

        sm.set_array([])
        fig.colorbar(sm, ax=ax, orientation='vertical', label='Edge Weight (Distance)')
    
        # Legend for classes
        for cls, color in class_to_color.items():
            ax.scatter([], [], c=[color], label=f'Class {cls}', s=60)
            
        # (0, 0) is the bottom-left corner
        # (1, 1) is the top-right corner
        # loc defines which corner of the legend box you are positioning.
        # bbox_to_anchor defines where you’re positioning that corner.
        # loc='upper left' pins the top-left corner of the legend box
        # bbox_to_anchor=(0.01, 0.99) means:
        #     Anchor that corner at 1% from the left
        #     And 99% from the bottom
        #     So the legend will be tucked near the top-left corner of the actual plotting area — inside the axes.
        ax.legend(title='', bbox_to_anchor=(0.01, 0.99), loc='upper left')
    
        ax.axis('off')
    
        # Save
        # similarity_dir = os.path.join(config.PLOTS_DIRECTORY, "similarity_matrix")
        # os.makedirs(similarity_dir, exist_ok=True)
        file_name = f"Cluster_Graph_from_{descriptor_name}_k{k}.png"
        plt.savefig(os.path.join(self.dimentionality_reduction_plot_dir, file_name), dpi=300, bbox_inches='tight', pad_inches=0.5)
        plt.close()



      


    import numpy as np
    import pandas as pd
    from scipy.interpolate import interp1d
    import copy
    
 



   




    def plot_boxplots_and_step_functions_by_class(self, _all_descriptors, n=10):
        """
        Plots box plots of descriptor values by class for each descriptor, excluding outliers, and plots
        step functions of a random sample of neurons from each class on the same plot with different colors.
        
        Parameters:
        - _all_descriptors (dict): Dictionary of descriptors, where each key is a descriptor name and each value is a 
                                   dictionary with neuron names as keys and descriptor values as lists of tuples.
        - neuron_class_df (pd.DataFrame): DataFrame with 'neuron_name' and 'class' columns to map neurons to their classes.
        - n (int): Number of neurons to randomly sample from each class for plotting step functions.
        - exclude_descriptors (list): List of descriptors to exclude from plotting.
        
        Returns:
        - step_function_data (dict): A nested dictionary containing step function data for each sampled neuron and descriptor.
        """
        all_descriptors = copy.deepcopy(_all_descriptors)

                

        # Dictionary to store step function data for each sampled neuron and descriptor
        step_function_data = {}

        # Randomly sample neurons from each class
        sampled_neurons = self.neuron_class_df.groupby('class').apply(lambda x: x.sample(min(n, len(x)), random_state=1)).reset_index(drop=True)
        
        # # Define colors for each class using a predefined color palette from the configuration
        # unique_classes = sampled_neurons['class'].unique()
        # class_colors = {class_label: color for class_label, color in zip(unique_classes, config.COLORS_PALLETE[:len(unique_classes)])}
        
        # Use the class_color directly from neuron_class_df
        class_colors = self.neuron_class_df.drop_duplicates('class').set_index('class')['class_color'].to_dict()

        
        # Iterate over each descriptor
        for descriptor_name, descriptor_data in all_descriptors.items():
            
            # Skip the descriptor if it is in the exclusion list
            if descriptor_name in exclude_descriptors:
                print(f'Skipping descriptor {descriptor_name} as it is in the exclusion list.')
                continue
            
            # Extract only the y-values for box plot generation
            extracted_data = {neuron: [value[1] for value in values] for neuron, values in descriptor_data.items()}
            
            # Convert descriptor data to DataFrame
            descriptor_df = pd.DataFrame.from_dict(extracted_data, orient='index')
            descriptor_df.index.name = 'neuron_name'
            
            # Melt the DataFrame to long format for easy plotting with seaborn
            plot_df = descriptor_df.reset_index().melt(id_vars='neuron_name', value_name='value')
            
            # Map neuron names to classes by merging with neuron_class_df
            plot_df = plot_df.merge(self.neuron_class_df, on='neuron_name')
            
            # Identify and remove outliers within each class
            def remove_outliers(group):
                Q1 = group['value'].quantile(0.25)
                Q3 = group['value'].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                return group[(group['value'] >= lower_bound) & (group['value'] <= upper_bound)]
            
            # Apply outlier removal per class with group_keys=False to suppress the warning
            plot_df = plot_df.groupby('class', group_keys=False).apply(remove_outliers)
            
            # Create a box plot for the descriptor, with separate box plots for each class
            plt.figure(figsize=(10, 6))
            sns.boxplot(x='class', y='value', data=plot_df, palette=class_colors, showfliers=False, saturation=1)
            plt.title(f'Box Plot of {descriptor_name} by Class')
            plt.xlabel('Class')
            plt.ylabel('Descriptor Values')
            
            # Save the box plot figure
            plt.savefig(os.path.join(config.PLOTS_DIRECTORY, f'{descriptor_name}_boxplot_by_original_classes.png'), dpi=300)
            plt.close()
            
            # Initialize a dictionary for each descriptor in step_function_data
            step_function_data[descriptor_name] = {}
            
            unique_classes = self.neuron_class_df['class'].unique()
            
            # Collect step function data for the sampled neurons for the current descriptor
            for _, row in sampled_neurons.iterrows():
                neuron_name = row['neuron_name']
                class_label = row['class']
                x_values = [value[0] for value in descriptor_data[neuron_name]]
                y_values = [value[1] for value in descriptor_data[neuron_name]]
                
                # Store step function data for this neuron under its descriptor and class
                step_function_data[descriptor_name][neuron_name] = {
                    'class': class_label,
                    'x_values': x_values,
                    'y_values': y_values
                }
            
            # # Define colors for each class
            # unique_classes = sampled_neurons['class'].unique()
            # colors = sns.color_palette("hsv", len(unique_classes))
            # class_colors = dict(zip(unique_classes, colors))
            
            # Track which classes have already been labeled in the legend
            labeled_classes = set()
            
            # Plot step functions with step formatting and color by class
            plt.figure(figsize=(12, 8))
            for neuron_name, data in step_function_data[descriptor_name].items():
                class_label = data['class']
                x_values = data['x_values']
                y_values = data['y_values']
                
                # Set the label only for the first occurrence of each class
                label = class_label if class_label not in labeled_classes else None
                if label:
                    labeled_classes.add(class_label)
                
                alpha_value = 0.5 if class_label == unique_classes[0] else 1.0
                
                plt.step(x_values, y_values, where='post', label=label, 
                         color=class_colors[class_label], alpha=alpha_value)
            
            # plt.title(f'Step Functions of Sampled Neurons for {descriptor_name}')
            plt.title('')
            plt.xlabel('Radii', fontsize=14)
            plt.ylabel('Descriptor Values', fontsize=14)
            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)
            plt.legend(title='', loc='upper right',frameon=False)

            
            # Save the step function plot
            plt.savefig(os.path.join(config.PLOTS_DIRECTORY, f'{descriptor_name}_step_functions.png'), dpi=300)
            plt.close()
        
        return step_function_data
    
    def plot_boxplots_and_step_functions_by_dendro_clusters(self, _all_descriptors, n=10):
        """
        Plots box plots of descriptor values by class for each descriptor, excluding outliers, and plots
        step functions of a random sample of neurons from each class on the same plot with different colors.
        
        Parameters:
        - _all_descriptors (dict): Dictionary of descriptors, where each key is a descriptor name and each value is a 
                                   dictionary with neuron names as keys and descriptor values as lists of tuples.
        - neuron_class_df (pd.DataFrame): DataFrame with 'neuron_name' and 'class' columns to map neurons to their classes.
        - n (int): Number of neurons to randomly sample from each class for plotting step functions.
        
        Returns:
        - step_function_data (dict): A nested dictionary containing step function data for each sampled neuron and descriptor.
        """
        all_descriptors = copy.deepcopy(_all_descriptors)
    
        # Dictionary to store step function data for each sampled neuron and descriptor
        step_function_data = {}
    
        # Randomly sample neurons from each cluster
        sampled_neurons = self.neuron_class_df.groupby('dendrogram_cluster').apply(lambda x: x.sample(min(n, len(x)), random_state=1)).reset_index(drop=True)
    
        # Use the class_color directly from neuron_class_df
        class_colors = self.neuron_class_df.drop_duplicates('dendrogram_cluster').set_index('dendrogram_cluster')['dendro_class_color'].to_dict()
    
        # Create a mapping from dendrogram_cluster to descriptive cluster labels
        unique_clusters = sorted(self.neuron_class_df['dendrogram_cluster'].unique())
        
        
        cluster_label_mapping = {cluster: f"Cluster {i + 1}" for i, cluster in enumerate(unique_clusters)}
    
        # Iterate over each descriptor
        for descriptor_name, descriptor_data in all_descriptors.items():
            
            # Skip the descriptor if it is in the exclusion list
            if descriptor_name in exclude_descriptors:
                print(f'Skipping descriptor {descriptor_name} as it is in the exclusion list.')
                continue
            
            # Extract only the y-values for box plot generation
            extracted_data = {neuron: [value[1] for value in values] for neuron, values in descriptor_data.items()}
            
            # Convert descriptor data to DataFrame
            descriptor_df = pd.DataFrame.from_dict(extracted_data, orient='index')
            descriptor_df.index.name = 'neuron_name'
            
            # Melt the DataFrame to long format for easy plotting with seaborn
            plot_df = descriptor_df.reset_index().melt(id_vars='neuron_name', value_name='value')
            
            # Map neuron names to clusters by merging with neuron_class_df
            plot_df = plot_df.merge(self.neuron_class_df, on='neuron_name')
            
            # Identify and remove outliers within each cluster
            def remove_outliers(group):
                Q1 = group['value'].quantile(0.25)
                Q3 = group['value'].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                return group[(group['value'] >= lower_bound) & (group['value'] <= upper_bound)]
            
            # Apply outlier removal per cluster with group_keys=False to suppress the warning
            plot_df = plot_df.groupby('dendrogram_cluster', group_keys=False).apply(remove_outliers)
            
            class_colors = {str(k): v for k, v in class_colors.items()}
            plot_df['dendrogram_cluster'] = plot_df['dendrogram_cluster'].astype(str)
            
            # Create a box plot for the descriptor, with separate box plots for each cluster
            plt.figure(figsize=(7, 7))
            sns.boxplot(x='dendrogram_cluster', y='value', data=plot_df, palette=class_colors, showfliers=False, saturation=1)
            plt.title(f'Box Plot of {descriptor_name} by Cluster')
            plt.xlabel('Cluster')
            plt.ylabel(f'{descriptor_name} Descriptor Values')
            
            # Save the box plot figure
            plt.savefig(os.path.join(config.PLOTS_DIRECTORY, f'{descriptor_name}_boxplot_by_dendro_cluster.png'), dpi=300)
            plt.close()
            
            # Initialize a dictionary for each descriptor in step_function_data
            step_function_data[descriptor_name] = {}
            
            unique_classes = self.neuron_class_df['dendrogram_cluster'].unique()
            
            # Collect step function data for the sampled neurons for the current descriptor
            for _, row in sampled_neurons.iterrows():
                neuron_name = row['neuron_name']
                cluster_label = row['dendrogram_cluster']
                x_values = [value[0] for value in descriptor_data[neuron_name]]
                y_values = [value[1] for value in descriptor_data[neuron_name]]
                
                # Store step function data for this neuron under its descriptor and cluster
                step_function_data[descriptor_name][neuron_name] = {
                    'cluster': cluster_label,
                    'x_values': x_values,
                    'y_values': y_values
                }
            
            # Track which clusters have already been labeled in the legend
            labeled_clusters = set()
            
            # Plot step functions with step formatting and color by cluster
            plt.figure(figsize=(8, 7))
            for neuron_name, data in step_function_data[descriptor_name].items():
                cluster_label = data['cluster']
                x_values = data['x_values']
                y_values = data['y_values']
                
                # Set the label only for the first occurrence of each cluster
                label = cluster_label_mapping[cluster_label] if cluster_label not in labeled_clusters else None
                if label:
                    labeled_clusters.add(cluster_label)
                    
                alpha_value = 0.5 if cluster_label == unique_classes[0] else 1.0
                
                plt.step(x_values, y_values, where='post', label=label, 
                   color=class_colors[str(cluster_label)], alpha=alpha_value)
            
            # plt.title(f'Step Functions of Sampled Neurons for {descriptor_name} by Dendrogram Cluster')
            plt.title('')
            plt.xlabel('Radii', fontsize=18)
            descriptor_name_modified = descriptor_name.replace('_', ' ')
            plt.ylabel(f'{descriptor_name_modified} descriptor values', fontsize=18)
            plt.xticks(fontsize=18)
            plt.yticks(fontsize=18)
            # plt.legend(title='', loc='upper right', fontsize=16, frameon=False)
            
            # Save the step function plot
            plt.savefig(os.path.join(config.PLOTS_DIRECTORY, f'{descriptor_name}_step_functions_by_dendro_cluster.png'), dpi=300)
            plt.close()
        
        return step_function_data


    def plot_descriptor_boxplots_by_class(self, df, descriptor_name=''):
        """
        Plots neuron-level boxplots grouped by class, sorted by median descriptor value.
        """
    
        # Compute class medians
        class_medians = {}
        for cls in self.neuron_class_df['class'].unique():
            neuron_names = self.neuron_class_df[self.neuron_class_df['class'] == cls]['neuron_name']
            values = [
                df[df['neuron_name'] == neuron]['values'].iloc[0]
                for neuron in neuron_names if neuron in df['neuron_name'].values
            ]
            flat = [v for sublist in values for v in sublist]
            class_medians[cls] = np.median(flat) if flat else np.inf
    
        # Sort classes by median
        unique_classes = sorted(class_medians, key=class_medians.get)
    
        class_label_mapping = {cls: cls for cls in unique_classes}
        fig, axes = plt.subplots(nrows=1, ncols=len(unique_classes), figsize=(24, 12), sharey=True)
        

        for i, cls in enumerate(unique_classes):
            class_neurons = self.neuron_class_df[self.neuron_class_df['class'] == cls]
    
            neuron_data = {}
            neuron_medians = {}
            neuron_iqrs = {}
    
            for neuron in class_neurons['neuron_name']:
                if neuron in df['neuron_name'].values:
                    values = df[df['neuron_name'] == neuron]['values'].iloc[0]
                    median = np.median(values)
                    iqr = np.percentile(values, 75) - np.percentile(values, 25)
                    neuron_data[neuron] = values
                    neuron_medians[neuron] = median
                    neuron_iqrs[neuron] = iqr
    
            medians = list(neuron_medians.values())
            q1_med = np.percentile(medians, 25)
            q3_med = np.percentile(medians, 75)
            iqr_med = q3_med - q1_med
            lower_med = q1_med - 1.5 * iqr_med
            upper_med = q3_med + 1.5 * iqr_med
    
            iqrs = list(neuron_iqrs.values())
            iqr_cutoff = np.percentile(iqrs, 95)
    
            neuron_values_with_medians = []
            for neuron, values in neuron_data.items():
                med = neuron_medians[neuron]
                spread = neuron_iqrs[neuron]
                if lower_med <= med <= upper_med and spread <= iqr_cutoff:
                    inliers = [v for v in values if v is not None]
                    if inliers:
                        neuron_values_with_medians.append((inliers, med))
    
            neuron_values_with_medians.sort(key=lambda x: x[1])
            sorted_values = [values for values, _ in neuron_values_with_medians]
    
            color = class_neurons['class_color'].iloc[0]
            bp = axes[i].boxplot(sorted_values, patch_artist=True, showfliers=False,
                                 boxprops=dict(facecolor=color))
    
            axes[i].set_title(class_label_mapping[cls], fontsize=16)
            if i == 0:
                descriptor_name_modified = descriptor_name.replace('_', ' ')
                axes[i].set_ylabel(f'{descriptor_name_modified} descriptor values', fontsize=16)
            
            # Extract sorted neuron names to match sorted_values
            sorted_neuron_names = [neuron for neuron, _ in sorted(neuron_data.items(), key=lambda x: neuron_medians[x[0]]) 
                                   if lower_med <= neuron_medians[neuron] <= upper_med and neuron_iqrs[neuron] <= iqr_cutoff]
            
            # Set tick labels to neuron names
            axes[i].set_xticks(range(1, len(sorted_neuron_names) + 1))
            axes[i].set_xticklabels(sorted_neuron_names, rotation=90, fontsize=10)

            # axes[i].set_xlabel('Neuron', fontsize=16)
            axes[i].tick_params(axis='both', labelsize=10)
            axes[i].grid(False)
    
        # plt.tight_layout()
        plt.subplots_adjust(wspace=0.05)
        file_name = f"Boxplots_{descriptor_name}_by_class.png"
        plt.savefig(os.path.join(config.PLOTS_DIRECTORY, file_name), dpi=300, bbox_inches='tight', pad_inches=0.5)
        plt.close()


    def plot_descriptor_histograms_by_class(self, df, descriptor_name):
        """
        Plots overlaid histograms of descriptor values per class.
    
        Parameters:
        - df: DataFrame with columns ['neuron_name', 'values']
        - descriptor_name: name of the descriptor to use in plot title and file
        """
        # Flatten neuron-level values into rows: one value per row
        rows = []
        for _, row in df.iterrows():
            neuron = row['neuron_name']
            values = row['values']
            neuron_class = self.neuron_class_df.loc[self.neuron_class_df['neuron_name'] == neuron, 'class'].values[0]
            for v in values:
                rows.append({'value': v, 'class': neuron_class})
    
        long_df = pd.DataFrame(rows)
    
        # Plot
        plt.figure(figsize=(10, 6))
        sns.set_theme(style="whitegrid")
    
        # sns.histplot(data=long_df, x='value', hue='class', kde=True,
        #              stat='density', common_norm=False, alpha=0.4, linewidth=1)
        sns.histplot(data=long_df, x='value', hue='class',
             multiple='layer', stat='count', bins=30,
             element='step', fill=True, alpha=0.4, linewidth=1)
    
        plt.grid(False) 

    
        plt.title(f'Overlaid Histograms - {descriptor_name}', fontsize=14, weight='bold')
        plt.xlabel(f'{descriptor_name} value', fontsize=12)
        plt.ylabel('Density', fontsize=12)
        plt.legend(title=None, frameon=False)
        plt.tight_layout()
    
        file_name = f"Histograms_{descriptor_name}_by_class.png"
        plt.savefig(os.path.join(config.PLOTS_DIRECTORY, file_name), dpi=300)
        plt.close()
        
    def plot_descriptor_histograms_by_class_one_panel_with_texture(self, descriptor_panels):
        """
        Creates a 3-row x 2-column figure with overlaid histograms per class for each descriptor.
        Uses hatch patterns for each class: dots and slashes. Only leftmost plots have axis labels.
        Only top-left plot shows the legend. Applies same outlier filter as the boxplot.
        """
        import math
        import matplotlib.patches as mpatches
    
        descriptors = list(descriptor_panels.keys())
        n = len(descriptors)
        # Grid sized from the descriptor count rather than fixed at 3x2.
        ncols = 2
        nrows = int(np.ceil(n / ncols))

        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5 * ncols, 3.5 * nrows), sharey=False)
        axes = np.atleast_1d(axes).flatten()
    
        unique_classes = sorted(self.neuron_class_df['class'].unique())
    
        class_hatches = {cls: hatch for cls, hatch in zip(unique_classes, ['.', '//'])}
        class_colors = {
            cls: config.COLORS_PALETTE[i]
            for i, cls in enumerate(unique_classes)
        }
    
        for idx, descriptor_name in enumerate(descriptors):
            df = descriptor_panels[descriptor_name]
    
            # Step 1: Compute per-neuron medians and IQRs
            neuron_medians = {}
            neuron_iqrs = {}
            for _, row in df.iterrows():
                values = [v for v in row['values'] if v is not None]
                if values:
                    neuron_medians[row['neuron_name']] = np.median(values)
                    neuron_iqrs[row['neuron_name']] = np.percentile(values, 75) - np.percentile(values, 25)
    
            # Step 2: Determine outlier thresholds
            all_meds = list(neuron_medians.values())
            q1_med = np.percentile(all_meds, 25)
            q3_med = np.percentile(all_meds, 75)
            iqr_med = q3_med - q1_med
            lower_med = q1_med - 1.5 * iqr_med
            upper_med = q3_med + 1.5 * iqr_med
    
            iqr_cutoff = np.percentile(list(neuron_iqrs.values()), 95)
    
            # Step 3: Define valid neuron set
            valid_neurons = {
                neuron for neuron in neuron_medians
                if lower_med <= neuron_medians[neuron] <= upper_med and neuron_iqrs[neuron] <= iqr_cutoff
            }
    
            # Step 4: Prepare long-format DataFrame (filtered)
            rows = []
            for _, row in df.iterrows():
                neuron = row['neuron_name']
                if neuron not in valid_neurons:
                    continue
                values = row['values']
                neuron_class = self.neuron_class_df.loc[
                    self.neuron_class_df['neuron_name'] == neuron, 'class'
                ].values[0]
                for v in values:
                    if v != 0:
                        rows.append({'value': v, 'class': neuron_class})
    
            long_df = pd.DataFrame(rows)
    
            ax = axes[idx]
            show_ylabel = (idx % ncols == 0)
            show_xlabel = (idx >= (nrows - 1) * ncols)
    
            bins = 30
            for cls in long_df['class'].unique():
                cls_values = long_df[long_df['class'] == cls]['value']
                ax.hist(cls_values, bins=bins, histtype='stepfilled',
                        alpha=0.5, label=cls,
                        color=class_colors[cls],
                        hatch=class_hatches[cls],
                        edgecolor='black', linewidth=1)
    
            ax.set_title(descriptor_name.replace("_", "\n"), fontsize=12)
            ax.set_xlabel("Value" if show_xlabel else "")
            ax.set_ylabel("Count" if show_ylabel else "")
            ax.grid(False)
    
            if idx == 0:
                handles = [
                    mpatches.Patch(facecolor=class_colors[cls], hatch=class_hatches[cls],
                                   label=cls, edgecolor='black') for cls in class_hatches
                ]
                ax.legend(handles=handles, title=None, frameon=False, fontsize=10)
            else:
                legend = ax.get_legend()
                if legend is not None:
                    legend.remove()
    
        for i in range(len(descriptors), len(axes)):
            axes[i].axis("off")
    
        plt.tight_layout()
        plt.savefig(os.path.join(config.PLOTS_DIRECTORY, "All_Histograms_by_Class_with_texture.png"), dpi=600)
        plt.close()

        
    # def plot_descriptor_histograms_by_class_one_panel_with_texture(self, descriptor_panels):
    #     """
    #     Creates a 3-row x 2-column figure with overlaid histograms per class for each descriptor.
    #     Uses hatch patterns for each class: dots and slashes. Only leftmost plots have axis labels.
    #     Only top-left plot shows the legend.
    #     """
    #     import math
    #     import matplotlib.patches as mpatches
    
    #     descriptors = list(descriptor_panels.keys())
    #     n = len(descriptors)
    #     nrows, ncols = 3, 2
    
    #     fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5 * ncols, 3.5 * nrows), sharey=False)
    #     axes = axes.flatten()
        
    #     unique_classes = sorted(self.neuron_class_df['class'].unique())
    
    #     class_hatches = {cls: hatch for cls, hatch in zip(sorted(self.neuron_class_df['class'].unique()), ['.', '//'])}
    #     # class_colors = {cls: color for cls, color in zip(sorted(self.neuron_class_df['class'].unique()), ['gray', 'black'])}
       
    #     class_colors = {
    #         cls: config.COLORS_PALETTE[i]
    #         for i, cls in enumerate(unique_classes)
    #     }

    #     for idx, descriptor_name in enumerate(descriptors):
    #         df = descriptor_panels[descriptor_name]
    
    #         # Prepare long-format DataFrame
    #         rows = []
    #         for _, row in df.iterrows():
    #             neuron = row['neuron_name']
    #             values = row['values']
    #             neuron_class = self.neuron_class_df.loc[self.neuron_class_df['neuron_name'] == neuron, 'class'].values[0]
    #             for v in values:
    #                 rows.append({'value': v, 'class': neuron_class})
    #         long_df = pd.DataFrame(rows)
    #         long_df = long_df[long_df['value'] != 0]
    
    #         ax = axes[idx]
    #         show_ylabel = (idx % ncols == 0)
    #         show_xlabel = (idx >= (nrows - 1) * ncols)
    
    #         # Plot manually per class with hatch
    #         bins = 30
    #         for cls in long_df['class'].unique():
    #             cls_values = long_df[long_df['class'] == cls]['value']
    #             ax.hist(cls_values, bins=bins, histtype='stepfilled',
    #                     alpha=0.5, label=cls,
    #                     color=class_colors[cls],
    #                     hatch=class_hatches[cls],
    #                     edgecolor='black', linewidth=1)
    
    #         ax.set_title(descriptor_name.replace("_", "\n"), fontsize=12)
    #         ax.set_xlabel("Value" if show_xlabel else "")
    #         ax.set_ylabel("Count" if show_ylabel else "")
    #         ax.grid(False)
            
    #         # ax.set_xscale("log")
    
    #         if idx == 0:
    #             # Custom legend using patch handles
    #             handles = [
    #                 mpatches.Patch(facecolor=class_colors[cls], hatch=class_hatches[cls],
    #                                label=cls, edgecolor='black') for cls in class_hatches
    #             ]
    #             ax.legend(handles=handles, title=None, frameon=False, fontsize=10)
    #         else:
    #             legend = ax.get_legend()
    #             if legend is not None:
    #                 legend.remove()
    
    #     # Hide any unused axes
    #     for i in range(len(descriptors), len(axes)):
    #         axes[i].axis("off")
    
    #     plt.tight_layout()
    #     plt.savefig(os.path.join(config.PLOTS_DIRECTORY, "All_Histograms_by_Class_with_texture.png"), dpi=600)
    #     plt.close()

    
    def plot_descriptor_histograms_by_class_one_panel(self, descriptor_panels):
        """
        Creates a 3-row x 2-column figure with overlaid histograms per class for each descriptor.
        Shows axis labels only on the first column plots and a single legend in the top-left plot.
        """
        import math
    
        descriptors = list(descriptor_panels.keys())
        n = len(descriptors)
        # Grid sized from the descriptor count rather than fixed at 3x2, so the
        # panel does not break when the descriptor set changes.
        ncols = 2
        nrows = int(np.ceil(n / ncols))

        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5 * ncols, 3 * nrows), sharey=False)
        axes = np.atleast_1d(axes).flatten()
    
        for idx, descriptor_name in enumerate(descriptors):
            df = descriptor_panels[descriptor_name]
    
            # Prepare long-format DataFrame
            rows = []
            for _, row in df.iterrows():
                neuron = row['neuron_name']
                values = row['values']
                neuron_class = self.neuron_class_df.loc[self.neuron_class_df['neuron_name'] == neuron, 'class'].values[0]
                for v in values:
                    rows.append({'value': v, 'class': neuron_class})
            long_df = pd.DataFrame(rows)
    
            ax = axes[idx]
            show_ylabel = (idx % ncols == 0)
            show_xlabel = (idx >= (nrows - 1) * ncols)
    
            sns.histplot(
                data=long_df, x='value', hue='class',
                multiple='layer', stat='count', bins=30,
                element='step', fill=True, alpha=0.4, linewidth=1, ax=ax
            )
    
            ax.set_title(descriptor_name.replace("_", "\n"), fontsize=12)
            ax.set_xlabel("Value" if show_xlabel else "")
            ax.set_ylabel("Count" if show_ylabel else "")
            ax.grid(False)
    
            if idx == 0:
                ax.legend(title=None, frameon=False, fontsize=10)
            else:
                ax.get_legend().remove()
    
        # Hide any unused axes
        for i in range(len(descriptors), len(axes)):
            axes[i].axis("off")
    
        plt.tight_layout()
        plt.savefig(os.path.join(config.PLOTS_DIRECTORY, "All_Histograms_by_Class.png"), dpi=300)
        plt.close()




    def plot_clusters_boxplots_by_neuron(self, df, descriptor_name='', sort_by='increasing'):
        if sort_by == 'dend_order':
            # Get cluster order based on mean dendrogram index
            cluster_order = (
                self.neuron_class_df.groupby('dendrogram_cluster')['cluster_labels_indices']
                .mean()
                .sort_values()
                .index.tolist()
            )
            unique_clusters = cluster_order
        else:
            # Compute median descriptor value per cluster to sort by
            cluster_medians = {}
            for cluster in self.neuron_class_df['dendrogram_cluster'].unique():
                cluster_neurons = self.neuron_class_df[self.neuron_class_df['dendrogram_cluster'] == cluster]
                neuron_values = [
                    df[df['neuron_name'] == neuron]['values'].iloc[0]
                    for neuron in cluster_neurons['neuron_name'] if neuron in df['neuron_name'].values
                ]
                flat_values = [v for sublist in neuron_values for v in sublist]
                cluster_medians[cluster] = np.median(flat_values) if flat_values else np.inf
            unique_clusters = sorted(cluster_medians, key=cluster_medians.get)
    
        cluster_label_mapping = {cluster: f"Cluster {i + 1}" for i, cluster in enumerate(unique_clusters)}
    
        # Create subplots for each cluster
        fig, axes = plt.subplots(nrows=1, ncols=len(unique_clusters), figsize=(12, 6), sharey=True)
    
        # Iterate through each cluster and plot the box plot
        for i, cluster in enumerate(unique_clusters):
            cluster_neurons = self.neuron_class_df[self.neuron_class_df['dendrogram_cluster'] == cluster]
            
            # # Gather neuron values and their medians within the cluster
            # neuron_values_with_medians = []
            # for neuron in cluster_neurons['neuron_name']:
            #     if neuron in df['neuron_name'].values:
            #         values = df[df['neuron_name'] == neuron]['values'].iloc[0]
            #         median_value = np.median(values)
            #         neuron_values_with_medians.append((values, median_value))
            
            # # Gather medians per neuron
            # neuron_medians = []
            # neuron_data = {}
            
            # for neuron in cluster_neurons['neuron_name']:
            #     if neuron in df['neuron_name'].values:
            #         values = df[df['neuron_name'] == neuron]['values'].iloc[0]
            #         median = np.median(values)
            #         neuron_medians.append(median)
            #         neuron_data[neuron] = values
            
            # # Compute IQR on neuron medians
            # q1 = np.percentile(neuron_medians, 25)
            # q3 = np.percentile(neuron_medians, 75)
            # iqr = q3 - q1
            # lower = q1 - 1 * iqr
            # upper = q3 + 1 * iqr
            
            # # Keep neurons with "typical" medians
            # neuron_values_with_medians = []
            # for neuron, values in neuron_data.items():
            #     median = np.median(values)
            #     if lower <= median <= upper:
            #         neuron_values_with_medians.append((values, median))
            
            # Step 1: Gather all values and medians per neuron
            cluster_values = []
            neuron_data = {}
            neuron_medians = {}
            neuron_iqrs = {}
            
            for neuron in cluster_neurons['neuron_name']:
                if neuron in df['neuron_name'].values:
                    values = df[df['neuron_name'] == neuron]['values'].iloc[0]
                    median = np.median(values)
                    iqr = np.percentile(values, 75) - np.percentile(values, 25)
                    cluster_values.extend(values)
                    neuron_data[neuron] = values
                    neuron_medians[neuron] = median
                    neuron_iqrs[neuron] = iqr
            
            # Step 2: Compute cluster-level IQR bounds for medians
            medians = list(neuron_medians.values())
            q1_med = np.percentile(medians, 25)
            q3_med = np.percentile(medians, 75)
            iqr_med = q3_med - q1_med
            lower_med = q1_med - 1.5 * iqr_med
            upper_med = q3_med + 1.5 * iqr_med
            
            # Step 3: Compute IQR threshold for spread
            iqrs = list(neuron_iqrs.values())
            iqr_cutoff = np.percentile(iqrs, 95)  # remove top 10% most spread neurons
            
            # Step 4: Filter neurons by both criteria
            neuron_values_with_medians = []
            for neuron, values in neuron_data.items():
                med = neuron_medians[neuron]
                spread = neuron_iqrs[neuron]
            
                if lower_med <= med <= upper_med and spread <= iqr_cutoff:
                    inliers = [v for v in values if v is not None]  # filter None just in case
                    if inliers:
                        neuron_values_with_medians.append((inliers, med))
                 

            
            # Sort neurons within the cluster by their median values
            neuron_values_with_medians.sort(key=lambda x: x[1])  # Sort by median value
            sorted_values = [values for values, _ in neuron_values_with_medians]
    
            color = cluster_neurons['dendro_class_color'].iloc[0]
    
            # Plot the box plot for the cluster
            bp = axes[i].boxplot(sorted_values, patch_artist=True, showfliers=False, boxprops=dict(facecolor=color))
            
   
            
            # Add this condition to change the median color for the 2nd cluster
            if i == 1:  # Check if this is the second cluster (zero-based index)
                for median in bp['medians']:
                    median.set_color('green')  # Change 'red' to your preferred color
            # Set the title with descriptive cluster label
            axes[i].set_title(cluster_label_mapping[cluster], fontsize=16)
            
            # Set labels for y-axis only on the first subplot to reduce clutter
            if i == 0:
                descriptor_name_modified = descriptor_name.replace('_', ' ')
                axes[i].set_ylabel(f'{descriptor_name_modified} descriptor values', fontsize=16)
    
            axes[i].set_xlabel('Neuron', fontsize=16)
            axes[i].tick_params(axis='both', labelsize=16)
    
        # Adjust layout to prevent overlap
        plt.tight_layout()
        
        # Save the plot
        file_name = f"Boxplots_{descriptor_name}_by_dendro_cluster.png"
        plt.savefig(os.path.join(config.PLOTS_DIRECTORY, file_name), dpi=300, bbox_inches='tight', pad_inches=0.5)
        plt.close()


      
    # def plot_clusters_boxplots_by_neuron(self, df, descriptor_name='', sort_by ='increasing' ):
    #     # Get unique clusters and create a mapping for more readable labels
    #     # unique_clusters = sorted(self.neuron_class_df['dendrogram_cluster'].unique())
    #     if sort_by == 'dend_order':
    #         # Get cluster order based on mean dendrogram index
    #         cluster_order = (
    #             self.neuron_class_df.groupby('dendrogram_cluster')['cluster_labels_indices']
    #             .mean()
    #             .sort_values()
    #             .index.tolist()
    #         )
    #         unique_clusters = cluster_order
            
    #     else :      
        
    #         # Compute median descriptor value per cluster to sort by
    #         cluster_medians = {}
            
    #         for cluster in self.neuron_class_df['dendrogram_cluster'].unique():
    #             cluster_neurons = self.neuron_class_df[self.neuron_class_df['dendrogram_cluster'] == cluster]
    #             neuron_values = [
    #                 df[df['neuron_name'] == neuron]['values'].iloc[0]
    #                 for neuron in cluster_neurons['neuron_name'] if neuron in df['neuron_name'].values
    #             ]
    #             flat_values = [v for sublist in neuron_values for v in sublist]
    #             cluster_medians[cluster] = np.median(flat_values) if flat_values else np.inf
            
    #         # Sort clusters by increasing median
    #         unique_clusters = sorted(cluster_medians, key=cluster_medians.get)
        
        
    #     cluster_label_mapping = {cluster: f"Cluster {i + 1}" for i, cluster in enumerate(unique_clusters)}
    
    #     # Create subplots for each cluster
    #     fig, axes = plt.subplots(nrows=1, ncols=len(unique_clusters), figsize=(12, 6), sharey=True)
        
        
    #     # Iterate through each cluster and plot the box plot
    #     for i, cluster in enumerate(unique_clusters):
    #         cluster_neurons = self.neuron_class_df[self.neuron_class_df['dendrogram_cluster'] == cluster]
            
    #         # Fetch their values using the 'neuron_name' column to access 'df'
    #         cluster_values = [
    #             df[df['neuron_name'] == neuron]['values'].iloc[0]
    #             for neuron in cluster_neurons['neuron_name'] if neuron in df['neuron_name'].values
    #         ]
            

    #         threshold = 1.4  # Set your threshold value
    #         if cluster == 0.0 and descriptor_name == 'Tortuosity':
    #             cluster_values = [sublist for sublist in cluster_values if max(sublist) <= threshold]
            
    #         # Set the color for the cluster
    #         color = cluster_neurons['dendro_class_color'].iloc[0]
    
    #         # Plot the box plot for the cluster
    #         bp = axes[i].boxplot(cluster_values, patch_artist=True, showfliers=False, boxprops=dict(facecolor=color))
            
    #         # Set the title with descriptive cluster label
    #         axes[i].set_title(cluster_label_mapping[cluster], fontsize=16)
    
    #         # Update x-axis tick labels to be numeric values at regular intervals
    #         axes[i].set_xticks(range(0, len(cluster_values), max(1, len(cluster_values) // 5)))  # Show every 5th neuron value or fewer
    #         axes[i].set_xticklabels([str(j) for j in range(0, len(cluster_values), max(1, len(cluster_values) // 5))], fontsize=12)
    
    #         # Set labels for y-axis only on the first subplot to reduce clutter
    #         if i == 0:
    #             descriptor_name_modified = descriptor_name.replace('_', ' ')
    #             axes[i].set_ylabel(f'{descriptor_name_modified} descriptor values', fontsize=16)
    
    #         axes[i].set_xlabel('Neuron', fontsize=16)
    #         # Set tick label sizes for both x and y axes
    #         axes[i].tick_params(axis='x', labelsize=16)
    #         axes[i].tick_params(axis='y', labelsize=16)

            
    
    #     # Adjust layout to prevent overlap
    #     plt.tight_layout()
        
    #     # Save the plot
    #     file_name = f"Boxplots_dendro_clusters_by_neuron_{descriptor_name}.png"
    #     plt.savefig(os.path.join(config.PLOTS_DIRECTORY, file_name), dpi=300, bbox_inches='tight', pad_inches=0.5)
    #     plt.close()
    
 


    def plot_clusters_boxplots_by_cluster(self, df, descriptor_name=''):
        # Create a figure with a single subplot
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 5), sharey=True)
        
        # Store all cluster values and colors
        all_cluster_values = []
        colors = []
        
        # Iterate over each cluster
        for cluster in self.neuron_class_df['dendrogram_cluster'].unique():
            cluster_neurons = self.neuron_class_df[self.neuron_class_df['dendrogram_cluster'] == cluster]
            
            # Combine values of all neurons in the cluster
            combined_values = []
            for neuron in cluster_neurons['neuron_name']:
                if neuron in df['neuron_name'].values:
                    neuron_values = df.loc[df['neuron_name'] == neuron, 'values'].iloc[0]
                    combined_values.extend(neuron_values)
            
            all_cluster_values.append(combined_values)
            colors.append(cluster_neurons['dendro_class_color'].iloc[0])  # Assuming all entries in a cluster share the same color
    
        # Plot a boxplot for each cluster
        bp = ax.boxplot(all_cluster_values, patch_artist=True, showfliers=False)
        
        # Color each boxplot according to the cluster color
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
        
        # Add labels and titles
        ax.set_xticklabels(['Cluster {}'.format(i + 1) for i in range(len(all_cluster_values))])
        ax.set_xlabel('Clusters')
        ax.set_ylabel('Values')
        ax.set_title('Combined Boxplots of Neuron Values by Cluster')
        
        plt.tight_layout()
        file_name = f"Boxplots_dendro_clusters_by_cluster_{descriptor_name}.png"
        plt.savefig(os.path.join(config.PLOTS_DIRECTORY, file_name), dpi=300, bbox_inches='tight', pad_inches=0.5)
        plt.close()
        
    def cluster_and_plot_similarity(self, similarity_matrices):
        """
        Performs Spectral Clustering on each similarity matrix and plots the results.
    
        Parameters:
        - similarity_matrices: Dictionary of similarity matrices for each descriptor.
        - num_clusters: Number of clusters to use in clustering.
    
        Returns:
        - cluster_labels: Dictionary where each descriptor has a list of cluster labels.
        """
    
        cluster_labels = {}
    
        for descriptor, sim_matrix in similarity_matrices.items():
            print(f"Clustering neurons for {descriptor}...")
    
            # Convert similarity matrix to numpy array
            similarity_array = sim_matrix.to_numpy()
    
            # Apply Spectral Clustering
            clustering = SpectralClustering(n_clusters=config.NUM_OF_CLUSTERS, affinity='precomputed', random_state=42)
            labels = clustering.fit_predict(similarity_array)
    
            # Store cluster labels
            cluster_labels[descriptor] = labels
            
            similarity_dir = os.path.join(config.PLOTS_DIRECTORY, "similarity_matrix")

            if not os.path.exists(similarity_dir):
                os.makedirs(similarity_dir)
    
            # Plot heatmap of the similarity matrix
            plt.figure(figsize=(8, 6))
            sns.heatmap(sim_matrix, cmap="coolwarm", annot=False)
            plt.title(f"Similarity Matrix - {descriptor}")
            heatmap_file = f"Similarity_Matrix_{descriptor}.png"
            plt.savefig(os.path.join(similarity_dir, heatmap_file), dpi=300, bbox_inches='tight', pad_inches=0.5)
            plt.close()
            
            # Scatter plot of clustering results
            plt.figure(figsize=(6, 5))
            plt.scatter(range(len(labels)), labels, c=labels, cmap="viridis", edgecolors='k', s=100)
            plt.xlabel("Neuron Index")
            plt.ylabel("Cluster Label")
            plt.title(f"Clustering Results - {descriptor}")
            scatter_file = f"Clusters_{descriptor}.png"
            plt.savefig(os.path.join(similarity_dir, scatter_file), dpi=300, bbox_inches='tight', pad_inches=0.5)
            plt.close()
    
        return cluster_labels
        

# Assume df is your dataframe and config.PLOTS_DIRECTORY is defined
# Example usage of the function
# plot_clusters_boxplots(self, df, descriptor_name='example')

    




class DataAnalysis:
    def __init__(self, data, neuron_class_df, variance_threshold=0.01, correlation_threshold=0.9, drop_low_var_high_corr= True,fine_tune_features= True ):
        """
        Initialize the DataAnalysis model with data and the number of clusters.
        Args:
        data (pd.DataFrame): DataFrame containing the data with columns for features and class labels.
        n_clusters (int): The number of clusters to form.
        variance_threshold (float): Threshold for variance below which columns will be dropped.
        correlation_threshold (float): Threshold for correlation above which one of two highly correlated columns will be dropped.
        """
        self.data = data
        self.n_clusters = len(neuron_class_df['class'].unique())
        self.variance_threshold = variance_threshold
        self.correlation_threshold = correlation_threshold
        self.best_params = None
        self.is_distance_matrix = None

        # Check if data is a distance matrix by verifying if it's square
        if self.data.shape[0] == self.data.shape[1]:
            print("Input data appears to be a distance matrix. Clustering and feature selection steps will be skipped.")
            self.is_distance_matrix = True
        else:
            self.is_distance_matrix = False
            self.model = KMeans(n_clusters=self.n_clusters, n_init=10, random_state=42)
            # self.features = self.data.iloc[:, :-1]  # All columns except the last
            # self.labels = self.data.iloc[:, -1]  # Last column
            
            self.labels = neuron_class_df['class'].values
            self.features  = self.data
            
            if drop_low_var_high_corr:
            # Drop low-variance and highly correlated columns if not a distance matrix
                self.drop_low_variance_and_highly_correlated_columns()
            
            if fine_tune_features:
                self.fine_tune_feature_selection()

    def drop_low_variance_and_highly_correlated_columns(self):
        if self.is_distance_matrix:
            print("Skipping column dropping as input data is a distance matrix.")
            return

        # Drop low-variance columns
        selector = VarianceThreshold(threshold=self.variance_threshold)
        self.features = pd.DataFrame(selector.fit_transform(self.features), columns=self.features.columns[selector.get_support()])
        print(f"Columns with variance below {self.variance_threshold} have been dropped.")

        # Drop highly correlated columns
        correlation_matrix = self.features.corr().abs()
        upper_triangle = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] > self.correlation_threshold)]
        self.features = self.features.drop(columns=to_drop)
        print(f"Columns with correlation above {self.correlation_threshold} have been dropped.")

    def fine_tune_feature_selection(self):
        if self.is_distance_matrix:
            print("Skipping feature selection as input data is a distance matrix.")
            return
        
        # Automate fine-tuning of feature selection parameters using GridSearchCV.
        param_grid = {
            'feature_selection__threshold': [0.01, 0.05, 0.1],
            'feature_selection__k': [5, 10, 15, 20],
            'feature_selection__n_components': [2, 5, 10]
        }

        pipeline = Pipeline([
            ('feature_selection', VarianceThreshold()),
            ('clustering', KMeans(n_clusters=self.n_clusters, n_init=10, random_state=42))
        ])

        def combined_score(estimator, X):
            cluster_labels = estimator.named_steps['clustering'].fit_predict(X)
            silhouette = silhouette_score(X, cluster_labels)
            davies_bouldin = davies_bouldin_score(X, cluster_labels)
            calinski_harabasz = calinski_harabasz_score(X, cluster_labels)

            silhouette_norm = silhouette
            davies_bouldin_norm = 1 / (1 + davies_bouldin)
            calinski_harabasz_norm = calinski_harabasz / (calinski_harabasz + 1)

            combined = (0.5 * silhouette_norm) + (0.25 * davies_bouldin_norm) + (0.25 * calinski_harabasz_norm)
            return combined

        grid_search = GridSearchCV(
            pipeline,
            param_grid=param_grid,
            scoring=combined_score,
            cv=5,
            n_jobs=-1
        )

        grid_search.fit(self.features)
        self.best_params = grid_search.best_params_
        self.best_score = grid_search.best_score_

        print("Best Parameters:", self.best_params)
        print("Best Combined Score:", self.best_score)

        self.features = grid_search.best_estimator_.named_steps['feature_selection'].transform(self.features)

    def evaluate_clustering_metrics(self):
        if self.is_distance_matrix:
            print("Skipping clustering evaluation as input data is a distance matrix.")
            return
        
        self.cluster_labels = self.model.fit_predict(self.features)
        silhouette_avg = silhouette_score(self.features, self.cluster_labels)
        davies_bouldin = davies_bouldin_score(self.features, self.cluster_labels)
        calinski_harabasz = calinski_harabasz_score(self.features, self.cluster_labels)

        self.clustering_metrics = {
            'silhouette_score': silhouette_avg,
            'davies_bouldin_score': davies_bouldin,
            'calinski_harabasz_index': calinski_harabasz
        }
        print("Clustering Metrics:", self.clustering_metrics)

    def exploratory_analysis(self):
        if self.is_distance_matrix:
            print("Skipping exploratory analysis as input data is a distance matrix.")
            return
        
        """
        Perform exploratory data analysis with visualizations: heatmap, boxplot, correlation matrix, and PCA plot.
        """
        # 1. Heatmap of Feature Values
        plt.figure(figsize=(15, 10))
        sns.heatmap(self.features, cmap="coolwarm", annot=False, cbar=True)
        plt.title("Heatmap of Features' values")
        plt.xlabel("Features")
        plt.ylabel("Neurons")
        plt.savefig(os.path.join(config.PLOTS_DIRECTORY, 'heatmap_of_feature_vectors.png'), bbox_inches='tight')
        plt.close()
        
                
        # 2. Correlation Matrix
        plt.figure(figsize=(12, 10))
        correlation_matrix = self.features.corr()
        sns.heatmap(correlation_matrix, cmap="coolwarm", annot=False, cbar=True)
        plt.title("Feature Correlation Matrix")
        plt.savefig(os.path.join(config.PLOTS_DIRECTORY, 'corrolation_matrix_of_feature_vectors.png'), bbox_inches='tight')
        plt.close()
        
        # # Create the directory for saving box plots if it does not exist
        # save_dir = os.path.join(config.SAVE_DIRECTORY, 'feature_boxplots')
        # os.makedirs(save_dir, exist_ok=True)
        
        # # Combine features and labels for plotting
        # numeric_features = self.features.select_dtypes(include=[np.number])  # Only numeric columns
        # plot_data = numeric_features.copy()
        # plot_data['class'] = self.labels  # Add class labels separately
    
        # # Iterate over each numeric column by position
        # for i in range(numeric_features.shape[1]):
        #     plt.figure(figsize=(8, 6))
        #     sns.boxplot(x='class', y=numeric_features.iloc[:, i], data=plot_data)
        #     plt.title(f'Box plot of feature {i+1} by class')
        #     plt.xlabel('Class')
        #     plt.ylabel('Values')
            
        #     # Save the figure with the feature index in the specified folder
        #     plt.savefig(os.path.join(save_dir, f'feature_{i+1}.png'), dpi=300)
        #     plt.close()
        
        # 4. PCA Plot for Dimensionality Reduction (2D)
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(self.features)
        
        # Encode class labels and create a color palette
        unique_labels = pd.Series(self.labels).unique()
        palette = sns.color_palette("viridis", len(unique_labels))
        label_to_color = {label: palette[i] for i, label in enumerate(unique_labels)}
        
        # Map labels to colors for plotting
        colors = [label_to_color[label] for label in self.labels]
    
        plt.figure(figsize=(10, 7))
        scatter = plt.scatter(
            pca_result[:, 0], pca_result[:, 1], 
            c=colors, alpha=0.7, edgecolors="k"
        )
        plt.title("PCA of Features (2D)")
        plt.xlabel("PCA Component 1")
        plt.ylabel("PCA Component 2")
        
        # Create custom legend
        handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10) 
                   for color in palette]
        plt.legend(handles, unique_labels, title="Class Labels", loc="best")
    
        plt.savefig(os.path.join(config.PLOTS_DIRECTORY, 'PCA_from_feature_vectors.png'), bbox_inches='tight')
        plt.close()
    


    def visualize_clusters(self):
        if self.is_distance_matrix:
            print("Skipping cluster visualization as input data is a distance matrix.")
            return
        
        """
        Visualize the results of the clustering along with the actual classes.
        """
        # Create a mapping for class labels to colors
        unique_classes = np.unique(self.labels)
        color_map = sns.color_palette("hsv", len(unique_classes))
        class_to_color = {cls: color for cls, color in zip(unique_classes, color_map)}

        # Mapping for cluster labels to markers
        markers = ['o', 'x', 's', '^', 'P', '*', '+']
        cluster_to_marker = {i: markers[i % len(markers)] for i in range(self.n_clusters)}

        plt.figure(figsize=(14, 10))
        for cls in unique_classes:
            for cluster in range(self.n_clusters):
                # Filter data by class and cluster using .loc for boolean indexing
                idx = (self.labels == cls) & (self.cluster_labels == cluster)
                plt.scatter(
                    self.features.loc[idx, self.features.columns[0]], 
                    self.features.loc[idx, self.features.columns[1]], 
                    color=class_to_color[cls], marker=cluster_to_marker[cluster],
                    label=f'Class {cls}, Cluster {cluster}' if np.any(idx) else ""
                )

        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.title('KMeans Clustering Results')
        plt.legend(title='Class, Cluster', loc='upper right', frameon=True)
        plt.grid(False)
        plt.savefig(os.path.join(config.PLOTS_DIRECTORY, 'kmeans_from_feature_vectors.png'), bbox_inches='tight')
        plt.close()

    def run_analysis(self):
        if self.is_distance_matrix:
            print("Analysis cannot proceed on a distance matrix.")
            return
        
        print("Performing Exploratory Data Analysis...")
        self.exploratory_analysis()
        print("Evaluating Clustering Metrics...")
        self.evaluate_clustering_metrics()
        print("Visualizing Clusters...")
        self.visualize_clusters()
        
        return self.features








# ==================================================


class DendrogramAnalysis:
    def __init__(self, data, neuron_class_df, save_directory = config.PLOTS_DIRECTORY, descriptor_name="", cutoff_dist = 0,  highlight_neurons=None, is_distance_matrix=True):
        """
        Initialize the DendrogramAnalysis with a distance matrix or feature vectors and neuron-class mapping.
        
        Args:
        - data (pd.DataFrame or np.array): Pairwise distance matrix or feature vectors DataFrame.
        - neuron_class_df (pd.DataFrame): DataFrame containing 'neuron_name' and 'class' columns.
        - descriptor_name (str): Name of the descriptor for labeling and saving plots.
        - method (str): The linkage method for hierarchical clustering (e.g., 'ward', 'complete').
        """
        self.data = data
        self.neuron_class_df = neuron_class_df
        self.descriptor_name = descriptor_name
        self.method = config.LINKAGE_METHOD
        self.labels = neuron_class_df['class'].values
        self.neuron_names = neuron_class_df['neuron_name'].values
        self.highlight_neurons = highlight_neurons
        self.mapping = None
        self.is_distance_matrix = is_distance_matrix
        self.cutoff_dist = cutoff_dist
        self.num_clusters = config.NUM_OF_CLUSTERS
        self.save_directory = save_directory

    


    '''
    1. Silhouette Score
    What It Measures: The Silhouette Score evaluates how similar an object is to 
    its own cluster compared to other clusters. The score is calculated for 
    each sample and can range from -1 to +1..
    
    Interpretation:
    
    +1 Score: The sample is far away from the neighboring clusters.
    
    0 Score: The sample is on or very close to the decision boundary between two neighboring clusters.
    
    -1 Score: The sample is placed in the wrong cluster.
    
    "Strong cluster structure" is indicated when the score is close to +1, which means clusters are dense and well-separated.
     =================================================================
    2. Adjusted Rand Index (ARI)
    What It Measures: The ARI measures the similarity between two clusterings 
    by considering all pairs of samples and counting pairs that are assigned in the same or
     different clusters in the predicted and true clusterings.
    
    Calculation Details:
    
    Unlike the Rand Index, ARI adjusts for the chance grouping of elements, 
    making it more robust and comparable across different datasets and clusterings.
    
    ARI can range from -1 to +1, where +1 indicates perfect agreement between two clusterings.
    
    Interpretation:
    
    "High agreement with true labels" is indicated when ARI is close to +1, showing that the clustering perfectly matches the true labels.
    
    Values closer to 0 or negative indicate random or poor clustering as compared to the true labels.
    ===================================================================
    3. Normalized Mutual Information (NMI)
    What It Measures: NMI is an adjustment of the Mutual Information (MI) score 
    that measures the mutual dependence between the two variables. 
    It is normalized to scale the result between 
    0 (no mutual information) and 1 (perfect correlation).
    
    Calculation Details:
    
    It compares how much information is shared between the clustering assignments 
    and the true classes; high values indicate a significant reduction
     in uncertainty about one variable given knowledge of the other.
    
    Interpretation:
    
    "Good mutual information" implies a high value (close to 1),
     indicating a strong association between cluster assignments and true classes.
    
    Lower values indicate less shared information, suggesting poor clustering 
    effectiveness with respect to the true classes.
    
    '''


    # def calculate_and_interpret_hierarchical_metrics(self, X, Z):
        
    #     X = X.select_dtypes(include=[np.number])
    #     # Assuming labels are stored in a column named 'class' in self.neuron_class_df
    #     label_encoder = LabelEncoder()
    #     labels = label_encoder.fit_transform(self.neuron_class_df['class'])
    
    #     # Calculate metrics
    #     silhouette = silhouette_score(X, fcluster(Z, t=2, criterion='maxclust')) if labels is not None else None
    #     ari = adjusted_rand_score(labels, fcluster(Z, t=2, criterion='maxclust')) if labels is not None else None
    #     nmi = normalized_mutual_info_score(labels, fcluster(Z, t=2, criterion='maxclust')) if labels is not None else None
    
    #     # Interpret metrics
    #     silhouette_interpretation = "Strong cluster structure" if silhouette > 0.5 else "Weak or overlapping clusters"
    #     ari_interpretation = "High agreement with true labels" if ari > 0.5 else "Low agreement with true labels"
    #     nmi_interpretation = "Good mutual information" if nmi > 0.5 else "Poor mutual information"
    
    #     return {
    #         'Silhouette': (silhouette, silhouette_interpretation),
    #         'ARI': (ari, ari_interpretation),
    #         'NMI': (nmi, nmi_interpretation)
    #     }
    
    def calculate_and_interpret_hierarchical_metrics(self, X, Z):
        from sklearn.preprocessing import LabelEncoder
        from scipy.cluster.hierarchy import fcluster
    
        # Convert DataFrame to numpy and drop rows with NaNs (must keep same row order as Z-based clusters)
        X_clean = pd.DataFrame(X).dropna()
        
        # Make sure the clustering labels match cleaned X
        valid_indices = X_clean.index
        cluster_labels = fcluster(Z, t=2, criterion='maxclust')
        cluster_labels = pd.Series(cluster_labels, index=X.index).loc[valid_indices].values
    
        label_encoder = LabelEncoder()
        labels_full = self.neuron_class_df['class'].values
    
        # Only keep labels corresponding to valid X rows
        labels = pd.Series(labels_full, index=self.neuron_class_df.index).loc[valid_indices].values
    
        X_clean = X_clean.values  # convert to numpy array
    
        # Now safely compute metrics
        silhouette = silhouette_score(X_clean, cluster_labels)
        ari = adjusted_rand_score(labels, cluster_labels)
        nmi = normalized_mutual_info_score(labels, cluster_labels)
        
        return {
            "Silhouette": (silhouette, "Higher = better separation"),
            "ARI": (ari, "Adjusted for random chance"),
            "NMI": (nmi, "Normalized agreement")
        }



    def plot_dendrogram(self):
        """
        Plot the dendrogram with a bar corresponding to the original classes.
        
        Args:
        - cutoff_dist (float): The distance threshold for coloring clusters in the dendrogram.
        """
        # Determine if the input is a distance matrix or feature vectors
        if isinstance(self.data, pd.DataFrame):
            if self.data.shape[0] == self.data.shape[1] and np.all(np.diag(self.data.values) == 0):
                print("Detected a distance matrix.")
                condensed_distance_matrix = squareform(self.data.values)
                Z = linkage(condensed_distance_matrix, method=self.method)
            else:
                print("Detected feature vectors.")
                feature_vectors = self.data.select_dtypes(include=[float, int])
                # Replace inf and NaN with a small number
                feature_vectors = feature_vectors.replace([np.inf, -np.inf], np.nan)
                feature_vectors = feature_vectors.fillna(1e-6)
                Z = linkage(feature_vectors.values, method=self.method)
                
                # feature_vectors = self.data.values
                # feature_vectors = self.data.select_dtypes(include=[float, int]).values
                # Z = linkage(feature_vectors, method=self.method)
        else:
            condensed_data = squareform(self.data) if self.is_distance_matrix else pdist(self.data)
            Z = linkage(condensed_data, method=self.method)
        
        self.mapping = self.cutoff_to_clusters(Z)
        # Call the internal method to plot the dendrogram with a color bar
        self._dendro_with_bar(Z)
    
    
    
    def _dendro_with_bar(self, Z):
        """
        Internal method to plot a dendrogram with a color bar corresponding to the original classes.
        
        Args:
        - Z: Linkage matrix for hierarchical clustering.
        - cutoff_dist (float): The distance threshold for coloring clusters.
        - highlight_neurons (list): Optional list of neuron names to highlight in a different color.
        """
    
     
        
        # Extract number of classes and total number of neurons
        num_of_original_classes = len(self.neuron_class_df['class'].unique())
        N = len(self.neuron_class_df)
    
        # Prepare the labels and colors for the bar
        label_encoder = LabelEncoder()
        df_labels = self.neuron_class_df.copy()
        df_labels['numeric'] = label_encoder.fit_transform(df_labels['class'])
    
        # # Create a color palette for the classes
        # palette = sns.color_palette("husl", n_colors=num_of_original_classes)
        # color_dict = {group: palette[i] for i, group in enumerate(df_labels["class"].unique())}
        # df_labels['color'] = df_labels['class'].map(color_dict)
        
        # Ensure that the number of unique classes does not exceed the number of available colors
        unique_classes = df_labels['class'].unique()
        if len(unique_classes) > len(config.COLORS_PALETTE):
            raise ValueError("Not enough colors in the palette for the number of classes.")
        
                # Create a color dictionary mapping each class to a color from the palette
        color_dict = {cls: config.COLORS_PALETTE[i % len(config.COLORS_PALETTE)] for i, cls in enumerate(unique_classes)}
        df_labels['color'] = df_labels['class'].map(color_dict)
        
        # color_dict = {cls: config.COLORS_PALLETE[i] for i, cls in enumerate(unique_classes)}
        # df_labels['color'] = df_labels['class'].map(color_dict)
    
        # Determine colors for neurons to highlight
        highlight_color = 'blue'  # Use a distinct color for highlighted neurons


        if self.highlight_neurons:
            # Only apply the highlight color to the bar without changing the legend colors
            df_labels['highlight'] = df_labels['neuron_name'].isin(self.highlight_neurons)
            df_labels['plot_color'] = np.where(df_labels['highlight'], highlight_color, df_labels['color'])
        else:
            df_labels['plot_color'] = df_labels['color']
    
        # Create the figure for the dendrogram
        # fig, ax = plt.figure(figsize=(8, 4))
        fig, ax = plt.subplots(figsize=(8, 4))

    
        # Determine the optimal cutoff distance if not provided
        if self.cutoff_dist == 0:
            distances = Z[:, 2]
            distance_gaps = np.diff(distances)
            max_gap_index = np.argmax(distance_gaps)
            self.cutoff_dist = distances[max_gap_index + 1]
            
        # Order the palette so that a given colour denotes the same anatomical
        # class in every dendrogram. scipy assigns palette entries to clusters in
        # left-to-right order, which follows the linkage structure and is
        # arbitrary with respect to anatomy, so the same colour can denote
        # different populations in two dendrograms built from different
        # representations. The palette is therefore permuted here so that the
        # first colour always goes to the cluster dominated by the first class in
        # sorted order.
        palette = list(config.DENDRO_CLUSTER_COLORS)
        try:
            probe = dendrogram(Z, no_plot=True, color_threshold=self.cutoff_dist)
            leaf_order = probe['leaves']
            memberships = fcluster(Z, t=len(unique_classes), criterion='maxclust')
            classes_by_leaf = df_labels['class'].to_numpy()
            leftmost_cluster = memberships[leaf_order[0]]
            in_leftmost = classes_by_leaf[memberships == leftmost_cluster]
            dominant = pd.Series(in_leftmost).value_counts().idxmax()
            # If the leftmost cluster is not dominated by the reference class,
            # swap the first two colours so the mapping is stable across figures.
            if dominant != sorted(unique_classes)[0]:
                palette[0], palette[1] = palette[1], palette[0]
        except Exception as error:  # noqa: BLE001
            print(f"Could not align dendrogram colours to class ({error}); "
                  f"using the palette order as given.")

        set_link_color_palette(palette)
        # Create the dendrogram without labels, but use the cutoff for color thresholding
        dend = dendrogram(Z, no_labels=True, color_threshold=self.cutoff_dist)
        
        
        # Calculate metrics and interpretations
        # metrics_with_interpretations = self.calculate_and_interpret_hierarchical_metrics(self.data, Z)
    
        # Extracting and formatting metric text from interpretations
        # metrics_text = "\n".join([f"{metric}: {values[0]:.2f} ({values[1]})" for metric, values in metrics_with_interpretations.items()])
        
        # # Annotate the plot with the metrics
        # ax.annotate(metrics_text, xy=(0.02, 0.98), xycoords='axes fraction', fontsize=12,
        #             verticalalignment='top', horizontalalignment='left', backgroundcolor='white', alpha=0.6)

        
        cluster_labels_colors = self.extract_cluster_labels_colors(dend)
        
        # Extract the color information for the bar
        # color = [df_labels.iloc[k]['plot_color'] for k in dend['leaves']]
        
        # Replace the relevant code for determining colors for the bar with:
        # Using the 'class_color' directly from your DataFrame aligned with the dendrogram leaves
        color = [self.neuron_class_df.iloc[leaf]['class_color'] for leaf in dend['leaves']]
        
        max_distance = Z[-1, 2]
    
        # Set the bar height as a proportion of max_distance (e.g., 20%)
        bar_height = 0.2 * max_distance
        bottom = -bar_height
        offset = 0.02 * max_distance  # Small offset for bar positioning
    
        # Create the bar that aligns with the leaves
        X = np.arange(N) * 10 + 5
        Y = np.ones(N) * (bar_height - offset)
        plt.bar(X, Y, bottom=bottom, width=10, color=color, edgecolor='none')
    
        # Get the labels for each leaf and align them with the bar
        labels = [self.neuron_class_df.iloc[leaf]['neuron_name'] for leaf in dend['leaves']]
        # for i, label in enumerate(labels):
        #     plt.text(X[i], bottom - offset, label, ha='center', va='top', rotation=90, fontsize=5)
    
        ax = plt.gca()
        for key, spine in ax.spines.items():
            spine.set_visible(False)
        
        ax.grid(False)
        plt.ylim(bottom, max_distance + 0.1 * max_distance)
        
        

        # Add legend for the class colors without including highlighted neurons
        label_name_num_map = dict(zip(df_labels['class'].unique(), color_dict.values()))  # Use original color_dict for legend
        handles = [plt.Rectangle((0, 0), 1, 1, color=label_name_num_map[label]) for label in label_name_num_map.keys()]
        labels_with_counts = [f"{label} ({(df_labels['class'] == label).sum()})" for label in label_name_num_map.keys()]
        plt.legend(handles, labels_with_counts, title='', title_fontsize='12', fontsize='12', loc='upper right',frameon=False)
    
        
    
        # Set the title and adjust the layout
        # plt.title(f'{self.descriptor_name} - Hierarchical Clustering Dendrogram', x=0.5, y=0.9)
        plt.tick_params(axis='y', labelsize=6)
        plt.tight_layout(rect=[0, 0.03, 1, 1])
        
        # save_folder = os.path.join(config.SAVE_DIRECTORY, "Dendrograms")
        # os.makedirs(save_folder, exist_ok=True)  # Create the folder if it doesn't exis
        # Save and show the dendrogram plot
        file_name = f"{self.descriptor_name}_dendrogram_with_bar.png" if self.descriptor_name else "dendrogram_with_bar.png"
        
        plt.savefig(os.path.join(self.save_directory, file_name), dpi=600, bbox_inches='tight', pad_inches=0.5)
        plt.close()
        
        
        
        if (self.is_distance_matrix):
            self.plot_pca_and_encircle_clusters()
            self.plot_pca_and_encircle_classes()
            self.plot_pca_with_class_decision_boundary()
            self.plot_pca_and_encircle_clusters_with_boundry_line()

        
 
    
        return self.neuron_class_df
    


    def extract_cluster_labels_colors(self, dend):
        from collections import defaultdict
        cluster_idxs = defaultdict(list)
        for color, pi in zip(dend['color_list'], dend['icoord']):
            for leg in pi[1:3]:  # consider only the middle two coordinates as cluster connections
                index = (leg - 5.0) / 10.0
                if abs(index - int(index)) < 1e-5:
                    cluster_idxs[color].append(int(index))
        
        labels = [self.neuron_class_df.iloc[leaf]['neuron_name'] for leaf in dend['leaves']]
        
        cluster_labels_indices = {color: [dend['leaves'][i] for i in indices] for color, indices in cluster_idxs.items()}
        cluster_labels_colors = {color: [labels[i] for i in indices] for color, indices in cluster_idxs.items()}

        # Update DataFrame with cluster assignments, colors, and indices
        cluster_number = 0
        for color, neuron_names in cluster_labels_colors.items():
            for name in neuron_names:
                idx = self.neuron_class_df[self.neuron_class_df['neuron_name'] == name].index
                self.neuron_class_df.loc[idx, 'dendrogram_cluster'] = cluster_number
                self.neuron_class_df.loc[idx, 'dendro_class_color'] = color
                # Set the dendrogram index for each neuron
                dend_idx = cluster_labels_indices[color][neuron_names.index(name)]
                self.neuron_class_df.loc[idx, 'cluster_labels_indices'] = dend_idx
            cluster_number += 1
        
        return cluster_labels_colors

    def cutoff_to_clusters(self, linkage_matrix, num_intervals=5):
        max_distance = max(linkage_matrix[:, 2])
        interval = max_distance / num_intervals
        cutoffs = np.linspace(0, max_distance, num_intervals + 1, endpoint=True)
    
        num_clusters = {}
    
        for cutoff in cutoffs:
            cutoff_str = str(round(cutoff, 2))  # Convert cutoff to string after rounding
            clusters = fcluster(linkage_matrix, t=float(cutoff_str), criterion='distance')  # Convert back to float for use
            num_clusters[cutoff_str] = len(np.unique(clusters))
    
        return num_clusters
    
    from sklearn.metrics import silhouette_score

    # def cutoff_to_clusters(self, linkage_matrix, num_intervals=5):
    #     max_distance = max(linkage_matrix[:, 2])
    #     interval = max_distance / num_intervals
    #     cutoffs = np.linspace(0, max_distance, num_intervals + 1, endpoint=True)
    
    #     cluster_quality = {}
    
    #     for cutoff in cutoffs:
    #         cutoff_str = str(round(cutoff, 2))
    #         clusters = fcluster(linkage_matrix, t=float(cutoff_str), criterion='distance')
    
    #         if len(set(clusters)) > 1 and len(set(clusters)) < len(clusters):
    #             score = silhouette_score(self.data, clusters, metric='precomputed')
    #         else:
    #             score = -1  # Invalid silhouette
    
    #         cluster_quality[cutoff_str] = {
    #             "num_clusters": len(np.unique(clusters)),
    #             "silhouette_score": round(score, 3)
    #         }
    
    #     return cluster_quality

    
    def plot_pca_and_encircle_clusters_with_boundry_line(self):
        from sklearn.linear_model import LogisticRegression
    
        # Perform MDS and PCA
        mds = MDS(n_components=2, dissimilarity="precomputed", random_state=42)
        mds_coords = mds.fit_transform(self.data)
        pca = PCA(n_components=2)
        pca_coords = pca.fit_transform(mds_coords)
        explained_var = pca.explained_variance_ratio_
        print(f"Explained variance by PC1 and PC2: {explained_var[0]:.2%}, {explained_var[1]:.2%}")

        sil_score = silhouette_score(self.data, self.neuron_class_df['dendrogram_cluster'], metric='precomputed')

        # Extract original class colors and dendrogram cluster assignments
        neuron_class_df = self.neuron_class_df.set_index('neuron_name').reindex(self.data.index)
        original_class_colors = neuron_class_df['class_color'].values
        dendrogram_labels = neuron_class_df['dendrogram_cluster'].values
    
        # Prepare color dictionary for dendrogram clusters using 'dendro_class_color'
        unique_clusters = np.unique(dendrogram_labels)
        cluster_color_dict = {
            cluster: neuron_class_df[neuron_class_df['dendrogram_cluster'] == cluster]['dendro_class_color'].iloc[0]
            for cluster in unique_clusters
        }
    
        # Plot PCA with class colors for dots
        fig, ax = plt.subplots(figsize=(10, 8))
        
        
        
    
        # Encircle and shade clusters using dendrogram cluster colors
        for cluster in unique_clusters:
            cluster_points = pca_coords[dendrogram_labels == cluster]
            if cluster_points.shape[0] > 2:
                hull = ConvexHull(cluster_points)
                poly = plt.Polygon(
                    cluster_points[hull.vertices], closed=True,
                    edgecolor=cluster_color_dict[cluster],
                    facecolor=cluster_color_dict[cluster],
                    alpha=0.2, linewidth=1.5, linestyle='--',zorder=1
                )
                ax.add_patch(poly)
                
        scatter = ax.scatter(
            pca_coords[:, 0], pca_coords[:, 1],
            c=original_class_colors, edgecolor='k', s=50, alpha=0.7, zorder=3
        )
        
        ax.set_xlim(pca_coords[:, 0].min() - 1000, pca_coords[:, 0].max() + 1000)
        ax.set_ylim(pca_coords[:, 1].min() - 1000, pca_coords[:, 1].max() + 1000)
        # ax.set_title(f"PCA Plot (Explained Variance: {explained_var[0]:.1%}, {explained_var[1]:.1%})", fontsize=14)
        # ax.set_title(f"PCA Plot\nExplained Variance: {explained_var[0]:.1%}, {explained_var[1]:.1%}\nSilhouette: {sil_score:.3f}", fontsize=14)

        # ==== Decision boundary between classes ====
        X = pca_coords
        y = neuron_class_df['class'].astype('category').cat.codes
        # clf = LogisticRegression().fit(X, y)
    
        # x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        # y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        # xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
        #                      np.linspace(y_min, y_max, 200))
        # Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    
        # ax.contour(xx, yy, Z, levels=[0.5], linewidths=2, colors='black', linestyles='--')
        
        clf = LogisticRegression().fit(X, y)

        # Manually shift the decision boundary left
        coef = clf.coef_[0]
        intercept = clf.intercept_[0] + 0.5  # ↓ Decrease this to shift left
        
        # Recompute decision boundary with shifted intercept
        x_vals = np.linspace(pca_coords[:, 0].min() - 1000, pca_coords[:, 0].max() + 1000, 200)
        y_vals = -(coef[0] / coef[1]) * x_vals - (intercept / coef[1])
        
        ax.plot(x_vals, y_vals, linestyle='--', color='black', linewidth=2, zorder=2)
        
        probs = clf.predict_proba(X)
        max_conf = probs.max(axis=1)
        mean_conf = np.mean(max_conf)
        print(f"Average class confidence: {mean_conf:.3f}")
        acc = clf.score(X, y)
        print(f"Classification accuracy in PCA space: {acc:.3f}")
        
        # Set title with all stats
        ax.set_title(
            f"PCA Plot\n"
            f"Explained Var: {explained_var[0]:.1%}, {explained_var[1]:.1%}  "
            f"Silhouette: {sil_score:.3f}\n"
            f"Avg Confidence: {mean_conf:.3f},  PCA Accuracy: {acc:.3f}",
            fontsize=14
        )

    
        # Set plot titles and labels
        # plt.title("")
        plt.xlabel("PC1", fontsize=16)
        plt.ylabel("PC2", fontsize=16)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
    
        # Legend for scatter points (original classes)
        unique_classes = neuron_class_df['class'].unique()
        class_color_dict = {
            cls: neuron_class_df[neuron_class_df['class'] == cls]['class_color'].iloc[0]
            for cls in unique_classes
        }
        scatter_handles = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10, label=cls)
            for cls, color in class_color_dict.items()
        ]
    
        # Legend for polygons (dendrogram clusters)
        cluster_handles = [
            plt.Line2D([0], [0], color=color, linewidth=2, linestyle='--', label=f"Cluster {i + 1}")
            for i, (cluster, color) in enumerate(cluster_color_dict.items())
        ]
    
        handles = scatter_handles + cluster_handles
        labels = [handle.get_label() for handle in handles]
        plt.legend(handles, labels, title="", loc='upper left', fontsize=10, frameon=False)
    
        file_name = f"PCA_from_dst_mtx_with_dendro_clusters_with_boundary_{self.descriptor_name}.png"
        plt.savefig(os.path.join(config.PLOTS_DIRECTORY, file_name), dpi=600, bbox_inches='tight', pad_inches=0.5)
        plt.close()


   


    def plot_pca_and_encircle_clusters(self):
        # Perform MDS and PCA
        mds = MDS(n_components=2, dissimilarity="precomputed", random_state=42)
        mds_coords = mds.fit_transform(self.data)
        pca = PCA(n_components=2)
        pca_coords = pca.fit_transform(mds_coords)
    
        # Extract original class colors and dendrogram cluster assignments
        neuron_class_df = self.neuron_class_df.set_index('neuron_name').reindex(self.data.index)
        original_class_colors = neuron_class_df['class_color'].values
        dendrogram_labels = neuron_class_df['dendrogram_cluster'].values
    
        # Prepare color dictionary for dendrogram clusters using 'dendro_class_color'
        unique_clusters = np.unique(dendrogram_labels)
        cluster_color_dict = {
            cluster: neuron_class_df[neuron_class_df['dendrogram_cluster'] == cluster]['dendro_class_color'].iloc[0]
            for cluster in unique_clusters
        }
    
        # Plot PCA with class colors for dots
        # fig, ax = plt.subplots(figsize=(12, 10))
        # Create the figure for the dendrogram
        fig, ax = plt.subplots(figsize=(6,4))
        
        scatter = ax.scatter(pca_coords[:, 0], pca_coords[:, 1], c=original_class_colors, edgecolor='k', s=50, alpha=0.7)
    
        # Encircle and shade clusters using dendrogram cluster colors from 'dendro_class_color'
        for cluster in unique_clusters:
            cluster_points = pca_coords[dendrogram_labels == cluster]
            if cluster_points.shape[0] > 2:
                hull = ConvexHull(cluster_points)
                poly = plt.Polygon(
                    cluster_points[hull.vertices], closed=True, edgecolor=cluster_color_dict[cluster],
                    facecolor=cluster_color_dict[cluster], alpha=0.2, linewidth=1.5, linestyle='--'
                )
                ax.add_patch(poly)
    
        # Set plot titles and labels
        plt.title("")
        plt.xlabel("PC1", fontsize=16)
        plt.ylabel("PC2", fontsize=16)
        plt.xticks(fontsize=8)
        plt.yticks(fontsize=9)
        plt.grid(False)
    
        # Create a custom legend
        # Legend for scatter points (original classes)
        unique_classes = neuron_class_df['class'].unique()
        class_color_dict = {
            cls: neuron_class_df[neuron_class_df['class'] == cls]['class_color'].iloc[0]
            for cls in unique_classes
        }
        scatter_handles = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10, label=cls)
            for cls, color in class_color_dict.items()
        ]
    
        # Legend for polygons (dendrogram clusters)
        cluster_handles = [
            plt.Line2D([0], [0], color=color, linewidth=2, linestyle='--', label=f"Cluster {i + 1}")
            for i, (cluster, color) in enumerate(cluster_color_dict.items())
        ]
    
        # Combine both legends 
        handles = scatter_handles + cluster_handles
        labels = [handle.get_label() for handle in handles]
        plt.legend(handles, labels, title="", loc='upper left', fontsize=10, frameon=False)
    
        # Optionally, save the plot
        file_name = f"PCA_from_dst_mtx_with_dendro_clusters_{self.descriptor_name}.png"
        plt.savefig(os.path.join(config.PLOTS_DIRECTORY, file_name), dpi=300, bbox_inches='tight', pad_inches=0.5)
        plt.close()  # or plt.show() if you wish to show the plot inline


    def plot_pca_and_encircle_classes(self):
        from sklearn.manifold import MDS
        from sklearn.decomposition import PCA
        from scipy.spatial import ConvexHull
        import matplotlib.pyplot as plt
        import numpy as np
        import os
    
        # MDS → PCA
        mds = MDS(n_components=2, dissimilarity="precomputed", random_state=42)
        mds_coords = mds.fit_transform(self.data)
        pca = PCA(n_components=2)
        pca_coords = pca.fit_transform(mds_coords)
    
        # Prepare class labels and colors
        neuron_class_df = self.neuron_class_df.set_index('neuron_name').reindex(self.data.index)
        class_labels = neuron_class_df['class'].values
        class_colors = neuron_class_df['class_color'].values
    
        # Class color map
        unique_classes = np.unique(class_labels)
        class_color_dict = {
            cls: neuron_class_df[neuron_class_df['class'] == cls]['class_color'].iloc[0]
            for cls in unique_classes
        }
    
        # Plot
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(pca_coords[:, 0], pca_coords[:, 1], c=class_colors, edgecolor='k', s=50, alpha=0.7)
    
        # Encircle by class
        for cls in unique_classes:
            mask = class_labels == cls
            points = pca_coords[mask]
            if points.shape[0] > 2:
                hull = ConvexHull(points)
                polygon = plt.Polygon(
                    points[hull.vertices],
                    closed=True,
                    edgecolor=class_color_dict[cls],
                    facecolor=class_color_dict[cls],
                    alpha=0.2,
                    linewidth=1.5,
                    linestyle='--'
                )
                ax.add_patch(polygon)
    
        # Labels and grid
        ax.set_xlabel("PC1", fontsize=16)
        ax.set_ylabel("PC2", fontsize=16)
        ax.set_title("")
        ax.tick_params(axis='x', labelsize=8)
        ax.tick_params(axis='y', labelsize=9)
        ax.grid(False)
    
        # Legend
        handles = [
            plt.Line2D([0], [0], marker='o', color='w', label=cls,
                       markerfacecolor=color, markersize=10)
            for cls, color in class_color_dict.items()
        ]
        ax.legend(handles=handles, title="", loc='upper left', fontsize=10, frameon=False)
    
        # Save
        file_name = f"PCA_from_dst_mtx_with_class_encircling_{self.descriptor_name}.png"
        plt.savefig(os.path.join(config.PLOTS_DIRECTORY, file_name), dpi=300, bbox_inches='tight', pad_inches=0.5)
        plt.close()

    def plot_pca_with_class_decision_boundary(self):
        from sklearn.manifold import MDS
        from sklearn.decomposition import PCA
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import LabelEncoder
        from scipy.spatial import ConvexHull
        import matplotlib.pyplot as plt
        import numpy as np
        import os
    
        # --- Step 1: MDS + PCA ---
        mds = MDS(n_components=2, dissimilarity="precomputed", random_state=42)
        mds_coords = mds.fit_transform(self.data)
        pca = PCA(n_components=2)
        pca_coords = pca.fit_transform(mds_coords)
    
        # --- Step 2: Class labels + colors ---
        neuron_class_df = self.neuron_class_df.set_index('neuron_name').reindex(self.data.index)
        class_labels = neuron_class_df['class'].values
        class_colors = neuron_class_df['class_color'].values
    
        # Create color dict for legend
        unique_classes = np.unique(class_labels)
        class_color_dict = {
            cls: neuron_class_df[neuron_class_df['class'] == cls]['class_color'].iloc[0]
            for cls in unique_classes
        }
    
        # --- Step 3: Plot base PCA scatter ---
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(pca_coords[:, 0], pca_coords[:, 1], c=class_colors, edgecolor='k', s=50, alpha=0.7)
    
        # --- Step 4: Logistic regression decision boundary ---
        le = LabelEncoder()
        y = le.fit_transform(class_labels)  # e.g., 'V1' → 0, 'V2' → 1
    
        if len(np.unique(y)) == 2:
            clf = LogisticRegression()
            clf.fit(pca_coords, y)
    
            # Meshgrid for decision boundary
            x_min, x_max = pca_coords[:, 0].min() - 1, pca_coords[:, 0].max() + 1
            y_min, y_max = pca_coords[:, 1].min() - 1, pca_coords[:, 1].max() + 1
            xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                                 np.linspace(y_min, y_max, 300))
            grid = np.c_[xx.ravel(), yy.ravel()]
            probs = clf.predict_proba(grid)[:, 1].reshape(xx.shape)
    
            # Contour = decision boundary (prob = 0.5)
            ax.contour(xx, yy, probs, levels=[0.5], linestyles='--', linewidths=2, colors='black')
        else:
            print("Skipping boundary: more than two classes.")
    
        # --- Step 5: Legend and layout ---
        handles = [
            plt.Line2D([0], [0], marker='o', color='w', label=cls,
                       markerfacecolor=color, markersize=10)
            for cls, color in class_color_dict.items()
        ]
        ax.legend(handles=handles, title=None, loc='upper left', fontsize=10, frameon=False)
    
        ax.set_xlabel("PC1", fontsize=16)
        ax.set_ylabel("PC2", fontsize=16)
        ax.set_title("")
        ax.tick_params(axis='x', labelsize=8)
        ax.tick_params(axis='y', labelsize=9)
        ax.grid(False)
    
        # --- Step 6: Save ---
        file_name = f"PCA_with_class_hyperplane_{self.descriptor_name}.png"
        plt.savefig(os.path.join(config.PLOTS_DIRECTORY, file_name), dpi=300, bbox_inches='tight', pad_inches=0.5)
        plt.close()
  
    



    # def plot_pca_and_encircle_clusters(self):
    #     # Perform MDS and PCA
    #     mds = MDS(n_components=2, dissimilarity="precomputed", random_state=42)
    #     mds_coords = mds.fit_transform(self.data)
    #     pca = PCA(n_components=2)
    #     pca_coords = pca.fit_transform(mds_coords)
    
    #     # Extract dendrogram cluster assignments
    #     dendrogram_labels = self.neuron_class_df.set_index('neuron_name').reindex(self.data.index)['dendrogram_cluster'].values
    
    #     # Ensure you have a color for each dendrogram cluster from config.COLORS_PALETTE
    #     unique_clusters = np.unique(dendrogram_labels)
    #     # Ensure the palette has enough colors or repeats colors if fewer are provided
    #     color_list = config.COLORS_PALETTE * (len(unique_clusters) // len(config.COLORS_PALETTE) + 1)
    #     cluster_color_dict = {cluster: color_list[i] for i, cluster in enumerate(unique_clusters)}
    
    #     # Plot PCA
    #     fig, ax = plt.subplots(figsize=(12, 10))
    #     # Assign colors to points based on cluster assignment
    #     point_colors = [cluster_color_dict[cluster] for cluster in dendrogram_labels]
    #     scatter = ax.scatter(pca_coords[:, 0], pca_coords[:, 1], c=point_colors, edgecolor='k', s=50, alpha=0.7)
    
    #     # Encircle and shade clusters with matching colors
    #     for cluster in unique_clusters:
    #         cluster_points = pca_coords[dendrogram_labels == cluster]
    #         if cluster_points.shape[0] > 2:
    #             hull = ConvexHull(cluster_points)
    #             poly = plt.Polygon(cluster_points[hull.vertices], edgecolor=cluster_color_dict[cluster], facecolor=cluster_color_dict[cluster], alpha=0.2, linewidth=1.5, linestyle='--')
    #             ax.add_patch(poly)
    
    #     plt.title("PCA with Dendrogram Clusters")
    #     plt.xlabel("Principal Component 1")
    #     plt.ylabel("Principal Component 2")
    #     plt.grid(False)
        
    #     # Optionally, save the plot
    #     file_name = f"PCA_from_dst_mtx_with_dendro_clusters_{self.descriptor_name}.png"
    #     plt.savefig(os.path.join(config.PLOTS_DIRECTORY, file_name), dpi=300, bbox_inches='tight', pad_inches=0.5)
    #     plt.close()

    # Suppose 'neuron_class_df' is your class DataFrame and 'df' is the DataFrame with values
    # # Call the function with your DataFrame
    # plot_cluster_boxplots(self.neuron_class_df, df)
  
    
  
  








    def run_analysis(self):
        """
        Run the complete analysis including plotting the dendrogram.
        
        Args:
        - cutoff_dist (float): Distance threshold for cutting the dendrogram.
        """
        self.plot_dendrogram()
        
        return self.neuron_class_df


###############################################################################
#############################   KEEP DO NOT DELETE   ##########################
###############################################################################

# def _dendro_with_bar(self, Z, cutoff_dist=0):
#     """
#     Internal method to plot a dendrogram with a color bar corresponding to the original classes.
    
#     Args:
#     - Z: Linkage matrix for hierarchical clustering.
#     - cutoff_dist (float): The distance threshold for coloring clusters.
#     """
#     # Extract number of classes and total number of neurons
#     num_of_original_classes = len(self.neuron_class_df['class'].unique())
#     N = len(self.neuron_class_df)

#     # Prepare the labels and colors for the bar
#     label_encoder = LabelEncoder()
#     df_labels = self.neuron_class_df.copy()
#     df_labels['numeric'] = label_encoder.fit_transform(df_labels['class'])

#     # Create a color palette for the classes
#     palette = sns.color_palette("husl", n_colors=num_of_original_classes)
#     color_dict = {group: palette[i] for i, group in enumerate(df_labels["class"].unique())}
#     df_labels['color'] = df_labels['class'].map(color_dict)

#     # Create the figure for the dendrogram
#     fig = plt.figure(figsize=(20, 10))

#     if cutoff_dist == 0:
#         cutoff_dist = Z[-num_of_original_classes + 1, 2]

#     # Create the dendrogram without labels, but use the cutoff for color thresholding
#     dend = dendrogram(Z, no_labels=True, color_threshold=cutoff_dist)

#     # Extract the color information for the bar
#     color = [df_labels.iloc[k]['color'] for k in dend['leaves']]
    
#     # extracts the largest linkage distance in the dendrogram from the linkage matrix Z
#     max_distance = Z[-1, 2]
    
#     bottom = -20  # Adjust this to position the bar higher or lower

#     # Create the bar that aligns with the leaves
#     X = np.arange(N) * 10 + 5  
#     # Increase the bar height- if the bar height is 20 (Y = np.ones(N) * 20 )
#     # and the bottom is -30 (bottom = -30 ) then the bar starts at -30 
#     # and goes up 20 so it ends up at -10
#     Y = np.ones(N) * 20 

#     # the width=10  is the width of teh small colored indicidual bars ..when it is wide enough it is all connected
#     plt.bar(X, Y, bottom=bottom, width=10, color=color, edgecolor='none')

#     # Get the labels for each leaf and align them with the bar
#     labels = [self.neuron_class_df.iloc[leaf]['neuron_name'] for leaf in dend['leaves']]
#     for i, label in enumerate(labels):
#         plt.text(X[i], bottom - 0.5, label, ha='center', va='top', rotation=90)

#     ax = plt.gca()
#     for key, spine in ax.spines.items():
#         spine.set_visible(False)

#     plt.gca().set_ylim((bottom - 0.05, None))

#     # Add legend for the class colors
#     label_name_num_map = dict(zip(df_labels['class'].unique(), df_labels['color'].unique()))
#     handles = [plt.Rectangle((0, 0), 1, 1, color=label_name_num_map[label]) for label in label_name_num_map.keys()]
#     labels_with_counts = [f"{label} ({(df_labels['class'] == label).sum()})" for label in label_name_num_map.keys()]
#     plt.legend(handles, labels_with_counts, title='Class', title_fontsize='16', fontsize='12', loc='upper right')

#     # Set the title and adjust the layout
#     plt.title(f'{self.descriptor_name} - Hierarchical Clustering Dendrogram', x=0.5, y=0.9)
#     plt.tick_params(axis='y', labelsize=12)
#     plt.tight_layout(rect=[0, 0.03, 1, 1])

#     # Save and show the dendrogram plot
#     file_name = f"{self.descriptor_name}_dendrogram_with_bar.png" if self.descriptor_name else "dendrogram_with_bar.png"
#     plt.savefig(os.path.join(config.SAVE_DIRECTORY, file_name), dpi=300, bbox_inches='tight', pad_inches=0.5)
#     plt.show()

#     # Create clusters and add them to neuron_class_df
#     clusters = fcluster(Z, cutoff_dist, criterion='distance')
#     self.neuron_class_df['cluster'] = clusters

#     # Add binary columns for each cluster
#     for i in range(1, clusters.max() + 1):
#         cluster_col_name = f'cluster_{i}'
#         self.neuron_class_df[cluster_col_name] = np.where(self.neuron_class_df['cluster'] == i, 1, 0)

#     return self.neuron_class_df