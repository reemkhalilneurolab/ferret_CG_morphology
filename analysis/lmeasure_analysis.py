import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu
import config
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import re
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import sys
PROJECT_ROOT = r"C:\Users\ahmad\Documents\GitHub\neurotopo"
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
from analysis.data_analysis import DataAnalysis, DendrogramAnalysis, Plotter
import neurotopo.utils as utl
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd


# L-Measure exports each metric as a Total, i.e. a sum over the objects it was
# measured on. For a quantity defined per compartment, per branch or per
# bifurcation, that sum equals the quantity's mean times the number of objects,
# so the column ends up reporting the object count rather than the quantity.
# Contraction, for instance, is a per-branch ratio averaging about 0.92, and its
# Total correlates with branch count at Spearman 0.995; dividing by the branch
# count reduces that to 0.222 and recovers the mean contraction.
#
# Left uncorrected, roughly a third of the columns are all reporting neuron size,
# which after z-scoring enters the Euclidean distance once per column.
#
# The denominators below record which objects each metric was measured over.
_PER_NEURON = ("Soma_Surface", "N_stems", "N_bifs", "N_branch", "N_tips",
               "Width", "Height", "Depth", "Length", "Surface", "Volume")
_PER_COMPARTMENT = ("Diameter", "Diameter_pow", "EucDistance", "PathDistance",
                    "Helix", "SectionArea", "Branch_Order", "Terminal_degree")
_PER_BRANCH = ("Branch_pathlength", "Contraction", "Fragmentation",
               "Taper_1", "Taper_2")
_PER_BIFURCATION = ("Bif_ampl_local", "Bif_ampl_remote", "Bif_tilt_local",
                    "Bif_tilt_remote", "Bif_torque_local", "Bif_torque_remote",
                    "Partition_asymmetry", "Pk", "Pk_2", "Pk_classic",
                    "Daughter_Ratio", "Parent_Daughter_Ratio",
                    "HillmanThreshold", "Diam_threshold", "Last_parent_diam")
_PER_TIP = ("TerminalSegment",)

LMEASURE_SCOPES = {}
for _group, _scope in ((_PER_NEURON, "per_neuron"),
                       (_PER_COMPARTMENT, "per_compartment"),
                       (_PER_BRANCH, "per_branch"),
                       (_PER_BIFURCATION, "per_bifurcation"),
                       (_PER_TIP, "per_tip")):
    for _metric in _group:
        LMEASURE_SCOPES[_metric] = _scope

# Columns that are not morphological measurements.
LMEASURE_NON_METRICS = {
    # Sum of SWC structure codes (soma 1, basal 3, apical 4). A bookkeeping
    # field recording which structure types were traced, not a measurement.
    "Type",
    # Reported as zero for all but three reconstructions.
    "Rall_Power",
    # No object count yields values of at least 1, so the exported total cannot
    # be resolved into a valid fractal dimension.
    "Fractal_Dim",
}


def to_per_object(numeric_data):
    """Convert L-Measure Totals to means over the objects they were measured on.

    Non-morphological columns are removed. Metrics already defined per neuron are
    returned unchanged. Total Fragmentation is compartments-per-branch summed
    over branches, so it also serves as the compartment count.
    """
    frame = numeric_data.rename(columns=lambda c: c.strip())
    counts = {
        "per_neuron": pd.Series(1.0, index=frame.index),
        "per_compartment": frame["Fragmentation"].astype(float),
        "per_branch": frame["N_branch"].astype(float),
        "per_bifurcation": frame["N_bifs"].astype(float),
        "per_tip": frame["N_tips"].astype(float),
    }
    converted = {}
    for metric in frame.columns:
        if metric in LMEASURE_NON_METRICS:
            continue
        scope = LMEASURE_SCOPES.get(metric)
        if scope is None:
            converted[metric] = frame[metric]
            continue
        denominator = counts[scope].replace(0, np.nan)
        converted[metric] = frame[metric].astype(float) / denominator
    return pd.DataFrame(converted, index=frame.index)


class LMeasureAnalysis:
    def __init__(self, neuron_class_df, lmeasure_data):
        """
        Initialize the class with neuron class data and L-Measure CSV file.

        Args:
        - neuron_class_df (pd.DataFrame): DataFrame containing neuron names and class labels.
        - lmeasure_csv (str): Path to the L-Measure CSV file.
        """
        self.neuron_class_df = neuron_class_df
        self.lmeasure_data = lmeasure_data
        self.n_clusters = config.NUM_OF_CLUSTERS
        self.output_directory = config.LMEASURE_FIGURES  # Output directory set in config
        self.pvalues_df = None  # Placeholder for statistical results
        self.pca_df = None
        self.cluster_labels = None
        self.kmeans_cluster_df = None
        self.feature_importance_df = None
        self.cluster_quality_metrics = None
        self.lmeasure_data_standarize = config.LMEASURE_DATA_STANDARIZE
        self.data, self.data_not_statndarized = self._load_and_clean_data()

    def _load_and_clean_data(self):
        """Loads L-Measure data, cleans filenames, removes excluded neurons, and keeps all columns."""
        print("Loading and cleaning L-Measure data...")

        # data = pd.read_csv(self.lmeasure_csv)
        
        data = self.lmeasure_data

        # Remove ".swc" at the end and clean filenames
        data['Filename'] = data['Filename'].apply(lambda x: x[:-4] if x.endswith('.swc') else x)
        data['Filename'] = data['Filename'].apply(lambda x: x.replace('.CNG', '').replace(' ', '_').replace('.', '_').replace('-', '_'))

        # Set Filename as index
        
        data.set_index("Filename", inplace=True)

        # Remove excluded neurons
        if hasattr(config, "EXCLUDED_NEURONS") and isinstance(config.EXCLUDED_NEURONS, list):
            data = data.drop(index=config.EXCLUDED_NEURONS, errors='ignore')
        
        # Split numeric / non-numeric
        numeric_data = data.select_dtypes(include=['number'])
        non_numeric_data = data.drop(columns=numeric_data.columns)

        # Express each metric per object measured rather than as an export total
        numeric_data = to_per_object(numeric_data)

        # Clean numeric data
        if config.CLEAN_DATA == True:
            numeric_data = utl.drop_constant_and_low_variance_columns(numeric_data)
        
        
        # data_not_statndarized = pd.DataFrame(
        #     numeric_data,
        #     index=numeric_data.index,
        #     columns=numeric_data.columns
        # )
        
        data_not_statndarized = pd.concat([non_numeric_data, numeric_data], axis=1)
        
        data_not_statndarized.columns = data_not_statndarized.columns.str.strip()
        
       
        # Optionally standardize
        if self.lmeasure_data_standarize is True and not numeric_data.empty:
            if config.LMEASURE_NORMALIZE_METHOD == "standard":
                scaler = StandardScaler()
            elif config.LMEASURE_NORMALIZE_METHOD  == "minmax":
                scaler = MinMaxScaler()
            elif config.LMEASURE_NORMALIZE_METHOD  == "robust":
                scaler = RobustScaler()
            else:
                raise ValueError(f"Unsupported standardization method: {config.LMEASURE_NORMALIZE_METHOD }")
        
            numeric_data = pd.DataFrame(
                scaler.fit_transform(numeric_data),
                index=numeric_data.index,
                columns=numeric_data.columns
            )
            
        # Combine cleaned numeric and non-numeric
        data = pd.concat([non_numeric_data, numeric_data], axis=1)
        data.columns = data.columns.str.strip()
        print("Data cleaning complete. Excluded neurons:", config.EXCLUDED_NEURONS)
        return data, data_not_statndarized
    
    '''
    Step-by-step Breakdown
    PCA transforms your data into a new coordinate system
    Each row (neuron) gets a new set of values (scores) in the space defined by the top principal components.
    
    These new values are called principal component scores
    They're stored in the matrix returned by PCA().fit_transform(...) → shape: (n_neurons, n_components)
    
    You can now measure how far each neuron (row) lies from the origin in that new PC space
    → This reflects how “informative” or “different” that neuron is, based on overall variance.
    
    '''

    def select_top_neurons_by_group(self, group_name ='V1' , n=51, n_components=3):
        """
        Selects top `n` neurons from a specific group based on PCA variance.
    
        Parameters:
        - df (DataFrame): Index = neuron names, includes numeric features and 'group' column
        - group_name (str): Name of the group to filter
        - group_column (str): Column name used for grouping (default = 'group')
        - n (int): Number of neurons to select
        - n_components (int): Number of PCA components to use for scoring
    
        Returns:
        - DataFrame with top `n` neurons from the group
        """
        # Filter the DataFrame by the specified group
        group_df = self.data[self.data['group'] == group_name]
    
        if group_df.empty:
            raise ValueError(f"No neurons found in group '{group_name}'.")
    
        # Keep only numeric features for PCA
        numeric_df = group_df.select_dtypes(include=[np.number]).dropna()
    
        if numeric_df.shape[0] < n:
            raise ValueError(f"Group '{group_name}' only has {numeric_df.shape[0]} valid neurons. Can't select {n}.")
    
        # Run PCA on numeric data
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(numeric_df)
    
        # Score each neuron based on projection magnitude
        scores = np.linalg.norm(X_pca, axis=1)
        top_indices = np.argsort(scores)[-n:][::-1]
    
        # Select top neurons and reattach non-numeric columns
        selected = group_df.loc[numeric_df.iloc[top_indices].index]
        
        # Remove the entire group from the original dataset
        df_without_group = self.data[self.data['group'] != group_name]

        # Combine the rest with selected top neurons
        updated_df = pd.concat([df_without_group, selected])
        
        # Update the original data
        self.data = updated_df.sort_index()
        self.neuron_class_df = self.neuron_class_df[self.neuron_class_df['neuron_name'].isin(self.data.index)].copy()




    # def create_grouped_boxplots(self):
    #     """
    #     Generate box plots for each numeric column grouped by 'group' and perform Mann-Whitney U test.

    #     Returns:
    #     - pvalues_df (pd.DataFrame): DataFrame with p-values and statistical significance.
    #     """
    #     print("Creating grouped boxplots...")

    #     # Ensure 'group' column exists
    #     if 'group' not in self.data_not_statndarized.columns:
    #         raise ValueError("The dataset must contain a 'group' column for grouping.")

    #     # Identify numeric columns
    #     numeric_columns = self.data_not_statndarized.select_dtypes(include=['number']).columns.tolist()

    #     # Define colors
    #     # palette_colors = ['#f77189', '#36ada4']
    #     palette_colors = ['#3274a0', '#e1812d']
        

    #     # Store p-values
    #     pvalues_data = []

    #     for column in numeric_columns:
    #         plt.figure(figsize=(8, 6))  # Set figure size
    #         ax = sns.boxplot(x='group', 
    #                          y=column, 
    #                          data=self.data_not_statndarized , 
    #                          showfliers=False, 
    #                          palette=palette_colors,
    #                          notch=True,  # Enable notches
    #                          width=0.3  # r
    #                          )

    #         # Perform Mann-Whitney U test
    #         group1 = self.data_not_statndarized[self.data_not_statndarized['group'] == self.data_not_statndarized['group'].unique()[0]][column]
    #         group2 = self.data_not_statndarized[self.data_not_statndarized['group'] == self.data_not_statndarized['group'].unique()[1]][column]
    #         stat, p_value = mannwhitneyu(group1, group2, alternative='two-sided')

    #         # Determine significance
    #         significance = "Significant" if p_value < 0.05 else "Not Significant"
    #         pvalues_data.append({"Column Name": column, "P-Value": p_value, "Significance": significance})

    #         # Update title with p-value
    #         ax.set_title(f'p-value: {p_value:.3e}', fontsize=16)
    #         # ax.set_title(f'Box Plot of {column} (p-value: {p_value:.3e})', fontsize=16)
    #         ax.set_xlabel('Group', fontsize=18)
    #         ax.set_ylabel(column, fontsize=18)
    #         ax.tick_params(axis='x', labelsize=16)
    #         ax.tick_params(axis='y', labelsize=16)
    #         plt.grid(False)
    #         plt.tight_layout()

    #         # Save plot
    #         plot_filename = os.path.join(self.output_directory, f"{column}_boxplot.png")
    #         plt.savefig(plot_filename)
    #         plt.close()

    #     # Store p-values in a DataFrame
    #     self.pvalues_df = pd.DataFrame(pvalues_data)
    #     print("Boxplots created and saved in:", self.output_directory)

    #     return self.pvalues_df 

    def create_grouped_boxplots(self):
         """
         Manually control box spacing for each numeric column grouped by 'group' using matplotlib.
         """
         print("Creating tightly controlled grouped boxplots...")
    
         if 'group' not in self.data_not_statndarized.columns:
             raise ValueError("The dataset must contain a 'group' column for grouping.")
    
         numeric_columns = self.data_not_statndarized.select_dtypes(include=['number']).columns.tolist()
         group_labels = self.data_not_statndarized['group'].unique()
         colors = ['#3274a0', '#e1812d']
         position_gap = 0.4
    
         pvalues_data = []
    
         for column in numeric_columns:
             group1_vals = self.data_not_statndarized[self.data_not_statndarized['group'] == group_labels[0]][column]
             group2_vals = self.data_not_statndarized[self.data_not_statndarized['group'] == group_labels[1]][column]
    
             data = [group1_vals.dropna(), group2_vals.dropna()]
             positions = [1, 1 + position_gap]
    
             fig, ax = plt.subplots(figsize=(5, 4))
             bplot = ax.boxplot(
                 data,
                 positions=positions,
                 widths=0.3,
                 patch_artist=True,
                 notch=True,
                 showfliers=False,
                 medianprops=dict(color='black', linewidth=2)
             )
    
             for patch, color in zip(bplot['boxes'], colors):
                 patch.set_facecolor(color)
                 patch.set_edgecolor('black')
    
             ax.set_xticks(positions)
             ax.set_xticklabels(group_labels, fontsize=12)
             ax.set_ylabel(column, fontsize=12)
    
             stat, p_value = mannwhitneyu(data[0], data[1], alternative='two-sided')
             pvalues_data.append({"Column Name": column, "P-Value": p_value, "Significance": "Significant" if p_value < 0.05 else "Not Significant"})
    
             # ax.set_title(f'{column}\nMann-Whitney U p = {p_value:.2e}', fontsize=13)
             ax.set_title(f'p-value = {p_value:.2e}', fontsize=13)
             ax.grid(False)
             plt.tight_layout()
    
             filename = os.path.join(self.output_directory, f"{column}_tight_boxplot.png")
             plt.savefig(filename, dpi=300)
             plt.close()
    
         self.pvalues_df = pd.DataFrame(pvalues_data)
         return self.pvalues_df
    

    
    def create_box_plot_for_selected_features(self, features):
        """
        Generate box plots for selected numeric columns grouped by 'group' and include p-values formatted to two significant figures.

        Parameters:
        - features (list): List of features to plot along with the 'group' column.
        """
        print("Creating box plot for selected features with p-values...")

        if 'group' not in self.data_not_statndarized.columns:
            raise ValueError("The dataset must contain a 'group' column for grouping.")

        missing_cols = [feature for feature in features if feature not in self.data_not_statndarized.columns]
        if missing_cols:
            raise ValueError(f"The following specified features do not exist in the dataset: {missing_cols}")

        features.append('group')
        selected_data = self.data_not_statndarized[features]

        melted_df = selected_data.melt(id_vars=['group'], var_name='descriptor', value_name='value')

        plt.figure(figsize=(12, 6))
        ax = sns.boxplot(
            data=melted_df,
            x="descriptor",
            y="value",
            hue="group",
            palette="tab10",
            showfliers=False,
            notch=True,  # Enable notches
            width=0.5  # Adjust box width, make it thinner
        )

        # Calculate and annotate p-values
        descriptors = melted_df['descriptor'].unique()
        for i, descriptor in enumerate(descriptors):
            group1 = melted_df[(melted_df['descriptor'] == descriptor) & (melted_df['group'] == 'control')]['value']
            group2 = melted_df[(melted_df['descriptor'] == descriptor) & (melted_df['group'] == 'susceptible')]['value']
            stat, p_value = mannwhitneyu(group1, group2, alternative='two-sided')
            y_max = max(group1.max(), group2.max())
            # Format p-value to two significant figures
            ax.text(i, y_max, f'p={p_value:.2e}', horizontalalignment='center', color='black', fontsize=12)

        plt.title("Distribution of Selected Features by Group with P-values")
        plt.xlabel("Feature")
        plt.ylabel("Value")
        plt.xticks(rotation=45)
        plt.legend(title="Group")
        plt.tight_layout()

        plot_filename = os.path.join(self.output_directory, "selected_features_boxplot.png")
        plt.savefig(plot_filename, dpi=300)
        plt.close()

        print(f"Box plot with p-values for selected features created and saved to {plot_filename}")
    
    def create_violin_plot_for_selected_features(self, features):
        """
        Generate a violin plot for selected numeric columns grouped by 'group' and include p-values.

        Parameters:
        - features (list): List of features to plot along with the 'group' column.
        """
        print("Creating violin plot for selected features with p-values...")

        if 'group' not in self.data_not_statndarized.columns:
            raise ValueError("The dataset must contain a 'group' column for grouping.")

        missing_cols = [feature for feature in features if feature not in self.data_not_statndarized.columns]
        if missing_cols:
            raise ValueError(f"The following specified features do not exist in the dataset: {missing_cols}")

        features.append('group')
        selected_data = self.data_not_statndarized[features]

        melted_df = selected_data.melt(id_vars=['group'], var_name='descriptor', value_name='value')

        plt.figure(figsize=(12, 6))
        ax = sns.violinplot(
            data=melted_df,
            x="descriptor",
            y="value",
            hue="group",
            split=True,
            palette="tab10",
            inner="quartile"
        )

        # Calculate and annotate p-values
        descriptors = melted_df['descriptor'].unique()
        for i, descriptor in enumerate(descriptors):
            group1 = melted_df[(melted_df['descriptor'] == descriptor) & (melted_df['group'] == 'control')]['value']
            group2 = melted_df[(melted_df['descriptor'] == descriptor) & (melted_df['group'] == 'susceptible')]['value']
            stat, p_value = mannwhitneyu(group1, group2, alternative='two-sided')
            y_max = max(group1.max(), group2.max())
            ax.text(i, y_max+10, f'p={p_value:.2e}', horizontalalignment='center', color='black', fontsize=12)

        plt.title("Distribution of Selected Features by Group")
        plt.xlabel("Feature")
        plt.ylabel("Value")
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles=handles, labels=labels, loc='best', frameon=False)

        plot_filename = os.path.join(self.output_directory, "selected_features_violinplot.png")
        plt.savefig(plot_filename, dpi=300)
        plt.close()

        print(f"Violin plot with p-values for selected features created and saved to {plot_filename}")

    




    
    def perform_pca_and_plot(self):
        """
        Perform PCA on numeric columns and create a scatter plot of the first two principal components,
        colored by the 'group' column.
    
        Returns:
        - pca_df (pd.DataFrame): DataFrame containing PCA results with 'PC1', 'PC2', and 'group' columns.
        """
        print("Performing PCA...")
    
        # Ensure 'group' column exists
        if 'group' not in self.data.columns:
            raise ValueError("The dataset must contain a 'group' column for grouping.")
    
        # Extract numeric columns for PCA
        numeric_columns = self.data.select_dtypes(include=['number']).columns.tolist()
    
        # Perform PCA (reduce to 2 components for visualization)
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(self.data[numeric_columns])
    
        # Create a DataFrame for the PCA results
        self.pca_df = pd.DataFrame(data=pca_result, columns=['PC1', 'PC2'], index=self.data.index)
        self.pca_df['group'] = self.data['group']
    
        # Create the PCA scatter plot
        plt.figure(figsize=(10, 8))
        palette_colors = ['#f77189', '#36ada4']
        sns.scatterplot(
            x='PC1', y='PC2', hue='group', data=self.pca_df, palette=palette_colors, s=100, alpha=0.8
        )
        plt.title('PCA Scatter Plot', fontsize=18)
        plt.xlabel('Principal Component 1', fontsize=14)
        plt.ylabel('Principal Component 2', fontsize=14)
        plt.legend(title='Group', fontsize=12, title_fontsize=14)
        plt.grid(False)  # Remove the grid for cleaner visuals
    
        # Save the PCA plot
        pca_plot_file = os.path.join(self.output_directory, "pca_scatter_plot.png")
        plt.savefig(pca_plot_file)
        plt.close()  # Close the plot to free up memory
    
        print(f"PCA plot saved at: {pca_plot_file}")
    
        return self.pca_df
    
    '''
    1. Silhouette Score
    What It Measures: The Silhouette Score assesses how well an object is matched 
    to its own cluster compared to other clusters. 
    A high value suggests that the object is well matched to its own cluster 
    and poorly matched to neighboring clusters.
    
            Interpretation Criteria:
    
    Good separation (> 0.7): Clusters are well-separated and clearly distinct 
    from each other. This indicates that the clustering configuration is appropriate and robust.
    
    Fair separation (0.25 to 0.7): Clusters are moderately well-separated 
    but might not be very distinct, suggesting a reasonable but not ideal clustering structure.
    
    Poor separation (< 0.25): Clusters overlap significantly, indicating that 
    the clustering algorithm struggled to discriminate between different clusters effectively.
    ===============================================================================
    2. Calinski-Harabasz Index
    What It Measures: This index evaluates cluster validity based on 
    the ratio of the sum of between-clusters dispersion and of within-cluster
    dispersion for all clusters. Essentially, it's a measure of cluster density and separation.
    
            Interpretation Criteria:
    
    Well-defined clusters (> 500): A high score suggests that the clusters are 
    dense and well-separated, which is indicative of a successful clustering outcome.
    
    Less distinct clusters (<= 500): A lower score indicates that the clusters 
    are not very dense or that the separation between different clusters is not clear. 
    This may suggest that the clustering configuration could be improved, 
    either by selecting better features or by tuning the clustering algorithm.
    =====================================================================
    3. Davies-Bouldin Index
    What It Measures: The Davies-Bouldin Index is based on a ratio of within-cluster 
    scatter to between-cluster separation. It seeks to identify sets of clusters 
    that are compact and well-separated.
    
            Interpretation Criteria:
    
    Compact and well-separated clusters (< 0.5): Indicates that the clusters are
    compact and significantly distanced from each other, which is ideal in clustering.
    
    Poor clustering (> 0.5): Suggests that the clusters have high within-cluster 
    scatter or low between-cluster separation, indicating suboptimal clustering.
    
    Practical Applications and Interpretations
    Silhouette Score: Provides insight into how appropriately data has been clustered.
    If the score is low, it might be necessary to consider increasing the number of 
    clusters or revising the clustering approach.
    
    Calinski-Harabasz Index: Useful for comparing the effectiveness of different 
    clustering schemes or when deciding the optimal number of clusters, particularly 
    because it balances cluster density with separation.
    
    Davies-Bouldin Index: Helps in identifying the best clustering algorithm 
    or settings as it directly penalizes models that produce diffuse clusters 
    or those that place clusters too close to each other.
    '''

    def calculate_clustering_metrics(self, data, labels):
        silhouette = silhouette_score(data, labels)  # higher is better, range -1 to 1
        calinski = calinski_harabasz_score(data, labels)  # higher is better
        davies = davies_bouldin_score(data, labels)  # lower is better, best if close to 0
        
        return silhouette, calinski, davies
    
    def interpret_metrics(self, silhouette, calinski, davies):
        interpretations = {}
        interpretations['Silhouette'] = "Good separation" if silhouette > 0.7 else "Poor separation" if silhouette < 0.25 else "Fair separation"
        interpretations['Calinski'] = "Well-defined clusters" if calinski > 500 else "Less distinct clusters"
        interpretations['Davies'] = "Compact and well-separated clusters" if davies < 0.5 else "Poor clustering"
        
        return interpretations
    

  
    def perform_kmeans_clustering_and_plot(self):
        print(f"Performing KMeans clustering with k={self.n_clusters}...")
    
        if 'group' not in self.data.columns:
            raise ValueError("The dataset must contain a 'group' column for grouping.")
    
        numeric_columns = self.data.select_dtypes(include=['number']).columns.tolist()
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(self.data[numeric_columns])
    
        # Calculate metrics and get interpretations
        silhouette, calinski, davies = self.calculate_clustering_metrics(self.data[numeric_columns], cluster_labels)
        interpretations = self.interpret_metrics(silhouette, calinski, davies)
    
        # Perform PCA for visualization
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(self.data[numeric_columns])
        self.kmeans_cluster_df = pd.DataFrame(pca_result, columns=['PC1', 'PC2'], index=self.data.index)
        self.kmeans_cluster_df['group'] = self.data['group']
        self.kmeans_cluster_df['cluster'] = cluster_labels
    
        # Plotting
        plt.figure(figsize=(10, 8))
        palette = sns.color_palette("husl", len(self.kmeans_cluster_df['group'].unique()))
        markers = ['o', 's', '^', '<', '>']
    
        for i, group in enumerate(self.kmeans_cluster_df['group'].unique()):
            for j, cluster in enumerate(self.kmeans_cluster_df['cluster'].unique()):
                mask = (self.kmeans_cluster_df['group'] == group) & (self.kmeans_cluster_df['cluster'] == cluster)
                sns.scatterplot(
                    x='PC1', y='PC2',
                    data=self.kmeans_cluster_df[mask],
                    color=palette[i],
                    marker=markers[j],
                    label=f'{group}, Cluster: {cluster}',
                    s=100, alpha=0.8
                )
    
        # Annotations with clustering metrics and their interpretations
        metrics_text = (f"Silhouette: {silhouette:.2f} ({interpretations['Silhouette']})\n"
                        f"Calinski-Harabasz: {calinski:.2f} ({interpretations['Calinski']})\n"
                        f"Davies-Bouldin: {davies:.2f} ({interpretations['Davies']})")
        plt.annotate(metrics_text, xy=(0.05, 0.95), xycoords='axes fraction', fontsize=12,
                     verticalalignment='top')  # Removed bbox styling for no border

    
        plt.title(f'KMeans Clustering (k={self.n_clusters}) with PCA', fontsize=18)
        plt.xlabel('Principal Component 1', fontsize=14)
        plt.ylabel('Principal Component 2', fontsize=14)
        plt.legend(fontsize=12, title_fontsize=14, bbox_to_anchor=(1.05, 1), loc='upper right')
        plt.grid(False)
    
        # Save the plot
        kmeans_plot_file = os.path.join(self.output_directory, "kmeans_scatter_plot.png")
        plt.savefig(kmeans_plot_file, bbox_inches='tight')
        plt.close()
    
        print(f"KMeans clustering plot saved at: {kmeans_plot_file}")
        return self.kmeans_cluster_df






    # def perform_kmeans_clustering_and_plot(self):
    #     print(f"Performing KMeans clustering with k={self.n_clusters}...")
    
    #     # Ensure 'group' column exists
    #     if 'group' not in self.data.columns:
    #         raise ValueError("The dataset must contain a 'group' column for grouping.")
    
    #     # Extract numeric columns for clustering
    #     numeric_columns = self.data.select_dtypes(include=['number']).columns.tolist()
    
    #     # Perform KMeans clustering
    #     kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
    #     cluster_labels = kmeans.fit_predict(self.data[numeric_columns])
    
    #     # Calculate clustering quality metrics
    #     silhouette = silhouette_score(self.data[numeric_columns], cluster_labels)
    #     calinski = calinski_harabasz_score(self.data[numeric_columns], cluster_labels)
    #     davies = davies_bouldin_score(self.data[numeric_columns], cluster_labels)
    
    #     # Perform PCA for visualization
    #     pca = PCA(n_components=2)
    #     pca_result = pca.fit_transform(self.data[numeric_columns])
    
    #     # Store PCA results and cluster assignments in a DataFrame
    #     self.kmeans_cluster_df = pd.DataFrame(data=pca_result, columns=['PC1', 'PC2'], index=self.data.index)
    #     self.kmeans_cluster_df['group'] = self.data['group']
    #     self.kmeans_cluster_df['cluster'] = cluster_labels
    
    #     # Plotting
    #     plt.figure(figsize=(10, 8))
    #     unique_groups = self.kmeans_cluster_df['group'].unique()
    #     unique_clusters = self.kmeans_cluster_df['cluster'].unique()
    #     palette = sns.color_palette("husl", len(unique_groups))
    #     markers = ['o', 's', '^', '<', '>']
    
    #     for i, group in enumerate(unique_groups):
    #         for j, cluster in enumerate(unique_clusters):
    #             mask = (self.kmeans_cluster_df['group'] == group) & (self.kmeans_cluster_df['cluster'] == cluster)
    #             sns.scatterplot(
    #                 x='PC1', y='PC2',
    #                 data=self.kmeans_cluster_df[mask],
    #                 color=palette[i],
    #                 marker=markers[j],
    #                 label=f'{group}, Cluster: {cluster}',
    #                 s=100, alpha=0.8
    #             )
    
    #     # Annotations with clustering metrics
    #     metrics_text = f'Silhouette: {silhouette:.2f}, Calinski-Harabasz: {calinski:.2f}, Davies-Bouldin: {davies:.2f}'
    #     plt.annotate(metrics_text, xy=(0.05, 0.95), xycoords='axes fraction', fontsize=12,
    #                  bbox=dict(boxstyle="round,pad=0.3", edgecolor='darkgray', facecolor='none'))
    
    #     plt.title(f'KMeans Clustering (k={self.n_clusters}) with PCA', fontsize=18)
    #     plt.xlabel('Principal Component 1', fontsize=14)
    #     plt.ylabel('Principal Component 2', fontsize=14)
    #     plt.legend(fontsize=12, title_fontsize=14, bbox_to_anchor=(1.05, 1), loc='upper right')
    #     plt.grid(False)
    
    #     # Save the plot
    #     kmeans_plot_file = os.path.join(self.output_directory, "kmeans_scatter_plot.png")
    #     plt.savefig(kmeans_plot_file, bbox_inches='tight')
    #     plt.close()
    
    #     print(f"KMeans clustering plot saved at: {kmeans_plot_file}")
    #     return self.kmeans_cluster_df




    # def perform_kmeans_clustering_and_plot(self):
    #     print(f"Performing KMeans clustering with k={self.n_clusters}...")
    
    #     # Ensure 'group' column exists
    #     if 'group' not in self.data.columns:
    #         raise ValueError("The dataset must contain a 'group' column for grouping.")
    
    #     # Extract numeric columns for clustering
    #     numeric_columns = self.data.select_dtypes(include=['number']).columns.tolist()
    
    #     # Perform KMeans clustering
    #     from sklearn.cluster import KMeans
    #     kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
    #     cluster_labels = kmeans.fit_predict(self.data[numeric_columns])
    
    #     # Perform PCA for visualization
    #     from sklearn.decomposition import PCA
    #     pca = PCA(n_components=2)
    #     pca_result = pca.fit_transform(self.data[numeric_columns])
    
    #     # Store PCA results and cluster assignments in a DataFrame
    #     self.kmeans_cluster_df = pd.DataFrame(data=pca_result, columns=['PC1', 'PC2'], index=self.data.index)
    #     self.kmeans_cluster_df['group'] = self.data['group']
    #     self.kmeans_cluster_df['cluster'] = cluster_labels
    
    #     # Plotting
    #     plt.figure(figsize=(10, 8))
    #     unique_groups = self.kmeans_cluster_df['group'].unique()
    #     unique_clusters = self.kmeans_cluster_df['cluster'].unique()
    #     palette = sns.color_palette("husl", len(unique_groups))
    #     markers = ['o', 's', '^', '<', '>']
    
    #     # Plot each group with different colors and each cluster with different markers
    #     for i, group in enumerate(unique_groups):
    #         for j, cluster in enumerate(unique_clusters):
    #             mask = (self.kmeans_cluster_df['group'] == group) & (self.kmeans_cluster_df['cluster'] == cluster)
    #             sns.scatterplot(
    #                 x='PC1', y='PC2',
    #                 data=self.kmeans_cluster_df[mask],
    #                 color=palette[i],
    #                 marker=markers[j],
    #                 label=f'{group}, kmeans: {cluster}',
    #                 s=100, alpha=0.8
    #             )
    
    #     plt.title(f'KMeans Clustering (k={self.n_clusters}) with PCA', fontsize=18)
    #     plt.xlabel('Principal Component 1', fontsize=14)
    #     plt.ylabel('Principal Component 2', fontsize=14)
    #     plt.legend( fontsize=12, title_fontsize=14, bbox_to_anchor=(1.05, 1), loc='upper right')
    #     plt.grid(False)
    
    #     # Save the plot
    #     kmeans_plot_file = os.path.join(self.output_directory, "kmeans_scatter_plot.png")
    #     plt.savefig(kmeans_plot_file, bbox_inches='tight')
    #     plt.close()
    
    #     print(f"KMeans clustering plot saved at: {kmeans_plot_file}")
    #     return self.kmeans_cluster_df

    
   
    
    
    def calculate_feature_importance(self):
        """
        Calculate feature importance using Random Forest and save the results to a CSV file.
    
        Returns:
        - self.feature_importance_df (pd.DataFrame): DataFrame containing feature importance scores.
        """
        print("Calculating feature importance...")
    
        # Ensure 'group' column exists
        if 'group' not in self.data.columns:
            raise ValueError("The dataset must contain a 'group' column for grouping.")
    
        # Prepare data for Random Forest
        X = self.data.select_dtypes(include=['number'])  # Numeric features
        y = self.data['group'].astype('category').cat.codes  # Encode group as numeric
    
        # Train Random Forest
        rf = RandomForestClassifier(random_state=42)
        rf.fit(X, y)
    
        # Get feature importance
        self.feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': rf.feature_importances_})
        self.feature_importance_df = self.feature_importance_df.sort_values(by='Importance', ascending=False)
    
        # Save to CSV
        feature_importance_file = os.path.join(self.output_directory, "feature_importance.csv")
        self.feature_importance_df.to_csv(feature_importance_file, index=False)
    
        print(f"Feature importance saved at: {feature_importance_file}")
    
        return self.feature_importance_df
    

    def evaluate_clustering_quality(self):
        """
        Evaluate the quality of clustering using Silhouette Score, Calinski-Harabasz Index, and Davies-Bouldin Index.
        Stores results in `self.cluster_quality_metrics`.
        """
        print(f"Evaluating clustering quality for k={self.n_clusters}...")
    
        # Extract numeric columns (already standardized)
        numeric_columns = self.data.select_dtypes(include=['number']).columns.tolist()
    
        # Perform KMeans clustering
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(self.data[numeric_columns])  # Get cluster labels
    
        # Compute clustering quality metrics
        metrics = {
            'Silhouette Score': silhouette_score(self.data[numeric_columns], cluster_labels),
            'Calinski-Harabasz Index': calinski_harabasz_score(self.data[numeric_columns], cluster_labels),
            'Davies-Bouldin Index': davies_bouldin_score(self.data[numeric_columns], cluster_labels)
        }
    
        # Store metrics in class attribute
        self.cluster_quality_metrics = metrics
    
        # Print results
        print("✅ Clustering Quality Metrics:")
        for metric, value in metrics.items():
            print(f"   {metric}: {value:.3f}")

    
        return self.cluster_quality_metrics

 
    
    def check_lmeasure_neuron_alignment(self):
        """
        Ensures neuron names in L-Measure data align with neuron_class_df.
        Filters and reorders the data accordingly.
        """
        data = self.data  # Use the cleaned and loaded L-Measure data
    
        lmeasure_filenames = set(data.index)
        neuron_names = set(self.neuron_class_df["neuron_name"])
    
        # Detect mismatches
        not_in_neuron = lmeasure_filenames - neuron_names
        not_in_lmeasure = neuron_names - lmeasure_filenames
    
        # Plain ASCII: on Windows a redirected stdout defaults to cp1252, which
        # cannot encode emoji and raised UnicodeEncodeError when this pipeline
        # was run with its output piped to a file.
        if not_in_neuron or not_in_lmeasure:
            print("Mismatch detected!")
            if not_in_neuron:
                print(f"[warn] In LMeasure but not in neuron_class_df ({len(not_in_neuron)}):",
                      sorted(not_in_neuron))
            if not_in_lmeasure:
                print(f"[warn] In neuron_class_df but not in LMeasure ({len(not_in_lmeasure)}):",
                      sorted(not_in_lmeasure))
            print(f"[warn] Removing {len(not_in_neuron)} unmatched entries from L-Measure.")
            data = data.drop(index=not_in_neuron, errors='ignore')
        else:
            print("[ok] Filenames in LMeasure and neuron_class_df match exactly.")
    
        # Reorder to match neuron_class_df
        valid_names = self.neuron_class_df["neuron_name"]
        data = data.loc[data.index.intersection(valid_names)]
        data = data.loc[valid_names]
    
        # Save the filtered and ordered data back to self.data
        self.data = data




    def run_analysis(self):
        """Runs all analysis functions and returns statistical results."""
        print("Running L-Measure Analysis...")
        self.check_lmeasure_neuron_alignment()
        # self.select_top_neurons_by_group()
        pvalues_df = self.create_grouped_boxplots()
        features = ['N_tips','N_bifs', 'N_branch']
        # features = ['PathDistance' ]
        violin_features = features.copy()
        self.create_violin_plot_for_selected_features(violin_features)
        boxplots_features = features.copy()
        self.create_box_plot_for_selected_features(boxplots_features)
        pca_df = self.perform_pca_and_plot()
        self.calculate_feature_importance()
        self.perform_kmeans_clustering_and_plot()
        print("Analysis complete!")
        

        data = self.data.select_dtypes(include=[float, int])
        dendrogram_analysis = DendrogramAnalysis(
            data = data,
            neuron_class_df = self.neuron_class_df,
            save_directory = self.output_directory,
            descriptor_name="Dendrogram_Lmeasure",
            highlight_neurons = config.HIGHLIGHT_NEURONS,
            is_distance_matrix = False
        )

        # Run the analysis to plot and save the dendrogram
        dendrogram_analysis.run_analysis()
        return data, pvalues_df, pca_df
