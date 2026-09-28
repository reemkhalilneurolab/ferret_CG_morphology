from metric_learn import LMNN
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
import os
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
# import umap
import matplotlib.pyplot as plt
import seaborn as sns
import config
import matplotlib.colors as mcolors
from sklearn.metrics import silhouette_score, davies_bouldin_score
from analysis.data_analysis import DataAnalysis, DendrogramAnalysis, Plotter
from sklearn.manifold import MDS
import pandas as pd
import numpy as np
from scipy.stats import ttest_rel, wilcoxon
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import accuracy_score
import itertools
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import accuracy_score
from sklearn.metrics import adjusted_rand_score
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import squareform
import matplotlib.pyplot as plt   # Re-import required packages after code execution state reset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import silhouette_samples
# Re-import after state reset

from sklearn.metrics import silhouette_samples
import scipy.stats
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.metrics import adjusted_rand_score, silhouette_score
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import StratifiedKFold
from collections import defaultdict

# excluded_descriptors = ["Tortuosity"]


class Detection:
    def __init__(self, distance_matrices: dict, neuron_class_df: pd.DataFrame):
        self.distance_matrices = distance_matrices  # {descriptor_name: distance_matrix}
        self.neuron_class_df = neuron_class_df      # index = neuron_name, column = 'Class'
        # self.features_df = features_df              # index = neuron_name, columns = feature vector (e.g., R^24)
        self.detection_results = {}
        self.beta_distributions = {}
        self.rankings = rankings={}

        
    
    
   
    
    # def compute_detection_rate(self, distance_matrix):
    #     # Initialize dictionaries to store final metrics per class
    #     detection_mean_scores = {}       # Line 293: average detection per class
    #     detection_threshold_scores = {}  # % neurons with detection ≥ threshold
    #     beta_distributions = {}          # Store raw beta scores per neuron for inspection
        
    #     # Precompute mapping: neuron_name → class
    #     name_to_class = self.neuron_class_df.set_index("neuron_name")["class"].to_dict()

    #     # Get all unique neuron class labels (C(N) in the paper, Line 288)
    #     class_labels = self.neuron_class_df['class'].unique()
    
    #     # Loop over each class C
    #     for C in class_labels:
    #         # Get neuron names belonging to current class C (Line 288)
    #         class_members = self.neuron_class_df[self.neuron_class_df['class'] == C]['neuron_name'].values
    #         total_class_size = len(class_members)
    #         beta_values_per_neuron = []
    
    #         # For each neuron N in class C (Line 289)
    #         for Ni in class_members:
    #             # Line 290–291: Compute distances from Ni to all other neurons (including other classes)
    #             all_distances = distance_matrix.loc[Ni]
    #             all_distances = all_distances.drop(index=Ni)  # Exclude self-distance (0 by definition)
    
    #             # Line 289: σ is the permutation that sorts neurons by distance to Ni
    #             sorted_neurons = all_distances.sort_values().index.tolist()
    
    #             # Class label of the current neuron Ni: C(N)
    #             class_of_Ni = name_to_class[Ni]  # ✅ FAST lookup using precomputed dictionary

    
    #             beta_list = []
    
    #             # Line 290–291: Loop over growing "balls" of i nearest neighbors
    #             for i in range(1, len(sorted_neurons) + 1):
    #                 # First i closest neurons (i.e., N_σ(1) to N_σ(i))
    #                 neighbors_in_ball = sorted_neurons[:i]
    
    #                 # Count neurons in the ball from the same class as Ni
    #                 same_class_count = 0  # ✅ Move this line just before the for-loop over `i`
    #                 for i, neighbor in enumerate(sorted_neurons, start=1):
    #                     if name_to_class[neighbor] == class_of_Ni:
    #                         same_class_count += 1

    #                 # Total number of neurons in the current ball is i (Line 291 denominator)
    #                 total_in_ball = i
                    
    #                 ''' 
                    
    #                 Ratio 1 — Class capture rate
    #                 # of same-class neighbors inside the ball / # of total same-class neurons in dataset                  ​
                     
    #                 What it tells you:
    #                 How much of neuron N's own class is included so far as the ball grows.
                    
    #                 Example:
    #                 If class C has 10 neurons in total and you’ve captured 5 of them inside the ball, then:
                    
    #                 Ratio 1 = 5/10 = 0.5
    #                 Interpretation:
    #                 This measures coverage: "How much of my own class have I found?"
    #                 =================================================================================
    #                 Ratio 2 — Ball purity
    #                 # of same-class neighbors inside the ball / # tof otal neurons inside the ball (of any class)                   ​
                     
    #                 What it tells you:
    #                 How many of the neurons inside the ball belong to the correct class.
                    
    #                 Example:
    #                 If the ball contains 6 neurons, and 5 are from the correct class, then:
                    
    #                 Ratio 2 = 5/6 = 0.83
    #                 Interpretation:
    #                 This measures purity: "How clean is the ball — does it contain mostly neurons of my class?"
    #                 ======================================================================================
    #                 ✅ Why take the minimum of both?
    #                 You want both coverage and purity to be high. If either is low, the detection is poor.
            
    #                 '''
                    
    #                 # Line 290 numerator: fraction of class C captured in the ball
    #                 ratio1 = same_class_count / total_class_size
    
    #                 # Line 291 numerator: purity of the ball (how many are correct class)
    #                 ratio2 = same_class_count / i
    
    #                 # Line 290–291: Detection score det_φ(N, i) is the min of the two ratios
    #                 beta_i = min(ratio1, ratio2)
    #                 beta_list.append(beta_i)
    
    #             # Line 293: Local detection rate det_φ(N) = max_i det_φ(N, i)
    #             beta_Ni = max(beta_list) if beta_list else 0
    #             beta_values_per_neuron.append(beta_Ni)
    
    #         # Final class-level metrics:
    #         # Line 293: Mean detection score over all neurons in class
    #         # How well, on average, the neurons in class C can be detected
    #         # It tells you, on average, how well neurons of this class are isolated from other classes using your distance metric
    #         # High (close to 100%) → Most neurons in class  C are tightly grouped and clearly separated from other classes.
    #         # Medium (50–70%) → Class is moderately detectable; might be overlapping with other classes.
    #         # Low (< 50%) → The class is hard to detect; neurons often mix with other classes.
    #         detection_mean_scores[C] = np.mean(beta_values_per_neuron) * 100
    
    #         # Additional: % of neurons in class with detection ≥ threshold (e.g., 70%)
    #         # How many neurons in the class are strongly detectable.
    #         # The percentage of neurons in class C whose detection score is above a specific quality threshold (e.g., 70%).
    #         # High (e.g. 100%) → All neurons in class C are well detected (consistent performance).
    #         # Low (e.g. 20%) → Only a few neurons are clearly distinguishable from other classes; others are ambiguous.
    #         threshold = 0.7
    #         count_above = sum(1 for b in beta_values_per_neuron if b >= threshold)
    #         detection_threshold_scores[C] = (count_above / total_class_size) * 100
            
    #         # Think of detection_mean_scores like your average exam score, and detection_threshold_scores like your pass rate (e.g. % of students scoring ≥70%).
    #         # Both are useful:
    #         # Mean score shows overall strength
    #         # Threshold score shows consistency across individuals
            
    #         # Save raw beta values per neuron in the class
    #         beta_distributions[C] = [
    #             {"neuron_name": class_members[i], "beta": beta}
    #             for i, beta in enumerate(beta_values_per_neuron)
    #         ]
    
    #     # Return all results: mean scores, thresholded scores, and raw values
    #     return detection_mean_scores, detection_threshold_scores, beta_distributions

    def compute_detection_rate(self):
        """Detection rate of each descriptor for each class.

        Implements supplementary section 1.4 of Khalil R, Kallel S, Farhat A,
        Dlotko P, Topological Sholl descriptors for neuronal clustering and
        classification, PLoS Comput Biol 2022;18(6):e1010229.

        A descriptor detects a class C at level n% if some ball B in that
        descriptor's metric holds at least n% of all members of C, and at least
        n% of the neurons in B belong to C. The detection level of a ball is
        therefore the smaller of those two ratios, class capture rate and ball
        purity, and the detection rate is the largest such value over balls.

        Balls are taken centred on each member of C and grown one nearest
        neighbour at a time, which enumerates every distinct membership the
        metric admits.

        Loops over all descriptors and computes:
        - Mean detection score per class
        - Threshold detection score per class
        - Per-neuron beta distributions

        Returns:
        - detection_mean_scores: {descriptor → {class → mean score}}
        - detection_threshold_scores: {descriptor → {class → % above threshold}}
        - beta_distributions: {descriptor → {class → [ {neuron_name, beta}, ... ]}}
        """
        # excluded_descriptors = []
        
        
        detection_max_scores = {}        # descriptor → class → mean detection score
        detection_threshold_scores = {}   # descriptor → class → % with beta ≥ 0.7
        beta_distributions = {}           # descriptor → class → list of beta per neuron
        global_beta_values_per_neuron ={}
        detection_composite_scores = {}
    
        # Precompute mapping: neuron_name → class
        name_to_class = self.neuron_class_df.set_index("neuron_name")["class"].to_dict()
    
        # Class labels C over which detection is evaluated
        class_labels = self.neuron_class_df['class'].unique()
    
        # Loop over all descriptors (each with its own distance matrix)
        for descriptor_name, distance_matrix in self.distance_matrices.items():
            detection_max_scores[descriptor_name] = {}
            detection_threshold_scores[descriptor_name] = {}
            beta_distributions[descriptor_name] = {}
            global_beta_values_per_neuron[descriptor_name] = {}
    
            # Loop over each class C
            for C in class_labels:
                # Get neuron names belonging to current class C (Line 288)
                class_members = self.neuron_class_df[self.neuron_class_df['class'] == C]['neuron_name'].values
                total_class_size = len(class_members)
                beta_values_per_neuron = []
    
                # For each neuron N in class C (Line 289)
                for Ni in class_members:
                    # Line 290–291: Compute distances from Ni to all other neurons (including other classes)
                    all_distances = distance_matrix.loc[Ni]
                    all_distances = all_distances.drop(index=Ni)  # Exclude self-distance (0 by definition)
    
                    # Line 289: σ is the permutation that sorts neurons by distance to Ni
                    sorted_neurons = all_distances.sort_values().index.tolist()
    
                    # Class label of the current neuron Ni: C(N)
                    class_of_Ni = name_to_class[Ni]  # ✅ FAST lookup using precomputed dictionary
    
                    beta_list = []
    
                    # Line 290–291: Loop over growing "balls" of i nearest neighbors
                    for i in range(1, len(sorted_neurons) + 1):
                        # First i closest neurons (i.e., N_σ(1) to N_σ(i))
                        neighbors_in_ball = sorted_neurons[:i]
    
                        # Count neurons in the ball from the same class as Ni
                        same_class_count = sum(
                            1 for neighbor in neighbors_in_ball if name_to_class[neighbor] == class_of_Ni
                        )
    
                        # Total number of neurons in the current ball is i (Line 291 denominator)
                        total_in_ball = i
    
                        ''' 
                        Ratio 1 — Class capture rate
                        # of same-class neighbors inside the ball / # of total same-class neurons in dataset
                        Tells you how much of neuron N's own class is included as the ball grows.
                        Example: if class C has 10 neurons, and 5 are in the ball: ratio1 = 5/10 = 0.5
    
                        Ratio 2 — Ball purity
                        # of same-class neighbors inside the ball / # of total neurons in the ball
                        Tells you how "clean" the ball is — how many of the captured neurons are from the correct class.
                        Example: if ball has 6 neurons, and 5 are correct: ratio2 = 5/6 ≈ 0.83
    
                        ✅ Take the min of both — poor coverage or poor purity means poor detection.
                        '''
    
                        # Ni lies at distance 0 from itself, so it is inside its
                        # own ball and is a member of class C. Both ratios must
                        # therefore count it: the ball holds i + 1 neurons, of
                        # which same_class_count + 1 belong to C. Omitting the
                        # centre understates both the class capture rate and the
                        # ball purity, and so understates detection for every
                        # descriptor.
                        ratio1 = (same_class_count + 1) / total_class_size
                        ratio2 = (same_class_count + 1) / (i + 1)
                        beta_i = min(ratio1, ratio2)
                        beta_list.append(beta_i)
    
                    # Line 293: Local detection rate det_φ(N) = max_i det_φ(N, i)
                    beta_Ni = max(beta_list) if beta_list else 0
                    beta_values_per_neuron.append(beta_Ni)
    

                detection_max_scores[descriptor_name][C] = np.max(beta_values_per_neuron) * 100
                global_beta_values_per_neuron[descriptor_name][C] = beta_values_per_neuron
    
                # % of neurons in class with detection ≥ threshold (e.g. 70%)
                threshold = 0.7
                count_above = sum(1 for b in beta_values_per_neuron if b >= threshold)
                detection_threshold_scores[descriptor_name][C] = (count_above / total_class_size) * 100
    
                # Store raw detection values for each neuron
                beta_distributions[descriptor_name][C] = [
                    {"neuron_name": class_members[i], "beta": beta}
                    for i, beta in enumerate(beta_values_per_neuron)
                ]
        
        alpha = 0.7  # weight between detection and silhouette 
        # Composite score calculation
        for descriptor_name, matrix in self.distance_matrices.items():
        
            labels = self.neuron_class_df.set_index("neuron_name").loc[matrix.index]["class"].values
            dist = matrix.values
            
            db = davies_bouldin_score(dist, labels)
            ch = calinski_harabasz_score(dist, labels)
        
            # Normalize DB index (lower is better → invert)
            db_norm = 1 / (db + 1e-10)
        
            silhouette = silhouette_score(dist, labels, metric="precomputed")
            detection_avg = np.mean(list(detection_max_scores[descriptor_name].values()))
            composite = alpha * detection_avg + (1 - alpha) * silhouette
            # Optionally, include DB and CH variants too
            # composite_weights_db = alpha * detection_avg + (1 - alpha) * db_norm
            # composite_weights_ch = alpha * detection_avg + (1 - alpha) * ch
            detection_composite_scores[descriptor_name] = composite
                        
        # Optional: visualize and save diagnostics
        self.analyze_detection_vs_structure(
            beta_distributions=beta_distributions,
            distance_matrices=self.distance_matrices,
            neuron_class_df=self.neuron_class_df
        )
            
    
        return detection_max_scores, detection_threshold_scores, beta_distributions, global_beta_values_per_neuron,detection_composite_scores
   


    def analyze_detection_vs_structure(self, beta_distributions, distance_matrices, neuron_class_df):
        """
        Full visual and tabular analysis per descriptor:
        1. β histogram per class
        2. Silhouette vs β plot (with mean silhouette and correlation)
        3. Intra- vs inter-class distance histogram (with OVL, BHA, FDR)
        Also generates a CSV with metric values and interpretation.
        """
        from sklearn.metrics import silhouette_samples
        import scipy.stats
    
        def overlap_coefficient(dist1, dist2, bins=50):
            hist1, bin_edges = np.histogram(dist1, bins=bins, density=True)
            hist2, _ = np.histogram(dist2, bins=bin_edges, density=True)
            return np.sum(np.minimum(hist1, hist2)) * np.diff(bin_edges).mean()
    
        def bhattacharyya_distance(dist1, dist2, bins=50):
            hist1, bin_edges = np.histogram(dist1, bins=bins, density=True)
            hist2, _ = np.histogram(dist2, bins=bin_edges, density=True)
            bc = np.sum(np.sqrt(hist1 * hist2)) * np.diff(bin_edges).mean()
            return -np.log(bc + 1e-10)
    
        def fisher_ratio(intra, inter):
            mu_diff = np.mean(inter) - np.mean(intra)
            var_sum = np.var(inter) + np.var(intra)
            return (mu_diff ** 2) / (var_sum + 1e-10)
    
        name_to_class = neuron_class_df.set_index("neuron_name")["class"].to_dict()
        descriptors = beta_distributions.keys()
    
        summary = {
            "Descriptor": [],
            "Mean Silhouette": [],
            "β-Silhouette Correlation": [],
            "Overlap Coefficient": [],
            "Bhattacharyya Distance": [],
            "Fisher Ratio": [],
            "Interpretation": []
        }
    
        for descriptor in descriptors:
            
            print(f"\n--- Analysis for descriptor: {descriptor} ---")
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))
            fig.suptitle(f"Descriptor: {descriptor}", fontsize=16)
    
            # 1. β histogram per class
            beta_flat = []
            for cls, entries in beta_distributions[descriptor].items():
                beta_vals = [entry['beta'] for entry in entries]
                beta_flat.extend([{'beta': b, 'class': cls} for b in beta_vals])
            beta_df = pd.DataFrame(beta_flat)
            sns.histplot(data=beta_df, x="beta", hue="class", element="step", common_norm=False, ax=axes[0])
            axes[0].set_title("β Score Distribution per Class")
            axes[0].set_xlabel("β Score")
            axes[0].set_ylabel("Count")
    
            # 2. Silhouette vs β
            dist_df = distance_matrices[descriptor]
            dist_matrix = dist_df.values
            labels = neuron_class_df.set_index("neuron_name").loc[dist_df.index]["class"].values
            sil_scores = silhouette_samples(dist_matrix, labels, metric="precomputed")
    
            neuron_order = dist_df.index.tolist()
            beta_lookup = {entry['neuron_name']: entry['beta']
                           for entries in beta_distributions[descriptor].values()
                           for entry in entries}
            beta_vals_ordered = [beta_lookup[n] for n in neuron_order]
    
            mean_sil = np.mean(sil_scores)
            corr, _ = scipy.stats.pearsonr(beta_vals_ordered, sil_scores)
            axes[1].scatter(beta_vals_ordered, sil_scores, alpha=0.7)
            axes[1].set_title(f"β vs. Silhouette\nMean Sil = {mean_sil:.2f}, Corr = {corr:.2f}")
            axes[1].set_xlabel("β Score")
            axes[1].set_ylabel("Silhouette Score")
    
            # 3. Intra- vs Inter-class distances
            intra_dists = []
            inter_dists = []
    
            for i in range(len(dist_df)):
                for j in range(i + 1, len(dist_df)):
                    name_i = dist_df.index[i]
                    name_j = dist_df.columns[j]
                    d = dist_df.iloc[i, j]
                    if name_to_class[name_i] == name_to_class[name_j]:
                        intra_dists.append(d)
                    else:
                        inter_dists.append(d)
    
            ovl = overlap_coefficient(intra_dists, inter_dists)
            bhatt = bhattacharyya_distance(intra_dists, inter_dists)
            fdr = fisher_ratio(intra_dists, inter_dists)
    
            axes[2].hist(intra_dists, bins=30, alpha=0.7, label="Intra-class")
            axes[2].hist(inter_dists, bins=30, alpha=0.7, label="Inter-class")
            axes[2].set_title(f"Intra vs. Inter Distances\nOVL={ovl:.2f}, BHA={bhatt:.2f}, FDR={fdr:.2f}")
            axes[2].set_xlabel("Distance")
            axes[2].set_ylabel("Frequency")
            axes[2].legend()
    
            # Save visual
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            out_path = os.path.join(config.DETECTION_DIRECTORY, f"detection_analysis_{descriptor}.png")
            plt.savefig(out_path, dpi=300)
            plt.close()
    
            # Interpretation logic
            explanation = []
            if mean_sil > 0.3:
                explanation.append("Good global class separation (silhouette > 0.3)")
            else:
                explanation.append("Weak global separation (silhouette ≤ 0.3)")
    
            if corr > 0.5:
                explanation.append("β agrees with silhouette (correlation > 0.5)")
            else:
                explanation.append("β does not track global structure well")
    
            if ovl < 0.5:
                explanation.append("Low overlap between classes (OVL < 0.5)")
            else:
                explanation.append("High overlap (OVL ≥ 0.5)")
    
            if bhatt > 0.5:
                explanation.append("High Bhattacharyya distance → distinct class distributions")
            else:
                explanation.append("Low Bhattacharyya → distributions overlap")
    
            if fdr > 1.0:
                explanation.append("Strong class separation (FDR > 1.0)")
            else:
                explanation.append("Weak separation (FDR ≤ 1.0)")
    
            summary["Descriptor"].append(descriptor)
            summary["Mean Silhouette"].append(round(mean_sil, 3))
            summary["β-Silhouette Correlation"].append(round(corr, 3))
            summary["Overlap Coefficient"].append(round(ovl, 3))
            summary["Bhattacharyya Distance"].append(round(bhatt, 3))
            summary["Fisher Ratio"].append(round(fdr, 3))
            summary["Interpretation"].append(" | ".join(explanation))
    
        df = pd.DataFrame(summary)
        csv_path = os.path.join(config.DETECTION_DIRECTORY, "descriptor_quality_summary.csv")
        df.to_csv(csv_path, index=False)
        return df

            
  


    # def analyze_detection_vs_structure(self, beta_distributions, distance_matrices, neuron_class_df):
    #     """
    #     Performs three checks:
    #     1. Histogram of beta scores per class per descriptor
    #     2. Compare silhouette score vs. beta per neuron
    #     3. Compare intra- vs inter-class distances per descriptor
    #     Saves plots to DETECTION_DIRECTORY.
    #     """
    #     name_to_class = neuron_class_df.set_index("neuron_name")["class"].to_dict()
    #     descriptors = beta_distributions.keys()
    
    #     for descriptor in descriptors:
    #         print(f"\n--- Analysis for descriptor: {descriptor} ---")
    #         fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    #         fig.suptitle(f"Descriptor: {descriptor}", fontsize=16)
    
    #         # 1. β histogram per class
    #         beta_flat = []
    #         for cls, entries in beta_distributions[descriptor].items():
    #             beta_vals = [entry['beta'] for entry in entries]
    #             beta_flat.extend([{'beta': b, 'class': cls} for b in beta_vals])
    #         beta_df = pd.DataFrame(beta_flat)
    #         sns.histplot(data=beta_df, x="beta", hue="class", element="step", common_norm=False, ax=axes[0])
    #         axes[0].set_title("β Score Distribution per Class")
    #         axes[0].set_xlabel("β Score")
    #         axes[0].set_ylabel("Count")
    
    #         # 2. Silhouette vs. β
    #         dist_matrix = distance_matrices[descriptor].values
    #         labels = neuron_class_df.set_index("neuron_name").loc[distance_matrices[descriptor].index]["class"].values
    #         sil_scores = silhouette_samples(dist_matrix, labels, metric="precomputed")
    
    #         neuron_order = distance_matrices[descriptor].index.tolist()
    #         beta_lookup = {entry['neuron_name']: entry['beta']
    #                        for entries in beta_distributions[descriptor].values()
    #                        for entry in entries}
    #         beta_vals_ordered = [beta_lookup[n] for n in neuron_order]
    
    #         axes[1].scatter(beta_vals_ordered, sil_scores, alpha=0.7)
    #         axes[1].set_title("β vs. Silhouette Score")
    #         axes[1].set_xlabel("β Score")
    #         axes[1].set_ylabel("Silhouette Score")
    
    #         # 3. Intra vs. Inter class distances
    #         dist_df = distance_matrices[descriptor]
    #         intra_dists = []
    #         inter_dists = []
    
    #         for i in range(len(dist_df)):
    #             for j in range(i + 1, len(dist_df)):
    #                 name_i = dist_df.index[i]
    #                 name_j = dist_df.columns[j]
    #                 d = dist_df.iloc[i, j]
    #                 if name_to_class[name_i] == name_to_class[name_j]:
    #                     intra_dists.append(d)
    #                 else:
    #                     inter_dists.append(d)
    
    #         axes[2].hist(intra_dists, bins=30, alpha=0.7, label="Intra-class")
    #         axes[2].hist(inter_dists, bins=30, alpha=0.7, label="Inter-class")
    #         axes[2].set_title("Intra- vs Inter-class Distances")
    #         axes[2].set_xlabel("Distance")
    #         axes[2].set_ylabel("Frequency")
    #         axes[2].legend()
    
    #         plt.tight_layout(rect=[0, 0, 1, 0.95])
    #         out_path = os.path.join(config.DETECTION_DIRECTORY, f"detection_analysis_{descriptor}.png")
    #         plt.savefig(out_path, dpi=300)
    #         plt.close()


    
    # def evaluate_all_detection_rates(self):
    #     all_detection_rates = {}
    #     all_beta_distributions = {}

    #     for descriptor_name, matrix in self.distance_matrices.items():
    #         mean_scores, threshold_scores, beta_distributions = self.compute_detection_rate(matrix)
    #         all_detection_rates[descriptor_name] = (mean_scores, threshold_scores)
    #         all_beta_distributions[descriptor_name] = beta_distributions

    #     self.detection_results = all_detection_rates
    #     self.beta_distributions = all_beta_distributions

    #     self.plot_all_detection_rates()
    #     self.save_detection_rates_to_csv()
    #     self.rank_descriptors_by_detection_quality(top_k= None)
    #     self.plot_composite_score_heatmap()
    #     explained_variances, selected_by_variance = self.explained_variance_from_distances()
    #     normalized_matrices = self.combine_descriptors_by_composite_score()
    #     # descriptor_pass_map = self.filter_descriptors_by_detection_thresholds(min_mean=60, 
    #     #                                                                     min_threshold=70, 
    #     #                                                                     min_beta_median=0.6)


    #     return all_detection_rates, self.rankings, normalized_matrices
    
    def evaluate_all_detection_rates(self):
        """
        Evaluate and store detection metrics across all descriptors,
        now using the internally-looping compute_detection_rate().
        """
    
        # Compute all detection metrics at once
        mean_scores, threshold_scores, beta_distributions, global_beta_values_per_neuron, detection_composite_scores = self.compute_detection_rate()
        # weighted_matrices, combined_matrix, results = self.combine_matrices_by_average_detection_score(mean_scores)
        # weighted_matrices, combined_matrix, results, metrics = self.combine_matrices_by_average_detection_score(mean_scores)
        weighted_matrices, combined_matrix, results, metrics = self.combine_matrices_by_average_detection_score(
        mean_scores,
        output_filename="Detection_rates.png",
        plot_title="Detection Scores by Descriptor (Direct)"
                        )

        # Combine scores in your original format
        all_detection_rates = {
            descriptor: (mean_scores[descriptor], threshold_scores[descriptor])
            for descriptor in mean_scores
        }
    
        # Store in self for downstream access
        self.detection_results = all_detection_rates
        self.beta_distributions = beta_distributions
    
        # Post-processing steps
        self.plot_all_detection_rates()
        self.save_detection_rates_to_csv()
        self.rank_descriptors_by_detection_quality(top_k=None)
        self.plot_composite_score_heatmap()
        explained_variances, selected_by_variance = self.explained_variance_from_distances()
        normalized_matrices = self.combine_descriptors_by_composite_score()
        
        # Optional filtering line commented by user
        # descriptor_pass_map = self.filter_descriptors_by_detection_thresholds(...)
    
        return all_detection_rates, self.rankings, normalized_matrices, global_beta_values_per_neuron, combined_matrix, results


    def plot_all_detection_rates(self):
        rows = []
        beta_dist_rows = []

        for descriptor, (mean_scores, threshold_scores) in self.detection_results.items():
            all_classes = set(mean_scores.keys()) | set(threshold_scores.keys())
            for cls in all_classes:
                rows.append({
                    'descriptor': descriptor,
                    'class': cls,
                    'mean': float(mean_scores.get(cls, np.nan)),
                    'threshold': float(threshold_scores.get(cls, np.nan))
                })

            for cls, beta_entries in self.beta_distributions[descriptor].items():
                for entry in beta_entries:
                    beta_dist_rows.append({
                        'descriptor': descriptor,
                        'class': cls,
                        'neuron_name': entry['neuron_name'],
                        'beta': entry['beta']
                    })


        df = pd.DataFrame(rows)
        beta_df = pd.DataFrame(beta_dist_rows)

        # Plot mean detection rate
        plt.figure(figsize=(10, 6))
        sns.barplot(
            data=df,
            x='descriptor',
            y='mean',
            hue='class',
            palette='tab10'
        )
        plt.title("Mean Detection Rate (%)")
        plt.ylabel("Detection Rate")
        plt.xlabel("Descriptor")
        plt.ylim(0, 110)
        plt.xticks(rotation=45)
        plt.legend(title="Class")
        plt.tight_layout()
        plt.savefig(os.path.join(config.DETECTION_DIRECTORY, "mean_detection_rates_grouped.png"), dpi=300)
        plt.close()

        # Plot threshold detection rate
        plt.figure(figsize=(10, 6))
        sns.barplot(
            data=df,
            x='descriptor',
            y='threshold',
            hue='class',
            palette='tab10'
        )
        plt.title("Threshold Detection Rate (%)")
        plt.ylabel("Detection Rate")
        plt.xlabel("Descriptor")
        plt.ylim(0, 110)
        plt.xticks(rotation=45)
        plt.legend(title="Class")
        plt.tight_layout()
        plt.savefig(os.path.join(config.DETECTION_DIRECTORY, "threshold_detection_rates_grouped.png"), dpi=300)
        plt.close()
        
        

        # Plot beta distributions
        plt.figure(figsize=(12, 6))
        sns.violinplot(
            data=beta_df,
            x="descriptor",
            y="beta",
            hue="class",
            split=True,
            palette="tab10",
            inner="quartile"
        )
        plt.title("Distribution of Beta Scores by Descriptor and Class")
        plt.xlabel("Descriptor")
        plt.ylabel("Beta Score")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(config.DETECTION_DIRECTORY, "beta_distributions_violinplot.png"), dpi=300)
        plt.close()

    def save_detection_rates_to_csv(self):
        rows = []
        beta_rows = []

        for descriptor, (mean_scores, threshold_scores) in self.detection_results.items():
            for cls, val in mean_scores.items():
                interpretation = (
                    "Strong separation" if val > 80 else
                    "Partial separation" if val > 60 else
                    "Poor or ambiguous separation" if val > 40 else
                    "No meaningful class structure"
                )
                plain_explanation = (
                    f"On average, each neuron in this class has {val:.1f}% of its "
                    "internal neighbors more distinguishable than external ones. "
                    "The class is only weakly separated from others — about half of "
                    "each neuron's neighbors from the same class are closer than the "
                    "closest outsiders." if 40 <= val <= 60 else ""
                )
                rows.append({
                    'descriptor': descriptor,
                    'class': cls,
                    'score_type': 'mean',
                    'value': val,
                    'interpretation': interpretation,
                    'explanation': plain_explanation
                })
                
            for cls, val in threshold_scores.items():
                explanation = (
                    "A neuron is considered well-detected only if its detection strength β is at least 70%. "
                    "This score reflects the percentage of neurons in the class that meet that threshold."
                )
                rows.append({
                    'descriptor': descriptor,
                    'class': cls,
                    'score_type': 'threshold',
                    'value': val,
                    'interpretation': "",
                    'explanation': explanation
                })

                    
            for cls, beta_list in self.beta_distributions[descriptor].items():
                for entry in beta_list:
                    beta_rows.append({
                        'descriptor': descriptor,
                        'class': cls,
                        'neuron_name': entry["neuron_name"],
                        'beta': entry["beta"]
                    })

           


        df = pd.DataFrame(rows)
        beta_df = pd.DataFrame(beta_rows)

        full_path = os.path.join(config.DETECTION_DIRECTORY, 'detection_rates.csv')
        beta_path = os.path.join(config.DETECTION_DIRECTORY, 'beta_distributions.csv')

        df.to_csv(full_path, index=False)
        beta_df.to_csv(beta_path, index=False)
        
        
    # def combine_matrices_by_average_detection_score(self, detection_scores, use_composite=False):
    #     """
    #     Computes one combined matrix by summing each descriptor matrix scaled by its detection score.
    
    #     Parameters:
    #     - detection_scores: 
    #         if use_composite = False → {descriptor → {class → max score}} (averaged per descriptor)
    #         if use_composite = True  → {descriptor → composite score}
    #     - use_composite: bool, whether to use global composite scores or per-class max scores
    
    #     Returns:
    #     - weighted_matrices: {descriptor → individually weighted matrix}
    #     - combined_matrix: total sum of all weighted matrices
    #     - results: clustering diagnostics from compare_clustering_quality
    #     """
    #     weighted_matrices = {}
    #     combined_sum = None
    
    #     for descriptor, matrix in self.distance_matrices.items():
    #         # Choose weighting strategy
    #         if use_composite:
    #             score = detection_scores.get(descriptor, 1.0)
    #         else:
    #             class_scores = detection_scores.get(descriptor, {})
    #             score = np.mean(list(class_scores.values())) if class_scores else 1.0
    
    #         # Weight the matrix
    #         weighted = matrix.values * score
    #         weighted_df = pd.DataFrame(weighted, index=matrix.index, columns=matrix.columns)
    #         weighted_matrices[descriptor] = weighted_df
    
    #         # Accumulate
    #         if combined_sum is None:
    #             combined_sum = weighted.copy()
    #         else:
    #             combined_sum += weighted
    
    #     combined_matrix = pd.DataFrame(combined_sum, index=matrix.index, columns=matrix.columns)
    
    #     # --- Plot detection scores ---
    #     if use_composite:
    #         df = pd.DataFrame([
    #             {"Descriptor": descriptor, "Composite Score": score}
    #             for descriptor, score in detection_scores.items()
    #         ])
    #         df["Descriptor"] = df["Descriptor"].apply(lambda x: x.replace("_", "\n"))
    
    #         plt.figure(figsize=(10, 6))
    #         sns.set_theme(style="white")
    #         ax = sns.barplot(data=df, x="Descriptor", y="Composite Score", color="gray")
    
    #         plt.title("Composite Scores by Descriptor", fontsize=14, weight='bold')
    #         plt.ylabel("Composite Score", fontsize=14)
    #         plt.xlabel("Descriptor", fontsize=14)
    #         plt.xticks(rotation=0)
    #         plt.ylim(0, 110)
    #         plt.tight_layout()
    #         plt.savefig(os.path.join(config.DETECTION_DIRECTORY, "Composite_scores.png"), dpi=300)
    #         plt.close()
    
    #         weights = detection_scores
    
    #     else:
    #         df = pd.DataFrame(detection_scores).T.reset_index().melt(
    #             id_vars='index', var_name='Class', value_name='Max Score'
    #         )
    #         df.rename(columns={'index': 'Descriptor'}, inplace=True)
    #         df["Descriptor"] = df["Descriptor"].apply(lambda x: x.replace("_", "\n"))
    
    #         plt.figure(figsize=(10, 6))
    #         sns.set_theme(style="white")
    #         ax = sns.barplot(data=df, x="Descriptor", y="Max Score", hue="Class", palette=config.COLORS_PALETTE)
    
    #         plt.title("Detection Scores by Descriptor", fontsize=14, weight='bold')
    #         plt.ylabel("Detection Score (%)", fontsize=14)
    #         plt.xlabel("Descriptor", fontsize=14)
    #         plt.xticks(rotation=0)
    #         plt.ylim(0, 110)
    #         plt.legend(title=None, frameon=False, fontsize=12)
    #         plt.tight_layout()
    #         plt.savefig(os.path.join(config.DETECTION_DIRECTORY, "Detection_rates.png"), dpi=300)
    #         plt.close()
    
    #         weights = {
    #             descriptor: np.mean(list(class_scores.values()))
    #             for descriptor, class_scores in detection_scores.items()
    #         }
    
    #     results = self.compare_clustering_quality(
    #         distance_matrices=self.distance_matrices,
    #         weights=weights,
    #         neuron_class_df=self.neuron_class_df
    #     )
    
    #     return weighted_matrices, combined_matrix, results

    

    def cross_validated_detection_scores(self, n_splits=5):
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
        neuron_names = list(self.neuron_class_df["neuron_name"])
        labels = self.neuron_class_df["class"].values
    
        # Collect detection scores for each fold
        all_fold_scores = {descriptor: [] for descriptor in self.distance_matrices}
    
        for train_index, test_index in skf.split(neuron_names, labels):
            train_names = [neuron_names[i] for i in train_index]
            test_names = [neuron_names[i] for i in test_index]
    
            # For each descriptor
            for descriptor, full_matrix in self.distance_matrices.items():
                train_matrix = full_matrix.loc[train_names, train_names]
                test_matrix = full_matrix.loc[test_names, train_names]  # test rows, train cols
    
                # Compute detection scores for test neurons vs. train neurons
                scores = []
                test_classes = self.neuron_class_df.set_index("neuron_name").loc[test_names]["class"]
    
                for test_neuron, true_class in zip(test_names, test_classes):
                    distances = test_matrix.loc[test_neuron]
                    neighbor_indices = distances.nsmallest(5).index
                    neighbor_classes = self.neuron_class_df.set_index("neuron_name").loc[neighbor_indices]["class"]
    
                    match_count = sum(neighbor_classes == true_class)
                    scores.append(match_count / len(neighbor_indices))
    
                fold_score = np.mean(scores)
                all_fold_scores[descriptor].append(fold_score * 100)  # percentage
    
        return all_fold_scores
    
    
    def plot_cross_validated_detection_scores(self, all_fold_scores):
        import seaborn as sns
        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
    
        # Prepare dataframe
        data = []
        for descriptor, scores in all_fold_scores.items():
            data.append({
                "Descriptor": descriptor.replace("_", "\n"),
                "Mean Score": np.mean(scores),
                "Std": np.std(scores)
            })
        df = pd.DataFrame(data)
    
        # Plot barplot
        plt.figure(figsize=(10, 6))
        sns.set_theme(style="whitegrid")
        ax = sns.barplot(data=df, x="Descriptor", y="Mean Score", palette="viridis")
    
        # Add error bars manually
        x_coords = range(len(df))
        ax.errorbar(x=x_coords, y=df["Mean Score"], yerr=df["Std"],
                    fmt='none', c='black', capsize=5)
    
        plt.ylim(0, 110)
        plt.ylabel("Detection Score (%)", fontsize=13)
        plt.title("Cross-Validated Detection Scores with Std", fontsize=14, weight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(config.DETECTION_DIRECTORY, "CV_detection_scores.png"), dpi=600)
        plt.close()
        
    
    def cross_validate_original_detection_method(self, n_splits=5):
        from sklearn.model_selection import StratifiedKFold
        from collections import defaultdict
    
        # Store original data
        full_class_df = self.neuron_class_df.copy()
        full_distance_matrices = self.distance_matrices.copy()
    
        neuron_names = list(full_class_df["neuron_name"])
        labels = full_class_df["class"].values
        class_list = sorted(set(labels))
    
        # Scores: descriptor → class → list of fold means
        all_fold_scores = {
            descriptor: {cls: [] for cls in class_list}
            for descriptor in full_distance_matrices
        }
    
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
        for train_index, test_index in skf.split(neuron_names, labels):
            train_neurons = [neuron_names[i] for i in train_index]
            test_neurons = [neuron_names[i] for i in test_index]
    
            # Subset class dataframe
            self.neuron_class_df = full_class_df[full_class_df["neuron_name"].isin(train_neurons)].reset_index(drop=True)
    
            # Subset distance matrices
            self.distance_matrices = {
                d: full_distance_matrices[d].loc[train_neurons, train_neurons]
                for d in full_distance_matrices
            }
    
            # Run original detection method on training data only
            mean_scores, _, _, _, _ = self.compute_detection_rate()
    
            # Evaluate scores on held-out test set
            # For now, just collect mean per class from training data
            for descriptor in mean_scores:
                for cls in mean_scores[descriptor]:
                    all_fold_scores[descriptor][cls].append(mean_scores[descriptor][cls])
    
        # Restore full dataset
        self.neuron_class_df = full_class_df
        self.distance_matrices = full_distance_matrices
    
        # Aggregate fold scores
        mean_scores = {
            descriptor: {
                cls: np.mean(score_list)
                for cls, score_list in class_scores.items()
            }
            for descriptor, class_scores in all_fold_scores.items()
        }
    
        # Use cross-validated scores to weight matrices
        weighted_matrices, combined_matrix, results, metrics = self.combine_matrices_by_average_detection_score(
            mean_scores,
            output_filename="Detection_scores_cv_original.png",
            plot_title="Cross-Validated Detection Scores (Original Method)"
        )
    
        return all_fold_scores, mean_scores, combined_matrix, results




    def cross_validated_detection_scores_per_class(self, n_splits=5):
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
        neuron_names = list(self.neuron_class_df["neuron_name"])
        labels = self.neuron_class_df["class"].values
        class_list = sorted(set(labels))
    
        # Initialize: {descriptor → {class → [fold scores]}}
        all_fold_scores = {
            descriptor: {cls: [] for cls in class_list}
            for descriptor in self.distance_matrices
        }
    
        for train_index, test_index in skf.split(neuron_names, labels):
            train_names = [neuron_names[i] for i in train_index]
            test_names = [neuron_names[i] for i in test_index]
    
            for descriptor, full_matrix in self.distance_matrices.items():
                train_matrix = full_matrix.loc[train_names, train_names]
                test_matrix = full_matrix.loc[test_names, train_names]
    
                test_classes = self.neuron_class_df.set_index("neuron_name").loc[test_names]["class"]
    
                # Track per-class scores
                class_scores = defaultdict(list)
    
                for test_neuron, true_class in zip(test_names, test_classes):
                    distances = test_matrix.loc[test_neuron]
                    neighbor_indices = distances.nsmallest(5).index
                    neighbor_classes = self.neuron_class_df.set_index("neuron_name").loc[neighbor_indices]["class"]
                    match_count = sum(neighbor_classes == true_class)
                    score = match_count / len(neighbor_indices)
                    class_scores[true_class].append(score)
    
                for cls in class_list:
                    if class_scores[cls]:  # avoid empty list
                        avg_score = np.mean(class_scores[cls])
                        all_fold_scores[descriptor][cls].append(avg_score * 100)  # as percentage
                        
        mean_scores = self.get_per_class_mean_scores_from_cv(all_fold_scores)
        
        # weighted_matrices, combined_matrix_weighted_detection_cv, results, metrics = self.combine_matrices_by_average_detection_score(mean_scores)
        weighted_matrices, combined_matrix_weighted_detection_cv, results, metrics = self.combine_matrices_by_average_detection_score(
                mean_scores,
                output_filename="Detection_rates_cv.png",
                plot_title="Detection Scores by Descriptor (Cross-Validated)"
            )   

    
        return all_fold_scores, mean_scores, combined_matrix_weighted_detection_cv
    
    
    # def get_mean_scores_from_cv(self, all_fold_scores):
    #     mean_scores = {}
    #     for descriptor, class_scores in all_fold_scores.items():
    #         # Flatten all per-class fold scores into one list
    #         all_scores = []
    #         for scores in class_scores.values():
    #             all_scores.extend(scores)
    #         if all_scores:
    #             mean_scores[descriptor] = np.mean(all_scores)
    #         else:
    #             mean_scores[descriptor] = 0  # or np.nan, depending on use
    #     return mean_scores
    
    def get_per_class_mean_scores_from_cv(self, all_fold_scores):
        per_class_scores = {}
        for descriptor, class_scores in all_fold_scores.items():
            class_means = {}
            for cls, scores in class_scores.items():
                if scores:
                    class_means[cls] = np.mean(scores)
                else:
                    class_means[cls] = 0  # or np.nan
            per_class_scores[descriptor] = class_means
        return per_class_scores



    def plot_cross_validated_detection_scores_per_class(self, all_fold_scores):

    
        data = []
        for descriptor, class_dict in all_fold_scores.items():
            for cls, scores in class_dict.items():
                data.append({
                    "Descriptor": descriptor.replace("_", "\n"),
                    "Class": cls,
                    "Mean": np.mean(scores),
                    "Std": np.std(scores)
                })
    
        df = pd.DataFrame(data)
    
        plt.figure(figsize=(10, 6))
        sns.set_theme(style="whitegrid")
    
        ax = sns.barplot(
            data=df,
            x="Descriptor",
            y="Mean",
            hue="Class",
            palette=config.COLORS_PALETTE  # reuse your existing colors
        )
    
        # Add error bars manually
        for bar, (_, row) in zip(ax.patches, df.iterrows()):
            height = bar.get_height()
            std = row["Std"]
            x = bar.get_x() + bar.get_width() / 2
            ax.errorbar(x=x, y=height, yerr=std, fmt='none', c='black', capsize=5)

    
        plt.ylim(0, 110)
        plt.ylabel("Detection Score (%)", fontsize=13)
        plt.title("Cross-Validated Detection Scores by Descriptor and Class", fontsize=14, weight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(config.DETECTION_DIRECTORY, "CV_detection_scores_by_class.png"), dpi=600)
        plt.close()



    def run_cross_validated_detection_analysis_by_class(self, n_splits=5):
        all_fold_scores, mean_scores, combined_matrix_weighted_detection_cv, results = self.cross_validate_original_detection_method(n_splits=n_splits)
        
        
        self.plot_cross_validated_detection_scores_per_class(all_fold_scores)
        return all_fold_scores, mean_scores, combined_matrix_weighted_detection_cv, results 


        
        
    def run_cross_validated_detection_analysis(self, n_splits=5):
        """
        Perform k-fold cross-validated detection analysis across all descriptors
        and generate a plot with mean ± std detection scores.
        """
        all_fold_scores = self.cross_validated_detection_scores(n_splits=n_splits)
        self.plot_cross_validated_detection_scores(all_fold_scores)
        return all_fold_scores



    
    def combine_matrices_by_average_detection_score(self, detection_max_scores, output_filename="Detection_rates.png", plot_title="Detection Scores by Descriptor"):
        """
        Computes one combined matrix by summing each descriptor matrix scaled by its average detection score.
        Also compares clustering performance before and after weighting.
    
        Parameters:
        - detection_max_scores: {descriptor → {class → max score}}
    
        Returns:
        - weighted_matrices: {descriptor → individually weighted matrix}
        - combined_matrix: total sum of all weighted matrices
        - results: output of compare_clustering_quality()
        - metrics: dict with ARI and silhouette for unweighted and weighted combinations
        """
        weighted_matrices = {}
        combined_sum = None

        def rescale(values):
            """Divide a distance matrix by its mean off-diagonal distance.

            Puts the descriptors on a common scale so that the detection weights,
            rather than the descriptors' physical units, determine how much each
            contributes. See config.NORMALIZE_DESCRIPTOR_DISTANCES.
            """
            if not getattr(config, 'NORMALIZE_DESCRIPTOR_DISTANCES', False):
                return values
            off_diagonal = values[~np.eye(len(values), dtype=bool)]
            mean_distance = off_diagonal.mean()
            return values / mean_distance if mean_distance > 0 else values

        for descriptor, matrix in self.distance_matrices.items():
            # Step 1: Retrieve max detection scores across classes for this descriptor
            class_scores = detection_max_scores.get(descriptor, {})
            avg_score = np.mean(list(class_scores.values())) if isinstance(class_scores, dict) else 1.0

            # Step 2: Rescale to a common footing, then weight by detection score
            weighted = rescale(matrix.values) * avg_score
            weighted_df = pd.DataFrame(weighted, index=matrix.index, columns=matrix.columns)
            weighted_matrices[descriptor] = weighted_df
    
            # Step 3: Accumulate into combined matrix
            if combined_sum is None:
                combined_sum = weighted.copy()
            else:
                combined_sum += weighted
    
        # Step 4: Final weighted matrix
        combined_matrix = pd.DataFrame(combined_sum, index=matrix.index, columns=matrix.columns)
    
        # Step 5: Unweighted matrix (equal weights). Rescaled the same way so the
        # weighted-versus-unweighted comparison isolates the effect of the
        # detection weights rather than of the normalisation.
        unweighted_sum = sum(rescale(m.values) for m in self.distance_matrices.values())
        unweighted_matrix = pd.DataFrame(
            unweighted_sum,
            index=next(iter(self.distance_matrices.values())).index,
            columns=next(iter(self.distance_matrices.values())).columns
        )
    
        # Step 6: Clustering metrics comparison
        labels = self.neuron_class_df.set_index("neuron_name").loc[combined_matrix.index]["class"].values
    
        def compute_metrics(matrix, labels):
            # squareform is required here: passing the square distance matrix
            # straight to linkage makes scipy treat each row as an observation
            # vector in n-dimensional Euclidean space and compute Euclidean
            # distances between those rows, which is not Ward linkage on the
            # supplied distances and yields a different partition.
            Z = linkage(squareform(matrix.values, checks=False), method="ward")
            predicted = fcluster(Z, t=len(set(labels)), criterion="maxclust")
            ari = adjusted_rand_score(labels, predicted)
            sil = silhouette_score(matrix.values, labels, metric="precomputed")
            return ari, sil
    
        ari_unweighted, sil_unweighted = compute_metrics(unweighted_matrix, labels)
        ari_weighted, sil_weighted = compute_metrics(combined_matrix, labels)
    
        print(f"Unweighted ARI: {ari_unweighted:.3f}, Silhouette: {sil_unweighted:.3f}")
        print(f"Weighted   ARI: {ari_weighted:.3f}, Silhouette: {sil_weighted:.3f}")
    
        # Step 7: Detection bar plot
        df = pd.DataFrame(detection_max_scores).T.reset_index().melt(
            id_vars='index', var_name='Class', value_name='Max Score'
        )
        df.rename(columns={'index': 'Descriptor'}, inplace=True)
        df["Descriptor"] = df["Descriptor"].apply(lambda x: x.replace("_", "\n"))
    
        plt.figure(figsize=(10, 6))
        sns.set_theme(style="white")
        ax = sns.barplot(data=df, x="Descriptor", y="Max Score", hue="Class", palette=config.COLORS_PALETTE)
    
        plt.title(plot_title, fontsize=14, weight='bold')
        plt.ylabel("Detection Score (%)", fontsize=14)
        plt.xlabel("Descriptor", fontsize=14)
        plt.xticks(rotation=0)
        plt.ylim(0, 110)
        plt.legend(title=None, frameon=False, fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(config.DETECTION_DIRECTORY, output_filename), dpi=600)

        plt.close()
    
        # Step 8: Weights for diagnostics
        weights = {
            descriptor: np.mean(list(class_scores.values()))
            for descriptor, class_scores in detection_max_scores.items()
            if isinstance(class_scores, dict)
        }
    
        results = self.compare_clustering_quality(
            distance_matrices=self.distance_matrices,
            weights=weights,
            neuron_class_df=self.neuron_class_df
        )
    
        metrics = {
            "ARI_unweighted": ari_unweighted,
            "ARI_weighted": ari_weighted,
            "Silhouette_unweighted": sil_unweighted,
            "Silhouette_weighted": sil_weighted
        }
        
        # Save metrics to CSV
        metrics_df = pd.DataFrame([metrics])
        metrics_csv_path = os.path.join(config.DETECTION_DIRECTORY, "clustering_metrics_before_after.csv")
        metrics_df.to_csv(metrics_csv_path, index=False)

    
        return weighted_matrices, combined_matrix, results, metrics

        
    # def combine_matrices_by_average_detection_score(self, detection_max_scores):
    #     """
    #     Computes one combined matrix by summing each descriptor matrix scaled by its average detection score.
    
    #     Parameters:
    #     - detection_max_scores: {descriptor → {class → max score}}
    
    #     Returns:
    #     - weighted_matrices: {descriptor → individually weighted matrix}
    #     - combined_matrix: total sum of all weighted matrices
    #     """
    #     weighted_matrices = {}
    #     combined_sum = None
    
    #     for descriptor, matrix in self.distance_matrices.items():
    #         # Step 1: Retrieve max detection scores across classes for this descriptor
    #         class_scores = detection_max_scores.get(descriptor, {})
    #         avg_score = np.mean(list(class_scores.values())) if class_scores else 1.0
    
    #         # Step 2: Multiply matrix by average score
    #         weighted = matrix.values * avg_score
    #         weighted_df = pd.DataFrame(weighted, index=matrix.index, columns=matrix.columns)
    #         weighted_matrices[descriptor] = weighted_df
    
    #         # Step 3: Accumulate into combined matrix
    #         if combined_sum is None:
    #             combined_sum = weighted.copy()
    #         else:
    #             combined_sum += weighted
    
    #     # Step 4: Return both dictionary of individual weighted matrices and their sum
    #     combined_matrix = pd.DataFrame(combined_sum, index=matrix.index, columns=matrix.columns)
    #     # combined_matrix = sum(mat.values for mat in self.distance_matrices.values()) / len(self.distance_matrices)
    #     # combined_matrix = pd.DataFrame(combined_matrix, index=next(iter(self.distance_matrices.values())).index,
    #     #                        columns=next(iter(self.distance_matrices.values())).columns)
        
        


    #     # Convert to DataFrame
    #     df = pd.DataFrame(detection_max_scores).T.reset_index().melt(id_vars='index', var_name='Class', value_name='Max Score')
    #     df.rename(columns={'index': 'Descriptor'}, inplace=True)
        
    #     # Plot setup
    #     plt.figure(figsize=(10, 6))
    #     sns.set_theme(style="white")  # No grid
        
    #     # Format descriptor names (e.g., "Branching Pattern" → "Branching\nPattern")
    #     df["Descriptor"] = df["Descriptor"].apply(lambda x: x.replace("_", "\n"))
        
    #     # Create bar plot
    #     ax = sns.barplot(data=df, x="Descriptor", y="Max Score", hue="Class", palette=config.COLORS_PALETTE)
        
    #     # Enhance aesthetics
    #     plt.title("Detection Scores by Descriptor", fontsize=14, weight='bold')
    #     plt.ylabel("Detection Score (%)", fontsize=14)
    #     plt.xlabel("Descriptor", fontsize=14)
    #     plt.xticks(rotation=0)  # Make x-tick labels horizontal
    #     plt.ylim(0, 110)  # Extend y-axis limit without affecting data
    #     plt.legend(title=None, frameon=False, fontsize=12)
    #     plt.tight_layout()
        
        
    #     # Save figure
    #     plt.savefig(os.path.join(config.DETECTION_DIRECTORY, "Detection_rates.png"), dpi=300)
    #     plt.close()
        
    #     # Example: detection_max_scores = {'desc1': {'A': 85.2, 'B': 90.1}, ...}
    #     weights = {
    #         descriptor: np.mean(list(class_scores.values()))
    #         for descriptor, class_scores in detection_max_scores.items()
    #             }

    #     results = self.compare_clustering_quality(
    #         distance_matrices=self.distance_matrices,
    #         weights=weights,
    #         neuron_class_df=self.neuron_class_df
    #         )


    #     # # 1. Intra- vs. Inter-class Distance Comparison (with bar plot)
    #     # class_distance_results = self.run_class_distance_tests_with_plot(self.distance_matrices, combined_matrix, labels)
        
    #     # # 2. Leave-One-Out kNN Accuracy (with accuracy bar plot)
    #     # knn_accuracy_results = self.run_knn_accuracy_with_plot(self.distance_matrices, combined_matrix, labels)
        
    #     # # 3. Permutation Test on Combined Matrix (with histogram of null distribution)
    #     # observed_diff, p_value = self.run_permutation_test_with_plot(self.combined_matrix.values, labels, n_permutations=1000)
                
        
    #     return weighted_matrices, combined_matrix, results
    


    def compare_clustering_quality(self, distance_matrices, weights, neuron_class_df, n_clusters=2):
        """
        Compares ARI and dendrogram class separation before and after detection-weighted matrix fusion.
        
        Parameters:
        - distance_matrices: dict of {descriptor_name: pd.DataFrame}
        - weights: dict of {descriptor_name: weight (float)}
        - neuron_class_df: pd.DataFrame with 'neuron_name' and 'class' columns
        - n_clusters: number of clusters to cut dendrogram
        
        Returns:
        - result: dict with ARI scores and dendrogram figures
        """
        neuron_names = distance_matrices[next(iter(distance_matrices))].index.tolist()
        labels_true = neuron_class_df.set_index("neuron_name").loc[neuron_names]["class"].values
    
        # Unweighted average matrix
        avg_matrix = sum(matrix.values for matrix in distance_matrices.values()) / len(distance_matrices)
    
        # Weighted matrix
        combined_matrix = sum(weights[d] * distance_matrices[d].values for d in distance_matrices)
    
        # Perform hierarchical clustering (average linkage)
        linkage_avg = linkage(avg_matrix[np.triu_indices_from(avg_matrix, k=1)], method='average')
        linkage_weighted = linkage(combined_matrix[np.triu_indices_from(combined_matrix, k=1)], method='average')
    
        # Get flat clusters
        labels_avg = fcluster(linkage_avg, n_clusters, criterion='maxclust')
        labels_weighted = fcluster(linkage_weighted, n_clusters, criterion='maxclust')
    
        # Compute ARI
        ari_avg = adjusted_rand_score(labels_true, labels_avg)
        ari_weighted = adjusted_rand_score(labels_true, labels_weighted)
    
        # Plot dendrograms
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        dendrogram(linkage_avg, labels=neuron_names, ax=axes[0])
        axes[0].set_title(f"Unweighted Dendrogram\nARI = {ari_avg:.2f}")
        dendrogram(linkage_weighted, labels=neuron_names, ax=axes[1])
        axes[1].set_title(f"Weighted Dendrogram\nARI = {ari_weighted:.2f}")
        plt.tight_layout()
        plt.tight_layout()
        plt.savefig(os.path.join(config.DETECTION_DIRECTORY, "compare_clustering_quality.png"), dpi=600)
        plt.close()
   
    
        return {
            "ARI (Unweighted)": ari_avg,
            "ARI (Weighted)": ari_weighted
        }



    
    # Inline-commented and visualization-enhanced versions of the analysis functions
    '''
    1. t-Tests on Class Distances
        Compare intra-class and inter-class distances:
        
        For each descriptor and for the combined matrix:
        
        Compute mean intra-class distance (same class).
        
        Compute mean inter-class distance (different classes).
        
        Perform a paired t-test (or Wilcoxon signed-rank test if not normally distributed) between:
        
        (intra-class distance - inter-class distance) for each descriptor
        
        (intra-class distance - inter-class distance) for combined matrix
        
        Large, significant differences imply better class separation.
        
    2. Distance-based Classification Accuracy (e.g., kNN)
        Use leave-one-out or k-fold cross-validation.
        
        Input: distance matrix → k-nearest neighbor classifier (e.g., k=1).
        
        Output: classification accuracy for each matrix.
        
        Run accuracy comparison between:
        
        Each individual descriptor
        
        The combined matrix
        
        Use a McNemar’s test to compare paired classifier performance.
        
    3. Permutation Tests
        Randomly permute class labels.
        
        Recompute average detection scores and combined matrix.
        
        Compare true inter-vs-intra class difference vs. null distribution.
        
        This tests whether the weighting scheme is better than chance.


    
    '''
    
    def _get_labels_and_matrix(self, matrix):
        """
        Safely retrieve class labels matching the matrix index order,
        and return a deep copy of the matrix.
        """
        matrix_copy = matrix.copy()
        neuron_names = matrix_copy.index.tolist()
    
        merged = pd.DataFrame({'neuron_name': neuron_names}).merge(
            self.neuron_class_df[['neuron_name', 'class']],
            on='neuron_name',
            how='left'
        )
    
        if merged['class'].isnull().any():
            missing = merged[merged['class'].isnull()]['neuron_name'].tolist()
            raise ValueError(f"Missing class labels for neurons: {missing}")
    
        labels = merged['class'].values
        return labels, matrix_copy

    
    # Compute intra- and inter-class distances
    def compute_class_distances(self, matrix, labels):
        intra_dists = []
        inter_dists = []
        for i, j in itertools.combinations(range(len(labels)), 2):
            if labels[i] == labels[j]:  # Same class → intra-class
                intra_dists.append(matrix[i, j])
            else:  # Different classes → inter-class
                inter_dists.append(matrix[i, j])
        return np.mean(intra_dists), np.mean(inter_dists)
    
    # Run distance analysis and visualize results
    def run_class_distance_tests_with_plot(self, distance_matrices, combined_matrix, labels):
        results_ttests = []
    
        # Loop through each descriptor matrix
        for name, mat_df in distance_matrices.items():
            labels, matrix = self._get_labels_and_matrix(mat_df)
            intra, inter = self.compute_class_distances(matrix.values, labels)
            results_ttests.append((name, intra, inter, intra - inter))
    
        # Combined matrix
        labels, matrix = self._get_labels_and_matrix(combined_matrix)
        combined_intra, combined_inter = self.compute_class_distances(matrix.values, labels)
        results_ttests.append(('combined', combined_intra, combined_inter, combined_intra - combined_inter))
    
        # Create DataFrame
        df = pd.DataFrame(results_ttests, columns=["Matrix", "Intra", "Inter", "Diff"])
    
        # Plot intra- vs. inter-class distances
        df_melted = df.melt(id_vars="Matrix", value_vars=["Intra", "Inter"], var_name="Type", value_name="Distance")
        plt.figure(figsize=(10, 6))
        sns.barplot(data=df_melted, x="Matrix", y="Distance", hue="Type")
        plt.title("Intra vs Inter Class Distances")
        plt.ylabel("Average Distance")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(config.DETECTION_DIRECTORY, "t-Tests_on_Class_Distances.png"), dpi=300)
        plt.show()
    
        return df
    


    def knn_accuracy(self, distance_matrix, labels):
        loo = LeaveOneOut()  # Leave-One-Out cross-validation
        preds = []
        for train_index, test_index in loo.split(distance_matrix):
            # For test sample, get distances to all training samples
            dists = distance_matrix[test_index[0], train_index]
            
            # Find the nearest training sample
            nearest_idx = train_index[np.argmin(dists)]
            
            # Predict the class label of the nearest neighbor
            preds.append(labels[nearest_idx])
        
        # Return accuracy and list of predicted labels
        return accuracy_score(labels, preds), preds

    
    # Compute kNN accuracy using Leave-One-Out strategy
    def run_knn_accuracy_with_plot(self, distance_matrices, combined_matrix, labels):
        acc_results = []
    
        # Combined matrix first
        labels, matrix = self._get_labels_and_matrix(combined_matrix)
        combined_acc, _ = self.knn_accuracy(matrix.values, labels)
        acc_results.append(("combined", combined_acc))
    
        # Each individual descriptor
        for name, mat_df in distance_matrices.items():
            labels, matrix = self._get_labels_and_matrix(mat_df)
            acc, _ = self.knn_accuracy(matrix.values, labels)
            acc_results.append((name, acc))
    
        # Convert to DataFrame
        df = pd.DataFrame(acc_results, columns=["Matrix", "Accuracy"])
    
        # Plot accuracy
        plt.figure(figsize=(10, 5))
        sns.barplot(data=df, x="Matrix", y="Accuracy")
        plt.title("kNN Classification Accuracy (Leave-One-Out)")
        plt.ylim(0, 1)
        plt.ylabel("Accuracy")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(config.DETECTION_DIRECTORY, "KNN_Classification_Accuracy.png"), dpi=300)
        plt.show()
    
        return df
    
    # Run permutation test and show histogram of null distribution
    def run_permutation_test_with_plot(self, combined_matrix, labels, n_permutations=1000):
        labels, matrix = self._get_labels_and_matrix(combined_matrix)
        orig_intra, orig_inter = self.compute_class_distances(matrix.values, labels)
        orig_diff = orig_intra - orig_inter
        diffs = []
    
        # Generate null distribution via label shuffling
        for _ in range(n_permutations):
            permuted = np.random.permutation(labels)
            intra, inter = self.compute_class_distances(matrix.values, permuted)
            diffs.append(intra - inter)
    
        # Compute empirical p-value
        p_value = np.mean(np.abs(diffs) >= np.abs(orig_diff))
    
        # Plot histogram of null distribution
        plt.figure(figsize=(10, 5))
        sns.histplot(diffs, bins=30, kde=True, color="gray")
        plt.axvline(orig_diff, color="red", linestyle="--", label=f'Observed Diff = {orig_diff:.4f}')
        plt.title(f"Permutation Test on Combined Matrix (p = {p_value:.4f})")
        plt.xlabel("Intra - Inter Class Distance")
        plt.ylabel("Frequency")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(config.DETECTION_DIRECTORY, "Permutation_tests.png"), dpi=300)
        plt.show()

        return orig_diff, p_value
    
    # Ready to run once user provides distance_matrices, combined_matrix, and labels.


       
# ================================================================

    def rank_descriptors_by_detection_quality(self, top_k=None):
        """
        Ranks and visualizes descriptors per class based on:
        - Mean detection score (average separability)
        - Threshold detection score (% neurons well-separated)
        - Median and variance of individual neuron detection scores (beta values)
    
        Parameters:
        - top_k: optional integer, keeps only top-k descriptors per class
    
        Returns:
        - Dictionary {class_name: ranked DataFrame}
        """
    
        # All descriptor names for which detection results exist
        descriptors = self.detection_results.keys()
    
        # All unique neuron classes (e.g., 'pyramidal', 'interneuron', etc.)
        classes = self.neuron_class_df["class"].unique()
    
        for cls in classes:
            rows = []
    
            for descriptor in descriptors:
                # Retrieve stored mean detection score for this class and descriptor
                mean_score = self.detection_results[descriptor][0].get(cls, 0)
    
                # Retrieve stored threshold score (e.g., % of neurons with beta ≥ 0.7)
                threshold_score = self.detection_results[descriptor][1].get(cls, 0)
    
                # Get list of per-neuron detection beta values for this class-descriptor combo
                beta_tuples = self.beta_distributions.get(descriptor, {}).get(cls, [])
    
                # Clean and extract raw numeric beta values
                beta_values = [
                    float(entry["beta"])
                    for entry in beta_tuples
                    if "beta" in entry and isinstance(entry["beta"], (int, float, np.floating))
                ]
    
                # Handle missing or empty beta lists gracefully
                if len(beta_values) == 0:
                    beta_median = 0                # No data → assume weak median
                    beta_variance = 1              # High penalty for instability
                else:
                    beta_median = np.median(beta_values)    # Central detection tendency
                    beta_variance = np.var(beta_values)     # Spread of detection (instability)
    
                # --- Composite Score ---
                # This weighted sum reflects both strength and reliability of detection
                # Weights:
                #   0.4 mean score      → captures average separability 	Core indicator of average detection; broadest summary
                #   0.3 threshold score → emphasizes confident detection in many neurons Captures consistency across neurons; favors reliable descriptors
                #   0.2 beta median     → stable central tendency, ignores outliers Robust to outliers; reflects the "typical" detection performance
                #  -0.1 beta variance   → penalizes inconsistency across neurons Penalizes inconsistency; filters out unstable descriptors
                composite_score = (
                    0.4 * mean_score +
                    0.3 * threshold_score +
                    0.2 * beta_median * 100 -          # scaled to match percentage format
                    0.1 * beta_variance * 100          # scaled to match others
                )
    
                # Store results in a structured row
                rows.append({
                    "descriptor": descriptor,
                    "mean": mean_score,
                    "threshold": threshold_score,
                    "beta_median": beta_median,
                    "beta_variance": beta_variance,
                    "composite_score": composite_score
                })
    
            # Convert list of dicts into DataFrame and sort by composite score (descending)
            df = pd.DataFrame(rows).sort_values("composite_score", ascending=False)
    
            # If user requests top_k only, slice the DataFrame
            if top_k:
                df = df.head(top_k)
    
            # Store the ranked list for this class
            self.rankings[cls] = df.reset_index(drop=True)
    
            # Save to CSV for inspection or further analysis
            df.to_csv(
                os.path.join(config.DETECTION_DIRECTORY, f"descriptor_ranking_class_{cls}.csv"),
                index=False
            )
    
            # Plot: horizontal bar chart of descriptor rankings for this class
            plt.figure(figsize=(10, 6))
            sns.barplot(data=df, x="descriptor", y="composite_score", palette="viridis")
            plt.title(f"Descriptor Ranking for Class: {cls}")
            plt.ylabel("Composite Score")
            plt.xlabel("Descriptor")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(
                os.path.join(config.DETECTION_DIRECTORY, f"descriptor_ranking_plot_class_{cls}.png"),
                dpi=300
            )
            plt.close()

        return self.rankings

    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    import numpy as np
    import os
    
    import pandas as pd
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt
    import os
    
    def combine_descriptors_by_composite_score(self, normalize=True):
        """
        Returns one weighted matrix per descriptor (all neurons included).
        Each matrix is scaled by the descriptor's average composite score across all classes.
        Uses `self.rankings`, which contains per-class composite scores.
        """
        descriptor_matrices = {}
    
        for descriptor, matrix in self.distance_matrices.items():
            # Step 1: Collect composite scores across all classes for this descriptor
            composite_scores = []
            for cls, df in self.rankings.items():
                df_indexed = df.set_index("descriptor")
                if descriptor in df_indexed.index:
                    composite_scores.append(df_indexed.loc[descriptor]["composite_score"])
            
            # Step 2: Compute average composite score (fallback to 1.0 if missing)
            avg_score = np.mean(composite_scores) if composite_scores else 1.0
    
            # Step 3: Multiply the distance matrix by the average composite score
            weighted_matrix = avg_score * matrix.values
    
            # Step 4: Normalize the matrix if requested
            if normalize:
                norm = np.linalg.norm(weighted_matrix)
                weighted_matrix = weighted_matrix / norm if norm > 0 else weighted_matrix
    
            # Step 5: Save matrix to dictionary
            descriptor_matrices[descriptor] = pd.DataFrame(
                weighted_matrix,
                index=matrix.index,
                columns=matrix.columns
            )
    
        return descriptor_matrices


    # def combine_descriptors_by_composite_score(self):
            
    #     # Combine matrices using composite scores as weights for each class
    #     combined_matrices = {}
    #     composite_scores_matrix = {}
        
    #     for cls, df in self.rankings.items():
    #         template = next(iter(self.distance_matrices.values()))
    #         index = template.index
        
    #         weighted_sum = np.zeros_like(template.values)
    #         total_score = df['composite_score'].sum()
            
    #         composite_scores_matrix[cls] = {}
        
    #         for _, row in df.iterrows():
    #             descriptor = row['descriptor']
    #             score = row['composite_score']
    #             weight = score / total_score if total_score > 0 else 0
        
    #             composite_scores_matrix[cls][descriptor] = score
    #             weighted_sum += weight * self.distance_matrices[descriptor].values
        
    #         combined_matrix = pd.DataFrame(weighted_sum, index=index, columns=index)
    #         combined_matrices[cls] = combined_matrix
                    
    #     # Normalize combined matrices
    #     normalized_matrices = {}
    #     for cls, mat in combined_matrices.items():
    #         norm = np.linalg.norm(mat.values)
    #         normalized = mat / norm if norm > 0 else mat
    #         normalized_matrices[cls] = pd.DataFrame(normalized, index=mat.index, columns=mat.columns)
         
    #     return normalized_matrices
    
            
    
    
    def plot_composite_score_heatmap(self):
        def interpret(score):
            if score > 75:
                return "Strong"
            elif score > 60:
                return "Partial"
            elif score > 40:
                return "Poor"
            else:
                return "None"
    
        score_matrix = {}
        interpretation_matrix = {}
    
        for cls, df in self.rankings.items():
            score_matrix[cls] = {}
            interpretation_matrix[cls] = {}
            for _, row in df.iterrows():
                descriptor = row["descriptor"]
                score = row["composite_score"]
                score_matrix[cls][descriptor] = score
                interpretation_matrix[cls][descriptor] = interpret(score)
    
        score_df = pd.DataFrame(score_matrix).T
        interpretation_df = pd.DataFrame(interpretation_matrix).T
    
        color_map = {
            "Strong": "#1a9850",
            "Partial": "#91cf60",
            "Poor": "#fee08b",
            "None": "#d73027"
        }
    
        # Convert interpretation labels to RGB colors
        rgb_matrix = interpretation_df.replace(color_map)
    
        # Prepare color matrix for heatmap
        color_array = rgb_matrix.applymap(mcolors.to_rgb).to_numpy()
    
        fig, ax = plt.subplots(figsize=(12, 6))
    
        # Create heatmap with RGB cell colors
        sns.heatmap(
            score_df,
            annot=score_df.round(1),
            fmt="",
            linewidths=0.5,
            cbar=False,
            cmap=None,
            mask=score_df.isna(),
            square=False,
            xticklabels=True,
            yticklabels=True,
            ax=ax
        )
    
        # Overlay color cells manually
        for i in range(score_df.shape[0]):
            for j in range(score_df.shape[1]):
                if not pd.isna(score_df.iloc[i, j]):
                    ax.add_patch(plt.Rectangle(
                        (j, i),
                        1,
                        1,
                        fill=True,
                        color=color_array[i, j],
                        linewidth=0
                    ))
    
        ax.set_title("Composite Score Interpretation per Class and Descriptor")
        ax.set_xlabel("Descriptor")
        ax.set_ylabel("Class")
        plt.xticks(rotation=45)
    
        import matplotlib.patches as mpatches
        legend_patches = [mpatches.Patch(color=color, label=label) for label, color in color_map.items()]
        plt.legend(
            handles=legend_patches,
            title="Interpretation",
            bbox_to_anchor=(1.05, 1),
            loc='upper left',
            borderaxespad=0.
        )
    
        plt.tight_layout()
        plt.savefig(os.path.join(config.DETECTION_DIRECTORY, "composite_score_interpretation_heatmap.png"), dpi=300)
        plt.close()
        
        
    def explained_variance_from_distances(self):
        explained_variances = {}
        summary_rows = []
    
        for name, dist_matrix in self.distance_matrices.items():
            mds = MDS(n_components=10, dissimilarity='precomputed', random_state=42)
            feature_matrix = mds.fit_transform(dist_matrix.values)
            pca = PCA()
            pca.fit(feature_matrix)
            explained_var = pca.explained_variance_ratio_
            explained_variances[name] = explained_var
    
            plt.figure()
            plt.plot(range(1, len(explained_var) + 1), explained_var.cumsum(), marker='o')
            plt.title(f"Cumulative Variance Explained by PCA (MDS on '{name}')")
            plt.xlabel("Principal Component")
            plt.ylabel("Cumulative Variance Explained")
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(config.DETECTION_DIRECTORY, f"pca_variance_mds_{name}.png"))
            plt.close()
    
            # Track thresholds
            cum_var = np.cumsum(explained_var)
            row = {"descriptor": name}
            for thresh in [0.5, 0.8, 0.9]:
                num_components = np.searchsorted(cum_var, thresh) + 1
                row[f"components_to_{int(thresh*100)}%"] = num_components
            summary_rows.append(row)
    
        summary_df = pd.DataFrame(summary_rows).set_index("descriptor")
    
        # Heatmap
        plt.figure(figsize=(8, 5))
        ax = plt.gca()
        im = ax.imshow(summary_df, cmap="YlGnBu", aspect="auto")
    
        for i in range(summary_df.shape[0]):
            for j in range(summary_df.shape[1]):
                ax.text(j, i, str(summary_df.iloc[i, j]), ha="center", va="center", color="black")
    
        ax.set_xticks(np.arange(len(summary_df.columns)))
        ax.set_xticklabels(summary_df.columns, rotation=45)
        ax.set_yticks(np.arange(len(summary_df.index)))
        ax.set_yticklabels(summary_df.index)
        plt.title("Components Required to Reach Variance Thresholds")
        plt.colorbar(im, label="Number of Components")
        plt.tight_layout()
        plt.savefig(os.path.join(config.DETECTION_DIRECTORY, f"variance_components_heatmap.png"))
        plt.close()
    
        # Select descriptors based on variance threshold
        selected_by_variance = summary_df[summary_df["components_to_80%"] <= 3].index.tolist()   
       
    
        return explained_variances, selected_by_variance  



    
   