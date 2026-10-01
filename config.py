import seaborn as sns
import os
import datetime

# Resolve inputs and outputs from this repository. The selected cohort remains
# data/Briggs (67 V1/V2 + 50 PMLS/PLLS/21a); data_with_stellates is not used.
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Dataset folder under data/. SWC files are read from
#     data/<PROJECT_NAME>/<class>/*.swc
# and each immediate subfolder name becomes the class label.
PROJECT_NAME = 'Briggs'

timestamp = datetime.datetime.now().strftime("%m%d%y_%H%M")

BASE_DIRECTORY = os.path.join(PROJECT_ROOT, 'data', PROJECT_NAME)

# Output folder for the current run. Every path below resolves from this
# repository, so a fresh clone runs without editing anything. Override with
# NEUROTOPO_RUN so a new run never overwrites a previous one.
RUN_NAME = os.environ.get('NEUROTOPO_RUN', 'results')
SAVE_DIRECTORY = os.path.join(PROJECT_ROOT, 'figures', RUN_NAME)

# Define the directories for plots and pickles
PLOTS_DIRECTORY = os.path.join(SAVE_DIRECTORY, 'plots')
PICKLE_DIRECTORY = os.path.join(SAVE_DIRECTORY, 'pickles')
DETECTION_DIRECTORY = os.path.join(PLOTS_DIRECTORY, "Detection")
METRIC_LEARNING_DIRECTORY = os.path.join(PLOTS_DIRECTORY, "Metric_Learning")
MORPHOMETRICS_DIRECTORY = os.path.join(PLOTS_DIRECTORY, "Morphometrics")
DESCRIPTORS_DIRECTORY = os.path.join(PLOTS_DIRECTORY, "Descriptors")

# **NEW**: Directory for L-Measure analysis figures
LMEASURE_FIGURES = os.path.join(SAVE_DIRECTORY, 'lmeasure_figures')
LMEASURE_CSV_PATH = os.path.join(BASE_DIRECTORY, "lmeasure_data.csv")
LMEASURE_DATA_STANDARIZE = True
# l_MEASURE STUFF
LMEASURE_NORMALIZE_METHOD ="standard"  #  minmax , robust

# Ensure the plots and pickles directories exist
# Ensure the save directory exists
os.makedirs(SAVE_DIRECTORY, exist_ok=True)
os.makedirs(PLOTS_DIRECTORY, exist_ok=True)
os.makedirs(PICKLE_DIRECTORY, exist_ok=True)
os.makedirs(LMEASURE_FIGURES, exist_ok=True)
os.makedirs(DETECTION_DIRECTORY, exist_ok=True)
os.makedirs(METRIC_LEARNING_DIRECTORY, exist_ok=True)
os.makedirs(MORPHOMETRICS_DIRECTORY, exist_ok=True)
os.makedirs(DESCRIPTORS_DIRECTORY, exist_ok=True)





# GREEN    = #91c2b3   -- Darker #65aa95 
# pink     = #f29696   -- Darker #ed7070
# pach     = #f2b888   
# Fuchsia  = #d81b60   

DISTANCE_MATRICES_COMBINATION_MODE=''
DISTANCE_METRIC = "l1"  #this is used in distances l1, wasserstein

# Rescale each descriptor distance matrix to unit mean before combining them.
#
# The descriptors measure different physical quantities and the L1 distance
# between step functions inherits those units: Spread is an area in um^2, Leaf a
# fraction bounded by 1, Branching a count, Energy a squared magnitude. Their
# mean pairwise distances consequently span roughly four orders of magnitude, so
# an unnormalised sum is governed by whichever descriptor carries the largest
# units rather than by what any of them measure. Restating a descriptor in
# different units would change the combined distance without changing any
# underlying measurement.
#
# The detection weights span about 1.3x, which cannot offset that imbalance, so
# without rescaling the weighting has no effect on the combined matrix.
#
# Dividing by the mean off-diagonal distance is a single positive scalar per
# matrix. It preserves each descriptor's internal rank ordering and leaves its
# own dendrogram unchanged, and detection scores depend only on neighbour rank
# order so they remain valid without recomputation.
NORMALIZE_DESCRIPTOR_DISTANCES = True
NUM_OF_CLUSTERS = 2
LINKAGE_METHOD = 'ward'
MAX_ITERATIONS = 1000000
# MAX_ITERATIONS = 100
CLEAN_DATA = True   # this will Drop constant Columns and Low variance Columns


ENGINEERED_FEATURES_AUTO_DROP = True
ENGINEERED_FEATURES_STANDARIZE =  True
ENGINEERED_FEATURES_STANDARIZE_METHOD ="standard"  #  minmax , robust
ENGINEERED_FEATURES_DROP_LOW_VARIANCE_CONSTANT = False


# Define default types of neuron components to be removed
# # Mapping from component names to SWC type numbers
# 0 - undefined
# 1 - soma
# 2 - axon
# 3 - (basal) dendrite
# 4 - apical dendrite
# 5+ - custom
REMOVE_TYPES = ['axon']  


# Setting for normalization in data processing FOR THE SWC FILE
NORMALIZE_DESCRIPTOR_RADII = True





# EXCLUDED_DESCRIPTORS = [ 'Polarity', 'Energy','Path','Volume','Tortuosity']
# EXCLUDED_DESCRIPTORS = [ 'Polarity', 'Energy','Path','Volume']
# EXCLUDED_DESCRIPTORS = ['Polarity', 'Energy']
EXCLUDED_DESCRIPTORS = []


PLOT_STEP_FUNCTION   = True
PLOT_SUB_TREE_3D     = False
PLOT_TREE_3D         = False
PLOT_POLARITY        = False
# ----------------------------------
PLOT_NEURON          = False  # Disabled for this figure-reproduction run.
PLOT_SWC_FILTER_TYPE = 2
# PLOT_SWC_FILTER_TYPE = None

# ----------------------------------
PLOT_NEURON_3D       = False  # utl.plot_graph_3d_with_plotly  (Called from neuron_processor)

FEATURE_NORMALIZATION_METHOD = "min-max"   # "z-score"  or "none"


# --------------------Dark Green --Brown--- 
CLASS_COLORS = []
# COLORS_PALETTE = ['#ea801c','#1a80bb','#7BC8F6','#C79FEF','#004d40','#846b54']
# DENDRO_CLUSTER_COLORS = ['#f77189','#36ada4','#f29696','#f2b888', '#d81b60','#91c2b3']

COLORS_PALETTE = ['#f77189','#36ada4','#7BC8F6','#C79FEF','#004d40','#846b54']
DENDRO_CLUSTER_COLORS = ['#ea801c','#1a80bb','#f29696','#f2b888', '#d81b60','#91c2b3']

# COLORS_PALLETE = sns.color_palette("husl", 10)

# # Generate a full HUSL color palette
# full_husl_palette = sns.color_palette("husl", 20)  # Generates 20 colors

# # Select specific colors by index, e.g., 0, 5, 10, and 15
# selected_indices = [0, 4]
# COLORS_PALLETE = [full_husl_palette[i] for i in selected_indices]





################*********************************************##################
# Define a scale factor
scale_factor = 0.8

# Configuration dictionary for Matplotlib plots
plot_settings = {
    'font.size': 10 * scale_factor,
    'axes.labelsize': 12 * scale_factor,
    'axes.titlesize': 14 * scale_factor,
    'xtick.labelsize': 10 * scale_factor,
    'ytick.labelsize': 10 * scale_factor,
    'legend.fontsize': 10 * scale_factor,
    'lines.linewidth': 1.5 * scale_factor,
    'lines.markersize': 6 * scale_factor,
}

# in the code use the folowing 

# with plt.rc_context(plot_settings):
#     plot......

################*********************************************##################

# HIGHLIGHT_NEURONS = ['62_1_2_PLLS','62_4_1_PMLS','62_3_1_PMLS','62_4_3_PMLS', 
#                      '62_4_2_PMLS','62_4_4_PMLS', '72513V1-3Slice3Cell1', '73014V12Slice9Cell1']



HIGHLIGHT_NEURONS = []

# 31_1_3_PMLS and 31_1_6_PMLS carry no apical label, but their dendritic fields
# are not radially symmetric (anisotropy 0.19 and 0.40 against a pyramidal
# population mean of 0.37, the latter above the population median) and both have
# a normal number of primary dendrites. There is no morphological evidence that
# they are non-pyramidal, so the missing label most likely reflects a tracing
# convention in the source archive rather than a stellate morphology. Spread is
# measured over the whole dendritic tree, so both remain evaluable and the cohort
# stays at 117.
EXCLUDED_NEURONS = ['S2A_2_3_28_b8_v1_L2_3_s1_N4_LH']

# EXCLUDED_NEURONS  = ['S2A-2_3_28_b8_v1_L2-3_s1_N4_LH']
# EXCLUDED_NEURONS  = ['S2A-2_3_28_b8_v1_L2-3_s1_N4_LH', 'S1A_3_15_NAC_MPFC_b3_v1_L2-3_s2_N1_RH', 'S1C_b9_26_b6_v1_L2-3_s9_N1_RH.swc',
#                       'S2A-2_3_28_b8_v1_L2-3_s1_N3_LH','S1C_b9_26_b6_v1_L2-3_s13_N5_LH','C2A_3-28_B8_V1_L2-3_S1_N3_LH','C1B_3-26_B8_V1_L2-3_S3_N1_RH']

# EXCLUDED_NEURONS = [
#     '121012V1-3Slice7Cell1', 
#     '32315-V1-4Slice6Cell2', 
#     '62_1_1_PLLS',
#     '84_4_1_21a',    
#     '121012V1-2Slice4Cell1',
#     '32315-V1-3Slice2Cell1',
#     '32315-V1-3Slice5Cell1',
#     '32315-V1-3Slice8Cell1',
#     '72513V1-3Slice9Cell1',
#     '72513V1-6Slice2Cell1',
#     '72715-V1-3Slice5Cell1',
#     '72715-V1-3Slice9Cell1',
#     '72715-V1-5Slice4Cell1',
#     '73014V1-2Slice11Cell1',
#     '73014V1-2Slice1Cell1',
#     '73014V1-2Slice1Cell2',
#     '73014V1-2Slice2Cell2',
#     '73014V1-2Slice2Cell3',
#     '73014V1-2Slice2Cell4',
#     '73014V1-2Slice3Cell1',
#     '73014V1-2Slice3Cell2',
#     '73014V1-2Slice4Cell1',
#     '73014V1-2Slice4Cell2',
#     '73014V1-2Slice5Cell1',
#     '73014V1-2Slice5Cell2',
#     '73014V1-2Slice8Cell1',
#     '73014V1-2Slice8Cell2',
#     '73014V1-2Slice9Cell2',
#     '73014V1-2Slice9Cell3',
#     '73014V1-3Slice4Cell1',
#     '73014V1-3Slice5Cell1',
#     '73014V1-3Slice5Cell2'
#     ]
