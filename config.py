
import os

# =============================================================================
# Directory Configuration
# =============================================================================

# Base directory (project root)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Data directory for downloaded datasets
DATA_DIR = os.path.join(BASE_DIR, "data")

# Results directory for experiment outputs
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# Figures directory for visualizations
FIGURES_DIR = os.path.join(BASE_DIR, "figures")

# Create directories if they don't exist
for directory in [DATA_DIR, RESULTS_DIR, FIGURES_DIR]:
    os.makedirs(directory, exist_ok=True)


# =============================================================================
# Dataset Configuration (Static Snapshots)
# =============================================================================
# For Scale-Down we treat the final citation graphs as static snapshots,
# plus the static Epinions trust network.

DATASETS = {
    # =========================================================================
    # TEMPORAL DATASETS (final snapshot) - 用于 Scale-Down
    # 这2个也有时间戳，用于 Back-in-Time
    # =========================================================================
    "cit-HepTh": {
        "url": "https://snap.stanford.edu/data/cit-HepTh.txt.gz",
        "filename": "cit-HepTh.txt.gz",
        "directed": True,
        "description": "ArXiv HEP-TH citation network (27,770 nodes, 352,807 edges)",
    },
    "cit-HepPh": {
        "url": "https://snap.stanford.edu/data/cit-HepPh.txt.gz",
        "filename": "cit-HepPh.txt.gz",
        "directed": True,
        "description": "ArXiv HEP-PH citation network (34,546 nodes, 421,578 edges)",
    },
    # =========================================================================
    # STATIC DATASETS - 只用于 Scale-Down (无时间戳)
    # =========================================================================
    "soc-Epinions1": {
        "url": "https://snap.stanford.edu/data/soc-Epinions1.txt.gz",
        "filename": "soc-Epinions1.txt.gz",
        "directed": True,
        "description": "Epinions who-trusts-whom network (75,879 nodes, 508,837 edges)",
    },
    "wiki-Vote": {
        "url": "https://snap.stanford.edu/data/wiki-Vote.txt.gz",
        "filename": "wiki-Vote.txt.gz",
        "directed": True,
        "description": "Wikipedia adminship voting network (7,115 nodes, 103,689 edges)",
    },
    "p2p-Gnutella31": {
        "url": "https://snap.stanford.edu/data/p2p-Gnutella31.txt.gz",
        "filename": "p2p-Gnutella31.txt.gz",
        "directed": True,
        "description": "Gnutella P2P file sharing network (62,586 nodes, 147,892 edges)",
    },
}


# =============================================================================
# Sampling Configuration
# =============================================================================

# Sampling ratios to test (fraction of original graph nodes)
SAMPLING_RATIOS = [0.10, 0.15, 0.20]

# Number of independent runs per configuration (for statistical stability)
NUM_RUNS = 3

# Random seed for reproducibility
RANDOM_SEED = 42


# =============================================================================
# Sampling Method Configuration
# =============================================================================

# Baseline methods from Leskovec & Faloutsos (2006)
BASELINE_METHODS = [
    "RN",   # Random Node sampling - uniform random node selection
    "RPN",  # Random PageRank Node sampling - PageRank-weighted selection
    "RDN",  # Random Degree Node sampling - degree-weighted selection
    "RW",   # Random Walk sampling - walk-based exploration with restart
    "RJ",   # Random Jump - random walk with random teleportation
    "FF"    # Forest Fire sampling - BFS-like burning exploration
]

# Hybrid method combinations: (node_selection_method, exploration_method)
# Our contribution: combining node selection with exploration strategies
HYBRID_COMBINATIONS = [
    ("RN", "RW"),    # Random + Random Walk
    ("RN", "FF"),    # Random + Forest Fire
    ("RPN", "RW"),   # PageRank + Random Walk
    ("RPN", "FF"),   # PageRank + Forest Fire
    ("RDN", "RW"),   # Degree + Random Walk
    ("RDN", "FF"),   # Degree + Forest Fire
]

# Alpha values for hybrid methods
# Alpha = fraction of nodes selected via node selection method
# (1-Alpha) = fraction explored via exploration method
HYBRID_ALPHA_VALUES = [0.3, 0.5, 0.7]


# =============================================================================
# Method-Specific Parameters
# =============================================================================
# Parameters from Leskovec & Faloutsos (2006), Section 4

# Random Walk restart probability (c = 0.15 is standard in literature)
# With probability c, the walk restarts from the starting node
RANDOM_WALK_RESTART_PROB = 0.15

# Maximum steps multiplier for random walk
# Total max steps = n_samples * RANDOM_WALK_MAX_STEPS_MULTIPLIER
RANDOM_WALK_MAX_STEPS_MULTIPLIER = 100

# =============================================================================
# Forest Fire Parameters
# =============================================================================
# 
# From Leskovec & Faloutsos (2006), Section 4.3.3:
#
# SCALE-DOWN GOAL (p_f = 0.7):
#   - "For Scale-down goal best performance was obtained with high values of p_f 
#     (p_f >= 0.6) where every fire eventually floods the connected component."
#   - Larger fires explore more of the node's vicinity, helping match overall 
#     graph properties like degree distribution and clustering.
#   - Setting p_f=0.7 means on average burning 1/(1-0.7) = 3.3 edges per node
#
# BACK-IN-TIME GOAL (p_f = 0.2):
#   - "For Back-in-time goal, we obtain good results for 0 <= p_f <= 0.4, 
#     obtaining the best D-statistic of 0.13 at p_f = 0.20."
#   - Smaller fires explore less vicinity, better mimicking the gradual temporal
#     evolution of the graph (fewer edges burned = slower growth simulation).
#   - Setting p_f=0.2 means on average burning 1/(1-0.2) = 1.25 edges per node
#
# Reference: Table in Section 4.3.3, page 635

FF_FORWARD_PROB_SCALEDOWN = 0.7   # High p_f for scale-down (larger fires)
FF_FORWARD_PROB_BACKTIME = 0.2   # Low p_f for back-in-time (smaller fires)
FF_BACKWARD_PROB = 0.0           # p_b = 0 (no backward burning, as in original paper)


# =============================================================================
# Evaluation Configuration
# =============================================================================

# Number of source nodes for hop-plot estimation (S5, S6)
# Higher values = more accurate but slower
HOP_PLOT_SAMPLES = 300

# Number of singular values to compute (S7, S8)
NUM_SINGULAR_VALUES = 30

# Whether to include S6 (hop-plot on largest WCC) by default
# S6 is important for comparing connectivity structure
INCLUDE_S6_DEFAULT = True

# Whether to use log-transform for power-law distributions
# Recommended for degree distributions, component sizes
USE_LOG_TRANSFORM_DEFAULT = True


# =============================================================================
# Back-in-Time Configuration
# =============================================================================

# Number of time snapshots for back-in-time evaluation
# Creates Snapshot_1, Snapshot_2, ..., Snapshot_N representing graph states over time
# NOTE: These are called "Snapshots" to avoid confusion with temporal METRICS T1-T5
NUM_TIME_SNAPSHOTS = 5

# Method for creating time snapshots
# "equal_time": Equal time intervals between snapshots
# "equal_nodes": Equal number of nodes added per snapshot
TIME_SNAPSHOT_METHOD = "equal_time"


# =============================================================================
# Evaluation Metrics
# =============================================================================

# Static metrics (S1-S9) - measured on single graph snapshot
STATIC_METRICS = [
    "S1_in_degree",      # In-degree distribution
    "S2_out_degree",     # Out-degree distribution
    "S3_wcc",            # Weakly connected component size distribution
    "S4_scc",            # Strongly connected component size distribution
    "S5_hop_plot",       # Hop-plot (reachable pairs at distance h)
    "S6_hop_plot_wcc",   # Hop-plot on largest WCC only
    "S7_singular_vec",   # First left singular vector distribution
    "S8_singular_val",   # Singular value distribution
    "S9_clustering",     # Clustering coefficient distribution (C_d by degree)
]

# Temporal metrics (T1-T5) - measured across time snapshots
# These capture how graph properties EVOLVE over time
TEMPORAL_METRICS = [
    "T1_dpl",            # Densification Power Law: e(t) ∝ n(t)^a
    "T2_diameter",       # Effective diameter over time
    "T3_cc_size",        # Largest connected component size over time
    "T4_singular",       # Largest singular value over time
    "T5_clustering",     # Average clustering coefficient over time
]


# =============================================================================
# Visualization Configuration
# =============================================================================

# Figure size (width, height) in inches
FIGURE_SIZE = (10, 6)

# DPI for saved figures
FIGURE_DPI = 150

# Figure format for saved files
FIGURE_FORMAT = "png"

# Color scheme for methods (consistent across all plots)
METHOD_COLORS = {
    # Baseline methods
    "RN": "#1f77b4",      # Blue
    "RPN": "#ff7f0e",     # Orange
    "RDN": "#2ca02c",     # Green
    "RW": "#d62728",      # Red
    "RJ": "#9467bd",      # Purple
    "FF": "#8c564b",      # Brown
    # Hybrid methods
    "HYB-RN-RW": "#e377c2",   # Pink
    "HYB-RN-FF": "#7f7f7f",   # Gray
    "HYB-RPN-RW": "#bcbd22",  # Yellow-green
    "HYB-RPN-FF": "#17becf",  # Cyan
    "HYB-RDN-RW": "#aec7e8",  # Light blue
    "HYB-RDN-FF": "#ffbb78",  # Light orange
}


# =============================================================================
# Utility Functions
# =============================================================================

def get_dataset_path(dataset_name: str) -> str:
    """Get the file path for a dataset."""
    if dataset_name in DATASETS:
        return os.path.join(DATA_DIR, DATASETS[dataset_name]["filename"])
    elif dataset_name in TEMPORAL_DATASETS:
        return os.path.join(DATA_DIR, TEMPORAL_DATASETS[dataset_name]["edges_file"])
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def get_ff_prob(goal: str = "scale_down") -> float:
    """
    Get Forest Fire forward burning probability for a given sampling goal.
    
    Args:
        goal: "scale_down" or "back_in_time"
    
    Returns:
        Forward burning probability p_f
    """
    if goal == "scale_down":
        return FF_FORWARD_PROB_SCALEDOWN
    elif goal == "back_in_time":
        return FF_FORWARD_PROB_BACKTIME
    else:
        raise ValueError(f"Unknown goal: {goal}. Use 'scale_down' or 'back_in_time'")


def is_temporal_dataset(dataset_name: str) -> bool:
    """Check if a dataset has temporal information (timestamps)."""
    return dataset_name in TEMPORAL_DATASETS


def list_all_datasets() -> dict:
    """List all available datasets grouped by type."""
    return {
        "static": list(DATASETS.keys()),
        "temporal": list(TEMPORAL_DATASETS.keys())
    }


def get_all_methods() -> list:
    """Get list of all sampling methods (baseline + hybrid)."""
    methods = BASELINE_METHODS.copy()
    for node_m, explore_m in HYBRID_COMBINATIONS:
        methods.append(f"HYB-{node_m}-{explore_m}")
    return methods
