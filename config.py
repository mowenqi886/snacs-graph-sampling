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

DATASETS = {
    # Collaboration networks (undirected)
    "hep-th": {
        "url": "https://snap.stanford.edu/data/ca-HepTh.txt.gz",
        "filename": "ca-HepTh.txt.gz",
        "directed": False,
        "description": "High Energy Physics Theory collaboration network"
    },
    "hep-ph": {
        "url": "https://snap.stanford.edu/data/ca-HepPh.txt.gz",
        "filename": "ca-HepPh.txt.gz",
        "directed": False,
        "description": "High Energy Physics Phenomenology collaboration network"
    },
    "astro-ph": {
        "url": "https://snap.stanford.edu/data/ca-AstroPh.txt.gz",
        "filename": "ca-AstroPh.txt.gz",
        "directed": False,
        "description": "Astrophysics collaboration network"
    },
    
    # Autonomous systems (undirected)
    "as": {
        "url": "https://snap.stanford.edu/data/as-733.tar.gz",
        "filename": "as-733.tar.gz",
        "directed": False,
        "description": "Autonomous Systems network"
    },
    
    # Trust network (directed)
    "epinions": {
        "url": "https://snap.stanford.edu/data/soc-Epinions1.txt.gz",
        "filename": "soc-Epinions1.txt.gz",
        "directed": True,
        "description": "Epinions trust network"
    }
}


# =============================================================================
# Temporal Dataset Configuration (With Timestamps for Back-in-Time)
# =============================================================================

TEMPORAL_DATASETS = {
    # Citation networks with timestamps
    "cit-HepTh": {
        "edges_url": "https://snap.stanford.edu/data/cit-HepTh.txt.gz",
        "dates_url": "https://snap.stanford.edu/data/cit-HepTh-dates.txt.gz",
        "edges_file": "cit-HepTh.txt.gz",
        "dates_file": "cit-HepTh-dates.txt.gz",
        "directed": True,
        "description": "ArXiv HEP-TH citation network with timestamps (1993-2003)",
        "time_range": "January 1993 - April 2003 (124 months)"
    },
    "cit-HepPh": {
        "edges_url": "https://snap.stanford.edu/data/cit-HepPh.txt.gz",
        "dates_url": "https://snap.stanford.edu/data/cit-HepPh-dates.txt.gz",
        "edges_file": "cit-HepPh.txt.gz",
        "dates_file": "cit-HepPh-dates.txt.gz",
        "directed": True,
        "description": "ArXiv HEP-PH citation network with timestamps (1993-2003)",
        "time_range": "January 1993 - April 2003 (124 months)"
    }
}


# =============================================================================
# Sampling Configuration
# =============================================================================

# Sampling ratios to test
SAMPLING_RATIOS = [0.10, 0.15, 0.20]

# Number of independent runs per configuration
NUM_RUNS = 10

# Random seed for reproducibility
RANDOM_SEED = 42


# =============================================================================
# Sampling Method Configuration
# =============================================================================

# Baseline methods
BASELINE_METHODS = [
    "RN",   # Random Node sampling
    "RPN",  # Random PageRank Node sampling  
    "RDN",  # Random Degree Node sampling
    "RW",   # Random Walk sampling
    "RJ",   # Random Jump (with restart)
    "FF"    # Forest Fire sampling
]

# Hybrid method combinations: (node_selection, exploration)
HYBRID_COMBINATIONS = [
    ("RN", "RW"),    # Random + Random Walk
    ("RN", "FF"),    # Random + Forest Fire
    ("RPN", "RW"),   # PageRank + Random Walk
    ("RPN", "FF"),   # PageRank + Forest Fire
    ("RDN", "RW"),   # Degree + Random Walk
    ("RDN", "FF"),   # Degree + Forest Fire
]

# Alpha values for hybrid methods (fraction from node selection)
HYBRID_ALPHA_VALUES = [0.3, 0.5, 0.7]


# =============================================================================
# Method-Specific Parameters
# =============================================================================

# Random Walk restart probability
RANDOM_WALK_RESTART_PROB = 0.15

# Maximum steps multiplier for random walk (max_steps = n_samples * multiplier)
RANDOM_WALK_MAX_STEPS_MULTIPLIER = 100

# Forest Fire forward burning probability
# Scale-down goal: p_f = 0.7 (match overall graph properties)
FF_FORWARD_PROB_SCALEDOWN = 0.7

# Back-in-time goal: p_f = 0.2 (emphasize earlier structure)
FF_FORWARD_PROB_BACKTIME = 0.2

# Forest Fire backward burning probability
FF_BACKWARD_PROB = 0.2


# =============================================================================
# Evaluation Configuration
# =============================================================================

# Number of source nodes for hop-plot estimation
HOP_PLOT_SAMPLES = 500

# Number of singular values to compute
NUM_SINGULAR_VALUES = 50

# Whether to include S6 (hop-plot on largest WCC) by default
INCLUDE_S6_DEFAULT = True

# Whether to use log-transform for power-law distributions
USE_LOG_TRANSFORM_DEFAULT = True


# =============================================================================
# Back-in-Time Configuration
# =============================================================================

# Number of time slices for back-in-time evaluation (T1, T2, T3, T4, T5)
NUM_TIME_SLICES = 5

# Method for creating time slices
# "equal_time": Equal time intervals
# "equal_nodes": Equal number of nodes per slice
TIME_SLICE_METHOD = "equal_time"


# =============================================================================
# Visualization Configuration
# =============================================================================

# Figure size
FIGURE_SIZE = (10, 6)

# DPI for saved figures
FIGURE_DPI = 150

# Figure format for saved files
FIGURE_FORMAT = "png"

# Color scheme for methods
METHOD_COLORS = {
    "RN": "#1f77b4",
    "RPN": "#ff7f0e", 
    "RDN": "#2ca02c",
    "RW": "#d62728",
    "RJ": "#9467bd",
    "FF": "#8c564b",
    "HYB-RN-RW": "#e377c2",
    "HYB-RN-FF": "#7f7f7f",
    "HYB-RPN-RW": "#bcbd22",
    "HYB-RPN-FF": "#17becf",
    "HYB-RDN-RW": "#aec7e8",
    "HYB-RDN-FF": "#ffbb78"
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
    """Get Forest Fire probability for a given goal."""
    if goal == "scale_down":
        return FF_FORWARD_PROB_SCALEDOWN
    elif goal == "back_in_time":
        return FF_FORWARD_PROB_BACKTIME
    else:
        raise ValueError(f"Unknown goal: {goal}. Use 'scale_down' or 'back_in_time'")


def is_temporal_dataset(dataset_name: str) -> bool:
    """Check if a dataset has temporal information."""
    return dataset_name in TEMPORAL_DATASETS


def list_all_datasets() -> dict:
    """List all available datasets."""
    return {
        "static": list(DATASETS.keys()),
        "temporal": list(TEMPORAL_DATASETS.keys())
    }