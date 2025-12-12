
import numpy as np
import networkx as nx
from scipy import sparse
from scipy.sparse.linalg import svds
from scipy.stats import ks_2samp
from typing import Dict, List, Tuple, Optional, Union
from collections import Counter, defaultdict
import warnings

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import HOP_PLOT_SAMPLES, NUM_SINGULAR_VALUES

# Suppress warnings for SVD convergence
warnings.filterwarnings('ignore', message='.*ARPACK.*')
warnings.filterwarnings('ignore', category=RuntimeWarning)


# =============================================================================
# Log-Transform Utility Functions
# =============================================================================

def log_transform_distribution(values: np.ndarray) -> np.ndarray:
    """
    Apply log transform to a distribution for KS testing.
    
    For power-law distributions (like degree distributions), comparing in 
    log-space reduces sensitivity to the tail region and provides more
    meaningful comparisons.
    
    Args:
        values: Array of values to transform
    
    Returns:
        Log-transformed array (values <= 0 are filtered out)
    """
    values = np.asarray(values, dtype=float)
    
    # Filter out non-positive values
    values = values[values > 0]
    
    if len(values) == 0:
        return np.array([0.0])
    
    return np.log1p(values)  # log(1 + x) to handle small values


def compute_ks_with_log_transform(dist1: np.ndarray, dist2: np.ndarray) -> float:
    """
    Compute KS statistic after log-transforming both distributions.
    
    This is the recommended way to compare power-law distributions,
    as it reduces sensitivity to extreme values in the tail.
    
    Args:
        dist1: First distribution (original graph)
        dist2: Second distribution (sampled graph)
    
    Returns:
        KS D-statistic computed in log-space
    """
    log_dist1 = log_transform_distribution(dist1)
    log_dist2 = log_transform_distribution(dist2)
    
    if len(log_dist1) == 0 or len(log_dist2) == 0:
        return 1.0
    
    try:
        statistic, _ = ks_2samp(log_dist1, log_dist2)
        return float(statistic)
    except Exception:
        return 1.0


# =============================================================================
# Graph Property Extraction Functions (S1-S9)
# =============================================================================

def get_in_degree_distribution(G: nx.Graph) -> np.ndarray:
    """
    S1: Extract in-degree distribution.
    
    For directed graphs: in-degree of each node
    For undirected graphs: degree of each node
    
    Args:
        G: NetworkX graph
    
    Returns:
        Array of in-degree values
    """
    if G.number_of_nodes() == 0:
        return np.array([0])
    
    if G.is_directed():
        degrees = np.array([d for n, d in G.in_degree()])
    else:
        degrees = np.array([d for n, d in G.degree()])
    
    return degrees


def get_out_degree_distribution(G: nx.Graph) -> np.ndarray:
    """
    S2: Extract out-degree distribution.
    
    For directed graphs: out-degree of each node
    For undirected graphs: degree of each node
    
    Args:
        G: NetworkX graph
    
    Returns:
        Array of out-degree values
    """
    if G.number_of_nodes() == 0:
        return np.array([0])
    
    if G.is_directed():
        degrees = np.array([d for n, d in G.out_degree()])
    else:
        degrees = np.array([d for n, d in G.degree()])
    
    return degrees


def get_wcc_size_distribution(G: nx.Graph) -> np.ndarray:
    """
    S3: Extract weakly connected component size distribution.
    
    Args:
        G: NetworkX graph
    
    Returns:
        Array of WCC sizes
    """
    if G.number_of_nodes() == 0:
        return np.array([0])
    
    if G.is_directed():
        components = list(nx.weakly_connected_components(G))
    else:
        components = list(nx.connected_components(G))
    
    sizes = np.array([len(c) for c in components])
    return sizes


def get_scc_size_distribution(G: nx.Graph) -> np.ndarray:
    """
    S4: Extract strongly connected component size distribution.
    
    For undirected graphs, this is equivalent to S3 (regular connected components).
    
    Args:
        G: NetworkX graph
    
    Returns:
        Array of SCC sizes
    """
    if G.number_of_nodes() == 0:
        return np.array([0])
    
    if G.is_directed():
        components = list(nx.strongly_connected_components(G))
    else:
        components = list(nx.connected_components(G))
    
    sizes = np.array([len(c) for c in components])
    return sizes


def get_hop_plot(G: nx.Graph, num_samples: int = HOP_PLOT_SAMPLES) -> np.ndarray:
    """
    S5: Extract hop-plot (number of pairs reachable within h hops).
    
    Computed on the FULL graph using sampling for efficiency.
    
    Args:
        G: NetworkX graph
        num_samples: Number of source nodes to sample
    
    Returns:
        Array where index h contains cumulative count of pairs reachable in ≤h hops
    """
    if G.number_of_nodes() == 0:
        return np.array([0])
    
    nodes = list(G.nodes())
    num_samples = min(num_samples, len(nodes))
    
    if num_samples == 0:
        return np.array([0])
    
    # Sample source nodes
    sample_nodes = np.random.choice(nodes, size=num_samples, replace=False)
    
    # Collect all shortest path lengths
    all_distances = []
    
    for source in sample_nodes:
        try:
            lengths = nx.single_source_shortest_path_length(G, source)
            all_distances.extend(lengths.values())
        except nx.NetworkXError:
            continue
    
    if not all_distances:
        return np.array([0])
    
    # Convert to hop-plot (cumulative count at each distance)
    max_dist = max(all_distances)
    hop_counts = np.zeros(max_dist + 1)
    
    for d in all_distances:
        if d <= max_dist:
            hop_counts[d] += 1
    
    # Cumulative sum
    cumulative = np.cumsum(hop_counts)
    
    return cumulative


def get_hop_plot_wcc(G: nx.Graph, num_samples: int = HOP_PLOT_SAMPLES) -> np.ndarray:
    """
    S6: Extract hop-plot on the LARGEST Weakly Connected Component only.
    
    This is different from S5 which operates on the full graph.
    S6 focuses on the largest connected component to avoid disconnected
    nodes affecting the hop-plot.
    
    Args:
        G: NetworkX graph
        num_samples: Number of source nodes to sample
    
    Returns:
        Array where index h contains cumulative count of pairs reachable in ≤h hops
        (computed only within the largest WCC)
    """
    if G.number_of_nodes() == 0:
        return np.array([0])
    
    # Get largest WCC
    if G.is_directed():
        components = list(nx.weakly_connected_components(G))
    else:
        components = list(nx.connected_components(G))
    
    if not components:
        return np.array([0])
    
    largest_wcc = max(components, key=len)
    
    if len(largest_wcc) == 0:
        return np.array([0])
    
    # Create subgraph of largest WCC
    wcc_graph = G.subgraph(largest_wcc).copy()
    
    # Compute hop-plot on largest WCC only
    return get_hop_plot(wcc_graph, num_samples)


def get_singular_vector_distribution(G: nx.Graph, k: int = 10) -> np.ndarray:
    """
    S7: Extract first left singular vector distribution.
    
    Computes the top singular vector of the adjacency matrix and returns
    the distribution of its absolute values.
    
    Args:
        G: NetworkX graph
        k: Number of singular values/vectors to compute
    
    Returns:
        Array of singular vector components (absolute values, sorted descending)
    """
    if G.number_of_nodes() < 3:
        return np.array([0])
    
    try:
        # Get adjacency matrix
        A = nx.adjacency_matrix(G).astype(float)
        
        # Compute SVD
        k = min(k, A.shape[0] - 2, A.shape[1] - 2)
        if k < 1:
            return np.array([0])
        
        U, s, Vt = svds(A, k=k)
        
        # Return absolute values of first left singular vector
        # svds returns singular values in ascending order, so -1 is largest
        first_sv = np.abs(U[:, -1])
        
        return np.sort(first_sv)[::-1]  # Sort descending
        
    except Exception:
        return np.array([0])


def get_singular_value_distribution(G: nx.Graph, 
                                     k: int = NUM_SINGULAR_VALUES) -> np.ndarray:
    """
    S8: Extract singular value distribution.
    
    Computes the top-k singular values of the adjacency matrix.
    
    Args:
        G: NetworkX graph
        k: Number of singular values to compute
    
    Returns:
        Array of singular values (sorted descending)
    """
    if G.number_of_nodes() < 3:
        return np.array([0])
    
    try:
        # Get adjacency matrix
        A = nx.adjacency_matrix(G).astype(float)
        
        # Adjust k based on matrix size
        k = min(k, A.shape[0] - 2, A.shape[1] - 2)
        if k < 1:
            return np.array([0])
        
        # Compute SVD
        U, s, Vt = svds(A, k=k)
        
        # Sort singular values in descending order
        singular_values = np.sort(s)[::-1]
        
        return singular_values
        
    except Exception:
        return np.array([0])


def get_clustering_coefficient_distribution(G: nx.Graph) -> np.ndarray:
    """
    S9: Extract clustering coefficient distribution C_d.
    
    FIXED: This now computes the AVERAGE clustering coefficient for each degree d,
    as defined in the original paper:
    
    "C_d is defined as the average C_v over all nodes v of degree d."
    
    This is different from just returning all node clustering coefficients!
    
    Args:
        G: NetworkX graph
    
    Returns:
        Array of average clustering coefficients per degree (C_d values)
    """
    if G.number_of_nodes() == 0:
        return np.array([0])
    
    try:
        # For directed graphs, convert to undirected for clustering
        if G.is_directed():
            G_undirected = G.to_undirected()
        else:
            G_undirected = G
        
        # Get clustering coefficient for each node
        clustering = nx.clustering(G_undirected)
        
        # Get degree for each node
        degrees = dict(G_undirected.degree())
        
        # Group clustering coefficients by degree
        degree_clustering = defaultdict(list)
        for node, cc in clustering.items():
            d = degrees[node]
            degree_clustering[d].append(cc)
        
        # Compute average clustering coefficient for each degree (C_d)
        C_d_values = []
        for d in sorted(degree_clustering.keys()):
            avg_cc = np.mean(degree_clustering[d])
            C_d_values.append(avg_cc)
        
        if not C_d_values:
            return np.array([0])
        
        return np.array(C_d_values)
        
    except Exception:
        return np.array([0])


# =============================================================================
# KS Statistic Functions
# =============================================================================

def compute_ks_statistic(dist1: np.ndarray, dist2: np.ndarray,
                          use_log_transform: bool = False) -> float:
    """
    Compute Kolmogorov-Smirnov D-statistic between two distributions.
    
    D = max_x |F'(x) - F(x)|
    
    Lower D indicates better match between distributions.
    
    Args:
        dist1: First distribution (original graph)
        dist2: Second distribution (sampled graph)
        use_log_transform: If True, apply log transform before KS test
                           (recommended for power-law distributions)
    
    Returns:
        KS D-statistic (float between 0 and 1)
    """
    # Handle empty distributions
    if len(dist1) == 0 or len(dist2) == 0:
        return 1.0
    
    # Convert to numpy arrays
    dist1 = np.asarray(dist1, dtype=float)
    dist2 = np.asarray(dist2, dtype=float)
    
    # Remove NaN and Inf values
    dist1 = dist1[np.isfinite(dist1)]
    dist2 = dist2[np.isfinite(dist2)]
    
    if len(dist1) == 0 or len(dist2) == 0:
        return 1.0
    
    # Apply log transform if requested (for power-law distributions)
    if use_log_transform:
        return compute_ks_with_log_transform(dist1, dist2)
    
    # Use scipy's KS test directly
    try:
        statistic, _ = ks_2samp(dist1, dist2)
        return float(statistic)
    except Exception:
        return 1.0


def compute_mean_statistics(stats_dict: Dict[str, List[float]]) -> Dict[str, float]:
    """
    Compute mean of statistics across multiple runs.
    
    Args:
        stats_dict: Dictionary with property names as keys and lists of values
    
    Returns:
        Dictionary with mean values
    """
    return {
        prop: float(np.mean(values)) for prop, values in stats_dict.items()
    }


# =============================================================================
# Comprehensive Graph Evaluator
# =============================================================================

class GraphEvaluator:
    """
    Comprehensive evaluator for comparing sampled graphs to original graphs.
    
    Computes all static properties (S1-S9) and their KS statistics.
    
    KEY FEATURES:
    - S6 (hop-plot on largest WCC) is included by default
    - Log-transform is applied to power-law distributions (degree, component sizes)
    - S9 is correctly computed as C_d (average clustering per degree)
    - All 9 properties are evaluated
    """
    
    # Properties that should use log-transform (power-law distributions)
    LOG_TRANSFORM_PROPERTIES = {'in_degree', 'out_degree', 'wcc', 'scc'}
    
    def __init__(self, original_graph: nx.Graph, 
                 hop_samples: int = HOP_PLOT_SAMPLES,
                 n_singular: int = NUM_SINGULAR_VALUES,
                 use_log_transform: bool = True):
        """
        Initialize evaluator with original graph.
        
        Args:
            original_graph: The original graph to compare against
            hop_samples: Number of samples for hop-plot estimation
            n_singular: Number of singular values to compute
            use_log_transform: Whether to use log-transform for power-law distributions
        """
        self.G = original_graph
        self.hop_samples = hop_samples
        self.n_singular = n_singular
        self.use_log_transform = use_log_transform
        
        # Precompute original graph properties (lazy loading)
        self._original_properties = {}
    
    def _get_original_property(self, property_name: str) -> np.ndarray:
        """
        Get (and cache) a property of the original graph.
        
        Args:
            property_name: Name of the property
        
        Returns:
            Property distribution as numpy array
        """
        if property_name not in self._original_properties:
            self._original_properties[property_name] = self._compute_property(
                self.G, property_name
            )
        return self._original_properties[property_name]
    
    def _compute_property(self, G: nx.Graph, property_name: str) -> np.ndarray:
        """
        Compute a specific property for a graph.
        
        Args:
            G: Graph to compute property for
            property_name: Name of the property
        
        Returns:
            Property distribution as numpy array
        """
        if property_name == "in_degree":
            return get_in_degree_distribution(G)
        elif property_name == "out_degree":
            return get_out_degree_distribution(G)
        elif property_name == "wcc":
            return get_wcc_size_distribution(G)
        elif property_name == "scc":
            return get_scc_size_distribution(G)
        elif property_name == "hop_plot":
            return get_hop_plot(G, self.hop_samples)
        elif property_name == "hop_plot_wcc":
            return get_hop_plot_wcc(G, self.hop_samples)
        elif property_name == "singular_vec":
            return get_singular_vector_distribution(G)
        elif property_name == "singular_val":
            return get_singular_value_distribution(G, self.n_singular)
        elif property_name == "clustering":
            return get_clustering_coefficient_distribution(G)
        else:
            raise ValueError(f"Unknown property: {property_name}")
    
    def evaluate_property(self, sampled_graph: nx.Graph, 
                          property_name: str) -> float:
        """
        Evaluate a single property by computing KS statistic.
        
        Args:
            sampled_graph: The sampled graph to evaluate
            property_name: Name of the property to evaluate
        
        Returns:
            KS D-statistic for the property
        """
        original_dist = self._get_original_property(property_name)
        sampled_dist = self._compute_property(sampled_graph, property_name)
        
        # Use log-transform for power-law distributions
        use_log = self.use_log_transform and property_name in self.LOG_TRANSFORM_PROPERTIES
        
        return compute_ks_statistic(original_dist, sampled_dist, use_log_transform=use_log)
    
    def evaluate_all(self, sampled_graph: nx.Graph,
                     include_s6: bool = True) -> Dict[str, float]:
        """
        Evaluate all properties and return KS statistics.
        
        Args:
            sampled_graph: The sampled graph to evaluate
            include_s6: Whether to include S6 (hop-plot on largest WCC)
        
        Returns:
            Dictionary with property names and their KS statistics
        """
        # Properties S1-S5
        properties = [
            "in_degree",     # S1
            "out_degree",    # S2
            "wcc",           # S3
            "scc",           # S4
            "hop_plot",      # S5
        ]
        
        # S6: hop-plot on largest WCC (included by default)
        if include_s6:
            properties.append("hop_plot_wcc")  # S6
        
        # S7-S9
        properties.extend([
            "singular_vec",  # S7
            "singular_val",  # S8
            "clustering",    # S9
        ])
        
        results = {}
        for prop in properties:
            results[prop] = self.evaluate_property(sampled_graph, prop)
        
        # Compute average
        results["AVG"] = float(np.mean(list(results.values())))
        
        return results
    
    def get_property_names(self, include_s6: bool = True) -> List[str]:
        """
        Get list of property names.
        
        Args:
            include_s6: Whether to include S6
        
        Returns:
            List of property names
        """
        props = ["in_degree", "out_degree", "wcc", "scc", "hop_plot"]
        if include_s6:
            props.append("hop_plot_wcc")
        props.extend(["singular_vec", "singular_val", "clustering"])
        return props


# =============================================================================
# Utility Functions
# =============================================================================

def evaluate_sample(original_graph: nx.Graph, sampled_graph: nx.Graph,
                    use_log_transform: bool = True,
                    include_s6: bool = True) -> Dict[str, float]:
    """
    Convenience function to evaluate a sampled graph against the original.
    
    Args:
        original_graph: The original graph
        sampled_graph: The sampled graph
        use_log_transform: Whether to use log-transform for power-law distributions
        include_s6: Whether to include S6
    
    Returns:
        Dictionary with KS statistics for all properties
    """
    evaluator = GraphEvaluator(original_graph, use_log_transform=use_log_transform)
    return evaluator.evaluate_all(sampled_graph, include_s6=include_s6)


def compare_methods(original_graph: nx.Graph, 
                    sampled_graphs: Dict[str, nx.Graph],
                    use_log_transform: bool = True,
                    include_s6: bool = True) -> Dict[str, Dict[str, float]]:
    """
    Compare multiple sampling methods.
    
    Args:
        original_graph: The original graph
        sampled_graphs: Dictionary with method names as keys and sampled graphs as values
        use_log_transform: Whether to use log-transform for power-law distributions
        include_s6: Whether to include S6
    
    Returns:
        Dictionary with method names and their evaluation results
    """
    evaluator = GraphEvaluator(original_graph, use_log_transform=use_log_transform)
    
    results = {}
    for method_name, sampled_graph in sampled_graphs.items():
        results[method_name] = evaluator.evaluate_all(sampled_graph, include_s6=include_s6)
    
    return results


def print_evaluation_results(results: Dict[str, Dict[str, float]], 
                              sort_by: str = "AVG") -> None:
    """
    Print evaluation results in a formatted table.
    
    Args:
        results: Dictionary with method names and their evaluation results
        sort_by: Property to sort by (default: AVG)
    """
    if not results:
        print("No results to display")
        return
    
    # Sort methods by the specified property
    sorted_methods = sorted(results.keys(), 
                            key=lambda m: results[m].get(sort_by, 1.0))
    
    # Get property names from first result
    first_result = list(results.values())[0]
    properties = [p for p in first_result.keys() if p != "AVG"] + ["AVG"]
    
    # Print header
    header = f"{'Method':<20}"
    for prop in properties:
        header += f"{prop:<12}"
    print(header)
    print("-" * len(header))
    
    # Print results
    for method in sorted_methods:
        row = f"{method:<20}"
        for prop in properties:
            value = results[method].get(prop, 1.0)
            row += f"{value:<12.4f}"
        print(row)


def get_summary_statistics(results: Dict[str, Dict[str, float]]) -> Dict[str, any]:
    """
    Compute summary statistics from evaluation results.
    
    Args:
        results: Dictionary with method names and their evaluation results
    
    Returns:
        Summary statistics dictionary
    """
    if not results:
        return {}
    
    # Find best method (lowest AVG)
    best_method = min(results.keys(), key=lambda m: results[m].get("AVG", 1.0))
    best_avg = results[best_method]["AVG"]
    
    # Find best method for each property
    properties = [p for p in list(results.values())[0].keys() if p != "AVG"]
    best_per_property = {}
    for prop in properties:
        best_prop_method = min(results.keys(), 
                               key=lambda m: results[m].get(prop, 1.0))
        best_per_property[prop] = {
            "method": best_prop_method,
            "value": results[best_prop_method][prop]
        }
    
    return {
        "best_overall": {
            "method": best_method,
            "AVG": best_avg
        },
        "best_per_property": best_per_property,
        "num_methods": len(results),
        "num_properties": len(properties)
    }
