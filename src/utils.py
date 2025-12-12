
import os
import sys
import logging
import numpy as np
import networkx as nx
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RANDOM_SEED


# =============================================================================
# Random Seed Management
# =============================================================================

def set_seed(seed: int = RANDOM_SEED) -> None:
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed value
    """
    np.random.seed(seed)
    import random
    random.seed(seed)


def get_seed_sequence(base_seed: int, n: int) -> List[int]:
    """
    Generate a sequence of seeds for multiple runs.
    
    Args:
        base_seed: Base seed value
        n: Number of seeds to generate
    
    Returns:
        List of seed values
    """
    return [base_seed + i for i in range(n)]


# =============================================================================
# Logging Configuration
# =============================================================================

def setup_logger(name: str = "graph_sampling", 
                 level: int = logging.INFO,
                 log_file: Optional[str] = None) -> logging.Logger:
    """
    Configure and return a logger.
    
    Args:
        name: Logger name
        level: Logging level
        log_file: Optional file path for logging
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Clear existing handlers
    logger.handlers = []
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(console_format)
        logger.addHandler(file_handler)
    
    return logger


# =============================================================================
# Graph Statistics
# =============================================================================

def compute_graph_statistics(G: nx.Graph) -> Dict[str, Union[int, float]]:
    """
    Compute basic statistics for a graph.
    
    Args:
        G: NetworkX graph
    
    Returns:
        Dictionary with statistics
    """
    stats = {
        'n_nodes': G.number_of_nodes(),
        'n_edges': G.number_of_edges(),
        'is_directed': G.is_directed(),
    }
    
    if G.number_of_nodes() == 0:
        return stats
    
    # Density
    stats['density'] = nx.density(G)
    
    # Degree statistics
    if G.is_directed():
        in_degrees = [d for n, d in G.in_degree()]
        out_degrees = [d for n, d in G.out_degree()]
        stats['avg_in_degree'] = np.mean(in_degrees)
        stats['avg_out_degree'] = np.mean(out_degrees)
        stats['max_in_degree'] = max(in_degrees)
        stats['max_out_degree'] = max(out_degrees)
    else:
        degrees = [d for n, d in G.degree()]
        stats['avg_degree'] = np.mean(degrees)
        stats['max_degree'] = max(degrees)
    
    # Connected components
    if G.is_directed():
        wccs = list(nx.weakly_connected_components(G))
        sccs = list(nx.strongly_connected_components(G))
        stats['n_wcc'] = len(wccs)
        stats['n_scc'] = len(sccs)
        stats['largest_wcc_size'] = max(len(c) for c in wccs) if wccs else 0
        stats['largest_scc_size'] = max(len(c) for c in sccs) if sccs else 0
    else:
        ccs = list(nx.connected_components(G))
        stats['n_cc'] = len(ccs)
        stats['largest_cc_size'] = max(len(c) for c in ccs) if ccs else 0
    
    return stats


def compute_effective_diameter(G: nx.Graph, 
                                percentile: float = 0.9,
                                num_samples: int = 500) -> float:
    """
    Compute effective diameter of a graph.
    
    The effective diameter is the minimum number of hops in which
    `percentile` fraction of all connected pairs can reach each other.
    
    Args:
        G: NetworkX graph
        percentile: Fraction of pairs (default: 0.9 = 90%)
        num_samples: Number of source nodes to sample
    
    Returns:
        Effective diameter
    """
    if G.number_of_nodes() < 2:
        return 0.0
    
    nodes = list(G.nodes())
    num_samples = min(num_samples, len(nodes))
    sample_nodes = np.random.choice(nodes, size=num_samples, replace=False)
    
    all_distances = []
    
    for source in sample_nodes:
        try:
            lengths = nx.single_source_shortest_path_length(G, source)
            all_distances.extend([d for d in lengths.values() if d > 0])
        except nx.NetworkXError:
            continue
    
    if not all_distances:
        return 0.0
    
    return float(np.percentile(all_distances, percentile * 100))


def compute_average_clustering(G: nx.Graph) -> float:
    """
    Compute average clustering coefficient.
    
    Args:
        G: NetworkX graph
    
    Returns:
        Average clustering coefficient
    """
    if G.number_of_nodes() < 3:
        return 0.0
    
    if G.is_directed():
        G_undirected = G.to_undirected()
        return nx.average_clustering(G_undirected)
    
    return nx.average_clustering(G)


# =============================================================================
# Results Formatting
# =============================================================================

def format_results_table(results: Dict[str, Dict[str, float]],
                          sort_by: str = "AVG") -> str:
    """
    Format results dictionary as a text table.
    
    Args:
        results: Dictionary {method: {metric: value}}
        sort_by: Metric to sort by
    
    Returns:
        Formatted string table
    """
    if not results:
        return "No results to display"
    
    # Get all metrics
    first_result = list(results.values())[0]
    metrics = list(first_result.keys())
    
    # Sort methods
    sorted_methods = sorted(results.keys(), 
                           key=lambda m: results[m].get(sort_by, 1.0))
    
    # Build header
    header = f"{'Method':<25}"
    for metric in metrics:
        header += f"{metric:<12}"
    
    lines = [header, "-" * len(header)]
    
    # Build rows
    for method in sorted_methods:
        row = f"{method:<25}"
        for metric in metrics:
            value = results[method].get(metric, 0.0)
            row += f"{value:<12.4f}"
        lines.append(row)
    
    return "\n".join(lines)


def format_time(seconds: float) -> str:
    """
    Format time duration as human-readable string.
    
    Args:
        seconds: Time in seconds
    
    Returns:
        Formatted string (e.g., "1h 23m 45s")
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.0f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"


def format_number(n: int) -> str:
    """
    Format large number with commas.
    
    Args:
        n: Number to format
    
    Returns:
        Formatted string (e.g., "1,234,567")
    """
    return f"{n:,}"


# =============================================================================
# File Utilities
# =============================================================================

def ensure_dir(path: str) -> str:
    """
    Ensure directory exists, create if not.
    
    Args:
        path: Directory path
    
    Returns:
        Same path (for chaining)
    """
    os.makedirs(path, exist_ok=True)
    return path


def get_timestamp() -> str:
    """
    Get current timestamp string.
    
    Returns:
        Timestamp string (e.g., "20240101_123456")
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def safe_filename(name: str) -> str:
    """
    Convert string to safe filename.
    
    Args:
        name: Original string
    
    Returns:
        Safe filename string
    """
    # Replace unsafe characters
    unsafe_chars = ['/', '\\', ':', '*', '?', '"', '<', '>', '|', ' ']
    result = name
    for char in unsafe_chars:
        result = result.replace(char, '_')
    return result


# =============================================================================
# Progress Display
# =============================================================================

def print_progress_bar(iteration: int, total: int, 
                       prefix: str = '', suffix: str = '',
                       length: int = 50, fill: str = '█') -> None:
    """
    Print a progress bar to console.
    
    Args:
        iteration: Current iteration
        total: Total iterations
        prefix: Prefix string
        suffix: Suffix string
        length: Bar length
        fill: Fill character
    """
    if total == 0:
        percent = 100
    else:
        percent = 100 * (iteration / float(total))
    
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    
    print(f'\r{prefix} |{bar}| {percent:.1f}% {suffix}', end='\r')
    
    if iteration == total:
        print()


# =============================================================================
# Validation Utilities
# =============================================================================

def validate_sampling_ratio(ratio: float) -> float:
    """
    Validate and normalize sampling ratio.
    
    Args:
        ratio: Sampling ratio (should be between 0 and 1)
    
    Returns:
        Validated ratio
    
    Raises:
        ValueError: If ratio is invalid
    """
    if not 0 < ratio <= 1:
        raise ValueError(f"Sampling ratio must be in (0, 1], got {ratio}")
    return ratio


def validate_graph(G: nx.Graph) -> None:
    """
    Validate that graph is suitable for sampling.
    
    Args:
        G: NetworkX graph
    
    Raises:
        ValueError: If graph is invalid
    """
    if G.number_of_nodes() == 0:
        raise ValueError("Graph has no nodes")
    
    if G.number_of_edges() == 0:
        raise ValueError("Graph has no edges")


# =============================================================================
# Demo
# =============================================================================

if __name__ == "__main__":
    print("="*70)
    print("UTILITIES MODULE DEMO")
    print("="*70)
    
    # Set seed
    set_seed(42)
    print("\n1. Random seed set to 42")
    
    # Create test graph
    G = nx.barabasi_albert_graph(500, 3, seed=42)
    
    # Compute statistics
    print("\n2. Graph Statistics:")
    stats = compute_graph_statistics(G)
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"   {key}: {value:.4f}")
        else:
            print(f"   {key}: {value}")
    
    # Effective diameter
    print("\n3. Effective Diameter (90th percentile):")
    diameter = compute_effective_diameter(G)
    print(f"   {diameter:.2f}")
    
    # Average clustering
    print("\n4. Average Clustering Coefficient:")
    clustering = compute_average_clustering(G)
    print(f"   {clustering:.4f}")
    
    # Format time
    print("\n5. Time Formatting:")
    for secs in [45, 125, 3725]:
        print(f"   {secs} seconds = {format_time(secs)}")
    
    # Format numbers
    print("\n6. Number Formatting:")
    for n in [1234, 1234567, 12345678901]:
        print(f"   {n} = {format_number(n)}")
    
    print("\n✓ Utilities demo completed!")
