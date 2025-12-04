import logging
import random
import numpy as np
import networkx as nx
from typing import Optional, List, Set, Tuple
import sys


def set_random_seed(seed: Optional[int] = None) -> None:
    """
    Set random seed for reproducibility across all random number generators.
    
    Args:
        seed: Random seed value. If None, uses system-generated seed.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        

def get_logger(name: str, level: str = 'INFO') -> logging.Logger:
    """
    Create and configure a logger instance.
    
    Args:
        name: Name of the logger (typically module name)
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    logger.setLevel(getattr(logging, level.upper()))
    return logger


def ensure_connected_sample(
    G: nx.Graph,
    sampled_nodes: Set[int],
    target_size: int
) -> Set[int]:
    """
    Ensure the sampled nodes form a connected subgraph by adding
    bridging nodes if necessary.
    
    Args:
        G: Original graph
        sampled_nodes: Set of initially sampled nodes
        target_size: Target number of nodes
        
    Returns:
        Updated set of sampled nodes
    """
    if len(sampled_nodes) == 0:
        return sampled_nodes
    
    # Create subgraph from sampled nodes
    subgraph = G.subgraph(sampled_nodes)
    
    # If already connected, return as is
    if G.is_directed():
        if nx.is_weakly_connected(subgraph):
            return sampled_nodes
    else:
        if nx.is_connected(subgraph):
            return sampled_nodes
    
    # Get connected components
    if G.is_directed():
        components = list(nx.weakly_connected_components(subgraph))
    else:
        components = list(nx.connected_components(subgraph))
    
    # Sort components by size (largest first)
    components = sorted(components, key=len, reverse=True)
    
    # Try to connect smaller components to the largest one
    largest_component = components[0]
    result_nodes = set(largest_component)
    
    for comp in components[1:]:
        if len(result_nodes) >= target_size:
            break
            
        # Find shortest path between this component and the result
        for node in comp:
            if len(result_nodes) >= target_size:
                break
            result_nodes.add(node)
    
    return result_nodes


def get_largest_component(G: nx.Graph) -> nx.Graph:
    """
    Get the largest (weakly) connected component of a graph.
    
    Args:
        G: Input graph
        
    Returns:
        Subgraph containing only the largest component
    """
    if G.is_directed():
        largest_cc = max(nx.weakly_connected_components(G), key=len)
    else:
        largest_cc = max(nx.connected_components(G), key=len)
    
    return G.subgraph(largest_cc).copy()


def compute_effective_diameter(G: nx.Graph, percentile: float = 0.9) -> float:
    """
    Compute the effective diameter of a graph.
    
    The effective diameter is defined as the minimum number of hops
    in which a given percentile of all connected pairs can reach each other.
    
    Args:
        G: Input graph
        percentile: Percentile for effective diameter (default 0.9 = 90%)
        
    Returns:
        Effective diameter value
    """
    if G.number_of_nodes() == 0:
        return 0.0
    
    # Sample nodes for large graphs
    nodes = list(G.nodes())
    n_samples = min(500, len(nodes))
    sample_nodes = np.random.choice(nodes, size=n_samples, replace=False)
    
    # Compute shortest path lengths
    all_lengths = []
    for node in sample_nodes:
        lengths = nx.single_source_shortest_path_length(G, node)
        all_lengths.extend([l for l in lengths.values() if l > 0])
    
    if not all_lengths:
        return 0.0
    
    # Compute percentile
    return np.percentile(all_lengths, percentile * 100)


def graph_statistics(G: nx.Graph) -> dict:
    """
    Compute basic statistics for a graph.
    
    Args:
        G: Input graph
        
    Returns:
        Dictionary containing graph statistics
    """
    stats = {
        'num_nodes': G.number_of_nodes(),
        'num_edges': G.number_of_edges(),
        'is_directed': G.is_directed(),
    }
    
    if G.number_of_nodes() > 0:
        degrees = [d for n, d in G.degree()]
        stats['avg_degree'] = np.mean(degrees)
        stats['max_degree'] = max(degrees)
        stats['min_degree'] = min(degrees)
        
        # Density
        n = G.number_of_nodes()
        m = G.number_of_edges()
        if n > 1:
            if G.is_directed():
                stats['density'] = m / (n * (n - 1))
            else:
                stats['density'] = 2 * m / (n * (n - 1))
        else:
            stats['density'] = 0.0
    
    return stats


def print_graph_info(G: nx.Graph, name: str = "Graph") -> None:
    """
    Print formatted information about a graph.
    
    Args:
        G: Input graph
        name: Name to display for the graph
    """
    stats = graph_statistics(G)
    
    print(f"\n{'='*50}")
    print(f" {name}")
    print(f"{'='*50}")
    print(f" Nodes:       {stats['num_nodes']:,}")
    print(f" Edges:       {stats['num_edges']:,}")
    print(f" Directed:    {stats['is_directed']}")
    if stats['num_nodes'] > 0:
        print(f" Avg Degree:  {stats['avg_degree']:.2f}")
        print(f" Max Degree:  {stats['max_degree']}")
        print(f" Density:     {stats['density']:.6f}")
    print(f"{'='*50}\n")


def safe_division(a: float, b: float, default: float = 0.0) -> float:
    """
    Perform division with safety check for zero denominator.
    
    Args:
        a: Numerator
        b: Denominator
        default: Value to return if denominator is zero
        
    Returns:
        Result of division or default value
    """
    return a / b if b != 0 else default


def normalize_distribution(values: List[float]) -> np.ndarray:
    """
    Normalize a list of values to form a probability distribution.
    
    Args:
        values: List of non-negative values
        
    Returns:
        Normalized numpy array summing to 1
    """
    values = np.array(values, dtype=float)
    total = values.sum()
    
    if total == 0:
        return np.ones_like(values) / len(values)
    
    return values / total


def sample_with_replacement_limit(
    items: List,
    weights: List[float],
    n: int,
    max_replacement: int = 1
) -> List:
    """
    Sample items with weighted probabilities, limiting how many times
    each item can be selected.
    
    Args:
        items: List of items to sample from
        weights: Sampling weights (will be normalized)
        n: Number of items to sample
        max_replacement: Maximum times an item can be selected
        
    Returns:
        List of sampled items
    """
    if n >= len(items) * max_replacement:
        return list(items) * max_replacement
    
    weights = normalize_distribution(weights)
    selected = []
    counts = {item: 0 for item in items}
    
    while len(selected) < n:
        # Create valid items and weights
        valid_items = [item for item in items if counts[item] < max_replacement]
        if not valid_items:
            break
            
        valid_weights = normalize_distribution([
            weights[items.index(item)] for item in valid_items
        ])
        
        # Sample one item
        idx = np.random.choice(len(valid_items), p=valid_weights)
        item = valid_items[idx]
        selected.append(item)
        counts[item] += 1
    
    return selected
