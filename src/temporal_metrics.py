
import numpy as np
import networkx as nx
from scipy.sparse.linalg import svds
from scipy.stats import ks_2samp
from typing import Dict, List, Tuple, Optional, Union
from collections import OrderedDict
import warnings

warnings.filterwarnings('ignore', category=RuntimeWarning)


# =============================================================================
# T1: Densification Power Law (DPL)
# =============================================================================

def compute_densification_exponent(snapshots: Dict[str, nx.Graph]) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    T1: Compute Densification Power Law exponent.
    
    DPL states that e(t) ∝ n(t)^a, where:
    - e(t) = number of edges at time t
    - n(t) = number of nodes at time t
    - a = densification exponent
    
    Typically a > 1, meaning the average degree of nodes increases over time
    (the graph becomes denser, not just larger).
    
    Args:
        snapshots: Dictionary of {snapshot_name: graph} ordered by time
    
    Returns:
        Tuple of (exponent_a, node_counts, edge_counts)
    """
    node_counts = []
    edge_counts = []
    
    # Sort snapshots by name to ensure temporal order
    for name in sorted(snapshots.keys()):
        G = snapshots[name]
        n = G.number_of_nodes()
        m = G.number_of_edges()
        
        if n > 0 and m > 0:
            node_counts.append(n)
            edge_counts.append(m)
    
    node_counts = np.array(node_counts, dtype=float)
    edge_counts = np.array(edge_counts, dtype=float)
    
    if len(node_counts) < 2:
        return 1.0, node_counts, edge_counts
    
    # Fit log(e) = a * log(n) + b using least squares
    # This gives us the densification exponent a
    log_n = np.log(node_counts)
    log_e = np.log(edge_counts)
    
    # Linear regression: log(e) = a * log(n) + b
    A = np.vstack([log_n, np.ones(len(log_n))]).T
    result = np.linalg.lstsq(A, log_e, rcond=None)
    a = result[0][0]
    
    return float(a), node_counts, edge_counts


def get_t1_dpl_distribution(snapshots: Dict[str, nx.Graph]) -> np.ndarray:
    """
    Get T1 DPL as a distribution for KS comparison.
    
    Returns edge counts normalized by node counts at each snapshot.
    This captures the densification pattern.
    
    Args:
        snapshots: Ordered dictionary of graph snapshots
    
    Returns:
        Array of edge/node ratios over time
    """
    _, node_counts, edge_counts = compute_densification_exponent(snapshots)
    
    if len(edge_counts) == 0 or len(node_counts) == 0:
        return np.array([0.0])
    
    # Return edges per node (average degree / 2 for undirected)
    with np.errstate(divide='ignore', invalid='ignore'):
        ratios = edge_counts / node_counts
        ratios = np.nan_to_num(ratios, nan=0.0, posinf=0.0, neginf=0.0)
    
    return ratios


# =============================================================================
# T2: Effective Diameter Over Time
# =============================================================================

def compute_effective_diameter(G: nx.Graph, percentile: float = 0.9, 
                                num_samples: int = 500) -> float:
    """
    Compute effective diameter of a graph.
    
    Effective diameter = minimum number of hops in which `percentile` fraction
    of all connected pairs of nodes can reach each other.
    
    Args:
        G: Graph
        percentile: Fraction of pairs (default 0.9 = 90%)
        num_samples: Number of source nodes to sample for estimation
    
    Returns:
        Effective diameter value
    """
    if G.number_of_nodes() < 2:
        return 0.0
    
    nodes = list(G.nodes())
    num_samples = min(num_samples, len(nodes))
    
    if num_samples == 0:
        return 0.0
    
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


def get_t2_diameter_over_time(snapshots: Dict[str, nx.Graph], 
                               num_samples: int = 500) -> np.ndarray:
    """
    T2: Compute effective diameter at each time snapshot.
    
    According to Leskovec & Faloutsos (2006), the effective diameter
    generally shrinks or stabilizes as the graph grows with time.
    
    Args:
        snapshots: Dictionary of {snapshot_name: graph} ordered by time
        num_samples: Number of samples for diameter estimation
    
    Returns:
        Array of diameter values over time
    """
    diameters = []
    
    for name in sorted(snapshots.keys()):
        G = snapshots[name]
        
        if G.number_of_nodes() == 0:
            diameters.append(0.0)
            continue
        
        # Use largest WCC for diameter computation (avoid disconnected nodes)
        if G.is_directed():
            wccs = list(nx.weakly_connected_components(G))
        else:
            wccs = list(nx.connected_components(G))
        
        if wccs:
            largest_wcc = max(wccs, key=len)
            G_wcc = G.subgraph(largest_wcc)
            d = compute_effective_diameter(G_wcc, num_samples=num_samples)
        else:
            d = 0.0
        
        diameters.append(d)
    
    return np.array(diameters)


# =============================================================================
# T3: Largest Connected Component Size Over Time
# =============================================================================

def get_t3_cc_size_over_time(snapshots: Dict[str, nx.Graph], 
                              normalize: bool = True) -> np.ndarray:
    """
    T3: Compute normalized size of largest connected component over time.
    
    For directed graphs, uses Weakly Connected Components.
    
    Args:
        snapshots: Dictionary of {snapshot_name: graph} ordered by time
        normalize: If True, return fraction of total nodes; else absolute size
    
    Returns:
        Array of CC sizes over time
    """
    cc_sizes = []
    
    for name in sorted(snapshots.keys()):
        G = snapshots[name]
        n = G.number_of_nodes()
        
        if n == 0:
            cc_sizes.append(0.0)
            continue
        
        if G.is_directed():
            wccs = list(nx.weakly_connected_components(G))
        else:
            wccs = list(nx.connected_components(G))
        
        if wccs:
            largest_size = max(len(c) for c in wccs)
            if normalize:
                cc_sizes.append(largest_size / n)
            else:
                cc_sizes.append(float(largest_size))
        else:
            cc_sizes.append(0.0)
    
    return np.array(cc_sizes)


# =============================================================================
# T4: Largest Singular Value Over Time
# =============================================================================

def get_largest_singular_value(G: nx.Graph) -> float:
    """
    Compute largest singular value of the adjacency matrix.
    
    The largest singular value is related to graph structure and
    grows as the graph becomes denser.
    
    Args:
        G: Graph
    
    Returns:
        Largest singular value
    """
    if G.number_of_nodes() < 3:
        return 0.0
    
    try:
        A = nx.adjacency_matrix(G).astype(float)
        
        # svds requires k < min(A.shape) - 1
        k = min(1, A.shape[0] - 2, A.shape[1] - 2)
        if k < 1:
            return 0.0
        
        _, s, _ = svds(A, k=k)
        return float(np.max(s))
    
    except Exception:
        return 0.0


def get_t4_singular_value_over_time(snapshots: Dict[str, nx.Graph]) -> np.ndarray:
    """
    T4: Compute largest singular value at each time snapshot.
    
    Args:
        snapshots: Dictionary of {snapshot_name: graph} ordered by time
    
    Returns:
        Array of singular values over time
    """
    singular_values = []
    
    for name in sorted(snapshots.keys()):
        G = snapshots[name]
        sv = get_largest_singular_value(G)
        singular_values.append(sv)
    
    return np.array(singular_values)


# =============================================================================
# T5: Average Clustering Coefficient Over Time
# =============================================================================

def get_t5_clustering_over_time(snapshots: Dict[str, nx.Graph]) -> np.ndarray:
    """
    T5: Compute average clustering coefficient at each time snapshot.
    
    C at time t is the average C_v of all nodes v in graph at time t,
    where C_v is the local clustering coefficient of node v.
    
    Args:
        snapshots: Dictionary of {snapshot_name: graph} ordered by time
    
    Returns:
        Array of average clustering coefficients over time
    """
    clustering_values = []
    
    for name in sorted(snapshots.keys()):
        G = snapshots[name]
        
        if G.number_of_nodes() < 3:
            clustering_values.append(0.0)
            continue
        
        try:
            if G.is_directed():
                G_undirected = G.to_undirected()
                avg_cc = nx.average_clustering(G_undirected)
            else:
                avg_cc = nx.average_clustering(G)
            
            clustering_values.append(float(avg_cc))
        except Exception:
            clustering_values.append(0.0)
    
    return np.array(clustering_values)


# =============================================================================
# Temporal Metrics Evaluator Class
# =============================================================================

class TemporalMetricsEvaluator:
    """
    Evaluator for temporal graph metrics T1-T5.
    
    Compares temporal evolution patterns between original graph snapshots
    and sampled graph snapshots using KS D-statistic.
    
    Usage:
        evaluator = TemporalMetricsEvaluator(original_snapshots)
        results = evaluator.evaluate(sampled_snapshots)
    """
    
    def __init__(self, original_snapshots: Dict[str, nx.Graph],
                 hop_samples: int = 500):
        """
        Initialize with original graph snapshots.
        
        Args:
            original_snapshots: Dict of {snapshot_name: graph} for original graph
            hop_samples: Number of samples for hop-plot/diameter computation
        """
        self.original_snapshots = OrderedDict(sorted(original_snapshots.items()))
        self.hop_samples = hop_samples
        self.num_snapshots = len(original_snapshots)
        
        # Precompute original temporal metrics
        self._original_metrics = {}
        self._compute_original_metrics()
    
    def _compute_original_metrics(self):
        """Precompute all temporal metrics for original snapshots."""
        print("  Computing original temporal metrics (T1-T5)...")
        
        # T1: Densification Power Law
        exponent, nodes, edges = compute_densification_exponent(self.original_snapshots)
        self._original_metrics['T1_exponent'] = exponent
        self._original_metrics['T1_nodes'] = nodes
        self._original_metrics['T1_edges'] = edges
        self._original_metrics['T1_dpl'] = get_t1_dpl_distribution(self.original_snapshots)
        
        # T2: Diameter over time
        self._original_metrics['T2_diameter'] = get_t2_diameter_over_time(
            self.original_snapshots, self.hop_samples
        )
        
        # T3: CC size over time
        self._original_metrics['T3_cc_size'] = get_t3_cc_size_over_time(
            self.original_snapshots
        )
        
        # T4: Singular value over time
        self._original_metrics['T4_singular'] = get_t4_singular_value_over_time(
            self.original_snapshots
        )
        
        # T5: Clustering over time
        self._original_metrics['T5_clustering'] = get_t5_clustering_over_time(
            self.original_snapshots
        )
        
        print(f"    T1 (DPL): exponent = {exponent:.4f}")
        print(f"    T2 (Diameter): range = [{self._original_metrics['T2_diameter'].min():.2f}, "
              f"{self._original_metrics['T2_diameter'].max():.2f}]")
        print(f"    T3 (CC Size): range = [{self._original_metrics['T3_cc_size'].min():.4f}, "
              f"{self._original_metrics['T3_cc_size'].max():.4f}]")
        print(f"    T4 (Singular): range = [{self._original_metrics['T4_singular'].min():.2f}, "
              f"{self._original_metrics['T4_singular'].max():.2f}]")
        print(f"    T5 (Clustering): range = [{self._original_metrics['T5_clustering'].min():.4f}, "
              f"{self._original_metrics['T5_clustering'].max():.4f}]")
        print("  ✓ Original temporal metrics computed")
    
    def get_original_metrics(self) -> Dict[str, Union[float, np.ndarray]]:
        """Get precomputed original metrics."""
        return self._original_metrics.copy()
    
    def compute_sampled_metrics(self, sampled_snapshots: Dict[str, nx.Graph]) -> Dict[str, Union[float, np.ndarray]]:
        """
        Compute temporal metrics for sampled snapshots.
        
        Args:
            sampled_snapshots: Dict of {snapshot_name: sampled_graph}
        
        Returns:
            Dictionary with all temporal metrics
        """
        sampled_snapshots = OrderedDict(sorted(sampled_snapshots.items()))
        metrics = {}
        
        # T1: Densification
        exponent, nodes, edges = compute_densification_exponent(sampled_snapshots)
        metrics['T1_exponent'] = exponent
        metrics['T1_nodes'] = nodes
        metrics['T1_edges'] = edges
        metrics['T1_dpl'] = get_t1_dpl_distribution(sampled_snapshots)
        
        # T2: Diameter
        metrics['T2_diameter'] = get_t2_diameter_over_time(sampled_snapshots, self.hop_samples)
        
        # T3: CC size
        metrics['T3_cc_size'] = get_t3_cc_size_over_time(sampled_snapshots)
        
        # T4: Singular value
        metrics['T4_singular'] = get_t4_singular_value_over_time(sampled_snapshots)
        
        # T5: Clustering
        metrics['T5_clustering'] = get_t5_clustering_over_time(sampled_snapshots)
        
        return metrics
    
    def _compute_ks_for_temporal(self, orig: np.ndarray, sampled: np.ndarray) -> float:
        """
        Compute KS statistic between two temporal sequences.
        
        Normalizes both sequences before comparison to focus on shape/pattern
        rather than absolute values.
        
        Args:
            orig: Original metric values over time
            sampled: Sampled metric values over time
        
        Returns:
            KS D-statistic
        """
        if len(orig) == 0 or len(sampled) == 0:
            return 1.0
        
        # Handle constant arrays
        orig_range = np.max(orig) - np.min(orig)
        sampled_range = np.max(sampled) - np.min(sampled)
        
        if orig_range < 1e-10 and sampled_range < 1e-10:
            return 0.0  # Both constant - consider them equal
        
        # Normalize to [0, 1] for comparison
        if orig_range > 1e-10:
            orig_norm = (orig - np.min(orig)) / orig_range
        else:
            orig_norm = np.zeros_like(orig)
        
        if sampled_range > 1e-10:
            sampled_norm = (sampled - np.min(sampled)) / sampled_range
        else:
            sampled_norm = np.zeros_like(sampled)
        
        try:
            stat, _ = ks_2samp(orig_norm, sampled_norm)
            return float(stat)
        except Exception:
            return 1.0
    
    def evaluate(self, sampled_snapshots: Dict[str, nx.Graph]) -> Dict[str, float]:
        """
        Evaluate sampled graph snapshots against original temporal patterns.
        
        Args:
            sampled_snapshots: Dict of {snapshot_name: sampled_graph}
        
        Returns:
            Dictionary with KS statistics for each temporal metric (T1-T5)
            and average (T_AVG)
        """
        # Compute sampled temporal metrics
        sampled_metrics = self.compute_sampled_metrics(sampled_snapshots)
        
        results = {}
        
        # T1: DPL
        results['T1_dpl'] = self._compute_ks_for_temporal(
            self._original_metrics['T1_dpl'],
            sampled_metrics['T1_dpl']
        )
        
        # T2: Diameter
        results['T2_diameter'] = self._compute_ks_for_temporal(
            self._original_metrics['T2_diameter'],
            sampled_metrics['T2_diameter']
        )
        
        # T3: CC Size
        results['T3_cc_size'] = self._compute_ks_for_temporal(
            self._original_metrics['T3_cc_size'],
            sampled_metrics['T3_cc_size']
        )
        
        # T4: Singular Value
        results['T4_singular'] = self._compute_ks_for_temporal(
            self._original_metrics['T4_singular'],
            sampled_metrics['T4_singular']
        )
        
        # T5: Clustering
        results['T5_clustering'] = self._compute_ks_for_temporal(
            self._original_metrics['T5_clustering'],
            sampled_metrics['T5_clustering']
        )
        
        # Compute average across all temporal metrics
        temporal_values = [results['T1_dpl'], results['T2_diameter'], 
                          results['T3_cc_size'], results['T4_singular'], 
                          results['T5_clustering']]
        results['T_AVG'] = float(np.mean(temporal_values))
        
        return results


# =============================================================================
# Convenience Functions
# =============================================================================

def compute_all_temporal_metrics(snapshots: Dict[str, nx.Graph], 
                                  hop_samples: int = 500) -> Dict[str, Union[float, np.ndarray]]:
    """
    Compute all temporal metrics for a sequence of graph snapshots.
    
    Args:
        snapshots: Dictionary of {snapshot_name: graph} ordered by time
        hop_samples: Number of samples for diameter estimation
    
    Returns:
        Dictionary with all temporal metrics
    """
    snapshots = OrderedDict(sorted(snapshots.items()))
    results = {}
    
    # T1: Densification
    exponent, nodes, edges = compute_densification_exponent(snapshots)
    results['T1_exponent'] = exponent
    results['T1_nodes'] = nodes
    results['T1_edges'] = edges
    results['T1_dpl'] = get_t1_dpl_distribution(snapshots)
    
    # T2: Diameter
    results['T2_diameter'] = get_t2_diameter_over_time(snapshots, hop_samples)
    
    # T3: CC size
    results['T3_cc_size'] = get_t3_cc_size_over_time(snapshots)
    
    # T4: Singular value
    results['T4_singular'] = get_t4_singular_value_over_time(snapshots)
    
    # T5: Clustering
    results['T5_clustering'] = get_t5_clustering_over_time(snapshots)
    
    return results


def print_temporal_metrics(metrics: Dict[str, Union[float, np.ndarray]], 
                           name: str = "Graph") -> None:
    """
    Print temporal metrics in formatted table.
    
    Args:
        metrics: Dictionary from compute_all_temporal_metrics
        name: Name to display
    """
    print(f"\n{'='*60}")
    print(f" Temporal Metrics (T1-T5): {name}")
    print(f"{'='*60}")
    
    if 'T1_exponent' in metrics:
        print(f"  T1 (Densification): exponent a = {metrics['T1_exponent']:.4f}")
    if 'T2_diameter' in metrics:
        d = metrics['T2_diameter']
        print(f"  T2 (Diameter):      [{d.min():.2f}, {d.max():.2f}] (mean: {d.mean():.2f})")
    if 'T3_cc_size' in metrics:
        c = metrics['T3_cc_size']
        print(f"  T3 (CC Size):       [{c.min():.4f}, {c.max():.4f}] (mean: {c.mean():.4f})")
    if 'T4_singular' in metrics:
        s = metrics['T4_singular']
        print(f"  T4 (Singular Val):  [{s.min():.2f}, {s.max():.2f}] (mean: {s.mean():.2f})")
    if 'T5_clustering' in metrics:
        cl = metrics['T5_clustering']
        print(f"  T5 (Clustering):    [{cl.min():.4f}, {cl.max():.4f}] (mean: {cl.mean():.4f})")
    
    print(f"{'='*60}")


def compare_temporal_metrics(original_metrics: Dict, sampled_metrics: Dict) -> Dict[str, float]:
    """
    Compare temporal metrics between original and sampled graphs using KS statistic.
    
    Args:
        original_metrics: Metrics from original graph snapshots
        sampled_metrics: Metrics from sampled graph snapshots
    
    Returns:
        Dictionary of KS D-statistics for each temporal metric
    """
    results = {}
    
    comparisons = [
        ('T1_dpl', 'T1_dpl'),
        ('T2_diameter', 'T2_diameter'),
        ('T3_cc_size', 'T3_cc_size'),
        ('T4_singular', 'T4_singular'),
        ('T5_clustering', 'T5_clustering'),
    ]
    
    for key, metric_key in comparisons:
        orig = original_metrics.get(metric_key, np.array([]))
        sampled = sampled_metrics.get(metric_key, np.array([]))
        
        if len(orig) > 0 and len(sampled) > 0:
            try:
                # Normalize for comparison
                orig_max = np.max(np.abs(orig)) + 1e-10
                sampled_max = np.max(np.abs(sampled)) + 1e-10
                
                stat, _ = ks_2samp(orig / orig_max, sampled / sampled_max)
                results[key] = float(stat)
            except Exception:
                results[key] = 1.0
        else:
            results[key] = 1.0
    
    # Average
    results['T_AVG'] = float(np.mean(list(results.values())))
    
    return results


# =============================================================================
# Demo and Testing
# =============================================================================

if __name__ == "__main__":
    print("="*70)
    print(" TEMPORAL METRICS (T1-T5) - TEST")
    print("="*70)
    
    # Create synthetic temporal graph sequence (simulating growth)
    print("\nCreating synthetic graph sequence...")
    snapshots = OrderedDict()
    
    np.random.seed(42)
    
    for i, n in enumerate([100, 200, 400, 800, 1600]):
        G = nx.barabasi_albert_graph(n, 3, seed=42+i)
        snapshots[f"Snapshot_{i+1}"] = G
        print(f"  Snapshot_{i+1}: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    # Compute metrics
    print("\nComputing temporal metrics...")
    metrics = compute_all_temporal_metrics(snapshots)
    print_temporal_metrics(metrics, "Synthetic BA Graph Sequence")
    
    # Test evaluator
    print("\nTesting TemporalMetricsEvaluator...")
    evaluator = TemporalMetricsEvaluator(snapshots)
    
    # Create "sampled" snapshots (just smaller versions for testing)
    sampled_snapshots = OrderedDict()
    for i, n in enumerate([50, 100, 200, 400, 800]):
        G = nx.barabasi_albert_graph(n, 3, seed=100+i)
        sampled_snapshots[f"Snapshot_{i+1}"] = G
    
    results = evaluator.evaluate(sampled_snapshots)
    
    print("\nEvaluation Results (KS D-statistics):")
    for key, value in results.items():
        print(f"  {key}: {value:.4f}")
    
    print("\n✓ Temporal metrics test completed successfully!")
